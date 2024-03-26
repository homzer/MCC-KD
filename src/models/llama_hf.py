from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)

from src.checkpoint import auto_split_huggingface_checkpoints
from src.models.modeling import ParallelModelForCausalLM, CausalLMOutputs, AttentionForCausalLM
from src.models.modeling_acts import RMSNorm, Clamp, RotaryEmbedding
from src.models.modeling_args import LlamaArgs, LoraLlamaArgs
from src.utils import set_barrier, apply_lora, logits_normalize, compute_position_ids, apply_rotary_pos_emb


class LlamaAttentionHf(AttentionForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        self.n_local_heads = args.n_heads // args.world_size
        self.head_dim = args.dim // args.n_heads

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        self.rotary_emb = None

    def init_weights(self):
        self.q_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.k_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.v_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.o_proj = RowParallelLinear(
            self.args.n_heads * self.head_dim,
            self.args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_heads, self.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seq_len + start_pos)
        position_ids = compute_position_ids(start_pos, seq_len).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        output = self.apply_attention(xq, xk, xv, mask)

        return self.o_proj(output)


class LlamaFeedForwardHf(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * args.dim) / 3)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.hidden_dim = hidden_dim
        self.dim = args.dim
        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None

    def init_weights(self):
        self.gate_proj = ColumnParallelLinear(
            self.dim, self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            self.hidden_dim, self.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            self.dim, self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaTransformerBlockHf(nn.Module):
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__()
        self.args = args
        self.self_attn = LlamaAttentionHf(args)
        self.mlp = LlamaFeedForwardHf(args)
        self.layer_id = layer_id
        self.clamp = Clamp(disable=not args.use_clamp)

        self.input_layernorm = None
        self.post_attention_layernorm = None

    def init_weights(self):
        self.self_attn.init_weights()
        self.mlp.init_weights()
        self.input_layernorm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.post_attention_layernorm = RMSNorm(self.args.dim, eps=self.args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache
    ):
        h = x + self.self_attn.forward(self.input_layernorm(x), start_pos, mask, use_cache)
        h = self.clamp.forward(h)
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        out = self.clamp.forward(out)
        return out


class LlamaModelHf(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.args = args

        self.embed_tokens = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LlamaTransformerBlockHf(layer_id, args))
        self.norm = None

    def init_weights(self):
        self.embed_tokens = ParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        )
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seq_len = tokens.shape
        h = self.embed_tokens(tokens)

        mask = None
        if seq_len > 1:
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, mask, use_cache)
        return self.norm(h)


class LlamaHf(ParallelModelForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.local_rank, args.world_size)
        self.args = args
        self.model = LlamaModelHf(args)
        self.lm_head = None

    def init_weights(self):
        self.model.init_weights()
        self.lm_head = ColumnParallelLinear(
            self.args.dim, self.args.vocab_size, bias=False, init_method=lambda x: x
        )

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos=0,
            use_cache=False
    ):
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h)
        return CausalLMOutputs(logits=logits_normalize(output), hidden_states=h)

    def load(self, ckpt_dir: str, verbose: bool = True, **kwargs):
        checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
        if len(checkpoints) != 0:  # normal loading
            super().load(ckpt_dir, verbose, **kwargs)
        else:  # splitting
            pl_ckpt_dir = auto_split_huggingface_checkpoints(
                ckpt_dir, world_size=self.world_size, local_rank=self.local_rank, verbose=verbose
            )
            set_barrier()
            super().load(pl_ckpt_dir, verbose, **kwargs)

    def flush(self):
        """ Clean cache in `LlamaAttention` module """
        for i in range(self.args.n_layers):
            self.model.layers[i].self_attn.flush()
        set_barrier()


class LoraLlamaAttentionHf(LlamaAttentionHf):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args

        self.lora_a_q_proj = None
        self.lora_b_q_proj = None
        self.lora_a_k_proj = None
        self.lora_b_k_proj = None
        self.lora_a_v_proj = None
        self.lora_b_v_proj = None
        self.lora_a_o_proj = None
        self.lora_b_o_proj = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_q_proj = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_q_proj = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_k_proj = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_k_proj = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_v_proj = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_v_proj = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_o_proj = RowParallelLinear(
            self.args.n_heads * self.head_dim,
            self.args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).type(self.args.lora_dtype)
        self.lora_b_o_proj = nn.Linear(
            self.args.r,
            self.args.dim,
            bias=False
        ).type(self.args.lora_dtype)
        init.zeros_(self.lora_b_wo.weight)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x) + apply_lora(x, self.lora_a_q_proj, self.lora_b_q_proj)
        xk = self.k_proj(x) + apply_lora(x, self.lora_a_k_proj, self.lora_b_k_proj)
        xv = self.v_proj(x) + apply_lora(x, self.lora_a_v_proj, self.lora_b_v_proj)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_heads, self.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seq_len + start_pos)
        position_ids = compute_position_ids(start_pos, seq_len).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        output = self.apply_attention(xq, xk, xv, mask)

        return self.o_proj(output) + apply_lora(output, self.lora_a_o_proj, self.lora_b_o_proj)


class LoraLlamaFeedForwardHf(LlamaFeedForwardHf):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args
        self.r = args.r

        self.lora_a_gate_proj = None
        self.lora_b_gate_proj = None
        self.lora_a_down_proj = None
        self.lora_b_down_proj = None
        self.lora_a_up_proj = None
        self.lora_b_up_proj = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_gate_proj = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_gate_proj = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_down_proj = RowParallelLinear(
            self.hidden_dim,
            self.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).type(self.args.lora_dtype)
        self.lora_b_down_proj = nn.Linear(
            self.r,
            self.dim,
            bias=False
        ).type(self.args.lora_dtype)
        init.zeros_(self.lora_b_w2.weight)
        self.lora_a_up_proj = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_up_proj = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)

    def forward(self, x):
        w1_x = self.gate_proj(x) + apply_lora(x, self.lora_a_gate_proj, self.lora_b_gate_proj)
        w3_x = self.up_proj(x) + apply_lora(x, self.lora_a_up_proj, self.lora_b_up_proj)
        out = F.silu(w1_x) * w3_x
        return self.down_proj(out) + apply_lora(out, self.lora_a_down_proj, self.lora_b_down_proj)


class LoraLlamaTransformerBlockHf(LlamaTransformerBlockHf):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.self_attn = LoraLlamaAttentionHf(args)
        self.mlp = LoraLlamaFeedForwardHf(args)


class LoraLlamaModelHf(LlamaModelHf):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LlamaTransformerBlockHf(layer_id, args))


class LoraLlamaHf(LlamaHf):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args
        self.model = LoraLlamaModelHf(args)
        self.lora_a_lm_head = None
        self.lora_b_lm_head = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_lm_head = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_lm_head = ColumnParallelLinear(
            self.args.r,
            self.args.vocab_size,
            bias=False,
            gather_output=True,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)

        # Freeze parameters
        self._freeze()

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h) + apply_lora(h, self.lora_a_lm_head, self.lora_b_lm_head)
        return CausalLMOutputs(logits=output, hidden_states=h)

    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)
