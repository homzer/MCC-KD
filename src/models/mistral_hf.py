from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)

from src.checkpoint import auto_split_huggingface_checkpoints
from src.models.mistral import repeat_kv
from src.models.modeling import ParallelModelForCausalLM, CausalLMOutputs, AttentionForCausalLM
from src.models.modeling_acts import RMSNorm, Clamp, RotaryEmbedding
from src.models.modeling_args import MistralArgsHf
from src.utils import set_barrier, compute_position_ids, apply_rotary_pos_emb


class MistralAttentionHf(AttentionForCausalLM):
    def __init__(self, args: MistralArgsHf):
        super().__init__(args.max_seq_len)
        self.args = args
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.num_local_heads = args.num_attention_heads // args.world_size
        self.num_key_value_heads = args.num_key_value_heads
        self.num_local_key_value_heads = self.num_key_value_heads // args.world_size
        self.n_rep = args.num_attention_heads // args.num_key_value_heads

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        self.rotary_emb = None

    def init_weights(self):
        self.q_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.args.num_attention_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.k_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.args.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.v_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.args.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.o_proj = RowParallelLinear(
            self.args.num_attention_heads * self.head_dim,
            self.args.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.args.max_position_embeddings,
            base=self.args.rope_theta
        )

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: torch.Tensor = None,
            use_cache: bool = False
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seqlen, self.num_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_local_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_local_key_value_heads, self.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seqlen + start_pos)
        position_ids = compute_position_ids(start_pos, seqlen).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)
        else:
            xk, xv = repeat_kv(xk, xv, self.n_rep)

        output = self.apply_attention(xq, xk, xv, mask[None, None, ...] if mask is not None else None)

        return self.o_proj(output)


class MistralFeedForwardHf(nn.Module):
    def __init__(self, args: MistralArgsHf):
        super().__init__()
        self.args = args

        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None

    def init_weights(self):
        self.gate_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.args.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            self.args.intermediate_size,
            self.args.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.args.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )

    def forward(self, x) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MistralTransformerBlockHf(nn.Module):
    def __init__(self, args: MistralArgsHf):
        super().__init__()
        self.args = args
        self.self_attn = MistralAttentionHf(args)
        self.mlp = MistralFeedForwardHf(args)
        self.clamp = Clamp(disable=not args.use_clamp)

        self.input_layernorm = None
        self.post_attention_layernorm = None

    def init_weights(self):
        self.self_attn.init_weights()
        self.mlp.init_weights()
        self.input_layernorm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache: bool
    ) -> torch.Tensor:
        h = x + self.self_attn.forward(self.input_layernorm(x), start_pos, mask, use_cache)
        h = self.clamp.forward(h)
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        out = self.clamp.forward(out)
        return out


class MistralModelHf(nn.Module):
    def __init__(self, args: MistralArgsHf):
        super().__init__()
        self.args = args

        self.embed_tokens = None
        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(MistralTransformerBlockHf(args))
        self.norm = None

    def init_weights(self):
        self.embed_tokens = ParallelEmbedding(
            self.args.vocab_size, self.args.hidden_size, init_method=lambda x: x
        )
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int,
            use_cache: bool
    ):
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


class MistralHf(ParallelModelForCausalLM):
    def __init__(self, args: MistralArgsHf):
        super().__init__(args.local_rank, args.world_size)
        self.args = args
        self.model = MistralModelHf(args)
        self.lm_head = None

    def init_weights(self):
        self.model.init_weights()
        self.lm_head = ColumnParallelLinear(
            self.args.hidden_size, self.args.vocab_size, bias=False, init_method=lambda x: x
        )

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h)
        return CausalLMOutputs(logits=output, hidden_states=h)

    # Copied from llama_hf.LlamaHf.load
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

    # Copied from llama_hf.LlamaHf.flush
    def flush(self):
        for i in range(self.args.num_hidden_layers):
            self.model.layers[i].self_attn.flush()
        set_barrier()
