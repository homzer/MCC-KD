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
from src.models.modeling import ParallelModelForCausalLM, CausalLMOutputs, AttentionForCausalLM
from src.models.modeling_acts import Clamp, RMSNorm, RotaryEmbedding
from src.models.modeling_args import QwenArgsHf
from src.utils import logits_normalize, set_barrier, compute_position_ids, apply_rotary_pos_emb


# class Qwen2RotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
#         super().__init__()
#
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#
#         # Build here to make `torch.jit.trace` work.
#         self._set_cos_sin_cache(
#             seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
#         )
#
#     def _set_cos_sin_cache(self, seq_len, device, dtype):
#         self.max_seq_len_cached = seq_len
#         t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
#
#         freqs = torch.outer(t, self.inv_freq)
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
#         self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
#
#     def forward(self, x, seq_len=None):
#         # x: [bs, num_attention_heads, seq_len, head_size]
#         if seq_len > self.max_seq_len_cached:
#             self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
#
#         return (
#             self.cos_cached[:seq_len].to(dtype=x.dtype),
#             self.sin_cached[:seq_len].to(dtype=x.dtype),
#         )

#
# # Copied from transformers.models.llama.modeling_llama.rotate_half
# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2:]
#     return torch.cat((-x2, x1), dim=-1)
#
#
# # Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
# def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.
#
#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`):
#             The position indices of the tokens corresponding to the query and key tensors. For example, this can be
#             used to pass offsetted position ids when working with a KV-cache.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos[position_ids].unsqueeze(unsqueeze_dim)
#     sin = sin[position_ids].unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed
#
#
# # Copied from src.models.mistral_hf.compute_position_ids
# def compute_position_ids(start_pos: int, seq_length: int):
#     position_ids = torch.arange(
#         start_pos, seq_length + start_pos, dtype=torch.long
#     )
#     position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#     return position_ids


class QwenAttention(AttentionForCausalLM):
    def __init__(self, args: QwenArgsHf):
        super().__init__(args.max_seq_len)
        self.args = args
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.num_local_heads = args.num_attention_heads // args.world_size
        self.num_key_value_heads = args.num_key_value_heads
        self.num_local_key_value_heads = self.num_key_value_heads // args.world_size
        self.n_rep = args.num_attention_heads // args.num_key_value_heads

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.args.max_position_embeddings,
            base=self.args.rope_theta,
        )

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

    def init_weights(self):
        self.q_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.args.num_attention_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.k_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x
        )
        self.v_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
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

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seqlen, _ = x.size()
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

        xk = self.repeat_kv(xk)
        xv = self.repeat_kv(xv)

        output = self.apply_attention(xq, xk, xv, mask)
        return self.o_proj(output)

    # Copied from src.models.llama_70B.LlamaAttention70B.repeat_kv
    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        bs, seqlen, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, seqlen, n_kv_heads, self.n_rep, head_dim)
            .reshape(bs, seqlen, n_kv_heads * self.n_rep, head_dim)
        )


class QwenFeedForward(nn.Module):
    def __init__(self, args: QwenArgsHf):
        super().__init__()
        self.args = args

        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None

    def init_weights(self):
        self.gate_proj = ColumnParallelLinear(
            self.args.hidden_size, self.args.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            self.args.intermediate_size, self.args.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            self.args.hidden_size, self.args.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenTransformerBlock(nn.Module):
    def __init__(self, args: QwenArgsHf):
        super().__init__()
        self.args = args
        self.self_attn = QwenAttention(args)
        self.mlp = QwenFeedForward(args)
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
            use_cache
    ):
        h = x + self.self_attn.forward(self.input_layernorm(x), start_pos, mask, use_cache)
        h = self.clamp.forward(h)
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        out = self.clamp.forward(out)
        return out


class QwenHead(nn.Module):
    def __init__(self, args: QwenArgsHf):
        super().__init__()
        self.args = args

        self.embed_tokens = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_hidden_layers):
            self.layers.append(QwenTransformerBlock(args))
        self.norm = None

    def init_weights(self):
        self.embed_tokens = ParallelEmbedding(
            self.args.vocab_size, self.args.hidden_size, init_method=lambda x: x
        )
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)

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


class Qwen(ParallelModelForCausalLM):
    def __init__(self, args: QwenArgsHf):
        super().__init__(args.local_rank, args.world_size)
        self.args = args
        self.model = QwenHead(args)
        self.lm_head = None

    def init_weights(self):
        self.model.init_weights()
        self.lm_head = ColumnParallelLinear(
            self.args.hidden_size, self.args.vocab_size, bias=False, init_method=lambda x: x
        )

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h)
        return CausalLMOutputs(logits=logits_normalize(output), hidden_states=h)

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
