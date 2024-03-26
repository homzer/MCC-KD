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

from src.models.modeling import ParallelModelForCausalLM, CausalLMOutputs, AttentionForCausalLM, \
    ParallelVerifier, VerifierOutputs
from src.models.modeling_acts import RMSNorm, Clamp
from src.models.modeling_args import MistralArgs, LoraMistralArgs
from src.utils import apply_rotary_emb, precompute_freqs_cis, set_barrier, logits_normalize, apply_lora


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


class MistralAttention(AttentionForCausalLM):
    def __init__(self, args: MistralArgs):
        super().__init__(args.max_seq_len)
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads
        self.n_local_heads = self.n_heads // args.world_size
        self.n_local_kv_heads = self.n_kv_heads // args.world_size

        self.repeats = self.n_local_heads // self.n_local_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim ** -0.5

        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None

    def init_weights(self):
        self.wq = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.args.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            self.args.dim,
            self.args.n_kv_heads * self.args.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            self.args.dim,
            self.args.n_kv_heads * self.args.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            self.args.n_heads * self.args.head_dim,
            self.args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

    def apply_cache(self, xk, xv, start_pos):
        bsz, seqlen, n_heads, head_dim = xk.shape
        positions = torch.arange(start_pos, start_pos + seqlen).to(xk.device)
        if self.cache_k is None:
            self.cache_k = torch.empty(
                (
                    bsz,
                    self.args.sliding_window,
                    self.n_local_kv_heads,
                    self.args.head_dim,
                )
            ).cuda()
        if self.cache_v is None:
            self.cache_v = torch.empty(
                (
                    bsz,
                    self.args.sliding_window,
                    self.n_local_kv_heads,
                    self.args.head_dim,
                )
            ).cuda()

            # The cache is a rotating buffer
        scatter_pos = (positions[-self.sliding_window:] % self.sliding_window)[None, :, None, None]
        scatter_pos = scatter_pos.repeat(bsz, 1, self.n_local_kv_heads, self.args.head_dim)
        self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window:])
        self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window:])

        if positions.shape[0] > 1:
            # prefill
            xk, xv = repeat_kv(xk, xv, self.repeats)
        else:
            cur_pos = positions[-1].item() + 1
            xk, xv = repeat_kv(
                self.cache_k[:bsz, :cur_pos, ...],
                self.cache_v[:bsz, :cur_pos, ...],
                self.repeats
            )
        return xk, xv

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: torch.Tensor = None,
            use_cache: bool = False
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)
        else:
            xk, xv = repeat_kv(xk, xv, self.repeats)

        output = self.apply_attention(xq, xk, xv, mask[None, None, ...] if mask is not None else None)

        return self.wo(output)


class MistralFeedForward(nn.Module):
    def __init__(self, args: MistralArgs):
        super().__init__()
        self.args = args
        self.w1 = None
        self.w2 = None
        self.w3 = None

    def init_weights(self):
        self.w1 = ColumnParallelLinear(
            self.args.dim,
            self.args.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            self.args.hidden_dim,
            self.args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            self.args.dim,
            self.args.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )

    def forward(self, x) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MistralTransformerBlock(nn.Module):
    def __init__(self, args: MistralArgs):
        super().__init__()
        self.args = args
        self.attention = MistralAttention(args)
        self.feed_forward = MistralFeedForward(args)
        self.clamp = Clamp(disable=not args.use_clamp)

        self.attention_norm = None
        self.ffn_norm = None

    def init_weights(self):
        self.attention.init_weights()
        self.feed_forward.init_weights()
        self.attention_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.ffn_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            use_cache: bool
    ) -> torch.Tensor:
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, use_cache)
        h = self.clamp.forward(h)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        out = self.clamp.forward(out)
        return out


class Mistral(ParallelModelForCausalLM):
    def __init__(self, args: MistralArgs):
        super().__init__(args.local_rank, args.world_size)
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = None
        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(MistralTransformerBlock(args))
        self.norm = None
        self.output = None

        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128000)

    def init_weights(self):
        self.tok_embeddings = ParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        )
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.output = ColumnParallelLinear(
            self.args.dim, self.args.vocab_size, bias=False, init_method=lambda x: x
        )

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        positions = torch.arange(start_pos, start_pos + seqlen).to(h.device)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[positions]

        mask = None
        if seqlen > 1:
            tensor = torch.full((seqlen, seqlen), dtype=h.dtype, fill_value=1, device=h.device)
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            # make the mask banded to account for sliding window
            mask = torch.triu(mask, diagonal=-self.args.sliding_window)
            mask = torch.log(mask)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, use_cache)
        h = self.norm(h)
        output = self.output(h)

        return CausalLMOutputs(logits=logits_normalize(output), hidden_states=h)

    def flush(self):
        """ Clean cache in `LlamaAttention` module """
        for i in range(self.args.n_layers):
            self.layers[i].attention.flush()
        set_barrier()


class MistralVerifier(ParallelVerifier):
    def __init__(self, args: MistralArgs):
        super().__init__(args.local_rank, args.world_size)
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = None
        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(MistralTransformerBlock(args))
        self.norm = None
        self.v_head = None

        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128000)

    def init_weights(self):
        self.tok_embeddings = ParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        )
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.v_head = nn.Linear(self.args.dim, 1, bias=False)

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        positions = torch.arange(0, seqlen).to(h.device)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[positions]

        mask = None
        if seqlen > 1:
            tensor = torch.full((seqlen, seqlen), dtype=h.dtype, fill_value=1, device=h.device)
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            # make the mask banded to account for sliding window
            mask = torch.triu(mask, diagonal=-self.args.sliding_window)
            mask = torch.log(mask)

        for layer in self.layers:
            h = layer(h, freqs_cis, positions, mask, use_cache=False)
        h = self.norm(h)
        scores = self.v_head(h.type_as(self.v_head.weight)).squeeze(-1)  # [b, s]
        return VerifierOutputs(scores=scores)


class LoraMistralAttention(MistralAttention):
    def __init__(self, args: LoraMistralArgs):
        super().__init__(args)
        self.args = args
        self.lora_a_wq = None
        self.lora_b_wq = None
        self.lora_a_wk = None
        self.lora_b_wk = None
        self.lora_a_wv = None
        self.lora_b_wv = None
        self.lora_a_wo = None
        self.lora_b_wo = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_wq = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_wq = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.args.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_wk = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_wk = ColumnParallelLinear(
            self.args.r,
            self.args.n_kv_heads * self.args.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_wv = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_wv = ColumnParallelLinear(
            self.args.r,
            self.args.n_kv_heads * self.args.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_wo = RowParallelLinear(
            self.args.n_heads * self.args.head_dim,
            self.args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).type(self.args.lora_dtype)
        self.lora_b_wo = nn.Linear(
            self.args.r,
            self.args.dim,
            bias=False
        ).type(self.args.lora_dtype)
        init.zeros_(self.lora_b_wo.weight)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: torch.Tensor = None,
            use_cache: bool = False
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x) + apply_lora(x, self.lora_a_wq, self.lora_b_wq)
        xk = self.wk(x) + apply_lora(x, self.lora_a_wk, self.lora_b_wk)
        xv = self.wv(x) + apply_lora(x, self.lora_a_wv, self.lora_b_wv)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)
        else:
            xk, xv = repeat_kv(xk, xv, self.repeats)

        output = self.apply_attention(xq, xk, xv, mask[None, None, ...] if mask is not None else None)
        return self.wo(output) + apply_lora(output, self.lora_a_wo, self.lora_b_wo)


class LoraMistralFeedForward(MistralFeedForward):
    def __init__(self, args: LoraMistralArgs):
        super().__init__(args)
        self.args = args
        self.lora_a_w1 = None
        self.lora_b_w1 = None
        self.lora_a_w2 = None
        self.lora_b_w2 = None
        self.lora_a_w3 = None
        self.lora_b_w3 = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_w1 = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_w1 = ColumnParallelLinear(
            self.args.r,
            self.args.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_w2 = RowParallelLinear(
            self.args.hidden_dim,
            self.args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).type(self.args.lora_dtype)
        self.lora_b_w2 = nn.Linear(
            self.args.r,
            self.args.dim,
            bias=False
        ).type(self.args.lora_dtype)
        init.zeros_(self.lora_b_w2.weight)
        self.lora_a_w3 = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_w3 = ColumnParallelLinear(
            self.args.r,
            self.args.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)

    def forward(self, x) -> torch.Tensor:
        w1_x = self.w1(x) + apply_lora(x, self.lora_a_w1, self.lora_b_w1)
        w3_x = self.w3(x) + apply_lora(x, self.lora_a_w3, self.lora_b_w3)
        out = F.silu(w1_x) * w3_x
        return self.w2(out) + apply_lora(out, self.lora_a_w2, self.lora_b_w2)


class LoraMistralTransformerBlock(MistralTransformerBlock):
    def __init__(self, args: LoraMistralArgs):
        super().__init__(args)
        self.args = args
        self.attention = LoraMistralAttention(args)
        self.feed_forward = LoraMistralFeedForward(args)


class LoraMistral(Mistral):
    def __init__(self, args: LoraMistralArgs):
        super().__init__(args)
        self.args = args
        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(LoraMistralTransformerBlock(args))
        self.lora_a_output = None
        self.lora_b_output = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_output = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_output = ColumnParallelLinear(
            self.args.r,
            self.args.vocab_size,
            bias=False,
            gather_output=True,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)

        # Freeze parameters
        self._freeze()

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        positions = torch.arange(start_pos, start_pos + seqlen).to(h.device)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[positions]

        mask = None
        if seqlen > 1:
            tensor = torch.full((seqlen, seqlen), dtype=h.dtype, fill_value=1, device=h.device)
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            # make the mask banded to account for sliding window
            mask = torch.triu(mask, diagonal=-self.args.sliding_window)
            mask = torch.log(mask)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, use_cache)
        h = self.norm(h)
        output = self.output(h) + apply_lora(h, self.lora_a_output, self.lora_b_output)

        return CausalLMOutputs(logits=logits_normalize(output), hidden_states=h)

    # lora op
    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)


class LoraMistralVerifier(MistralVerifier):
    def __init__(self, args: LoraMistralArgs):
        super().__init__(args)
        self.args = args
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(LoraMistralTransformerBlock(args))

    def init_weights(self):
        super().init_weights()

        # Freeze parameters
        self._freeze()

    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name and 'v_head' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)
