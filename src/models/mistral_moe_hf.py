from typing import Optional

import torch
import torch.nn as nn

from src.models import MistralHf
from src.models.mistral_hf import MistralModelHf, MistralTransformerBlockHf, MistralFeedForwardHf
from src.models.modeling_acts import RMSNorm
from src.models.modeling_args import MistralMoeArgsHf


class MistralMoeLayer(nn.Module):
    def __init__(self, args: MistralMoeArgsHf):
        super().__init__()
        assert args.num_local_experts > 0
        self.args = args
        self.experts = nn.ModuleList([MistralFeedForwardHf(args) for _ in range(args.num_local_experts)])
        self.gate = None

    def forward(self, x):
        x_squashed = x.view(-1, x.shape[-1])
        gate_logits = self.gate(x_squashed)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = nn.functional.softmax(weights, dim=1, dtype=torch.float).type_as(x)
        results = torch.zeros_like(x_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                x_squashed[batch_idx]
            )
        return results.view_as(x)

    def init_weights(self):
        for expert in self.experts:
            expert.init_weights()
        self.gate = nn.Linear(self.args.hidden_size, self.args.num_local_experts, bias=False)


class MistralMoeTransformerBlockHf(MistralTransformerBlockHf):
    def __init__(self, args: MistralMoeArgsHf):
        super().__init__(args)
        self.block_sparse_moe = MistralMoeLayer(args)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache: bool
    ) -> torch.Tensor:
        h = x + self.self_attn.forward(self.input_layernorm(x), start_pos, mask, use_cache)
        h = self.clamp.forward(h)
        out = h + self.block_sparse_moe.forward(self.post_attention_layernorm(h))
        out = self.clamp.forward(out)
        return out

    def init_weights(self):
        self.self_attn.init_weights()
        self.block_sparse_moe.init_weights()
        self.input_layernorm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)


class MistralMoeModelHf(MistralModelHf):
    def __init__(self, args: MistralMoeArgsHf):
        super().__init__(args)

        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(MistralMoeTransformerBlockHf(args))


class MistralMoeHf(MistralHf):
    def __init__(self, args: MistralMoeArgsHf):
        super().__init__(args)
        self.model = MistralMoeModelHf(args)
