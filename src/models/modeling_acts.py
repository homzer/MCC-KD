import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.mappings import (
    scatter_to_model_parallel_region,
    reduce_from_model_parallel_region,
    gather_from_model_parallel_region,
    copy_to_model_parallel_region
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class ColumnParallelLinearPartitioned(nn.Module):
    """ Warning: Partitioned size should be equal across all ranks """
    def __init__(self,
                 in_features: int,
                 output_size_per_partition: int,
                 bias: bool = True,
                 gather_output: bool = True,
                 init_method=nn.init.xavier_normal_):
        super().__init__()
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.Tensor(output_size_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size_per_partition))
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        init_method(self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class RowParallelLinearPartitioned(nn.Module):
    """ Warning: Partitioned size should be equal across all ranks """
    def __init__(self,
                 input_size_per_partition: int,
                 out_features: int,
                 bias: bool = True,
                 input_is_parallel: bool = False,
                 init_method=nn.init.xavier_normal_):
        super().__init__()
        self.input_is_parallel = input_is_parallel
        self.weight = nn.Parameter(torch.Tensor(out_features, input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        init_method(self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output


class Clamp:
    def __init__(self, disable: bool = False):
        self.disable = disable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clamp inf values to enable fp16 training.
        Will slow down speed, disable it when you don't need it.
        """
        if self.disable or not x.requires_grad:  # disable when inference
            return x
        if x.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(x).any(),
                torch.finfo(x.dtype).max - 1000,
                torch.finfo(x.dtype).max
            ).item()
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x


# Copied from Huggingface
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).type(self.args.lora_dtype).to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


# class MistralRotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
#         super().__init__()
#
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#
#         # Build here to make `torch.jit.trace` work.
#         self._set_cos_sin_cache(
#             seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
#         )
#
#     def _set_cos_sin_cache(self, seq_len, device, dtype):
#         self.max_seq_len_cached = seq_len
#         t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
#
#         freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
#         self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
#
#     def forward(self, x, seq_len=None):
#         # x: [bs, num_attention_heads, seq_len, head_size]
#         if seq_len > self.max_seq_len_cached:
#             self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
#
#         return (
#             self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#             self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#         )
