import collections
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import set_barrier, merge_lora_state_dict

CausalLMOutputs = collections.namedtuple('CausalLMOutputs', ['logits', 'hidden_states'])
Seq2SeqLMOutputs = collections.namedtuple('Seq2SeqLMOutputs', ['logits', 'hidden_states'])
MaskedLMOutputs = collections.namedtuple('MaskedLMOutputs', ['logits', 'hidden_states'])
VerifierOutputs = collections.namedtuple('VerifierOutputs', ['scores'])


class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def device(self):
        return next(self.parameters()).device

    def init_weights(self):
        raise NotImplementedError

    def load(self, ckpt_file: str, verbose: bool = True, **kwargs):
        if verbose:
            print(f'Loading model from {ckpt_file} .....')
        state_dict = torch.load(ckpt_file, map_location='cpu')
        outputs = self.load_state_dict(state_dict, strict=False)
        if verbose:
            for missing_key in outputs.missing_keys:
                print(f"MISSING KEY: {missing_key}")
            for unexpected_key in outputs.unexpected_keys:
                print(f"UNEXPECTED KEY: {unexpected_key}")
            print("Loading done!")
        return self

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f'Saving model to {save_dir} ......')
        torch.save(self.state_dict(), os.path.join(save_dir, f'pytorch_model.bin'))
        print(f'Saving done !')


# Decoder-only
class ModelForCausalLM(Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError


# Encoder-only
class ModelForMaskedLM(Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            tokens: torch.Tensor,
    ) -> MaskedLMOutputs:
        raise NotImplementedError


# Encoder-decoder
class ModelForSeq2SeqLM(Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            input_ids: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            use_cache: bool = False,
            start_pos: int = None
    ) -> Seq2SeqLMOutputs:
        raise NotImplementedError


class Verifier(Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        raise NotImplementedError


class ParallelModule(Module):
    def __init__(self, local_rank, world_size):
        super().__init__()
        self.local_rank = local_rank
        self.world_size = world_size

    def init_weights(self):
        raise NotImplementedError

    def load(self, ckpt_dir: str, verbose: bool = True, **kwargs):
        if verbose:
            print(f'Loading model from {ckpt_dir} .....')
        checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
        assert self.world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {self.world_size}"
        ckpt_path = checkpoints[self.local_rank]
        state_dict = torch.load(str(ckpt_path), map_location="cpu")
        if kwargs.get("merge_lora", False):
            state_dict = merge_lora_state_dict(state_dict)
        outputs = self.load_state_dict(state_dict, strict=False)
        self.cuda(self.local_rank)
        set_barrier()
        if verbose:
            for missing_key in outputs.missing_keys:
                print(f"MISSING KEY: {missing_key}")
            for unexpected_key in outputs.unexpected_keys:
                print(f"UNEXPECTED KEY: {unexpected_key}")
            print(f'Loading done !')

    def save(self, save_path):
        if self.local_rank == 0:
            os.makedirs(save_path, exist_ok=True)
        print(f'Saving model to {save_path} ......')
        set_barrier()
        torch.save(self.state_dict(), os.path.join(save_path, f'consolidated.0{self.local_rank}.pth'))
        set_barrier()
        print(f'Saving done !')


class ParallelModelForCausalLM(ParallelModule):
    def __init__(self, local_rank, world_size):
        super().__init__(local_rank, world_size)

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError


class ParallelModelForMaskedLM(ParallelModule):
    def __init__(self, local_rank, world_size):
        super().__init__(local_rank, world_size)

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            tokens: torch.Tensor,
    ) -> MaskedLMOutputs:
        raise NotImplementedError


class ParallelModelForSeq2SeqLM(ParallelModule):
    def __init__(self, local_rank, world_size):
        super().__init__(local_rank, world_size)

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            input_ids: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            use_cache: bool = False,
            start_pos: int = None
    ) -> Seq2SeqLMOutputs:
        raise NotImplementedError


class ParallelVerifier(ParallelModule):
    def __init__(self, local_rank, world_size):
        super().__init__(local_rank, world_size)

    def init_weights(self):
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        raise NotImplementedError


class AttentionForCausalLM(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.cache_k = None
        self.cache_v = None

    def apply_cache(self, xk, xv, start_pos):
        bsz, seqlen, n_heads, head_dim = xk.shape
        if self.cache_k is None:
            self.cache_k = torch.zeros(
                (bsz, self.max_seq_len, n_heads, head_dim)
            )
        if self.cache_v is None:
            self.cache_v = torch.zeros(
                (bsz, self.max_seq_len, n_heads, head_dim)
            )

        self.cache_k = self.cache_k.to(xk)
        self.cache_v = self.cache_v.to(xv)
        self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

        xk = self.cache_k[:bsz, : start_pos + seqlen]
        xv = self.cache_v[:bsz, : start_pos + seqlen]
        return xk, xv

    @staticmethod
    def apply_attention(xq, xk, xv, mask):
        bsz, seqlen, n_heads, head_dim = xq.shape
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return output

    def flush(self):
        """ Clean self.cache for next inference. """
        self.cache_v = None
        self.cache_k = None
