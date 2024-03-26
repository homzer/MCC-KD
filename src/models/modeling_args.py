import json
from dataclasses import dataclass

import torch
import transformers

HF_CONFIG_MAP = {
    "hidden_size": "dim",
    "num_attention_heads": "n_heads",
    "num_hidden_layers": "n_layers",
    "rms_norm_eps": "norm_eps",
}


class Args:
    def __post_init__(self):
        lora_dtype = getattr(self, 'lora_dtype', None)
        if lora_dtype is not None:
            if lora_dtype in ['float32', 'fp32']:
                setattr(self, 'lora_dtype', torch.float32)
            elif lora_dtype in ['float16', 'fp16']:
                setattr(self, 'lora_dtype', torch.float16)
            elif lora_dtype in ['bfloat16', 'bf16']:
                setattr(self, 'lora_dtype', torch.bfloat16)
            else:
                raise ValueError(lora_dtype)

    def _set_attribute(self, name, value):
        try:
            if getattr(self, name, None) is None:
                setattr(self, name, value)
        except AttributeError as err:
            print(f"Can't set `{name}` with value `{value}` for {self}")
            raise err

    def show(self):
        param_str = '\n'.join(['%30s = %s' % (k, v) for k, v in sorted(vars(self).items())])
        print('%30s   %s\n%s\n%s\n' % ('ATTRIBUTE', 'VALUE', '_' * 60, param_str))

    def from_json(self, filename: str):
        with open(filename, 'r', encoding='utf-8') as reader:
            config_dict = json.load(reader)
        for key, value in config_dict.items():
            if not hasattr(self, key):
                continue
            self._set_attribute(key, value)
        return self


@dataclass
class GPT2Args(Args):
    max_seq_len: int
    attn_pdrop: int = None
    embd_pdrop: int = None
    layer_norm_epsilon: float = None
    n_ctx: int = None
    n_embd: int = None
    n_head: int = None
    n_layer: int = None
    n_positions: int = None
    resid_pdrop: float = None
    vocab_size: int = None
    activation_function: str = None
    use_clamp: bool = False


@dataclass
class LlamaArgs(Args):
    max_seq_len: int
    local_rank: int
    world_size: int

    dim: int = None
    n_layers: int = None
    n_heads: int = None
    vocab_size: int = None  # defined later by tokenizer
    multiple_of: int = None  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = None

    # for 70b
    ffn_dim_multiplier: float = None
    n_kv_heads: int = None

    use_clamp: bool = False


@dataclass
class LoraLlamaArgs(LlamaArgs):
    r: int = None  # Rank of lora
    lora_dtype: str = "float32"


@dataclass
class MistralArgs(Args):
    max_seq_len: int
    local_rank: int
    world_size: int

    dim: int = None
    n_layers: int = None
    head_dim: int = None
    hidden_dim: int = None
    n_heads: int = None
    n_kv_heads: int = None
    norm_eps: float = None
    vocab_size: int = None

    # For rotary embeddings. If not set, will be infered from sliding window.
    rope_theta: float = None
    # If this is set, use sliding window attention rotating cache.
    sliding_window: int = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe = None

    use_clamp: bool = False


@dataclass
class MistralArgsHf(Args):
    max_seq_len: int
    local_rank: int
    world_size: int

    hidden_size: int = None
    num_hidden_layers: int = None
    intermediate_size: int = None
    max_position_embeddings: int = None
    num_attention_heads: int = None
    num_key_value_heads: int = None
    rms_norm_eps: float = None
    vocab_size: int = None

    # For rotary embeddings. If not set, will be infered from sliding window.
    rope_theta: int = None
    # If this is set, use sliding window attention rotating cache.
    sliding_window: int = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe = None

    use_clamp: bool = False


@dataclass
class OpenChatArgs(MistralArgs):
    pass


@dataclass
class LoraMistralArgs(MistralArgs):
    r: int = None
    lora_dtype: str = "float32"


@dataclass
class QwenArgsHf(Args):
    max_seq_len: int
    local_rank: int
    world_size: int

    hidden_size: int = None
    intermediate_size: int = None
    max_position_embeddings: int = None
    max_window_layers: int = None
    num_attention_heads: int = None
    num_hidden_layers: int = None
    num_key_value_heads: int = None
    rms_norm_eps: float = None
    rope_theta: int = None
    sliding_window: int = None
    tie_word_embeddings: bool = None
    use_sliding_window: bool = None
    vocab_size: int = None

    use_clamp: bool = False


class MistralMoeArgsHf(MistralArgsHf):
    num_local_experts: int
    num_experts_per_tok: int


class T5Config(transformers.T5Config):
    def __init__(self, max_input_len=128, max_output_len=384, **kwargs):
        super().__init__(**kwargs)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len


class LoraT5Config(T5Config):
    def __init__(self, r=16, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        self.lora_dtype = "float32"
