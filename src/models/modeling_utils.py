from src.models import (
    ParallelModule,
    LoraLlama30B,
    LoraMistral,
    LoraLlama70B,
    LoraLlama,
    Mistral,
    MistralHf,
    MistralMoeHf,
    Llama30B,
    Llama70B,
    Llama,
    Qwen
)
from src.models.modeling_args import (
    LlamaArgs,
    MistralArgs,
    LoraLlamaArgs,
    LoraMistralArgs,
    QwenArgsHf,
    MistralArgsHf,
    MistralMoeArgsHf
)
from src.tokenizers import (
    Tokenizer,
    LlamaTokenizer,
    MistralTokenizer,
    MistralChatTokenizer,
    QwenChatTokenizer,
    QwenTokenizer
)


ARGS = {
    "llama-1-7b": LlamaArgs,
    "llama-1-13b": LlamaArgs,
    "llama-2-7b": LlamaArgs,
    "llama-2-13b": LlamaArgs,
    "llama-1-30b": LlamaArgs,
    "llama-2-70b": LlamaArgs,
    "lora-llama-1-7b": LoraLlamaArgs,
    "lora-llama-1-13b": LoraLlamaArgs,
    "lora-llama-2-7b": LoraLlamaArgs,
    "lora-llama-2-13b": LoraLlamaArgs,
    "lora-llama-1-30b": LoraLlamaArgs,
    "lora-llama-2-70b": LoraLlamaArgs,

    "mistral-7b": MistralArgs,
    "lora-mistral-7b": LoraMistralArgs,
    "mistral-7b-instruct-v0.2": MistralArgsHf,
    "mixtral-8x7b-instruct-v0.1": MistralMoeArgsHf,

    "qwen-7b": QwenArgsHf,
    "qwen-14b": QwenArgsHf,
    "qwen-72b": QwenArgsHf,
    "qwen-7b-chat": QwenArgsHf,
    "qwen-14b-chat": QwenArgsHf,
    "qwen-72b-chat": QwenArgsHf,
}


MODELS = {
    "llama-1-7b": Llama,
    "llama-1-13b": Llama,
    "llama-2-7b": Llama,
    "llama-2-13b": Llama,
    "llama-1-30b": Llama30B,
    "llama-2-70b": Llama70B,
    "lora-llama-1-7b": LoraLlama,
    "lora-llama-1-13b": LoraLlama,
    "lora-llama-2-7b": LoraLlama,
    "lora-llama-2-13b": LoraLlama,
    "lora-llama-1-30b": LoraLlama30B,
    "lora-llama-2-70b": LoraLlama70B,

    "mistral-7b": Mistral,
    "lora-mistral-7b": LoraMistral,
    "mistral-7b-instruct-v0.2": MistralHf,
    "mixtral-8x7b-instruct-v0.1": MistralMoeHf,

    "qwen-7b": Qwen,
    "qwen-14b": Qwen,
    "qwen-72b": Qwen,
    "qwen-7b-chat": Qwen,
    "qwen-14b-chat": Qwen,
    "qwen-72b-chat": Qwen,
}

TOKENIZERS = {
    "llama-1-7b": LlamaTokenizer,
    "llama-1-13b": LlamaTokenizer,
    "llama-2-7b": LlamaTokenizer,
    "llama-2-13b": LlamaTokenizer,
    "llama-1-30b": LlamaTokenizer,
    "llama-2-70b": LlamaTokenizer,
    "lora-llama-1-7b": LlamaTokenizer,
    "lora-llama-1-13b": LlamaTokenizer,
    "lora-llama-2-7b": LlamaTokenizer,
    "lora-llama-2-13b": LlamaTokenizer,
    "lora-llama-1-30b": LlamaTokenizer,
    "lora-llama-2-70b": LlamaTokenizer,

    "mistral-7b": MistralTokenizer,
    "lora-mistral-7b": MistralTokenizer,
    "mistral-7b-instruct-v0.2": MistralChatTokenizer,
    "mixtral-8x7b-instruct-v0.1": MistralChatTokenizer,

    "qwen-7b": QwenTokenizer,
    "qwen-14b": QwenTokenizer,
    "qwen-72b": QwenTokenizer,
    "qwen-7b-chat": QwenChatTokenizer,
    "qwen-14b-chat": QwenChatTokenizer,
    "qwen-72b-chat": QwenChatTokenizer,
}


def get_parallel_model(
        model_type: str,
        config_file: str,
        local_rank: int,
        world_size: int,
        max_seq_len: int,
        tokenizer_file: str,
        lora_rank: int
) -> (ParallelModule, Tokenizer):
    if local_rank > 0:
        model_type = "lora-" + model_type
    args = ARGS[model_type](
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank
    ).from_json(config_file)
    model = MODELS[model_type](args)
    tokenizer = TOKENIZERS[model_type](tokenizer_file)
    model.init_weights()
    return model, tokenizer
