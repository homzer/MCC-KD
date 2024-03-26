import os
from typing import List

from transformers import LlamaTokenizerFast

from src.tokenizers import Tokenizer


class OpenChatTokenizer(Tokenizer):
    def __init__(self, model_file: str):
        model_path, _ = os.path.split(model_file)
        config_file = os.path.join(model_path, 'tokenizer_config.json')
        assert os.path.isfile(model_file), model_file
        assert os.path.isfile(config_file), config_file
        self.model = LlamaTokenizerFast.from_pretrained(model_path)
        super().__init__(
            vocab_size=len(self.model),
            bos_id=self.model.bos_token_id,
            eos_id=self.model.eos_token_id,
            pad_id=32001
        )

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        encode = self.model.encode(s, add_special_tokens=bos)
        return encode.append(self.eos_id) if eos else encode

    def decode(self, t: List[int]) -> str:
        return self.model.decode(t, skip_special_tokens=True)
