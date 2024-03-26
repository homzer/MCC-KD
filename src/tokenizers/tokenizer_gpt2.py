from typing import List

from transformers import GPT2Tokenizer as BaseGPT2Tokenizer

from src.tokenizers.tokenizer import Tokenizer


class GPT2Tokenizer(Tokenizer):
    def __init__(self, model_path: str):
        self.model = BaseGPT2Tokenizer.from_pretrained(model_path)
        super().__init__(
            vocab_size=self.model.vocab_size,
            bos_id=self.model.bos_token_id,
            eos_id=self.model.eos_token_id,
            pad_id=self.model.pad_token_id if self.model.pad_token_id else self.model.bos_token_id
        )

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        t = []
        if bos:
            t = [self.bos_id]
        t.extend(self.model.encode(s))
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: List[int]) -> str:
        return self.model.decode(t, skip_special_tokens=True)
