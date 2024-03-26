from typing import List

from transformers import LlamaTokenizerFast

from src.tokenizers import Tokenizer
from src.tokenizers.tokenizer_llama import LlamaTokenizer


class MistralTokenizer(LlamaTokenizer):
    def __init__(self, model_file: str):
        super().__init__(model_file)


class MistralChatTokenizer(Tokenizer):
    def __init__(self, model_dir: str):
        self.model = LlamaTokenizerFast.from_pretrained(model_dir)
        super().__init__(
            vocab_size=self.model.vocab_size,
            bos_id=self.model.bos_token_id,
            eos_id=self.model.eos_token_id,
            pad_id=self.model.eos_token_id
        )

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert eos is False  # TODO
        s = self.model.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=False,
            add_generation_prompt=True
        )
        return self.model.encode(s)

    def decode(self, t: List[int]) -> str:
        return self.model.decode(t, skip_special_tokens=True)
