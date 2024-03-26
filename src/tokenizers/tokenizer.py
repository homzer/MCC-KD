from typing import List


class Tokenizer:
    def __init__(self, vocab_size, bos_id, eos_id, pad_id):
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        raise NotImplementedError

    def decode(self, t: List[int]) -> str:
        raise NotImplementedError
