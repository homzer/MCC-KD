from typing import List

from transformers import Qwen2Tokenizer

from src.tokenizers import Tokenizer


class QwenTokenizer(Tokenizer):
    def __init__(self, model_dir: str):
        self.model = Qwen2Tokenizer.from_pretrained(model_dir)
        super().__init__(
            vocab_size=self.model.vocab_size,
            bos_id=self.model.convert_tokens_to_ids('<|im_start|>'),
            eos_id=self.model.eos_token_id,
            pad_id=self.model.pad_token_id
        )

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        encode = self.model.encode(s)
        if bos and encode[0] != self.bos_id:
            encode.insert(0, self.bos_id)
        if eos:
            encode.append(self.eos_id)
        return encode

    def decode(self, t: List[int]) -> str:
        return self.model.decode(t, skip_special_tokens=True)


class QwenChatTokenizer(QwenTokenizer):
    def __init__(self, model_dir: str):
        super().__init__(model_dir)

    def apply_chat_template(self, conversation: List[dict]) -> str:
        """
        :param conversation: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "greetings!"}]
        :return:
        """
        return self.model.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert eos is False  # TODO
        s = self.apply_chat_template([{"role": "user", "content": s}])
        return super().encode(s, bos, eos)

    def decode(self, t: List[int]) -> str:
        return super().decode(t)

# import base64
# import os
# from typing import Dict, List
#
# import tiktoken
#
# from src.tokenizers.tokenizer import Tokenizer
#
# PAT_STR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|" \
#           r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
# END_OF_TEXT = "<|endoftext|>"
# IM_START = "<|im_start|>"
# IM_END = "<|im_end|>"
# EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
# SPECIAL_START_ID = 151643
# SPECIAL_TOKENS = tuple(
#     enumerate(
#         (
#                 (
#                     END_OF_TEXT,
#                     IM_START,
#                     IM_END,
#                 )
#                 + EXTRAS
#         ),
#         start=SPECIAL_START_ID,
#     )
# )
#
#
# def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
#     with open(tiktoken_bpe_file, "rb") as f:
#         contents = f.read()
#     return {
#         base64.b64decode(token): int(rank)
#         for token, rank in (line.split() for line in contents.splitlines() if line)
#     }
#
#
# class QWenTokenizer(Tokenizer):
#     def __init__(self, model_file: str):
#         assert os.path.isfile(model_file), model_file
#         self.merge_ranks = _load_tiktoken_bpe(model_file)
#         self.special_tokens = {
#             token: index
#             for index, token in SPECIAL_TOKENS
#         }
#         self.model = tiktoken.Encoding(
#             "QWen",
#             pat_str=PAT_STR,
#             mergeable_ranks=self.merge_ranks,
#             special_tokens=self.special_tokens,
#         )
#         super().__init__(
#             vocab_size=self.model.n_vocab,
#             bos_id=self.model.eot_token,
#             eos_id=self.model.eot_token,
#             pad_id=self.model.eot_token
#         )
#
#     def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
#         encode = self.model.encode(s, allowed_special="all", disallowed_special=())
#         if eos:
#             encode.append(self.eos_id)
#         return encode
#
#     def decode(self, t: List[int]) -> str:
#         t = [i for i in t if i < self.eos_id]
#         return self.model.decode(t)
