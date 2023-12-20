from typing import List

import torch

from src.model import LLaMA
from src.tokenizer import Tokenizer
from src.utils import sample_top_p


class Generator:
    def __init__(self, model: LLaMA, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
            self,
            prompts: List[str],
            max_gen_len: int,
            temperature: float = 0.8,
            top_p: float = 0.95,
    ) -> List[dict]:
        bsz = len(prompts)
        params = self.model.params
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long).cuda()
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, use_cache=True)[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            unfinished_sequences = unfinished_sequences * (
                    next_token != self.tokenizer.eos_id).cuda().long()
            if unfinished_sequences.max() == 0:
                break
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            prompt_length = len(prompt_tokens[i])
            # cut to max gen len
            t = t[: prompt_length + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(dict(
                instruction=self.tokenizer.decode(t[:prompt_length]),
                output=self.tokenizer.decode(t[prompt_length:])))
        self.model.flush()
        return decoded
