import collections
from typing import List, Union

import torch

from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, ParallelVerifier, Verifier
from src.tokenizers.tokenizer import Tokenizer
from src.utils import sample_top_p, masked_mean, truncate


class GeneratorForCausalLM:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int
    ):
        self.model = model
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def forward(self, instructions: List[str], t: float = 0.0, p: float = 0.5) -> List[dict]:
        self.model.eval()
        bsz = len(instructions)
        prompt_tokens = []
        for x in instructions:
            x = self.tokenizer.encode(x, bos=True, eos=False)
            prompt_tokens.append(x[: self.max_seq_len])
        min_prompt_size = min([len(t) for t in prompt_tokens])
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).cuda().long()
        for k, tks in enumerate(prompt_tokens):
            tokens[k, : len(tks)] = torch.tensor(tks).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long).cuda()
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                outputs = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, use_cache=True)
                logits = outputs.logits[:, -1, :]
            if t > 0:
                probs = torch.softmax(logits / t, dim=-1)
                next_token = sample_top_p(probs, p)
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
        for i, tks in enumerate(tokens.tolist()):
            # cut to max gen len
            tks = tks[: self.max_seq_len]
            prompt_length = len(prompt_tokens[i])
            instruction = self.tokenizer.decode(tks[:prompt_length])
            tks = tks[prompt_length:]
            # cut to eos tok if any
            if self.tokenizer.eos_id in tks:
                tks = tks[: tks.index(self.tokenizer.eos_id)]
            output = self.tokenizer.decode(tks)
            decoded.append(dict(
                instruction=instruction,
                output=output
            ))
        self.model.flush()
        return decoded


class GeneratorForVerifier:
    def __init__(
            self,
            model: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            max_seq_len: int,
            reduce: str = "mean"
    ):
        self.model = model
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.reduce = reduce
        assert self.reduce in ["mean", "last"]

    # def _truncating_strategy(self, instruction_ids, output_ids):
    #     instruction_length = len(instruction_ids)
    #     output_length = len(output_ids)
    #     if instruction_length >= self.max_seq_len:
    #         print(f'WARNING: Length of instruction {instruction_length} '
    #               f'exceeds the max input length {self.max_seq_len}')
    #         instruction_ids = instruction_ids[:self.max_seq_len]
    #         instruction_length = len(instruction_ids)
    #     sequence_length = instruction_length + output_length
    #     if sequence_length > self.max_seq_len:
    #         exceed_length = sequence_length - self.max_seq_len
    #         output_ids = output_ids[:-exceed_length]
    #     return instruction_ids, output_ids

    def _prepare_for_generation(
            self,
            instructions: Union[List[str], List[List[int]]],
            outputs: Union[List[str], List[List[int]]]
    ):
        bsz = len(instructions)
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        masks = torch.full((bsz, self.max_seq_len), False)
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            if type(instruction) is str:
                instruction_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
            elif type(instruction) is list:
                assert type(instruction[0]) is int, type(instruction[0])
                instruction_ids = instruction
            else:
                raise TypeError(type(instruction))
            if type(output) is str:
                output_ids = self.tokenizer.encode(output, bos=False, eos=True)
            elif type(output) is list:
                assert type(output[0]) is int, type(output[0])
                output_ids = output
            else:
                raise TypeError(type(output))
            instruction_ids, output_ids = truncate(instruction_ids, output_ids, self.max_seq_len)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            masks[i, instr_len: instr_len + output_len] = True
        Output = collections.namedtuple('Outputs', ['tokens', 'masks'])
        return Output(tokens=tokens, masks=masks)

    def forward(self, instructions: Union[List[str], List[List[int]]], outputs: Union[List[str], List[List[int]]]):
        self.model.eval()
        examples = self._prepare_for_generation(instructions, outputs)
        with torch.no_grad():
            tokens_scores = self.model.forward(examples.tokens).scores
        result_tokens_scores = []
        for i, score in enumerate(tokens_scores):
            result_tokens_scores.append(torch.masked_select(score, examples.masks[i]).tolist())
        if self.reduce == "mean":
            scores = masked_mean(tokens_scores, examples.masks).tolist()
        else:  # "last"
            scores = []
            for i, score in enumerate(tokens_scores):
                ids = examples.masks[i].nonzero()
                scores.append(score[ids[-1].item() if len(ids) > 0 else -1].item())
        Output = collections.namedtuple('Output', ['scores', 'tokens_scores'])
        return Output(scores=scores, tokens_scores=result_tokens_scores)
