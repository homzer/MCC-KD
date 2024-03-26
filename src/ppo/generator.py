import collections
from typing import List, Union

import numpy as np
import torch

from src.generator import GeneratorForVerifier
from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, ParallelVerifier, Verifier
from src.tokenizers import Tokenizer
from src.utils import sample_top_p, truncate

ActionGeneratorOutputs = collections.namedtuple("ActionGeneratorOutputs", [
    'outputs', 'logits', 'hidden_states', 'obs', 'actions', 'action_logits', 'action_masks'
])

SolverGeneratorOutputs = collections.namedtuple("SolverGeneratorOutputs", [
    'outputs', 'actions', 'action_masks'
])

LogitsGeneratorOutputs = collections.namedtuple("LogitsGeneratorOutputs", [
    'logits'
])


def sampling_strategy(logits: torch.Tensor, t: float, p: float):
    assert len(logits.shape) == 3
    seq_length = logits.shape[1]
    # only perform sampling on the last token
    last_logits = logits[:, -1, :]  # [b, v]
    if t > 0:  # Top-p Sampling
        next_tokens = sample_top_p(torch.softmax(last_logits / t, dim=-1), p)
    else:  # Greedy Sampling
        next_tokens = torch.argmax(last_logits, dim=-1, keepdim=True)
    if seq_length > 1:
        next_tokens = torch.cat([torch.argmax(logits[:, :-1, :], dim=-1), next_tokens], dim=-1)
    return next_tokens  # [b, s]


class SolverGeneratorForCausalLM:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        self.model = model
        self.vocab_size = tokenizer.vocab_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def _prepare_for_generation(self, prompts: List[str], eos: bool = False):
        bsz = len(prompts)
        prompt_tokens = []
        for x in prompts:
            x = self.tokenizer.encode(x, bos=True, eos=eos)
            prompt_tokens.append(x[: self.max_seq_len])
        min_prompt_size = min([len(t) for t in prompt_tokens])
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_masks = tokens != self.tokenizer.pad_id
        Outputs = collections.namedtuple("Outputs", [
            'tokens', 'input_masks', 'start_pos'
        ])
        return Outputs(
            tokens=tokens.to(self.model.device()),
            input_masks=input_masks.to(self.model.device()),
            start_pos=min_prompt_size,
        )

    def _model_forward(self, tokens, input_masks=None, start_pos=None, t=0.0, p=0.8):
        bsz = tokens.shape[0]
        prev_pos = 0
        tokens = tokens.clone()
        hidden_states = None
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long, device=self.model.device())
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                outputs = self.model.forward(
                    tokens[:, prev_pos: cur_pos], prev_pos, use_cache=True
                )
            if hidden_states is None:
                hidden_states = torch.zeros(
                    (*tokens.shape, outputs.hidden_states.shape[-1]),
                ).to(outputs.hidden_states)
            hidden_states[:, prev_pos: cur_pos, :] = outputs.hidden_states
            next_tokens = sampling_strategy(outputs.logits, t, p)  # [b, s]
            next_token = next_tokens[:, -1].reshape(-1)
            next_token = torch.where(
                input_masks[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            unfinished_sequences = unfinished_sequences * (
                    next_token != self.tokenizer.eos_id
            ).long()
            if unfinished_sequences.max() == 0:
                break

        self.model.flush()
        Outputs = collections.namedtuple("Outputs", ['tokens'])
        return Outputs(tokens=tokens)

    def _get_output_masks(self, tokens, prompt_lengths):
        output_masks = torch.full_like(tokens, fill_value=True)
        for i, t in enumerate(tokens.tolist()):
            output_masks[i][: prompt_lengths[i] - 1] = False
            if self.tokenizer.eos_id in t[1:]:
                # find index of eos
                end = t.index(self.tokenizer.eos_id, 1)
                output_masks[i][end:] = False
            else:
                output_masks[i][-1:] = False
        return output_masks.to(torch.bool)

    def _decode_response(self, tokens, output_masks):
        responses = []
        # shift right
        shifted_output_masks = torch.full_like(output_masks, fill_value=False)
        shifted_output_masks[:, 1:] = output_masks[:, :-1]
        for t, m in zip(tokens, shifted_output_masks):
            responses.append(self.tokenizer.decode(t[m].tolist()))
        return responses

    def forward(self, instructions: List[str], t: float = 0.0, p: float = 0.8) -> SolverGeneratorOutputs:
        self.model.eval()
        prep_outputs = self._prepare_for_generation(instructions)
        forward_outputs = self._model_forward(
            prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos, t=t, p=p
        )
        prompt_lengths = torch.sum(prep_outputs.input_masks, dim=-1)
        output_masks = self._get_output_masks(forward_outputs.tokens, prompt_lengths)
        outputs = self._decode_response(forward_outputs.tokens, output_masks)
        # input tokens shift left to get output tokens
        output_tokens = torch.zeros_like(forward_outputs.tokens)
        output_tokens[:, :-1] = forward_outputs.tokens[:, 1:]
        return SolverGeneratorOutputs(
            outputs=outputs,
            actions=output_tokens,
            action_masks=output_masks,
        )


class LogitsGeneratorForCausalLM:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        self.model = model
        self.vocab_size = tokenizer.vocab_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    # def _truncating_strategy(self, instruction_ids, output_ids):
    #     """ TODO: duplicated code with `Trainer` """
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

    def _prepare_for_generation(self, instructions, outputs):
        bsz = len(instructions)
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            instruction_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
            output_ids = self.tokenizer.encode(output, bos=False, eos=True)
            instruction_ids, output_ids = truncate(instruction_ids, output_ids, self.max_seq_len)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
        Output = collections.namedtuple('Outputs', ['tokens'])
        return Output(tokens=tokens)

    def _model_forward(self, tokens):
        tokens = tokens.clone()
        with torch.no_grad():
            outputs = self.model.forward(tokens)

        Outputs = collections.namedtuple('Outputs', ['logits'])
        return Outputs(logits=outputs.logits)

    def forward(self, instructions: List[str], outputs: List[str]) -> LogitsGeneratorOutputs:
        self.model.eval()
        prep_outputs = self._prepare_for_generation(instructions, outputs)
        forward_outputs = self._model_forward(prep_outputs.tokens)
        return LogitsGeneratorOutputs(forward_outputs.logits)


class LogitsGeneratorForCausalLMV0(SolverGeneratorForCausalLM):
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)

    def _model_forward(self, tokens, input_masks=None, start_pos=None, t=0.0, p=0.8):
        tokens = tokens.clone()
        with torch.no_grad():
            outputs = self.model.forward(tokens)

        Outputs = collections.namedtuple('Outputs', ['logits'])
        return Outputs(logits=outputs.logits)

    def forward(self, instructions: List[str], t: float = 0.0, p: float = 0.8) -> LogitsGeneratorOutputs:
        self.model.eval()
        prep_outputs = self._prepare_for_generation(instructions, eos=True)
        forward_outputs = self._model_forward(prep_outputs.tokens)
        return LogitsGeneratorOutputs(forward_outputs.logits)


class ActorGeneratorForCausalLM(SolverGeneratorForCausalLM):
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)

    def _model_forward(self, tokens, input_masks=None, start_pos=None, t=0.0, p=0.8):
        bsz = tokens.shape[0]
        prev_pos = 0
        tokens = tokens.clone()
        hidden_states = None
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long, device=self.model.device())
        logits = torch.zeros((*tokens.shape, self.vocab_size))
        tokens_logits = torch.zeros(tokens.shape)
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                outputs = self.model.forward(
                    tokens[:, prev_pos: cur_pos], prev_pos, use_cache=True
                )
            if hidden_states is None:
                hidden_states = torch.zeros(
                    (*tokens.shape, outputs.hidden_states.shape[-1]),
                    device=self.model.device()
                )
            hidden_states[:, prev_pos: cur_pos, :] = outputs.hidden_states
            logits = logits.to(outputs.logits)
            logits[:, prev_pos: cur_pos, :] = outputs.logits
            next_tokens = sampling_strategy(outputs.logits, t, p)
            tokens_logits = tokens_logits.to(outputs.logits)
            tokens_logits[:, prev_pos: cur_pos] = torch.gather(
                outputs.logits, dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)
            next_token = next_tokens[:, -1].reshape(-1)
            next_token = torch.where(
                input_masks[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            unfinished_sequences = unfinished_sequences * (
                    next_token != self.tokenizer.eos_id
            ).long()
            if unfinished_sequences.max() == 0:
                break

        self.model.flush()
        Outputs = collections.namedtuple("Outputs", [
            'tokens', 'hidden_states', 'logits', 'tokens_logits'])
        return Outputs(tokens=tokens, hidden_states=hidden_states, logits=logits, tokens_logits=tokens_logits)

    def forward(self, instructions: List[str], t: float = 0.0, p: float = 0.8) -> ActionGeneratorOutputs:
        self.model.eval()
        prep_outputs = self._prepare_for_generation(instructions)
        forward_outputs = self._model_forward(
            prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos, t=t, p=p
        )

        prompt_lengths = torch.sum(prep_outputs.input_masks, dim=-1)
        output_masks = self._get_output_masks(forward_outputs.tokens, prompt_lengths)
        outputs = self._decode_response(forward_outputs.tokens, output_masks)
        # input tokens shift left to get output tokens
        output_tokens = torch.zeros_like(forward_outputs.tokens)
        output_tokens[:, :-1] = forward_outputs.tokens[:, 1:]
        return ActionGeneratorOutputs(
            outputs=outputs,
            logits=forward_outputs.logits,
            hidden_states=forward_outputs.hidden_states,
            obs=forward_outputs.tokens,
            actions=output_tokens,
            action_logits=forward_outputs.tokens_logits,
            action_masks=output_masks,
        )


class CriticGeneratorForCausalLM:
    def __init__(self, verifier: Union[Verifier, ParallelVerifier], tokenizer: Tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.generator = GeneratorForVerifier(verifier, tokenizer, max_seq_len)

    def forward(self, obs: List[str], actions: np.ndarray, action_masks: np.ndarray) -> List[List[float]]:
        outputs = []
        for action, action_mask in zip(actions, action_masks):
            outputs.append(action[action_mask].tolist())
        return self.generator.forward(obs, outputs).tokens_scores
