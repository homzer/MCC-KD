import collections
from typing import List

import torch
import torch.nn as nn

from src.models.modeling import ModelForCausalLM, Module, ParallelModule, ParallelModelForCausalLM
from src.ppo.generator import ActorGeneratorForCausalLM

PolicyForwardOutputs = collections.namedtuple(
    "PolicyForwardOutputs", ["obs", "actions", "values", "action_logits", "action_masks"]
)
PolicyEvaluateOutputs = collections.namedtuple(
    "PolicyEvaluateOutputs", ["values", "action_logits"]
)
ActorForwardOutputs = collections.namedtuple(
    "ActorForwardOutputs", ["obs", "actions", "action_logits", "action_masks"]
)
CriticForwardOutputs = collections.namedtuple(
    "CriticForwardOutputs", ["values"]
)


class AbstractPolicyForCausalLM(Module):
    """ Abstract Actor-Critic Policy """
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> PolicyForwardOutputs:
        raise NotImplementedError

    def evaluate_actions(self, *args, **kwargs) -> PolicyEvaluateOutputs:
        raise NotImplementedError

    def predict_values(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def predict_actions(self, *args, **kwargs):
        raise NotImplementedError


class AbstractParallelPolicyForCausalLM(ParallelModule):
    """ Abstract Actor-Critic Policy """
    def __init__(self, local_rank, world_size):
        super().__init__(local_rank, world_size)

    def init_weights(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> PolicyForwardOutputs:
        raise NotImplementedError

    def evaluate_actions(self, *args, **kwargs) -> PolicyEvaluateOutputs:
        raise NotImplementedError

    def predict_values(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def predict_actions(self, *args, **kwargs):
        raise NotImplementedError


class ActorCriticPolicyForCausalLM(AbstractPolicyForCausalLM):
    def __init__(self, model: ModelForCausalLM, generator: ActorGeneratorForCausalLM, dim: int):
        super().__init__()
        self.model = model
        self.ln = nn.LayerNorm(dim, elementwise_affine=False).float()
        self.value = nn.Linear(dim, 1, bias=False).float()
        self.generator = generator

    def init_weights(self):
        self.model.init_weights()

    def forward(self, obs: List[str]) -> PolicyForwardOutputs:
        outputs = self.generator.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        return PolicyForwardOutputs(
            obs=outputs.input_tokens,  # [b, s]
            actions=outputs.output_tokens,  # [b, s]
            values=values,  # [b, s]
            action_logits=outputs.tokens_logits,
            action_masks=outputs.output_masks
        )

    def predict_values(self, obs) -> torch.Tensor:
        outputs = self.model.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        return values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> PolicyEvaluateOutputs:
        outputs = self.model.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        actions_logits = torch.gather(
            outputs.logits,
            dim=-1,
            index=actions.unsqueeze(-1)
        ).squeeze(-1)

        return PolicyEvaluateOutputs(
            values=values,
            action_logits=actions_logits,
        )

    def predict_actions(self, prompts: List[str]) -> List[dict]:
        outputs = self.generator.forward(prompts)
        return outputs.input_tokens  # TODO


class ParallelActorCriticPolicyForCausalLM(AbstractParallelPolicyForCausalLM):
    def __init__(self, model: ParallelModelForCausalLM, generator: ActorGeneratorForCausalLM, dim: int):
        super().__init__(model.local_rank, model.world_size)
        self.model = model
        self.ln = nn.LayerNorm(dim, elementwise_affine=False).float()
        self.value = nn.Linear(dim, 1, bias=False).float()
        self.generator = generator

    def init_weights(self):
        self.model.init_weights()

    def forward(self, obs: List[str]) -> PolicyForwardOutputs:
        outputs = self.generator.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        return PolicyForwardOutputs(
            obs=outputs.input_tokens,  # [b, s]
            actions=outputs.output_tokens,  # [b, s]
            values=values,  # [b, s]
            action_logits=outputs.tokens_logits,
            action_masks=outputs.output_masks
        )

    def predict_values(self, obs) -> torch.Tensor:
        outputs = self.model.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        return values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> PolicyEvaluateOutputs:
        outputs = self.model.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        actions_logits = torch.gather(
            outputs.logits,
            dim=-1,
            index=actions.unsqueeze(-1)
        ).squeeze(-1)

        return PolicyEvaluateOutputs(
            values=values,
            action_logits=actions_logits,
        )

    def predict_actions(self, prompts: List[str]) -> List[dict]:
        outputs = self.generator.forward(prompts)
        return outputs.input_tokens  # TODO
