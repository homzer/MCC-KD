import collections

import torch

from src.criterion import MSELoss
from src.models.modeling import ParallelModelForCausalLM, ParallelVerifier
from src.ppo.buffer import RolloutBufferSample
from src.ppo.policy import AbstractPolicyForCausalLM, AbstractParallelPolicyForCausalLM
from src.trainer import ParallelTrainer, Trainer
from src.utils import masked_std


class PPOTrainerForCausalLM(Trainer):
    def __init__(self, policy: AbstractPolicyForCausalLM, optimizer: torch.optim.Optimizer):
        super().__init__(policy, optimizer)
        self.policy = policy
        self.optimizer = optimizer
        # TODO: schedule function
        self.clip_range = 0.07
        self.vf_coef = 0.1
        self.step = 0
        self.criterion = MSELoss()

    def forward(self, rollout_data: RolloutBufferSample):
        self.policy.train()
        self.step += 1

        obs = rollout_data.observations.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        advantages = rollout_data.advantages.to(self.policy.device())
        old_action_logits = rollout_data.old_action_logits.to(self.policy.device())
        returns = rollout_data.returns.to(self.policy.device())

        outputs = self.policy.evaluate_actions(obs=obs, actions=actions)

        # Normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.masked_select(advantages.view(-1), action_masks.view(-1))
        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(outputs.action_logits - old_action_logits)
        ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))
        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

        # Value loss using the TD(Temporal Difference)(gae_lambda) target
        # Regression training for value function (or critic)
        value_loss = self.criterion.forward(outputs.values, returns, action_masks)

        loss = policy_loss + self.vf_coef * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', ['loss', 'policy_loss', 'value_loss'])
        return Outputs(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item()
        )


class ParallelPPOTrainerForCausalLM(ParallelTrainer):
    def __init__(self, policy: AbstractParallelPolicyForCausalLM, optimizer: torch.optim.Optimizer):
        super().__init__(policy, optimizer)
        self.policy = policy
        self.optimizer = optimizer
        self.clip_range = 0.07
        self.vf_coef = 0.1
        self.step = 0
        self.criterion = MSELoss()

    def forward(self, rollout_data: RolloutBufferSample):
        self.policy.train()
        self.step += 1

        obs = rollout_data.observations.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        advantages = rollout_data.advantages.to(self.policy.device())
        old_action_logits = rollout_data.old_action_logits.to(self.policy.device())
        returns = rollout_data.returns.to(self.policy.device())

        outputs = self.policy.evaluate_actions(obs=obs, actions=actions)

        # Normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.masked_select(advantages.view(-1), action_masks.view(-1))
        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(outputs.action_logits - old_action_logits)
        ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))
        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

        # Value loss using the TD(Temporal Difference)(gae_lambda) target
        # Regression training for value function (or critic)
        value_loss = self.criterion.forward(outputs.values, returns, action_masks)

        loss = policy_loss + self.vf_coef * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', ['loss', 'policy_loss', 'value_loss'])
        return Outputs(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item()
        )


class ParallelActorTrainerForCausalLM(ParallelTrainer):
    def __init__(self, actor: ParallelModelForCausalLM, optimizer: torch.optim.Optimizer):
        super().__init__(actor, optimizer)
        self.actor = actor
        self.clip_range = 0.07
        self.step = 0

    def forward(self, rollout_data: RolloutBufferSample):
        self.actor.train()
        self.step += 1

        obs = rollout_data.observations.to(self.actor.device())
        actions = rollout_data.actions.to(self.actor.device())
        action_masks = rollout_data.action_masks.to(self.actor.device())
        advantages = rollout_data.advantages.to(self.actor.device())
        old_action_logits = rollout_data.old_action_logits.to(self.actor.device())

        outputs = self.actor.forward(obs)
        action_logits = torch.gather(
            outputs.logits,
            dim=-1,
            index=actions.unsqueeze(-1)
        ).squeeze(-1)

        # Normalize advantage
        advantages = torch.masked_select(advantages.view(-1), action_masks.view(-1))
        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_logits - old_action_logits)
        ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))
        # clipped surrogate loss
        actor_loss_1 = advantages * ratio
        actor_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        loss = - torch.min(actor_loss_1, actor_loss_2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', ['loss'])
        return Outputs(loss=loss.item())


class ParallelCriticTrainerForCausalLM(ParallelTrainer):
    def __init__(self, critic: ParallelVerifier, optimizer: torch.optim.Optimizer):
        super().__init__(critic, optimizer)
        self.critic = critic
        self.step = 0
        self.criterion = MSELoss()

    def forward(self, rollout_data: RolloutBufferSample):
        self.critic.train()
        self.step += 1

        obs = rollout_data.observations.to(self.critic.device())
        action_masks = rollout_data.action_masks.to(self.critic.device())
        returns = rollout_data.returns.to(self.critic.device())

        values = self.critic.forward(obs).scores
        values = values / (masked_std(values, action_masks, keepdim=True) + 1e-12)

        loss = self.criterion.forward(values, returns, action_masks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', ['loss'])
        return Outputs(loss=loss.item())
