import torch
import torch.nn as nn

from src.utils import powmax, masked_mean


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class KLDivLoss(Loss):
    def __init__(self, eps=7e-5):
        super().__init__()
        self.eps = eps

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            masks: torch.Tensor = None,
            T: float = 1.0
    ):
        """
        Compute KL-Divergence loss.
        :param T: Temperature, default to be 1.
        :param logits: the logits of the estimated distribution, before `softmax`
        :param targets: the target logits, before `softmax`.
        :param masks: Optional. For masked selection.
        Shape is identical to the shape of `logits` up to last dim.
        :return: scalar loss.
        """
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, targets.size(-1)).to(logits)
        estimates = torch.softmax(logits.float(), dim=-1).type_as(logits)
        targets = torch.softmax(targets.float() / T, dim=-1).type_as(targets)
        estimates = powmax(estimates + self.eps)
        targets = powmax(targets + self.eps)

        loss = targets * (torch.log(targets) - torch.log(estimates))
        loss = torch.sum(loss, dim=-1)
        if masks is not None:
            masks = masks.view(-1).to(logits.device)
            loss = torch.masked_select(loss, masks)
        return torch.mean(loss)


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            masks: torch.Tensor = None,
    ):
        loss = (logits - targets) ** 2
        loss = masked_mean(loss, masks)
        return loss.mean()


class RewardLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            chosen_rewards: torch.Tensor,
            rejected_rewards: torch.Tensor,
            chosen_masks: torch.Tensor = None,
            rejected_masks: torch.Tensor = None
    ):
        bzs = chosen_rewards.shape[0]
        chosen_rewards = chosen_rewards.view(bzs, -1)
        rejected_rewards = rejected_rewards.view(bzs, -1)
        if chosen_masks is not None:
            chosen_masks = chosen_masks.view(bzs, -1)
        if rejected_masks is None:
            rejected_masks = rejected_masks.view(bzs, -1)

        c_rewards = masked_mean(chosen_rewards, chosen_masks, dim=-1)  # [b]
        r_rewards = masked_mean(rejected_rewards, rejected_masks, dim=-1)  # [b]

        loss = - torch.log(torch.sigmoid(c_rewards - r_rewards)).mean()
        return loss


if __name__ == '__main__':
    criterion = RewardLoss()
    _chosen_rewards = torch.tensor([[2, 2, 3, 3, 3]])
    _rejected_rewards = torch.tensor([[1, 1, 1, 1, 4]])
    _chosen_masks = torch.tensor([[False, False, False, False, False]])
    _rejected_masks = torch.tensor([[True, True, True, True, False]])
    _loss = criterion.forward(_chosen_rewards, _rejected_rewards, _chosen_masks, _rejected_masks)
    print(_loss)
