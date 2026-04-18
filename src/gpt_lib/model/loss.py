from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn.functional as F

from gpt_lib.utils.schemas import ModelOutput, LossConfig 

# TODO: from cut_cross_entropy import linear_cross_entropy

class BaseLoss(ABC):
    @abstractmethod
    def __call__(
        self,
        logits: Union[torch.Tensor, ModelOutput],
        labels: torch.Tensor,
        ref_output: Optional[ModelOutput] = None,
    ) -> torch.Tensor:
        ...

class CrossEntropyLoss(BaseLoss):
    def __init__(self, ignore_index=-1, reduction="mean"):
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, logits: Union[torch.Tensor, ModelOutput], labels: torch.Tensor, ref_output=None, reduction=None):
        if isinstance(logits, ModelOutput):
            logits = logits.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction if reduction is None else reduction,
        )
        return loss

def cross_entropy_loss(logits: Union[torch.Tensor, ModelOutput], labels: torch.Tensor, ignore_index=-1, reduction="mean") -> torch.Tensor:
    if isinstance(logits, ModelOutput):
        logits = logits.logits
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-1,
        reduction=reduction,
    )
    return loss

# class CutCrossEntropyLoss(BaseLoss):
#     def __init__(self, ignore_index=-1, reduction="mean"):
#         self.ignore_index = ignore_index
#         self.reduction = reduction

#     def __call__(self, logits: Union[torch.Tensor, ModelOutput], labels: torch.Tensor, ref_output=None, reduction=None):
#         raise NotImplementedError("CutCrossEntropyObjective is not yet supported.")
#         if isinstance(logits, ModelOutput):
#             logits = logits.logits

#         loss = linear_cross_entropy(
#             logits,
#             labels,
#             ignore_index=self.ignore_index,
#             reduction=self.reduction if reduction is None else reduction,
#             impl="torch_compile"
#         )
#         return loss


class KLDivLoss(BaseLoss):
    def __init__(self, epsilon: float = 0.1, ignore_index: int = -1):
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def __call__(self, logits: Union[torch.Tensor, ModelOutput], labels: torch.Tensor, ref_output=None, reduction=None):
        if isinstance(logits, ModelOutput):
            log_probs = logits.log_probs
        else:
            log_probs = logits

        vocab_size = log_probs.size(-1)

        mask = labels != self.ignore_index

        with torch.no_grad():
            target_dist = torch.zeros_like(log_probs)
            target_dist.fill_(self.epsilon / (vocab_size - 1))
            target_dist.scatter_(
                -1, labels.unsqueeze(-1), 1.0 - self.epsilon
            )

        kl = F.kl_div(
            log_probs[mask],
            target_dist[mask],
            reduction="batchmean",
            log_target=False,
        )
        return kl


def build_loss(config: Optional[LossConfig] = None) -> "BaseLoss":
    if config is None:
        config = LossConfig()

    loss_type = config.loss_fn
    loss_kwargs = config.kwargs
    if loss_type == "cross_entropy":
        return cross_entropy_loss
    elif loss_type == "kl_divergence":
        return KLDivLoss(
            epsilon=loss_kwargs.get("epsilon", 0.1),
            ignore_index=loss_kwargs.get("ignore_index", -1),
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    