from gpt_lab.utils.distributed import ComputeContext
import math
from types import SimpleNamespace

import pytest
import torch

from gpt_lab.evaluate.bpb import compute_bpb


class _FixedLossModel:
    def get_device(self):
        return torch.device("cpu")

    def __call__(self, x, y, reduction):
        assert reduction == "none"
        return SimpleNamespace(loss=x.float(), logits=None)


def test_compute_bpb_reports_mean_loss_per_token():
    batches = [
        (torch.tensor([[1, 2]]), torch.tensor([[0, 1]]), None),
        (torch.tensor([[3, 0]]), torch.tensor([[1, -1]]), None),
    ]
    token_bytes = torch.tensor([0, 1])

    result = compute_bpb(
        _FixedLossModel(),
        batches,
        len(batches),
        token_bytes,
        comp_info=ComputeContext(device=torch.device("cpu")),
    )

    assert result["loss"] == 2.0
    assert result["bpb"] == pytest.approx((2 + 3) / (2 * math.log(2.0)))
