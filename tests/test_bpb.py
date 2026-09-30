import math
from types import SimpleNamespace

import pytest
import torch

from gpt_lab.evaluate.bpb import compute_bpb


class _ConstantLossModel:
    def __init__(self, loss_per_token: float, vocab_size: int):
        self.loss_per_token = loss_per_token
        self.vocab_size = vocab_size

    def get_device(self):
        return torch.device("cpu")

    def __call__(self, x, y, reduction):
        assert reduction == "none"
        loss = torch.full(y.shape, self.loss_per_token, dtype=torch.float32)
        logits = torch.empty((*y.shape, self.vocab_size), dtype=torch.float32)
        return SimpleNamespace(loss=loss, logits=logits)


def test_compute_bpb_reports_mean_loss_per_token():
    batch_size = 4
    sequence_length = 3
    steps = 2
    loss_per_token = 2.5
    vocab_size = 4

    x = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    y = torch.ones((batch_size, sequence_length), dtype=torch.long)
    batches = [(x, y, None) for _ in range(steps)]
    token_bytes = torch.ones(vocab_size, dtype=torch.int64)
    model = _ConstantLossModel(loss_per_token, vocab_size)

    result = compute_bpb(
        model,
        batches,
        steps,
        token_bytes,
        dist_info={"IS_DDP_INITIALIZED": False, "WORLD_SIZE": 1},
    )

    assert result["loss"] == pytest.approx(loss_per_token)
    assert result["bpb"] == pytest.approx(loss_per_token / math.log(2.0))
