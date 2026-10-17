"""Adapted from https://github.com/zichongli5/NorMuon/blob/main/normuon.py.

Copyright (c) 2025 zichongli5. MIT; see LICENSES/NorMuon-MIT.txt.
Newton-Schulz implementation credited to Keller Jordan; see
LICENSES/modded-nanogpt-MIT.txt. Batched and compiled for GPT-Lab.
"""
import torch
from torch import Tensor


@torch.compile(dynamic=False, fullgraph=True)
def normuon_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    _red_dim: int,
    _compute_dtype: torch.dtype,
) -> None:
    """Reference-style update with BF16 NS and row-wise normalization.

    Shares MuonStrategy's kernel signature; its axis/dtype arguments do not
    override the reference algorithm. Compiler fusion may change intermediate
    BF16 rounding relative to the eager reference.
    """
    momentum = momentum_t.to(stacked_grads.dtype)
    beta2 = beta2_t.to(second_momentum_buffer.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    update = stacked_grads.lerp(momentum_buffer, momentum)
    x = update.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        gram = x @ x.mT
        polynomial = b * gram + c * gram @ gram
        x = a * x + polynomial @ x
    if transposed:
        x = x.mT
    update = x.to(stacked_grads.dtype)

    original_norm = update.norm(dim=(-2, -1), keepdim=True)
    second_momentum_buffer.lerp_(update.square().mean(dim=-1, keepdim=True), 1 - beta2)
    update.mul_(1 / (second_momentum_buffer.sqrt() + 1e-10))
    update.mul_(original_norm / (update.norm(dim=(-2, -1), keepdim=True) + 1e-10))
    update.mul_(max(1.0, stacked_grads.size(-2) / stacked_grads.size(-1)) ** 0.5)

    stacked_params.mul_(1 - lr_t * wd_t)
    stacked_params.sub_(lr_t * update)
