"""Adapted from https://github.com/zichongli5/NorMuon/blob/main/normuon.py.

Copyright (c) 2025 zichongli5. MIT; see LICENSES/NorMuon-MIT.txt.
Newton-Schulz implementation credited to Keller Jordan; see
LICENSES/modded-nanogpt-MIT.txt. Batched and compiled for GPT-Lab.

ANVIL2 maps adapted from Deven Pietrzak's archived modded-nanogpt record:
records/track_1_short/2026-08-30_ANVIL2/this_pr/01b2f094-4536-479c-b0dd-cb14eba4844c.txt
Embedded train_gpt.py SHA256: c6da246b1d87a525c0130f212d40e9a32e7eba01738e97686011d0854dd6945a.
MIT; see LICENSES/modded-nanogpt-MIT.txt.
"""
import torch
from torch import Tensor


ANVIL2_MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)


def _orthogonalize(update: Tensor, ns_steps: int, anvil_spectral: bool) -> Tensor:
    x = update.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    if anvil_spectral:
        gram = x @ x.mT
        norm = gram.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None].sqrt()
        denominator = norm * 1.05 + 1e-6
        x = (x.float() / denominator).bfloat16()
        gram = (gram.float() / denominator.square()).bfloat16()
        coefficients = ANVIL2_MAPS
    else:
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        coefficients = ((3.4445, -4.7750, 2.0315),) * ns_steps
    for index, (a, b, c) in enumerate(coefficients):
        if not anvil_spectral or index > 0:
            gram = x @ x.mT
        polynomial = b * gram + c * gram @ gram
        x = a * x + polynomial @ x
    return x.mT if transposed else x


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
    anvil_spectral: bool = False,
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
    x = _orthogonalize(update, ns_steps, anvil_spectral)
    update = x.to(stacked_grads.dtype)

    original_norm = update.norm(dim=(-2, -1), keepdim=True)
    second_momentum_buffer.lerp_(update.square().mean(dim=-1, keepdim=True), 1 - beta2)
    update.mul_(1 / (second_momentum_buffer.sqrt() + 1e-10))
    update.mul_(original_norm / (update.norm(dim=(-2, -1), keepdim=True) + 1e-10))
    update.mul_(max(1.0, stacked_grads.size(-2) / stacked_grads.size(-1)) ** 0.5)

    stacked_params.mul_(1 - lr_t * wd_t)
    stacked_params.sub_(lr_t * update)
