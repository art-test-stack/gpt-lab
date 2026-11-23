import torch
import torch.distributed as dist
from torch import Tensor

def _newton_schulz_batch(X: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Batched quintic Newton-Schulz iteration.

    Iteratively maps X → a matrix with orthonormal rows (Stiefel manifold),
    which gives the update approximately unit operator norm. This is the
    core operation that distinguishes Muon from every other optimizer.

    Coefficients (a, b, c) are tuned for fast convergence with 5 steps,
    from Keller Jordan's original Muon release (2024).

    X : (B, m, n)
    Returns same shape, orthogonalized.
    """
    a, b, c = 3.4445, -4.7750, 2.0315

    # Normalize to unit Frobenius norm for numerical stability
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Always iterate on the smaller square (n × n where n = min(m, n))
    # so the inner A = X @ Xᵀ stays O(n²·max) not O(m²·n)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mT.contiguous()

    # Quintic iteration: X ← a·X + (b·A + c·A²)·X,  A = X·Xᵀ
    for _ in range(steps):          # unrolled at compile time (steps is a Python int)
        A  = X @ X.mT               # (B, n, n)
        B_ = b * A + c * (A @ A)    # (B, n, n)
        X  = a * X + B_ @ X         # (B, n, n) or (B, n, m)

    if transposed:
        X = X.mT.contiguous()

    return X

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt
Copyright (c) 2024 Keller Jordan
Licensed under the MIT License; see LICENSES/modded-nanogpt-MIT.txt.
Modified for GPT-Lab by Arthur Testard.

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _packed_complex_mm(ar: Tensor, ai: Tensor, br: Tensor, bi: Tensor, dtype: torch.dtype):
    """Evaluate (ar + i ai) @ (br + i bi) with one real GEMM."""
    top = torch.cat((ar, -ai), dim=-1)
    bottom = torch.cat((ai, ar), dim=-1)
    left = torch.cat((top, bottom), dim=-2).to(dtype)
    right = torch.cat((br, bi), dim=-2).to(dtype)
    return (left @ right).float().chunk(2, dim=-2)


def paired_polar_express(
    proposal: Tensor, ns_steps: int, compute_dtype: torch.dtype,
    pair_indices: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Apply one Hermitian polar map per head, optionally re-pairing its rows."""
    if pair_indices is not None:
        order, inverse = pair_indices
        proposal = proposal.index_select(-2, order)
    dtype = torch.bfloat16 if compute_dtype == torch.bfloat16 else torch.float32
    x = proposal.to(dtype).float()
    # Normalize jointly over both components. Keep scalar and polynomial arithmetic in FP32.
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    xr, xi = x[..., ::2, :], x[..., 1::2, :]
    for a, b, c in polar_express_coeffs[:ns_steps]:
        hr, hi = _packed_complex_mm(xr, xi, xr.mT, -xi.mT, dtype)
        h2r, h2i = _packed_complex_mm(hr, hi, hr, hi, dtype)
        pr = b * hr + c * h2r
        pi = b * hi + c * h2i
        dr, di = _packed_complex_mm(pr, pi, xr, xi, dtype)
        xr, xi = a * xr + dr, a * xi + di
    output = torch.stack((xr, xi), dim=-2).flatten(-3, -2).to(dtype)
    return output if pair_indices is None else output.index_select(-2, inverse)


def _real_polar_express(g: Tensor, ns_steps: int, compute_dtype: torch.dtype) -> Tensor:
    # Skip the low-precision cast for FP16, whose exponent range is too narrow here.
    X = g.bfloat16() if compute_dtype == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def _head_norm(heads: Tensor) -> Tensor:
    return torch.linalg.vector_norm(heads.float(), dim=(-2, -1))


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
    compute_dtype: torch.dtype,     # torch.float16, torch.bfloat16, or torch.float32
    *,
    paired_heads: int = 0,
    joint_qk_heads: int = 0,
    pair_indices: tuple[Tensor, Tensor] | None = None,
    monitor_indices: tuple[Tensor, Tensor] | None = None,
) -> Tensor | None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    if monitor_indices is not None:
        gradient_norm = _head_norm(stacked_grads[monitor_indices])

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    if monitor_indices is not None:
        proposal_norm = _head_norm(g[monitor_indices])

    if joint_qk_heads:
        q, k, v = g.split(joint_qk_heads, dim=-3)
        joint_qk = torch.cat((q, k), dim=-1)
        polar = paired_polar_express if paired_heads else _real_polar_express
        q, k = polar(joint_qk, ns_steps, compute_dtype).chunk(2, dim=-1)
        X = torch.cat((q, k, _real_polar_express(v, ns_steps, compute_dtype)), dim=-3)
    elif paired_heads:
        qk, v = g[..., :paired_heads, :, :], g[..., paired_heads:, :, :]
        qk = paired_polar_express(qk, ns_steps, compute_dtype, pair_indices)
        X = torch.cat((qk, _real_polar_express(v, ns_steps, compute_dtype)), dim=-3)
    else:
        X = _real_polar_express(g, ns_steps, compute_dtype)
    if monitor_indices is not None:
        polar_norm = _head_norm(X[monitor_indices])
    g = X.to(stacked_params.dtype)

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    update = lr * g
    decay = lr * wd * stacked_params * mask
    if monitor_indices is not None:
        before = stacked_params[monitor_indices]
        update_norm = _head_norm(update[monitor_indices])
        decay_norm = _head_norm(decay[monitor_indices])
    stacked_params.sub_(update + decay)
    if monitor_indices is not None:
        displacement = _head_norm(stacked_params[monitor_indices].float() - before.float())
        weight_norm = _head_norm(before)
        ratio = torch.where(weight_norm > 0, displacement / weight_norm, float("nan"))
        return torch.stack((gradient_norm, proposal_norm, polar_norm, update_norm, decay_norm, displacement, ratio), dim=-1)
    return None
