import torch


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


@torch.compile(dynamic=False, fullgraph=True)
def muon_step(
    params:       torch.Tensor,   # (B, m, n) — stacked weight matrices
    grads:        torch.Tensor,   # (B, m, n)
    momentum_buf: torch.Tensor,   # (B, m, n) — Nesterov buffer
    second_buf:   torch.Tensor,   # (B, 1, 1) — per-tensor RMS EMA   ← shape fixed
    momentum:     torch.Tensor,   # scalar
    lr:           torch.Tensor,   # scalar
    wd:           torch.Tensor,   # scalar
    beta:         torch.Tensor,   # scalar — EMA decay for second_buf
    ns_steps:     int,            # Python int — loop unrolled at trace time
) -> None:
    # ── 1. Nesterov momentum ──────────────────────────────────────────────
    # Standard (non-EMA) form matching Keller Jordan's reference implementation:
    #   buf_t = μ · buf_{t-1} + g_t          (accumulate)
    #   ĝ     = g_t + μ · buf_t              (lookahead)
    momentum_buf.mul_(momentum).add_(grads)
    g = grads + momentum * momentum_buf     # non-mutating: grads tensor untouched

    # ── 2. Newton-Schulz orthogonalization ───────────────────────────────
    # Cast to fp32 for the iteration; cast back afterward.
    # After this, g has approximately unit operator norm per matrix.
    g = _newton_schulz_batch(g.float(), ns_steps).to(grads.dtype)

    # ── 3. Per-tensor RMS adaptive scaling ───────────────────────────────
    # Tracks the running RMS of the orthogonalized update so the effective
    # step size is consistent across layers of different rank/sparsity.
    rms = g.square().mean(dim=(-2, -1), keepdim=True)      # (B, 1, 1)
    second_buf.lerp_(rms.to(second_buf.dtype), 1 - beta)   # EMA update
    g = g * second_buf.clamp_min(1e-10).rsqrt().to(g.dtype)

    # ── 4. Decoupled weight decay + gradient step ─────────────────────────
    params.sub_(lr * (g + wd * params))