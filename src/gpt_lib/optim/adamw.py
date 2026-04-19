import torch


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step(
    p:           torch.Tensor,   # parameter
    grad:        torch.Tensor,   # gradient
    exp_avg:     torch.Tensor,   # m_t  — first  moment buffer
    exp_avg_sq:  torch.Tensor,   # v_t  — second moment buffer
    step_t:      torch.Tensor,   # scalar: current step (1-indexed)
    lr_t:        torch.Tensor,   # scalar: learning rate
    beta1_t:     torch.Tensor,   # scalar: β₁
    beta2_t:     torch.Tensor,   # scalar: β₂
    eps_t:       torch.Tensor,   # scalar: ε
    wd_t:        torch.Tensor,   # scalar: weight decay λ
) -> None:
    # 1. Decoupled weight decay: p ← p · (1 - lr·λ)
    p.mul_(1 - lr_t * wd_t)

    # 2. Moment EMA updates
    exp_avg.lerp_(grad, 1 - beta1_t)           # m_t = β₁·m_{t-1} + (1-β₁)·g
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)  # v_t = β₂·v_{t-1} + (1-β₂)·g²

    # 3. Bias correction
    bias1 = 1 - beta1_t ** step_t              # 1 - β₁ᵗ
    bias2 = 1 - beta2_t ** step_t              # 1 - β₂ᵗ
    m_hat = exp_avg / bias1                    # m̂_t
    v_hat = exp_avg_sq / bias2                 # v̂_t

    # 4. Adam update — all tensor ops, no .item() needed for fullgraph
    #    p ← p - lr · m̂ / (√v̂ + ε)
    p.sub_(lr_t * m_hat / (v_hat.sqrt() + eps_t))