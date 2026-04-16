import torch

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linear to zero over the course of training)
# def get_weight_decay(it):
#     return weight_decay_scaled * (1 - it / num_iterations)
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params,
                    momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta_t,
                    ns_steps, red_dim):

    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    X = g / (g.norm(dim=(-2, -1), keepdim=True) + 1e-6)
    g = X

    beta = beta_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.dtype), 1 - beta)

    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    g = g * step_size.to(g.dtype)

    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    stacked_params.sub_(lr * g + lr * wd * stacked_params)

