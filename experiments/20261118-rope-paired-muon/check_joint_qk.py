"""One CPU step on a causal key-selection task using PR #53's paired polar kernel."""

import math

import torch
import torch.nn.functional as F

from gpt_lab.optim.kernels.muon import paired_polar_express


def pointer_loss(weights, tokens, phase):
    # One head, four positions, four real coordinates (two genuine rotary planes).
    qk = tokens @ weights.mT
    pairs = torch.view_as_complex(qk.reshape(2, 4, 2, 2).contiguous())
    rotated = torch.view_as_real(pairs * phase).flatten(-2)
    q, k = F.rms_norm(rotated, (4,), eps=1e-5)
    logits = (q @ k.T) / math.sqrt(4)
    future = torch.ones_like(logits, dtype=torch.bool).triu(1)
    # At query positions 1, 2 and 3 the target key is the preceding position.
    return F.cross_entropy(logits.masked_fill(future, -torch.inf)[1:], torch.arange(3))


def polar(proposal):
    return paired_polar_express(proposal, ns_steps=5, compute_dtype=torch.float32)


def joint_direction(proposal, k_rows):
    q, k = proposal
    # Each adjacent row pair is one RoPE plane. Only K's plane order varies.
    packed = torch.cat((q, k[k_rows]), dim=-1)
    uq, uk = polar(packed).chunk(2, dim=-1)
    return torch.stack((uq, uk[torch.argsort(k_rows)]))


def main():
    torch.set_num_threads(1)
    tokens = torch.randn(4, 4, generator=torch.Generator().manual_seed(42))
    angles = torch.outer(torch.arange(4.0), torch.tensor([1.0, 0.01]))
    phase = torch.polar(torch.ones_like(angles), angles)
    print("weight_seed  initial_CE    separate     aligned     swapped   cos(sep,aligned)")
    for seed in range(4):
        generator = torch.Generator().manual_seed(seed)
        weights = (0.5 * torch.randn(2, 4, 4, generator=generator)).requires_grad_()
        initial = pointer_loss(weights, tokens, phase)
        (gradient,) = torch.autograd.grad(initial, weights)
        directions = [
            polar(gradient),
            joint_direction(gradient, torch.arange(4)),
            joint_direction(gradient, torch.tensor([2, 3, 0, 1])),
        ]
        with torch.no_grad():
            losses = []
            for direction in directions:
                step = 0.05 * direction / direction.norm()
                torch.testing.assert_close(step.norm(), torch.tensor(0.05))
                losses.append(pointer_loss(weights - step, tokens, phase).item())
            cosine = F.cosine_similarity(directions[0].flatten(), directions[1].flatten(), dim=0)
        loss_columns = " ".join(f"{loss:11.6f}" for loss in losses)
        print(f"{seed:11} {initial.item():11.6f} {loss_columns}   {cosine.item():.6f}")


if __name__ == "__main__":
    main()
