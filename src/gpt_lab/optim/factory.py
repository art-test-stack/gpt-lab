# factory.py
from __future__ import annotations

import torch
import torch.distributed as dist
from typing import Any

from gpt_lab.optim.adamw import adamw_step
from gpt_lab.optim.muon  import muon_step


# ── Required keys per optimizer type ──────────────────────────────────────────
# Validated at construction so misconfigured YAML groups fail immediately
# instead of blowing up mid-training on step 1.
_ADAMW_REQUIRED: frozenset[str] = frozenset({"opt", "lr", "betas", "eps", "weight_decay"})
_MUON_REQUIRED:  frozenset[str] = frozenset({"opt", "lr", "momentum", "beta", "ns_steps", "weight_decay"})


def _validate_group(group: dict[str, Any], idx: int) -> None:
    """
    Raise a descriptive ValueError for any malformed param group.

    Called once at construction — much better than cryptic KeyErrors or
    silent wrong-key-fallthrough during an actual training step.
    The YAML `default` block can silently inject keys (e.g. `betas`, `eps`)
    into Muon groups; those are harmless but this catches genuinely missing ones.
    """
    opt = group.get("opt")
    if opt == "adamw":
        missing = _ADAMW_REQUIRED - group.keys()
    elif opt == "muon":
        missing = _MUON_REQUIRED - group.keys()
    else:
        raise ValueError(
            f"Param group [{idx}]: unknown optimizer type {opt!r}. "
            f"Expected 'adamw' or 'muon'."
        )
    if missing:
        raise ValueError(
            f"Param group [{idx}] (opt={opt!r}) is missing required keys: {missing}. "
            f"Check that your YAML config or code defaults supply all required fields."
        )


# ══════════════════════════════════════════════════════════════════════════════
# BASE
# ══════════════════════════════════════════════════════════════════════════════

class _BaseFactory(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict]):
        for idx, group in enumerate(param_groups):
            _validate_group(group, idx)

            # Factory owns initial_lr — set it HERE, before super().__init__
            # copies the group dicts, so every downstream path (update_hyperparams,
            # scheduler, etc.) always has a stable reference point.
            #
            # NOTE: build_optimizer() sets initial_lr again after construction:
            #   for group in optimizer.param_groups:
            #       group["initial_lr"] = group["lr"]   ← redundant, safe to remove
            # setdefault means that loop is a no-op, not a bug.
            group.setdefault("initial_lr", group["lr"])

        super().__init__(param_groups, defaults={})
        self._init_adamw_scalars()

    # ------------------------------------------------------------------
    # Persistent CPU scalar tensors for AdamW.
    # Re-used every step via .fill_() — avoids re-allocation and keeps
    # the torch.compile graph stable (same tensor identity every call).
    # ------------------------------------------------------------------
    def _init_adamw_scalars(self) -> None:
        self._adamw_step_t  = torch.tensor(0.0, device="cpu")
        self._adamw_lr_t    = torch.tensor(0.0, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, device="cpu")
        self._adamw_eps_t   = torch.tensor(0.0, device="cpu")
        self._adamw_wd_t    = torch.tensor(0.0, device="cpu")

    # ──────────────────────────────────────────────────────────────────
    # AdamW local step
    # ──────────────────────────────────────────────────────────────────
    def _step_adamw_local(self, group: dict) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if not state:
                state["step"]       = 0
                state["exp_avg"]    = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1

            self._adamw_step_t .fill_(state["step"])
            self._adamw_lr_t   .fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t  .fill_(group["eps"])
            self._adamw_wd_t   .fill_(group["weight_decay"])

            adamw_step(
                p, p.grad,
                state["exp_avg"], state["exp_avg_sq"],
                self._adamw_step_t, self._adamw_lr_t,
                self._adamw_beta1_t, self._adamw_beta2_t,
                self._adamw_eps_t, self._adamw_wd_t,
            )

    # ──────────────────────────────────────────────────────────────────
    # Muon local step
    #
    # build_optimizer() creates one group per unique weight shape
    # (shape bucketing). Each group gets its own _muon_state on first
    # call, so N buckets → N independent momentum/second-moment buffers.
    # ──────────────────────────────────────────────────────────────────
    def _step_muon_local(self, group: dict) -> None:
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return

        # All params in the group share the same shape (guaranteed by bucketing).
        grads          = torch.stack([p.grad for p in params])   # (B, m, n)
        params_stacked = torch.stack([p.data for p in params])   # (B, m, n)

        # State is stored on the group dict, not per-parameter, because Muon
        # maintains a single batched buffer across all params in the bucket.
        state = group.setdefault("_muon_state", {})
        if "momentum_buf" not in state:
            B, *_ = params_stacked.shape
            dev, dt = params_stacked.device, params_stacked.dtype
            state["momentum_buf"] = torch.zeros_like(params_stacked)  # (B, m, n)
            state["second_buf"]   = torch.zeros(B, 1, 1, device=dev, dtype=dt)  # (B, 1, 1)

        dev, dt = params_stacked.device, params_stacked.dtype
        muon_step(
            params_stacked,
            grads,
            state["momentum_buf"],
            state["second_buf"],
            torch.tensor(group["momentum"],     device=dev, dtype=dt),
            torch.tensor(group["lr"],           device=dev, dtype=dt),
            torch.tensor(group["weight_decay"], device=dev, dtype=dt),
            torch.tensor(group["beta"],         device=dev, dtype=dt),
            group["ns_steps"],   # Python int — unrolled at compile time
        )

        for p, updated in zip(params, params_stacked.unbind(0)):
            p.data.copy_(updated)


# ══════════════════════════════════════════════════════════════════════════════
# BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

class _LocalBackend:
    def __init__(self, opt: _BaseFactory):
        self.opt = opt

    def step(self) -> None:
        for group in self.opt.param_groups:
            if group["opt"] == "adamw":
                self.opt._step_adamw_local(group)
            else:
                self.opt._step_muon_local(group)

class _DistributedBackend:
    def __init__(self, opt: _BaseFactory):
        self.opt = opt

    def step(self) -> None:
        # ── 1. Gradient sync ──────────────────────────────────────────
        all_grads = [
            p.grad
            for group in self.opt.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        if all_grads:
            self._allreduce_grads(all_grads)

        # ── 2. Local step (same on every rank by construction) ────────
        for group in self.opt.param_groups:
            opt_type = group["opt"]
            if opt_type == "adamw":
                self.opt._step_adamw_local(group)
            elif opt_type == "muon":
                self.opt._step_muon_local(group)
            # TODO
            # elif opt_type == "shampoo":
                # self.opt._step_shampoo_local(group) 
            else:
                raise ValueError(f"Unknown optimizer type {opt_type!r} in distributed step.")

    @staticmethod
    def _allreduce_grads(grads: list[torch.Tensor]) -> None:
        """
        Flatten-reduce-unflatten over (device, dtype) buckets.

        Replaces the deprecated all_reduce_coalesced while preserving
        its key property: one all_reduce kernel launch per bucket instead
        of one per tensor, keeping communication overhead minimal.

        Steps:
          1. Group gradients by (device, dtype) — mixed-precision models
             may have bf16 and fp32 grads simultaneously.
          2. Cat each bucket into one flat contiguous buffer.
          3. Fire all_reduce(async_op=True) on every buffer concurrently.
          4. Wait on each handle, then scatter the averaged values back
             into the original gradient tensors via offset arithmetic.
        """
        # ── bucket by (device, dtype) ─────────────────────────────────
        buckets: dict[tuple, list[torch.Tensor]] = {}
        for g in grads:
            key = (g.device, g.dtype)
            buckets.setdefault(key, []).append(g)

        # ── launch all reductions concurrently ────────────────────────
        pending: list[tuple[dist.Work, torch.Tensor, list[torch.Tensor]]] = []
        for (device, dtype), bucket in buckets.items():
            flat = torch.cat([g.view(-1) for g in bucket])  # one contiguous buffer
            work = dist.all_reduce(flat, op=dist.ReduceOp.AVG, async_op=True)
            pending.append((work, flat, bucket))

        # ── wait and scatter back ─────────────────────────────────────
        for work, flat, bucket in pending:
            work.wait()
            offset = 0
            for g in bucket:
                numel = g.numel()
                g.copy_(flat[offset : offset + numel].view_as(g))
                offset += numel

# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

class OptimizerFactory(_BaseFactory):
    def __init__(self, param_groups: list[dict], distributed: bool = False):
        super().__init__(param_groups)
        self.backend = _DistributedBackend(self) if distributed else _LocalBackend(self)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.backend.step()
        return loss

    @torch.no_grad()
    def update_hyperparams(
        self,
        lrm: float                  = 1.0,
        muon_momentum: float | None = None,
        weight_decay:  float | None = None,
    ) -> tuple[float, float | None, float | None]:
        """
        Called by the LR scheduler each step.

        lrm            — applied to ALL groups via initial_lr (set at construction).
        weight_decay   — if given, overrides weight_decay on ALL groups.
                         build_optimizer() already applied weight_decay_scale to
                         the transformer groups before construction, so passing
                         an absolute value here (e.g. from a WD schedule) is safe.
        muon_momentum  — if given, overrides momentum only on Muon groups.
        """
        for group in self.param_groups:
            group["lr"] = group["initial_lr"] * lrm 

            if weight_decay is not None:
                group["weight_decay"] = weight_decay

            if group["opt"] == "muon" and muon_momentum is not None:
                group["momentum"] = muon_momentum

        return lrm, muon_momentum, weight_decay