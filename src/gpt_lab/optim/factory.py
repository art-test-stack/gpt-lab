# factory.py
from __future__ import annotations

import torch
import torch.distributed as dist
from typing import Any, Optional

from gpt_lab.optim.adamw import adamw_step_fused as adamw_step
from gpt_lab.optim.muon  import muon_step_fused as muon_step
from gpt_lab.utils.distributed import get_dist_info


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
        # self._init_adamw_scalars()

    # ------------------------------------------------------------------
    # Persistent CPU scalar tensors for AdamW.
    # Re-used every step via .fill_() — avoids re-allocation and keeps
    # the torch.compile graph stable (same tensor identity every call).
    # ------------------------------------------------------------------
    # def _init_adamw_scalars(self) -> None:
    #     self._adamw_step_t  = torch.tensor(0.0, device="cpu")
    #     self._adamw_lr_t    = torch.tensor(0.0, device="cpu")
    #     self._adamw_beta1_t = torch.tensor(0.0, device="cpu")
    #     self._adamw_beta2_t = torch.tensor(0.0, device="cpu")
    #     self._adamw_eps_t   = torch.tensor(0.0, device="cpu")
    #     self._adamw_wd_t    = torch.tensor(0.0, device="cpu")
    
    # def _init_muon_scalars(self) -> None:
    #     self._muon_momentum_t = torch.tensor(0.0, device="cpu")
    #     self._muon_lr_t       = torch.tensor(0.0, device="cpu")
    #     self._muon_wd_t       = torch.tensor(0.0, device="cpu")
    #     self._muon_beta2_t    = torch.tensor(0.0, device="cpu")

    # # ──────────────────────────────────────────────────────────────────
    # # AdamW local step
    # # ──────────────────────────────────────────────────────────────────
    # def _step_adamw_local(self, group: dict) -> None:
    #     for p in group["params"]:
    #         if p.grad is None:
    #             continue

    #         state = self.state[p]
    #         if not state:
    #             state["step"]       = 0
    #             state["exp_avg"]    = torch.zeros_like(p)
    #             state["exp_avg_sq"] = torch.zeros_like(p)

    #         state["step"] += 1

    #         self._adamw_step_t .fill_(state["step"])
    #         self._adamw_lr_t   .fill_(group["lr"])
    #         self._adamw_beta1_t.fill_(group["betas"][0])
    #         self._adamw_beta2_t.fill_(group["betas"][1])
    #         self._adamw_eps_t  .fill_(group["eps"])
    #         self._adamw_wd_t   .fill_(group["weight_decay"])

    #         adamw_step(
    #             p, p.grad,
    #             state["exp_avg"], state["exp_avg_sq"],
    #             self._adamw_step_t, self._adamw_lr_t,
    #             self._adamw_beta1_t, self._adamw_beta2_t,
    #             self._adamw_eps_t, self._adamw_wd_t,
    #         )

    # # ──────────────────────────────────────────────────────────────────
    # # Muon local step
    # #
    # # build_optimizer() creates one group per unique weight shape
    # # (shape bucketing). Each group gets its own _muon_state on first
    # # call, so N buckets → N independent momentum/second-moment buffers.
    # # ──────────────────────────────────────────────────────────────────
    # def _step_muon_local(self, group: dict) -> None:
    #     params = [p for p in group["params"] if p.grad is not None]
    #     if not params:
    #         return

    #     p = params[0]  # any param will do, since all share the same
    #     state = self.state[p]
    #     n_params = len(params)
    #     shape, device, dtype = p.shape, p.device, p.dtype

    #     # State is stored on the group dict, not per-parameter, because Muon
    #     # maintains a single batched buffer across all params in the bucket.
    #     state = group.setdefault("_muon_state", {})
    #     if "momentum_buf" not in state:
    #         state["momentum_buf"] = torch.zeros_like(n_params, *shape, dtype=dtype, device=device)  # (B, m, n)
    #     momentum_buffer = state["momentum_buf"]
    #     if "second_buf" not in state:   
    #         state_shape = (n_params, shape[-2], 1) if shape[-2] >= shape[-1] else (n_params, 1, shape[-1])
    #         state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
    #     second_momentum_buffer = state["second_momentum_buffer"]
        
    #     red_dim = -1 if shape[-2] >= shape[-1] else -2

    #     # Stack grads and params (NOTE: this assumes all params have the same shape)
    #     stacked_grads = torch.stack([p.grad for p in params])
    #     stacked_params = torch.stack(params)

    #     # Fill all the 0-D tensors with current values
    #     self._muon_momentum_t.fill_(group["momentum"])
    #     self._muon_beta2_t.fill_(group["beta"] if group["beta"] is not None else 0.0)
    #     self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
    #     self._muon_wd_t.fill_(group["weight_decay"])

    #     muon_step(
    #         stacked_grads,
    #         stacked_params,
    #         momentum_buffer,
    #         second_momentum_buffer,
    #         self._muon_momentum_t,
    #         self._muon_lr_t,
    #         self._muon_wd_t,
    #         self._muon_beta2_t,
    #         group["ns_steps"],   # Python int — unrolled at compile time
    #         red_dim,
    #         dtype,
    #     )

    #     torch._foreach_copy_(params, list(stacked_params.unbind(0)))

# ══════════════════════════════════════════════════════════════════════════════
# BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

class _BaseBackend:
    def __init__(self, opt: _BaseFactory):
        self.opt = opt

    def _step_group(self, group: dict) -> None:
        opt_type = group["opt"]
        if opt_type == "adamw":
            return self.opt._step_adamw_local(group)
        elif opt_type == "muon":
            return self.opt._step_muon_local(group)
        else:
            raise ValueError(f"Unknown optimizer type {opt_type!r} in local step.")

class _LocalBackend(_BaseBackend):
    def step(self) -> None:
        for group in self.opt.param_groups:
            self._step_group(group)

class _DistributedBackend(_BaseBackend):
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
            self._step_group(group)

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

    

# nanochat optimizer
from torch import Tensor
# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # Fill 0-D tensors with current values
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            # Fused update: weight_decay -> momentum -> bias_correction -> param_update
            adamw_step(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        params: list[Tensor] = group['params']
        if not params:
            return

        # Get or create group-level buffers (stored in first param's state for convenience)
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum for every individual parameter
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer is factored, either per-row or per-column
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (NOTE: this assumes all params have the same shape)
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Fill all the 0-D tensors with current values
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta"] if group["beta"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
        muon_step(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
            dtype
        )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['opt'] == 'adamw':
                self._step_adamw(group)
            elif group['opt'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    See MuonAdamW for the algorithmic details of each optimizer. This class adds
    distributed communication to enable multi-GPU training without PyTorch DDP.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (caller's responsibility).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer, second_momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim, dtype
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['opt'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['opt'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['opt'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['opt'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

class OptimizerFactory(_BaseFactory):
    def __init__(self, param_groups: list[dict], dist_info: Optional[dict] = None):
        super().__init__(param_groups)
        if dist_info is None:
            dist_info = get_dist_info()
        self.dist_info = dist_info
        # self.backend = _DistributedBackend(self) if dist_info["IS_DDP_INITIALIZED"] else _LocalBackend(self)
        self.backend = DistMuonAdamW(param_groups) if dist_info["IS_DDP_INITIALIZED"] else MuonAdamW(param_groups)

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