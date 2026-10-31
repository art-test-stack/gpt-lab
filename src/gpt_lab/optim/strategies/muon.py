"""Muon optimizer strategy (MomentUm Orthogonalized by Newton-Schulz)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from gpt_lab.optim.kernels.muon import muon_step_fused as muon_step
from gpt_lab.optim.strategy import OptimizerStrategy

if TYPE_CHECKING:
    from gpt_lab.optim.factory import OptimizerFactory


class MuonStrategy(OptimizerStrategy):
    """Existing Polar Express/NorMuon-style variant with cautious weight decay."""

    step_kernel = staticmethod(muon_step)

    @staticmethod
    def _normalization_dim(shape) -> int:
        return -1 if shape[-2] >= shape[-1] else -2

    @staticmethod
    def _learning_rate(group: dict, shape) -> float:
        return group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5

    def _kernel_kwargs(self, group: dict) -> dict:
        return {}
    
    def local_step(self, group: dict) -> None:
        params: list[torch.Tensor] = group["params"]
        if not params:
            return
        
        p = params[0]
        state = self.opt.state[p]
        shape, device, dtype = p.shape, p.device, p.dtype
        red_dim = self._normalization_dim(shape)
        
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (len(params), shape[-2], 1) if red_dim == -1
                else (len(params), 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        
        self.opt.scalars.muon_momentum_t.fill_(group["momentum"])
        self.opt.scalars.muon_beta2_t.fill_(group.get("beta") or 0.0)
        self.opt.scalars.muon_lr_t.fill_(self._learning_rate(group, shape))
        self.opt.scalars.muon_wd_t.fill_(group["weight_decay"])
        
        self.step_kernel(
            stacked_grads, stacked_params, state["momentum_buffer"],
            state["second_momentum_buffer"], self.opt.scalars.muon_momentum_t,
            self.opt.scalars.muon_lr_t, self.opt.scalars.muon_wd_t,
            self.opt.scalars.muon_beta2_t, group["ns_steps"], red_dim,
            self.opt.comp_info.compute_dtype,
            **self._kernel_kwargs(group),
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))
    
    def dist_reduce(self, group: dict, world_size: int) -> dict:
        params = group["params"]
        if not params:
            return {"work": None, "grad_chunk": None}

        return self._reduce_bucket(group, list(range(len(params))), world_size)

    def _reduce_bucket(
        self, group: dict, param_indices: list[int], world_size: int,
    ) -> dict:
        """Reduce whole matrices to their original group owners.

        All ranks must supply the same nonempty index list and collective order.
        owner_positions maps received matrices to the existing local state slots.
        """
        params = group["params"]
        if not param_indices or len(set(param_indices)) != len(param_indices):
            raise ValueError("Muon buckets require nonempty, unique parameter indices")
        if any(index < 0 or index >= len(params) for index in param_indices):
            raise ValueError("Muon bucket parameter index is outside the optimizer group")

        owner_chunk_size = (len(params) + world_size - 1) // world_size
        indices_by_owner = [[] for _ in range(world_size)]
        for index in param_indices:
            indices_by_owner[index // owner_chunk_size].append(index)

        average = dist.get_backend() == "gloo"
        reduce_op = dist.ReduceOp.SUM if average else dist.ReduceOp.AVG

        chunk_size = max(len(indices) for indices in indices_by_owner)
        padded = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        stacked_grads = torch.empty(padded, *shape, dtype=dtype, device=device)
        for owner, indices in enumerate(indices_by_owner):
            start = owner * chunk_size
            if indices:
                gradients = torch.stack([params[index].grad for index in indices])
                stacked_grads[start:start + len(indices)].copy_(gradients)
            stacked_grads[start + len(indices):start + chunk_size].zero_()
        
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        work = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=reduce_op, async_op=True
        )
        
        return {
            "work": work,
            "average": average,
            "grad_chunk": grad_chunk,
            "stacked_grads": stacked_grads,
            "chunk_size": chunk_size,
            "param_indices": list(param_indices),
            "owner_positions": [
                index % owner_chunk_size
                for index in indices_by_owner[dist.get_rank()]
            ],
        }
    
    def dist_compute(
        self, group: dict, info: dict, gather_list: list, rank: int, world_size: int,
    ) -> None:
        """Consume one full-group reduction or {"buckets": complete_partition}.

        Assemble gradients into the original owner-local slots before one group
        update and gather. Bucket boundaries do not change checkpoint state layout.
        """
        params = group["params"]
        if not params:
            return

        buckets = info.get("buckets", [info])
        indices = [index for bucket in buckets for index in bucket["param_indices"]]
        if sorted(indices) != list(range(len(params))):
            raise ValueError("Muon buckets must cover every group parameter exactly once")

        chunk_size = (len(params) + world_size - 1) // world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        red_dim = self._normalization_dim(shape)
        
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        if len(buckets) == 1 and buckets[0]["owner_positions"] == list(range(num_owned)):
            owned_grads = buckets[0]["grad_chunk"]
        else:
            owned_grads = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        for bucket in buckets:
            bucket["work"].wait()
            if bucket["average"]:
                bucket["grad_chunk"].div_(world_size)
            if owned_grads is not bucket["grad_chunk"]:
                for slot, position in enumerate(bucket["owner_positions"]):
                    owned_grads[position].copy_(bucket["grad_chunk"][slot])
        
        state = self.opt.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (chunk_size, shape[-2], 1) if red_dim == -1
                else (chunk_size, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        
        updated_params = info.get("updated_params")
        if updated_params is None:
            updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        
        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)
            
            self.opt.scalars.muon_momentum_t.fill_(group["momentum"])
            self.opt.scalars.muon_beta2_t.fill_(group["beta"])
            self.opt.scalars.muon_lr_t.fill_(
                self._learning_rate(group, shape)
            )
            self.opt.scalars.muon_wd_t.fill_(group["weight_decay"])
            
            self.step_kernel(
                owned_grads[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned],
                state["second_momentum_buffer"][:num_owned],
                self.opt.scalars.muon_momentum_t, self.opt.scalars.muon_lr_t,
                self.opt.scalars.muon_wd_t, self.opt.scalars.muon_beta2_t,
                group["ns_steps"], red_dim, self.opt.comp_info.compute_dtype,
                **self._kernel_kwargs(group),
            )
            updated_params[:num_owned].copy_(stacked_owned)
        
        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()
        
        gathered_buffer = (
            buckets[0]["stacked_grads"] if len(buckets) == 1
            else torch.empty(chunk_size * world_size, *shape, dtype=dtype, device=device)
        )
        work = dist.all_gather_into_tensor(
            gathered_buffer, updated_params, async_op=True
        )
        gather_list.append({
            "work": work,
            "stacked_params": gathered_buffer,
            "params": params,
        })


