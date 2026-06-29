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
    """Muon optimizer strategy for single and distributed training."""
    
    def local_step(self, group: dict) -> None:
        params: list[torch.Tensor] = group["params"]
        if not params:
            return
        
        p = params[0]
        state = self.opt.state[p]
        shape, device, dtype = p.shape, p.device, p.dtype
        
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (len(params), shape[-2], 1) if shape[-2] >= shape[-1]
                else (len(params), 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        
        self.opt.scalars.muon_momentum_t.fill_(group["momentum"])
        self.opt.scalars.muon_beta2_t.fill_(group.get("beta") or 0.0)
        self.opt.scalars.muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self.opt.scalars.muon_wd_t.fill_(group["weight_decay"])
        
        muon_step(
            stacked_grads, stacked_params, state["momentum_buffer"],
            state["second_momentum_buffer"], self.opt.scalars.muon_momentum_t,
            self.opt.scalars.muon_lr_t, self.opt.scalars.muon_wd_t,
            self.opt.scalars.muon_beta2_t, group["ns_steps"], red_dim, dtype,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))
    
    def dist_reduce(self, group: dict, world_size: int) -> dict:
        params = group["params"]
        if not params:
            return {"future": None, "grad_chunk": None}
        
        chunk_size = (len(params) + world_size - 1) // world_size
        padded = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded:
            stacked_grads[len(params):].zero_()
        
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
        ).get_future()
        
        return {
            "future": future,
            "grad_chunk": grad_chunk,
            "stacked_grads": stacked_grads,
            "chunk_size": chunk_size,
        }
    
    def dist_compute(
        self, group: dict, info: dict, gather_list: list, rank: int, world_size: int,
    ) -> None:
        if info["future"] is None:
            return
        
        info["future"].wait()
        params = group["params"]
        chunk_size = info["chunk_size"]
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        
        state = self.opt.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1]
                else (chunk_size, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        
        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)
            
            self.opt.scalars.muon_momentum_t.fill_(group["momentum"])
            self.opt.scalars.muon_beta2_t.fill_(group["beta"])
            self.opt.scalars.muon_lr_t.fill_(
                group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
            )
            self.opt.scalars.muon_wd_t.fill_(group["weight_decay"])
            
            muon_step(
                info["grad_chunk"][:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned],
                state["second_momentum_buffer"][:num_owned],
                self.opt.scalars.muon_momentum_t, self.opt.scalars.muon_lr_t,
                self.opt.scalars.muon_wd_t, self.opt.scalars.muon_beta2_t,
                group["ns_steps"], red_dim, dtype,
            )
            updated_params[:num_owned].copy_(stacked_owned)
        
        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()
        
        gathered_buffer = info["stacked_grads"]
        future = dist.all_gather_into_tensor(
            gathered_buffer, updated_params, async_op=True
        ).get_future()
        gather_list.append({
            "future": future,
            "stacked_params": gathered_buffer,
            "params": params,
        })


