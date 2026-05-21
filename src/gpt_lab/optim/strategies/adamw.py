"""AdamW optimizer strategy."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from gpt_lab.optim.kernels.adamw import adamw_step_fused as adamw_step
from gpt_lab.optim.strategy import OptimizerStrategy

if TYPE_CHECKING:
    from gpt_lab.optim.factory import OptimizerFactory


class AdamWStrategy(OptimizerStrategy):
    """AdamW optimizer strategy for single and distributed training."""
    
    def local_step(self, group: dict) -> None:
        """AdamW update for each parameter in the group (single GPU)."""
        for p in group["params"]:
            if p.grad is None:
                continue
            
            grad = p.grad
            state = self.opt.state[p]
            
            # Lazy state initialization
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            
            # Fill 0-D scalars with current values
            self.opt.scalars.adamw_step_t.fill_(state["step"])
            self.opt.scalars.adamw_lr_t.fill_(group["lr"])
            self.opt.scalars.adamw_beta1_t.fill_(group["betas"][0])
            self.opt.scalars.adamw_beta2_t.fill_(group["betas"][1])
            self.opt.scalars.adamw_eps_t.fill_(group["eps"])
            self.opt.scalars.adamw_wd_t.fill_(group["weight_decay"])
            
            # Fused kernel: weight_decay → momentum → bias_correction → param_update
            adamw_step(
                p,
                grad,
                exp_avg,
                exp_avg_sq,
                self.opt.scalars.adamw_step_t,
                self.opt.scalars.adamw_lr_t,
                self.opt.scalars.adamw_beta1_t,
                self.opt.scalars.adamw_beta2_t,
                self.opt.scalars.adamw_eps_t,
                self.opt.scalars.adamw_wd_t,
            )
    
    def dist_reduce(self, group: dict, world_size: int) -> dict:
        """Launch async gradient reductions for AdamW parameters.
        
        Small params (<1024 els): all_reduce, replicated update.
        Large params: reduce_scatter, sharded state (ZeRO-2 style).
        """
        param_infos = {}
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            
            if p.numel() < 1024:
                # Small: all_reduce (no scatter/gather)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = {"future": future, "grad_slice": grad, "is_small": True}
            else:
                # Large: reduce_scatter for sharding
                assert (
                    grad.shape[0] % world_size == 0
                ), f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                param_infos[p] = {"future": future, "grad_slice": grad_slice, "is_small": False}
        
        return {"param_infos": param_infos}
    
    def dist_compute(
        self,
        group: dict,
        info: dict,
        gather_list: list,
        rank: int,
        world_size: int,
    ) -> None:
        """Wait for reduces, compute updates, launch all_gather for large params."""
        param_infos = info["param_infos"]
        
        for p in group["params"]:
            if p not in param_infos:
                continue
            
            pinfo = param_infos[p]
            pinfo["future"].wait()
            grad_slice = pinfo["grad_slice"]
            state = self.opt.state[p]
            
            # Determine param slice (full for small, sharded for large)
            if pinfo["is_small"]:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size : (rank + 1) * rank_size]
            
            # Lazy state initialization
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)
            
            state["step"] += 1
            
            # Fill 0-D scalars and run fused kernel
            self.opt.scalars.adamw_step_t.fill_(state["step"])
            self.opt.scalars.adamw_lr_t.fill_(group["lr"])
            self.opt.scalars.adamw_beta1_t.fill_(group["betas"][0])
            self.opt.scalars.adamw_beta2_t.fill_(group["betas"][1])
            self.opt.scalars.adamw_eps_t.fill_(group["eps"])
            self.opt.scalars.adamw_wd_t.fill_(group["weight_decay"])
            
            adamw_step(
                p_slice,
                grad_slice,
                state["exp_avg"],
                state["exp_avg_sq"],
                self.opt.scalars.adamw_step_t,
                self.opt.scalars.adamw_lr_t,
                self.opt.scalars.adamw_beta1_t,
                self.opt.scalars.adamw_beta2_t,
                self.opt.scalars.adamw_eps_t,
                self.opt.scalars.adamw_wd_t,
            )
            
            # Large params need all_gather to sync back
            if not pinfo["is_small"]:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append({"future": future, "params": None})
