"""Strategy interface and backends for optimizer execution.

Separates optimizer logic (per-strategy) from execution mode (local vs distributed).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from gpt_lab.optim.factory import OptimizerFactory


@dataclass
class ScalarCache:
    """0-D CPU tensor cache for torch.compile stability.
    
    Strategies fill these once per step via .fill_() instead of creating
    new tensors, keeping the torch.compile graph stable (same tensor identity).
    """
    # AdamW scalars
    adamw_step_t: torch.Tensor = None
    adamw_lr_t: torch.Tensor = None
    adamw_beta1_t: torch.Tensor = None
    adamw_beta2_t: torch.Tensor = None
    adamw_eps_t: torch.Tensor = None
    adamw_wd_t: torch.Tensor = None
    # Muon scalars
    muon_momentum_t: torch.Tensor = None
    muon_lr_t: torch.Tensor = None
    muon_wd_t: torch.Tensor = None
    muon_beta2_t: torch.Tensor = None

    def __post_init__(self) -> None:
        """Initialize all scalars on first instantiation."""
        if self.adamw_step_t is None:
            for attr in self.__dataclass_fields__:
                setattr(self, attr, torch.tensor(0.0, dtype=torch.float32, device="cpu"))


class OptimizerStrategy(ABC):
    """Abstract base for optimizer strategies.
    
    Each concrete optimizer (AdamW, Muon, etc.) implements these methods.
    Strategies have access to the factory instance for state, param groups, and scalars.
    """
    
    def __init__(self, opt: OptimizerFactory) -> None:
        self.opt = opt
    
    @abstractmethod
    def local_step(self, group: dict) -> None:
        """Single-GPU update step for a parameter group."""
        pass
    
    @abstractmethod
    def dist_reduce(self, group: dict, world_size: int) -> dict:
        """Launch async gradient reduction operations. Returns info dict for dist_compute."""
        pass
    
    @abstractmethod
    def dist_compute(
        self,
        group: dict,
        info: dict,
        gather_list: list,
        rank: int,
        world_size: int,
    ) -> None:
        """Wait for reduces, compute updates, launch gathers."""
        pass


class _LocalBackend:
    """Single-GPU backend: local_step only."""
    
    def __init__(self, opt: OptimizerFactory) -> None:
        self.opt = opt
        # Import here to avoid circular dependency
        from gpt_lab.optim.registry import OPTIMIZER_REGISTRY
        self.strategies = {
            name: spec.strategy_class(opt)
            for name, spec in OPTIMIZER_REGISTRY.items()
        }
    
    def step(self) -> None:
        """Execute one optimizer step locally."""
        for group in self.opt.param_groups:
            strategy = self.strategies[group["opt"]]
            strategy.local_step(group)


class _DistributedBackend:
    """Distributed backend: 3-phase async pattern (reduce, compute, gather)."""
    
    def __init__(self, opt: OptimizerFactory) -> None:
        self.opt = opt
        # Import here to avoid circular dependency
        from gpt_lab.optim.registry import OPTIMIZER_REGISTRY
        self.strategies = {
            name: spec.strategy_class(opt)
            for name, spec in OPTIMIZER_REGISTRY.items()
        }
    
    def step(self) -> None:
        """Execute one distributed optimizer step with async communication."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Phase 1: Launch all async reduce ops (no waiting)
        reduce_infos: list[dict] = []
        for group in self.opt.param_groups:
            strategy = self.strategies[group["opt"]]
            reduce_infos.append(strategy.dist_reduce(group, world_size))
        
        # Phase 2: Wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.opt.param_groups, reduce_infos):
            strategy = self.strategies[group["opt"]]
            strategy.dist_compute(group, info, gather_list, rank, world_size)
        
        # Phase 3: Wait for all gathers and copy back (Muon only)
        self._finish_gathers(gather_list)
    
    @staticmethod
    def _finish_gathers(gather_list: list) -> None:
        """Wait for all async gathers and copy params back from comm buffers."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(
                    info["params"],
                    list(info["stacked_params"][: len(info["params"])].unbind(0)),
                )
