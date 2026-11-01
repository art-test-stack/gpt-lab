"""Optimizer factory with registry-based architecture.

Simple factory that assembles optimizers from registered strategies.
Automatically selects single-GPU or distributed backend.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from gpt_lab.optim.registry import OPTIMIZER_REGISTRY, OptimizerSpec, _validate_group, register_optimizer
from gpt_lab.optim.strategy import ScalarCache, _DistributedBackend, _LocalBackend
from gpt_lab.optim.strategies import AdamWStrategy, MuonStrategy, NorMuonStrategy
from gpt_lab.utils.distributed import ComputeContext, get_comp_info

# ── Register built-in optimizers ──────────────────────────────────────────────

register_optimizer(
    OptimizerSpec(
        name="adamw",
        required_keys=frozenset({"opt", "lr", "betas", "eps", "weight_decay"}),
        strategy_class=AdamWStrategy,
    )
)

register_optimizer(
    OptimizerSpec(
        name="muon",
        required_keys=frozenset({"opt", "lr", "momentum", "beta", "ns_steps", "weight_decay"}),
        strategy_class=MuonStrategy,
    )
)

register_optimizer(
    OptimizerSpec(
        name="normuon",
        required_keys=frozenset({"opt", "lr", "momentum", "beta", "ns_steps", "weight_decay"}),
        strategy_class=NorMuonStrategy,
    )
)



class OptimizerFactory(torch.optim.Optimizer):
    """Registry-based optimizer factory.
    
    Supports mixed AdamW, Muon, and NorMuon groups.
    Automatically detects and uses distributed backend if DDP is initialized.
    """
    
    def __init__(
        self, param_groups: list[dict[str, Any]], comp_info: Optional[ComputeContext] = None
    ) -> None:
        # Validate all groups early — fail before training starts
        for idx, group in enumerate(param_groups):
            _validate_group(group, idx)
            group.setdefault("initial_lr", group["lr"])
            group.setdefault("initial_weight_decay", group["weight_decay"])
            if group["opt"] in ("muon", "normuon"):
                group.setdefault("initial_momentum", group["momentum"])
        
        super().__init__(param_groups, defaults={})
        
        # Allocate scalar cache once, reused every step
        self.scalars = ScalarCache()
        
        # Detect distributed mode
        if comp_info is None:
            comp_info = get_comp_info()
        self.comp_info = comp_info
        self.communication = None
        
        # Select backend
        if comp_info.is_distributed:
            self.backend = _DistributedBackend(self)
        else:
            self.backend = _LocalBackend(self)

    def configure_communication(self, mode: str, bucket_bytes: int, timing: bool = False) -> None:
        """Select an experimental transport without changing optimizer state/groups."""
        if self.communication is not None:
            self.communication.close()
            self.communication = None
        if mode != "baseline":
            from gpt_lab.optim.communication import BucketCommunication
            self.communication = BucketCommunication(self, mode, bucket_bytes, timing)

    def begin_backward_communication(self, final_microstep: bool) -> None:
        if self.communication is not None:
            self.communication.begin_backward(final_microstep)

    def finish_gradient_communication(self) -> None:
        """Drain optimizer reductions before an independent trainer collective."""
        if self.communication is not None:
            self.communication.finish_reductions()
    
    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        """Execute one optimizer step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.backend.step()
        return loss
    
    def state_dict(self) -> dict:
        """Return optimizer state dict."""
        return super().state_dict()
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state dict."""
        if self.communication is not None:
            self.communication.reset()
        super().load_state_dict(state_dict)
        # Older checkpoints predate the schedule baselines.
        for group in self.param_groups:
            group.setdefault("initial_lr", group["lr"])
            group.setdefault("initial_weight_decay", group["weight_decay"])
            if group["opt"] in ("muon", "normuon"):
                group.setdefault("initial_momentum", group["momentum"])
    
    @torch.no_grad()
    def update_hyperparams(
        self,
        lrm: float = 1.0,
        muon_momentum: Optional[float] = None,
        weight_decay: Optional[float] = None,
    ) -> tuple[float, Optional[float], Optional[float]]:
        """Update learning rate and other hyperparameters.
        
        Called by LR scheduler each step.
        
        Args:
            lrm: Learning rate multiplier, applied via initial_lr to all groups.
            muon_momentum: Momentum multiplier for Muon and NorMuon groups.
            weight_decay: Weight-decay multiplier for all groups.
        """
        for group in self.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            
            if weight_decay is not None:
                group["weight_decay"] = group["initial_weight_decay"] * weight_decay
            
            if group["opt"] in ("muon", "normuon") and muon_momentum is not None:
                group["momentum"] = group["initial_momentum"] * muon_momentum
        
        return lrm, muon_momentum, weight_decay


# ══════════════════════════════════════════════════════════════════════════════
# Backward compatibility: thin aliases so existing code doesn't break
# ══════════════════════════════════════════════════════════════════════════════
MuonAdamW = OptimizerFactory
DistMuonAdamW = OptimizerFactory
