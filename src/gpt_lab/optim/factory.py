"""Optimizer factory with registry-based architecture.

Simple factory that assembles optimizers from registered strategies.
Automatically selects single-GPU or distributed backend.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from gpt_lab.optim.registry import OPTIMIZER_REGISTRY, OptimizerSpec, _validate_group, register_optimizer
from gpt_lab.optim.strategy import ScalarCache, _DistributedBackend, _LocalBackend
from gpt_lab.optim.strategies import AdamWStrategy, MuonStrategy
from gpt_lab.utils.distributed import get_dist_info

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



class OptimizerFactory(torch.optim.Optimizer):
    """Registry-based optimizer factory.
    
    Supports mixed optimizer groups (AdamW + Muon) with minimal code duplication.
    Automatically detects and uses distributed backend if DDP is initialized.
    """
    
    def __init__(
        self, param_groups: list[dict[str, Any]], dist_info: Optional[dict] = None
    ) -> None:
        # Validate all groups early — fail before training starts
        for idx, group in enumerate(param_groups):
            _validate_group(group, idx)
            group.setdefault("initial_lr", group["lr"])
        
        super().__init__(param_groups, defaults={})
        
        # Allocate scalar cache once, reused every step
        self.scalars = ScalarCache()
        
        # Detect distributed mode
        if dist_info is None:
            dist_info = get_dist_info()
        self.dist_info = dist_info
        
        # Select backend
        if dist_info["IS_DDP_INITIALIZED"]:
            self.backend = _DistributedBackend(self)
        else:
            self.backend = _LocalBackend(self)
    
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
        super().load_state_dict(state_dict)
    
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
            muon_momentum: Override momentum for Muon groups only.
            weight_decay: Override weight_decay for all groups.
        """
        for group in self.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            
            if weight_decay is not None:
                group["weight_decay"] = weight_decay
            
            if group["opt"] == "muon" and muon_momentum is not None:
                group["momentum"] = muon_momentum
        
        return lrm, muon_momentum, weight_decay


# ══════════════════════════════════════════════════════════════════════════════
# Backward compatibility: thin aliases so existing code doesn't break
# ══════════════════════════════════════════════════════════════════════════════
MuonAdamW = OptimizerFactory
DistMuonAdamW = OptimizerFactory
