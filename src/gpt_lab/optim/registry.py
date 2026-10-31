"""Optimizer registry and validation.

Enables adding new optimizers with a single registry entry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpt_lab.optim.strategy import OptimizerStrategy


@dataclass(frozen=True)
class OptimizerSpec:
    """Specification for an optimizer registered in the factory.
    
    Fields:
        name: Human-readable optimizer name (e.g., "adamw", "muon").
        required_keys: Frozenset of required YAML config keys for this optimizer.
        strategy_class: Strategy class implementing OptimizerStrategy for this optimizer.
    """
    name: str
    required_keys: frozenset[str]
    strategy_class: type[OptimizerStrategy]


# Registry mapping optimizer name to spec
OPTIMIZER_REGISTRY: dict[str, OptimizerSpec] = {}


def register_optimizer(spec: OptimizerSpec) -> None:
    """Register an optimizer spec in the global registry."""
    OPTIMIZER_REGISTRY[spec.name] = spec


def _validate_group(group: dict[str, Any], idx: int) -> None:
    """Validate a parameter group against its optimizer's spec.
    
    Raises ValueError if required keys are missing or optimizer is unknown.
    Called at construction — catches misconfigured YAML before training starts.
    """
    opt_name = group.get("opt")
    if opt_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Param group [{idx}]: unknown optimizer type {opt_name!r}. "
            f"Available: {set(OPTIMIZER_REGISTRY.keys())}"
        )
    
    spec = OPTIMIZER_REGISTRY[opt_name]
    missing = spec.required_keys - group.keys()
    if missing:
        raise ValueError(
            f"Param group [{idx}] (opt={opt_name!r}) is missing required keys: {missing}. "
            f"Check that your YAML config or code defaults supply all required fields."
        )

    if opt_name == "normuon":
        if not isinstance(group.get("anvil_spectral", False), bool):
            raise ValueError("NorMuon anvil_spectral must be a boolean")
        for key in ("lr", "weight_decay", "momentum", "beta"):
            value = group[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"NorMuon {key} must be finite and non-negative")
            if key in ("momentum", "beta") and value >= 1:
                raise ValueError(f"NorMuon {key} must be less than 1")
        steps = group["ns_steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("NorMuon ns_steps must be a positive integer")
