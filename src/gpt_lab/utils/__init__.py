"""Utility exports loaded only when accessed."""

from importlib import import_module


_EXPORTS = {
    "Board": "board",
    "get_banner": "common",
    "print0": "common",
    "print0_dict": "common",
    "is_rank0": "common",
    "get_rank": "common",
    "init_dist_groups": "distributed",
    "cleanup_dist_groups": "distributed",
    "get_device_type": "distributed",
    "get_dist_info": "distributed",
    "init_logger": "logging",
    "log0": "logging",
    "log_dict": "logging",
    "SpecialTokens": "special_tokens",
    "get_git_info": "system",
    "get_gpu_info": "system",
    "get_system_info": "system",
    "estimate_cost": "system",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "Board",
    "get_banner",
    "print0",
    "print0_dict",
    "init_dist_groups",
    "cleanup_dist_groups",
    "is_rank0",
    "get_device_type",
    "get_dist_info",
    "init_logger",
    "log0",
    "log_dict",
    "SpecialTokens",
    "get_git_info",
    "get_gpu_info",
    "get_system_info",
    "estimate_cost",
]
