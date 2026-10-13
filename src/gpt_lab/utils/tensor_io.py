"""Safetensors files with JSON metadata for nested training state."""

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import save_file


def save_tensor_state(state, path: str | Path) -> None:
    """Save tensors, containers, scalars, NumPy arrays, devices, and dtypes."""
    tensors = {}

    def add_tensor(value):
        key = f"tensor_{len(tensors)}"
        tensors[key] = value.detach().to("cpu", copy=True).contiguous()
        return key

    def encode(value):
        if isinstance(value, torch.Tensor):
            return {"type": "tensor", "key": add_tensor(value)}
        if isinstance(value, (np.ndarray, np.generic)):
            return {"type": "numpy" if isinstance(value, np.ndarray) else "numpy_scalar",
                    "key": add_tensor(torch.from_numpy(np.asarray(value).copy()))}
        if isinstance(value, dict):
            return {"type": "dict", "items": [[encode(key), encode(item)] for key, item in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"type": "tuple" if isinstance(value, tuple) else "list", "items": [encode(item) for item in value]}
        if isinstance(value, torch.device):
            return {"type": "device", "value": str(value)}
        if isinstance(value, torch.dtype):
            return {"type": "dtype", "value": str(value).removeprefix("torch.")}
        if isinstance(value, float) and not math.isfinite(value):
            return {"type": "float", "value": str(value)}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f"Unsupported tensor state type: {type(value).__name__}")

    # Flat model state dicts stay readable by ordinary safetensors consumers.
    if isinstance(state, dict) and all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        tensors = {key: value.detach().to("cpu", copy=True).contiguous() for key, value in state.items()}
        metadata = None
    else:
        metadata = {"gpt_lab_state": json.dumps({"version": 1, "state": encode(state)}, allow_nan=False)}
    save_file(tensors, path, metadata=metadata)


def load_tensor_state(path: str | Path, device="cpu"):
    """Load a flat tensor dictionary or reconstruct nested training state."""
    with safe_open(path, framework="pt", device=str(device)) as file:
        tensors = {key: file.get_tensor(key) for key in file.keys()}
        metadata = file.metadata() or {}
    if "gpt_lab_state" not in metadata:
        return tensors
    payload = json.loads(metadata["gpt_lab_state"])
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported tensor state metadata version")

    def decode(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if not isinstance(value, dict):
            raise ValueError("Invalid tensor state metadata")
        kind = value.get("type")
        if kind == "tensor":
            return tensors[value["key"]]
        if kind in ("numpy", "numpy_scalar"):
            array = tensors[value["key"]].cpu().numpy().copy()
            return array[()] if kind == "numpy_scalar" else array
        if kind == "dict":
            return {decode(key): decode(item) for key, item in value["items"]}
        if kind in ("list", "tuple"):
            items = [decode(item) for item in value["items"]]
            return tuple(items) if kind == "tuple" else items
        if kind == "device":
            return torch.device(value["value"])
        if kind == "dtype":
            dtype = getattr(torch, value["value"], None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError("Invalid tensor state dtype")
            return dtype
        if kind == "float" and value["value"] in ("inf", "-inf", "nan"):
            return float(value["value"])
        raise ValueError(f"Unsupported tensor state metadata type: {kind!r}")

    return decode(payload["state"])


def migrate_legacy_tensor_state(path: str | Path, *, weights_only=True) -> Path:
    """Convert once, retain the original, and publish a complete safetensors file."""
    path = Path(path)
    target = path.with_suffix(".safetensors")
    if target.exists():
        return target
    state = torch.load(path, map_location="cpu", weights_only=weights_only)
    if isinstance(state, BaseModel):
        state = state.model_dump()
    # Ranks can migrate shared files concurrently; readers must see a complete file.
    with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".safetensors", delete=False) as file:
        temporary = Path(file.name)
    try:
        save_tensor_state(state, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
