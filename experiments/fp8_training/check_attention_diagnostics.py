"""Offline failure diagnostics; no kernel downloads or CUDA execution."""

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch

from gpt_lab.utils import import_utils


def fail_get_kernel(_name):
    raise ImportError("diagnostic sentinel: missing CUDA symbol")


def test_availability_failure_reports_original_traceback(monkeypatch):
    monkeypatch.setattr(import_utils, "is_torch_cuda_available", lambda: True)
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=fail_get_kernel))
    check = import_utils.is_flash_attn3_available_from_kernel
    check.cache_clear()
    try:
        with pytest.warns(UserWarning, match="Traceback") as captured:
            assert check() is False
        message = str(captured[0].message)
        assert "ImportError: diagnostic sentinel: missing CUDA symbol" in message
        assert "fail_get_kernel" in message
    finally:
        check.cache_clear()


def test_backend_import_failure_reports_original_traceback(monkeypatch):
    monkeypatch.setattr(import_utils, "is_torch_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=fail_get_kernel))
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    path = Path(__file__).resolve().parents[2] / "src/gpt_lab/model/flash_attn.py"
    with pytest.warns(RuntimeWarning, match="Traceback") as captured:
        state = runpy.run_path(str(path))
    assert "ImportError: diagnostic sentinel: missing CUDA symbol" in str(captured[0].message)
    assert state["_flash_attn"] is None
    assert state["_flash_attn2"] is None
    assert state["flash_attention_is_installed"] is False
    with pytest.raises(RuntimeError, match="refusing SDPA fallback") as failure:
        state["require_flash_attn3"]()
    assert isinstance(failure.value.__cause__, ImportError)
    assert "diagnostic sentinel" in str(failure.value.__cause__)

    from gpt_lab.model import layers
    from gpt_lab.utils.schemas import TransformerConfig

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(layers, "require_flash_attn3", state["require_flash_attn3"])
    config = TransformerConfig(vocab_size=64, d_model=32, n_heads=2, d_head=16,
                               n_layers=2, max_context=8, attn_impl="fa3")
    with pytest.raises(RuntimeError, match="refusing SDPA fallback"):
        layers.CausalSelfAttention(config, layer_idx=0)
    layers.CausalSelfAttention(config.model_copy(update={"attn_impl": "sdpa"}), layer_idx=0)

    state["require_flash_attn3"].__globals__["_flash_attn"] = SimpleNamespace()
    state["require_flash_attn3"]()
