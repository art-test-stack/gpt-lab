import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from safetensors.torch import save_file

from gpt_lab.utils.tensor_io import load_tensor_state, migrate_legacy_tensor_state, save_tensor_state


def test_unsupported_state_fails_before_creating_a_file(tmp_path):
    path = tmp_path / "state.safetensors"
    with pytest.raises(TypeError, match="Unsupported tensor state type"):
        save_tensor_state({"unsupported": object()}, path)
    assert not path.exists()


@pytest.mark.parametrize("payload", [
    {"version": 2, "state": None},
    {"version": 1, "state": {"type": "object"}},
    {"version": 1, "state": {"type": "dtype", "value": "load"}},
])
def test_invalid_state_metadata_is_rejected(tmp_path, payload):
    path = tmp_path / "state.safetensors"
    save_file({}, path, metadata={"gpt_lab_state": json.dumps(payload)})
    with pytest.raises(ValueError):
        load_tensor_state(path)


@pytest.mark.parametrize("extension", ["pt", "pth"])
def test_legacy_conversion_is_reused_and_preserves_original(tmp_path, monkeypatch, extension):
    path = tmp_path / f"state.{extension}"
    state = {"weights": torch.arange(4)}
    torch.save(state, path)
    original = path.read_bytes()
    with ThreadPoolExecutor(max_workers=2) as workers:
        converted = list(workers.map(migrate_legacy_tensor_state, [path, path]))
    assert converted == [path.with_suffix(".safetensors")] * 2
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".state.safetensors.*"))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Legacy state must only be decoded once"))
    torch.testing.assert_close(load_tensor_state(migrate_legacy_tensor_state(path))["weights"], state["weights"])


def test_failed_conversion_does_not_publish_or_use_partial_state(tmp_path, monkeypatch):
    path = tmp_path / "state.pt"
    torch.save({"weights": torch.ones(2)}, path)
    original = path.read_bytes()

    def fail_after_writing(state, temporary):
        temporary.write_bytes(b"incomplete")
        raise OSError("simulated conversion failure")

    monkeypatch.setattr("gpt_lab.utils.tensor_io.save_tensor_state", fail_after_writing)
    with pytest.raises(OSError, match="conversion failure"):
        migrate_legacy_tensor_state(path)
    assert path.read_bytes() == original
    assert not path.with_suffix(".safetensors").exists()
    assert list(tmp_path.iterdir()) == [path]
