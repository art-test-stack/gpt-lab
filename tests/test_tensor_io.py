import json

import pytest
from safetensors.torch import save_file

from gpt_lab.utils.tensor_io import load_tensor_state, save_tensor_state


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
