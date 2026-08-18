import pytest
import torch
import numpy as np
import random
from gpt_lab.model.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    make_default_run_name,
    save_checkpoint,
    set_rng_state,
)
from gpt_lab.utils.schemas import CheckpointState, TrainerState

def numpy_rng_equal(s1, s2):
    if s1[0] != s2[0]:  # algorithm name
        return False
    if not np.array_equal(s1[1], s2[1]):  # state array
        return False
    return s1[2:] == s2[2:] 

def assert_rng_state_consistency(rng_state1, rng_state2, raise_msg="RNG states do not match."):
    assert torch.equal(rng_state1["torch"], rng_state2["torch"]), f"{raise_msg} Torch RNG states do not match."
    assert numpy_rng_equal(rng_state1["numpy"], rng_state2["numpy"]), f"{raise_msg} Numpy RNG states do not match."
    assert rng_state1["python"] == rng_state2["python"], f"{raise_msg} Python random RNG states do not match."
    if torch.cuda.is_available():
        cuda1, cuda2 = rng_state1["cuda"], rng_state2["cuda"]
        if isinstance(cuda1, (list, tuple)):
            for s1, s2 in zip(cuda1, cuda2):
                assert torch.equal(s1, s2), f"{raise_msg} CUDA RNG states do not match."
        else:
            assert torch.equal(cuda1, cuda2), f"{raise_msg} CUDA RNG states do not match."

@pytest.mark.fast
def test_checkpoint_rng_state_capture():
    # Capture the RNG state before saving the checkpoint
    rng_state_before = capture_rng_state()

    # Simulate saving and loading a checkpoint (you can replace this with actual checkpoint code)
    # For this test, we'll just set the RNG state back to the captured state
    set_rng_state(rng_state_before)

    # Capture the RNG state after loading the checkpoint
    rng_state_after = capture_rng_state()

    # Assert that the RNG states are the same
    assert_rng_state_consistency(rng_state_before, rng_state_after, "RNG state was not preserved across checkpoint save/load.")

@pytest.mark.fast
def test_checkpoint_rng_state_with_seed():
    random_seed = [42, 1234, 9999]
    for seed in random_seed:
        # Set the RNG state to a specific seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Capture the RNG state after setting it
        rng_state_set = capture_rng_state()

        # Set the RNG state again to the same seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Capture the RNG state again
        rng_state_set_again = capture_rng_state()

        # Assert that the RNG states are the same
        assert_rng_state_consistency(rng_state_set, rng_state_set_again, f"RNG state was not consistent for seed {seed}.")

@pytest.mark.fast
def test_checkpoint_rng_state_with_random_values():
    # Set a random seed and capture the RNG state
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng_state_before = capture_rng_state()

    # Generate some random values
    torch.rand(5)
    np.random.rand(5)
    random.random()

    # Capture the RNG state after generating random values
    rng_state_after = capture_rng_state()

    # Assert that the RNG states are different after generating random values
    with pytest.raises(AssertionError):
        assert_rng_state_consistency(rng_state_before, rng_state_after, "RNG state should change after generating random values.")

    # Now set the RNG state back to the captured state
    set_rng_state(rng_state_before)

    # Capture the RNG state again
    rng_state_restored = capture_rng_state()

    # Assert that the restored RNG state matches the original captured state
    assert_rng_state_consistency(rng_state_before, rng_state_restored, "RNG state was not correctly restored after setting it back.")


class _MarkerOptimizer:
    def __init__(self, marker):
        self.marker = marker

    def state_dict(self):
        return {"marker": torch.tensor(self.marker)}


@pytest.mark.fast
def test_sharded_checkpoint_round_trips_each_rank_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    checkpoint_state = CheckpointState(best_eval_step=7, best_eval_value=1.5)

    for rank in range(2):
        torch.manual_seed(100 + rank)
        save_checkpoint(
            model=model,
            checkpoint_dir=tmp_path,
            step=7,
            trainer_state=TrainerState(step=7, n_tokens=1000 + rank),
            checkpoint_state=checkpoint_state,
            optimizer=_MarkerOptimizer(rank),
            mode="shard",
            dist_info={
                "RANK": rank,
                "WORLD_SIZE": 2,
                "DEVICE": torch.device("cpu"),
                "IS_DDP_INITIALIZED": False,
            },
        )

    step_dir = tmp_path / "checkpoint_step_000007"
    assert (step_dir / "optim_rank0.pt").exists()
    assert (step_dir / "optim_rank1.pt").exists()
    assert (step_dir / "trainer_state_rank0.json").exists()
    assert (step_dir / "trainer_state_rank1.json").exists()
    assert (step_dir / "rng_state_rank0.pt").exists()
    assert (step_dir / "rng_state_rank1.pt").exists()

    for rank in range(2):
        loaded = load_checkpoint(
            tmp_path,
            step=7,
            mode="shard",
            load_scaler=False,
            dist_info={
                "RANK": rank,
                "WORLD_SIZE": 2,
                "DEVICE": torch.device("cpu"),
                "IS_DDP_INITIALIZED": False,
            },
        )
        assert loaded.optimizer_state["marker"].item() == rank
        assert loaded.trainer_state.n_tokens == 1000 + rank
        assert loaded.checkpoint_state.best_eval_step == 7
        assert loaded.checkpoint_state.best_eval_value == pytest.approx(1.5)
        assert loaded.rng_state is not None


@pytest.mark.fast
def test_sharded_checkpoint_rejects_world_size_change(tmp_path):
    model = torch.nn.Linear(2, 2)
    save_checkpoint(
        model=model,
        checkpoint_dir=tmp_path,
        step=1,
        trainer_state=TrainerState(step=1),
        checkpoint_state=CheckpointState(),
        optimizer=_MarkerOptimizer(0),
        mode="shard",
        dist_info={
            "RANK": 0,
            "WORLD_SIZE": 2,
            "DEVICE": torch.device("cpu"),
            "IS_DDP_INITIALIZED": False,
        },
    )

    with pytest.raises(ValueError, match="same world size"):
        load_checkpoint(
            tmp_path,
            step=1,
            mode="shard",
            load_scaler=False,
            dist_info={
                "RANK": 0,
                "WORLD_SIZE": 4,
                "DEVICE": torch.device("cpu"),
                "IS_DDP_INITIALIZED": False,
            },
        )


@pytest.mark.fast
def test_default_run_name_is_broadcast_from_rank_zero(monkeypatch):
    def fake_broadcast(names, src):
        assert src == 0
        names[0] = "shared-run-name"

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake_broadcast)

    run_name = make_default_run_name(
        depth=12,
        name="ic1",
        dist_info={
            "RANK": 1,
            "IS_DDP_INITIALIZED": True,
            "DEVICE_NAME": "GPU",
        },
    )

    assert run_name == "shared-run-name"
