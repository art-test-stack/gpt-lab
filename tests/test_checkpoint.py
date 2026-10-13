from types import SimpleNamespace
import json
from gpt_lab.utils.distributed import ComputeContext
import pytest
import torch
import numpy as np
import random
from gpt_lab.model.checkpoint import (
    CheckpointManager,
    capture_rng_state,
    load_checkpoint,
    load_optimizer_state,
    make_default_run_name,
    save_checkpoint,
    set_rng_state,
)
from gpt_lab.utils.schemas import CheckpointState, MetaConfig, TrainerConfig, TrainerState, TransformerConfig

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
@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_sharded_checkpoint_round_trips_each_rank_state(tmp_path, legacy_metadata):
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
            optimizer_state_layout="sharded",
            comp_info=SimpleNamespace(rank=rank, world_size=2, device=torch.device("cpu"), is_distributed=False),
        )

    step_dir = tmp_path / "checkpoint_step_000007"
    metadata_path = step_dir / "checkpoint_meta.json"
    assert json.loads(metadata_path.read_text()) == {"optimizer_state_layout": "sharded", "world_size": 2}
    if legacy_metadata:
        metadata_path.write_text(json.dumps({"mode": "shard", "world_size": 2}))
    assert (step_dir / "model.safetensors").exists()
    assert (step_dir / "optim_rank0.safetensors").exists()
    assert (step_dir / "optim_rank1.safetensors").exists()
    assert (step_dir / "trainer_state_rank0.json").exists()
    assert (step_dir / "trainer_state_rank1.json").exists()
    assert (step_dir / "rng_state_rank0.safetensors").exists()
    assert (step_dir / "rng_state_rank1.safetensors").exists()

    for rank in range(2):
        loaded = load_checkpoint(
            tmp_path,
            step=7,
            optimizer_state_layout="sharded",
            load_scaler=False,
            comp_info=SimpleNamespace(rank=rank, world_size=2, device=torch.device("cpu"), is_distributed=False),
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
        optimizer_state_layout="sharded",
        comp_info=SimpleNamespace(rank=0, world_size=2, device=torch.device("cpu"), is_distributed=False),
    )

    with pytest.raises(ValueError, match="same world size"):
        load_checkpoint(
            tmp_path,
            step=1,
            optimizer_state_layout="sharded",
            load_scaler=False,
            comp_info=SimpleNamespace(rank=0, world_size=4, device=torch.device("cpu"), is_distributed=False),
        )


@pytest.mark.fast
@pytest.mark.parametrize("device_name, prefix", [
    ("Apple M2 (10-core GPU)", "apple_m2"),
    ("Apple M2 Pro (19-core GPU)", "apple_m2_pro"),
    ("NVIDIA GeForce RTX 4090", "4090"),
    (None, "none"),
])
def test_default_run_name_preserves_apple_chip_label(monkeypatch, device_name, prefix):
    monkeypatch.setattr("gpt_lab.utils.system.run_command", lambda _command: "abc123")
    run_name = make_default_run_name(
        depth=12,
        name="ic1",
        comp_info=SimpleNamespace(rank=0, is_distributed=False, device_name=device_name),
    )
    assert run_name.startswith(f"{prefix}_ic1_d12_cmt_abc123_dt_")


@pytest.mark.fast
def test_default_run_name_is_broadcast_from_rank_zero(monkeypatch):
    def fake_broadcast(names, src):
        assert src == 0
        names[0] = "shared-run-name"

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake_broadcast)

    run_name = make_default_run_name(
        depth=12,
        name="ic1",
        comp_info=SimpleNamespace(rank=1, world_size=1, device=torch.device("cpu"), is_distributed=True, device_name="GPU"),
    )

    assert run_name == "shared-run-name"


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_restores_optimizer_scaler_and_rng_continuation(tmp_path, legacy):
    from gpt_lab.optim.factory import OptimizerFactory

    comp_info = ComputeContext(device=torch.device("cpu"))
    model = torch.nn.Linear(2, 2)

    def optimizer_for(model):
        return OptimizerFactory([dict(params=list(model.parameters()), opt="adamw", lr=0.01,
                                      betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)], comp_info=comp_info)

    optimizer = optimizer_for(model)
    scaler = torch.amp.GradScaler("cpu")
    inputs = torch.tensor([[0.5, -0.2]])

    def step(model, optimizer, scaler):
        optimizer.zero_grad()
        scaler.scale(model(inputs).square().mean()).backward()
        scaler.step(optimizer)
        scaler.update()

    step(model, optimizer, scaler)
    checkpoint_dir = tmp_path / "demo" / "run" / "base"
    checkpoint_state = CheckpointState(best_eval_step=7, best_eval_value=1.5)
    saved_rng = capture_rng_state()
    save_checkpoint(model, checkpoint_dir, step=7, trainer_state=TrainerState(step=7),
                    checkpoint_state=checkpoint_state, optimizer=optimizer, scaler=scaler, comp_info=comp_info)
    step_dir = checkpoint_dir / "checkpoint_step_000007"
    metadata_path = step_dir / "checkpoint_meta.json"
    assert json.loads(metadata_path.read_text()) == {"optimizer_state_layout": "full", "world_size": 1}
    if legacy:
        metadata_path.write_text(json.dumps({"mode": "ddp", "world_size": 1}))
        for name, state in (("model", model.state_dict()), ("optim", optimizer.state_dict()),
                            ("rng_state", saved_rng), ("scaler", scaler.state_dict())):
            (step_dir / f"{name}.safetensors").unlink()
            torch.save(state, step_dir / f"{name}.pt")
        (step_dir / "checkpoint_state.safetensors").unlink()
        torch.save(checkpoint_state.model_dump(), step_dir / "checkpoint_state.pkl")
    else:
        assert not list(step_dir.glob("*.pt"))
        assert not list(step_dir.glob("*.pkl"))
    originals = {path: path.read_bytes() for path in step_dir.iterdir() if path.suffix in (".pt", ".pkl")}

    expected_random = (torch.rand(3), np.random.random(3), random.random())
    loaded = load_checkpoint(checkpoint_dir, step=7, comp_info=comp_info)
    for path, original in originals.items():
        assert path.with_suffix(".safetensors").exists()
        assert path.read_bytes() == original
    assert_rng_state_consistency(saved_rng, loaded.rng_state)
    set_rng_state(loaded.rng_state)
    torch.testing.assert_close(torch.rand(3), expected_random[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.random(3), expected_random[1])
    assert random.random() == expected_random[2]
    assert loaded.checkpoint_state == checkpoint_state
    assert loaded.scaler_state == scaler.state_dict()
    assert isinstance(loaded.optimizer_state["param_groups"][0]["betas"], tuple)
    assert all(isinstance(key, int) for key in loaded.optimizer_state["state"])

    restored = torch.nn.Linear(2, 2)
    restored.load_state_dict(loaded.model_state)
    restored_optimizer = optimizer_for(restored)
    restored_optimizer.load_state_dict(load_optimizer_state(
        "demo", run_name="run", model_cachedir=tmp_path, step=7, comp_info=comp_info,
    ))
    restored_scaler = torch.amp.GradScaler("cpu")
    restored_scaler.load_state_dict(loaded.scaler_state)
    step(model, optimizer, scaler)
    step(restored, restored_optimizer, restored_scaler)
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], weight, rtol=0, atol=0)


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_manager_round_trips_training_config_and_best_step(tmp_path, legacy):
    comp_info = ComputeContext(device=torch.device("cpu"), compute_dtype=torch.float32)
    run_dir = tmp_path / "demo" / "run"
    MetaConfig(name="demo", run_name="run", dirname=run_dir, model_cfg=TransformerConfig(), base_train={})
    manager = CheckpointManager("demo", model_run="run", model_cachedir=tmp_path, comp_info=comp_info)
    trainer_config = TrainerConfig(comp_info=comp_info)
    manager.save_training_config(trainer_config)
    path = manager.source_dir / "training_config.safetensors"
    assert TrainerConfig.from_disk(path).model_dump() == trainer_config.model_dump()
    if legacy:
        path.unlink()
        torch.save(trainer_config, manager.source_dir / "training_config.pkl")
    assert manager.load_training_config() == trainer_config
    assert path.exists()
    manager.save(7, torch.nn.Linear(2, 2), checkpoint_state=CheckpointState(best_eval_step=7, best_eval_value=1.5))
    manager.save(8, torch.nn.Linear(2, 2), trainer_state=TrainerState(step=8))
    manager.mark_best_bpb(8, 1.0)
    if legacy:
        manager._ckpt_state_path.unlink()
        torch.save(manager._ckpt_state.model_dump(), manager.source_dir / "checkpoint_state.pkl")
    reloaded = CheckpointManager("demo", model_run="run", model_cachedir=tmp_path, comp_info=comp_info)
    assert reloaded._ckpt_state.best_eval_step == 8
    assert (manager.source_dir / "checkpoint_state.safetensors").exists()
    assert CheckpointManager("demo", model_run="best", model_cachedir=tmp_path, comp_info=comp_info).run_name == "run"
    model_state = manager.load_states("best", load_optim=False)
    assert model_state.trainer_state.step == 8
