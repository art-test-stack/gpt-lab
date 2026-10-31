import os
import json
import socket
import sys
import subprocess
from types import SimpleNamespace
from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpt_lab.utils.distributed import (
    ComputeContext,
    broadcast_model,
    cleanup_dist_groups,
    detect_compute_dtype,
    get_comp_info,
    get_peak_flops,
    init_dist_groups,
)
from gpt_lab.utils.schemas import TrainerConfig
from gpt_lab.utils.tensor_io import load_tensor_state, save_tensor_state


@pytest.mark.parametrize("override,cuda,capability,expected", [
    (None, False, (0, 0), torch.float32),
    (None, True, (7, 5), torch.float32),
    (None, True, (8, 0), torch.bfloat16),
    ("float16", True, (7, 5), torch.float16),
    ("float32", True, (8, 0), torch.float32),
    ("bfloat16", False, (0, 0), torch.bfloat16),
])
def test_compute_dtype_fallbacks(monkeypatch, override, cuda, capability, expected):
    monkeypatch.delenv("GPTLAB_DTYPE", raising=False)
    if override is not None:
        monkeypatch.setenv("GPTLAB_DTYPE", override)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    detect_compute_dtype.cache_clear()
    try:
        assert detect_compute_dtype()[0] == expected
        context = ComputeContext(device=torch.device("cuda" if cuda else "cpu"))
        assert context.compute_dtype == expected
    finally:
        detect_compute_dtype.cache_clear()


def test_cpu_device_name():
    context = ComputeContext(device=torch.device("cpu"))
    assert context.device_type == "cpu"
    assert context.device_name == "cpu"
    assert context.gpu_peak_flops == float("inf")


@pytest.mark.parametrize("chip, cores, peak", [
    ("Apple M2", "10", 3.6e12),
    ("Apple M2", "8", float("inf")),
    ("Apple M2 Pro", "19", float("inf")),
])
def test_mps_peak_matches_gpu_variant(monkeypatch, chip, cores, peak):
    calls = []

    def profile(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps({"SPDisplaysDataType": [{
            "spdisplays_vendor": "sppci_vendor_Apple",
            "sppci_model": chip,
            "sppci_cores": cores,
        }]}))

    monkeypatch.setattr("gpt_lab.utils.distributed.subprocess.run", profile)
    context = ComputeContext(device=torch.device("mps"))
    assert context.device_name == f"{chip} ({cores}-core GPU)"
    assert context.gpu_peak_flops == peak
    assert len(calls) == 1
    assert get_peak_flops(context.device_name.upper()) == peak


@pytest.mark.parametrize("output", ["invalid JSON", '{"SPDisplaysDataType": []}'])
def test_unavailable_mps_metadata_keeps_unknown_peak(monkeypatch, output):
    monkeypatch.setattr("gpt_lab.utils.distributed.subprocess.run",
                        lambda *_args, **_kwargs: SimpleNamespace(stdout=output))
    context = ComputeContext(device=torch.device("mps"))
    assert context.device_name is None
    assert context.gpu_peak_flops == float("inf")


def test_mps_detection_timeout_keeps_unknown_peak(monkeypatch):
    def profile(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("gpt_lab.utils.distributed.subprocess.run", profile)
    assert ComputeContext(device=torch.device("mps")).gpu_peak_flops == float("inf")


def test_single_device_and_legacy_config(monkeypatch, tmp_path):
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    cleanup_dist_groups()
    context = init_dist_groups("cpu")
    assert context.device == torch.device("cpu")
    assert (context.rank, context.dp_rank, context.world_size, context.local_rank) == (0, 0, 1, 0)
    assert context.mesh is None and context.dp_mesh is None
    assert not context.is_distributed and not dist.is_initialized()
    with pytest.raises(FrozenInstanceError):
        context.device = torch.device("mps")
    config = TrainerConfig(comp_info=context)
    path = tmp_path / "config.safetensors"
    save_tensor_state(config.model_dump(), path)
    saved = load_tensor_state(path)
    assert "comp_info" not in saved
    assert TrainerConfig.model_validate(saved).comp_info.device == context.device
    saved["dist_info"] = {"RANK": 7, "WORLD_SIZE": 8, "DEVICE": "cuda:7"}
    assert TrainerConfig.model_validate(saved).comp_info.rank == 0
    cleanup_dist_groups()


def _mesh_worker(rank, port, output_dir, adopt_existing):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    if adopt_existing:
        dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    context = init_dist_groups("cpu")
    try:
        assert context.rank == context.dp_rank == context.local_rank == rank
        assert context.world_size == 2 and context.is_distributed
        assert context.mesh.mesh_dim_names == ("dp",)
        assert context.dp_mesh.size() == 2
        assert get_comp_info("cpu") is context
        config = TrainerConfig(comp_info=context)
        assert "comp_info" not in config.model_dump()
        path = output_dir / f"config-{rank}.safetensors"
        save_tensor_state(config.model_dump(), path)
        assert TrainerConfig.model_validate(load_tensor_state(path)).comp_info.rank == rank
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(rank + 1)
        broadcast_model(model, context)
        assert model.weight.item() == 1
        total = torch.tensor(float(rank + 1))
        dist.all_reduce(total, group=context.dp_mesh.get_group())
        assert total.item() == 3
    finally:
        cleanup_dist_groups()
    assert not dist.is_initialized()


@pytest.mark.parametrize("adopt_existing", [True, False], ids=["existing-group", "fresh-group"])
def test_two_process_cpu_mesh(tmp_path, adopt_existing):
    # Use fresh workers for each case to avoid racing process-group reinitialization.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_mesh_worker, args=(port, tmp_path, adopt_existing), nprocs=2, join=True)


def _optimizer_worker(rank, port):
    from gpt_lab.optim.factory import OptimizerFactory

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    context = init_dist_groups("cpu")
    try:
        shapes = [(8,), (32, 32), (2, 3), (2, 3), (2, 3), (3, 2), (3, 2), (3, 2)]
        parameters = [torch.nn.Parameter(torch.linspace(-0.3, 0.7, torch.Size(shape).numel()).reshape(shape))
                      for shape in shapes]
        reference = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]

        def groups(params):
            return [
                dict(params=params[:2], opt="adamw", lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1),
                dict(params=params[2:5], opt="muon", lr=0.01, momentum=0.95, beta=0.9, ns_steps=2, weight_decay=0.1),
                dict(params=params[5:], opt="normuon", lr=0.01, momentum=0.95, beta=0.95, ns_steps=5, weight_decay=0.1),
            ]

        optimizer = OptimizerFactory(groups(parameters), comp_info=context)
        local = OptimizerFactory(groups(reference), comp_info=ComputeContext(device=torch.device("cpu")))
        # Exercise real optimizer kernels without testing the compiler toolchain.
        with torch.compiler.set_stance("force_eager"):
            for step in range(2):
                for index, (parameter, expected) in enumerate(zip(parameters, reference)):
                    base = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                    gradients = [torch.sin(base + index + step + worker) * (worker + 1) for worker in range(2)]
                    parameter.grad = gradients[rank].clone()
                    expected.grad = (gradients[0] + gradients[1]) / 2
                optimizer.step()
                local.step()
                for parameter, expected in zip(parameters, reference):
                    torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)
                for index in range(2):
                    actual_state = optimizer.state[parameters[index]]
                    expected_state = local.state[reference[index]]
                    assert actual_state["step"] == expected_state["step"] == step + 1
                    for key in ("exp_avg", "exp_avg_sq"):
                        expected = expected_state[key]
                        if index == 1:
                            expected = expected.chunk(2)[rank]
                        torch.testing.assert_close(actual_state[key], expected)
                owned = 2 if rank == 0 else 1
                for start in (2, 5):
                    for key in ("momentum_buffer", "second_momentum_buffer"):
                        expected = local.state[reference[start]][key][rank * 2:rank * 2 + owned]
                        torch.testing.assert_close(optimizer.state[parameters[start]][key][:owned], expected)
    finally:
        cleanup_dist_groups()


def test_two_process_gloo_optimizer_matches_averaged_local_updates():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_optimizer_worker, args=(port,), nprocs=2, join=True)
