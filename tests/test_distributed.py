import os
import socket
import sys
from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpt_lab.utils.distributed import (
    DistributedContext,
    broadcast_model,
    cleanup_dist_groups,
    get_dist_info,
    init_dist_groups,
)
from gpt_lab.utils.schemas import TrainerConfig
from gpt_lab.utils.tensor_io import load_tensor_state, save_tensor_state


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
    config = TrainerConfig(dist_info=context)
    path = tmp_path / "config.safetensors"
    save_tensor_state(config.model_dump(), path)
    saved = load_tensor_state(path)
    assert "dist_info" not in saved
    assert TrainerConfig.model_validate(saved).dist_info.device == context.device
    saved["dist_info"] = {"RANK": 7, "WORLD_SIZE": 8, "DEVICE": "cuda:7"}
    assert TrainerConfig.model_validate(saved).dist_info.rank == 0
    cleanup_dist_groups()


def _mesh_worker(rank, port, output_dir):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    # Also cover adopting an existing process group and creating a fresh one.
    for cycle in range(2):
        if cycle == 0:
            dist.init_process_group("gloo", timeout=timedelta(seconds=30))
        context = init_dist_groups("cpu")
        try:
            assert context.rank == context.dp_rank == context.local_rank == rank
            assert context.world_size == 2 and context.is_distributed
            assert context.mesh.mesh_dim_names == ("dp",)
            assert context.dp_mesh.size() == 2
            assert get_dist_info("cpu") is context
            config = TrainerConfig(dist_info=context)
            assert "dist_info" not in config.model_dump()
            path = output_dir / f"config-{rank}-{cycle}.safetensors"
            save_tensor_state(config.model_dump(), path)
            assert TrainerConfig.model_validate(load_tensor_state(path)).dist_info.rank == rank
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


def test_two_process_cpu_mesh(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_mesh_worker, args=(port, tmp_path), nprocs=2, join=True)
