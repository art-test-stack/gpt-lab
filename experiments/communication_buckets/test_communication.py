"""Real collective correctness; NCCL coverage is user-launched on two GPUs."""
from datetime import timedelta
import copy
import json
import runpy
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpt_lab.optim.factory import OptimizerFactory
from gpt_lab.utils.distributed import cleanup_dist_groups, init_dist_groups
from gpt_lab.utils.tensor_io import load_tensor_state, save_tensor_state


def _test_options():
    optimizer = os.environ.get("GPTLAB_COMM_TEST_OPTIMIZER", "normuon").strip().lower()
    if optimizer not in ("muon", "normuon"):
        raise ValueError("GPTLAB_COMM_TEST_OPTIMIZER must be muon or normuon")
    mode_text = os.environ.get("GPTLAB_COMM_TEST_MODES", "buckets,pipeline,backward")
    modes = tuple(mode.strip() for mode in mode_text.split(",") if mode.strip())
    allowed = {"buckets", "pipeline", "backward"}
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= allowed:
        raise ValueError("GPTLAB_COMM_TEST_MODES must be a comma list of unique buckets,pipeline,backward modes")
    smoke = os.environ.get("GPTLAB_COMM_TEST_SMOKE_ONLY", "0") == "1"
    return optimizer, modes, smoke


def _parameters(device):
    shapes = [(8,), (32, 32), *[(2, 3)] * 5, *[(3, 2)] * 3, (8,), (4,)]
    return [torch.nn.Parameter(torch.linspace(-0.3, 0.7, torch.tensor(shape).prod().item(),
                                             device=device).reshape(shape).to(
                                                 torch.bfloat16 if index == 10 else torch.float32))
            for index, shape in enumerate(shapes)]


def _groups(parameters, optimizer_name="normuon"):
    if optimizer_name == "muon":
        matrix_beta, matrix_weight_decay = .9, .28
    elif optimizer_name == "normuon":
        matrix_beta, matrix_weight_decay = .95, .1
    else:
        raise ValueError(f"Unsupported communication test optimizer {optimizer_name!r}")
    return [
        dict(params=parameters[:2] + parameters[10:], opt="adamw", lr=.01,
             betas=(.9, .95), eps=1e-8, weight_decay=.1),
        dict(params=parameters[2:7], opt=optimizer_name, lr=.02, momentum=.95,
             beta=matrix_beta, ns_steps=5, weight_decay=matrix_weight_decay),
        dict(params=parameters[7:10], opt=optimizer_name, lr=.02, momentum=.95,
             beta=matrix_beta, ns_steps=5, weight_decay=matrix_weight_decay),
    ]


def _backward(optimizer, parameters, rank, update):
    for microstep in range(2):
        optimizer.begin_backward_communication(microstep == 1)
        terms = []
        for index, parameter in enumerate(parameters[:-1]):
            if update >= 2 and microstep == 1 and index == 0:
                continue  # Its accumulated gradient exists, but no final-microstep hook fires.
            base = torch.arange(parameter.numel(), device=parameter.device).reshape(parameter.shape)
            gradient = torch.sin(base + index + update + microstep + rank) * (rank + 1)
            gradient = gradient.to(parameter.dtype)
            terms.append((parameter * gradient).sum())
            if index == 0:
                # A tied alias contributes to the same leaf and gets one hook/collective entry.
                alias = parameters[0]
                terms.append((alias * gradient).sum() * .5)
        (sum(terms) / 2).backward()
        if microstep == 0 and optimizer.communication is not None:
            assert optimizer.communication.next_bucket == 0


def _assert_state_matches(actual, expected, parameters, reference):
    errors = {"parameter_max_abs": 0.0, "gradient_max_abs": 0.0, "state_max_abs": 0.0}
    for index, (parameter, target) in enumerate(zip(parameters, reference)):
        tolerance = dict(rtol=0.03, atol=0.003) if parameter.dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(parameter, target, **tolerance, msg=f"parameter {index}")
        errors["parameter_max_abs"] = max(errors["parameter_max_abs"], (parameter.float() - target.float()).abs().max().item())
        assert (parameter.grad is None) == (target.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, target.grad, **tolerance, msg=f"gradient {index}")
            errors["gradient_max_abs"] = max(errors["gradient_max_abs"], (parameter.grad.float() - target.grad.float()).abs().max().item())
        assert set(actual.state[parameter]) == set(expected.state[target])
        for key, value in actual.state[parameter].items():
            other = expected.state[target][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, **tolerance, msg=f"state {index}.{key}")
                errors["state_max_abs"] = max(errors["state_max_abs"], (value.float() - other.float()).abs().max().item())
            else:
                assert value == other
    return errors


def _write_correctness_report(base_path, rank, record):
    if not base_path:
        return
    path = Path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rank_path = path.with_name(f"{path.stem}.rank{rank}{path.suffix or '.json'}")
    temporary = rank_path.with_suffix(rank_path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(rank_path)


def _worker(rank, port, backend, output_dir, world):
    optimizer_name, modes, smoke = _test_options()
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GPTLAB_DTYPE="bfloat16",
                      GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    try:
        context = init_dist_groups("cuda" if backend == "nccl" else "cpu")
        device = context.device
        report_modes = {}
        optimizer_execution = "force_eager" if backend == "gloo" or smoke else "default"
        with torch.compiler.set_stance(optimizer_execution):
            for mode in modes:
                mode_error = {"parameter_max_abs": 0.0, "gradient_max_abs": 0.0, "state_max_abs": 0.0}
                caps = (128,) if smoke else (128, 1024)
                updates = 4
                for cap in caps:
                    parameters, reference = _parameters(device), _parameters(device)
                    optimizer = OptimizerFactory(_groups(parameters, optimizer_name), comp_info=context)
                    baseline = OptimizerFactory(_groups(reference, optimizer_name), comp_info=context)
                    optimizer.configure_communication(mode, cap, timing=mode == "pipeline" and cap == 1024)
                    addresses = None
                    for update in range(updates):
                        _backward(baseline, reference, rank, update)
                        baseline.step()
                        _backward(optimizer, parameters, rank, update)
                        manager = optimizer.communication
                        if mode == "backward" and manager.prepared:
                            assert manager.next_bucket > 0  # Real reductions launched by leaf hooks.
                        if update == 3:
                            optimizer.finish_gradient_communication()
                            probe = torch.ones(1, device=device)
                            dist.all_reduce(probe)
                            torch.testing.assert_close(probe, torch.full_like(probe, world))
                        optimizer.step()
                        errors = _assert_state_matches(optimizer, baseline, parameters, reference)
                        for key, value in errors.items():
                            mode_error[key] = max(mode_error[key], value)
                        assert parameters[-1].grad is None
                        assert not optimizer.state[parameters[-1]]
                        assert [id(p) for g in optimizer.param_groups for p in g["params"]] == [
                            id(p) for g in _groups(parameters, optimizer_name) for p in g["params"]]
                        current = [(bucket.packed.data_ptr(), bucket.received.data_ptr()) for bucket in manager.buckets]
                        current.extend(tuple(info[key].data_ptr() for key in ("grad_chunk", "stacked_grads", "updated_params"))
                                       for info in manager.infos if "grad_chunk" in info)
                        if addresses is not None:
                            assert current == addresses
                        addresses = current
                        for bucket in manager.buckets:
                            owner = rank if bucket.kind == "reduce_scatter" else 0
                            count = bucket.lengths[owner]
                            torch.testing.assert_close(bucket.received[count:], torch.zeros_like(bucket.received[count:]))
                        # Independently average original accumulated gradients for consumer checks.
                        for index, parameter in enumerate(parameters[:-1]):
                            base = torch.arange(parameter.numel(), device=device).reshape(parameter.shape)
                            workers = []
                            for worker in range(world):
                                micros = [
                                    (torch.sin(base + index + update + micro + worker) * (worker + 1)).to(parameter.dtype)
                                    for micro in range(2)
                                ]
                                if update >= 2 and index == 0:
                                    micros[1].zero_()
                                workers.append((micros[0] / 2 + micros[1] / 2) * (1.5 if index == 0 else 1))
                            averaged = sum(workers) / world
                            if index < 2 or index == 10:
                                info = manager.infos[0]["param_infos"][parameter]
                                expected = averaged if info["is_small"] else averaged.chunk(world)[rank]
                                if parameter.dtype != torch.bfloat16:
                                    torch.testing.assert_close(info["grad_slice"], expected, rtol=1e-5, atol=1e-6)
                            else:
                                group_index, group_start = (1, 2) if index < 7 else (2, 7)
                                chunk = manager.infos[group_index]["chunk_size"]
                                if (index - group_start) // chunk == rank:
                                    slot = (index - group_start) % chunk
                                    if optimizer_name == "normuon":
                                        torch.testing.assert_close(manager.infos[group_index]["grad_chunk"][slot], averaged)
                        assert manager.stats()["padding_bytes"] >= 0
                        assert manager.stats()["control_collective_count"] == 1
                        assert manager.stats()["control_payload_bytes"] == 4
                        assert manager.stats()["validation_collectives_total"] == manager.step_count
                        if manager.timing:
                            assert manager.stats()["per_update"][-1]["host_launch_ms"] >= 0
                            assert manager.stats()["per_update"][-1]["validation_ms"] >= 0
                        if update == 1:
                            if smoke:
                                saved_state = copy.deepcopy(optimizer.state_dict())
                            else:
                                directory = Path(output_dir)
                                directory.mkdir(parents=True, exist_ok=True)
                                path = directory / f"{backend}-{mode}-{cap}-rank{rank}.safetensors"
                                save_tensor_state(optimizer.state_dict(), path)
                            optimizer.communication.close()
                            restored = OptimizerFactory(_groups(parameters, optimizer_name), comp_info=context)
                            restored.configure_communication(mode, cap if smoke else 512)
                            restored.load_state_dict(
                                saved_state if smoke else load_tensor_state(path, device=device)
                            )
                            optimizer = restored
                            addresses = None
                        optimizer.zero_grad(set_to_none=True)
                        baseline.zero_grad(set_to_none=True)
                    optimizer.communication.close()
                report_modes[mode] = mode_error

            if not smoke:
                # Whole Muon matrices can exceed a packed-byte target and must stay whole.
                if "pipeline" in modes:
                    parameters = [torch.nn.Parameter(torch.zeros(2, 3, device=device)) for _ in range(5)]
                    optimizer = OptimizerFactory([dict(params=parameters, opt=optimizer_name, lr=.02,
                        momentum=.95, beta=.9 if optimizer_name == "muon" else .95, ns_steps=5,
                        weight_decay=.28 if optimizer_name == "muon" else .1)], comp_info=context)
                    optimizer.configure_communication("pipeline", 32)
                    for parameter in parameters:
                        parameter.grad = torch.ones_like(parameter)
                    optimizer.step()
                    assert all(bucket["oversized"] for bucket in optimizer.communication.stats()["buckets"])
                    assert optimizer.communication.stats()["padding_bytes"] == 5 * 24 * (world - 1)
                    optimizer.communication.close()

                # A rank-local missing gradient after warmup is rejected everywhere, without
                # stranded data collectives or partially updated parameters/momenta.
                for mode in modes:
                    parameters = [torch.nn.Parameter(torch.zeros(2, 3, device=device)) for _ in range(3)]
                    optimizer = OptimizerFactory([dict(params=parameters, opt=optimizer_name, lr=.02,
                        momentum=.95, beta=.9 if optimizer_name == "muon" else .95, ns_steps=5,
                        weight_decay=.28 if optimizer_name == "muon" else .1)], comp_info=context)
                    optimizer.configure_communication(mode, 48)
                    optimizer.begin_backward_communication(True)
                    sum((p * (i + 1)).sum() for i, p in enumerate(parameters)).backward()
                    optimizer.step()
                    before = [p.detach().clone() for p in parameters]
                    state = copy.deepcopy(optimizer.state_dict())
                    optimizer.zero_grad(set_to_none=True)
                    optimizer.begin_backward_communication(True)
                    sum((p * (i + 2)).sum() for i, p in enumerate(parameters) if not (rank == 0 and i == 1)).backward()
                    with pytest.raises(ValueError, match="presence/layout changed"):
                        optimizer.step()
                    for p, value in zip(parameters, before):
                        torch.testing.assert_close(p, value, rtol=0, atol=0)
                    current = optimizer.state_dict()
                    assert current["param_groups"] == state["param_groups"]
                    for key, value in state["state"][0].items():
                        torch.testing.assert_close(current["state"][0][key], value, rtol=0, atol=0)
                    optimizer.communication.close()

                # All ranks reject unsupported initial masks before any gradient data collective.
                if "backward" in modes:
                    for rank_dependent in (False, True):
                        parameters = [torch.nn.Parameter(torch.zeros(2, 3, device=device)) for _ in range(2)]
                        optimizer = OptimizerFactory([dict(params=parameters, opt=optimizer_name, lr=.02,
                            momentum=.95, beta=.9 if optimizer_name == "muon" else .95, ns_steps=5,
                            weight_decay=.28 if optimizer_name == "muon" else .1)], comp_info=context)
                        optimizer.configure_communication("backward", 128)
                        parameters[0].grad = torch.ones_like(parameters[0])
                        if rank_dependent and rank == 1:
                            parameters[1].grad = torch.ones_like(parameters[1])
                        with pytest.raises(ValueError, match="missing gradients|presence"):
                            optimizer.step()
                        assert not optimizer.state
                        optimizer.communication.close()
        report_path = os.environ.get("GPTLAB_COMM_TEST_REPORT")
        if report_path:
            # A failed rank never writes an `ok` report; this collective also
            # keeps successful ranks from publishing before peers finish.
            completed = torch.ones((), dtype=torch.int32, device=device)
            dist.all_reduce(completed, op=dist.ReduceOp.MIN)
            _write_correctness_report(report_path, rank, {
                "backend": backend,
                "optimizer": optimizer_name,
                "optimizer_execution": optimizer_execution,
                "world_size": world,
                "updates_per_mode_and_cap": 4,
                "modes": list(modes),
                "ok": bool(completed.item()),
                "max_abs_errors": report_modes,
            })
    finally:
        cleanup_dist_groups()


def _run(backend, tmp_path):
    world = int(os.environ.get("GPTLAB_COMM_TEST_WORLD_SIZE", "2")) if backend == "nccl" else 2
    if world not in (2, 4, 8):
        raise ValueError("NCCL correctness checks support 2, 4, or 8 ranks")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(port, backend, str(tmp_path), world), nprocs=world, join=True)


def test_gloo_buckets_accumulation_padding_ties_and_resume(tmp_path):
    _run("gloo", tmp_path)


@pytest.mark.gpu
@pytest.mark.skipif(not dist.is_nccl_available() or torch.cuda.device_count() < int(os.environ.get("GPTLAB_COMM_TEST_WORLD_SIZE", "2")),
                    reason="Requires the requested 2/4/8 user-provided CUDA GPUs with NCCL")
def test_nccl_buckets_accumulation_padding_ties_and_resume(tmp_path):
    _run("nccl", tmp_path)


@pytest.mark.parametrize("optimizer", ["muon", "normuon"])
def test_diagnostic_cli_preserves_native_overrides_and_scheduler_horizon(tmp_path, optimizer):
    runner = runpy.run_path(str(Path(__file__).resolve().parents[2] / "experiments/communication_buckets/communication.py"))
    from gpt_lab.cli.parser import build_parser
    args, forwarded = runner["parse_args"]([
        "--optimizer", optimizer, "--communication", "backward", "--bucket-mib", "32",
        "--device-batch-size", "4", "--random-seed", "7", "--num-steps", "150",
        "--eval-bpb-every", "10", "--save-every", "5",
    ])
    resolved = build_parser().parse_args([
        "train", "auto", *runner["default_cli_args"](tmp_path),
        *forwarded, *runner["invariant_cli_args"](args.optimizer),
    ])
    assert args.communication == "backward" and args.bucket_mib == 32
    assert resolved.device_batch_size == 16 and resolved.random_seed == 42
    assert resolved.num_steps == 2317 and resolved.stop_after_step == 150
    assert resolved.act_func == "swiglu" and resolved.attn_impl == "fa2"
    assert resolved.eval_bpb_every == resolved.eval_core_every == resolved.sample_every == resolved.save_every == 0
    assert resolved.optim_config_path.endswith("configs/optim.yaml" if optimizer == "muon" else "configs/normuon.yaml")


def test_diagnostic_failure_preserves_error_without_new_collectives(tmp_path, monkeypatch):
    from gpt_lab.train.trainer import Trainer
    from gpt_lab.model.checkpoint import CheckpointManager
    runner = runpy.run_path(str(Path(__file__).resolve().parents[2] / "experiments/communication_buckets/communication.py"))
    for name in ("__init__", "train", "train_step", "optimizer_step", "prepare_batch", "save_checkpoint"):
        monkeypatch.setattr(Trainer, name, getattr(Trainer, name))  # Restore runner-installed wrappers after this test.
    monkeypatch.setattr(CheckpointManager, "save_training_config", CheckpointManager.save_training_config)

    def failed_train(self):
        raise RuntimeError("original rank failure")

    monkeypatch.setattr(Trainer, "train", failed_train)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_gather_object", lambda *_: pytest.fail("summary collective after failure"))
    args = SimpleNamespace(communication="baseline", profile=None, warmup_updates=25)
    runner["install_runner"](args, tmp_path, 42, {})
    assert Trainer.save_checkpoint(None) is None
    assert CheckpointManager.save_training_config(None, None) is None
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(comp_info=SimpleNamespace(rank=0))
    trainer.state = SimpleNamespace(completed_steps=3)
    with pytest.raises(RuntimeError, match="original rank failure"):
        trainer.train()
    import json
    record = json.loads((tmp_path / "communication-error.rank0.json").read_text())
    assert record == dict(status="failed", rank=0, error_type="RuntimeError", completed_steps=3)
