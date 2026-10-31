"""Real CUDA/NCCL gate. Run only through a user-launched eight-H100 job."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def fingerprint(value):
    import torch
    import numpy as np
    sha = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            sha.update(str((item.shape, item.dtype)).encode())
            sha.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            sha.update(str((item.shape, item.dtype)).encode())
            sha.update(item.tobytes())
        elif isinstance(item, dict):
            for key, part in item.items():
                sha.update(str(key).encode())
                add(part)
        elif isinstance(item, (list, tuple)):
            for part in item:
                add(part)
        else:
            sha.update(repr(item).encode())
    add(value)
    return sha.hexdigest()


def assert_tree(actual, expected):
    import torch
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=1e-5, atol=1e-6)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_tree(a, b)
    else:
        assert actual == expected


def owner_checks(context, output):
    import torch
    import torch.distributed as dist
    from gpt_lab.optim.factory import OptimizerFactory
    from gpt_lab.optim.kernels.normuon import _orthogonalize
    from gpt_lab.model.checkpoint import save_checkpoint, load_checkpoint, set_rng_state
    from gpt_lab.utils.schemas import TrainerState, CheckpointState, OptimizerConfig
    from gpt_lab.utils.distributed import ComputeContext
    orthogonalize = torch.compile(_orthogonalize, fullgraph=True, dynamic=False)
    for shape in ((16, 32), (32, 16)):
        for rank in (0, 1, 16):
            matrix = torch.zeros(shape, device=context.device)
            matrix[:rank, :rank] = torch.eye(rank, device=context.device)
            if rank == 1:
                matrix[1, 1] = 1e-7  # Near-rank-one guard, rather than exact rank one only.
            result = orthogonalize(matrix, 5, True)
            assert torch.isfinite(result).all()
            singular = torch.linalg.svdvals(result.float())
            torch.testing.assert_close(singular[:rank], torch.ones(rank, device=context.device), rtol=0, atol=.06)
            if rank == 0:
                assert not result.count_nonzero()
    records = []
    for opt, spectral in (("muon", False), ("normuon", False), ("normuon", True)):
        module = torch.nn.ParameterList([torch.nn.Parameter(torch.arange(512, device=context.device).reshape(16, 32).float()/1000+i/10) for i in range(17)])
        reference = torch.nn.ParameterList([torch.nn.Parameter(p.detach().clone()) for p in module])
        profile = "muon" if opt == "muon" else ("b_spectral" if spectral else "a_normuon")
        selected = OptimizerConfig.from_file(f"experiments/optimizer_comparison/configs/{profile}.yaml").group_configs()["transformer"]
        def group(parameters):
            return dict(params=list(parameters), **selected)
        optimizer = OptimizerFactory([group(module)], comp_info=context)
        local = OptimizerFactory([group(reference)], comp_info=ComputeContext(context.device))
        checkpoint = output/f"owner-{opt}-{spectral}"
        def update(step, compare_local=True):
            optimizer.param_groups[0]["lr"] = .02/(step+1)
            local.param_groups[0]["lr"] = .02/(step+1)
            for i, (p, q) in enumerate(zip(module, reference)):
                base = torch.arange(512, device=context.device).reshape(16, 32)
                p.grad = ((base+i+step) % 7 - 3 + context.rank).float()/32
                q.grad = torch.stack([((base+i+step) % 7 - 3 + r).float()/32 for r in range(8)]).mean(0)
            optimizer.step()
            if compare_local:
                local.step()
                for p, q in zip(module, reference):
                    torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
                owned = min(3, max(0, 17-context.rank*3))
                for key, value in optimizer.state[module[0]].items():
                    if owned:
                        torch.testing.assert_close(value[:owned], local.state[reference[0]][key][context.rank*3:context.rank*3+owned], rtol=1e-5, atol=1e-6)
                    assert not value[owned:].count_nonzero()
        update(0)
        save_checkpoint(model=module, checkpoint_dir=checkpoint, step=0, optimizer=optimizer,
            trainer_state=TrainerState(step=0, completed_steps=1), checkpoint_state=CheckpointState(),
            optimizer_state_layout="sharded", comp_info=context)
        update(1)
        expected_weights, expected_state = cpu_copy(module.state_dict()), cpu_copy(optimizer.state_dict())
        loaded = load_checkpoint(checkpoint, step=0, comp_info=context, load_scaler=False, optimizer_state_layout="sharded")
        assert loaded.trainer_state.completed_steps == 1
        module.load_state_dict(loaded.model_state)
        optimizer.load_state_dict(loaded.optimizer_state)
        set_rng_state(loaded.rng_state)
        update(1, compare_local=False)
        assert_tree(module.state_dict(), expected_weights)
        assert_tree(optimizer.state_dict(), expected_state)
        hashes = [None]*8
        dist.all_gather_object(hashes, fingerprint(module.state_dict()))
        assert len(set(hashes)) == 1
        records.append(dict(optimizer=opt, spectral=spectral, owner_state_and_disk_resume="passed"))
    return records


def model_check(context, output, settings):
    import torch
    import torch.distributed as dist
    from torch._inductor.select_algorithm import extern_kernels
    from gpt_lab.model.checkpoint import save_checkpoint, load_checkpoint, set_rng_state
    from gpt_lab.model.flash_attn import require_flash_attn3
    from gpt_lab.train.trainer import Trainer
    from gpt_lab.utils.schemas import TrainerConfig, OptimizerConfig, TrainerState, CheckpointState
    from gpt_lab.tokenizer import Tokenizer
    from experiments.optimizer_comparison.recipe import model_and_horizon, storage_summary
    if settings["attn_impl"] == "fa3":
        require_flash_attn3()
    tokenizer = Tokenizer.from_pretrained(settings["tokenizer_model"])
    torch.manual_seed(settings["random_seed"])
    model, horizon = model_and_horizon(context.device, settings, context, tokenizer.vocab_size)
    model._enforce_precision(torch.bfloat16)
    model.init_weights()
    parameter_storage = storage_summary(model)
    config = TrainerConfig(comp_info=context, **horizon["base_training"], fp8_ffn_forward=True,
        lr_warmup_steps=settings["warmup_steps"], lr_warmdown_ratio=settings["warmdown_ratio"],
        final_lr_ratio=settings["final_lr_frac"], eval_bpb_every=0, eval_core_every=0,
        optim_config=OptimizerConfig.from_file(settings["optim_config_path"]))
    trainer = Trainer(model, tokenizer, None, None, config=config,
        checkpoint_manager=SimpleNamespace(source_dir=output/"model"))
    trainer.compiled_model = torch.compile(model, dynamic=False)
    trainer.compiled_training_loss = torch.compile(model.training_loss, dynamic=False)
    original_step = trainer.optimizer_step
    def checked_step(should_log):
        assert any(p.grad is not None for p in model.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        original_step(should_log)
    trainer.optimizer_step = checked_step
    records, originals = [], {}
    # Observe actual Inductor extern GEMM operands, not only FX intentions.
    for name in ("_scaled_mm", "mm", "addmm"):
        if not hasattr(extern_kernels, name):
            continue
        originals[name] = getattr(extern_kernels, name)
        def wrapped(*args, _name=name, _original=originals[name], **kwargs):
            tensors = [a for a in args if isinstance(a, torch.Tensor) and a.ndim == 2]
            if len(tensors) >= 2:
                records.append(dict(operator=_name, shapes=[list(t.shape) for t in tensors[:2]],
                                    dtypes=[str(t.dtype) for t in tensors[:2]]))
            return _original(*args, **kwargs)
        setattr(extern_kernels, name, wrapped)
    losses = []
    try:
        generator = torch.Generator(device=context.device).manual_seed(100+context.rank)
        x = torch.randint(tokenizer.vocab_size, (config.device_batch_size, model.config.max_context), generator=generator, device=context.device)
        y = torch.randint(tokenizer.vocab_size, (config.device_batch_size, model.config.max_context), generator=generator, device=context.device)
        def batches():
            while True:
                yield x, y, None
        iterator = batches()
        batch = trainer.prepare_batch(next(iterator))
        torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            trainer.state.step = step
            batch, result = trainer.train_step(iterator, batch, False)
            losses.append(float(result["loss"]))
            assert torch.isfinite(result["loss"])
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert all(torch.isfinite(v).all() for state in trainer.optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor))
        save_checkpoint(model=model, checkpoint_dir=output/"full-model-resume", step=1,
            optimizer=trainer.optimizer, trainer_state=TrainerState(step=1, completed_steps=2),
            checkpoint_state=CheckpointState(), optimizer_state_layout="sharded", comp_info=context)
        records.clear()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            trainer.state.step = 2
            batch, result = trainer.train_step(iterator, batch, False)
            assert torch.isfinite(result["loss"])
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(output/f"trace_rank_{context.rank}.json"))
        forward = [r for r in records if r["operator"] == "_scaled_mm"]
        expected = 2 * model.config.n_layers * config.n_acc_steps
        assert len(forward) == expected, f"Native compiled FP8 dispatch not proven: {len(forward)} / {expected}; inspect trace"
        assert all(r["dtypes"] == ["torch.float8_e4m3fn"]*2 for r in forward)
        backward = [r for r in records if r["operator"] != "_scaled_mm" and any(width in shape for shape in r["shapes"] for width in (model.blocks[0].ffn.w_1.out_features, model.blocks[0].ffn.w_2.in_features))]
        assert len(backward) == expected * 2, f"BF16 FFN backward dispatch not proven: {len(backward)} / {expected * 2}"
        assert all(r["dtypes"] == ["torch.bfloat16"]*2 for r in backward)
        events = json.loads((output/f"trace_rank_{context.rank}.json").read_text())["traceEvents"]
        flash = [e["name"] for e in events if e.get("cat") == "kernel" and "flash" in e.get("name", "").lower()]
        if settings["attn_impl"] in ("fa2", "fa3"):
            assert flash, "Compiled attention CUDA kernels not identified; inspect trace before training"
        dispatch_records = copy.deepcopy(records)
        expected_weights = cpu_copy(model.state_dict())
        expected_optimizer = cpu_copy(trainer.optimizer.state_dict())
        expected_loss = result["loss"].detach().clone()
        loaded = load_checkpoint(output/"full-model-resume", step=1, comp_info=context,
                                 load_scaler=False, optimizer_state_layout="sharded")
        assert loaded.trainer_state.completed_steps == 2
        model.load_state_dict(loaded.model_state)
        trainer.optimizer.load_state_dict(loaded.optimizer_state)
        assert storage_summary(model) == parameter_storage
        set_rng_state(loaded.rng_state)
        batch, resumed = trainer.train_step(iterator, batch, False)
        torch.testing.assert_close(resumed["loss"], expected_loss, rtol=1e-5, atol=1e-6)
        assert_tree(model.state_dict(), expected_weights)
        assert_tree(trainer.optimizer.state_dict(), expected_optimizer)
        del expected_weights, expected_optimizer, loaded
        # Eval must use the same parameters with FP8 conversion disabled.
        model.eval()
        with torch.no_grad(), trainer.val_context(model):
            assert all("Float8" not in type(m).__name__ for m in model.modules())
            evaluation_loss = model.training_loss(x[:1], y[:1])
            assert torch.isfinite(evaluation_loss)
        return dict(horizon=horizon, parameter_storage=parameter_storage, losses=losses, full_model_fp8_disk_resume="passed", evaluation_loss=float(evaluation_loss),
                    actual_gemm_dispatch=dispatch_records, fa3_kernel_names=sorted(set(flash)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    note="Synthetic requested-microbatch memory/dispatch check; not a learning-quality or throughput result")
    finally:
        for name, original in originals.items():
            setattr(extern_kernels, name, original)


def main():
    import torch
    import torch.distributed as dist
    from gpt_lab.utils.distributed import init_dist_groups, cleanup_dist_groups
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    context = init_dist_groups("cuda", random_seed=settings["random_seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        assert context.world_size == 8 and dist.get_backend() == "nccl"
        names = [torch.cuda.get_device_name(i) for i in range(8)]
        assert all("H100" in name for name in names) and torch.cuda.get_device_capability() == (9, 0)
        print("Allocated GPUs:", names, flush=True)
        owners = owner_checks(context, args.output)
        torch.manual_seed(42)
        report = model_check(context, args.output, settings)
        report.update(status="passed", owner_checks=owners, gpu_models=names, rank=context.rank)
        (args.output/f"rank_{context.rank}.json").write_text(json.dumps(report, indent=2)+"\n")
        dist.barrier()
        if context.rank == 0:
            reports = [json.loads((args.output/f"rank_{rank}.json").read_text()) for rank in range(8)]
            assert all(r["status"] == "passed" for r in reports)
            (args.output/"passed.json").write_text(json.dumps(dict(status="passed", ranks=8, horizon=report["horizon"]), indent=2)+"\n")
    finally:
        cleanup_dist_groups()


if __name__ == "__main__":
    main()
