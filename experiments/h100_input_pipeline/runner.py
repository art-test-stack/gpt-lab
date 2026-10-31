"""One H100 worker: real production updates, matched data, isolated input variants."""
import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

from recipe import (SHA, train_args, batch_config, BATCH_SIZES, ORDER, TOKENS, cpu_copy, tensor_hash,
                    tensor_dtypes, write, hardware_report, timing, restore_training_state)
from cache import capture, Replay, cache_bytes, digest_batch, verify
from tracing import FetchRange, ranges, summarize


def restore(trainer, weights, optimizer, state, rng, factory):
    from gpt_lab.model.checkpoint import set_rng_state
    iterator, batch = restore_training_state(trainer, weights, optimizer, state, rng, factory)
    # Input preparation must not shift the compute RNG relative to resident replay.
    set_rng_state(rng)
    return iterator, batch


def run_updates(trainer, iterator, batch, start, count, profile=False):
    """The production step includes its original boundary synchronizations."""
    import torch
    seconds, losses, schedules = [], [], []
    total_dt = []
    for step in range(start, start+count):
        trainer.state.step = step
        marker = torch.profiler.record_function(f"diagnostic/update/{step}") if profile else nullcontext()
        before = time.perf_counter()
        with marker:
            batch, metrics = trainer.train_step(iterator, batch, False)
            trainer.update_metrics(metrics, total_dt)
        seconds.append(time.perf_counter()-before)
        assert metrics["tokens"] == trainer.config.total_batch_size
        losses.append(float(metrics["loss"]))
        schedules.append({key:metrics[key] for key in ("lrm", "muon_momentum", "weight_decay")})
        trainer.state.completed_steps = step+1
    return batch, seconds, losses, schedules


def diagnostic(trainer, factory, output, warmup, measured, preflight):
    import torch
    from gpt_lab.model.checkpoint import capture_rng_state
    from gpt_lab.data.loader import DistDataLoader
    cfg = trainer.model.config
    assert (cfg.d_model,cfg.d_ffn,cfg.max_context,cfg.n_layers,cfg.act_func,cfg.attn_impl) == (768,3072,2048,12,"swiglu","fa3")
    resolved = batch_config(trainer.config,cfg.max_context)
    print("Resolved production batch: " + json.dumps(resolved),flush=True)
    assert resolved == preflight["resolved_batch"]
    assert trainer.config.device_batch_size in BATCH_SIZES
    assert (trainer.config.n_steps,trainer.config.lr_warmup_steps,trainer.config.total_batch_size) == (2317,300,TOKENS)
    assert trainer.dtype == torch.bfloat16 and trainer.scaler is None
    assert not trainer.config.fp8_ffn_forward and trainer.config.comp_info.world_size == 1
    model = trainer.model
    weights = cpu_copy(model.state_dict())
    optim = cpu_copy(trainer.optimizer.state_dict())
    state = copy.deepcopy(trainer.state)
    rng = capture_rng_state()
    parameter_dtypes = {name:str(t.dtype) for name,t in model.named_parameters()}
    identities = {name:id(t) for name,t in model.named_parameters()}
    count = resolved["n_acc_steps"]*(warmup+measured)+1  # include the production step's final prefetch
    loader = factory()
    if type(loader) is not DistDataLoader:
        raise RuntimeError("Only the default production DistDataLoader is supported")
    start = time.perf_counter()
    packed, batches, hashes = capture(loader, count)
    capture_seconds = time.perf_counter()-start
    order_hash = hashlib.sha256("".join(hashes).encode()).hexdigest()
    assert order_hash == preflight["data_order_sha256"], "CPU preflight and GPU worker data differ"
    replay = Replay(packed, batches, trainer.device)
    # Verify every replay and every independently owned resident batch before timing.
    for expected in batches:
        actual = next(replay)
        for a,b in zip(actual[:2],expected[:2]):
            verify(a,b)
    torch.cuda.synchronize()
    # Check the actual production order too, in a separate untimed fresh iterator.
    validation_loader = factory()
    for expected_hash in hashes:
        assert digest_batch(next(validation_loader)) == expected_hash
    del validation_loader, loader
    gc.collect()
    overhead = cache_bytes(packed,batches)
    overhead["replay_staging_cpu_bytes"] = replay.cpu.numel()*replay.cpu.element_size()
    overhead["replay_staging_gpu_bytes"] = sum(t.numel()*t.element_size() for t in (replay.gpu,replay.inputs,replay.targets))
    report = dict(production_sha=SHA, harness_sha=os.environ["INPUT_HARNESS_SHA"],
                  resolved_batch=resolved, initial_optimizer_groups=optim["param_groups"],
                  model=cfg.model_dump(), trainer=trainer.config.model_dump(mode="json"),
                  torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                  gpu=torch.cuda.get_device_name(), gpu_properties=str(torch.cuda.get_device_properties(0)),
                  parameter_storage_dtypes=parameter_dtypes, optimizer_initial_dtypes=tensor_dtypes(optim),
                  initial_model_sha256=tensor_hash(weights.items()), data_order_sha256=order_hash,
                  first_row_sha256=digest_batch((batches[0][0][0],batches[0][1][0])),
                  cached_microbatches=count, cache_memory=overhead, cache_capture_seconds=capture_seconds,
                  warmup_updates=warmup, measured_updates=measured, cpu_preflight=preflight,
                  order=ORDER, legs={})
    write(output/"report.json",report)
    # Match production compilation policy and keep it across all legs.
    model.train()
    trainer.compiled_model = torch.compile(model, dynamic=False)
    trainer.compiled_training_loss = torch.compile(model.training_loss, dynamic=False)

    def reset(kind):
        if kind == "production":
            selected_factory = factory
        else:
            replay.index = 0
            replay.gpu_resident = kind == "gpu-cache"
            selected_factory = lambda:replay
        iterator,batch = restore(trainer,weights,optim,state,rng,selected_factory)
        assert identities == {name:id(t) for name,t in model.named_parameters()}
        assert parameter_dtypes == {name:str(t.dtype) for name,t in model.named_parameters()}
        torch.cuda.synchronize()
        return iterator,batch

    # Production disables cyclic GC after its first update; keep all legs equal.
    gc.disable()
    for name in ORDER:
        kind = "production" if name.startswith("production") else name
        setup_start = time.perf_counter()
        iterator,batch = reset(kind)
        batch,warm_times,_,_ = run_updates(trainer,iterator,batch,0,warmup)
        startup = time.perf_counter()-setup_start
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        batch,seconds,losses,schedules = run_updates(trainer,iterator,batch,warmup,measured)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-before
        result = dict(timing=timing(seconds,TOKENS), measured_wall_seconds=elapsed,
                      global_tokens_per_second=measured*TOKENS/elapsed,
                      startup_and_warmup_seconds=startup, warmup_ms=[t*1000 for t in warm_times],
                      losses=losses, schedules=schedules,
                      final_prefetch_sha256=digest_batch((*batch[0],batch[1])),
                      last_consumed_loader_state=trainer.state.train_loader_state.model_dump(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      final_model_sha256=tensor_hash(model.state_dict().items()),
                      optimizer_final_dtypes=tensor_dtypes(trainer.optimizer.state_dict()))
        report["legs"][name] = result
        write(output/"report.json",report)
        print(f"{name}: {result['global_tokens_per_second']:.0f} tokens/s; median {result['timing']['median_ms']:.3f} ms",flush=True)
        del iterator,batch
    first,last = (report["legs"][name] for name in (ORDER[0],ORDER[-1]))
    for result in report["legs"].values():
        for key in ("schedules","final_prefetch_sha256","last_consumed_loader_state","optimizer_final_dtypes"):
            assert first[key] == result[key], f"Unmatched {key}"
    report["controls"] = dict(median_drift_ms=abs(first["timing"]["median_ms"]-last["timing"]["median_ms"]),
                              throughput_drift_percent=100*(last["global_tokens_per_second"]/first["global_tokens_per_second"]-1),
                              max_loss_difference=max(abs(a-b) for a,b in zip(first["losses"],last["losses"])))
    report["comparisons"] = {}
    for name in ("cpu-cache","gpu-cache"):
        result = report["legs"][name]
        saving = min(first["timing"]["median_ms"],last["timing"]["median_ms"])-result["timing"]["median_ms"]
        spread = max(item["timing"]["p95_ms"]-item["timing"]["p05_ms"] for item in (first,last,result))
        report["comparisons"][name] = dict(median_saving_vs_faster_control_ms=saving,
            saving_exceeds_control_drift_and_within_leg_width=saving>max(report["controls"]["median_drift_ms"],spread),
            max_loss_difference=max(abs(a-b) for a,b in zip(first["losses"],result["losses"])))
    write(output/"report.json",report)
    # Separate short trace after all uninstrumented measurements, using real production input.
    iterator,batch = reset("production")
    with ranges(trainer,iterator):
        traced = FetchRange(iterator)
        batch,_,_,_ = run_updates(trainer,traced,batch,0,warmup)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as trace:
            run_updates(trainer,traced,batch,warmup,2,profile=True)
        torch.cuda.synchronize()
    trace.export_chrome_trace(str(output/"trace.json"))
    report["profile"] = summarize(output/"trace.json",expected_microsteps=resolved["n_acc_steps"])
    report["complete"] = True
    report["interpretation"] = "Production versus CPU cache removes reads/tokenization/packing; CPU versus GPU cache removes staging/transfer. Differences are interventions, not additive stage costs. Require repeat-control stability and trace evidence before attribution. All caches remain allocated in every timed leg; report their overhead separately."
    write(output/"report.json",report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-batch-size",type=int,choices=BATCH_SIZES,required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--measured", type=int, default=50)
    args = parser.parse_args()
    if args.warmup<25 or args.measured<50 or args.warmup+args.measured>2317:
        parser.error("Need >=25 warmup and >=50 measured updates within horizon 2317")
    assert subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip() == SHA
    import torch
    if torch.cuda.device_count()!=1 or "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("Requires exactly one H100")
    from gpt_lab.model.flash_attn import require_flash_attn3
    require_flash_attn3()
    from gpt_lab.data import loader
    from gpt_lab.train.trainer import Trainer
    from gpt_lab.model.checkpoint import CheckpointManager
    build, factories = loader.build_dataloader, []
    def remember(**kwargs):
        if kwargs["split"] == "train":
            factories.append(lambda:build(**kwargs))
        return build(**kwargs)
    loader.build_dataloader = remember
    Trainer.train = lambda self:diagnostic(self,factories[0],args.output,args.warmup,args.measured,
                                          json.loads(args.preflight.read_text())["batches"][str(args.device_batch_size)])
    Trainer.save_checkpoint = lambda self:None
    CheckpointManager.save_training_config = lambda self,config:None
    # Auto-configuration normally autosaves metadata; keep it in report.json only.
    from gpt_lab.utils.schemas import MetaConfig
    original_post_init = MetaConfig.model_post_init
    def metadata_only(self, context):
        self.autosave = False
        original_post_init(self, context)
    MetaConfig.model_post_init = metadata_only
    from gpt_lab.cli import main as cli
    write(args.output/"harness_manifest.json", {str(path.relative_to(Path(__file__).parent.parent)):hashlib.sha256(path.read_bytes()).hexdigest()
          for folder in (Path(__file__).parent,Path(__file__).parent.parent/"h100_fp8_diagnostic") for path in folder.glob("*.py")})
    hardware_report(args.output/"nvidia-smi-start.xml")
    try:
        cli([*train_args(args.device_batch_size),"--run-name",args.output.name,
             "--model-dir",".tmp/codex/h100-input-pipeline/models",
             "--board-dir",".tmp/codex/h100-input-pipeline/board",
             "--stop-after-step",str(args.warmup+args.measured)])
    except BaseException:
        write(args.output/"failure.json",dict(traceback=traceback.format_exc()))
        raise
    finally:
        hardware_report(args.output/"nvidia-smi-end.xml")


if __name__ == "__main__":
    main()
