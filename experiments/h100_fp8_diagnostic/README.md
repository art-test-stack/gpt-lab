# One-worker H100 FP8 diagnostic

Production code is frozen at **3fcd250669bb2f0bb5c25397c058f8cf7bd7d429**.
No production code, dependencies, architecture, optimizer mathematics or precision
storage policy is changed by this harness. Its local source is uploaded separately;
`harness_manifest.json` records SHA256 hashes of every Python file uploaded.
No production push is required: Modal fetches the already published pinned SHA.

Run from the repository root (this command launches paid compute **only when you run it**):

```sh
env -u VIRTUAL_ENV uv run --locked --extra cpu --with 'modal==1.6.2' python experiments/h100_fp8_diagnostic/launch.py --run-name h100-fp8-diag-01 --detach
```

Use a fresh name after a failed or completed run. Append `--dry-run` to print the
fixed plan without importing Modal or contacting any service. `--warmup 20
--measured 50` are the defaults; smaller counts are rejected. The single detached
worker owns the whole diagnostic, so client disconnection does not interrupt the
sequence. There is no separate GPU function per leg and no automatic retry.

Requires existing `gpt-lab-cache` / `gpt-lab-runs` volumes in Modal's **main**
environment and `HF_TOKEN` in the existing `gpt-lab` secret. Dataset and tokenizer
cache presence are checked on CPU before allocating a GPU. FA3 and FP8 support
are checked inside the worker. Missing support is an error. The image uses the
same locked PyTorch GPU dependencies plus the existing optional torchao extra
for every leg. Hub downloads of code/binary attention kernels may occur; dataset
downloads are not requested. The native loader receives `base_url=None`.

## Fixed recipe and state

One H100, world size 1; BF16 autocast; SwiGLU width 768, hidden width 3072,
projections **768→6144 and 3072→768**; 12 layers; microbatch **16×2048**;
**16 accumulation microsteps**, or 524,288 tokens per complete optimizer update.
FA3, SSSL/512 windows, full-vocabulary head/loss, embedding tying, optimizer groups,
residuals and data packing are from the pinned production recipe. Horizon remains
**2317**, LR warmup **300**, warmdown ratio .65, final LR fraction .05. FP8 is
the existing forward-only FFN E4M3 rowwise recipe with fast accumulation disabled;
backward GEMMs remain BF16. Parameter identity/sharing and parameter/gradient/
optimizer tensor dtypes are checked or recorded. No caches, FP8 backward, head
conversion or QKV conversion are added.

The real CLI builds the model, loader, optimizer, precision policy and schedules
once. Only `Trainer.train` is replaced, inside the diagnostic subprocess: updates
call the production `Trainer.train_step` and compiled `model.training_loss`.
No alternate loss or synthetic batches are used. Each leg copies the same initial
model and optimizer snapshot from CPU RAM, restores trainer and Python/NumPy/
Torch/CUDA RNG state, and rebuilds the production loader from the same start.
Snapshots are never written to disk. First/last prefetched batches and the final
loader state must match across ABA legs. Model parameter objects remain shared
with the optimizer. The existing `DisableFP8` context supplies the BF16 controls.

Every leg trains warmup updates 0–19 before sampling at identical schedule
positions. Unprofiled samples cover updates **20–69**. Short profiler legs cover
updates **20–21**. These are training updates, not an alternative scheduling horizon.
The native execution cap is 70 completed updates with the default counts.
The first compilation-containing update and all warmup durations are saved
separately; they are excluded from the measured distribution. Evaluation, sampling,
checkpoint writes, periodic logging and W&B are disabled for all legs. Reports
and raw traces replace W&B for this diagnostic. Initial JSON configuration files
are written only to scratch space; no model/optimizer checkpoints are exported.

## Order and evidence

1. Warm BF16 and collect two full-update CPU/CUDA traces.
2. Reset, warm FP8-forward and collect two full-update traces. At actual compiled
   Inductor extern GEMM calls, record operand dtypes, dimensions, generated-code
   callsites and direction. Require 768 FP8 forward calls in two updates, using
   E4M3 operands at both real projection shapes, and 1536 BF16 backward GEMMs.
   Correlate the forward calls with CUDA GEMM kernels using profiler external IDs /
   launch correlation IDs. Missing or unexpected dispatch stops the diagnostic;
   a flag, module type or eager dispatch observation does not pass the gate.
3. After those gates, independently reset/warm/time **BF16 → FP8 → BF16**, each
   with 20 warmup and 50 measured complete production updates. No profiler,
   runtime operand wrappers or profiling/debug compiler options are active in
   these timed legs. All 16 accumulation microsteps, prefetch and optimizer work
   are inside the trainer's synchronized update timing. Logging and report writes
   are outside it. Initial state, data and scheduling position match each leg.

Profiler-only annotations observe external GEMMs and existing generated Triton
launchers; they do not change kernel arithmetic or introduce graph breaks.
Inductor's post-grad node provenance attributes FFN forward/backward, conversion /
casts, attention, head/loss and optimizer work. Raw generated graph metadata and
code are saved to audit that attribution. Quantization fused with activation or
other work is labelled as shared fused work, **not** assigned an invented pure
quantization cost. Backend-internal setup inside a GEMM remains part of that GEMM.
Unattributed and mixed kernels remain visible rather than being guessed away.

The CUDA timeline is partitioned across all streams into **exclusive** component
time, overlap, and exposed GPU idle. Those buckets sum to the complete update
window; component rollups and per-kernel raw duration sums must not be added to
them again. Exposed idle can include host/launch/data/synchronization gaps and
profiler overhead; it does not prove a particular CPU bottleneck. The unprofiled
ABA measurements determine speed, not profiler timings.

`report.json` compares the median saving against the faster BF16 leg, BF16 drift,
and the largest within-leg p05–p95 width. The report marks a gain beyond observed
variation only when the median saving exceeds both drift and that width; this is
a conservative descriptive check, not a significance test. Paired losses, peak
allocated/reserved memory, useful FLOP/s and **BF16-peak-normalized utilization**
are also recorded. Useful FLOPs stay unchanged between precisions. This does not
establish equal model quality or a general H100 performance result.

## Reports

Everything is saved under `gpt-lab-runs/h100-fp8-diag-01/`, including partial
evidence on failure. Download only this diagnostic's reports/traces:

```sh
modal volume get -e main gpt-lab-runs /h100-fp8-diag-01 .tmp/codex/h100-fp8-diag-01
```

Start with `report.json`, `worker.log`, each profile leg's `trace_summary.json`,
`gemm_operands.json`, `compiled_graphs.json`, and `trace.json`. Chrome traces can
be opened in Perfetto. NVIDIA-SMI XML records the worker's hardware/driver state
before and after; compiler caches stay in disposable container scratch space.

## Local checks (no GPU or network)

```sh
UV_CACHE_DIR=.tmp/codex/h100-diagnostic/uv-cache TORCHINDUCTOR_CACHE_DIR=.tmp/codex/h100-diagnostic/inductor-cache uv run --locked --extra cpu --group dev --no-sync pytest -q experiments/h100_fp8_diagnostic/check.py --basetemp .tmp/codex/h100-diagnostic/pytest-tmp -o cache_dir=.tmp/codex/h100-diagnostic/pytest-cache
```

These check the pinned recipe/plan, snapshot restoration with the real optimizer
factory, loader/RNG reset, variation reporting, complete-update trace accounting,
overlap/correlation handling, dispatch rejection and a real small compiled CPU
forward/backward operand observation. They do **not** verify native CUDA dispatch,
GPU provenance coverage, profiler permissions or speed. Those are checked only
when the user launches the H100 worker.
