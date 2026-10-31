# H100 production input diagnostic

Production is pinned to **cd7ca922d5d062a4b7f537c1d2e6e672b8a91514** on
B's `fix/swiglu` checkout, inspected clean before branching. B advanced one
commit beyond `3fcd250669bb2f0bb5c25397c058f8cf7bd7d429`; the intervening
commit adds only `experiments/h100_fp8_diagnostic/` and its index entry.
No production implementation changed between those SHAs. No A/C code is merged.
All new files are confined to this experiment directory.

## What inspection establishes

The default `workflows/train_base.py` calls `build_dataloader` with
`use_nanochat=False`. The path is:

1. `ShardManager.iterate` opens existing local Parquet shards, reads a complete
   text row group with PyArrow, converts it to Python strings, and yields lists
   of 128 strings. Reads, decompression and conversion happen on demand in the
   training thread; PyArrow may use its own internal threads.
2. `ShardedDataset` invokes the production tokenizer separately for each
   document, prepending BOS, and creates an int64 tensor for each token list.
   With the GPT2 tiktoken backend, the single-string branch uses
   `encode_ordinary`; the loader's `threads=4` argument does not parallelize
   that branch. A locally cached GPT2 tokenizer is selected by the same
   `load_tokenizer` and `Tokenizer.from_config` path as training.
3. `DistDataLoader._refill` fills a deque with a token budget of `B*T*16`.
   This is synchronous lookahead, not a background prefetch worker. Each batch
   tops up this deque. Packing copies document slices into one persistent CPU
   int64 buffer of `B*(T+1)` elements.
4. On CUDA that staging buffer is pinned. Despite an earlier source comment
   mentioning asynchronous copies, the executed transfer is explicitly
   **`self.gpu.copy_(self.cpu, non_blocking=False)`**. The loader then copies
   shifted views into contiguous input and target views of a persistent output
   allocation. Both outputs are reused on every fetch.
5. `Trainer.prepare_batch` only unpacks tensors, metadata and token count.
   `Trainer.train_step` enqueues forward/backward, then fetches the next batch
   inside every accumulation iteration, including the final prefetch. CPU
   preparation can overlap previously enqueued GPU work. The blocking copy
   can wait on the stream as well as transfer data. The step also checks finite
   loss, schedules hyperparameters, updates the optimizer, clears gradients,
   and preserves its original update-boundary synchronizations.

Python tokenization/packing and the blocking transfer are plausible limits.
Their exposed cost on H100 is **unmeasured**. CPU preparation time is not GPU
waiting time, and the older A100 result does not settle this question.

## Comparison

One worker, one H100. Run the following sequence independently for device batch
16, then device batch 32:

- `production-before`: unchanged production loader and `Trainer.train_step`.
- `cpu-cache`: real packed data already in CPU RAM, copied into a persistent
  staging buffer with the same pinned-memory policy, followed by the literal
  blocking production transfer tail and output-view copies.
- `gpu-cache`: independently owned copies of the same real output batches,
  already on GPU; no loading, staging or H2D transfer.
- `production-after`: repeat the control.

The harness reuses B's recipe, state-copy, restoration, hashing, distribution
and hardware-report helpers from the sibling experiment. Both directories are
uploaded by the launcher; there is no manual artifact dependency. The worker
runs a pristine checkout of the production SHA, separate from uploaded tooling.

The production optimizer profile (`src/gpt_lab/optim/base.yaml`) and parameter
storage policy are unchanged. BF16, SwiGLU, FA3, depth 12, width 768, FFN 3072,
device batch 16 or 32, sequence 2048, global batch 524288,
horizon 2317 and LR warmup 300 are fixed. The recorded B launch's
`target-param-data-ratio=8`, warmdown .65 and final LR fraction .05 are retained.
There is no FP8 conversion. Actual storage dtypes, optimizer configuration,
versions and hardware are recorded.

Both CLI recipes explicitly pass `--device-batch-size <16|32>`,
`--total-batch-size 524288` and **`--n-acc-steps -1`**. Production AutoConfig
infers accumulation; the harness never divides global tokens to infer it.
The CPU preflight calls the production `AutoGPTConfig.compute_accumulation_steps`
method. The GPU runner reads the actual resolved production `TrainerConfig`,
logs it, checks it against preflight, and asserts:

```text
device_batch_size * seq_len * world_size * n_acc_steps == total_batch_size
```

On one H100, production resolution gives:

| Device batch | Sequence | Global tokens/update | Resolved microsteps/update |
| --- | --- | --- | --- |
| 16 | 2048 | 524288 | 16 |
| 32 | 2048 | 524288 | 8 |

Cache lengths, validation lengths and trace range counts use that resolved
configuration. Timing uses complete global updates for both shapes.

Every leg restores the same initial model, optimizer, trainer and RNG state,
starts at the beginning of the same data stream, warms up for at least 25 full
updates and measures at least 50 full updates. Optimizer updates stay enabled.
Each shape runs in a fresh subprocess on the same GPU worker, with separate
TorchInductor cache directories. Compilation follows production's policy and is
retained across the four input legs within a shape; every leg has its own
warmup. The combined report verifies identical initial weights, optimizer groups,
architecture, storage dtypes, trainer settings (except batch/accumulation),
schedules and first-row contents across the two shape runs. Each starts from
the beginning of the same production data stream. Final trained weights across
shapes need not be bit-identical because accumulation/reduction order changes.
Cyclic GC is disabled for the comparison, as in steady
production training. Production `update_metrics` also runs, without logging.
Evaluation, sampling, checkpoint/config saves, metadata autosave and final saves
are bypassed in the harness. Snapshot state lives in RAM, not checkpoint files.

Caching includes all warmup and measurement batches plus the final prefetched
batch (1201 microbatches for B=16; 601 for B=32 at default update counts).
Outputs are cloned before the loader can
reuse its buffers. Copies finish before cache capture advances. Every batch is
checked for contents, shape, dtype and strides; production re-iteration, CPU
replay and CPU preflight data hashes must agree before timing starts.

Default cache overhead is 314988672 bytes of packed CPU tensors and 629669888
bytes of GPU outputs for B=16; for B=32 it is 315250944 and 630194176 bytes,
respectively. Each shape holds only its own caches, plus small persistent
staging buffers and Python/state objects. Exact tensor bytes are reported.
Model/optimizer snapshots add CPU RAM
overhead separately from input caching. All input caches stay allocated across
all four timed legs so cached GPU allocations cannot bias only one leg's peak
memory. Peak allocated/reserved GPU memory is reported after warmup.

Throughput uses the whole measured loop with synchronization at its boundaries.
Median/p95 update times cover full production steps and normal metric updates.
There are no added per-fetch synchronizations in timed paths. The existing
blocking transfer and production synchronizations are preserved. Capture/hash
verification synchronizations occur only outside timing. Startup and warmup
are reported separately, including first-leg compilation.

For each shape, a separate production run warms up for at least 25 updates,
then captures two CPU/CUDA updates with `input/fetch`, `input/transfer`, `compute/forward`,
`compute/backward`, `compute/optimizer` and complete-update ranges. Python method
instrumentation is active only in that run. GPU annotations remain in the raw trace and are
excluded from host-range counts; numeric CPU and GPU process IDs can collide.
The analyzer selects host `user_annotation` events using CPU process and thread
metadata, verifies complete range counts and expected update indices, and
computes GPU idle gaps from the union of CUDA kernels, memcpy
and memset across streams. It reports the largest gaps and their temporal
intersection with input ranges, plus exclusive idle-association buckets for
fetch only, transfer only, both and neither. Those buckets partition idle time
without double-counting. Fetch contains transfer; overlap columns and
variant timing differences are **not additive**. Read the raw trace before
attributing a gap to input rather than launch/optimizer/other host work.

## User launch

No remote preflight or GPU run was launched during implementation.
Run from this branch's worktree, with Modal configured for environment `main`:

```sh
uv run --no-project --with modal==1.6.2 python experiments/h100_input_pipeline/launch.py \
  --run-name h100-input-pipeline-b16-b32-01
```

This builds the pinned image, then invokes a **CPU-only** preflight against the
existing `gpt-lab-cache` volume. It requires contiguous production shards and
existing valid metadata, uses the actual loader/tokenizer at both B=16 and B=32,
T=2048,
reports initialization, first-batch latency, sustained tokens/s and median/p95
batch latency, and hashes the whole comparison stream. Only after that succeeds
does the launcher invoke the single H100 worker. GPU minimum/buffer containers
are zero; registering the function does not request a warm GPU container.
Both functions mount the input volume read-only; no dataset is downloaded or
replaced. The image build may fetch GPT2 tokenizer assets and Python/kernel
software. CPU and GPU functions use 16 allocated CPUs, 64 GiB RAM, OMP/MKL threads
1; the CPU preflight is not evidence about concurrent H100 training contention.

Use a fresh run name. Outputs are preserved under
`gpt-lab-runs/<run-name>/`: `cpu-preflight.json`, `worker.log`, `report.json`,
and `batch-16/` and `batch-32/` subdirectories containing per-shape `report.json`,
`trace.json`, hardware telemetry and a tooling hash manifest. The top-level
`report.json` includes the six batch/path configurations plus both repeated
production controls, each with throughput, median/p95 update time and peak
memory. It separately reports batch-32/batch-16 ratios for each input path;
per-shape reports retain input-path gains, control drift and traces. Partial
failures are retained, with `complete=false` in a partial top-level report.
`--dry-run` prints the plan without importing Modal or making
remote calls. `--cpu-only` invokes just the remote CPU preflight; it reserves
that run name, so use another fresh name for a later full comparison.
Launching requires committed tooling; the report records its commit SHA and
per-file hashes as well as the separate production SHA.

## Local checks and limits

The Mac had no production Parquet shards in its default gpt-lab cache. The local
fixture uses small generated text Parquet files, the production tokenizer
wrapper with a byte vocabulary, and the actual production read/tokenize/pack
path. Its timings are **fixture-only**, not GPT2/climbmix throughput and not
H100 evidence.

```sh
uv run --locked --extra cpu --group dev pytest experiments/h100_input_pipeline/check.py \
  --basetemp=.tmp/codex/h100-input-pipeline/pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked --extra cpu --group dev python \
  experiments/h100_input_pipeline/fixture.py \
  --output .tmp/codex/h100-input-pipeline/fixture-cpu.json
```

The focused checks exercise real Parquet reads, owned caches despite loader
reuse, noncontiguous stride preservation, matching execution across all four
paths with real optimizer updates and state resets, the frozen recipe/transfer
tail, dry-run safety and non-double-counted trace idle intervals. The tiny CPU
optimizer executes eagerly: these tests verify state transitions and arithmetic,
not TorchInductor or CUDA compilation. One initial attempt to compile the CPU
optimizer hit stale shared TorchInductor precompiled headers; the focused test
now explicitly forces eager execution without changing GPU worker behavior.
The updated local check run passed **11 tests**, including the existing trainer
accumulation/prefetch and CLI AutoConfig override regressions. Launcher dry-run also passed with the installed
Modal 1.6.2 interpreter; no remote APIs were invoked. The checks explicitly
verify production AutoConfig resolves B=16 to 16 microsteps and B=32 to 8,
exercise real optimizer updates/cache replays with two locally scaled shapes,
validate both profiler counts, and reject mismatched cross-shape initialization.

The preserved original B=16 fixture run on macOS 26.5.2 arm64, PyTorch 2.9.1, one Torch thread, CPU-only, B=16/T=2048:
128 fixture batches measured about **9.77M tokens/s**, **3.32 ms median**,
**3.96 ms p95**; initialization 14.0 ms and first batch 63.8 ms. This repeated,
byte-level fixture is useful only for checking that the tooling executes.
PyArrow emitted sandbox CPU-cache-discovery warnings. Raw fixture outputs stay
under `.tmp/codex/`, outside Git.

A stable production-versus-CPU-cache gain would implicate read/tokenize/packing;
a further CPU-cache-versus-GPU-cache gain would implicate staging, transfer or
its stream synchronization. Require agreement with the trace and gains larger
than control drift/within-leg variability. Interpret each shape using its own real-cache preflight, repeated controls and
H100 trace; the corrected batch-16 findings are recorded below, and batch 32
remains unmeasured. The two requested microbatch sizes are the only shapes compared. A speedup
from B=32 alone can reflect larger GEMMs or fewer accumulation/launch iterations
and does not demonstrate a loader bottleneck. No loader changes, optimizations,
quality claim or A100-to-H100 extrapolation is included.


## Corrected saved batch-16 analysis and batch-32 continuation

The original `h100-input-pipeline-b16-b32-01` run completed all four unprofiled
batch-16 timing legs, then stopped during trace analysis. PyTorch emitted
identically named CPU `user_annotation` and `gpu_user_annotation` ranges. The
old analyzer selected names alone, counting four update events instead of two.
The corrected selection combines category with CPU process/thread metadata;
it also handles Kineto's numeric CPU/GPU process-ID collisions. Regression
fixtures contain same-named CPU/GPU annotations, an incorrectly tagged GPU-lane
annotation, overlapping kernels on different streams, missing host ranges and
missing metadata. GPU annotations remain intact in the original raw trace.

The saved trace was reanalyzed **offline**. It has exactly updates **25 and 26**;
each has **16 fetch, 16 transfer, 16 forward, 16 backward and one optimizer**
host ranges. GPU busy time is a union across CUDA kernels, copies and memset,
so overlapping activity is counted once:

| Update | Profile wall ms | GPU busy union ms | GPU idle ms | Idle during fetch only ms | Idle during fetch + transfer ms | Other idle ms |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | 1413.378 | 1373.931 | 39.447 | 14.280 | 11.778 | 13.389 |
| 26 | 1410.987 | 1383.859 | 27.127 | 2.462 | 12.624 | 12.041 |

Transfer-only idle is zero in these windows; transfer is nested in fetch.
These are temporal associations in a separately instrumented trace, not causal
stage costs or additional throughput measurements. CPU fetch spans total about
1161/1162 ms per update, mostly overlapping CUDA work or waiting in the blocking
copy; those durations are not exposed GPU idle. One 10.877 ms GPU idle gap in
update 25 coincides with fetching, with only 0.049 ms inside the transfer range.
The trace does not separately identify whether that preparation gap was a
read, tokenization, packing, scheduling or profiler overhead.

The saved unprofiled measurements are independent of the analyzer bug. The
uploaded runner's hash matches its saved manifest. In that runner all timing
legs ended at line 151, before the separate profiler at line 174 and the failing
analysis at line 179. `reanalyze.py` verifies that ordering against the actual
recorded Git revision, recomputes throughput from saved global-token counts and
wall times, and recomputes median/p95 from the original 50 samples per leg:

| Input path | Global tokens/s | Median update ms | p95 update ms |
| --- | --- | --- | --- |
| Production before | 372736.396 | 1406.481 | 1412.220 |
| CPU cache | 373020.554 | 1405.840 | 1410.697 |
| GPU cache | 374433.359 | 1400.108 | 1405.247 |
| Production after | 372722.162 | 1406.525 | 1413.317 |

On this single NVIDIA H100 80GB HBM3, BF16, B=16/T=2048, 524288 tokens/update,
the gains versus the faster production control remain **+0.08% CPU cache** and
**+0.46% GPU cache**. Their median savings (0.641/6.373 ms) are both below the
conservative variability threshold (12.399 ms). This run does not establish a
material input bottleneck at batch 16; it does not establish the batch-32 result.

The original downloaded `report.json`, `trace.json` and manifest remain under
`.tmp/codex/h100-input-pipeline-trace-failure/artifacts/batch-16/`, unchanged.
The separate corrected report is
`.tmp/codex/h100-input-pipeline-trace-failure/batch-16-corrected.json`. It retains
the original measured legs and records source hashes, corrected counts/idle
metrics, verified throughput gains and analysis-source hashes. The offline
`reanalyze.py --trace ... --report ... --output <new-path>` command never trains
or overwrites existing artifacts. It requires the saved sibling manifest and
recorded runner revision to verify measurement independence.

To continue with **batch 32 only**, use a fresh run name:

```sh
uv run --no-project --python 3.12 --with modal==1.6.2 python experiments/h100_input_pipeline/launch.py \
  --device-batch-size 32 --run-name h100-input-pipeline-b32-01
```

This restricts both the real-cache CPU preflight and GPU comparison to batch 32.
Production AutoConfig still receives `--n-acc-steps -1`, resolves **eight**
microsteps on one H100, and validates the global-token invariant. All three
input paths, the repeated production control, shape-specific compilation,
25-update warmup, 50 measured complete updates and the separate corrected
profiler remain enabled. The original batch-16 run is not repeated or modified.
The single-shape top-level report is complete for its requested batch only;
it makes no cross-shape claim. The existing offline `recipe.combine_reports`
helper can compare corrected batch-16 and future batch-32 reports after their
initialization/schedule checks pass. No remote computation was launched while
making this correction.
