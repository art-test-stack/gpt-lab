# H100 input-pipeline throughput

2026-10-31. **No material production input-pipeline bottleneck was demonstrated
for either device batch size.** Bypassing CPU preparation or host-to-device
transfer changed throughput by less than 0.5%; neither gain exceeded the
harness's conservative variability threshold.

## Setup

One NVIDIA H100 80GB HBM3; PyTorch 2.9.1+cu128; BF16; 12 layers, width 768,
FFN 3072, SwiGLU and FA3. Dataset: climbmix-base Parquet, production GPT2
tokenizer. Production optimizer profile (`src/gpt_lab/optim/base.yaml`) and
parameter-storage policy were unchanged.

Sequence length was 2048 and global batch was 524288 tokens/update.
`--n-acc-steps -1` let production AutoConfig resolve **16 microsteps at device
batch 16** and **8 at batch 32**. The schedule retained horizon 2317, LR warmup
300, warmdown ratio .65 and final LR fraction .05.

For each shape, the order was production → CPU cache → GPU cache → production.
CPU caching bypassed reads/tokenization/packing while retaining production
staging and blocking transfer. GPU caching used the same real batches resident
on device. Contents, shapes, dtypes, strides, initialization and starting data
position were matched. Each shape compiled separately; each path warmed for
25 updates and measured 50 complete production optimizer updates.
Evaluation, sampling and checkpoint writes were disabled.

## Results

Throughput covers complete global updates. Times are milliseconds. Peak memory
is allocated GPU memory, including the resident input cache (about 0.59 GiB).
Caches remained allocated across all timed paths within each shape.

| Device batch | Input path | Tokens/s | Median update | p95 update | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 | Production before | 372736 | 1406.481 | 1412.220 | 27.88 |
| 16 | CPU cache | 373021 | 1405.840 | 1410.697 | 27.88 |
| 16 | GPU cache | 374433 | 1400.108 | 1405.247 | 27.88 |
| 16 | Production after | 372722 | 1406.525 | 1413.317 | 27.88 |
| 32 | Production before | 375353 | 1396.474 | 1401.841 | 50.96 |
| 32 | CPU cache | 374918 | 1399.096 | 1402.820 | 50.96 |
| 32 | GPU cache | 376499 | 1392.944 | 1397.601 | 50.96 |
| 32 | Production after | 375237 | 1397.577 | 1401.774 | 50.96 |

Relative to the faster production control, CPU/GPU caching changed throughput
by **+0.08% / +0.46% at batch 16** and **−0.12% / +0.31% at batch 32**.
Production-control throughput drift was 0.004% and 0.031%, respectively.
Neither cache gain exceeded the larger of control median drift and the maximum
within-leg p05–p95 update-time width; this is a descriptive, conservative check,
not a statistical significance test.

Increasing device batch from 16 to 32 improved production throughput by only
**0.70%**, while peak allocated memory increased from **27.88 to 50.96 GiB**.
This is a microbatch-size effect and does not establish a loader bottleneck.
These results do not justify loader tuning for this recipe; the remaining
compute-side limit was not isolated.

## Trace verification and limits

Separate traces cover updates 25 and 26: each has 16 fetch/transfer/forward/
backward ranges at batch 16, eight at batch 32, and one optimizer range.
GPU idle averaged **33.29 ms / 22.10 ms** per profiled update; **20.57 ms /
10.53 ms** coincided with fetching, including transfer. CUDA activity is unioned
across streams to avoid double-counting. These are temporal associations in
instrumented traces, not additive stage costs or proof of causation.

The batch-16 analyzer initially counted duplicate CPU/GPU annotations. The fix
uses CPU categories and process/thread metadata. Offline reanalysis preserved
the original trace and confirmed unchanged timing samples; batch 16 was not
rerun. Batch 32 completed normally. Eleven focused local checks passed.
Findings apply to this model, hardware and training recipe.

## Provenance and reproduction

Production SHA: `cd7ca922d5d062a4b7f537c1d2e6e672b8a91514`.
Batch-16 harness: `ba1add7eb610d19cd0235ff690a3795599eaf4ff`.
Corrected analyzer and batch-32 harness: `ff30bc277e94f6e9e81e88c8c29818b9df982fba`.
Production source files were unchanged by this experiment.

Original reports, traces and manifests remain on Modal volume `gpt-lab-runs`:

- `h100-input-pipeline-b16-b32-01/batch-16/`
- `h100-input-pipeline-b32-01/batch-32/`

Local offline outputs, preserved outside Git:

- `.tmp/codex/h100-input-pipeline-trace-failure/batch-16-corrected.json`
- `.tmp/codex/h100-input-pipeline-comparison/combined-report.json`

From the repository root, with Modal environment `main` and the existing cached
data volume, reproduce both shapes using a fresh run name:

```sh
uv run --no-project --python 3.12 --with modal==1.6.2 \
  python experiments/h100_input_pipeline/launch.py \
  --run-name h100-input-pipeline-repro-01
```

`--dry-run` prints the plan without allocating compute. `--device-batch-size 32`
selects only batch 32. A real-cache CPU preflight runs before GPU allocation;
input data is read-only and no replacement dataset is downloaded.
