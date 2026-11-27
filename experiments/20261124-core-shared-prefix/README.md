# CORE shared-prefix scoring

CORE keeps separate choice rows as its default. A question is eligible for packing when every cropped row has the same nonempty prefix at the same positions, every continuation starts immediately after that prefix, and every choice has at least one scored token. Ineligible questions use the separate-row scorer. Schema and language-modeling tasks use the default common-suffix and exact-match scoring paths.

## Time to returned CORE scores

`core_timing.py` is the user-facing command for the opt-in mode. It calls `evaluate_core` with a task scorer that batches up to four multiple-choice questions within each task. Unpackable questions use separate rows, and schema and language-modeling tasks use the stock task scorer. The trainer still calls CORE with its default options.

On Apple M2, stock master took a median of **106.47 s** and the opt-in mode took **57.01 s**. The median paired speedup was **1.87×**, saving **49.46 s per evaluation**. Both used the frozen two-layer FP32 GPT-Lab checkpoint, the byte tokenizer, and a 2,048-token logical context. All 352 predictions and all 22 task scores matched. Centered CORE was `0.03214244121780599` in both arms. The machine paged during the run, with noticeable variation across the three pairs.

| Pair | Stock master | Batched wide, prefetch and deferred readback | Time saved |
| --- | ---: | ---: | ---: |
| 1 | 106.47 s | 57.01 s | 49.46 s |
| 2 | 124.34 s | 76.32 s | 48.03 s |
| 3 | 85.93 s | 10.14 s | 75.80 s |

Timing used stock `61f066250420afbc121524037b996ee4a3524c0a` and candidate `4aab1cd2fd031abfe8714714b6ebec2adf988f94`. The checkpoint SHA-256 was `c8102efd75dfd7f2cfcba814916a53eb0db43e41cc86c0ab3b9c29b63cca595f`.

Stage the CORE bundle and use the frozen `frozen.pt` from the MPS parity run below. From the repository root, set `FROZEN` to that checkpoint and `RESULTS` to a directory under `.tmp/codex/`.

```bash
script=experiments/20261124-core-shared-prefix/core_timing.py
export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2
uv run --locked --extra cpu python "$script" --checkpoint "$FROZEN" --output "$RESULTS/core"
uv run --locked --extra cpu python "$script" --checkpoint "$FROZEN" --output "$RESULTS/paired" \
  --mode paired --isolated-trials
uv run --locked --extra cpu python "$script" --checkpoint "$FROZEN" --output "$RESULTS/overlap" --mode overlap
uv run --locked --extra cpu python "$script" --checkpoint "$FROZEN" --output "$RESULTS/profile" --mode profile
```

The paired command runs stock master at `61f066250420afbc121524037b996ee4a3524c0a` and records the candidate commit. Each trial starts after model load and warm-up, before CORE loads task data and renders prompts, and ends after scores are on the host and the device is synchronized. Fresh processes keep Metal allocations from carrying across arms. A separate pass records multiple-choice and schema winners, scored language-model predictions, and the exact cropped tokens and scoring spans. The report includes all 352 examples across 22 tasks, three alternating timing pairs, per-task score deltas, and changed predictions.

The prepared-token overlap command keeps the same packed groups and 36 fallbacks across all four configurations. Prefetch prepares one upcoming batch on a worker. Deferred readback concatenates the small winner tensors and reads them in one call. All configurations include readback and aggregation. The component profile separately reports host dispatch and instrumented MPS intervals. Overlapping intervals and profiling overhead make these durations unsuitable for summing into an uninstrumented wall time.

| Four-question loop | Median wall | Questions/s | Sampled peak live tensors | GPU idle in trace | Collation overlapping GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 5.061 s | 34.78 | 285.1 MiB | 1.853 s | 0.00 ms |
| Prefetch only | 4.832 s | 36.42 | 285.7 MiB | 9.445 s | 0.51 ms |
| Deferred readback only | 4.747 s | 37.07 | 293.3 MiB | 2.015 s | 0.00 ms |
| Both | 4.824 s | 36.49 | 293.3 MiB | 1.916 s | 2.76 ms |

The separate Metal traces count time with no active GPU interval for the benchmark process. The prefetch-only trace was an outlier at 12.22 s, while the other traces took 4.55–4.75 s. With prefetch enabled, every subsequent batch finished collation by the end of the preceding CPU submission. The traces showed no collation on that critical path, so a Rust prototype was unnecessary.

In the component pass, rendering took 220 ms, byte tokenization took 5.9 ms, tokenization and span discovery together took 114 ms, grouping took 0.10 ms, and collation took 148 ms including 18 ms to stack position IDs. The single deferred host copy including device wait took 0.61 ms in this pass. Transferring the resulting tensor to Python values took 0.02 ms and accuracy aggregation took 0.10 ms.

| Stage | Instrumented MPS interval total |
| --- | ---: |
| Attention | 3.662 s |
| Dense branch masks | 0.652 s |
| Native RoPE table gather | 0.013 s |
| Q/K rotation | 0.092 s |
| Vocabulary projection | 0.015 s |
| FP32 cross-entropy | 0.200 s |
| Option reduction | 0.244 s |

The largest mask allocation sampled inside a construction call was 154 MiB above its entry allocation. The sweep forwarded 421,272 physical token slots, including 2,986 padding slots. Its largest physical row was 2,056 and its highest logical position was 2047. The profile reports the number of valid MPS intervals per stage, since Metal dropped some event pairs. Attention and mask construction accounted for most of the sampled intervals. The measurements did not warrant RoPE caching or selective vocabulary projection on this checkpoint.

Record Metal GPU execution separately from wall timing. Compile the optional signpost library, then run the following for each of `current`, `prefetch`, `deferred`, and `both` as `CONFIG`. Set `PYTHON` to the repository environment's absolute Python executable.

```bash
mkdir -p "$RESULTS/traces"
xcrun clang -dynamiclib -O2 experiments/20261124-core-shared-prefix/core_signposts.c \
  -o "$RESULTS/traces/core_signposts.dylib"
xcrun xctrace record --template 'Metal System Trace' --instrument 'Points of Interest' --no-prompt \
  --env "PYTHONPATH=$PWD/src" --output "$RESULTS/traces/$CONFIG.trace" \
  --launch -- "$PYTHON" "$script" --mode overlap --configuration "$CONFIG" --repeats 1 \
  --metal-trace --signpost-library "$RESULTS/traces/core_signposts.dylib" \
  --checkpoint "$FROZEN" --output "$RESULTS/traces/$CONFIG"
```

Export the two interval tables from each recording and summarize them with `trace_summary.py`.

```bash
for SCHEMA in OSSignpostIntervals metal-gpu-intervals; do
  xcrun xctrace export --input "$RESULTS/traces/$CONFIG.trace" \
    --xpath "/trace-toc/run[@number='1']/data/table[@schema='$SCHEMA']" \
    --output "$RESULTS/traces/$CONFIG-$SCHEMA.xml"
done
python experiments/20261124-core-shared-prefix/trace_summary.py "$RESULTS/traces"
```

The packed row contains the prefix followed by each choice in order. A branch starts at the prefix length in RoPE positions and can see the prefix and its own causal history. Each layer applies its left window to the difference in logical positions. The final prefix logit supplies all first-token losses. Each later token uses the loss from its own predecessor. The score is mean continuation NLL, with ties going to the first minimum.

Packed SDPA uses the model's explicit `complete_attn_masks` input to supply these visibility rules. Ordinary `attn_mask` tensors restrict the model's usual causal and sliding-window rules.

## Apple MPS parity

Apple M2, macOS 27.2.1, PyTorch 2.11.0. The same seeded two-layer GPT-Lab model ran both paths in FP32 eval mode with SDPA. It has width 32, two heads of width 16, a 2,048-token context, a four-token left window in the first layer, and full causal attention in the final layer. The tokenizer uses UTF-8 bytes and a separate BOS.

The sample used 16 examples per task across all 22 CORE tasks. Of the 176 multiple-choice questions, 140 packed and 36 used separate rows.

| Measurement | Result |
| --- | ---: |
| Absolute FP32 tolerance, logits and NLL (`rtol=0`) | `2e-5` |
| Max absolute scored-logit difference | `7.1525574e-7` |
| Max absolute per-choice NLL difference | `9.5367432e-7` |
| Multiple-choice winner changes | 0 |
| CORE score, both paths | 0.0321424412 |
| Mean accuracy, both paths | 0.2301136364 |
| Max absolute B-logit difference after changing only A's suffix | 0 |

The same check covers a single-token answer, unequal choice lengths, aligned and misaligned crops, the exact sliding-window edge, and first-minimum tie breaking.

From the checkout root, stage the existing CORE bundle and run the parity script.

```bash
export GPTLAB_CACHE_DIR="$PWD/.tmp/codex/core-shared-prefix-cache"
mkdir -p "$GPTLAB_CACHE_DIR/data/core"
curl -fsSL https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip \
  -o "$GPTLAB_CACHE_DIR/data/core/eval_bundle.zip"
unzip -q "$GPTLAB_CACHE_DIR/data/core/eval_bundle.zip" -d "$GPTLAB_CACHE_DIR/data/core"
DEVELOPMENT=0 PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --locked --extra cpu \
  python experiments/20261124-core-shared-prefix/parity.py --device mps --max-per-task 16
```

The measured archive has SHA-256 `90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200`.

### Paired SDPA timing

With the same bundle staged, run the prepared-token timing harness on the M2.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --locked --extra cpu \
  python experiments/20261124-core-shared-prefix/timing.py --repeats 3
```

Each eligible question runs through the reference, the separate-row scoring control, and the packed scorer in a rotated order. The control applies the packed scorer's first-token calculation to separate rows and skips the unused argmax. The benchmark warms up the entire sample and reports trial totals and medians. It synchronizes each scorer call and includes packing, mask construction, model forward, and scoring. Prompt preparation and parity assertions run outside the timed sections. Each trial measures the fallback scorer once per ineligible question and shares its cost across the three full-sample totals.

### Four questions per forward on MPS

This experiment uses one frozen FP32 SDPA checkpoint from the seeded parity model and the same byte tokenizer for all paths. It archives master at `61f066250420afbc121524037b996ee4a3524c0a` and runs that source in a persistent subprocess. Before timing, the harness checks all 352 examples for exact winners, scored-token argmaxes, per-task scores, and centered CORE. It also reports bitwise equality and maximum absolute differences for logits and option NLLs.

The four-question layouts group the 176 multiple-choice questions by packed width. Strict mode sends packed widths above 2,048 to the separate-row scorer. Wide mode accepts those rows when their largest logical position is at most 2047. Both preserve per-choice cropping, exclude padding keys and targets, and compute option reductions and first-minimum selection on-device. The 2,160-token fixture runs separately from the timing sample and checks that mutating a branch or padding leaves unrelated logits exactly equal.

After staging the CORE bundle as above, run the comparison from the checkout root.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --locked --extra cpu \
  python experiments/20261124-core-shared-prefix/batching.py \
  --device mps --work-dir .tmp/codex/core-question-batching --repeats 3
```

The work directory stores the frozen checkpoint and `report.json`. The harness warms each scorer before running paired sweeps with CPU prefetch off and on. Prefetch builds one upcoming batch's inputs on a worker thread. The timed sweep includes CPU collation, transfer, mask construction, forward, and scoring, with MPS synchronization at both boundaries. Prompt rendering, tokenization, parity checks, and subprocess startup run outside the timed sweeps.

On Apple M2 with PyTorch 2.11.0, all three packed paths matched pinned master on all 352 examples and 33,244 scored-token argmaxes. All 22 per-task scores matched and centered CORE was `0.03214244121780599`. Each path had the same largest absolute differences, `7.1525574e-7` for scored logits and `9.5367432e-7` for option NLLs. Bitwise equality was false for both quantities. The 2,160-token fixture reached logical position 2047. The unaffected logits after branch or padding mutation matched exactly.

The measured frozen checkpoint has SHA-256 `c8102efd75dfd7f2cfcba814916a53eb0db43e41cc86c0ab3b9c29b63cca595f`.

### Packing with four questions in both arms

The separate-row arm groups up to four questions and forwards one full row per answer. The wide arm uses one packed row per question. Both arms use the same device option-NLL reducer, first-minimum selection, and one combined readback per forward. Prefetch is off. Prompt rendering and tokenization run before timing. [Separate-row batching in nanochat #526](https://github.com/karpathy/nanochat/pull/526)

The eligible comparison uses identical length-sorted question groups. The full comparison also lets the separate-row arm batch questions that require a separate forward as packed fallbacks. Both arms run the parity sweep before timing to warm those shapes. Timed trials alternate arm order. MPS peak memory is the highest sampled live tensor allocation in a separate pass.

With the bundle staged and the frozen checkpoint from the earlier run, use

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --locked --extra cpu \
  python experiments/20261124-core-shared-prefix/packing_pair.py \
  --checkpoint .tmp/codex/core-question-batching/frozen.pt \
  --report .tmp/codex/core-question-batching/packing-pair.json --repeats 3
```

Append `--batch-sizes 4 8 16 32` to measure both scorers at each question batch size. Each size reports the eligible and full cohorts separately, plus the fastest observed size for each scorer and cohort. The harness records allocation failures per scorer and size. For isolated Metal driver allocations, run the command once per size with a separate report path. Sampled peak memory covers standalone evaluation tensors.

The fixed-width sweep rounds physical widths up to the next bucket. Separate rows still cap width at 2,048. Padding does not change real tokens' logical positions or loss spans. `--isolate-trials` runs each warm-up check, timed trial, and memory pass in a fresh process. It warms the scorer before timing, then clears unused tensor allocations outside the timer. Trial order alternates. Process start, prompt preparation, and warm-up are outside timed sweeps.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2 uv run --locked --extra cpu \
  python experiments/20261124-core-shared-prefix/packing_pair.py \
  --checkpoint .tmp/codex/core-question-batching/frozen.pt \
  --report .tmp/codex/core-question-batching/packing-pair-isolated.json \
  --batch-sizes 4 8 16 32 --width-buckets 512 1024 2048 2160 --isolate-trials --repeats 3
```

## Two-H100 follow-up

Use the same CORE bundle and a single GPT-Lab RoPE checkpoint with its tokenizer. Set `GPTLAB_BENCH_MODEL_DIR` to the checkpoint cache root, `GPTLAB_BENCH_MODEL` and `GPTLAB_BENCH_RUN` to its model and run names, and `GPTLAB_BENCH_STEP` to the explicit checkpoint step. Point `GPTLAB_CACHE_DIR` at the staged CORE bundle and tokenizer cache on the node.

```bash
CUDA_VISIBLE_DEVICES=0,1 GPTLAB_DTYPE=bfloat16 uv run --locked --extra gpu \
  --with 'nvidia-cutlass-dsl==4.4.2' --with 'apache-tvm-ffi>=0.1.5,<0.2' \
  --with einops --with torch-c-dlpack-ext \
  torchrun --standalone --nproc-per-node=2 \
  experiments/20261124-core-shared-prefix/benchmark.py \
  --model-dir "${GPTLAB_BENCH_MODEL_DIR:?set the checkpoint cache root}" \
  --model-name "${GPTLAB_BENCH_MODEL:?set the model name}" \
  --run-name "${GPTLAB_BENCH_RUN:?set the run name}" \
  --step "${GPTLAB_BENCH_STEP:?set the checkpoint step}" --max-per-task 64
```

| Arm | Attention | Choice scoring |
| --- | --- | --- |
| A | Current native FA3 | Separate rows and reference scoring |
| B | Native FA4 with `causal=True` | Same separate rows and reference scoring |
| B-score | Same native FA4 as B | Separate rows with the packed scorer's first-token NLL calculation on eligible examples |
| C | Native FA4 with branch-aware `mask_mod` | Packed prefix on eligible examples, B on fallback |

FA4 uses Hub v0 at commit `a3944dc87df804a42f0248d16c780fed90f84508`. The public entry point in this build has no runtime metadata parameter. C calls the pinned native forward implementation with device tensors for token positions, branch IDs, and the layer window. A, B, B-score, and C each get a CUDA profiler probe and must show the expected Hopper attention kernel. [Pinned FA4 source](https://huggingface.co/kernels-community/flash-attn4/tree/a3944dc87df804a42f0248d16c780fed90f84508/build/torch-cuda)

Before warming up real questions, each rank runs the same cropped sibling-mutation fixture through native FA4. The fixture requires exact equality of B's scored logits after mutating A and a matching separate-row winner. A direct attention probe exposes the native mask with one output channel per key. It checks the exact set of visible keys under the four-token and full windows. Preflight compilation is reported separately.

Each rank warms up its full CORE shard before timing. The synchronized sweep includes prompt preparation, mask construction, forward, and scoring. The report includes CuTe compile time, unique questions divided by the slower rank's elapsed time, peak PyTorch-allocated bytes per GPU, and speedup over A. Every arm uses the same example IDs, few-shot selections, and context limit. Each forward handles one question.

BF16 parity uses an absolute tolerance of `0.05` nats per choice with zero relative tolerance. The report includes the largest NLL difference, any decision flips with each arm's winner-to-runner-up margin, and a failure status for differences outside tolerance. H100 kernel execution, parity, and throughput are pending.

### Paired four-question FA4

The opt-in comparison uses the same eligible and full-sample plans as the MPS harness. Both arms call native BF16 FA4. The packed mask reads its position and branch metadata using the batch row and token indices and excludes padding keys. The preflight covers the 2,160-token cross-row fixture. Each rank checks the native kernel symbols, option NLL tolerance, and decision margins before the timed sweeps. Each trial uses the slower rank's wall time and records the actual CUDA peak allocated tensor bytes per rank.

```bash
CUDA_VISIBLE_DEVICES=0,1 GPTLAB_DTYPE=bfloat16 uv run --locked --extra gpu \
  --with 'nvidia-cutlass-dsl==4.4.2' --with 'apache-tvm-ffi>=0.1.5,<0.2' \
  --with einops --with torch-c-dlpack-ext \
  torchrun --standalone --nproc-per-node=2 \
  experiments/20261124-core-shared-prefix/benchmark.py --compare-four-question \
  --model-dir "${GPTLAB_BENCH_MODEL_DIR:?set the checkpoint cache root}" \
  --model-name "${GPTLAB_BENCH_MODEL:?set the model name}" \
  --run-name "${GPTLAB_BENCH_RUN:?set the run name}" \
  --step "${GPTLAB_BENCH_STEP:?set the checkpoint step}" --max-per-task 64
```

Append `--batch-sizes 4 8 16 32` to the paired FA4 command for the same throughput sweep. If either rank runs out of CUDA memory, that scorer and size are recorded as an allocation failure. The runner keeps measuring the other scorer and requested sizes.

## Single-A100 PyTorch SDPA sweep

Append `--validate-backends` to run the single-A100 follow-up. The runner replays the three previous batch-1 winner changes, forces both arms onto the same SDPA implementation, and probes the pinned Hub FA4 source for native SM80 dispatch, sibling isolation, padding, and the sliding-window boundary. It tries SDPA math if matched memory-efficient SDPA fails validation. Both matched SDPA and native FA4 get timed sweeps when their winners, sampled task scores, and 1% relative NLL criterion agree with pinned master. Each passing backend gets an all-task score comparison before its all-task timing. `validated-report.json` records the diagnostics even when timing is skipped. The local Hugging Face Jobs launcher stages the optional pinned FA4 dependencies and uses the existing cache bucket for results.

Run from the repository root with CORE staged at `GPTLAB_CACHE_DIR/data/core/eval_bundle`. Use a trained base RoPE checkpoint with a 2,048-token context and its matching tokenizer. The runner loads the tokenizer from checkpoint metadata. For a relocated local tokenizer, set `GPTLAB_BENCH_TOKENIZER_DIR` to its directory.

```bash
GPTLAB_DTYPE=bfloat16 uv run --locked --extra gpu \
  python experiments/20261124-core-shared-prefix/benchmark_a100.py \
  --model-dir "$GPTLAB_BENCH_MODEL_DIR" --model-name "$GPTLAB_BENCH_MODEL" \
  --run-name "$GPTLAB_BENCH_RUN" --step "$GPTLAB_BENCH_STEP" --output "$GPTLAB_BENCH_OUTPUT"
```

The sweep also accepts a pretrained Llama-family model through Transformers. For the locked environment, omit the optional Hub kernels package to use PyTorch SDPA.

```bash
uv sync --locked --extra gpu --no-install-package kernels
GPTLAB_DTYPE=bfloat16 uv run --no-sync \
  python experiments/20261124-core-shared-prefix/benchmark_a100.py \
  --hf-model HuggingFaceTB/SmolLM2-360M \
  --hf-revision f8027fd0eaeea54caa13c31d31b9fdc459c38b49 --output "$GPTLAB_BENCH_OUTPUT"
```

This path uses the same Hub weights, tokenizer, and Transformers SDPA forward in both scorers. The pinned master still supplies the separate-row CORE reference. Prompts and choices are cropped to 2,048 logical tokens with the model's pretrained RoPE settings.

The sweep starts with the PR's 176 multiple-choice candidates and checks packing eligibility using the checkpoint tokenizer. It extends the seeded task pools if fewer than 128 qualify. It balances selection across tasks and chooses evenly spaced sequence lengths within each task. It saves the question IDs, physical lengths, and batch membership in `selected.json`. Both scorers use those same prepared questions at sizes 1, 4, 8, and 16, with 128, 32, 16, and 8 forwards per sweep respectively.

`master-reference.json` comes from the pinned master's original separate-row forward and option reduction. `preflight.json` records the branch and padding fixture and BF16 parity against that reference. Each size gets three alternating timed pairs after warm-up, with CUDA synchronization and prefetch off. `batch-<size>.json` and `report.json` record SDPA operators and GPU kernel symbols from a probe for each batch shape, raw timing and peak CUDA allocation, paired packing speedup, and each scorer's throughput relative to its own size 1. If a configuration runs out of CUDA memory, its partner can still be checked against the saved master reference. The A100 parity gate requires the mean absolute relative change across option NLLs to be at most 1% of the pinned reference. A zero reference NLL contributes zero when the candidate is also zero. An undefined relative change fails the gate. Winner changes, their margins, task scores, and the largest absolute and relative NLL differences are reported for every configuration. The runner records parity before timing, saves the timing for every runnable size, and profiles afterward. Profiling errors are recorded with the results. The command returns a failed parity status after saving the reports when a configuration exceeds the limit.
