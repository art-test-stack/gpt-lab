# CORE shared-prefix scoring

CORE keeps separate choice rows as its default. A question is eligible for packing when every cropped row has the same nonempty prefix at the same positions, every continuation starts immediately after that prefix, and every choice has at least one scored token. Ineligible questions use the separate-row scorer. Schema and language-modeling tasks use the default common-suffix and exact-match scoring paths.

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

Each rank warms up its full CORE shard before timing. The synchronized sweep includes prompt preparation, mask construction, forward, and scoring. The report includes CuTe compile time, unique questions divided by the slower rank's elapsed time, peak PyTorch-allocated bytes per GPU, and speedup over A. Every arm uses the same example IDs, few-shot selections, and context limit. Each forward handles one question.

BF16 parity uses an absolute tolerance of `0.05` nats per choice with zero relative tolerance. The report includes the largest NLL difference, any decision flips with each arm's winner-to-runner-up margin, and a failure status for differences outside tolerance. H100 kernel execution, parity, and throughput are pending.
