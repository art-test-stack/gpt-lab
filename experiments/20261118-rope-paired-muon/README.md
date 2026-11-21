# RoPE-paired Muon

The paired map follows sections B.2–B.4 of Michal's *RoPE-Paired Muon for Query
and Key Optimizations*. Each head's adjacent real rows represent one complex
row. Five Polar Express iterations operate on the Hermitian Gram matrix of
the whole packed head. The result is unpacked to the real head shape.

| Arm | Experiment CLI | Q/K polar map |
| --- | --- | --- |
| A | `--pairing real auto --muon-qkv head` | Repository per-head real Muon |
| B | `--pairing rope auto --muon-qkv head` | Genuine pairs `(0,1), (2,3), …` |
| C | `--pairing shuffled auto --muon-qkv head` | Fixed shuffled pairs |

Use `train.py` for all three instrumented arms. Training uses the standard
fused `qkv_proj.weight`, forward GEMM, gradient accumulation and whole-matrix
reduce-scatter. V uses the original per-head real polar map. All arms keep
the repository's Nesterov proposal, variance normalization, learning-rate
shape scale based on the real head, and cautious decay.

The shuffled arm uses a dedicated RNG seeded with `20261118`. It accepts a
row permutation only when none of its pairs share a genuine RoPE plane. It
uses that same order for every Q and K head of the same width, then restores
the original row order before the rest of the update. The exact order is
stored in the optimizer checkpoint and checked on load. Resume B/C with
`train.py --pairing rope|shuffled resume` and the saved arm, dataset and
world size. The ordinary training CLI retains its existing modes.

## Dashboard metrics

For the depth-16 preset, the samples are layers 0, 7 and 15 and heads 0, 5
and 11 for each of Q and K. The owner rank measures the FP32 Frobenius norms
on update 1, every 50 completed updates and at the endpoint. The dashboard
keys use `qk/{q,k}/layer_<layer>/head_<head>/...`. The field
`qk/completed_update` gives the one-based update number.

- `weight_gradient_norm` measures the accumulated, reduced gradient before
  the in-place momentum interpolation.
- `momentum_proposal_norm` and `raw_polar_output_norm` bracket the polar map.
- `lr_scaled_update_norm` measures the variance-normalized and LR-scaled
  update before decay. `decay_contribution_norm` measures the separate
  masked and LR-scaled decay term.
- `displacement_norm` measures the difference between the stored weights
  before and after the update. `displacement_to_weight_norm` divides that
  displacement norm by the original weight norm.

The diagnostic reduction sends only the sampled norms to rank 0. It runs
after the ordinary optimizer update and weight gather.

## 400-update comparison

The matched comparison runs A, B, and C sequentially on one node with eight
H100s. All three share the same model seed, prepared `climbmix-base` shards,
GPT-2 tokenizer, and evaluation configuration.
GPT-Lab resolves the batch size and schedules 15 tokens per scaling parameter.

The recipe uses depth 16, BF16, sequence length 2,048, device batch 16,
and two accumulation microsteps for 524,288 tokens per update. It schedules
19,490 updates with a 1,300-update warmup and stops after update 400. At that
endpoint it runs BPB on 4,194,304 tokens and CORE on up to 500 examples per
task, then saves the final sharded checkpoint. W&B shows training metrics on
every update, plus evaluation metrics and the sampled Q/K norms.

With `--guard-mfu`, each arm skips the first 10
completed updates and checks the median `train/mfu` over updates 11–20,
21–30, and each subsequent full window. A median at or below 40% ends the
allocation before the next update, evaluation, or final checkpoint. W&B
records `train/mfu_window_median` and marks a failure with
`train/mfu_guard_tripped`.

## Polar check

```bash
uv run --locked --extra cpu python experiments/20261118-rope-paired-muon/check_polar.py
uv run --locked --extra gpu python experiments/20261118-rope-paired-muon/check_polar.py --device cuda
uv run --locked --extra cpu python experiments/20261118-rope-paired-muon/check_polar.py --exact
```

The default fixture consists of two seeded `[128, 1536]` momentum proposals.
Use `--proposals PATH` for a saved real tensor with trailing dimensions
`[head rows, input columns]` in original head order. The finite check feeds
the same proposal to the production real kernel and the native complex
reference for each pairing. `--pairing rope` or `--pairing shuffled` selects
one pairing. The script reports relative L2 error, cosine similarity and
output-norm ratio per proposal. The CUDA invocation compiles the real kernel
and disables TF32 for the FP32 comparison.

FP32 keeps the solver in FP32. BF16 first rounds the incoming proposal to
BF16, then uses FP32 for the joint Frobenius norm, `1.01 * norm + 1e-6`, and
elementwise polynomial arithmetic. Each packed real GEMM takes BF16 operands
and returns BF16, with the two dot products for each component summed in the
same GEMM. The complex reference rounds each component at the same operand
and result boundaries. Both paths round the final output to BF16. Fused
addition and GEMM accumulation order can still differ between the paths.

`--exact` independently compares the FP64 real block SVD polar with the
complex SVD polar. It uses the same relative singular-value cutoff for both
representations and reports the polar factor on the supported subspace.

Numerical results, timings, GPU-hours and the pilot budget are pending.

## Sharing one polar step between a query head and its key head

`check_joint_qk.py` runs a one-step CPU example with four tokens and two RoPE
planes. At each query position after the first, the target key is the preceding
token. Q and K keep separate weights. Arm B supplies the reference direction
by applying the five-step polar map separately to the two gradients.
The joint variant concatenates their proposals along input columns. Each
complex row then contains the Q and K proposals for the same rotary frequency.
The resulting Hermitian Gram is the sum of the Q and K row Grams.

A control swaps the two complete K planes during the polar step and restores
their order before updating the weights. All three displacements have combined
Frobenius norm 0.05. The script reports the cross-entropy on the same tokens
before and after each step. It uses raw gradients as proposals and omits
momentum history, variance reduction and weight decay.

```bash
uv run --locked --extra cpu python experiments/20261118-rope-paired-muon/check_joint_qk.py
```
