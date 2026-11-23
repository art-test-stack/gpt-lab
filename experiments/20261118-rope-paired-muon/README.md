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
| D | `--pairing joint-real auto --muon-qkv head` | One real map per matched Q/K head |
| E | `--pairing joint-rope auto --muon-qkv head` | One genuine RoPE-paired map per matched Q/K head |

Use `train.py` for all five instrumented arms. Training uses the standard
fused `qkv_proj.weight`, forward GEMM, gradient accumulation and whole-matrix
reduce-scatter. V uses the original per-head real polar map. All arms keep
the repository's Nesterov proposal, variance normalization, learning-rate
shape scale based on the real head, and cautious decay.

The shuffled arm uses a dedicated RNG seeded with `20261118`. It accepts a
row permutation only when none of its pairs share a genuine RoPE plane. It
uses that same order for every Q and K head of the same width, then restores
the original row order before the rest of the update. The exact order is
stored in the optimizer checkpoint and checked on load. Resume with
`train.py --pairing <saved-arm> resume` and the saved dataset and world size.
The dashboard run config records the arm under `pairing`.

D and E require equal query and key head counts. For each layer and head
index, they concatenate the Q and K Nesterov proposals along the input
columns and apply one five-step polar map. E treats adjacent output rows as
the real and imaginary components of one RoPE plane, so its complex matrix
has shape `[d_head/2, 2*d_model]` and Gram `Z_Q Z_Qᴴ + Z_K Z_Kᴴ`.
Both arms split the output back into Q and K before the per-head variance
normalization, diagnostics, and cautious decay.

```bash
uv run --locked --extra gpu python experiments/20261118-rope-paired-muon/train.py --pairing joint-real auto --muon-qkv head
uv run --locked --extra gpu python experiments/20261118-rope-paired-muon/train.py --pairing joint-rope auto --muon-qkv head
```

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

## Comparison protocol

The historical A/B/C runs completed 400 updates with seed 42 before the
ClimbMix loader change. For fresh comparisons, use the same current loader
revision, prepared `climbmix-base` shards, model configuration, GPT-2 tokenizer,
and model and data seeds across arms.

B versus C is the primary comparison of genuine and shuffled RoPE pairs.
D versus A and E versus B are follow-ups that measure the effect of sharing
one polar map per matched Q/K head. The primary metric is held-out ClimbMix
`eval/loss` (mean next-token NLL in nats) at equal training token counts.

The eight-H100 reference recipe uses depth 16, BF16, sequence length 2,048,
device batch 16, and two accumulation microsteps for 524,288 tokens per update.
GPT-Lab schedules 19,490 updates at 15 tokens per scaling parameter, with a
1,300-update warmup. Use `--stop-after-step 400` for a short run on that schedule.

Evaluate BPB/NLL and CORE every 400 completed updates with the routine
budgets below. The same budgets apply at any short stop.

```bash
--eval-bpb-every 400 --n-bpb-tokens 4194304 --eval-core-every 400 --n-core-tokens 500
```

At the full 19,490-update endpoint, score the final checkpoint on up to
20,971,520 held-out validation tokens (about 21M) and uncapped CORE
(`max_per_task=-1`). The training CLI uses one budget per metric throughout
a run, so use a separate evaluation pass on the final checkpoint for these
larger budgets. Use the same held-out split and evaluation procedure across arms.

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

The separate-head fixture consists of two seeded `[128, 1536]` momentum proposals.
Use `--proposals PATH` for a saved real tensor with trailing dimensions
`[head rows, input columns]` in original head order. The finite check feeds
the same proposal to the production packed kernel and the native complex
reference. `--pairing rope` or `--pairing shuffled` narrows this part of the
check. The CUDA invocation compiles the kernels and disables TF32 for the
FP32 comparison.

The finite check also dispatches A, B, D, and E through the training
optimizer on one fused QKV weight with two distinguishable heads. Each
proposal has 16 output rows and 32 input columns per head. It compares D
with a real five-step reference and E with an independent complex five-step
reference, followed by the same per-head variance normalization. Each
matched head must stay within relative L2 error `2e-4` in FP32 and `0.05` in
BF16. The swapped-K reference must exceed that tolerance for both heads.
V must match A across both updates, and resuming D or E from a serialized
checkpoint must reproduce the next weights and optimizer buffers.

For each matched head, the checker reports the combined Q/K Frobenius norm
and the update-direction cosine against its reference and separate-head
baseline. Weight decay is zero, and per-head variance normalization
preserves the five-step polar output norm. A shared full-row-rank exact
polar has the Frobenius norm of one polar factor. The combined norm of two
separate factors in the same representation is `sqrt(2)` times as large,
so the exact shared-to-separate ratio is `1/sqrt(2)`.

The local CPU five-step results below span the two matched heads.

| Comparison | Precision | Joint Q/K norm | Separate Q/K norm | Joint / separate | Direction cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| D vs A | FP32 | 4.139–4.168 | 5.792–5.924 | 0.704–0.715 | 0.937–0.944 |
| D vs A | BF16 | 4.071–4.111 | 5.836–5.896 | 0.697–0.698 | 0.936–0.943 |
| E vs B | FP32 | 2.953–2.957 | 4.092–4.151 | 0.712–0.722 | 0.975–0.980 |
| E vs B | BF16 | 2.961–2.987 | 4.028–4.121 | 0.719–0.742 | 0.974–0.979 |

FP32 keeps the solver in FP32. The paired BF16 map first rounds the incoming proposal to
BF16, then uses FP32 for the joint Frobenius norm, `1.01 * norm + 1e-6`, and
elementwise polynomial arithmetic. Each packed real GEMM takes BF16 operands
and returns BF16, with the two dot products for each component summed in the
same GEMM. The complex reference rounds each component at the same operand
and result boundaries. Both paths round the final output to BF16. Fused
addition and GEMM accumulation order can still differ between the paths.

`--exact` only compares the FP64 real block SVD polar with the
complex SVD polar. It uses the same relative singular-value cutoff for both
representations and reports the polar factor on the supported subspace.

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
