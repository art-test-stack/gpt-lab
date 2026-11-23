# RoPE-paired Muon

| Arm | `--pairing` | Five-step Q/K polar map |
| --- | --- | --- |
| A | `real` | Real map per head |
| B | `rope` | Complex map per head using genuine RoPE row pairs |
| C | `shuffled` | Complex map per head using fixed shuffled row pairs |
| D | `joint-real` | One real map per matched Q/K head |
| E | `joint-rope` | One complex map per matched Q/K head using genuine RoPE row pairs |

All arms keep fused QKV, whole-matrix reduce-scatter, Nesterov momentum,
per-head variance normalization and LR scaling, and cautious decay. V uses
the real per-head map. C uses seed `20261118`, breaks every genuine RoPE pair,
and restores the original row order after the map.

D and E require equal Q and K head counts. They concatenate each matched
head's proposals along input columns and split the output before per-head
normalization. E uses adjacent rows as complex components, giving a
`[d_head/2, 2*d_model]` matrix with Gram `Z_Q Z_Qᴴ + Z_K Z_Kᴴ`.

Use `train.py --pairing <arm> auto --muon-qkv head` with the name from the
table. Run from the repository root with the same training options for each arm.

```bash
uv run --locked --extra gpu python experiments/20261118-rope-paired-muon/train.py --pairing joint-real auto --muon-qkv head
uv run --locked --extra gpu python experiments/20261118-rope-paired-muon/train.py --pairing joint-rope auto --muon-qkv head
```

The dashboard config and optimizer checkpoint record the arm. C also saves
its shuffle order. Resume rejects arm changes. Use
`train.py --pairing <saved-arm> resume` with the saved dataset and world size.

## Comparison protocol

**B vs C** is the primary comparison of genuine and shuffled RoPE pairs.
**D vs A** and **E vs B** test sharing. Compare held-out ClimbMix `eval/loss`
(mean next-token NLL in nats) at equal training tokens.

The historical A/B/C runs completed 400 updates with seed 42 before the
ClimbMix loader change. For fresh runs, match the current loader revision,
`climbmix-base` shards, model configuration, GPT-2 tokenizer, and model and
data seeds across arms.

The eight-H100 recipe uses depth 16, BF16, context 2,048, device batch 16,
and two accumulation microsteps for 524,288 tokens per update. The
19,490-update schedule uses 15 tokens per scaling parameter and 1,300
warmup updates. `--stop-after-step 400` keeps that schedule.

Evaluate BPB/NLL and CORE every 400 updates and at short stops, using
4,194,304 validation tokens and up to 500 examples per CORE task.

```bash
--eval-bpb-every 400 --n-bpb-tokens 4194304 --eval-core-every 400 --n-core-tokens 500
```

At update 19,490, evaluate the final checkpoint in a separate pass on the
same split with up to 20,971,520 validation tokens and uncapped CORE
(`max_per_task=-1`). The training CLI uses one budget per metric for the run.

## Dashboard metrics

At update 1, every 50 updates, and the endpoint, the depth-16 preset logs
FP32 Frobenius norms for Q and K heads 0, 5, and 11 in layers 0, 7, and 15.
Keys use `qk/{q,k}/layer_<layer>/head_<head>/...` with the one-based
`qk/completed_update`. The norms cover the reduced gradient, Nesterov
proposal, raw polar output, LR-scaled update, decay, and weight displacement
(absolute and relative to the weight norm).

`--guard-mfu` stops when the median MFU in a 10-update window after the
first 10 updates is at or below 40%. It logs `train/mfu_window_median` and
`train/mfu_guard_tripped`.

## Polar check

```bash
exp=experiments/20261118-rope-paired-muon
uv run --locked --extra cpu python "$exp/check_polar.py"
uv run --locked --extra cpu python "$exp/check_polar.py" --exact
uv run --locked --extra gpu python "$exp/check_polar.py" --device cuda
uv run --locked --extra cpu python "$exp/check_joint_qk.py"
```

`check_polar.py` compares B/C on two seeded `[128, 1536]` proposals.
`--proposals PATH` supplies a tensor with trailing dimensions
`[head rows, input columns]`. Use `--pairing rope` or `--pairing shuffled`
to select one of those checks. `--exact` runs the separate FP64 real-block
and complex SVD comparison. CUDA checks compile the kernels with TF32 disabled.

The five-step check also dispatches D/E through the training optimizer on
two distinguishable `[16, 32]` heads, using real and independent complex
references on identical Nesterov proposals. Relative L2 error must stay
within `2e-4` in FP32 and `0.05` in BF16. Swapping K heads must exceed those
limits. V must match A, and checkpoint resume must reproduce the next
weights and optimizer buffers exactly.

Below are the combined Q/K Frobenius norms and update cosines across the two
matched heads on CPU. The check uses zero decay, and variance normalization
preserves the five-step polar norm. For full-row-rank exact polars in the
same representation, sharing gives `1/sqrt(2)` of the combined norm of two
separate factors.

| Comparison | Precision | Joint norm | Separate norm | Joint / separate | Cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| D vs A | FP32 | 4.139–4.168 | 5.792–5.924 | 0.704–0.715 | 0.937–0.944 |
| D vs A | BF16 | 4.071–4.111 | 5.836–5.896 | 0.697–0.698 | 0.936–0.943 |
| E vs B | FP32 | 2.953–2.957 | 4.092–4.151 | 0.712–0.722 | 0.975–0.980 |
| E vs B | BF16 | 2.961–2.987 | 4.028–4.121 | 0.719–0.742 | 0.974–0.979 |

`check_joint_qk.py` is a four-token key-selection example. It compares
separate, aligned joint, and swapped-plane joint steps at a fixed combined
displacement norm of 0.05 using raw gradients.
