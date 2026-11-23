# RoPE-paired Muon

| `--pairing` | Five-step Q/K polar map |
| --- | --- |
| `real` | Real map per head |
| `rope` | Complex map per head using genuine RoPE row pairs |
| `shuffled` | Complex map per head using fixed shuffled row pairs |
| `joint-real` | One real map per matched Q/K head |
| `joint-rope` | One complex map per matched Q/K head using genuine RoPE row pairs |
| `joint-shuffled` | One complex map per matched Q/K head using fixed shuffled row pairs |

All arms use genuine RoPE in the forward pass and real per-head Muon for V.
They keep fused QKV, whole-matrix reduce-scatter, Nesterov momentum,
per-head variance normalization and LR scaling, and cautious decay.

The joint arms require equal Q and K head counts. They concatenate each
matched head's proposals along input columns and split the result before
per-head normalization. Both shuffled arms use the same fixed permutation
from a private RNG seeded with `20261118`. It breaks every genuine RoPE pair.
`joint-shuffled` applies it to both Q and K and restores their row order
after the shared complex map.

## Comparison protocol

At equal training tokens, do genuine RoPE pairs improve held-out ClimbMix
next-token NLL (`eval/loss`, in nats) over fixed shuffled pairs? Compare
`rope` with `shuffled` for the primary study. Compare `joint-rope` with
`joint-shuffled` to ask the same question when matched Q and K share an update.

The historical `real`, `rope`, and `shuffled` runs completed 400 updates
with seed 42 before the ClimbMix loader change. Use the fresh runs below
for matched comparisons on the current loader.

## Launch on eight H100s

Run one fresh process per arm from the repository root. This recipe uses
`configs/llama3.yaml` at depth 16 and aspect ratio 96, giving width 1,536
and 12 Q and 12 KV heads of dimension 128. It uses BF16 mixed precision,
FlashAttention 3, GPT-2 tokenization, and the prepared `climbmix-base`
shards in the same order.
The Torch and dataloader seeds are both 42. Context 2,048, device batch 16,
and two accumulation steps give 524,288 tokens per update. The 15N
schedule has 19,490 updates and 1,300 warmup updates.

```bash
exp=experiments/20261118-rope-paired-muon
arm=joint-shuffled
GPTLAB_DTYPE=bfloat16 uv run --locked --extra gpu torchrun --standalone --nproc-per-node=8 \
  "$exp/train.py" --pairing "$arm" auto --model-name rope-muon-d16-study \
  --device cuda --board wandb --architecture-config configs/llama3.yaml \
  --depth 16 --aspect-ratio 96 --tokenizer-model gpt2 --ds-name climbmix-base \
  --random-seed 42 --attn-impl fa3 --max-seq-len 2048 --device-batch-size 16 \
  --total-batch-size 524288 --target-param-data-ratio 15 --warmup-steps 1300 \
  --eval-bpb-every 400 --n-bpb-tokens 4194304 --eval-core-every 400 --n-core-tokens 500 --log-every 1
```

Set `arm` to any table entry. Add `--stop-after-step N` to cap completed
updates. Set `GPTLAB_CACHE_DIR` if the prepared shards are outside your usual
cache, and use `--model-dir` to choose the checkpoint location.

The local Modal runner uses the same recipe with the prepared `gpt-lab-cache`
volume and writes checkpoints to `gpt-lab-runs`. With the runner present,
`modal run "$exp/modal_trainbase.py" --arm "$arm"` starts one full arm and
`--stop-after-step N` caps it. Use `--arm all` to run the six arms in table
order, each with its own `torchrun` process. Each arm starts after the
previous one finishes successfully.

The W&B project is `trainbase_rope-muon-d16-study`. The run name, run ID,
and checkpoint directory name are all `joint-shuffled-s42` for the full
run, `joint-shuffled-s42-stop400` with `--stop-after-step 400`, and
`joint-shuffled-s42-stop1949` with `--stop-after-step 1949`. A cap limits
execution and leaves the 19,490-update schedule intact.

The dashboard config and optimizer checkpoint record the arm. The shuffled
arms also save their permutation. To continue an existing run, use
`train.py --pairing <saved-arm> resume --checkpoint-dir <saved-run-dir>`
at the same world size, with the same dataset and precision. Resume keeps
the selected run ID and rejects a different optimizer arm.

Trainer evaluates BPB/NLL on 4,194,304 held-out ClimbMix targets and CORE
on up to 500 examples per task every 400 updates and at an early stop.
An early stop on a scheduled update gets one evaluation. Every run saves
a standard final checkpoint in its run directory.

At update 19,490, Trainer saves the checkpoint first and runs one full
evaluation in place of the routine endpoint pass. It scores every eligible
target in the held-out split, including each rank's final partial batch,
and evaluates every example of every CORE task. Each pass saves its results
and any new best scores in the final checkpoint's `checkpoint_state.safetensors`.
If evaluation stops, use the same resume command. It finishes any pending
pass with zero additional optimizer steps. Once both passes finish, a resume
reuses their results and leaves the saved weights untouched.

The same W&B run receives one row at the final update with `eval/bpb`, mean
token NLL in `eval/loss`, `eval/scored_targets`, aggregate CORE scores, and
per-task results under `core/<task>/`. `eval/full_endpoint_update` records the
original update even if an older run already committed that history row.

## Download and inspect a checkpoint

The command prints the final checkpoint path. On Modal, `/runs` is the
mount for `gpt-lab-runs`. The full `joint-shuffled-s42` run saves
`/runs/rope-muon-d16-study/joint-shuffled-s42/base/checkpoint_step_019489`.
Download the run directory to keep its `meta.json`, trainer configuration,
and checkpoint together.

```bash
run=joint-shuffled-s42
mkdir -p study-download/rope-muon-d16-study
modal volume get gpt-lab-runs "/rope-muon-d16-study/$run" study-download/rope-muon-d16-study
```

From the repository root, load the final weights and the GPT-2 tokenizer
recorded in `meta.json` on a CPU. Evaluation on CPU defaults to SDPA. The
explicit override below selects SDPA for this load, and `meta.json` still
records FA3 for training. Use the study's BF16 precision to retain the saved
tensor dtypes. The first tokenizer load may fetch the public GPT-2 vocabulary
and merge rules.

```bash
uv run --locked --extra cpu python - <<'PY'
import torch
from gpt_lab.model.checkpoint import CheckpointManager
from gpt_lab.utils.distributed import ComputeContext

checkpoint = CheckpointManager(
    model_name="rope-muon-d16-study",
    model_run="joint-shuffled-s42",
    model_cachedir="study-download",
    comp_info=ComputeContext(device=torch.device("cpu"), compute_dtype=torch.bfloat16),
)
model, tokenizer, _, _ = checkpoint.load(step="latest", phase="eval", attn_impl="sdpa")
print(model.config)
print(tokenizer.config)
PY
```

For capped runs, use the run name with its `-stop<N>` suffix in both commands.
For later Hub publication, keep `model.safetensors`, `meta.json`, and the
tokenizer configuration as inputs for packaging the gpt-lab model.

## Dashboard metrics

The study command uses the standard `Trainer` with sampled Q/K norms,
benchmark timing, and the MFU guard disabled. Enable `--monitor-qk-norms`
before `auto` or `resume` to log FP32 Frobenius norms at update 1, every 50
updates, and the endpoint. For depth 16 it samples Q and K heads 0, 5, and 11
in layers 0, 7, and 15.
Keys use `qk/{q,k}/layer_<layer>/head_<head>/...` with the one-based
`qk/completed_update`. The norms cover the reduced gradient, Nesterov
proposal, raw polar output, LR-scaled update, decay, and weight displacement
(absolute and relative to the weight norm).

`--benchmark-timing` enables synchronized phase timings. `--guard-mfu`
stops when the median MFU in a 10-update window after the first 10 updates
is at or below 40%. It logs `train/mfu_window_median` and
`train/mfu_guard_tripped`. Both flags are independent of Q/K norm monitoring.

## Polar check

```bash
exp=experiments/20261118-rope-paired-muon
uv run --locked --extra cpu python "$exp/check_polar.py"
uv run --locked --extra cpu python "$exp/check_polar.py" --exact
uv run --locked --extra gpu python "$exp/check_polar.py" --device cuda
uv run --locked --extra cpu python "$exp/check_joint_qk.py"
```

`check_polar.py` checks `rope` and `shuffled` on two seeded `[128, 1536]` proposals.
`--proposals PATH` supplies a tensor with trailing dimensions
`[head rows, input columns]`. Use `--pairing rope` or `--pairing shuffled`
to select one of those checks. `--exact` runs the separate FP64 real-block
and complex SVD comparison. CUDA checks compile the kernels with TF32 disabled.

The five-step check also dispatches all six arms through the training
optimizer on two distinguishable `[16, 32]` heads. It checks each Q and K
head against an independent reference for its arm, and checks V against
the real reference. Each check enforces relative L2 tolerances of `2e-4` in
FP32 and `0.05` in BF16. The joint fixtures detect mismatched K heads. The
`joint-shuffled` fixture also detects leaving K unshuffled or skipping
restoration of the original row order. Checkpoint resume must reproduce
the joint arms' next weights and optimizer buffers exactly.

The earlier CPU comparison below shows combined Q/K Frobenius norms and
update cosines across the two matched heads. The fixture uses zero decay,
and variance normalization preserves the five-step polar norm. For
full-row-rank exact polars in the same representation, sharing gives
`1/sqrt(2)` of the combined norm of two separate factors.

| Comparison | Precision | Joint norm | Separate norm | Joint / separate | Cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| `joint-real` vs `real` | FP32 | 4.139–4.168 | 5.792–5.924 | 0.704–0.715 | 0.937–0.944 |
| `joint-real` vs `real` | BF16 | 4.071–4.111 | 5.836–5.896 | 0.697–0.698 | 0.936–0.943 |
| `joint-rope` vs `rope` | FP32 | 2.953–2.957 | 4.092–4.151 | 0.712–0.722 | 0.975–0.980 |
| `joint-rope` vs `rope` | BF16 | 2.961–2.987 | 4.028–4.121 | 0.719–0.742 | 0.974–0.979 |
| `joint-shuffled` vs `shuffled` | FP32 | 2.783–3.033 | 3.892–4.196 | 0.663–0.779 | 0.977–0.982 |
| `joint-shuffled` vs `shuffled` | BF16 | 2.777–3.037 | 3.896–4.111 | 0.676–0.779 | 0.976–0.978 |

`check_joint_qk.py` is a four-token key-selection example. It compares
separate, aligned joint, and swapped-plane joint steps at a fixed combined
displacement norm of 0.05 using raw gradients.
