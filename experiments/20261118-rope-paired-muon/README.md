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

Launch one training job per arm from the repository root. This recipe uses
`configs/llama3.yaml` at depth 16 and aspect ratio 96, giving width 1,536
and 12 Q and 12 KV heads of dimension 128. It uses BF16 mixed precision,
FlashAttention 3, GPT-2 tokenization, and the prepared `climbmix-base`
shards in the same order.
The study fixes the dataloader seed at 42, so every fresh run uses the same
shard order. The example sets the Torch seed to 42. Change `seed` to vary
Torch initialization. Context 2,048, device batch 16, and two
accumulation steps give 524,288 tokens per update. The 15N schedule has
19,490 updates and 1,300 warmup updates.

```bash
exp=experiments/20261118-rope-paired-muon
arm=joint-shuffled
seed=42
model_dir=./study-runs
GPTLAB_DTYPE=bfloat16 uv run --locked --extra gpu torchrun --standalone --nproc-per-node=8 \
  "$exp/train.py" --pairing "$arm" --study-eval auto \
  --model-name rope-muon-d16-study --model-dir "$model_dir" \
  --device cuda --board wandb --architecture-config configs/llama3.yaml \
  --depth 16 --aspect-ratio 96 --tokenizer-model gpt2 --ds-name climbmix-base \
  --random-seed "$seed" --attn-impl fa3 --max-seq-len 2048 --device-batch-size 16 \
  --total-batch-size 524288 --target-param-data-ratio 15 --warmup-steps 1300 \
  --eval-bpb-every 400 --n-bpb-tokens 4194304 --n-core-tokens 500 --log-every 1
```

Set `arm` to any table entry. Add `--stop-after-step N` to cap completed
updates. Set `GPTLAB_CACHE_DIR` if the prepared shards are outside your usual
cache, and use `--model-dir` to choose the checkpoint location.

The W&B project is `trainbase_rope-muon-d16-study`. The run name, run ID,
and checkpoint directory name are all `joint-shuffled-s42` for the full
run, `joint-shuffled-s42-stop400` with `--stop-after-step 400`, and
`joint-shuffled-s42-stop1949` with `--stop-after-step 1949`. A cap limits
execution and leaves the 19,490-update schedule intact.

If setup fails before saving a checkpoint or opening a W&B run, fix the
error and rerun the same launch command. An active launch or existing run
blocks a duplicate. The launcher suggests `resume` when it finds a complete
training checkpoint.

The dashboard config and optimizer checkpoint record the arm. The shuffled
arms also save their permutation. To continue an existing run, use
`train.py --pairing <saved-arm> --study-eval resume --checkpoint-dir <saved-run-dir>`
at the same world size, with the same dataset and precision. Resume keeps
the selected run ID and rejects a different optimizer arm. Repeat
`--study-eval` on every resume of a protocol run. To continue a capped run
to the full horizon, add `--stop-after-step 19490` to the resume command.

The study runs routine BPB/NLL on 4,194,304 held-out ClimbMix targets every
400 completed updates. A full run gets 48 passes, ending at update 19,200.
Capped CORE uses up to 500 examples per task at updates 4,873, 9,745, and
14,618, the first integer updates at or beyond 25%, 50%, and 75% of the
full training horizon. A cap adds no off-schedule passes, and resume keeps
the original milestones.

At update 19,490, training saves `base/checkpoint_step_019489` in the run
directory, then scores every eligible target in the held-out ClimbMix split,
including each rank's final partial batch, on the eight training H100s.
The pass saves its results in the final checkpoint's
`checkpoint_state.safetensors` and logs `eval/bpb`, `eval/loss` (mean token
NLL in nats), `eval/scored_targets`, and `eval/full_endpoint_update` to the
training W&B run. Training then exits. If endpoint validation is interrupted,
resume with `--study-eval` to finish it without an optimizer update.

For the 12-update smoke check, omit `--study-eval` and use
`--stop-after-step 12 --eval-bpb-every -1 --n-bpb-tokens 262144 --eval-core-every -1 --n-core-tokens 1`
in place of the evaluation flags above. This runs one validation batch
across all eight ranks and up to one CORE example per task at the cap.

## Run endpoint CORE on one H100

After training exits, run this from the repository root in a separate job
with one H100. Set `model_dir` to the shared checkpoint root or a local copy
of it, and use the same `arm` and `seed` as training. These select the run
directory `<model_dir>/rope-muon-d16-study/<arm>-s<seed>/`. If the run
started with a cap and was resumed to completion, set `run_name` to its
original name with the `-stop<N>` suffix. Keep the run's `meta.json` and
endpoint checkpoint directory together. Step 19,489 is the zero-based
checkpoint index for 19,490 completed updates.
The evaluation load uses the saved tokenizer and model without restoring
the eight-rank optimizer. The job prints aggregate and per-task CORE results.

```bash
arm=joint-shuffled
seed=42
model_dir=./study-runs
run_name="${arm}-s${seed}"
GPTLAB_DTYPE=bfloat16 DEVELOPMENT=0 uv run --locked --extra gpu python - "$model_dir" "$run_name" <<'PY'
import json
import sys

import torch

from gpt_lab.evaluate.core import evaluate_core
from gpt_lab.model.checkpoint import CheckpointManager
from gpt_lab.utils.distributed import ComputeContext

model_dir, run_name = sys.argv[1:]
device = torch.device("cuda:0")
context = ComputeContext(device=device, compute_dtype=torch.bfloat16)
checkpoint = CheckpointManager("rope-muon-d16-study", run_name, model_dir, comp_info=context)
model, tokenizer, state, _ = checkpoint.load(step=19489, phase="eval")
if state.trainer_state.completed_steps != 19490:
    raise ValueError("Expected the checkpoint after 19,490 completed updates")
with torch.autocast("cuda", dtype=torch.bfloat16):
    results = evaluate_core(model, tokenizer, device, max_per_task=-1)
print(json.dumps(results, indent=2))
PY
```

## Inspect a checkpoint

The training command prints the final checkpoint path. The example below
uses a local copy of the full run at
`study-download/rope-muon-d16-study/joint-shuffled-s42/`. Keep the run's
`meta.json`, trainer configuration, and checkpoint together.

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

For capped runs, use the run name with its `-stop<N>` suffix as `model_run`.
For later Hub publication, keep `model.safetensors`, `meta.json`, and the
tokenizer configuration as inputs for packaging the gpt-lab model.

## Dashboard metrics

By default, the study command logs sampled Q/K Frobenius norms in FP32 at
update 1, every 10 completed updates, and the endpoint. For depth 16 it
samples Q and K heads 0, 5, and 11 in layers 0, 7, and 15. Pass
`--no-monitor-qk-norms` before `auto` or `resume` to disable these diagnostics.
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

`check_joint_qk.py` is a four-token key-selection example. It compares
separate, aligned joint, and swapped-plane joint steps at a fixed combined
displacement norm of 0.05 using raw gradients.
