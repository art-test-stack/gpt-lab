# RoPE-paired Muon

| `--pairing` | Five-step Q/K polar map |
| --- | --- |
| `real` | Real map per head |
| `rope` | Complex map per head using genuine RoPE row pairs |
| `shuffled` | Complex map per head using fixed shuffled row pairs |

All arms use genuine RoPE in the forward pass and real per-head Muon for V.
They keep fused QKV, whole-matrix reduce-scatter, Nesterov momentum,
per-head variance normalization and LR scaling, and cautious decay.

## Comparison protocol

At equal training tokens, do genuine RoPE pairs improve held-out ClimbMix
next-token NLL (`eval/loss`, in nats) over fixed shuffled pairs? Compare
`rope` with `shuffled` for the primary study. Use `real` as the reference
for each paired arm. Report the matched `rope` minus `shuffled` NLL at
shared token counts, including full held-out NLL at the chosen endpoint.

Run `real`, `rope`, and `shuffled` once each in the pilot. Within each
replicate, match the initialization seed and dataloader seed across all
three arms so they start from identical weights and see the same batches.
Keep the loader revision, shard manifest, model, tokenizer, precision,
and rank layout fixed. Use one seed pair per replicate and change both
seeds together for any later replicate.

The `shuffled` arm uses the same permutation in every replicate, from a
private RNG seeded with `20261118`. It breaks every genuine RoPE pair.
Try a second permutation only if the primary comparison looks promising.

The historical `real`, `rope`, and `shuffled` runs completed 400 updates
with seed 42 before the ClimbMix loader change. Use the fresh runs below
for matched comparisons on the current loader.

## Warmup and training budget

Choose the training horizon before the warmup check. The reference horizon
is 19,490 updates, about 10.22B training tokens per arm at 524,288 tokens per
update. If compute is tight, lower `--num-steps` equally across the pilot
arms and keep depth 16 and width 1,536. Set `--num-steps` to the actual
horizon and recalculate the 65% LR warmdown, evaluation milestones, and
training budget for that horizon.

Run one `real` check with seeds `(42, 42)`, the intended pilot horizon,
and `--stop-after-step 400`. Inspect training loss and sampled applied Q/K
update norms over the 270 updates after warmup, plus held-out ClimbMix NLL
at update 400. If training is stable, launch fresh `real`, `rope`, and
`shuffled` pilot runs with the same 130-update warmup.

| Stage | Runs | Initialization / data seeds | Updates per run | Total training tokens | Estimated training H100-hours |
| --- | ---: | --- | ---: | ---: | ---: |
| Warmup check | 1 | `(42, 42)` | 400 | 209,715,200 | 0.55 |
| Three-arm pilot | 3 | `(42, 42)` for all arms | 19,490 | 30,655,119,360 | 81.1 |
| Two additional replicates, optional | 6 | `(43, 43)` and `(44, 44)`, three arms each | 19,490 | 61,310,238,720 | 162.2 |

The estimates use 49.80% MFU, the lowest measured arm median, with
approximately `4.69e9` FLOPs per token and GPT-Lab's BF16 peak of
989 TFLOP/s per H100. [Historical MFU](https://github.com/art-test-stack/gpt-lab/pull/53#issuecomment-6992210577)

Budget startup, checkpointing, and evaluation separately. The warmup check
adds one bounded ClimbMix pass at update 400. Each pilot or replicate arm
adds the routine evaluations below, one full ClimbMix pass on eight H100s,
and a separate uncapped CORE job on one H100.

For a chosen horizon `H`, use `H × 524,288` tokens per arm and
`total training tokens × 4.69e9 / (0.498 × 989e12 × 3600)` for training
H100-hours. For example, 4,000 updates give 2,097,152,000 tokens per arm,
2,600 warmdown updates, nine routine ClimbMix passes through update 3,600,
and capped CORE at 1,000, 2,000, and 3,000. Full evaluation moves to 4,000.
The pilot then budgets 6.29B tokens and 16.6 training H100-hours, and the
optional replicates budget 12.58B tokens and 33.3 training H100-hours.

## Launch on eight H100s

Launch one training job per arm from the repository root. This recipe uses
`configs/llama3.yaml` at depth 16 and aspect ratio 96, giving width 1,536
and 12 Q and 12 KV heads of dimension 128. It uses BF16 mixed precision,
FlashAttention 3, GPT-2 tokenization, and prepared `climbmix-base` shards.
Context 2,048, device batch 16, and two accumulation steps give 524,288
tokens per update. `--random-seed` sets the initialization seed and
`--data-seed` sets the training shard order. W&B records both for fresh
runs, and the training checkpoint saves the dataloader's order and cursor.

```bash
exp=experiments/20261118-rope-paired-muon
init_seed=42
data_seed=42
horizon=19490
model_dir=./study-runs

run_study() {
  local arm="$1"
  shift
  GPTLAB_DTYPE=bfloat16 uv run --locked --extra gpu torchrun --standalone --nproc-per-node=8 \
    "$exp/train.py" --pairing "$arm" --data-seed "$data_seed" --study-eval auto \
    --model-name rope-muon-d16-study --model-dir "$model_dir" \
    --device cuda --board wandb --architecture-config configs/llama3.yaml \
    --depth 16 --aspect-ratio 96 --tokenizer-model gpt2 --ds-name climbmix-base \
    --random-seed "$init_seed" --attn-impl fa3 --max-seq-len 2048 --device-batch-size 16 \
    --total-batch-size 524288 --target-param-data-ratio 15 --num-steps "$horizon" \
    --warmup-steps 130 --eval-bpb-every 400 --n-bpb-tokens 4194304 \
    --n-core-tokens 500 --log-every 1 "$@"
}
```

Run the warmup check with the settings above and the intended pilot horizon.

```bash
run_study real --stop-after-step 400 --eval-core-every 0 \
  --run-name "real-s${init_seed}-d${data_seed}-h${horizon}-warmup130-stop400"
```

If the check is stable, submit each fresh pilot invocation as a separate job
using the same horizon and seed pair. For an optional replicate, change
both seed variables in the common settings and repeat the three invocations.

```bash
run_study real
run_study rope
run_study shuffled
```

Set `GPTLAB_CACHE_DIR` if the prepared shards are outside your usual cache.
Use `model_dir` to choose the checkpoint location.

The W&B project is `trainbase_rope-muon-d16-study`. The run name, run ID,
and checkpoint directory name default to `<arm>-s<init_seed>`, with
`-stop<N>` appended for a capped run. An explicit `--run-name` sets all
three, as in the warmup check. A cap limits execution and preserves the
chosen scheduling horizon.

If setup fails before saving a checkpoint or opening a W&B run, fix the
error and rerun the same launch command. An active launch or existing run
blocks a duplicate. The launcher suggests `resume` when it finds a complete
training checkpoint.

The dashboard config and optimizer checkpoint record the arm. The shuffled
arms also save their permutation. To continue an existing run, use
`train.py --pairing <saved-arm> resume --checkpoint-dir <saved-run-dir>`
at the same world size, with the same dataset and precision. Resume keeps
the selected run ID and rejects a different optimizer arm. Keep
`study-eval.enabled` with the run directory so resume selects the saved
study schedule automatically. If you have a protocol checkpoint from an
older launcher that did not save this file, pass `--study-eval` once on resume.
To continue a capped run to the full horizon, set `--stop-after-step` to
the original horizon.

The study runs routine BPB/NLL on 4,194,304 held-out ClimbMix targets every
400 completed updates before the endpoint. With horizon `H`, that's
`floor((H − 1) / 400)` passes, or 48 through update 19,200 at `H = 19,490`.
Capped CORE uses up to 500 examples per task at the first integer updates
at or beyond 25%, 50%, and 75% of `H`, which are 4,873, 9,745, and 14,618
at the reference horizon. A cap adds no off-schedule passes, and resume
keeps the original milestones. The start command records each evaluation budget.
With `--study-eval`, `--eval-bpb-every 0` disables both BPB passes and
`--eval-bpb-every -1` selects full BPB at the endpoint only. CORE uses the
quarter milestones when enabled. Set `--eval-core-every 0` to disable it
independently and `--n-core-tokens` to set its per-task cap.

At update `H`, training saves the checkpoint at zero-based index `H − 1`
(`base/checkpoint_step_019489` at the reference horizon), then scores every
eligible target in the held-out ClimbMix split,
including each rank's final partial batch, on the eight training H100s.
The pass writes `climbmix-full.json` in the run directory and logs
`eval/bpb`, `eval/loss` (mean token NLL in nats), and
`eval/full_endpoint_update` to the training W&B run. Training then exits.
If endpoint validation is interrupted, resume from the saved checkpoint
to finish it without an optimizer update. Keep `climbmix-full.json` with
the run so a later resume can reuse the completed pass.

For the 12-update smoke check, omit `--study-eval` and use
`--stop-after-step 12 --eval-bpb-every -1 --n-bpb-tokens 262144 --eval-core-every -1 --n-core-tokens 1`
in place of the evaluation flags above. This runs one validation batch
across all eight ranks and up to one CORE example per task at the cap.

## Run endpoint CORE on one H100

After training exits, run this from the repository root in a separate job
with one H100. Set `model_dir`, `run_name`, and `horizon` to the checkpoint
root, actual run name, and planned number of completed updates from training.
Keep the run's `meta.json` and endpoint checkpoint directory together.
The evaluation load uses the saved tokenizer and model without restoring
the eight-rank optimizer. The job prints aggregate and per-task CORE results.

```bash
model_dir=./study-runs
run_name=rope-s42
horizon=19490
GPTLAB_DTYPE=bfloat16 DEVELOPMENT=0 uv run --locked --extra gpu python - "$model_dir" "$run_name" "$horizon" <<'PY'
import json
import sys

import torch

from gpt_lab.evaluate.core import evaluate_core
from gpt_lab.model.checkpoint import CheckpointManager
from gpt_lab.utils.distributed import ComputeContext

model_dir, run_name, horizon = sys.argv[1:]
horizon = int(horizon)
device = torch.device("cuda:0")
context = ComputeContext(device=device, compute_dtype=torch.bfloat16)
checkpoint = CheckpointManager("rope-muon-d16-study", run_name, model_dir, comp_info=context)
model, tokenizer, state, _ = checkpoint.load(step=horizon - 1, phase="eval")
if state.trainer_state.completed_steps != horizon:
    raise ValueError(f"Expected the checkpoint after {horizon:,} completed updates")
with torch.autocast("cuda", dtype=torch.bfloat16):
    results = evaluate_core(model, tokenizer, device, max_per_task=-1)
results_json = json.dumps(results, indent=2)
(checkpoint.source_dir.parent / "core-full.json").write_text(results_json + "\n", encoding="utf-8")
print(results_json)
PY
```

## Inspect a checkpoint

The training command prints the final checkpoint path. The example below
uses a local copy of the full run at
`study-download/rope-muon-d16-study/rope-s42/`. Keep the run's
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
    model_run="rope-s42",
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
`qk/completed_update`. Use `lr_scaled_update_norm` to compare either paired
arm with `real` at matched updates. It measures the optimizer update after
variance rescaling and multiplication by the applied learning rate.
`decay_contribution_norm` measures the separate cautious weight-decay term.
Use these traces to check whether a difference versus `real` coincides with
a change in update magnitude. The raw polar output norm and actual weight
displacement remain available as separate diagnostics.

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

## Further Work

Joint Q/K is the next experiment. When Q and K have equal head counts,
`joint-real`, `joint-rope`, and `joint-shuffled` concatenate each matched
head's proposals along input columns, apply a shared map, and split the
outputs before per-head variance normalization. `joint-shuffled` applies
the fixed permutation to both Q and K and restores their row order after
the shared complex map. Compare `joint-rope` with `joint-shuffled` at
matched token counts to test pairing under the shared update.
