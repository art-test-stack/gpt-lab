# Optional torchao FP8 FFN training

This first stage implements A (existing BF16) and B (FP8 FFN forward GEMMs,
BF16 backward GEMMs). No flag means the existing training path: no torchao
import, layer replacement, quantization or optional dependency installation.
The legacy `--fp8` flag is still unused. The new opt-in flag is
`--fp8-ffn-forward`.

Install the optional backend on a compatible Linux GPU machine:

```sh
uv sync --locked --extra gpu --extra fp8 --group dev
```

The `fp8` extra pins `torchao==0.15.0`, matching the project's PyTorch 2.9.1
according to the [torchao compatibility table](https://github.com/pytorch/ao/issues/2919).
The adapter uses torchao's Float8Linear and its low-level training matmul API;
the exact pin is intentional. torchao owns quantization and GEMM dispatch.
The small subclass retains gpt-lab's temporary BF16 weight cast so both backward
operands are BF16 while the trainable Parameter and stored gradient remain FP32.

## Precision and scope

| Operation/storage | This stage |
| --- | --- |
| FFN forward matrix operands | E4M3 FP8, dynamic rowwise scaling |
| FFN forward accumulation | Standard accumulation requested with `use_fast_accum=False` |
| FFN forward result | BF16 |
| FFN input-gradient / weight-gradient GEMMs | BF16 operands, existing PyTorch reduction settings |
| Stored FFN parameters / gradients | FP32, same Parameter objects |
| Optimizer states and communication | Existing project behavior |
| Attention, QKV, embeddings, vocabulary head, loss | Existing project paths |

Each call computes current row maxima and power-of-two-rounded scales, then
casts current BF16 inputs and temporary BF16 weight copies to E4M3. It uses no
FP8 parameter storage, emulation, weight cache, delayed scaling history,
FP8 all-gather, automatic padding, or dual-layout quantization. Input and weight
casts, quantization reductions, temporary buffers and layout handling remain
costs that must be included in timings. The torchao autograd function saves
BF16 operands for backward; it does not reuse FP8 operands for backward here.

The initial native target is Hopper SM90 (H100/H200). A100 runs use the ordinary
BF16 path without the flag; explicitly requesting FP8 on A100 raises a clear
error. Both matrix dimensions and flattened token counts must be divisible by
16. Incompatible cases fail rather than silently report an FP8 result.

All precision comparisons explicitly use `--act-func swiglu`, preserving the
existing SwiGLU width and residual composition. The new model default is squared
ReLU, so omitting this option would change the architecture under comparison.
The output projection still computes the full vocabulary objective. Useful
FLOPs are unchanged between the BF16 and FP8 runs of this same architecture.
For mixed precision, the trainer's existing `train/mfu` means
**BF16-peak-normalized utilization**, not utilization against an FP8 peak.

## Checks

CPU safeguards (no torchao installation or GPU required):

```sh
uv run --locked --extra cpu --group dev pytest -q \
  experiments/fp8_training/check_cpu.py
```

These check the untouched BF16 loss/gradients, lazy imports, device rejection,
conversion wiring, parameter identity, evaluation fallback and saved flag/state
keys. Conversion wiring uses a BF16 test double; it is not FP8 numerical evidence.

Run on H100/H200 before trusting the experimental path:

```sh
uv run --locked --extra gpu --extra fp8 --group dev python \
  -m experiments.fp8_training.check_native \
  --output .tmp/codex/fp8-training/native-check-01
```

The native probe compares a small SwiGLU FFN's outputs, input gradients and both
weight gradients using the same initialization, input and upstream gradient.
It reports absolute and relative L2 errors, norms, finiteness, actual scaled-mm
operand dtypes, scale ranges, clipping and saturation. It checks eager and
compiled results and exports a CUDA profiler trace. Errors are reported without
an invented acceptance tolerance. Finite values alone do not establish adequate
accuracy. Inspect the trace's CUDA kernels before claiming native acceleration.

Quantization and prequantized-GEMM CUDA timings are reported separately, as is
compiled whole-FFN forward/backward latency. Isolated synchronized timings do
not add up exactly to training-step latency. There is no cache refresh timing
because no cache exists. No padded/dual-layout or refresh speedup is claimed.
No native measurements have been made by the authoring agent on this CPU-only
machine. Distributed correctness and matched short-run loss remain hardware
validation tasks.

## Matched real-pipeline runs on Modal

After committing and pushing production and lockfile changes, the existing local
`runs/modal_trainbase.py` launcher installs the optional extra only when the new
flag is present. The launcher requires matching local/origin HEAD and a clean
tracked worktree; experiment files can remain untracked. It rejects an explicit
FP8/A100 request before GPU submission. Native GPU code may still fail validation;
run the native probe first on a compatible GPU environment.

The following commands launch paid GPU work only when the user executes them.
Both use one H100 and the same SwiGLU, real data, attention, optimizer and
schedule settings. The scheduling horizon stays 2317; execution ends at 10
completed optimizer steps. Early steps include compilation. Compare the same
attention backend and steady steps, and do not infer model quality from ten steps.

```sh
# Run A (BF16), then B (FP8 FFN forward) after A succeeds.
env -u VIRTUAL_ENV uv run --locked --extra cpu --with 'modal==1.6.2' \
  python experiments/fp8_training/run_h100_compare.py \
  --run-prefix fp8-1h100-01
```

The wrapper uses `runs/modal_trainbase.py` unchanged for each launch and waits
for both runs; keep the command connected. Each run requests one H100 in a
separate allocation, so the physical GPU may differ. Global batch stays at
524,288 tokens: microbatch 16 × sequence length 2048 × 16 accumulation steps.
Append `--dry-run` to print the paired commands without Git or Modal requests.
The run names are `fp8-1h100-01-bf16` and `fp8-1h100-01-fp8-forward`.

The launcher retains HF/W&B secrets and reports the selected FA2/FA3 backend.
It does not change attention to FP8. Checkpoints retain ordinary weight keys;
the saved trainer flag reconstructs FP8 modules for training resume. Inference
loads ordinary linears. Dynamic scaling is recomputed from current operands on
resume and needs no additional checkpoint history. CPU wiring/state checks cover
the contract; native/distributed resume equivalence has not been established.

Later stages C (FP8 FFN backward), D (output projection), E (QKV), and any cache
or dual-layout optimization require separate correctness and performance evidence.
