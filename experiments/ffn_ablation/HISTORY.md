# Archived FFN experiment notes

Preserved from stash commit `8491845`; historical commands/source paths below
are provenance, not newly executed or newly validated entry points.

# Experiments

Run these modules from the repository root with the installed project environment.
They are separate from the `gpt-lab` CLI; the old `scripts/benchmark/` entry points
have moved here. Use `--help` for all options and defaults.

## Dataloader benchmark

```bash
uv run --locked --extra cpu python -m experiments.dataloaders \
  --mode synthetic --loader all --batch-size 2 --seq-len 128 \
  --num-batches 5 --num-docs 100 --buffer-size 32
```

Synthetic mode runs offline. `glab` selects both gpt-lab loaders; `all` adds
nanochat. The on-the-fly fixture parses integer text, so it does not measure
natural-language BPE. Initialization and two warm-up batches are excluded.
`--buffer-size` means tokens for gpt-lab and documents for nanochat;
uninstrumented crop/search metrics report `N/A`.

Real mode requires `--bin tokens.bin --idx tokens.idx`: raw uint32 token IDs and
uint64 document offsets, including zero and the final token boundary. It downloads
no data. Hyphenated and underscore options work (`--batch-size`/`--batch_size`);
`--parquet_dir` and `--loader packed` are unsupported.

## Independent FFN architecture ablations

### Native configuration

New `TransformerConfig` and `AutoGPTConfig` objects default to
`act_func="relu_squared"`. Fresh training accepts
`gpt-lab train auto --act-func relu_squared ...` or `--act-func swiglu`.
`relu_squarred` is accepted as an alias. Explicit Python `None` selects SwiGLU;
checkpoint metadata missing `act_func` (or storing null) is migrated to SwiGLU.
Both FFNs preserve the internal residual. New squared-ReLU configs are saved
with their activation so checkpoint reconstruction selects the correct matrices.

`DenseTransformer.estimate_flops()` is the source for native AutoGPTConfig
training FLOPs and MFU. It counts actual projection weights (including the
padded vocabulary head) plus inclusive causal attention pairs. It excludes
elementwise operations, optimizer work, conversion and recomputation. This
replaces the old rectangular-attention estimate, so historical MFU values use
a different numerator. It can also change automatically chosen horizons when
targeting FLOPs; loaded trainer configurations retain their saved settings.
Native auto-scaling uses the selected FFN for probe, target and reference models,
so automatic token targets and optimizer scaling can differ from historical
experiments that initialized and sized a SwiGLU model before swapping its FFN.

Offline integration check: `python -m experiments.check_native_ffn`.

Baseline: post-#37 `10ac35ff94e9d3ebec68f00280fc3c35913d8cdd`.
`ffn_architecture` provides experiment-only variants; it does not modify the
production model configuration or checkpoint loader:

| Flag | FFN | Inner residual |
| --- | --- | --- |
| `--architecture A` | Production SwiGLU, hidden width `d_ffn` | Preserved |
| `--architecture B` | Same SwiGLU | Removed |
| `--architecture C` | Squared ReLU, hidden width `d_ffn` | Preserved |

With the automatic `d_ffn=4*d_model`, A/B use a packed `d→8d` input projection
and `4d→d` output. C uses `d→4d→d`. A/B isolates residual composition; A/C
changes the FFN structure and matrix budget together. No quality conclusion
follows from the synthetic correctness checks.

Run locally with the installed environment:

```bash
uv run --locked --extra cpu --group dev --no-sync python -m experiments.ffn_architecture \
  --architecture A --self-check
uv run --locked --extra cpu --no-sync python -m experiments.ffn_architecture \
  --architecture C --config .tmp/codex/ffn-fp8/baseline.json
```

The config argument accepts a JSON `TransformerConfig` or a manifest with a
`model` key. Without `--self-check`, the command only reports configuration,
parameter counts and useful FLOPs; it does not allocate a model or train.
The saved inspection manifest records the local versions and the historical
resolved configuration; it is a local artifact, not a packaged example.

For a later training harness, initialize a production model, call
`apply_architecture(model, flag)` once, then create its optimizer. B reuses both
Parameter objects. C copies the first half of A's input projection and reuses
the output projection, preserving non-FFN initialization and consuming no RNG.
C uses the standalone `SquaredReLUFeedForward` component in
`src/gpt_lab/model/layers.py`; its constructor takes `d_in`, `d_latent`, and
`dropout` (plus an optional device). It returns `MLP(x) + x`, matching the
measured C variant. `DecoderLayer` now selects its FFN from `config.act_func`.
The production initialization rule (including zero output projection) is kept.
The self-check also uses nonzero FFN output weights to exercise both branches
and both weight gradients, checks BF16 loss/gradients against production for A,
and checks the explicit FFN equations for each variant. It checks finite
gradients, storage dtypes, counts, and strict base-state loading for A/B.

The decoder composes `h + MLP(norm(h)) + norm(h)` in A/C and
`h + MLP(norm(h))` in B, after attention. Its norm epsilon is currently `1e-8`,
while the outer model norm uses the configured epsilon. Learned per-layer
scales blend the incoming stream and initial normalized embeddings before
attention. Both residual additions occur in the earliest imported history;
their combined intent is unknown. The comparison reference is
[nanochat at 92d63d4](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py),
whose squared-ReLU MLP has no inner residual. C only matches its FFN activation
and width, not its residual composition or full architecture.

Counts use unique stored parameters and dense forward/backward projection
work (2 FLOPs per multiply-add), plus inclusive useful causal attention pairs.
The padded vocabulary head is included. Elementwise operations, normalization,
optimizer work, quantization/conversion, GEMM padding and recomputation are
outside that FLOPs convention. A/B share the same useful FLOPs. No timings,
FP8 kernels, matched training comparisons or utilization results are reported
by this first architecture step. Do not save B/C through the production
checkpoint API: its configuration cannot identify their changed semantics.

### A100 BF16 timing on Modal (user launch)

From the repository root, with the existing Modal CLI/account configured:

```bash
# Offline plan only; starts no Modal build or function:
python3 experiments/modal_ffn.py
# Paid GPU execution begins only when you run this command:
modal run -e main --detach experiments/modal_ffn.py \
  --run-name ffn-a100-01 --warmup-steps 8 --steps 10
# After completion, download reports and logs:
mkdir -p .tmp/codex/ffn-fp8/modal-results
modal volume get -e main gpt-lab-runs /ffn/ffn-a100-01 \
  .tmp/codex/ffn-fp8/modal-results/ffn-a100-01
```

One A100-80GB runs fresh processes in A/B/C/C/B/A order, with a one-hour
job timeout and no retries. Use a new run name for each launch; existing output
is never overwritten. The existing `gpt-lab-runs` volume must exist in `main`.
To rerun only C through the full-model training benchmark against saved A
results, use:

```bash
modal run -e main --detach experiments/modal_ffn.py \
  --run-name ffn-c-a100-01 --architecture C --warmup-steps 8 --steps 10
```

This launches one C process, including full-model forward/backward, accumulation,
the production optimizer and schedules, and a separate whole-FFN timing.
The Modal flags use optimizer-step terminology; the direct benchmark also
accepts the older `--warmup` and `--updates` aliases for existing scripts.

The image checks out the pinned baseline, installs locked GPU dependencies,
and uploads experiment files plus the modified native model/config/CLI source files. No dataset/cache
volume or credentials are attached. Attention kernel code may be fetched by
the existing production kernel loader; no datasets are downloaded. The runner
requires A100 and FlashAttention 2 and fails rather than changing the attention
control. Production source other than the explicitly uploaded files must match the
pinned baseline; all uploaded source hashes are recorded and verified remotely.

`ffn_baseline.json` preserves the resolved saved model/trainer configuration:
12 layers, width 768, FFN width 3072, context 2048, vocabulary 50257 padded to
50304, six heads of width 128, SSSL windows of 512 tokens. All cases use BF16,
microbatch 16, accumulation 4, 131072 tokens/update, and the saved Muon/AdamW
profile, horizon 2317 and LR warmup 300. All start from the same seed, zero
optimizer state, schedule position and deterministic four-batch synthetic ring.
Only the architecture differs. The baseline json's historical source path is
provenance; the runner reads the checked-in values and needs no local artifacts.

Each process warms up eight updates before ten timed updates. The first warmup
also checks gradient finiteness and storage dtypes and records its gradient
norm. Steady-state timing excludes those diagnostics; loss and final parameter
finiteness are checked. Compilation counters must stay unchanged in each timed
window. Startup/warmup is reported separately. Whole-FFN forward, backward and
combined timings use a separate freshly initialized FFN with BF16 inputs and
fixed upstream gradients, with synchronization between forward and backward.
Training peak allocated memory, per-update losses, timing samples/variability,
tokens/s, source hashes, versions, configuration and useful FLOPs are saved.
Utilization explicitly uses 312 TFLOP/s dense BF16 per A100. It is independent
of the FFN conversion and optimizer work excluded from useful model FLOPs.

`launch.json` records status/completed cases, `summary.json` collects results,
and each case has `results.json` plus a sibling log. Failure logs/status are
committed to the volume before the exception propagates. This is a single-GPU
synthetic benchmark; it excludes data loading/H2D and distributed communication.
Short synthetic losses establish neither equal model quality nor convergence.
FP8 is disabled throughout and must be tested separately on supported hardware.

Local workflow smoke (real CPU BF16 trainer/optimizer math, no GPU):

```bash
uv run --locked --extra cpu --group dev --no-sync python -m experiments.check_ffn_benchmark
```

### C with the actual four-GPU training pipeline

```bash
modal run -e main --detach experiments/modal_train_ffn_c.py \
  --run-name ffn-c-4xa100-01
```

This separate launcher uses the defaults and cached-input preflight from
`runs/modal_trainbase.py`: four A100-80GB GPUs, cached climbmix-base/gpt2,
12 layers, width 768, context 2048, microbatch 16 per GPU, global batch 524288
tokens (four accumulation microsteps), FlashAttention 2, BF16, the production
distributed optimizer and schedules. The native `--num-steps 2317` defines the
schedule and `--stop-after-step 10` ends execution after ten completed steps.
There are no extra benchmark warmup steps. The LR warmup remains 300 steps.

The experiment adapter calls the normal CLI with `--act-func relu_squared` and
`Trainer.train()`. Native AutoGPTConfig selects and sizes C before initialization
and computes trainer FLOPs through `model.estimate_flops()`; the adapter no longer
replaces FFNs or overrides FLOPs. `resolved-rankN.json` records the native config.
Checkpoint
exports, evaluations and sampling are disabled. W&B settings and the `gpt-lab`
Modal secret (`HF_TOKEN`, `WANDB_API_KEY`) match `runs/modal_trainbase.py`.
The secret is injected into the GPU worker and inherited by its subprocesses;
token values are not recorded in the launch manifest. Existing cache/run volumes are required;
preflight checks their data before requesting GPU work. The existing attention
loader may fetch kernel code, but no datasets are downloaded.

Reports and streaming logs are saved to
`gpt-lab-runs:/ffn-training/<run-name>`. Use a fresh run name. The first steps
include compilation/startup; compare the later per-step timings with the same
timing boundary in the saved four-GPU baseline. The logged MFU uses useful C
FLOPs and the dense BF16 peak; faster tokens/s need not imply higher MFU.

