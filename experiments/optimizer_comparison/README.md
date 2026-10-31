# Optimizer comparison: existing implementations
Preserves explicit Muon, NorMuon (A), and NorMuon with ANVIL spectral maps (B); this is not full ANVIL2.
B adds all six guarded BF16 spectral maps from update zero, retaining NorMuon row normalization and one momentum rail.
Twin-rail momentum, delayed blending, terminal averaging and parameter-storage changes are not implemented.
Profiles: `configs/muon.yaml`, `configs/a_normuon.yaml`, `configs/b_spectral.yaml`; select with `--optim-config-path`.
Muon beta2/decay are 0.9/0.28; NorMuon A/B are 0.95/0.1; base matrix LR is 0.02 for all three.
CPU spectral/norm/state/resume and owner-update checks passed; compiled spectral GPU/NCCL checks and learning quality remain unverified.
Current recipe/ownership: [INTEGRATION.md](INTEGRATION.md), [selected-runs.json](selected-runs.json); Agent A owns GPU validation and remaining run prerequisites.
Entry: `uv run --locked --extra cpu --group dev pytest experiments/optimizer_comparison`; evidence and coefficients: `results.json`, `REPORT.md`.

Evaluation now runs after completed updates (including the final update), and
`--save-on-best` saves independently for improved BPB and CORE. Checkpoint
directory indices and `best_eval_step`/`best_core_step` stay zero-based for resume
compatibility; new optional `best_eval_completed_steps`/`best_core_completed_steps`
record evaluated update counts. Older checkpoints omit these fields and remain
loadable; their historical evaluation timing is not rewritten.

## Selected H100 runs (prepared, hardware validation pending)

The current selection is depth **20**, width 1280, ReLU-squared, FA3, BF16 compute
with FP8 E4M3 FFN-forward GEMMs and BF16 FFN backward. Parameter storage retains
the existing policy: untied input/value embeddings are BF16; a table tied to the
output head stays FP32; other parameters are FP32. Storage and initialization
are identical across optimizer variants. FP8 is a compute setting, not a storage
change. Capture is disabled and communication uses the baseline backend.

The revised commands resolve to **1,048,576 global tokens/update**, **two
accumulation microsteps**, and **4,364 updates**. They use window 258,
softcap 12, 4,194,300 requested BPB tokens, and final-only periodic checkpoint
saving (`--save-every -1`; independent best saves remain enabled). Model names
retain the requested `ic1-d16-d20` form; run names are arbitrary labels.

No recipe-equality checks or naming/dependency rules are imposed by the launcher.
Native CLI configuration determines the actual model, batch and scheduler. Training starts directly through the native CLI. No experiment gate, fingerprint
hooks or prior-run checks are run automatically.

Run the following direct commands one at a time from the clean published branch.
Replace `PUBLISHED_SHA` with its full commit SHA. Each allocates eight H100s and
starts training directly. There is no execution cap unless
`--stop-after-step N` is explicitly supplied. No `--full-run` option is needed.

```bash
uv run --no-project --python 3.12 --with modal==1.6.2 python runs/modal_trainbase.py \
  --sha PUBLISHED_SHA --gpu H100 --gpus 8 --dtype bfloat16 \
  --cpu 32 --memory 262144 --timeout 86400 --checkpoints \
  --run-name ic1-d20-20261031-muon-v2 -- \
  --model-name ic1-d16-d20-muon --optim-config-path experiments/optimizer_comparison/configs/muon.yaml \
  --ds-name climbmix-base --tokenizer-model gpt2 --random-seed 42 --device cuda \
  --depth 20 --aspect-ratio 64 --d-head 128 --act-func relu_squared \
  --attn-impl fa3 --max-seq-len 2048 --tie-embeddings 'embs;value_embs;head' \
  --window-pattern SSSL --window-size 258 --softcap 12.0 \
  --device-batch-size 32 --total-batch-size -1 \
  --num-steps -1 --target-param-data-ratio 10 --target-flops -1 --target-time -1 \
  --warmup-steps 40 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --fp8-ffn-forward --cuda-graph-mode disabled \
  --eval-bpb-every 400 --n-bpb-tokens 4194300 --eval-core-every 400 --n-core-tokens 500 \
  --save-on-best --save-every -1 --sample-every 0 --log-every 10 --board wandb
```

```bash
uv run --no-project --python 3.12 --with modal==1.6.2 python runs/modal_trainbase.py \
  --sha PUBLISHED_SHA --gpu H100 --gpus 8 --dtype bfloat16 \
  --cpu 32 --memory 262144 --timeout 86400 --checkpoints \
  --run-name ic1-d20-20261031-normuon-v2 -- \
  --model-name ic1-d16-d20-normuon --optim-config-path experiments/optimizer_comparison/configs/a_normuon.yaml \
  --ds-name climbmix-base --tokenizer-model gpt2 --random-seed 42 --device cuda \
  --depth 20 --aspect-ratio 64 --d-head 128 --act-func relu_squared \
  --attn-impl fa3 --max-seq-len 2048 --tie-embeddings 'embs;value_embs;head' \
  --window-pattern SSSL --window-size 258 --softcap 12.0 \
  --device-batch-size 32 --total-batch-size -1 \
  --num-steps -1 --target-param-data-ratio 10 --target-flops -1 --target-time -1 \
  --warmup-steps 40 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --fp8-ffn-forward --cuda-graph-mode disabled \
  --eval-bpb-every 400 --n-bpb-tokens 4194300 --eval-core-every 400 --n-core-tokens 500 \
  --save-on-best --save-every -1 --sample-every 0 --log-every 10 --board wandb
```

```bash
uv run --no-project --python 3.12 --with modal==1.6.2 python runs/modal_trainbase.py \
  --sha PUBLISHED_SHA --gpu H100 --gpus 8 --dtype bfloat16 \
  --cpu 32 --memory 262144 --timeout 86400 --checkpoints \
  --run-name ic1-d20-20261031-normuon-spectral-v2 -- \
  --model-name ic1-d16-d20-normuon-spectral --optim-config-path experiments/optimizer_comparison/configs/b_spectral.yaml \
  --ds-name climbmix-base --tokenizer-model gpt2 --random-seed 42 --device cuda \
  --depth 20 --aspect-ratio 64 --d-head 128 --act-func relu_squared \
  --attn-impl fa3 --max-seq-len 2048 --tie-embeddings 'embs;value_embs;head' \
  --window-pattern SSSL --window-size 258 --softcap 12.0 \
  --device-batch-size 32 --total-batch-size -1 \
  --num-steps -1 --target-param-data-ratio 10 --target-flops -1 --target-time -1 \
  --warmup-steps 40 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --fp8-ffn-forward --cuda-graph-mode disabled \
  --eval-bpb-every 400 --n-bpb-tokens 4194300 --eval-core-every 400 --n-core-tokens 500 \
  --save-on-best --save-every -1 --sample-every 0 --log-every 10 --board wandb
```

Append `--dry-run` before `--` to verify source and print a command without
contacting Modal or launching compute. The image installs the FP8 extra when
`--fp8-ffn-forward` is present. Jobs use the existing `gpt-lab` secret, volumes and
Forge W&B defaults. The three commands are independent blocking launches. Run one at a time to
respect the eight-GPU concurrency limit.

The standalone `check_h100.py` diagnostic is preserved for explicit use; it is
not part of these training commands. Native FP8 backend/device requirements
still apply. Existing CUDA/NCCL validation remains unverified; no paid compute
has been launched by the agent.
