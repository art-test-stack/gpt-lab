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

Eight H100s × 32 sequences × 2048 tokens × one microstep = 524,288 tokens/update.
The literal `ic1-d16` prefix is retained, but names also include `d20`. The
resolved model has 1,165,886,680 unique stored parameters and 457,606,320 scaling
parameters. Scaling counts decoder blocks plus the output table (including a
tied output table once), excluding other embedding tables and residual scalars.
Ratio 10 targets 4,576,063,200 tokens. The existing floor rule gives **8,728
updates**, or 4,575,985,664 tokens (77,536 below target).

Unspecified settings are frozen in `recipe.py` and `selected-runs.json`: seed 42,
GPT2/climbmix-base, 40-update LR warmup, 0.65 warmdown ratio, final LR fraction
0.05, SSSL/512 attention windows, and softcap 18. BPB and CORE run after every
400 completed updates and the final update, using the same budgets for all
variants (41,943,040 BPB tokens; 500 CORE examples/task). Evaluation disables FP8.
Improved BPB and CORE checkpoints are saved independently; the final checkpoint
is resumable. Checkpoint directories retain zero-based indices; reports include
completed-update counts. No checkpoint retention/deletion policy is introduced.

Run from a clean, published `exp/anvil-normuon` checkout. Replace `PUBLISHED_SHA`
with its full commit SHA; the launcher verifies it against the live branch.
Offline planning needs no Modal import or network:

```bash
python experiments/optimizer_comparison/launch.py --sha PUBLISHED_SHA --run-name selected-h100-check-v1 --mode validate --dry-run
```

Only the following user-launched commands allocate paid GPUs:

```bash
uv run --no-project --python 3.12 --with modal==1.6.2 python experiments/optimizer_comparison/launch.py --sha PUBLISHED_SHA --run-name selected-h100-check-v1 --mode validate
uv run --no-project --python 3.12 --with modal==1.6.2 python experiments/optimizer_comparison/launch.py --sha PUBLISHED_SHA --run-name selected-h100-training-v1 --mode train --detach
```

Training mode repeats validation on its allocation, then runs Muon, NorMuon,
and spectral NorMuon sequentially. It stops on any validation or training failure.
No more than eight GPUs are requested concurrently. The jobs use the existing
`gpt-lab` secret (`HF_TOKEN`, `WANDB_API_KEY`) and `gpt-lab-cache`/`gpt-lab-runs`
volumes in Modal's `main` environment; training data is not downloaded.
Tracking uses the existing Forge W&B host/entity and `trainbase_ic1` project.
Each W&B run name includes the launch prefix to keep repeated launches distinct.

The hardware gate checks all eight ranks: compiled spectral zero/near-rank-one,
tall/wide cases; NCCL owner states versus averaged local updates; same-variant
sharded checkpoint/resume; the resolved full model at batch 32; actual compiled
FP8 GEMM operands and BF16 backward; FA3 CUDA kernels; finite losses/gradients,
parameters and state; memory; full-model FP8 checkpoint/resume; and BF16 evaluation.
Dispatch evidence must be observed, not inferred from an enabled flag. The gate
may reject an unrecognized compiler lowering; inspect its trace before changing
acceptance checks. Resume comparisons use rtol 1e-5 / atol 1e-6, not bitwise claims.
The full-model gate uses synthetic tokens and does not establish learning quality.

Artifacts are on `gpt-lab-runs/<run-name>/`: validation reports and traces under
`validation/`, process logs and `launch.json` at the root, and each variant's
models, W&B files, `initial_rank_N.json`, and `selected_result.json` in its own
folder. Initial model/RNG and first data batch/position fingerprints must match
before later variants execute their first update. The result report records the
resolved horizon, rounded/actual tokens, storage policy and best/final paths.
Validation checkpoints are small owner fixtures plus one full-model snapshot;
these are diagnostic artifacts, separate from production checkpoints.

CPU checks do not verify native FP8, CUDA/NCCL, H100 memory fit or learning quality.
No paid compute has been launched by the agent.
