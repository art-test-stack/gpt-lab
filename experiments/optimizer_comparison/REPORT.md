# NorMuon / ANVIL2 ablations

Inspection date: 2026-10-30. No paid compute launched.

## Preserved starting point

- Existing branch: `feat/comm-backward`, commit `192bbe2921e4774d24324b2d07852311b7bc4dcf`.
- New branch: `exp/anvil-normuon`, created directly from that commit.
- The existing branch had no tracked working-tree changes. Its commits and the ignored local `runs/modal_trainbase.py` are preserved.
- A/B are implemented in `experiments/optimizer_comparison/configs/a_normuon.yaml` and `experiments/optimizer_comparison/configs/b_spectral.yaml`, selected by the optional Boolean `anvil_spectral` group setting. Omitting it keeps existing NorMuon behavior. The second option, momentum variants C/D, and diagnostic instrumentation remain pending the next implementation steps.

## Historical source and provenance

Use the source embedded in the ANVIL2 record, rather than today's refactored trainer:

- [Historical record](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2026-08-30_ANVIL2).
- [Archived run containing the source](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2026-08-30_ANVIL2/this_pr/01b2f094-4536-479c-b0dd-cb14eba4844c.txt).
- Immutable run blob: [`d2f2d975d7ed87287e91bbe00306194090b5e4f9`](https://api.github.com/repos/KellerJordan/modded-nanogpt/git/blobs/d2f2d975d7ed87287e91bbe00306194090b5e4f9). Its Git blob hash was verified after download.
- Embedded `train_gpt.py` SHA256: `c6da246b1d87a525c0130f212d40e9a32e7eba01738e97686011d0854dd6945a`. The prefix matches the record README's shipped-source hash.
- Relevant archived source lines: 213–319 (maps and velocity), 3382–3395 (lookahead schedule), 3604–3634 (engagement and scalar assignment).

## Starting implementation

`src/gpt_lab/optim/kernels/normuon.py` has a single EMA momentum rail, a Nesterov blend using the same coefficient, five repetitions of the original quintic Newton–Schulz map, BF16 orthogonalization, row-wise adaptive second moments, and norm restoration after the row rescale. Parameters and optimizer state follow the existing storage policy. NorMuon uses ordinary decoupled decay.

The shared Muon strategy already preserves whole-matrix ownership and can assemble complete bucket partitions into original owner-local gradient slots before one batched update/gather. Its current backend still launches reductions inside `optimizer.step()`, after backward.

The older optimizer named `muon` has a five-map Polar Express schedule, shape-dependent adaptive normalization, and cautious decay. Those similarities are not an implementation of ANVIL2's six maps or twin rails, and `muon` is not the requested NorMuon baseline.

Neither ANVIL2 mechanism was present in the starting branch. The spectral-only option has since been added; the baseline path retains the original arithmetic.

## Equations extracted from the archive

For reduced gradient `g_t`, fast state `f_t`, slow state `s_t`, fast decay `b_t`, slow decay `0.98`, fast weight `w_t`, and Nesterov coefficient `mu_t`:

```text
f_t = b_t f_(t-1) + (1 - b_t) g_t
s_t = 0.98 s_(t-1) + 0.02 g_t
m_t = w_t f_t + (1 - w_t) s_t
v_t = (1 - mu_t) g_t + mu_t m_t
```

Both rails start at zero and update from step zero. Before archived update 514, `w_t = 1` and `b_t = mu_t`. Starting at update 514, `w_t = 0.4385` and `b_t = 0.85`. The Nesterov coefficient continues to use the schedule below; it does not become 0.85 at engagement.

The source uses 1,122 scheduled steps, 52 added schedule-growth steps, and 20 extension steps: `T_ref = 1194`. For zero-based source update `u`:

```text
mu(u) = 0.85 + 0.08 * u/240                  if u < 240
        0.93                                if 240 <= u <= 1194 - 50
        0.93 - 0.08 * (u - (1194 - 50))/50   otherwise
```

The six spectral maps take the form

```text
X_(k+1) = a_k X_k + (b_k A_k + c_k A_k^2) X_k
A_k = X_k X_k^T
sigma_(k+1) = a_k sigma_k + b_k sigma_k^3 + c_k sigma_k^5
```

Tall matrices use the equivalent right-multiplication form with `A_k = X_k^T X_k`.

| Map | a | b | c |
|---|---:|---:|---:|
| 1 | 3.923798038567 | -6.095026865488 | 3.905234618423 |
| 2 | 3.278126713798 | -3.328923386476 | 0.989127286973 |
| 3 | 3.505298394150 | -5.137358782410 | 1.968325560615 |
| 4 | 2.815058591845 | -3.685181239622 | 1.417196497642 |
| 5 | 2.245503932403 | -2.443826979899 | 0.963091710461 |
| 6 | 2.256537145403 | -2.166840097229 | 0.929501253245 |

The historical entry normalization uses `d = 1.05 * sqrt(trace(A_raw)) + 1e-6`, where the first Gram is formed from the BF16 velocity. Both `X` and that first Gram are divided by `d` and `d^2`, respectively. This also has a particular rounding path; a straightforward compiled PyTorch implementation will not reproduce the custom Triton kernels bit-for-bit.

Baseline NorMuon divides its BF16 input by `||X||_F + 1e-7`. The historical safety factor is a material requirement of these six maps, not an interchangeable implementation detail. The spectral option includes this approved input guard while retaining NorMuon's row-wise normalization. It always executes all six maps; the matched `ns_steps: 5` field still controls the baseline algorithm when the option is off.

## Proposed horizon mapping, not yet implemented

For a target scheduler horizon `H`, use continuous reference time `u = 1194 * t / H`, where `t` is the completed-update count before the next update. Evaluate the historical schedule at `u`, and engage the slow-rail blend when `u >= 514`. Keep slow decay `0.98`, post-engagement fast decay `0.85`, and weight `0.4385` unchanged. This preserves schedule positions as fractions of training progress; it does not rescale EMA time constants or reproduce the historical changing token batch sizes.

With the launcher's existing `H = 2317`, fast-beta warmup ends near update 466, blending begins at update 998, and cooldown begins near update 2220. An execution cap of 150 must not change any of these boundaries or the LR/decay horizon. At the last diagnostic update, `t = 149`, the proposed Nesterov coefficient is approximately `0.875594303`.

Consequently, a 150-update training diagnostic will exercise two rail updates in C/D but not a nontrivial slow-rail blend. Correctness examples must separately exercise the engagement boundary and cooldown; learning-quality comparisons require longer runs. Compressing engagement into the diagnostic would change the experiment.

## Scalar spectral inspection performed

These are CPU float64 scalar polynomial calculations, not measured BF16 matrix behavior or GPU performance. The sweep used 10,001 logarithmically spaced singular values.

- On `[0.003, 1/1.05]`, the historical six-map cascade outputs approximately `[0.995067199, 1.000401391]`.
- NorMuon's repeated five-map cascade on the same interval outputs approximately `[0.681831462, 1.202368602]`.
- The six-map derivative at zero is about `643.133485`; the repeated five-map derivative is about `484.876287`. Exact zero singular values remain zero.
- With unguarded input `sigma = 0.97`, the historical six maps produce approximately `5.1e176`; near `0.99` the scalar calculation overflows. Near-rank-one matrices can therefore fail if only the coefficient tuples are replaced.

The NorMuon row rescale preserves the incoming spectral update's Frobenius norm rather than forcing every algorithm to the same target norm. Equal LR therefore does not imply equal update size. For equal nonzero input spectra, guarded-six-map / baseline pre-rescale Frobenius ratios are approximately `1.436458` (rank 1), `0.902572` (rank 2), `0.937324` (rank 8), and `1.453036` (rank 64).

Small CPU BF16 matrix examples on the M-series Mac, PyTorch 2.9.1, with shape `4x8` and `8x4` produced nonzero output singular values `1.0078125` for rank one and `0.99609375` for rank four. Zero matrices stayed zero. A diagonal input spectrum `[1, 0.1, 0.01, 0.003]` produced singular values `[1.0078125, 1, 1, 0.9921875]`, versus approximately `[1.109375, 0.78125, 0.6875, 0.671875]` with the baseline. These are correctness examples, not GPU or throughput measurements.

## Optimizer-state memory accounting

For a same-shaped NorMuon group of `N` matrices with shape `m x n`, world size `W`, original owner allocation `K = ceil(N/W)`, and state element size `q` bytes:

```text
A/B state per rank = q K (mn + m)
C/D state per rank = q K (2mn + m)
additional twin-rail state = q K mn
```

The row-wise second moment contributes `q K m`; padding in owner state is included. For local execution use `K=N`. Existing BF16 training uses FP32 matrix storage/state for the relevant transformer groups, so normally `q=4`. The experiment must preserve that policy. Excluded from these formulas: parameters, gradients, communication buffers, activations, allocator overhead, compiler workspaces, and tiny schedule/step metadata.

For `N=5`, `W=2`, `q=4`: a `2x4` group uses 120 bytes per rank in A/B and 216 bytes in C/D; a `4x2` group uses 144 and 240 bytes. Either shape adds exactly 96 bytes of momentum state per rank. A/B state sizes, FP32 state dtype, and same-variant disk checkpoint round trips are checked on small examples. C/D values are estimates pending implementation.

## Implementation and diagnostic constraints

- A: unchanged existing NorMuon; B: spectral stage only; C: momentum/blend only; D: both.
- All profiles must share NorMuon beta2, LR/decay values and schedules, group membership/order, architecture, data order, precision policy, and scheduler horizon.
- Do not import ANVIL2's alternative lane-equalizer axis, cautious decay, decay multipliers, mantissa sidecar, terminal averaging, storage changes, architecture changes, FP8 paths, or data schedules.
- A/B should retain existing momentum state shape. C/D need one extra matrix-sized rail plus checkpointed update position/horizon sufficient to resume across engagement. Existing A/B checkpoints must remain loadable in their original variant; cross-variant continuation is not a matched baseline and must not silently invent a slow rail.
- User-launched diagnostics should cap actual updates at 150 with `stop_after_step`, disable both evaluation paths and generation, and suppress the unconditional final checkpoint as well as periodic saves.
- Report optimizer-step CUDA-event duration including optimizer communication, and end-to-end global token throughput. Report compilation/startup separately from a stated steady-state window; use the same window and settings for all four variants. Keep instrumentation in a diagnostic runner where possible.
- Longer learning-quality runs need matched tokens, initialization/data seeds, and separate evaluation. A 150-update timing run cannot establish convergence quality.

The spectral checks cover the scalar envelope, BF16 rank-zero/rank-one/full-rank matrices, update-norm restoration, FP32 state bytes, and same-variant resume through a `.safetensors` round trip. Existing baseline reference-update checks remain in place. The real two-process Gloo bucket test also includes the spectral variant and compares updated parameters and owner-local momentum to local averaged-gradient updates. Exact reference and Gloo checks use `force_eager`; actual compiled execution is checked separately.

Validation: `uv run --locked --extra cpu --group dev pytest tests/test_optim.py tests/test_cli_training_config.py tests/test_distributed.py -k 'not two_process_cpu_mesh'` passed 46 tests (one unrelated mesh test deselected). `git diff --check` passed. A CPU smoke check executed the actual compiled full-graph A/B kernels for two updates with batched `2x4x2` matrices and FP32 parameters/state; both remained finite. Maximum parameter differences from eager were approximately `0.001752764` for A and `0.000598073` for B on that example. Compiler fusion changes BF16 intermediate rounding; these are not bitwise-reference results or performance measurements. GPU/NCCL execution and diagnostic timings remain unverified.
