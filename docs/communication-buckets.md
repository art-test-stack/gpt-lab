# Communication bucketing and overlap experiment

This note preserves the original implementation handoff. For measured Muon
64/128 MiB results and the new optimizer-selectable NorMuon sweep, use
[Optimizer-specific communication sweeps](communication-optimizer-sweeps.md).
The original launcher/source remains pinned in the original checkout for
same-SHA profiling; the new launcher lives on `exp/comm-optimizer-sweeps`.

Pinned starting code: `192bbe2921e4774d24324b2d07852311b7bc4dcf`.
The standalone performance handoff explicitly selects **Muon**, using
`configs/optim.yaml`; NorMuon is checked separately for compatibility. Being on
a branch containing NorMuon does not select that optimizer.
Experiment branch: `exp/comm-buckets`. The separate dirty `exp/anvil-normuon`
worktree remains unchanged. No paid compute has been launched.

## Existing communication

Every rank holds a complete model replica and computes local gradients. The
optimizer shards its state and update work; this is not parameter-sharded FSDP.
`model.build_optimizer()` creates groups in this order: embeddings, head, value
embeddings, residual hidden coefficients, residual input coefficients, then
transformer groups in sorted shape order. Parameter order within each shape
group follows the original module order. Empty groups remain in the list.

| Parameters | Gradient collective | State/update ownership | Parameter synchronization |
|---|---|---|---|
| AdamW, fewer than 1,024 elements | All-reduce in place | Full state and update replicated on every rank | None needed |
| AdamW, at least 1,024 elements | Reduce-scatter along dimension 0 | Rank `r` owns rows `[r*d0/W, (r+1)*d0/W)` | All-gather after owner update |
| NorMuon, every same-shaped matrix group, including small matrices | Owner-packed reduce-scatter | With `K=ceil(N/W)`, matrix index `i` belongs to rank `i//K`, state slot `i%K` | One full padded group all-gather after one batched owner update |

Large AdamW tensors currently require `shape[0] % W == 0`; the baseline does not
pad or repartition them. NorMuon matrix owners do not change with bucket size.
Muon/NorMuon momenta and row moments remain keyed under the original group's
first parameter, with leading dimension `K`, including state padding.

NCCL reductions use `ReduceOp.AVG`. Gloo uses `SUM`, then the consumer divides
once by `W` after waiting. There is no additional trainer gradient all-reduce.
The accumulation loop scales each microbatch loss by `1/n_acc_steps`, runs all
backward microsteps, then invokes the optimizer once and clears gradients.

The backend's ordering is fixed:

1. Launch all gradient reductions in group order and parameter order.
2. Visit groups in that same order. Wait for their reduction(s), perform the
   original optimizer update(s), and launch the corresponding parameter gather.
3. Wait for gathers in append order and copy NorMuon gathered matrices back into
   their original parameters. AdamW gathers directly into the full parameter.

All collectives currently use the default process group. The ordering must be
the same on every rank, including interactions with non-optimizer collectives.

## Existing bucketing and overlap

The whole NorMuon shape group is already a communication bucket. The private
`_reduce_bucket()` helper can pack a subset into original owner segments,
zero-pad unequal owner populations, and return original owner-local positions.
`dist_compute()` accepts a complete partition of results, validates coverage,
assembles original owner-local gradients, and updates/gathers the group once.
Production still calls that helper with the whole group. There is no byte limit,
gradient-readiness hook, or backward communication scheduler.

AdamW currently uses one gradient collective per active parameter; its small
all-reduces are not coalesced. Large tensor gradients are reduce-scattered
directly from the existing gradient storage. NorMuon freshly allocates packed
inputs and receive chunks, including intermediate `torch.stack()` copies.
Its full-group path reuses the packed input allocation as the gather output.

Async launches already permit some overlap: an early reduction may execute while
later gradients are packed, and later reductions may execute while an earlier
group is updated. Parameter gathers also remain in flight until the final
drain. This is existing overlap, not a new C improvement. All of it starts after
backward. Any new C result must be compared with A as well as with B.

## Other synchronization and collectives

- Fresh distributed initialization broadcasts each unique model parameter.
- The trainer has an initial training barrier and CUDA device synchronizations
  at the outer step boundary and the start/end of `train_step()`.
- Checking accumulated-loss finiteness and reading loss metrics synchronize
  device values with the host.
- FP16/scaler mode all-reduces overflow flags before an optimizer update. This
  communication is absent from the BF16/scaler-free experiment.
- Optional gradient monitoring all-reduces statistics, not training gradients.
- Per-step timing performs a scalar all-reduce with `MAX`. Evaluations have
  their own communication. These are separate from optimizer payload counts.

Do not replace these with gradient reductions, count them as optimizer payload,
or introduce a second all-reduce before the existing sharded reduce-scatter.

## Precision, ties, missing gradients, and resume

Buckets must not cross device, gradient dtype, collective kind, or ownership
boundaries. BF16 training does not make every gradient BF16: transformer weights
and the output head retain FP32 storage/gradients, while untied lookup embeddings
can use BF16. A head tied to lookup embeddings retains one FP32 parameter.
Do not cast payloads to increase bucket occupancy.

Parameter ties are object aliases. The model reestablishes them after device
moves and loads; `parameters()` deduplicates, and optimizer construction excludes
shared embedding/head/value-embedding parameters from duplicate groups. Build
the plan from the actual original optimizer parameters, deduplicating/checking
object identity, without changing their ownership or registering aliases twice.

AdamW skips `grad is None`; a globally absent gradient must not advance its step,
moments, or decay. Rank-dependent missing gradients in the pinned baseline risk
different collective schedules. NorMuon requires every grouped gradient and
errors when one is missing. Substituting zero and updating its momentum/decay
would change the baseline equations. Missing-gradient handling and any
cross-rank presence validation must therefore be explicit, not silently added
as padding. The experiment validates an identical initial active mask/layout
across ranks, excludes uniformly absent AdamW parameters, and preserves the
NorMuon missing-gradient error. It requires that active set remain fixed after
warmup; parameters may still contribute only on earlier accumulation microsteps.
One additional 4-byte `MAX` control all-reduce per update checks presence,
contiguity, and layout before any optimizer update. B/C check before launching
gradient collectives. D first flushes its canonical bucket suffix, padding an
invalid/missing piece with zeros only to drain the error path, then checks and
waits on all ranks before rejecting. It never substitutes zero into an optimizer
update for an absent gradient. Initial rank-dependent masks and post-warmup
rank-local disappearance are tested. Dynamic active-set changes and
noncontiguous custom gradient layouts are rejected, not silently reinterpreted.

Communication buffers are transient state and should be rebuilt outside capture
after a checkpoint load. Original optimizer state keys, parameter order, owners,
step counts, scheduler horizon, accumulated-gradient scaling, and data position
must remain the same. Rebuild any tensor views after a load replaces state.
Changing bucket sizes on same-topology resume must not change optimizer state.

## Static payload accounting

For `W` ranks, matrix size `M` elements, dtype size `q`, and `N` matrices in one
NorMuon group, the baseline packed reduction input and full gathered output each
contain `S = W * ceil(N/W) * M * q` bytes. Padding is
`(W*ceil(N/W)-N)*M*q`. A subset bucket with owner populations `n[r]` contains
`W * max(n[r]) * M * q` bytes, including padding. Whole-matrix preservation can
make even a singleton exceed a requested byte target.

For an AdamW tensor with `L` elements, gradient payload is `L*q`: its all-reduce
operates on that full tensor, or its reduce-scatter returns `L*q/W` bytes per
rank and parameter all-gather reconstructs `L*q` bytes. With every gradient
present, optimizer collective count is
`n_small_adam + 2*n_large_adam + 2*n_nonempty_normuon_groups`.

Payload bytes are not physical link traffic. Under a simple ring model, an
all-reduce sends approximately `2*(W-1)/W*S` bytes per rank; reduce-scatter and
all-gather each send approximately `(W-1)/W*S`. This excludes receives and
protocol overhead. Actual NCCL algorithms, topology, NVLink/NVSwitch/PCIe paths,
and transport counters must be reported separately when measured.

A metadata-only model inspection was run on CPU with no training or GPU work:
depth 12, aspect ratio 64, width 768, FFN width 3072, 12 layers, head dimension
128, vocabulary 50,304, context 2,048, and literal embedding setting
`embs;value_embs;head` (separate embedding/head parameters). It has 64 unique
parameter objects and 422,314,440 parameter elements. The table assumes every
parameter receives a gradient and the existing BF16 storage policy applies.

| World size | Optimizer collectives/update | Reduction input bytes/rank | Gather output bytes/rank | Reduction output bytes/rank |
|---:|---:|---:|---:|---:|
| 2 | 28 | 1,148,389,152 | 1,148,389,056 | 574,194,624 |
| 4 | 28 | 1,148,389,728 | 1,148,389,632 | 287,097,504 |
| 8 | 28 | 1,299,384,672 | 1,299,384,576 | 162,423,168 |

These are analytical sums over actual model metadata, not network measurements,
peak memory, or throughput. The layout was inspected from local model metadata; the temporary inspection
output is not a runtime dependency or part of the published handoff.

At eight ranks, the minimum packed singleton for the existing transformer
groups is already the following. These lower bounds assume the existing FP32
matrix gradients, not BF16 communication.

| Matrix shape | One matrix | Minimum nonempty owner-packed input |
|---|---:|---:|
| `768x768` | 2.25 MiB | 18 MiB |
| `768x3072` | 9 MiB | 72 MiB |
| `2304x768` | 6.75 MiB | 54 MiB |
| `6144x768` | 18 MiB | 144 MiB |

Thus a nominal 64 MiB target cannot bound every message while retaining complete
matrices. Also avoid filling buckets with contiguous indices from one owner
when other owners' matrices can fit the same bucket: that can multiply padding
and total payload. Bucket membership may interleave owners deterministically
while retaining every original parameter index, owner, and local state slot.

## Coordination and proposed first comparison

The ANVIL coordination review confirmed that transport must pass exactly the
same averaged matrices into the owner-local kernel and retain group identity.
The separate ANVIL experiment's optional spectral kernel argument is not part
of this pinned branch and must not be merged into the communication ablation.

Agent 3 agreed a first boundary with host-driven communication outside capture.
Read-only inspection of the active `exp/cuda-graphs` worktree confirms its
`CapturedForwardBackward` captures only a fixed-shape forward/backward
microbatch, replays once per accumulation microstep, and leaves the optimizer
outside capture. It preserves gradient addresses using `set_to_none=False`.
Communication buffers may be persistent without requiring `.grad` to alias them.
Do not edit that worktree or add communication hooks inside its captured region.

Implemented comparisons, as selected by the user:

| Variant | Runner mode | Reduction timing |
|---|---|---|
| A | `baseline` | Exact original optimizer communication path |
| B | `buckets` | Pack, launch, wait and unpack one bucket at a time, after accumulation |
| C | `pipeline` | Launch reductions while packing later buckets, then wait and unpack |
| D | `backward` | Launch the same canonical bucket sequence from final-microstep leaf-readiness hooks; flush any suffix after backward |

Call `optimizer.configure_communication(mode, bucket_bytes, timing=False)` to
select the transport. The trainer signals final accumulation microsteps through
`begin_backward_communication()`. The existing strategies still execute each
parameter/group update exactly once; they consume completed reduced gradients
instead of launching a second reduction. Small replicated AdamW all-reduces
are coalesced only with compatible dtype/ownership entries. Large AdamW tensors
may be fragmented into owner-aligned pieces; their original row ownership is
unchanged. NorMuon matrices remain whole and owner packed. Buckets can combine
different original groups without changing optimizer groups or update equations.

Before optional gradient-statistics collectives, the trainer calls
`finish_gradient_communication()` to drain the canonical schedule. The later
optimizer step reuses those results without repeating reductions or the safety
check. Statistics still read original local gradients, as in the baseline.
The real Gloo test inserts an independent collective between drain and update
and checks that only one validation reduction occurs per completed update.

B/C/D share fixed reverse group order and owner-interleaved whole matrices.
The plan is identical for all three modes at the same target size. Each bucket
has its own persistent packed and received storage; each original group has
persistent reduced-gradient/gather storage. This prevents an in-flight collective
from reading a buffer being repacked. The cap is a soft message target,
including padding, not a limit on total resident communication memory. All
persistent allocations, including the validation scalar, are counted in results.

D uses a first non-overlapped step to prepare/validate the layout before hooks
launch reductions. It is rejected with FP16 scaling or forward/backward CUDA
capture. Its first version is an eager host hook scheduler around the compiled
training path; it does not attempt to capture or replay Python hooks/NCCL calls.
B/C preserve Agent 3's optimizer-outside-capture boundary. Tensor address
stability is tested, but combined graph replay and NCCL capture are not verified.

Do not claim useful overlap merely because `async_op=True` is set. A CUDA/NCCL
timeline must show independent packing/compute executing during communication,
and measured exposed wait must include the required stream dependency.

The agreed dependency chain is: accumulated backward produces gradients; pack
waits for the producer stream; NCCL waits for pack writes; owner update waits
for reduction completion; gather waits for owner update; parameter copyback and
the next forward wait for gather completion. Keep Work objects alive until
their waits, and keep each input/output buffer immutable while its operation
uses it. CUDA `Work.wait()` establishes a stream dependency; its host return is
not proof of device completion. Graph segments require fixed tensor/scalar
addresses, shapes, and stream dependencies. Enclosing NCCL capture/replay is a
separate GPU verification, not established by this inspection.

## Measurement requirements for subsequent steps

Sweep explicitly labeled MiB targets rather than assuming 64 MiB is best. The
target must account for packed reduce-scatter padding and oversized whole-matrix
exceptions. Log useful bytes, padded bytes, and actual bucket sizes for every
target. Bound/report total persistent buffer memory separately from message size.

Report collective count and type, API payload sizes, any topology-qualified
traffic estimate, pack/unpack CUDA duration and host launch overhead, optimizer
duration, exposed communication wait on the GPU critical path, peak allocated
and reserved memory, and end-to-end global tokens/second. Exclude compile/warmup
from a stated steady-state window but report startup separately. Avoid inserting
per-collective device synchronizations into the timed throughput comparison;
use an instrumented/profile run separately from throughput runs.

Correctness checks must compare reduced gradients and parameter/state updates
against A, with accumulation, uneven ownership and padding, dtype boundaries,
ties, supported missing-gradient cases, and same-topology checkpoint/resume.
CPU/Gloo verifies layout and state transitions; it cannot prove NCCL averaging,
stream overlap, capture/replay, or GPU performance. Supported-GPU NCCL commands
follow below. No paid GPU job has been launched by the agent.

## Reference investigation

The X page was not readable through the browser. Its linked historical
[May 24 FasterReduce source](https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/records/track_1_short/2025-05-24_FasterReduce/23f40b75-06fb-4c3f-87a8-743524769a35.txt)
does expose persistent flat buffers, async all-reduce, later waits/unpacking,
and a literal `64.0 * 1024**2` bucket setting (despite an inconsistent adjacent
comment). It is an all-reduce design, not a replacement for this optimizer's
sharded gradient path. Its hyperparameter-setting work between launch and wait
also does not prove significant GPU overlap. This is a historical example,
not a size sweep or evidence that 64 MiB is optimal for our setup.

## Checks performed and remaining hardware checks

Real two-process CPU/Gloo checks cover B/C/D at two byte targets, averaged
gradients and baseline parameter/moment equivalence, two-microstep accumulation,
a parameter active only on an earlier microstep, tied contributions to one
leaf, BF16 versus FP32 boundaries, padding, oversized whole matrices, stable
communication-buffer pointers, absent AdamW gradients, missing NorMuon gradients,
and rank-local gradient disappearance after warmup. They also round-trip the
optimizer through `.safetensors` and change bucket size on same-topology resume.
The trainer accumulation-boundary test and native diagnostic CLI argument test
are included. No transport buffers are added to optimizer checkpoints.

The final handoff checks passed 11 focused CPU tests, with one NCCL test skipped
because CUDA GPUs are unavailable. An additional explicit Muon/Gloo smoke run
passed, matching parameters, gradients, and optimizer state exactly for all
three candidate modes over four updates and two accumulation microsteps.
The full Gloo check uses NorMuon and also tests disk resume with a
changed bucket target. The Modal smoke checks use in-memory state restore and
write only JSON, XML, logs, and traces; they never write model/optimizer
checkpoints. Tiny correctness uses eager optimizer execution to avoid compiling
many small fixture shapes; performance runs use the existing compiled kernels.
At the original handoff no GPU result was available. Subsequent user-launched
Muon 64/128 MiB sweeps have measured NCCL correctness, peak memory and throughput;
see the linked results report. Optimizer-time/packing/wait profiles and actual
overlap remain unmeasured. CPU/MPS checks cannot establish those results.

The first user-launched Modal correctness run stopped during pytest setup:
the fresh image lacked the parent of the explicit `--basetemp` directory.
The launcher now creates that parent before invoking pytest; a local regression
checks this from an empty temporary repository. No NCCL comparison ran in that
failed attempt. Retry with a fresh run name, such as `comm-nccl-20261031-r2`.

## User-launched Modal handoff

The commands and 35-update screens in this section describe the **original
`exp/comm-buckets` checkout at `8956f6e2cb8aae856c467c40ff2e3f37824dff75`**.
Use the linked optimizer-specific report for the new branch's explicit optimizer
selection, 55-update screens, 1% decision rule, and NorMuon commands.

All source and runners are tracked relative to the experiment worktree's Git
root. `scripts/benchmark/modal_communication.py` verifies a clean checkout, the
full SHA, its published `origin/exp/comm-buckets` tip, and required tracked files
before any Modal call. The remote image fetches that exact SHA. `--dry-run`
performs source verification without importing Modal or submitting work.

Prerequisites: an authenticated Modal account, existing `gpt-lab-cache` and
`gpt-lab-runs` volumes, secret `gpt-lab` with `HF_TOKEN`, and the existing cached
`climbmix-base` parquet shards and GPT-2 tokenizer. CPU preflight rejects absent
or noncontiguous cached data before allocating the GPU worker. These commands
launch paid resources **only when the user runs them**. Run from the clean
`exp/comm-buckets` Git root; choose fresh run names if these names already exist.

```bash
SHA=$(git rev-parse HEAD)

# Tiny two-rank NCCL: A versus B/C/D, four updates, two microsteps/update,
# parameter/gradient/optimizer-state comparisons, padding/ties, in-memory resume.
uv run --no-project --python 3.12 --with modal==1.6.2 python \
  scripts/benchmark/modal_communication.py correctness \
  --sha "$SHA" --run-name comm-nccl-20261031

# One 64 MiB packed-byte target initially, four A100-80GB GPUs.
uv run --no-project --python 3.12 --with modal==1.6.2 python \
  scripts/benchmark/modal_communication.py sweep \
  --sha "$SHA" --run-name comm-a100-20261031 --bucket-mib 64
```

The sweep first gates training on an actual four-rank NCCL correctness check.
It screens A `baseline`, B `buckets` (immediate wait), C `pipeline`
(post-backward launch/packing overlap), and D `backward` (readiness hooks):
35 updates each, excluding the first 25 for compilation/warmup. It chooses the
fastest **candidate B/C/D** by global steady wall tokens/s, runs the small NorMuon
compatibility check for that mode, then runs **150 updates each for A and the
candidate**. A candidate is still compared against A if all candidates screened
slower. One 64 MiB soft target includes reduce-scatter padding; whole-matrix
exceptions and total persistent bytes are reported. This initial screen does
not identify an optimal bucket size. Repeat with a fresh run name and another
`--bucket-mib` only after inspecting this handoff's results.

Every timed candidate runs in a fresh process with seed 42 and the original data
start. The launcher verifies initial model and optimizer SHA-256 fingerprints,
resolved optimizer groups, first prepared batch hash and position, and every
consumed update's data position against A (matching prefixes for 35 versus 150
updates). No model checkpoint files are needed for these resets. All candidates
use SwiGLU, BF16 compute with existing parameter storage precision, explicit
Muon `configs/optim.yaml`, FA2 routing on A100, microbatch **16/GPU**, sequence
**2048**, and global batch **524,288**: four accumulation microsteps at four
ranks. The scheduler still uses horizon **2317**, LR warmup **300**, warmdown
0.65 and final LR fraction 0.05; the short stop cap does not rescale it.

Evaluation, sampling, periodic saves, initial saves and unconditional final
saves are disabled. `Trainer.save_checkpoint` and the workflow's separate
training-configuration checkpoint writer are replaced with no-ops before
trainer construction. Captured forward/backward and compiler CUDA graphs are
disabled for every candidate, including D. Ordinary compiled model and optimizer
execution remains enabled. Resolved optimizer groups are logged explicitly.

After the sweep completes, these independent commands repeat the tiny NorMuon
check on two GPUs and collect profiler traces on four GPUs. The sweep already
includes its own winning-mode NorMuon check; this repeat is optional. Profiling
uses the completed sweep's winner and bucket target, and never enters the timed
comparison or selects the winner.

```bash
uv run --no-project --python 3.12 --with modal==1.6.2 python \
  scripts/benchmark/modal_communication.py compatibility \
  --sha "$SHA" --run-name comm-normuon-20261031 \
  --results-run comm-a100-20261031

uv run --no-project --python 3.12 --with modal==1.6.2 python \
  scripts/benchmark/modal_communication.py profile \
  --sha "$SHA" --run-name comm-profile-20261031 \
  --results-run comm-a100-20261031
```

`gpt-lab-runs/<run-name>/manifest.json` is atomically persisted after every
candidate, with per-candidate `communication-results.json`, `console.log`, exact
command, and per-rank correctness JSON. Correctness errors include maximum
absolute parameter, gradient, and optimizer-state differences by mode. Any
failed assertion, skipped NCCL check, nonzero child exit, source/reset mismatch,
or invalid timing aborts the sweep. NCCL process-group deadlines are 60 seconds
for tiny checks and 300 seconds for training; a whole candidate has an 1800
second deadline (600 for tiny checks), after which its process tree is killed.
Failures and available logs are committed to the results volume. Modal retries
and torchrun restarts are disabled.

Results report global steady wall tokens/s, native synchronized train-step
tokens/s, nearest-rank median/p95 step milliseconds, per-rank peak allocated and
reserved memory after warmup, resident memory at the warmup boundary, optimizer
collective counts/bytes/padding, total persistent buffer bytes, and startup time.
Process startup through the first completed update includes initialization,
hashing, compilation, and prefetch; the narrower train-loop startup is also
reported. Modal queueing and image building are outside those timers. Optimizer
CUDA time, pack/unpack cost and exposed communication wait are measured in the
separate profile stage, with per-rank traces; they are not overhead-free numbers
from the uninstrumented sweep. Retrieve a manifest, for example:

```bash
uv run --no-project --python 3.12 --with modal==1.6.2 modal volume get \
  gpt-lab-runs comm-a100-20261031/manifest.json ./comm-a100-manifest.json
```

## Interpreting the measurements

- Collective counts and payload bytes refer to optimizer collectives. B/C/D
  report one extra 4-byte validation reduction separately from gradient traffic.
  Initial object-layout exchanges and the trainer's existing metric collectives
  are outside the steady optimizer payload count. A's payloads are inferred from
  its observed first-update active set, assumed fixed; verify launches in its trace.
- `pack_ms` and `unpack_ms` measure bucket-copy phases on the current CUDA
  stream. Profile annotations use `gradient_communication::*`. A reports
  `pack_and_launch_stream_ms` and host launch time, also encompassing its
  allocation/enqueue phase; these are not isolated copy-bandwidth measurements.
  Bucket modes also record host packing/unpacking and collective enqueue time.
- Gradient `Work.wait()` intervals measure exposed current-stream dependencies,
  not total network duration. In D, validation queues behind gradient reductions,
  so its interval includes remaining gradient communication plus validation cost.
  Interpret it together with gradient-wait time. Gather wait/copy includes
  parameter copyback. Use the NCCL timeline to separate kernels, copies and
  concurrent useful work.
- Optimizer CUDA events time the post-backward optimizer phase. D's hook packing
  and communication started earlier and must not be treated as free. Their
  complete cost is included in end-to-end throughput and the packing trace.
  Compiled autograd may deliver leaf gradients together near the end of a
  compiled backward region. Hooks firing is therefore not evidence of a useful
  overlap window; D needs a GPU timeline showing concurrent remaining work.
- Report both native synchronized train-step throughput and steady wall-step
  throughput including metrics/logging/Python between updates. Total training-wall
  throughput includes startup/compilation; time to first update is labeled
  separately and is not a pure compilation measurement.
- Peak allocated/reserved GPU memory resets after warmup, with resident memory
  at that boundary also recorded. Persistent bytes count communication tensor
  storage without double-counting all-reduce input/output aliases. Do not
  mistake the message cap for total memory or the planned ring-send estimate
  for measured physical NVLink/PCIe traffic.

The 150-update runs are systems diagnostics, not optimizer learning-quality
evidence. Longer matched-token runs and separate evaluation are needed for
learning conclusions, with equations, precision and data settings still fixed.
