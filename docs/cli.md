# gpt-lab CLI

Install the package with Python 3.12 (`uv sync --extra cpu` or `uv sync --extra gpu`),
then use `uv run gpt-lab ...`. With an activated environment, use `gpt-lab ...`
directly. `python -m gpt_lab.cli ...` is equivalent and works outside the checkout.
Paths supplied on the command line are relative to the working directory.
`configs/data.yaml` and `configs/optim.yaml` remain repository examples, not bundled
configuration files; pass absolute paths when running elsewhere.

| Purpose | Commands |
| --- | --- |
| Tokenization | `gpt-lab tokenizer train` |
| Training | `gpt-lab train base auto`, `gpt-lab train base resume` |
| Benchmarking | `gpt-lab benchmark dataloaders` |
| Experiments | `gpt-lab experiment tokenizer-scaling` |
| Data preparation | `gpt-lab data reshard` |
| Cache management | `gpt-lab cache clear` |
| Inference | `gpt-lab chat console`, `gpt-lab chat app` |

All groups require a subcommand. Missing/unknown subcommands and malformed arguments
exit with status 2. Help and parser construction use only the standard library;
they do not import Torch, initialize distributed groups, inspect checkpoints,
create cache directories, load datasets, or import Gradio.

```bash
gpt-lab --help
gpt-lab tokenizer train --help
gpt-lab benchmark dataloaders --help
gpt-lab experiment tokenizer-scaling --help
gpt-lab train base auto --help
gpt-lab train base resume --help
```

## Common workflows

Train on an existing UTF-8 text file, using the currently implemented backend:

```bash
gpt-lab tokenizer train --trainer huggingface --vocab-size 32000 \
  --name ic1_tokenizer --corpus-path ./.data/corpus.txt
```

A corpus directory produced by `TokenizerCorpus` is also accepted. Add
`--write-corpus --max-chars 1000000` to sample the default source mixture into that
path **as a directory**. This downloads source data and writes a corpus. The current
source writer budgets UTF-8 bytes: generation samples up to four bytes per requested
character; training applies the exact `--max-chars` and `--chars-per-doc` character
limits. `-1` leaves existing-corpus training unlimited. New corpus generation
requires a positive `--max-chars`, because the old unlimited source-writing path
could not safely bound streaming sources. Model/tokenizer artifacts use
`GPTLAB_CACHE_DIR`, defaulting to `~/.cache/gpt_lab`.

The legacy tokenizer default remains `--trainer tiktoken` and all six trainer
choices remain parseable. The current library only implements **huggingface**
training; other choices fail with an explicit explanation before loading data.
`--fast` remains a documented, unused legacy flag.

A small offline CPU benchmark:

```bash
gpt-lab benchmark dataloaders --mode synthetic --loader all \
  --batch-size 2 --seq-len 128 --num-batches 5 --num-docs 100 --buffer-size 32
```

Both hyphenated and original underscore spellings work, such as `--batch-size`
and `--batch_size`. `glab` selects both gpt-lab variants; `all` also runs nanochat.
The tokenized variant uses the production `DistDataLoader`. The on-the-fly fixture
parses integer text, so its timing **does not measure natural-language BPE**.
The nanochat variant uses the production best-fit loader on temporary local
Parquet files. Initialization and two warm-up batches are excluded from timing.
`--buffer-size` is a token budget for gpt-lab and a document count for nanochat.
Uninstrumented crop/search metrics report `N/A`.

Real mode takes `--bin tokens.bin --idx tokens.idx`: raw uint32 token IDs and uint64
document offsets, including a leading zero and the final token boundary. It does
not download a tokenizer or dataset. The original advertised `--parquet_dir` and
`--loader packed` options never existed in the parser and are not supported.

An experimental tokenizer sweep (downloads training/evaluation data and baselines):

```bash
gpt-lab experiment tokenizer-scaling --num-procs 2 \
  --vocab-sizes 20000,50000 --pat-strs gpt2,cl100k_base \
  --corpus-sizes-gb 0.1,0.5 --write-corpus --result-dir ./scaling-results
```

Sizes are GiB, as in the old implementation (`1024**3` bytes). A new sweep requires
`--corpus-sizes-gb`; no automatic size range was implemented. To continue, use
`--resume --result-dir ./scaling-results`; saved metadata supplies the sweep
configuration and completed `.pkl` results are skipped. `--corpus-dir` overrides
`<cache>/data/corpus/<result-directory-name>`. Without `--resume`, an existing result
directory is preserved under a numbered sibling. `--compare-truncated-baselines`
now uses the existing comparison helper. The `--board` option remains unused.
See [the experimental methodology and its limitations](tokenizer_scaling.md).

Base training on prepared local `climbmix-base` shards:

```bash
gpt-lab train base auto --model-name ic1 --run-name baseline --depth 12 \
  --tokenizer-model gpt2 --optim-config-path configs/optim.yaml --board dummy
```

This is an expensive training job. `auto` retains the original `AutoGPTConfig`
scaling rules and `TrainerConfig` settings, including depth 12, context 2048,
device batch 32, parameter:data target 11, and automatic step/batch horizons.
The default metrics board remains `wandb`; `--board dummy` disables external board
logging. Tokenizer resolution and evaluation can access external data when run.
The old `--fp8`, `--truncate-tokenizer`, and `--ds-config-path` arguments were not
wired into this training script; they remain parseable and explicitly documented
as unused. Dataset selection still uses `--ds-name` and prepared local shards.

Resume with the latest checkpoint in a specific run:

```bash
gpt-lab train base resume --model-name ic1 --run-name baseline --board dummy
```

Omit `--run-name` to retain automatic run discovery. Use `--checkpoint-step 2000`
for an exact step, `0` for step zero, `-1`/`-2` for the latest/second-latest saved
step, or `latest`/`best`. Selection is delegated to `CheckpointManager`.
Saved model, trainer, optimizer, dataloader, checkpoint-best, RNG and scaler states
retain the existing restore path. Saved trainer settings take precedence over
common CLI training defaults during resume.

An explicit directory also works:

```bash
gpt-lab train base resume \
  --checkpoint-dir ~/.cache/gpt_lab/models/ic1/baseline/base/checkpoint_step_002000
```

It must identify a run containing `meta.json`, its `base` directory, or a
`checkpoint_step_N` directory in the normal layout
`<model-dir>/<model-name>/<run-name>/base/checkpoint_step_N`.
It overrides `--model-dir`, `--model-name`, and `--run-name`. A step directory also
selects that exact step; a conflicting `--checkpoint-step` is rejected. The old
`--checkpoint-dir` was parsed but ignored; this is now wired. `--checkpoint-path`
was mentioned in old help but was never an accepted option.

## Distributed launch

Use torchrun's **module mode**, with launcher arguments before `-m` and all CLI
arguments after `gpt_lab.cli`:

```bash
torchrun --standalone --nproc_per_node=8 -m gpt_lab.cli \
  train base auto --model-name ic1 --run-name baseline --depth 12

torchrun --standalone --nproc_per_node=8 -m gpt_lab.cli \
  train base resume --model-name ic1 --run-name baseline --checkpoint-step -1
```

With uv, prefix these commands with `uv run`. Each worker reaches the same package
entry point; Torch supplies `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` through the
environment. Existing CUDA/NCCL setup and per-rank checkpoint behavior are
preserved. CPU/MPS are single-device workflows; this change does not add distributed
CPU training or alter checkpoint world-size restrictions. Distributed cleanup and
board closure now also happen when training raises an exception. Tests parse the
torchrun form and mock the training lifecycle; they do not start worker groups.

## Data, cache, and chat

```bash
gpt-lab data reshard --ds-name climbmix --mode local --streaming --max-shards 2
gpt-lab cache clear --subdirs tokenizer
gpt-lab chat console --model-name ic1 --run-name baseline --max-tokens 64
gpt-lab chat app
```

Resharding defaults to `bucket` mode. Remote `bucket`/`dataset` modes require
`--repo-id` and use `HF_TOKEN`; existing output shard names may be overwritten.
Local output is below `<cache>/data/<configured-output-dir>`.

Cache deletion prompts unless `--force` is supplied. Omitted/empty `--subdirs`,
or `all`, clears the whole cache. `tokenizer` maps to `tokenizers`, and `corpus`
maps to `data/corpus`. Both `--cache-dir` and old `--cache_dir` are accepted.
Subdirectory symlinks are not followed when deleting, and filesystem/home roots
are rejected.

The console is a local checkpoint prompt loop using the existing greedy generation
engine, without an instruction-model chat template. The old console file was only
an import stub. The Gradio app delegates to the existing experimental UI: it can
query Hugging Face when launched, remote completion needs `OPENAI_BASE_URL` and
`HF_TOKEN`, and local web completion is still unimplemented. It additionally needs
`openai` and `python-dotenv`; the incorrect `load_dotenv` module import was fixed.
The existing UI uses newer Gradio component arguments than the project's pinned
Gradio 3.18.0 supports. UI modernization is outside this CLI migration; the app
was tested through a mocked launcher, not a browser session.

## Compatibility and extension

All eight user-facing scripts remain thin wrappers. Both direct execution with
an installed package and the old `python -m scripts...` forms delegate to the same
CLI parser and handlers. `torchrun -m scripts.train_base auto ...` also continues
to work from the checkout. The old `gradio scripts/chat_app.py --demo-name=app` autoreload form is no longer
supported: the wrapper no longer creates a module-global UI during import. Use
`gpt-lab chat app` for explicit startup.

The old nonfunctional `arch`/`cust` training placeholders
are removed, and no SFT/RL placeholder is exposed.

The complete [pre-migration script inventory](cli-migration.md) records original
arguments, defaults, requirements, side effects, and necessary fixes.

Parser-only modules live under `gpt_lab.cli.commands`. Each module implements
`register(subparsers)` and stores a lazy `module:handler` target. Handlers accept
an `argparse.Namespace` and can be called programmatically; heavier orchestration
lives in `gpt_lab.workflows`. Application config objects are built only at execution.
The root parser does not know the names of training stages.

To add a real `train sft` or `train rl` implementation:

1. Add a lightweight command module with `register(stages)`; register its own
   arguments/subcommands and set `_handler` to an importable handler.
2. Put its configuration construction and execution in an importable workflow.
3. Add the command module to `TRAINING_STAGES` in `cli/commands/training.py`.
4. Add parser/dispatch tests. No root parser or base-training-loop rewrite is needed.

## Validation

```bash
python -m pytest -q tests/cli
python -m pytest -q -m 'not slow'
```

The focused suite covers all help/dispatch paths, legacy defaults and wrappers,
configuration forwarding, checkpoint paths and restored states, cache safety,
experiment resume/result helpers, a tiny offline CPU benchmark, torchrun parsing,
and a `python -S -m gpt_lab.cli` smoke test outside the checkout. Training, UI, and
external services are mocked. No new lint/format tool is introduced.
