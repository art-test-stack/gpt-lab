# gpt-lab CLI

Use Python 3.12. Install with `uv sync --locked --extra cpu` (or `--extra gpu`
on compatible Linux hardware), then use `uv run --extra cpu gpt-lab ...`, keeping
the same extra. In an activated environment, use `gpt-lab ...` or
`python -m gpt_lab.cli ...`.

Paths are relative to the working directory. Repository examples such as
`configs/optim.yaml` are not bundled; use absolute paths outside the checkout.
Artifacts use `GPTLAB_CACHE_DIR` (default: `~/.cache/gpt_lab`). Cache paths expand
to absolute paths without resolving symlinks. Help uses only the standard library
and does not create directories or load runtime dependencies.

| Purpose | Command |
| --- | --- |
| Tokenization | `gpt-lab tokenizer train` |
| Training | `gpt-lab train base auto`, `gpt-lab train base resume` |
| Data preparation | `gpt-lab data reshard` |
| Cache management | `gpt-lab cache clear` |
| Inference | `gpt-lab chat console`, `gpt-lab chat app` |

Append `--help` for arguments and defaults. Groups require a subcommand;
invalid commands or arguments exit with status 2.
Standalone benchmarks and tokenizer scaling live in [experiments](../experiments/README.md).

## Tokenizer training

```bash
gpt-lab tokenizer train --trainer huggingface --vocab-size 32000 \
  --name ic1_tokenizer --corpus-path ./.data/corpus.txt
```

Accepts a UTF-8 file or `TokenizerCorpus` directory. Only `huggingface` training
works; select it explicitly because the legacy default is `tiktoken`. `--fast`
is unused. Saved tokenizers use `source="local"`.

`--max-chars` and `--chars-per-doc` limit training characters; `-1` is unlimited.
Add `--write-corpus` and a positive `--max-chars` to download and write a corpus
**directory**. Generation budgets up to four UTF-8 bytes per requested character;
training applies the exact character limits.

## Base training and resume

```bash
gpt-lab train base auto --model-name ic1 --run-name baseline --depth 12 \
  --tokenizer-model ic1_tokenizer --optim-config-path configs/optim.yaml --board dummy

gpt-lab train base resume --model-name ic1 --run-name baseline --board dummy

gpt-lab train base resume \
  --checkpoint-dir ~/.cache/gpt_lab/models/ic1/baseline/base/checkpoint_step_002000
```

Uses prepared local shards selected by `--ds-name` (default: `climbmix-base`).
Defaults include depth 12, context 2048, device batch 32, parameter:data target 11,
and automatic step/batch horizons. The default board is `wandb`; `--board dummy`
disables external board logging. Tokenizer resolution and evaluation may access
external data. `--fp8`, `--truncate-tokenizer`, and `--ds-config-path` are unused.

Resume restores model, trainer, optimizer, dataloader, RNG, scaler, and best-checkpoint
state. Saved trainer settings override CLI defaults. Omit `--run-name` for automatic
run discovery. `--checkpoint-step` accepts an exact step (including `0`), `-1`/`-2`
for latest/second-latest, or `latest`/`best`.

`--checkpoint-dir` accepts a run containing `meta.json`, its `base` directory, or
`checkpoint_step_N` under `<model-dir>/<model-name>/<run-name>/base/`. It overrides
model/run selection; a step directory selects that step and rejects a conflicting
`--checkpoint-step`.

## Distributed launch

```bash
uv run --extra gpu torchrun --standalone --nproc-per-node=8 -m gpt_lab.cli \
  train base auto --model-name ic1 --run-name baseline --depth 12

uv run --extra gpu torchrun --standalone --nproc-per-node=8 --no-python \
  gpt-lab train base resume --model-name ic1 --run-name baseline --checkpoint-step -1
```

Use module mode (`-m gpt_lab.cli`) or `--no-python` for the installed executable.
Launcher arguments precede the entry point; CLI arguments follow it. Multi-GPU
training uses CUDA/NCCL; CPU/MPS are single-device. Existing checkpoint world-size
restrictions apply. Tests mock distributed worker lifecycles.


## Data, cache, and chat

```bash
gpt-lab data reshard --ds-name climbmix --mode local --streaming --max-shards 2
gpt-lab cache clear --subdirs tokenizer
gpt-lab chat console --model-name ic1 --run-name baseline --max-tokens 64
gpt-lab chat app
```

- **Resharding:** defaults to `bucket`. Remote `bucket`/`dataset` modes need
  `--repo-id` and use `HF_TOKEN`. Existing shard names may be overwritten; local
  output is below `<cache>/data/<configured-output-dir>`.
- **Cache:** prompts unless `--force` is supplied. Empty/omitted `--subdirs` or
  `all` deletes the whole cache; `tokenizer` maps to `tokenizers`, `corpus` to
  `data/corpus`. `--cache-dir`/`--cache_dir` selects the root. Subdirectory symlinks
  are not followed; filesystem and home roots are rejected.
- **Console:** local checkpoint prompt loop with greedy generation and no
  instruction-model chat template.
- **App:** experimental Gradio UI; may query Hugging Face at startup. Remote
  completion needs `OPENAI_BASE_URL` and `HF_TOKEN`; local web completion is
  unimplemented. Also requires `openai` and `python-dotenv`, and uses component
  arguments incompatible with pinned Gradio 3.18.0. Browser operation is unverified.

## Compatibility and development

Legacy scripts delegate to the CLI, including `python -m scripts...` and
`torchrun -m scripts.train_base auto ...` from the checkout. Gradio script autoreload
is unsupported; use `gpt-lab chat app`. No `arch`, `cust`, SFT, or RL placeholders
are exposed.

Parser modules in `gpt_lab.cli.commands` register arguments and lazy `module:handler`
targets. Handlers accept `argparse.Namespace`; orchestration lives in
`gpt_lab.workflows`. Add training stages via a command module, workflow, entry in
`TRAINING_STAGES`, and parser/dispatch tests.

```bash
uv run --locked --extra cpu --group dev pytest -q tests/cli
uv run --locked --extra cpu --group dev pytest -q -m 'not slow'
```

CLI checks cover help, dispatch, wrappers, checkpoints, and cache safety.
Model training, UI, and external services
are mocked; the tokenizer save/load regression uses real tiny training.
