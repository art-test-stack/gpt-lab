"""gpt-lab CLI: tokenizer training, model training, data, cache, and chat.

Use Python 3.12 and install with ``uv sync --locked --extra cpu`` (or ``gpu``
on compatible Linux hardware). Keep the same extra when running commands.
An activated environment supports ``gpt-lab`` or ``python -m gpt_lab.cli``.
Paths are relative to the working directory; repository configs are not bundled.
Artifacts use GPTLAB_CACHE_DIR, defaulting to ~/.cache/gpt_lab. Imports and help
perform no application setup; help loads no runtime dependencies.

Commands (append --help for arguments and defaults):
    gpt-lab tokenizer train --trainer huggingface --corpus-path ./.data/corpus.txt
    gpt-lab train --model-name ic1 --depth 12 --board dummy
    gpt-lab train resume --model-name ic1 --run-name baseline
    gpt-lab data reshard --ds-name climbmix --mode local --max-shards 2
    gpt-lab cache clear --subdirs tokenizer
    gpt-lab chat console --model-name ic1 --run-name baseline
    gpt-lab chat app

Tokenizer training accepts a UTF-8 file or TokenizerCorpus directory. Only
huggingface works; select it explicitly instead of the legacy tiktoken default.
--max-chars/--chars-per-doc limit characters (-1 means unlimited). --write-corpus
requires positive --max-chars and writes a directory; saved tokenizers are local.

Base training requires prepared local shards (--ds-name defaults to climbmix-base).
Defaults: depth 12, context 2048, device batch 32, data-to-parameter ratio 11,
and automatic batch/step horizons. --board dummy disables the default wandb board.
Tokenizer resolution and evaluation may access external data.
No auto subcommand is required. Explicit vocabulary, batch, step, and accumulation
values replace inference. Optimizer values come from the bundled base YAML;
--optim-config-path supplies file overrides before scaling. Learning-rate and
weight-decay CLI overrides are not supported. Resume accepts checkpoint/runtime
options, not architecture flags, and rejects conflicting optimizer profiles.

Python entry point (file loading stays outside the workflow):
    from gpt_lab.utils.schemas import OptimizerConfig
    from gpt_lab.workflows.train_base import train_base
    train_base(depth=12, optim_config=OptimizerConfig.from_file("custom.yaml"))

Tying defaults off: --tie-embeddings 'embs;h_embs;head'. Commas share tables;
semicolons separate groups. 'embs,head' ties input/head; 'h_embs:' shares values;
'embs,head;h_embs:' creates two shared tables; 'embs,head,h_embs:' shares all.
Bare h_embs alone keeps values separate; in a comma group it ties every value
table to the other named table, just like h_embs:. Empty semicolon groups are
ignored; omitted names remain separate. Colon slicing is not supported yet.
Tying values to input/head requires n_kv_heads * d_head == d_model.
Shared tables use input settings when embs is included, otherwise head settings
when head is included. Storage is counted once; output FLOPs still include it.
Resume retains saved tying; legacy no-op tying flags load as disabled.

Resume restores model, trainer, optimizer, dataloader, RNG, scaler, and best state.
Saved settings take precedence; omit --run-name to discover the latest run.
--checkpoint-step accepts an exact step, -1/-2 (latest/second latest), latest, or
best. --checkpoint-dir accepts a run, base directory, or checkpoint_step_N;
it overrides model/run selection and rejects conflicting step arguments.
Changing tying modes on resume is not a checkpoint conversion.

Distributed CUDA/NCCL launch (CPU/MPS remain single-device):
    uv run --extra gpu torchrun --standalone --nproc-per-node=8 -m gpt_lab.cli \\
      train --model-name ic1 --depth 12
Use module mode or --no-python for the installed executable. Launcher arguments
precede the entry point; CLI arguments follow it. Checkpoint world-size limits apply.

Resharding defaults to bucket; remote bucket/dataset modes need --repo-id and
HF_TOKEN. Shard names may be overwritten; local output is below the data cache.
Cache clearing prompts unless --force is set. Omitted/empty --subdirs or all
deletes the whole cache; tokenizer maps to tokenizers, corpus to data/corpus.
--cache-dir selects the root; home/filesystem roots and subdirectory symlinks
are protected. Console chat is greedy local generation without a chat template.
The experimental Gradio app may access Hugging Face; remote completion needs
OPENAI_BASE_URL and HF_TOKEN. Local web completion is unimplemented. It also needs
openai/python-dotenv and uses arguments incompatible with pinned Gradio 3.18.0.

Unused flags: tokenizer --fast; training --fp8, --truncate-tokenizer,
--ds-config-path. Legacy scripts delegate to the CLI; app autoreload is unsupported.
Benchmarks and tokenizer scaling live in experiments/. Parser command modules
register lazy module:handler targets; workflows handle orchestration. Add commands
with parser/dispatch tests. Invalid arguments exit 2.
CLI checks: uv run --locked --extra cpu --group dev pytest -q tests/cli
Training, UI, and external services are mocked; tiny tokenizer save/load is real.
"""

from collections.abc import Sequence
from importlib import import_module


def main(argv: Sequence[str] | None = None) -> int:
    """Parse explicit arguments (or sys.argv) and run exactly one workflow."""
    from .parser import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    module_name, handler_name = args._handler.split(":")
    handler = getattr(import_module(module_name), handler_name)
    try:
        return handler(args) or 0
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
