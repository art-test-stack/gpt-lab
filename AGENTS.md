# Repository Guidelines

## Project Scope

These instructions apply to `https://github.com/art-test-stack/gpt-lab`.

gpt-lab is an educational PyTorch project for implementing, training, and evaluating small language models and running ablation studies. Prefer transparent implementations that make the underlying mechanisms understandable. Keep claims about supported scale and hardware consistent with the implementation.

## Project Structure

Production code lives in `src/gpt_lab/`. Core areas include `model/`, `data/`, `tokenizer/`, `optim/`, `train/`, and `evaluate/`. CLI code lives in `cli/`; application and interface code lives in `app/` and `interface/`. Shared helpers belong in `utils/`.

Use `configs/` for experiment configuration, `scripts/` for runnable training, evaluation, and benchmark entry points, `tests/` for tests, and `docs/` for technical notes. Follow the current checkout's structure when it differs from this overview.

## Environment and Development Commands

Use `pyproject.toml`, `.python-version`, and `uv.lock` as the source of truth for Python and dependencies. The current project targets Python 3.12.

- `uv sync --locked --extra cpu --group dev` installs CPU dependencies and development tools.
- `uv sync --locked --extra gpu --group dev` installs the CUDA configuration and development tools on compatible Linux hardware.
- `uv run --locked --extra cpu --group dev pytest tests/<test_file>.py` runs a focused test module with the CPU environment.

The `cpu` and `gpu` extras are mutually exclusive. Use the extra appropriate to the current machine and keep it consistent across sync and run commands. Inspect pytest markers and individual tests before selecting broader checks; some tests require a GPU or take substantial time.

## Development Principles

- Keep implementations simple, explicit, and suitable for controlled experiments.
- Match nearby Python style and existing configuration conventions.
- Keep changes to architecture, data processing, optimization, and training behavior scoped and documented.
- Avoid adding abstractions or dependencies without a concrete requirement.
- Credit adapted implementations and preserve applicable license notices.

## Correctness and Experimentation

For changes to tokenization, packing, attention masks, loss computation, or evaluation, verify the affected semantics with a small reproducible example or focused test.

For changes to distributed training, gradient accumulation, checkpointing, or dataloader resume behavior, verify the affected state transitions and reductions. Document any changes to checkpoint compatibility or reproducibility.

Performance claims must come from measurements. State the relevant model, sequence length, batch size, precision, device, and world size. Distinguish measured throughput or memory use from estimates such as FLOPs and MFU.

Use small models and data samples for correctness checks. Keep long training runs and hardware benchmarks separate from routine tests.

## Documentation and Pull Requests

Update the relevant documentation, configuration examples, or CLI help when supported behavior changes. Keep commits focused and use descriptive subjects.

Pull requests should explain the concrete problem, the resulting behavior, and the checks performed. Report unavailable hardware checks as not run rather than implying they passed.

Do not commit credentials, downloaded datasets, model checkpoints, local caches, or generated training artifacts.
