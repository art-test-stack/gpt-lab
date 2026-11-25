"""Train and evaluate the Q/K Muon arms."""

import json
import os
import re
import shlex
from argparse import ArgumentParser, BooleanOptionalAction
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from time import perf_counter

from filelock import FileLock, Timeout
from safetensors import SafetensorError, safe_open

from gpt_lab.cli.commands.base import register, resolve_checkpoint, run


ARMS = ("real", "rope", "shuffled", "joint-real", "joint-rope", "joint-shuffled")
STUDY_MODEL = "rope-muon-d16-study"
STUDY_DATA_SEED = 42


def _complete_checkpoint_file(path):
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            if path.name == "meta.json":
                if not isinstance(data, str):
                    return False
                data = json.loads(data)
            return isinstance(data, dict)
        # Validate the header and file length without loading tensors.
        with safe_open(path, framework="pt", device="cpu"):
            return True
    except (OSError, ValueError, SafetensorError):
        return False


def _resumable_checkpoint(directory):
    source = directory / "base"
    required = (directory / "meta.json", source / "training_config.safetensors")
    if not all(map(_complete_checkpoint_file, required)):
        return None
    run_state = source / "checkpoint_state.safetensors"
    if run_state.exists() and not _complete_checkpoint_file(run_state):
        return None
    for checkpoint in sorted(source.glob("checkpoint_step_*"), reverse=True):
        step = checkpoint.name.removeprefix("checkpoint_step_")
        if not step.isdecimal() or checkpoint.name != f"checkpoint_step_{int(step):06d}":
            continue
        try:
            layout = json.loads((checkpoint / "checkpoint_meta.json").read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(layout, dict):
            continue
        world_size = layout.get("world_size")
        if type(world_size) is not int or world_size < 1:
            continue
        if layout.get("optimizer_state_layout") == "full" and world_size == 1:
            rank_files = ("optim.safetensors",)
        elif layout.get("optimizer_state_layout") == "sharded":
            rank_files = (
                f"{stem}_rank{rank}.{extension}"
                for rank in range(world_size)
                for stem, extension in (("optim", "safetensors"), ("trainer_state", "json"), ("rng_state", "safetensors"))
            )
        else:
            continue
        shared_files = ("model.safetensors", "trainer_state.json", "rng_state.safetensors")
        step_state = checkpoint / "checkpoint_state.safetensors"
        if step_state.exists() and not _complete_checkpoint_file(step_state):
            continue
        if all(_complete_checkpoint_file(checkpoint / name) for name in shared_files) and all(
            _complete_checkpoint_file(checkpoint / name) for name in rank_files
        ):
            return checkpoint
    return None


@contextmanager
def _reserve_run(directory, board):
    from trainer import STUDY_EVAL_FILE

    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError(f"Run path {directory} already exists.")
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".launch.lock"
    try:
        reservation = FileLock(lock_path).acquire(timeout=0)
    except Timeout as exc:
        raise ValueError(f"A launch is already active for {directory}.") from exc
    with reservation:
        source = directory / "base"
        metadata = directory / "meta.json"
        setup = directory / ".setup-board"
        evaluation = directory / STUDY_EVAL_FILE
        training_config = source / "training_config.safetensors"
        other_artifacts = set(directory.iterdir()) - {lock_path, metadata, source, setup, evaluation}
        checkpoint_artifacts = source.is_symlink() or (source.exists() and (
            not source.is_dir() or bool(set(source.iterdir()) - {training_config})
        ))
        # Older launchers wrote the training config just before opening W&B.
        may_have_wandb = training_config.is_file() and (
            not setup.is_file() or (setup.read_text() == "wandb" and board != "wandb")
        )
        if other_artifacts or checkpoint_artifacts or metadata.is_dir() or training_config.is_dir() or may_have_wandb:
            checkpoint = _resumable_checkpoint(directory)
            hint = f" Use resume --checkpoint-dir {shlex.quote(str(checkpoint))}." if checkpoint else ""
            raise ValueError(f"Run directory {directory} already contains run artifacts.{hint}")
        training_config.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
        evaluation.unlink(missing_ok=True)
        setup.write_text(board)
        yield


def _start_wandb_trainer(directory, factory, **kwargs):
    if not kwargs["board"].main.disabled:
        (directory / ".wandb-run").touch()
    if factory is None:
        from gpt_lab.train.trainer import Trainer

        factory = Trainer
    return factory(**kwargs)


def main():
    worker_start = perf_counter()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", choices=ARMS, required=True)
    parser.add_argument("--data-seed", type=int, help="Training shard seed for a fresh study run (default: 42).")
    parser.add_argument(
        "--study-eval", action="store_true",
        help="Save the study schedule for this run. Resume reads the saved schedule automatically.",
    )
    parser.add_argument(
        "--monitor-qk-norms", action=BooleanOptionalAction, default=True,
        help="Log sampled Q/K update norms at the first update, every 10 updates, and the endpoint (default).",
    )
    parser.add_argument("--benchmark-timing", action="store_true", help="Print synchronized timings per rank.")
    parser.add_argument("--guard-mfu", action="store_true", help="Require median MFU above 40%% per 10 updates after startup.")
    register(parser).set_defaults(warmup_steps=130)
    args = parser.parse_args()
    pairing = args.pairing
    data_seed = args.data_seed
    study_eval = args.study_eval
    monitor_qk_norms = args.monitor_qk_norms
    benchmark_timing = args.benchmark_timing
    guard_mfu = args.guard_mfu
    del args.pairing
    del args.data_seed
    del args.study_eval
    del args.monitor_qk_norms
    del args.benchmark_timing
    del args.guard_mfu
    if args.model_init == "auto":
        if args.muon_qkv not in (None, "head"):
            parser.error("the experiment requires --muon-qkv head")
        args.muon_qkv = "head"

    try:
        if getattr(args, "checkpoint_dir", None) is not None:
            args = resolve_checkpoint(args)
            args.checkpoint_dir = None

        if study_eval and args.model_name != STUDY_MODEL:
            raise ValueError(f"--study-eval requires --model-name {STUDY_MODEL}")
        study_model = args.model_name == STUDY_MODEL
        if data_seed is not None and (not study_model or args.model_init != "auto"):
            raise ValueError(f"--data-seed is for fresh runs with --model-name {STUDY_MODEL}")
        if study_model:
            args.data_seed = STUDY_DATA_SEED if data_seed is None else data_seed
            if os.environ.setdefault("GPTLAB_DTYPE", "bfloat16") != "bfloat16" or getattr(args, "fp8_ffn_forward", False):
                raise ValueError("The study requires BF16 mixed precision with FP8 disabled")
            if args.use_nanochat_dataloader:
                raise ValueError("The study requires the native validation loader")
            if args.model_init == "auto":
                if getattr(args, "attn_impl", "fa3") != "fa3":
                    raise ValueError("The study requires --attn-impl fa3")
                args.attn_impl = "fa3"
            os.environ["DEVELOPMENT"] = "0"

        trainer_factory = None
        if study_model or monitor_qk_norms or benchmark_timing or guard_mfu:
            from trainer import BenchmarkTrainer, ExperimentTrainer

            trainer = ExperimentTrainer
            if benchmark_timing:
                trainer = partial(BenchmarkTrainer, worker_start=worker_start)
            trainer_factory = partial(
                trainer, study_eval=study_eval if study_model else None,
                monitor_qk_norms=monitor_qk_norms, guard_mfu=guard_mfu,
            )

        reservation = nullcontext()
        if study_model and args.model_init == "auto":
            cap = args.stop_after_step
            run_name = f"{pairing}-s{args.random_seed}"
            if cap is not None:
                if cap < 0:
                    raise ValueError("--stop-after-step must be nonnegative")
                run_name += f"-stop{cap}"
            if args.run_name is not None:
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name):
                    raise ValueError(
                        "--run-name must start with a letter or digit and contain only "
                        "letters, digits, dots, underscores, or hyphens"
                    )
                run_name = args.run_name
            args.run_name = run_name
            # W&B must also reject duplicates when checkpoints are stored elsewhere.
            if args.board == "wandb":
                os.environ["WANDB_RESUME"] = "never"
            if os.environ.get("RANK", "0") == "0":
                directory = Path(args.model_dir) / args.model_name / run_name
                reservation = _reserve_run(directory, args.board)
                if args.board == "wandb":
                    trainer_factory = partial(_start_wandb_trainer, directory, trainer_factory)

        from optimizer import configure_experiment_muon

        with reservation:
            run(
                args,
                configure_optimizer=partial(configure_experiment_muon, pairing=pairing, monitor_qk_norms=monitor_qk_norms),
                trainer_factory=trainer_factory,
            )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
