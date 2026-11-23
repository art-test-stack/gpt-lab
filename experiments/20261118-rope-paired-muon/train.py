"""Train and evaluate the Q/K Muon arms."""

import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from time import perf_counter

from gpt_lab.cli.commands.base import register, resolve_checkpoint, run


ARMS = ("real", "rope", "shuffled", "joint-real", "joint-rope", "joint-shuffled")
STUDY_MODEL = "rope-muon-d16-study"


def main():
    worker_start = perf_counter()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", choices=ARMS, required=True)
    parser.add_argument("--benchmark-timing", action="store_true", help="Print synchronized timings per rank.")
    parser.add_argument("--guard-mfu", action="store_true", help="Require median MFU above 40%% per 10 updates after startup.")
    register(parser)
    args = parser.parse_args()
    pairing = args.pairing
    benchmark_timing = args.benchmark_timing
    guard_mfu = args.guard_mfu
    del args.pairing
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

        trainer_factory = None
        if args.model_name != STUDY_MODEL or benchmark_timing or guard_mfu:
            from trainer import BenchmarkTrainer, ExperimentTrainer

            trainer_factory = partial(ExperimentTrainer, guard_mfu=guard_mfu)
            if benchmark_timing:
                trainer_factory = partial(BenchmarkTrainer, worker_start=worker_start, guard_mfu=guard_mfu)

        if args.model_name == STUDY_MODEL and args.model_init == "auto":
            cap = args.stop_after_step
            run_name = f"{pairing}-s{args.random_seed}"
            if cap is not None:
                if cap < 0:
                    raise ValueError("--stop-after-step must be nonnegative")
                run_name += f"-stop{cap}"
            if args.run_name not in (None, run_name):
                raise ValueError(f"This study run must use the name {run_name!r}")
            args.run_name = run_name
            # W&B must also reject duplicates when checkpoints are stored elsewhere.
            if args.board == "wandb":
                os.environ["WANDB_RESUME"] = "never"
            if os.environ.get("RANK", "0") == "0":
                directory = Path(args.model_dir) / args.model_name / run_name
                try:
                    directory.mkdir(parents=True)
                except FileExistsError as exc:
                    raise ValueError(f"Run directory {directory} already exists. Use resume to continue it.") from exc

        from optimizer import configure_experiment_muon

        run(
            args,
            configure_optimizer=partial(configure_experiment_muon, pairing=pairing),
            trainer_factory=trainer_factory,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
