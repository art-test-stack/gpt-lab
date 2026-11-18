"""Train the per-head real and paired Q/K Muon arms with dashboard diagnostics."""

from argparse import ArgumentParser
from functools import partial
from time import perf_counter

from gpt_lab.cli.commands.base import register, run


def main():
    worker_start = perf_counter()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", choices=("real", "rope", "shuffled"), required=True)
    parser.add_argument("--benchmark-timing", action="store_true", help="Print synchronized timings per rank.")
    register(parser)
    args = parser.parse_args()
    pairing = args.pairing
    benchmark_timing = args.benchmark_timing
    del args.pairing
    del args.benchmark_timing
    if args.model_init == "auto":
        if args.muon_qkv not in (None, "head"):
            parser.error("the experiment requires --muon-qkv head")
        args.muon_qkv = "head"

    from optimizer import configure_experiment_muon
    from trainer import BenchmarkTrainer, ExperimentTrainer

    trainer_factory = partial(BenchmarkTrainer, worker_start=worker_start) if benchmark_timing else ExperimentTrainer
    try:
        run(
            args,
            configure_optimizer=partial(configure_experiment_muon, pairing=pairing),
            trainer_factory=trainer_factory,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
