"""Generate independent run commands; scheduler submission is intentionally separate."""
import argparse
import itertools
from pathlib import Path
import shlex

from experiments.value_vocab.run import SHARING, parser as run_parser, positive_int


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--vocab-sizes", type=positive_int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--sharing", choices=list(SHARING), nargs="+", default=list(SHARING))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--result-dir", type=Path, default=Path("experiments/results/value_vocab"))
    p.add_argument("run_args", nargs=argparse.REMAINDER, help="common runner arguments after --")
    return p


def commands(args):
    common = args.run_args[1:] if args.run_args[:1] == ["--"] else args.run_args
    reserved = {"--vocab-size", "--sharing", "--seed", "--output"}
    if any(arg.split("=", 1)[0] in reserved for arg in common):
        raise ValueError("set sweep axes and --result-dir on the submitter, not after --")
    run_parser().parse_args(common)
    axes = (args.vocab_sizes, args.sharing, args.seeds)
    if any(len(axis) != len(set(axis)) for axis in axes):
        raise ValueError("sweep axes must not contain duplicates")
    for vocab, sharing, seed in itertools.product(*axes):
        name = f"v{vocab}-{sharing}-seed{seed}"
        output = args.result_dir / f"{name}.jsonl"
        # The truncation API writes config/token-byte caches; jobs must not share them.
        cache = args.result_dir / ".tokenizer-cache" / name
        yield ["env", f"GPTLAB_CACHE_DIR={cache}", "python", "-m", "experiments.value_vocab.run", *common,
               "--vocab-size", str(vocab), "--sharing", sharing, "--seed", str(seed),
               "--output", str(output)]


def main():
    for command in commands(parser().parse_args()):
        print(shlex.join(command))


if __name__ == "__main__":
    main()
