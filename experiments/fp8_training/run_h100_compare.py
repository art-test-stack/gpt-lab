"""Run the real training pipeline twice: one H100 BF16, then one H100 FP8.

Run via uv with the Modal extra tool dependency, as shown in README.md.
Each launch blocks until completion; the second starts only if the first succeeds.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def commands(prefix):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", prefix):
        raise ValueError("run-prefix must contain 1–80 letters, digits, underscores or hyphens")
    common = [sys.executable, str(ROOT/"runs/modal_trainbase.py"),
              "--gpu", "H100", "--gpus", "1", "--dtype", "bfloat16"]
    training = ["--act-func", "swiglu", "--max-seq-len", "2048",
                "--device-batch-size", "16", "--total-batch-size", "524288",
                "--n-acc-steps", "16", "--num-steps", "2317",
                "--warmup-steps", "300", "--stop-after-step", "10"]
    return [
        [*common, "--run-name", prefix+"-bf16", "--", *training],
        [*common, "--run-name", prefix+"-fp8-forward", "--", *training, "--fp8-ffn-forward"],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", default="", help="Run names receive -bf16 and -fp8-forward suffixes; defaults to a UTC timestamp")
    parser.add_argument("--dry-run", action="store_true", help="Print both commands without contacting Git or Modal")
    args = parser.parse_args(argv)
    prefix = args.run_prefix or datetime.now(timezone.utc).strftime("fp8-1h100-%Y%m%d-%H%M%S")
    try:
        plan = commands(prefix)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(plan, indent=2), flush=True)
    if not args.dry_run:
        for command in plan:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
