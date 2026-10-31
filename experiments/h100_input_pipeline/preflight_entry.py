"""Run the CPU preflight in the locked project interpreter, before GPU allocation."""
import argparse
from pathlib import Path

from cpu import preflight
from recipe import write, BATCH_SIZES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-batch-sizes",type=int,nargs="+",choices=BATCH_SIZES,default=BATCH_SIZES)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--measured", type=int, required=True)
    args = parser.parse_args()
    report = preflight(args.run_name,args.warmup,args.measured,args.device_batch_sizes)
    root = Path("/runs")/args.run_name
    root.mkdir(exist_ok=False)
    write(root/"cpu-preflight.json",report)
