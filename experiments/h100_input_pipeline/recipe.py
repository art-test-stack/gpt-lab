"""Pinned recipe; reuse B's offline helpers without importing CUDA dependencies."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h100_fp8_diagnostic"))
from common import TRAIN_ARGS, REPOSITORY, distribution, union_length
from worker import cpu_copy, tensor_hash, tensor_dtypes, write, hardware_report, restore_training_state

SHA = "cd7ca922d5d062a4b7f537c1d2e6e672b8a91514"
TRAIN_ARGS = [*TRAIN_ARGS, "--target-param-data-ratio", "8"]
TOKENS = 524288
ORDER = ["production-before", "cpu-cache", "gpu-cache", "production-after"]


def timing(seconds, tokens):
    result = distribution([value * 1000 for value in seconds])
    result["tokens_per_second"] = tokens / (result["mean_ms"] / 1000)
    return result
