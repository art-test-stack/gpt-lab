"""Pinned recipe; reuse B's offline helpers without importing CUDA dependencies."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h100_fp8_diagnostic"))
from common import TRAIN_ARGS as BASE_TRAIN_ARGS, REPOSITORY, distribution, union_length
from worker import cpu_copy, tensor_hash, tensor_dtypes, write, hardware_report, restore_training_state

SHA = "cd7ca922d5d062a4b7f537c1d2e6e672b8a91514"
TOKENS = 524288
BATCH_SIZES = (16, 32)
ORDER = ["production-before", "cpu-cache", "gpu-cache", "production-after"]


def train_args(device_batch_size):
    args = [*BASE_TRAIN_ARGS, "--target-param-data-ratio", "8"]
    for flag,value in (("--device-batch-size",str(device_batch_size)),
                       ("--total-batch-size",str(TOKENS)), ("--n-acc-steps","-1")):
        args[args.index(flag)+1] = value
    return args


def batch_config(config, seq_len):
    resolved = dict(device_batch_size=config.device_batch_size, seq_len=seq_len,
                    world_size=config.comp_info.world_size, n_acc_steps=config.n_acc_steps,
                    total_batch_size=config.total_batch_size)
    assert resolved["n_acc_steps"] > 0
    assert (resolved["device_batch_size"] * resolved["seq_len"] * resolved["world_size"]
            * resolved["n_acc_steps"] == resolved["total_batch_size"])
    return resolved


def resolve_cpu_batch(device_batch_size):
    """Use the very same AutoConfig method training uses; never infer accumulation here."""
    import torch
    from gpt_lab.cli.parser import build_parser
    from gpt_lab.model.auto import AutoGPTConfig
    from gpt_lab.utils.distributed import ComputeContext
    args = build_parser().parse_args(train_args(device_batch_size))
    auto = AutoGPTConfig(run_name="input-preflight", device_batch_size=args.device_batch_size,
                         max_seq_len=args.max_seq_len, total_batch_size=args.total_batch_size,
                         n_acc_steps=args.n_acc_steps, comp_info=ComputeContext(torch.device("cpu")))
    auto.n_acc_steps = auto.compute_accumulation_steps(auto.total_batch_size)
    return batch_config(auto,auto.max_seq_len)


def combine_reports(reports):
    """Separate same-shape input interventions from same-path microbatch changes."""
    first,second = (reports[str(size)] for size in BATCH_SIZES)
    for key in ("initial_model_sha256","initial_optimizer_groups","model","parameter_storage_dtypes","first_row_sha256"):
        assert first[key] == second[key], f"Unmatched batch-size initialization: {key}"
    for key,value in first["trainer"].items():
        if key not in ("device_batch_size","n_acc_steps"):
            assert value == second["trainer"][key], f"Unmatched trainer setting: {key}"
    rows = []
    for size in BATCH_SIZES:
        report = reports[str(size)]
        assert report["complete"]
        for name in ORDER:
            leg = report["legs"][name]
            assert leg["schedules"] == first["legs"][name]["schedules"]
            rows.append(dict(device_batch_size=size,input_path=name,
                             n_acc_steps=report["resolved_batch"]["n_acc_steps"],
                             tokens_per_second=leg["global_tokens_per_second"],
                             median_update_ms=leg["timing"]["median_ms"],p95_update_ms=leg["timing"]["p95_ms"],
                             peak_allocated_bytes=leg["peak_allocated_bytes"],peak_reserved_bytes=leg["peak_reserved_bytes"]))
    return dict(configurations=rows, batch32_vs_batch16_same_path={
        name:dict(throughput_ratio=second["legs"][name]["global_tokens_per_second"]/first["legs"][name]["global_tokens_per_second"],
                  median_update_ratio=second["legs"][name]["timing"]["median_ms"]/first["legs"][name]["timing"]["median_ms"])
        for name in ORDER},
        interpretation="Within each batch size, compare production/CPU cache/GPU cache with repeated controls and trace evidence. Across batch sizes, same-path ratios measure the effect of larger microbatches and fewer accumulation iterations. Batch 32 being faster alone does not establish an input bottleneck; differences are not additive.")


def timing(seconds, tokens):
    result = distribution([value * 1000 for value in seconds])
    result["tokens_per_second"] = tokens / (result["mean_ms"] / 1000)
    return result
