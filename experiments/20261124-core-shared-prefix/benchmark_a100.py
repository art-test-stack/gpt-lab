"""Matched question batching on one A100 using BF16 PyTorch SDPA."""
import argparse
import io
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import torch

from gpt_lab.evaluate import core
from gpt_lab.model import checkpoint
from gpt_lab.utils.distributed import ComputeContext
from gpt_lab.utils.paths import TOKENIZERS_FOLDER


CONTEXT = 2048
SAMPLE = 128
SIZES = (1, 4, 8, 16)
WIDTHS = (512, 1024, 2048, 2160)
NLL_ATOL = 0.05
ARMS = ("separate", "packed")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_cached_tokenizer(config, build):
    directory = os.getenv("GPTLAB_BENCH_TOKENIZER_DIR")
    if config.source == "local":
        path = Path(directory) if directory else TOKENIZERS_FOLDER / config.dirname.name
        config = config.model_copy(update={"dirname": path})
    elif directory:
        if config.source != "huggingface":
            raise ValueError("A tokenizer directory requires local BPE or Hugging Face tokenizer metadata")
        from gpt_lab.tokenizer.hf import HuggingFaceTokenizerWrapper

        return HuggingFaceTokenizerWrapper.from_directory(directory, special_tokens=config.special_tokens)
    return build(config)


def load_checkpoint_model(args, device):
    load, build = checkpoint.load_checkpoint, checkpoint.build_tokenizer

    def weights_only(*args, **kwargs):
        return load(*args, **(kwargs | {"load_rng": False, "load_trainer": False}))

    # Cross-device evaluation needs neither optimizer nor trainer RNG state.
    with patch.multiple(checkpoint, load_checkpoint=weights_only, set_rng_state=lambda _: None,
                        build_tokenizer=lambda config: build_cached_tokenizer(config, build)):
        model, tokenizer, _, _ = checkpoint.build_model(
            args.model_name, run_name=args.run_name, model_cachedir=args.model_dir, step=args.step,
            phase="eval", comp_info=ComputeContext(device, compute_dtype=torch.bfloat16), attn_impl="sdpa",
        )
    if model.max_seq_len != CONTEXT or model.config.positional_encoding != "rope":
        raise ValueError("Use a trained RoPE checkpoint with a 2,048-token context")
    return model, tokenizer


def choose_examples(tokenizer):
    from batching import Example
    from scoring import pack_choices

    path, tasks, baselines = core.load_core_bundle()
    tasks = [task for task in tasks if task["icl_task_type"] == "multiple_choice"]
    # Index refers to the position in CORE's seeded shuffle for this task.
    sources = [(task["label"], *core.load_core_task(path, task, -1)) for task in tasks]
    pools = {label: [] for label, _, _ in sources}
    checked = 0

    def add_candidates(start, stop):
        nonlocal checked
        for label, data, meta in sources:
            # Preserve the PR sample's few-shot pool for its first 16 examples.
            fewshot_pool = data[:16] if start == 0 else data
            for index in range(start, min(stop, len(data))):
                checked += 1
                try:
                    prepared = core.prepare_example(index, tokenizer, fewshot_pool, meta, CONTEXT)
                except AssertionError as error:
                    if str(error) != "this should never happen right?":
                        raise
                    continue  # Cropping passed the start of an answer span.
                packed = pack_choices(*prepared)
                if packed is not None:
                    pools[label].append(Example(label, index, "multiple_choice", data[index]["gold"], prepared, packed))

    add_candidates(0, 16)
    assert checked == 176, "CORE primary sample changed"
    limit = 16
    while sum(map(len, pools.values())) < SAMPLE:
        if all(len(data) <= limit for _, data, _ in sources):
            raise ValueError(f"Only {sum(map(len, pools.values()))} eligible questions after checking {checked}")
        add_candidates(limit, 2 * limit)
        limit *= 2

    quotas = dict.fromkeys(pools, 0)
    remaining = SAMPLE
    while remaining:
        for label, pool in pools.items():
            if remaining and quotas[label] < len(pool):
                quotas[label] += 1
                remaining -= 1

    selected = []
    for label, pool in pools.items():
        pool.sort(key=lambda example: (max(map(len, example.prepared[0])), example.index))
        count = quotas[label]
        selected.extend(pool[(2 * i + 1) * len(pool) // (2 * count)] for i in range(count))
    selected.sort(key=lambda example: (len(example.packed.tokens), example.label, example.index))
    assert len({(example.label, example.index) for example in selected}) == SAMPLE
    assert all(max(ex.packed.positions) < CONTEXT and max(map(len, ex.prepared[0])) <= CONTEXT for ex in selected)
    return selected, baselines, {"checked": checked, "eligible_by_task": {k: len(v) for k, v in pools.items()}}


def master_reference(args, device):
    if not Path(core.__file__).is_relative_to(Path(os.environ["PYTHONPATH"])):
        raise RuntimeError("Pinned-master worker imported an unexpected CORE module")
    model, tokenizer = load_checkpoint_model(args, device)
    questions = json.loads(args.master_input.read_text())
    pad = tokenizer.get_bos_token_id()
    predictions = []
    with torch.autocast(device.type, dtype=torch.bfloat16):
        for question in questions:
            tokens, starts, ends = question["prepared"]
            rows = core.stack_sequences(tokens, pad).to(device)
            losses, _ = core.forward_model(model, rows)
            nll = [losses[i, start - 1:end - 1].mean().item()
                   for i, (start, end) in enumerate(zip(starts, ends))]
            predictions.append({"task": question["task"], "index": question["index"],
                                "winner": nll.index(min(nll)), "option_nll": nll})
    write_json(args.output / "master-reference.json", predictions)


def get_master_reference(args, examples, master):
    repo = Path(__file__).resolve().parents[2]
    archive = subprocess.check_output(["git", "archive", master, "src"], cwd=repo)
    with TemporaryDirectory(prefix="core-a100-master-") as temporary:
        work = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(work, filter="data")
        inputs = work / "prepared.json"
        write_json(inputs, [{"task": ex.label, "index": ex.index, "prepared": ex.prepared} for ex in examples])
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--master-input", str(inputs)],
            env=os.environ | {"PYTHONPATH": str(work / "src")}, check=True,
        )
    return {(item["task"], item["index"]): (item["option_nll"], item["winner"])
            for item in json.loads((args.output / "master-reference.json").read_text())}


def task_scores(scores, examples, baselines):
    correct, counts = {}, {}
    for ex in examples:
        correct[ex.label] = correct.get(ex.label, 0) + int(scores[ex.label, ex.index][1] == ex.gold)
        counts[ex.label] = counts.get(ex.label, 0) + 1
    report = {}
    for task, count in counts.items():
        accuracy, baseline = correct[task] / count, baselines[task] / 100
        report[task] = {"questions": count, "accuracy": accuracy, "centered": (accuracy - baseline) / (1 - baseline)}
    return report


def validate(reference, scores, examples, baselines, expected_tasks):
    from benchmark import compare

    result = compare(reference, scores)
    result["task_scores"] = task_scores(scores, examples, baselines)
    result["task_scores_exact"] = result["task_scores"] == expected_tasks
    result["passed"] = not (result["outside_tolerance"] or result["paired_nan"] or result["flips"]
                            or not result["task_scores_exact"] or result["max_nll_abs"] > NLL_ATOL)
    return result


def profile_shapes(plan, examples, run):
    from packing_pair import bucket_width

    distinct = {}
    for packed, group in plan:
        if packed:
            rows, longest = len(group), max(len(examples[i].packed.tokens) for i in group)
            width = bucket_width(longest, WIDTHS) or longest
        else:
            lengths = [len(row) for i in group for row in examples[i].prepared[0]]
            rows, longest = len(lengths), max(lengths)
            width = min(bucket_width(longest, WIDTHS) or longest, CONTEXT)
        distinct.setdefault((rows, width), (packed, group))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as trace:
        run(list(distinct.values()))
        torch.cuda.synchronize()
    operators, kernels = set(), set()
    for event in trace.events():
        if event.name.startswith("aten::_scaled_dot_product_"):
            operators.add(event.name)
            # The math fallback's GPU kernels belong to its child operators.
            children = [event]
            while children:
                child = children.pop()
                kernels.update(kernel.name for kernel in child.kernels)
                children.extend(child.cpu_children)
    if not operators or not kernels:
        raise RuntimeError("PyTorch profiler did not capture CUDA SDPA operators and their GPU kernels")
    return {"shapes": list(distinct), "sdpa_operators": sorted(operators), "cuda_kernel_symbols": sorted(kernels)}


def attempt(run, record, phase):
    try:
        return run()
    except torch.cuda.OutOfMemoryError as error:
        record["allocation_error"] = {"phase": phase, "message": str(error)}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return None


def summarize_trial(run):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    run()
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    return {"seconds": seconds, "questions_per_second": SAMPLE / seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def benchmark(args, device):
    from batching import MASTER
    from batched_scoring import check_wide_fixture
    from fixture_model import small_model
    from packing_pair import comparison_plans, score_plan

    metadata = checkpoint.load_meta_config(args.model_name, run_name=args.run_name, model_cachedir=args.model_dir)
    tokenizer = build_cached_tokenizer(metadata.tokenizer_cfg, checkpoint.build_tokenizer)
    if tokenizer.vocab_size != metadata.model_cfg.vocab_size:
        raise ValueError("Checkpoint and tokenizer vocabulary sizes differ")
    examples, baselines, selection = choose_examples(tokenizer)
    configs = {size: comparison_plans(examples, size)["eligible"] for size in SIZES}
    assert all(len(group) == size for size, arms in configs.items() for plan in arms.values() for _, group in plan)
    write_json(args.output / "selected.json", {
        **selection, "tokenizer": tokenizer.config.model_dump(mode="json"),
        "questions": [{"task": ex.label, "index": ex.index, "gold": ex.gold,
                       "packed_width": len(ex.packed.tokens), "max_position": max(ex.packed.positions)} for ex in examples],
        "groups_by_size": {size: [group for _, group in arms["separate"]] for size, arms in configs.items()},
    })
    reference = get_master_reference(args, examples, MASTER)
    assert len(reference) == SAMPLE and all(math.isfinite(nll) for losses, _ in reference.values() for nll in losses)
    task_reference = task_scores(reference, examples, baselines)
    model, tokenizer = load_checkpoint_model(args, device)
    pad = tokenizer.get_bos_token_id()
    results = {
        "master": MASTER, "torch": torch.__version__, "cuda": torch.version.cuda,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": torch.cuda.get_device_name(), "dtype": "bfloat16", "prefetch": False, "max_context": CONTEXT,
        "model_config": model.config.model_dump(mode="json"),
        "model": {"dir": str(args.model_dir), "name": args.model_name, "run": args.run_name, "step": args.step},
        "width_buckets": WIDTHS, "questions": SAMPLE, "option_nll_atol": NLL_ATOL,
        "reference_task_scores": task_reference, "sizes": {},
    }

    def run(plan):
        return score_plan(plan, examples, model, pad, device, width_buckets=WIDTHS)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        results["isolation_fixture"] = check_wide_fixture(small_model(device, head_dim=64), 256, device)
        write_json(args.output / "preflight.json", results)
        for size, arms in configs.items():
            print(f"Checking batch size {size} against pinned master", flush=True)
            record = {arm: {"forwards": len(arms[arm]), "trials": []} for arm in ARMS}
            for arm, plan in arms.items():
                scores = attempt(lambda: run(plan), record[arm], "parity_warmup")
                if scores is not None:
                    record[arm]["parity"] = validate(reference, scores, examples, baselines, task_reference)
            results["sizes"][size] = record
            write_json(args.output / "preflight.json", results)
            if any(not details["parity"]["passed"] for details in record.values() if "parity" in details):
                raise SystemExit(f"Pinned-master parity failed at size {size}; see preflight.json")

        for size, arms in configs.items():
            print(f"Timing batch size {size}", flush=True)
            record = results["sizes"][size]
            for arm, plan in arms.items():
                if "allocation_error" not in record[arm]:
                    kernels = attempt(lambda: profile_shapes(plan, examples, run), record[arm], "profile_warmup")
                    if kernels is not None:
                        record[arm].update(kernels)
            for i in range(3):
                for arm in (ARMS if i % 2 == 0 else ARMS[::-1]):
                    if "allocation_error" in record[arm]:
                        continue
                    trial = attempt(lambda: summarize_trial(lambda: run(arms[arm])), record[arm], "timing")
                    if trial is not None:
                        record[arm]["trials"].append(trial)
            for arm in ARMS:
                trials = record[arm]["trials"]
                if len(trials) == 3:
                    seconds = median(trial["seconds"] for trial in trials)
                    qps = SAMPLE / seconds
                    record[arm].update(median_seconds=seconds, questions_per_second=qps,
                                       peak_allocated_bytes=max(trial["peak_allocated_bytes"] for trial in trials))
                    baseline = results["sizes"][1][arm].get("questions_per_second")
                    record[arm]["batching_speedup_vs_size_1"] = qps / baseline if baseline is not None else None
            if all(len(record[arm]["trials"]) == 3 for arm in ARMS):
                record["paired_packing_speedup"] = median(
                    a["seconds"] / b["seconds"] for a, b in zip(record["separate"]["trials"], record["packed"]["trials"])
                )
                record["separate_median_over_packed_median"] = (
                    record["separate"]["median_seconds"] / record["packed"]["median_seconds"]
                )
            write_json(args.output / f"batch-{size}.json", record)
            write_json(args.output / "report.json", results)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-input", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.step < 0:
        parser.error("an explicit nonnegative checkpoint step is required")
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        parser.error("exactly one CUDA A100 is required")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    if args.master_input:
        master_reference(args, device)
    else:
        benchmark(args, device)


if __name__ == "__main__":
    main()
