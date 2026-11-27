"""Run the opt-in batched CORE scorer or compare its returned results with stock master."""
import argparse
from contextlib import nullcontext, redirect_stdout
from functools import partial
import hashlib
import io
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from fixture_model import ByteTokenizer, load_frozen_checkpoint


HERE = Path(__file__).resolve().parent
CONFIGS = {"current": (False, False), "prefetch": (True, False),
           "deferred": (False, True), "both": (True, True)}


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(run, device):
    synchronize(device)
    start = perf_counter()
    result = run()
    synchronize(device)
    return perf_counter() - start, result


def planned_winners(plan, examples, model, pad, device, *, prefetch, defer, width_buckets=(), fa4=None):
    from batch_timing import batches
    from packing_pair import collate_group, score_group

    prepare = partial(collate_group, examples=examples, pad_id=pad, max_context=model.max_seq_len,
                      width_buckets=width_buckets)
    output, pending, indices = {}, [], []
    for packed, group, batch in batches(plan, prepare, prefetch):
        _, winners = score_group(packed, batch, model, device, fa4=fa4)
        if defer:
            indices.extend(group)
            pending.append(winners)
        else:
            output.update(zip(group, winners.cpu().tolist()))
    if pending:
        output.update(zip(indices, torch.cat(pending).cpu().tolist()))
    return output


def score_task(model, tokenizer, data, device, meta, *, batch_size, prefetch, defer, records=None, fa4=None):
    from gpt_lab.evaluate import core
    from batching import Example
    from packing_pair import comparison_plans
    from scoring import pack_choices

    if meta["task_type"] != "multiple_choice":
        return core.evaluate_task(model, tokenizer, data, device, meta)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        raise ValueError("The experimental batched CORE entry point requires one rank")
    examples = []
    for index, item in enumerate(data):
        prepared = core.prepare_example(index, tokenizer, data, meta, model.max_seq_len)
        examples.append(Example(meta["dataset_uri"], index, "multiple_choice", item["gold"],
                                prepared, pack_choices(*prepared)))
    plan = comparison_plans(examples, batch_size)["all"]["packed"]
    predictions = planned_winners(plan, examples, model, tokenizer.get_bos_token_id(), device,
                                  prefetch=prefetch, defer=defer, fa4=fa4)
    correct = 0
    for ex in examples:
        winner = predictions[ex.index]
        correct += winner == ex.gold
        if records is not None:
            records.append({"dataset": ex.label, "index": ex.index, "task_type": ex.task_type,
                            "winner": winner, "correct": winner == ex.gold, "prepared": ex.prepared})
    return correct / len(examples)


def evaluate(model, tokenizer, device, *, batch_size=4, configuration="both", records=None, fa4=None):
    from gpt_lab.evaluate import core

    prefetch, defer = CONFIGS[configuration]
    return core.evaluate_core(
        model, tokenizer, device, max_per_task=16,
        score_task=partial(score_task, batch_size=batch_size, prefetch=prefetch, defer=defer,
                           records=records, fa4=fa4),
    )


def record_predictions(model, tokenizer, device, *, candidate=None):
    from pinned_master import record_core

    records, results = record_core(model, tokenizer, device, evaluate=candidate, capture_logits=False)
    return [{key: record[key] for key in ("dataset", "index", "winner", "correct", "prepared")}
            | ({"predictions": [row.tolist() for row in record["predictions"]]}
               if record["task_type"] == "language_modeling" else {}) for record in records], results


def stock_worker(model, tokenizer, device, output):
    from gpt_lab.evaluate import core

    if not Path(core.__file__).is_relative_to(Path(os.environ["PYTHONPATH"])):
        raise RuntimeError("Stock worker imported CORE outside the archived revision")
    with redirect_stdout(sys.stderr):
        records, _ = record_predictions(model, tokenizer, device)
        (output / "stock-predictions.json").write_text(json.dumps(records))
        core.evaluate_core(model, tokenizer, device, max_per_task=16)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        if line.strip() == "run":
            with redirect_stdout(sys.stderr):
                elapsed, results = timed(partial(core.evaluate_core, model, tokenizer, device, max_per_task=16), device)
            print(json.dumps({"seconds": elapsed, "scores": results}), flush=True)


def compare_returned(stock, candidate):
    assert stock["all_core_results"].keys() == candidate["all_core_results"].keys()
    return {
        "accuracy": candidate["core/accuracy"] - stock["core/accuracy"],
        "centered_core": candidate["core/core"] - stock["core/core"],
        "tasks": {
            task: {key: result[key] - stock["all_core_results"][task][key]
                   for key in ("accuracy", "centered_result")}
            for task, result in candidate["all_core_results"].items()
        },
    }


def compare_predictions(stock, candidate):
    keys = lambda record: (record["dataset"], record["index"])
    stock, candidate = sorted(stock, key=keys), sorted(candidate, key=keys)
    assert len(stock) == len(candidate) == 352
    changes = []
    for ref, actual in zip(stock, candidate):
        assert keys(ref) == keys(actual) and tuple(ref["prepared"]) == tuple(actual["prepared"]), keys(ref)
        if any(ref.get(key) != actual.get(key) for key in ("winner", "predictions", "correct")):
            changes.append({"dataset": ref["dataset"], "index": ref["index"],
                            "stock": ref.get("winner"), "candidate": actual.get("winner"),
                            "correctness_changed": ref["correct"] != actual["correct"]})
    return changes


def paired(args, model, tokenizer, device, *, extra_worker_args=(), fa4=None):
    from batching import MASTER, read_worker

    candidate = partial(evaluate, model, tokenizer, device, configuration=args.configuration, fa4=fa4)
    trials = {"stock": [], "candidate": []}
    archive = subprocess.check_output(["git", "archive", MASTER, "src"], cwd=HERE.parents[1])
    with TemporaryDirectory(dir=args.output, prefix="master-") as folder:
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(folder, filter="data")
        worker = subprocess.Popen(
            [sys.executable, __file__, "--stock-worker", "--device", str(device), "--output", str(args.output),
             *extra_worker_args],
            env=os.environ | {"PYTHONPATH": str(Path(folder) / "src"), "DEVELOPMENT": "0"},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            assert read_worker(worker)["ready"]
            stock_records = json.loads((args.output / "stock-predictions.json").read_text())
            candidate_records, _ = record_predictions(model, tokenizer, device, candidate=lambda records:
                                                       candidate(records=records))
            changes = compare_predictions(stock_records, candidate_records)
            candidate()
            for i in range(args.repeats):
                for arm in (("stock", "candidate") if i % 2 == 0 else ("candidate", "stock")):
                    if arm == "stock":
                        worker.stdin.write("run\n")
                        worker.stdin.flush()
                        run = read_worker(worker)
                    else:
                        seconds, scores = timed(candidate, device)
                        run = {"seconds": seconds, "scores": scores}
                    trials[arm].append(run)
                    print(json.dumps({"pair": i + 1, "arm": arm, "seconds": run["seconds"]}), flush=True)
        finally:
            worker.stdin.close()
            worker.wait()
    if worker.returncode:
        raise RuntimeError(f"Stock worker exited with status {worker.returncode}")
    pairs = list(zip(trials["stock"], trials["candidate"]))
    return {
        "stock_sha": MASTER, "candidate_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                                                     cwd=HERE).strip(),
        "configuration": args.configuration, "tasks": 22, "examples": 352,
        "trials": trials, "changed_predictions": changes,
        "score_deltas": [compare_returned(a["scores"], b["scores"]) for a, b in pairs],
        "seconds_saved": [a["seconds"] - b["seconds"] for a, b in pairs],
        "median_seconds_saved": median(a["seconds"] - b["seconds"] for a, b in pairs),
        "median_paired_speedup": median(a["seconds"] / b["seconds"] for a, b in pairs),
    }


def overlap(args, model, tokenizer, device):
    from batching import load_examples
    from packing_pair import comparison_plans, mps_peak, plan_counts

    examples, _ = load_examples(tokenizer)
    examples = [example for example in examples if example.task_type == "multiple_choice"]
    plan = comparison_plans(examples, 4)["all"]["packed"]
    pad = tokenizer.get_bos_token_id()

    def run(config):
        prefetch, defer = CONFIGS[config]
        scores = planned_winners(plan, examples, model, pad, device, prefetch=prefetch, defer=defer)
        return scores, sum(winner == examples[i].gold for i, winner in scores.items())

    configs = [args.configuration] if args.metal_trace else list(CONFIGS)
    warm = {config: run(config) for config in configs}
    assert all(result == warm[configs[0]] for result in warm.values())
    samples = {config: [] for config in configs}
    for trial in range(args.repeats):
        for config in (configs if trial % 2 == 0 else configs[::-1]):
            profile = (torch.mps.profiler.profile(mode="interval", wait_until_completed=False)
                       if args.metal_trace else nullcontext())
            with profile:
                seconds, output = timed(partial(run, config), device)
            assert output == warm[config]
            samples[config].append(seconds)
            print(json.dumps({"trial": trial + 1, "configuration": config, "seconds": seconds}), flush=True)
    report = {"plan": plan_counts(plan, "packed"), "configs": {}}
    for config, values in samples.items():
        elapsed = median(values)
        report["configs"][config] = {"raw_seconds": values, "median_seconds": elapsed,
                                     "questions_per_second": len(examples) / elapsed}
        if device.type == "mps" and not args.metal_trace:
            report["configs"][config].update(mps_peak(partial(run, config)))
    return report


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("core", "paired", "overlap"), default="core")
    parser.add_argument("--configuration", choices=CONFIGS, default="both")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--metal-trace", action="store_true", help="Emit MPS signposts for a separate trace run.")
    parser.add_argument("--stock-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.metal_trace and (args.device != "mps" or args.mode != "overlap"):
        parser.error("--metal-trace requires --device mps and --mode overlap")
    os.environ["DEVELOPMENT"] = "0"
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if args.stock_worker:
        with redirect_stdout(sys.stderr):
            model, tokenizer = load_frozen_checkpoint(args.checkpoint, device), ByteTokenizer()
        stock_worker(model, tokenizer, device, args.output)
        return
    model, tokenizer = load_frozen_checkpoint(args.checkpoint, device), ByteTokenizer()
    if args.mode == "paired":
        report = paired(args, model, tokenizer, device,
                        extra_worker_args=("--checkpoint", str(args.checkpoint.resolve())))
    elif args.mode == "overlap":
        report = overlap(args, model, tokenizer, device)
    else:
        elapsed, scores = timed(partial(evaluate, model, tokenizer, device, configuration=args.configuration), device)
        report = {"seconds": elapsed, "scores": scores}
    report.update(device=str(device), torch=torch.__version__, max_context=model.max_seq_len,
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    (args.output / f"{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
