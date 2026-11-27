import argparse
from dataclasses import dataclass
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

import torch

from gpt_lab.evaluate import core
from batch_timing import logits_result, observe_rows, time_sweep
from batched_scoring import CONTEXT, check_wide_fixture, collate_packed, score_packed_batch
from fixture_model import ByteTokenizer, freeze_checkpoint, load_frozen_checkpoint, wide_fixture
from scoring import PackedChoices, pack_choices, packed_sdpa


MASTER = "61f066250420afbc121524037b996ee4a3524c0a"
MODES = ("master", "one_question", "strict", "wide")


@dataclass
class Example:
    label: str
    index: int
    task_type: str
    gold: int | None
    prepared: tuple
    packed: PackedChoices | None


def load_examples(tokenizer):
    data_path, tasks, baselines = core.load_core_bundle()
    examples = []
    for task in tasks:
        data, meta = core.load_core_task(data_path, task, 16)
        for index, item in enumerate(data):
            prepared = core.prepare_example(index, tokenizer, data, meta, CONTEXT)
            assert max(map(len, prepared[0])) <= CONTEXT
            kind = meta["task_type"]
            packed = pack_choices(*prepared) if kind == "multiple_choice" else None
            examples.append(Example(task["label"], index, kind, item.get("gold"), prepared, packed))
    return examples, baselines


def plan_groups(examples, indices, mode, *, batch_size=4):
    """Keep contiguous groups of similar-width eligible questions."""
    if mode == "one_question":
        batch_size = 1
    groups, pending = [], []
    for index in indices:
        packed = examples[index].packed
        eligible = (mode != "master" and packed is not None
                    and (mode != "strict" or len(packed.tokens) <= CONTEXT))
        if eligible:
            pending.append(index)
        if pending and (not eligible or len(pending) == batch_size):
            groups.append((True, pending))
            pending = []
        if not eligible:
            groups.append((False, [index]))
    if pending:
        groups.append((True, pending))
    return groups


def make_inputs(group, examples, mode, pad):
    is_packed, indices = group
    if is_packed:
        questions = [pack_choices(*examples[i].prepared) for i in indices]
        if mode == "strict":
            assert max(len(question.tokens) for question in questions) <= CONTEXT
        payload = questions[0] if mode == "one_question" else collate_packed(questions, pad)
    else:
        rows, starts, ends = examples[indices[0]].prepared
        payload = core.stack_sequences(rows, pad), starts, ends
    return is_packed, indices, payload


def score_inputs(inputs, examples, model, mode, device, *, capture=False):
    is_packed, indices, payload = inputs
    if not is_packed:
        rows, starts, ends = payload
        rows = rows.to(device)
        if capture:
            return [observe_rows(core, model, rows, starts, ends, task_type=examples[indices[0]].task_type)]
        losses, _ = core.forward_model(model, rows)
        core.first_minimum(core.choice_nll(losses, starts, ends))
    elif mode == "one_question":
        nll, logits = packed_sdpa(model, payload, device, capture_logits=capture)
        winner = core.first_minimum(nll)
        if capture:
            return [logits_result(logits, nll, winner)]
    else:
        means, winners, logits = score_packed_batch(model, payload, device, capture_logits=capture)
        if capture:
            return [
                logits_result(scores, means[i, :len(scores)], winner)
                for i, (scores, winner) in enumerate(zip(logits, winners.tolist()))
            ]


def compare(reference, candidate, totals, key):
    assert reference["winner"] == candidate["winner"], (key, reference["nll"], candidate["nll"])
    assert len(reference["logits"]) == len(candidate["logits"]), key
    for i, (ref, actual) in enumerate(zip(reference["logits"], candidate["logits"])):
        expected = reference["predictions"][i] if "predictions" in reference else ref.argmax(-1)
        assert torch.equal(expected, actual.argmax(-1)), (key, i)
        totals["scored_tokens"] += ref.size(0)
        compare_values(ref, actual, totals["logits"])
    if reference["nll"] is not None:
        compare_values(reference["nll"], candidate["nll"], totals["option_nll"])


def compare_values(reference, candidate, totals):
    assert reference.shape == candidate.shape and reference.dtype == candidate.dtype
    bitwise = torch.equal(reference.contiguous().view(torch.uint8), candidate.contiguous().view(torch.uint8))
    totals["bitwise_equal"] &= bitwise
    if reference.numel():
        diff = (reference - candidate).abs().max().item()
        assert diff < float("inf"), diff
        totals["max_abs"] = max(totals["max_abs"], diff)


def empty_totals():
    return {
        "logits": {"bitwise_equal": True, "max_abs": 0.0},
        "option_nll": {"bitwise_equal": True, "max_abs": 0.0},
        "scored_tokens": 0,
    }


def parity(golden, examples, indices, baselines, model, device, pad):
    records, stock = golden["records"], golden["results"]
    labels = list(stock["all_core_results"])
    assert len(records) == len(examples) == 352 and len(labels) == 22
    separate = []
    for i, (ex, ref) in enumerate(zip(examples, records)):
        assert ex.label == labels[i // 16] and ex.index == ref["index"] and ex.task_type == ref["task_type"]
        assert ex.prepared == ref["prepared"], (ex.label, ex.index)
        group = (False, [i])
        separate.extend(score_inputs(make_inputs(group, examples, "master", pad), examples,
                                     model, "master", device, capture=True))
    report = {}
    for mode in MODES[1:]:
        observed = list(separate)
        for group in plan_groups(examples, indices, mode):
            if group[0]:
                inputs = make_inputs(group, examples, mode, pad)
                outputs = score_inputs(inputs, examples, model, mode, device, capture=True)
                for i, result in zip(group[1], outputs):
                    observed[i] = result
        totals, scores = empty_totals(), dict.fromkeys(labels, 0)
        for ex, ref, actual in zip(examples, records, observed):
            compare(ref, actual, totals, (mode, ex.label, ex.index))
            if ex.task_type == "language_modeling":
                (row,), (start,), (end,) = ex.prepared
                correct = actual["logits"][0].argmax(-1).tolist() == row[start:end]
            else:
                correct = actual["winner"] == ex.gold
            assert correct == ref["correct"], (mode, ex.label, ex.index)
            scores[ex.label] += int(correct)
        accuracies = {label: count / 16 for label, count in scores.items()}
        centered = {label: (acc - 0.01 * baselines[label]) / (1 - 0.01 * baselines[label])
                    for label, acc in accuracies.items()}
        for label, result in stock["all_core_results"].items():
            assert accuracies[label] == result["accuracy"] and centered[label] == result["centered_result"]
        core_score = sum(centered.values()) / len(centered)
        assert core_score == stock["core/core"]
        report[mode] = {**totals, "centered_core": core_score, "per_task_accuracy": accuracies,
                        "selected_answers_exact": True, "scored_argmaxes_exact": True}
    return report


def fixture_check(golden, model, pad, device):
    isolation = check_wide_fixture(model, pad, device)
    examples = [Example("fixture", i, "multiple_choice", None, choice, pack_choices(*choice))
                for i, choice in enumerate(wide_fixture(pad))]
    assert plan_groups(examples, (1, 0), "strict") == [(True, [1]), (False, [0])]
    wide_group, = plan_groups(examples, (1, 0), "wide")
    assert wide_group == (True, [1, 0])
    outputs = score_inputs(make_inputs(wide_group, examples, "wide", pad), examples, model, "wide", device, capture=True)
    totals = empty_totals()
    for i, result in zip(wide_group[1], outputs):
        compare(golden["fixture"][i], result, totals, ("fixture", i))
    return {**isolation, **totals, "strict_fallback_questions": 1}


def dimensions(examples, groups, mode):
    allocated = padding = fallback = width_fallback = max_width = max_position = 0
    for is_packed, indices in groups:
        if is_packed:
            packed = [examples[i].packed for i in indices]
            lengths = [len(item.tokens) for item in packed]
            position = max(max(item.positions) for item in packed)
        else:
            lengths = list(map(len, examples[indices[0]].prepared[0]))
            position = max(lengths) - 1
            fallback += int(mode != "master")
            width_fallback += int(mode == "strict" and examples[indices[0]].packed is not None)
        width = max(lengths)
        slots = len(lengths) * width
        allocated += slots
        padding += slots - sum(lengths)
        max_width = max(max_width, width)
        max_position = max(max_position, position)
    return dict(allocated_tokens=allocated, padding_tokens=padding, fallback_questions=fallback,
                width_fallback_questions=width_fallback,
                forwards=len(groups), max_physical_width=max_width, max_logical_position=max_position)


def benchmark(worker, examples, indices, model, device, pad, repeats):
    groups = {mode: plan_groups(examples, indices, mode) for mode in MODES}
    configs = [(mode, ahead) for mode in MODES for ahead in (False, True)]
    timings = {config: [] for config in configs}

    def measure(mode, ahead):
        if mode == "master":
            request = json.dumps({"indices": indices, "prefetch": ahead})
            worker.stdin.write(request + "\n")
            worker.stdin.flush()
            return read_worker(worker)["seconds"]
        return time_sweep(
            groups[mode], lambda plan: make_inputs(plan, examples, mode, pad),
            lambda batch: score_inputs(batch, examples, model, mode, device),
            device, ahead,
        )

    for mode in MODES:
        print(f"Warming {mode}", flush=True)
        measure(mode, False)
    for trial in range(repeats):
        offset = trial % len(configs)
        order = configs[offset:] + configs[:offset]
        for mode, ahead in order:
            seconds = measure(mode, ahead)
            timings[mode, ahead].append(seconds)
            print(json.dumps({"trial": trial + 1, "mode": mode, "prefetch": ahead, "seconds": seconds}), flush=True)
    report = {}
    for mode, ahead in configs:
        trial_times = timings[mode, ahead]
        seconds = median(trial_times)
        reference = timings["master", ahead]
        report[f"{mode}/prefetch_{'on' if ahead else 'off'}"] = {
            **dimensions(examples, groups[mode], mode), "trial_seconds": trial_times,
            "median_seconds": seconds, "questions_per_second": len(indices) / seconds,
            "median_paired_speedup_vs_master": median(a / b for a, b in zip(reference, trial_times)),
        }
    return report


def read_worker(worker):
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError("Pinned master stopped before returning its result")
    return json.loads(line)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Compare CORE question batching with pinned master.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--repeats", type=int, default=3, help="Use 0 for parity only.")
    args = parser.parse_args()
    if args.repeats < 0:
        parser.error("repeats must be nonnegative")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("PyTorch reports MPS unavailable")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "frozen.pt"
    freeze_checkpoint(checkpoint, device)
    model, tokenizer = load_frozen_checkpoint(checkpoint, device), ByteTokenizer()
    examples, baselines = load_examples(tokenizer)
    indices = [i for i, ex in enumerate(examples) if ex.task_type == "multiple_choice"]
    indices.sort(key=lambda i: len(examples[i].packed.tokens) if examples[i].packed else float("inf"))
    scripts = Path(__file__).resolve().parent
    repo = scripts.parents[1]
    archive = subprocess.check_output(["git", "archive", MASTER, "src"], cwd=repo)
    with TemporaryDirectory(dir=work, prefix="master-") as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(temporary, filter="data")
        sources = Path(temporary) / "src"
        golden_path = Path(temporary) / "golden.pt"
        worker = subprocess.Popen(
            [sys.executable, str(scripts / "pinned_master.py"), str(checkpoint), str(golden_path), args.device],
            env=os.environ | {"PYTHONPATH": str(sources), "DEVELOPMENT": "0"},
            text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        try:
            ready = read_worker(worker)
            assert ready["ready"] and Path(ready["source"]).is_relative_to(sources) and ready["examples"] == 352
            golden = torch.load(golden_path, weights_only=True)
            pad = tokenizer.get_bos_token_id()
            report = {
                "master": MASTER, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "device": str(device), "torch": torch.__version__, "max_context": CONTEXT,
                "examples": len(examples), "multiple_choice_questions": len(indices),
                "fixture": fixture_check(golden, model, pad, device),
                "parity": parity(golden, examples, indices, baselines, model, device, pad),
            }
            print(json.dumps(report), flush=True)
            if args.repeats:
                report["timing"] = benchmark(worker, examples, indices, model, device, pad, args.repeats)
            (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            worker.stdin.close()
            worker.wait()
    if worker.returncode:
        raise RuntimeError(f"Pinned master exited with status {worker.returncode}")


if __name__ == "__main__":
    main()
