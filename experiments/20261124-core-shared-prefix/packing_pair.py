"""Compare separate and packed answer rows at matched question batch sizes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from threading import Event, Thread
from time import perf_counter

import torch

from batching import Example, load_examples, plan_groups
from batched_scoring import (
    check_wide_fixture, collate_packed, collate_separate, score_packed_batch, score_separate_batch,
)
from fixture_model import ByteTokenizer, load_frozen_checkpoint, wide_fixture
from scoring import pack_choices


ARMS = ("separate", "packed")


def comparison_plans(examples, batch_size=4):
    eligible = sorted(
        (i for i, example in enumerate(examples) if example.packed is not None),
        key=lambda i: len(examples[i].packed.tokens),
    )
    fallback = sorted(
        (i for i, example in enumerate(examples) if example.packed is None),
        key=lambda i: max(map(len, examples[i].prepared[0])),
    )
    indices = eligible + fallback
    matched = [eligible[i:i + batch_size] for i in range(0, len(eligible), batch_size)]
    comparisons = {
        "eligible": {
            "separate": [(False, group) for group in matched],
            "packed": [(True, group) for group in matched],
        },
        "all": {
            "separate": [(False, indices[i:i + batch_size]) for i in range(0, len(indices), batch_size)],
            "packed": plan_groups(examples, indices, "wide", batch_size=batch_size),
        },
    }
    for scope, arms in comparisons.items():
        expected = eligible if scope == "eligible" else indices
        for plan in arms.values():
            assert [i for _, group in plan for i in group] == expected
    return comparisons


def bucket_width(width, buckets):
    return next((bound for bound in buckets if width <= bound), None)


def score_plan(plan, examples, model, pad_id, device, *, fa4=None, width_buckets=()):
    scores = {}
    for packed, indices in plan:
        if packed:
            questions = [examples[i].packed for i in indices]
            width = None
            if width_buckets:
                width = bucket_width(max(len(question.tokens) for question in questions), width_buckets)
            batch = collate_packed(questions, pad_id, max_context=model.max_seq_len, padded_width=width)
            nll, winners, _ = score_packed_batch(model, batch, device, fa4=fa4)
        else:
            prepared = [examples[i].prepared for i in indices]
            width = None
            if width_buckets:
                widest = max(len(row) for tokens, _, _ in prepared for row in tokens)
                width = bucket_width(widest, width_buckets)
                if width is not None:
                    width = min(width, model.max_seq_len)
            batch = collate_separate(prepared, pad_id, padded_width=width)
            nll, winners = score_separate_batch(model, batch, device)
        # One readback per forward in both arms, for all options and winners.
        results = nll.new_empty((len(indices), nll.size(1) + 1))
        results[:, :-1] = nll
        results[:, -1] = winners
        for index, row in zip(indices, results.cpu().tolist()):
            example = examples[index]
            choices = len(example.prepared[0])
            scores[example.label, example.index] = row[:choices], int(row[-1])
    return scores


def plan_counts(plan, arm):
    fallbacks = [indices for packed, indices in plan if arm == "packed" and not packed]
    return {
        "questions": sum(len(indices) for _, indices in plan),
        "forwards": len(plan),
        "packed_forwards": sum(packed for packed, _ in plan),
        "fallback_forwards": len(fallbacks),
        "fallback_questions": sum(map(len, fallbacks)),
    }


def check_fp32(reference, candidate):
    assert reference.keys() == candidate.keys()
    max_abs = 0.0
    for key, (ref, winner) in reference.items():
        nll, packed_winner = candidate[key]
        assert winner == packed_winner, (key, ref, nll)
        expected, actual = torch.tensor(ref), torch.tensor(nll)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-5, equal_nan=True, msg=str(key))
        max_abs = max(max_abs, (actual - expected).abs().nan_to_num().max().item())
    return {"max_option_nll_abs": max_abs, "winners_exact": True}


def mps_peak(score):
    done = Event()
    peak = torch.mps.current_allocated_memory()
    samples = 1

    def sample():
        nonlocal peak, samples
        while not done.wait(0.001):
            peak = max(peak, torch.mps.current_allocated_memory())
            samples += 1

    worker = Thread(target=sample)
    worker.start()
    try:
        score()
        torch.mps.synchronize()
    finally:
        done.set()
        worker.join()
    return {"sampled_peak_mps_tensor_bytes": peak, "memory_samples": samples}


def mps_attempt(run):
    try:
        return run(), None
    except RuntimeError as error:
        message = str(error)
        if not any(text in message.lower() for text in ("out of memory", "invalid buffer size", "failed to allocate")):
            raise
    torch.mps.synchronize()
    torch.mps.empty_cache()
    return None, message


def mps_compare(scope, arms, score, repeats, batch_size):
    warmup, failures = {}, {}
    for arm in ARMS:
        result, failure = mps_attempt(lambda: score(arm))
        if failure is None:
            warmup[arm] = result
        else:
            failures[arm] = failure
    parity = None
    if len(warmup) == len(ARMS):
        parity = check_fp32(warmup["separate"], warmup["packed"])

    def time_arm(arm):
        torch.mps.synchronize()
        start = perf_counter()
        score(arm)
        torch.mps.synchronize()
        return perf_counter() - start

    trials = {arm: [] for arm in ARMS}
    for trial in range(repeats):
        for arm in (ARMS if trial % 2 == 0 else ARMS[::-1]):
            if arm in failures:
                continue
            seconds, failure = mps_attempt(lambda: time_arm(arm))
            if failure is None:
                trials[arm].append(seconds)
            else:
                failures[arm] = failure
            print(json.dumps({
                "batch_size": batch_size, "scope": scope, "arm": arm, "trial": trial + 1,
                "seconds": seconds, "allocation_error": failure,
            }), flush=True)

    comparison = {}
    for arm, plan in arms.items():
        counts = plan_counts(plan, arm)
        if arm in failures:
            comparison[arm] = {**counts, "allocation_error": failures[arm]}
            continue
        seconds = median(trials[arm])
        memory, failure = mps_attempt(lambda: mps_peak(lambda: score(arm)))
        comparison[arm] = {
            **counts, "trial_seconds": trials[arm], "median_seconds": seconds,
            "questions_per_second": counts["questions"] / seconds,
            **(memory if failure is None else {"memory_allocation_error": failure}),
        }
    comparison["paired_speedup"] = None if failures else median(
        a / b for a, b in zip(trials["separate"], trials["packed"])
    )
    return comparison, parity


def fastest_sizes(sizes):
    winner = {}
    for scope in ("eligible", "all"):
        winner[scope] = {}
        for arm in ARMS:
            measured = [
                (size, results["comparisons"][scope][arm])
                for size, results in sizes.items()
                if "questions_per_second" in results["comparisons"][scope][arm]
            ]
            if measured:
                size, result = max(measured, key=lambda item: item[1]["questions_per_second"])
                winner[scope][arm] = {"batch_size": size, **result}
    return winner


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Measure CORE packing throughput across question batch sizes on MPS.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4])
    parser.add_argument("--width-buckets", type=int, nargs="+", default=[])
    args = parser.parse_args()
    if not torch.backends.mps.is_available() or args.repeats < 1:
        parser.error("an MPS device and at least one timed trial are required")
    if min(args.batch_sizes) < 1 or len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("batch sizes must be distinct positive integers")
    if args.width_buckets != sorted(set(args.width_buckets)) or any(width < 1 for width in args.width_buckets):
        parser.error("width buckets must be increasing positive integers")
    torch.set_num_threads(1)
    device = torch.device("mps")
    memory_fraction = os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if memory_fraction is not None:
        # Honor an explicit cap after GPT-Lab initializes the backend.
        torch.mps.set_per_process_memory_fraction(float(memory_fraction))
    model, tokenizer = load_frozen_checkpoint(args.checkpoint, device), ByteTokenizer()
    pad_id = tokenizer.get_bos_token_id()
    fixture = check_wide_fixture(model, pad_id, device)
    fixture_examples = [
        Example("fixture", i, "multiple_choice", None, choices, pack_choices(*choices))
        for i, choices in enumerate(wide_fixture(pad_id))
    ]
    group = [0, 1]
    fixture.update(check_fp32(
        score_plan([(False, group)], fixture_examples, model, pad_id, device, width_buckets=args.width_buckets),
        score_plan([(True, group)], fixture_examples, model, pad_id, device, width_buckets=args.width_buckets),
    ))
    print(json.dumps({"fixture": fixture}), flush=True)

    examples, _ = load_examples(tokenizer)
    examples = [example for example in examples if example.task_type == "multiple_choice"]
    assert len(examples) == 176 and sum(example.packed is not None for example in examples) == 140
    sizes = {}
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "device": "mps", "torch": torch.__version__, "prefetch": False,
        "width_buckets": args.width_buckets, "mps_high_watermark_ratio": memory_fraction,
        "fixture": fixture, "batch_sizes": sizes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for batch_size in args.batch_sizes:
        comparisons, parity = {}, {}
        for scope, arms in comparison_plans(examples, batch_size).items():
            comparisons[scope], parity[scope] = mps_compare(
                scope, arms, lambda arm: score_plan(
                    arms[arm], examples, model, pad_id, device, width_buckets=args.width_buckets,
                ),
                args.repeats, batch_size,
            )
        sizes[batch_size] = {"comparisons": comparisons, "parity": parity}
        report["fastest_sizes"] = fastest_sizes(sizes)
        if len(args.batch_sizes) == 1:
            report.update(sizes[batch_size])
        staged = args.report.with_suffix(".tmp")
        staged.write_text(json.dumps(report, indent=2) + "\n")
        staged.replace(args.report)


if __name__ == "__main__":
    main()
