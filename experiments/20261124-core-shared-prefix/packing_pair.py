"""Compare separate and packed answer rows with the same four-question batching."""
import argparse
import hashlib
import json
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


def comparison_plans(examples):
    eligible = sorted(
        (i for i, example in enumerate(examples) if example.packed is not None),
        key=lambda i: len(examples[i].packed.tokens),
    )
    fallback = sorted(
        (i for i, example in enumerate(examples) if example.packed is None),
        key=lambda i: max(map(len, examples[i].prepared[0])),
    )
    indices = eligible + fallback
    matched = [eligible[i:i + 4] for i in range(0, len(eligible), 4)]
    comparisons = {
        "eligible": {
            "separate": [(False, group) for group in matched],
            "packed": [(True, group) for group in matched],
        },
        "all": {
            "separate": [(False, indices[i:i + 4]) for i in range(0, len(indices), 4)],
            "packed": plan_groups(examples, indices, "wide"),
        },
    }
    for scope, arms in comparisons.items():
        expected = eligible if scope == "eligible" else indices
        for plan in arms.values():
            assert [i for _, group in plan for i in group] == expected
    return comparisons


def score_plan(plan, examples, model, pad_id, device, *, fa4=None):
    scores = {}
    for packed, indices in plan:
        if packed:
            batch = collate_packed([examples[i].packed for i in indices], pad_id, max_context=model.max_seq_len)
            nll, winners, _ = score_packed_batch(model, batch, device, fa4=fa4)
        else:
            batch = collate_separate([examples[i].prepared for i in indices], pad_id)
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Measure the incremental benefit of packing four questions on MPS.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch.backends.mps.is_available() or args.repeats < 1:
        parser.error("an MPS device and at least one timed trial are required")
    torch.set_num_threads(1)
    device = torch.device("mps")
    model, tokenizer = load_frozen_checkpoint(args.checkpoint, device), ByteTokenizer()
    pad_id = tokenizer.get_bos_token_id()
    fixture = check_wide_fixture(model, pad_id, device)
    fixture_examples = [
        Example("fixture", i, "multiple_choice", None, choices, pack_choices(*choices))
        for i, choices in enumerate(wide_fixture(pad_id))
    ]
    group = [0, 1]
    fixture.update(check_fp32(
        score_plan([(False, group)], fixture_examples, model, pad_id, device),
        score_plan([(True, group)], fixture_examples, model, pad_id, device),
    ))
    print(json.dumps({"fixture": fixture}), flush=True)

    examples, _ = load_examples(tokenizer)
    examples = [example for example in examples if example.task_type == "multiple_choice"]
    plans = comparison_plans(examples)
    assert len(examples) == 176 and len(plans["eligible"]["packed"]) == 35
    assert len(plans["all"]["separate"]) == 44
    assert plan_counts(plans["all"]["packed"], "packed")["fallback_questions"] == 36

    def score(scope, arm):
        return score_plan(plans[scope][arm], examples, model, pad_id, device)

    # Both full parity passes exercise the exact groups and device paths used in timing.
    parity = {scope: check_fp32(score(scope, "separate"), score(scope, "packed")) for scope in plans}
    print(json.dumps({"parity": parity}), flush=True)
    comparisons = {}
    for scope, arms in plans.items():
        trials = {arm: [] for arm in ARMS}
        for trial in range(args.repeats):
            for arm in (ARMS if trial % 2 == 0 else ARMS[::-1]):
                torch.mps.synchronize()
                start = perf_counter()
                score(scope, arm)
                torch.mps.synchronize()
                seconds = perf_counter() - start
                trials[arm].append(seconds)
                print(json.dumps({"scope": scope, "arm": arm, "trial": trial + 1, "seconds": seconds}), flush=True)
        arms_report = {}
        for arm, plan in arms.items():
            counts = plan_counts(plan, arm)
            seconds = median(trials[arm])
            arms_report[arm] = {
                **counts, "trial_seconds": trials[arm], "median_seconds": seconds,
                "questions_per_second": counts["questions"] / seconds,
                **mps_peak(lambda: score(scope, arm)),
            }
        comparisons[scope] = {
            **arms_report,
            "paired_speedup": median(a / b for a, b in zip(trials["separate"], trials["packed"])),
        }
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "device": "mps", "torch": torch.__version__, "prefetch": False,
        "fixture": fixture, "parity": parity, "comparisons": comparisons,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
