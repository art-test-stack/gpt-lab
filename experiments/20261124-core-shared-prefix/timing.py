import argparse
import json
from statistics import median
from time import perf_counter

import torch

from gpt_lab.evaluate.core import first_minimum, load_core_bundle, load_core_task, prepare_example
from parity import ATOL, ByteTokenizer, small_model
from scoring import (
    pack_choices, packed_sdpa, separate_choices, separate_choices_with_prefix_scoring, shared_prefix_length,
)


def timed(score, example):
    torch.mps.synchronize()
    start = perf_counter()
    nll = score(example)
    winner = first_minimum(nll)
    torch.mps.synchronize()
    return perf_counter() - start, nll, winner


def paired_trial(examples, scorers, repetition):
    paths = tuple(scorers)
    elapsed = dict.fromkeys((*paths, "shared_fallback"), 0.0)
    for i, (example, eligible) in enumerate(examples):
        if not eligible:
            seconds, _, _ = timed(scorers["reference"], example)
            elapsed["shared_fallback"] += seconds
            continue

        order = paths if i % 2 == 0 else paths[::-1]
        first = (i + repetition) % len(order)
        results = {}
        for path in order[first:] + order[:first]:
            seconds, nll, winner = timed(scorers[path], example)
            elapsed[path] += seconds
            results[path] = nll, winner
        ref_nll, ref_winner = results["reference"]
        for path, (nll, winner) in results.items():
            assert winner == ref_winner, (path, i, nll, ref_nll)
            torch.testing.assert_close(torch.tensor(nll), torch.tensor(ref_nll), atol=ATOL, rtol=0)
    return elapsed


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Time paired CORE scorers with PyTorch SDPA on MPS.")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("use at least one timed trial")
    if not torch.backends.mps.is_available():
        parser.error("PyTorch reports MPS unavailable")

    torch.set_num_threads(1)
    device = torch.device("mps")
    model, tokenizer = small_model(device), ByteTokenizer()
    pad = tokenizer.get_bos_token_id()
    data_path, tasks, _ = load_core_bundle()
    examples = []
    for task in tasks:
        if task["icl_task_type"] != "multiple_choice":
            continue
        data, meta = load_core_task(data_path, task, 16)
        for idx in range(len(data)):
            example = prepare_example(idx, tokenizer, data, meta, model.max_seq_len)
            examples.append((example, shared_prefix_length(*example) is not None))
    eligible_count = sum(eligible for _, eligible in examples)
    if not eligible_count:
        raise RuntimeError("the timed sample has no eligible questions")

    def packed(example):
        layout = pack_choices(*example)
        assert layout is not None
        return packed_sdpa(model, layout, device)[0]

    scorers = {
        "reference": lambda example: separate_choices(model, *example, pad, device)[0],
        "separate_scoring_control": lambda example: separate_choices_with_prefix_scoring(model, *example, pad, device),
        "packed": packed,
    }
    print(f"Warming up on {len(examples)} questions ({eligible_count} eligible)", flush=True)
    paired_trial(examples, scorers, repetition=-1)
    trials = []
    for repetition in range(args.repeats):
        elapsed = paired_trial(examples, scorers, repetition)
        trials.append(elapsed)
        print(json.dumps({"trial": repetition + 1, "seconds": elapsed}), flush=True)
    median_seconds = {path: median(trial[path] for trial in trials) for path in trials[0]}
    print(json.dumps({
        "torch": torch.__version__, "questions": len(examples),
        "eligible_questions": eligible_count, "fallback_questions": len(examples) - eligible_count,
        "trials": args.repeats, "median_seconds": median_seconds,
        "eligible_speedup_reference_over_packed": median_seconds["reference"] / median_seconds["packed"],
        "eligible_speedup_control_over_packed": median_seconds["separate_scoring_control"] / median_seconds["packed"],
        "median_seconds_including_shared_fallback": {
            path: median(trial[path] + trial["shared_fallback"] for trial in trials)
            for path in scorers
        },
    }, indent=2))


if __name__ == "__main__":
    main()
