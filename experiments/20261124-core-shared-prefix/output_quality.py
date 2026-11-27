"""Compare the scored outputs and task accuracy of two CORE evaluations."""
import math

import torch
import torch.nn.functional as F


def scored_outputs(logits, prepared):
    tokens, starts, ends = prepared
    assert len(logits) == len(tokens)
    log_probs, predictions = [], []
    for rows, row, start, end in zip(logits, tokens, starts, ends):
        # Match the stock slices even for a cropped span starting at zero.
        targets = row[start:end] if start > 0 else []
        assert rows.size(0) == len(targets), (rows.shape, start, end)
        target = torch.tensor(targets, dtype=torch.long, device=rows.device)
        log_probs.append((-F.cross_entropy(rows.float(), target, reduction="none")).cpu().tolist())
        predictions.append(rows.argmax(-1).cpu().tolist())
    return log_probs, predictions


def finite_nlls(values):
    return None if values is None else [x if math.isfinite(x) else None for x in values]


def differences(reference, candidate):
    assert len(reference) == len(candidate)
    pairs = [(a, b) for a, b in zip(reference, candidate)
             if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    deltas = torch.tensor([b - a for a, b in pairs], dtype=torch.float64)
    absolute = deltas.abs()
    non_finite = len(reference) - len(pairs)
    return {
        "values": len(reference), "non_finite": non_finite, "exact": not non_finite and bool((deltas == 0).all()),
        "mean_signed_delta": deltas.mean().item() if pairs else None,
        "mean_absolute_delta": absolute.mean().item() if pairs else None,
        "p95_absolute_delta": absolute.quantile(0.95).item() if pairs else None,
        "max_absolute_delta": absolute.max().item() if pairs else None,
    }


def winner_record(record):
    values, winner = record["option_nll"], record["winner"]
    margin = None
    if values is not None and winner is not None and values[winner] is not None:
        margin = min((value - values[winner] for i, value in enumerate(values)
                      if i != winner and value is not None), default=None)
    return {"winner": winner, "option_nll": values, "decision_margin_nats": margin, "correct": record["correct"]}


def compare_quality(stock, candidate, stock_scores, candidate_scores, nll_limit):
    def by_task(records, scores):
        offset, indexed = 0, {}
        for task, result in scores["all_core_results"].items():
            count = result["num_examples"]
            block = records[offset:offset + count]
            assert sorted(record["index"] for record in block) == list(range(count)), task
            indexed.update({(task, record["index"]): record for record in block})
            offset += count
        assert offset == len(records)
        return indexed

    stock = by_task(stock, stock_scores)
    candidate = by_task(candidate, candidate_scores)
    assert stock.keys() == candidate.keys() and len(stock) == 352
    log_probs, option_nlls = [[], []], [[], []]
    argmax_changes = correct_to_incorrect = incorrect_to_correct = 0
    changed = []
    for key, ref in stock.items():
        actual = candidate[key]
        assert all(ref[field] == actual[field] for field in ("dataset", "prepared", "task_type")), key
        assert len(ref["predictions"]) == len(actual["predictions"])
        question_argmax_changes = 0
        for i, (a, b) in enumerate(zip(ref["predictions"], actual["predictions"])):
            x, y = ref["target_log_probs"][i], actual["target_log_probs"][i]
            assert len(a) == len(b) == len(x) == len(y), key
            question_argmax_changes += sum(left != right for left, right in zip(a, b))
            log_probs[0].extend(x)
            log_probs[1].extend(y)
        argmax_changes += question_argmax_changes
        if ref["option_nll"] is not None:
            assert len(ref["option_nll"]) == len(actual["option_nll"])
            option_nlls[0].extend(ref["option_nll"])
            option_nlls[1].extend(actual["option_nll"])
        correct_to_incorrect += ref["correct"] and not actual["correct"]
        incorrect_to_correct += actual["correct"] and not ref["correct"]
        winner_changed = ref["winner"] != actual["winner"]
        if winner_changed or question_argmax_changes or ref["correct"] != actual["correct"]:
            changed.append({
                "task": key[0], "dataset": ref["dataset"], "index": key[1], "task_type": ref["task_type"],
                "changed_scored_token_argmaxes": question_argmax_changes, "winner_changed": winner_changed,
                "stock": winner_record(ref), "candidate": winner_record(actual),
            })

    token_stats = differences(*log_probs)
    nll_stats = differences(*option_nlls)
    relative = [abs(b - a) / abs(a) if a else 0.0
                for a, b in zip(*option_nlls) if a is not None and b is not None and (a != 0 or b == 0)]
    relative_drift = math.fsum(relative) / len(relative) if relative else None
    passes = len(relative) == nll_stats["values"] and relative_drift is not None and relative_drift <= nll_limit
    scores = {}
    for name, source, records in (("stock", stock_scores, stock), ("candidate", candidate_scores, candidate)):
        correct = sum(record["correct"] for record in records.values())
        scores[name] = {
            "raw_accuracy": source["core/accuracy"], "correct_examples": correct,
            "sample_accuracy": correct / len(records), "centered_core": source["core/core"],
            "tasks": {task: {k: score[k] for k in ("accuracy", "centered_result", "num_examples")}
                      for task, score in source["all_core_results"].items()},
        }
    before, after = scores["stock"], scores["candidate"]
    assert before["tasks"].keys() == after["tasks"].keys() and len(before["tasks"]) == 22
    core_delta = after["centered_core"] - before["centered_core"]
    scores["delta"] = {
        "raw_accuracy": after["raw_accuracy"] - before["raw_accuracy"],
        "sample_accuracy": after["sample_accuracy"] - before["sample_accuracy"], "centered_core": core_delta,
        "tasks": {task: {k: score[k] - before["tasks"][task][k] for k in ("accuracy", "centered_result")}
                  for task, score in after["tasks"].items()},
    }
    return {
        "target_log_probability": token_stats, "option_mean_nll": nll_stats,
        "changed_scored_token_argmaxes": argmax_changes,
        "changed_winners": sum(record["winner_changed"] for record in changed),
        "changed_predictions": changed, "exact_prediction_parity": not changed,
        "exact_numeric_parity": token_stats["exact"] and nll_stats["exact"],
        "option_nll_gate": {"limit": nll_limit, "mean_absolute_relative_delta": relative_drift,
                            "undefined": nll_stats["values"] - len(relative), "passed": passes},
        "core": scores, "exact_core_score_parity": before == after,
        "correct_to_incorrect": correct_to_incorrect,
        "incorrect_to_correct": incorrect_to_correct, "centered_core_drop_review": core_delta < 0,
    }
