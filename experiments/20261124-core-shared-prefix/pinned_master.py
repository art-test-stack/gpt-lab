"""Run with PYTHONPATH pointing at the archived master source tree."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from gpt_lab.evaluate import core
from batch_timing import observe_rows, time_sweep
from fixture_model import ByteTokenizer, load_frozen_checkpoint, wide_fixture


def record_core(model, tokenizer, device, *, evaluate=None, capture_logits=True, capture_log_probs=False):
    records = []
    batch = {name: getattr(core, name) for name in (
        "batch_sequences_mc", "batch_sequences_schema", "batch_sequences_lm",
    )}
    stock_stack, stock_forward, stock_evaluate = core.stack_sequences, core.forward_model, core.evaluate_example
    current = {}
    logits = []

    def remember_spans(batch_sequences):
        def wrapped(*args):
            result = batch_sequences(*args)
            if "index" in current:
                current["uncropped"] = result
            return result
        return wrapped

    def remember_rows(tokens, pad_id):
        if "index" not in current:
            return stock_stack(tokens, pad_id)
        original, starts, ends = current.pop("uncropped")
        shift = [len(row) - len(cropped) for row, cropped in zip(original, tokens)]
        current["prepared"] = (
            tokens, [start - drop for start, drop in zip(starts, shift)],
            [end - drop for end, drop in zip(ends, shift)],
        )
        return stock_stack(tokens, pad_id)

    def remember_losses(model, rows):
        losses, predictions = stock_forward(model, rows)
        _, starts, ends = current["prepared"]
        if capture_logits:
            current["logits"] = [logits[0][i, start - 1:end - 1].cpu()
                                 for i, (start, end) in enumerate(zip(starts, ends))]
        current["predictions"] = [predictions[i, start - 1:end - 1].cpu()
                                  for i, (start, end) in enumerate(zip(starts, ends))]
        if capture_log_probs:
            from output_quality import scored_outputs

            scored = [logits[0][i, start - 1:end - 1] for i, (start, end) in enumerate(zip(starts, ends))]
            current["target_log_probs"], _ = scored_outputs(scored, current["prepared"])
        if current["task_type"] != "language_modeling":
            nll = [losses[i, start - 1:end - 1].mean().item() for i, (start, end) in enumerate(zip(starts, ends))]
            current.update(nll=torch.tensor(nll), winner=nll.index(min(nll)))
        return losses, predictions

    def remember_example(idx, model, tokenizer, data, device, meta, **kwargs):
        current.update(index=idx, task_type=meta["task_type"], nll=None, winner=None)
        current["dataset"] = meta["dataset_uri"]
        current["correct"] = stock_evaluate(idx, model, tokenizer, data, device, meta, **kwargs)
        records.append(dict(current))
        current.clear()
        logits.clear()
        return records[-1]["correct"]

    def remember_logits(_, args, output):
        if "index" in current:
            logits.append(output.logits)

    hook = model.register_forward_hook(remember_logits) if capture_logits or capture_log_probs else None
    replacements = {name: remember_spans(fn) for name, fn in batch.items()}
    try:
        with patch.multiple(core, **replacements, stack_sequences=remember_rows,
                            forward_model=remember_losses, evaluate_example=remember_example):
            results = (core.evaluate_core(model, tokenizer, device, max_per_task=16)
                       if evaluate is None else evaluate(records))
    finally:
        if hook is not None:
            hook.remove()
    return records, results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("golden", type=Path)
    parser.add_argument("device", choices=("cpu", "mps"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    with redirect_stdout(sys.stderr):
        model = load_frozen_checkpoint(args.checkpoint, device)
        tokenizer = ByteTokenizer()
        records, results = record_core(model, tokenizer, device)
        pad = tokenizer.get_bos_token_id()
        fixture = [
            observe_rows(core, model, core.stack_sequences(tokens, pad).to(device), starts, ends,
                         task_type="multiple_choice")
            for tokens, starts, ends in wide_fixture(pad)
        ]
        torch.save({"records": records, "results": results, "fixture": fixture}, args.golden)
    print(json.dumps({"ready": True, "source": core.__file__, "examples": len(records)}), flush=True)

    def prepare(index):
        tokens, starts, ends = records[index]["prepared"]
        return core.stack_sequences(tokens, pad), starts, ends

    def score(batch):
        rows, starts, ends = batch
        losses, _ = core.forward_model(model, rows.to(device))
        nll = [losses[i, start - 1:end - 1].mean().item() for i, (start, end) in enumerate(zip(starts, ends))]
        nll.index(min(nll))

    for line in sys.stdin:
        command = json.loads(line)
        elapsed = time_sweep(command["indices"], prepare, score, device, command["prefetch"])
        print(json.dumps({"seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
