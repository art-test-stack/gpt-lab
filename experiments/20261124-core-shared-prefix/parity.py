import argparse
import json
import platform

import torch

from gpt_lab.evaluate.core import evaluate_core, first_minimum, prepare_example
from gpt_lab.model.gpt import DenseTransformer
from gpt_lab.utils.schemas import TransformerConfig
from scoring import pack_choices, packed_sdpa, separate_choices, separate_choices_with_prefix_scoring, tree_masks


ATOL = 2e-5
MC = dict(task_type="multiple_choice", num_fewshot=0, continuation_delimiter=" ")


class ByteTokenizer:
    """Deterministic UTF-8 tokenizer with a separate BOS for the small model."""
    def get_bos_token_id(self):
        return 256

    def __call__(self, texts, *, prepend):
        return [[prepend, *text.encode("utf-8")] for text in texts]


def small_model(device, *, head_dim=16):
    torch.manual_seed(29)
    model = DenseTransformer(TransformerConfig(
        vocab_size=257, max_context=2048, n_layers=2, n_heads=2, d_head=head_dim,
        d_model=2 * head_dim, d_ffn=4 * head_dim, window_pattern="SL", window_size=4, attn_impl="sdpa",
    ))
    model.init_weights()
    # Training initialization zeros residual projections. Exercise both layers.
    with torch.no_grad():
        model.lm_head.weight.normal_(std=0.08)
        for block in model.blocks:
            block.attention.o_proj.weight.normal_(std=0.08)
            block.ffn.w_2.weight.normal_(std=0.05)
    return model.to(device).eval()


class Parity:
    def __init__(self):
        self.packed = 0
        self.fallback = 0
        self.max_logit_abs = 0.0
        self.max_nll_abs = 0.0

    def __call__(self, model, tokens, starts, ends, pad_id, device):
        args = model, tokens, starts, ends, pad_id, device
        packed = pack_choices(tokens, starts, ends)
        if packed is None:
            self.fallback += 1
            return separate_choices(*args)[0]
        self.packed += 1
        ref_nll, ref_logits = separate_choices(*args, capture_logits=True)
        nll, logits = packed_sdpa(model, packed, device, capture_logits=True)
        for actual, reference in zip(logits, ref_logits):
            torch.testing.assert_close(actual, reference, atol=ATOL, rtol=0)
            self.max_logit_abs = max(self.max_logit_abs, (actual - reference).abs().max().item())
        for actual, reference in zip(nll, ref_nll):
            diff = abs(actual - reference)
            assert diff <= ATOL, (actual, reference)
            self.max_nll_abs = max(self.max_nll_abs, diff)
        assert first_minimum(nll) == first_minimum(ref_nll), (nll, ref_nll)
        return nll


def prepare_choice_fixture(tokenizer, query, choices, limit=None):
    data = [dict(query=query, choices=choices, gold=0)]
    return prepare_example(0, tokenizer, data, MC, limit)


def cropped_sibling_examples(tokenizer):
    query = "0123456789abcdefghij"
    aligned = prepare_choice_fixture(tokenizer, query, ["ABCDEF", "GHIJKL"], 16)
    mutated = prepare_choice_fixture(tokenizer, query, ["AXYZEF", "GHIJKL"], 16)
    return aligned, mutated


def check_sibling_isolation(original, mutated, score):
    assert original.positions == mutated.positions and original.branches == mutated.branches
    assert original.tokens[:original.prefix] == mutated.tokens[:mutated.prefix]
    assert original.tokens[original.offsets[1]:] == mutated.tokens[mutated.offsets[1]:]
    nll, before = score(original)
    _, after = score(mutated)
    assert not torch.equal(before[0], after[0])
    torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)
    return nll, (before[1] - after[1]).abs().max().item()


def boundary_checks(model, tokenizer, device):
    parity = Parity()
    pad = tokenizer.get_bos_token_id()

    unequal = prepare_choice_fixture(tokenizer, "Question?", ["A", "BCDEFGHI"])
    assert [end - start for start, end in zip(unequal[1], unequal[2])] == [1, 8]
    nll = parity(model, *unequal, pad, device)
    control = separate_choices_with_prefix_scoring(model, *unequal, pad, device)
    torch.testing.assert_close(torch.tensor(control), torch.tensor(nll), atol=ATOL, rtol=0)

    aligned, mutated = cropped_sibling_examples(tokenizer)
    assert [row[0] for row in aligned[0]] == [ord("b")] * 2
    assert aligned[1] == [10, 10]
    parity(model, *aligned, pad, device)
    misaligned = prepare_choice_fixture(tokenizer, "0123456789abcdefghij", ["ABCDEF", "GHIJKLM"], 16)
    assert misaligned[1] == [10, 9]
    assert parity(model, *misaligned, pad, device) == separate_choices(model, *misaligned, pad, device)[0]
    assert (parity.packed, parity.fallback) == (2, 1)

    original = pack_choices(*aligned)
    positions = torch.tensor(original.positions, device=device)
    branches = torch.tensor(original.branches, device=device)
    short, full = tree_masks(positions, branches, model.window_sizes)
    a, b = original.offsets
    p = original.prefix
    assert short[b, p - 4] and not short[b, p - 5] and full[b, p - 5]
    assert not full[b, a] and not full[a, b] and not full[p - 1, a]
    assert short[b + 4, b] and not short[b + 5, b]
    _, max_abs = check_sibling_isolation(
        original, pack_choices(*mutated),
        lambda layout: packed_sdpa(model, layout, device, capture_logits=True),
    )
    assert first_minimum([2.0, 2.0, 3.0]) == 0
    return {**vars(parity), "branch_b_max_logit_abs": max_abs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--max-per-task", type=int, default=16)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("PyTorch reports MPS unavailable")
    if args.max_per_task < 11:
        parser.error("the CORE sample requires at least 11 examples per task for ten-shot prompts")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    model, tokenizer = small_model(device), ByteTokenizer()
    boundaries = boundary_checks(model, tokenizer, device)
    sample = Parity()
    reference = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)
    packed = evaluate_core(
        model, tokenizer, device, max_per_task=args.max_per_task, score_multiple_choice=sample,
    )
    for key in ("core/core", "core/accuracy"):
        assert reference[key] == packed[key], key
    for task, result in reference["all_core_results"].items():
        assert result["accuracy"] == packed["all_core_results"][task]["accuracy"], task
    assert sample.packed > 0
    print(json.dumps({
        "device": str(device), "platform": platform.platform(), "torch": torch.__version__,
        "fp32_atol": ATOL, "rtol": 0, "boundaries": boundaries, "sample": vars(sample),
        "sample_size": sum(result["num_examples"] for result in reference["all_core_results"].values()),
        "core_by_path": {name: results["core/core"] for name, results in (("separate", reference), ("packed", packed))},
        "accuracy_by_path": {name: results["core/accuracy"] for name, results in (("separate", reference), ("packed", packed))},
    }, indent=2))


if __name__ == "__main__":
    main()
