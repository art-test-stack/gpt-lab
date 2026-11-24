import argparse
import json
import math
import os
from time import perf_counter

import torch
import torch.distributed as dist
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from gpt_lab.evaluate.core import (
    first_minimum, load_core_bundle, load_core_task, prepare_example, rank_example_indices,
)
from gpt_lab.model.checkpoint import build_model
from gpt_lab.model.flash_attn import require_flash_attn3
from gpt_lab.utils.distributed import cleanup_dist_groups, init_dist_groups
from fa4 import FA4_REPO, FA4_REVISION, NativeFA4
from parity import ByteTokenizer, check_sibling_isolation, cropped_sibling_examples, small_model
from scoring import pack_choices, separate_choices, separate_choices_with_prefix_scoring, tree_masks


ARMS = ("A", "B", "B-score", "C")
BF16_NLL_ATOL = 0.05


def check_native_fa4(model, fa4, device):
    tokenizer = ByteTokenizer()
    aligned, mutated = cropped_sibling_examples(tokenizer)
    original = pack_choices(*aligned)
    for block in model.blocks:
        block.attention.flash_attn.flash_attn_func = fa4
    reference, _ = separate_choices(model, *aligned, tokenizer.get_bos_token_id(), device)
    packed, max_b_diff = check_sibling_isolation(
        original, pack_choices(*mutated),
        lambda layout: fa4.packed_choices(model, layout, device, capture_logits=True),
    )
    torch.testing.assert_close(torch.tensor(packed), torch.tensor(reference), atol=BF16_NLL_ATOL, rtol=0)
    assert first_minimum(packed) == first_minimum(reference), (packed, reference)

    # Uniform QK and one channel per key expose the native mask in the output.
    length = len(original.tokens)
    q = torch.zeros((1, length, 1, 64), device=device, dtype=torch.bfloat16)
    v = torch.eye(length, 64, device=device, dtype=q.dtype).view_as(q)
    max_window_diff = 0.0
    with fa4.packed_layout(original, device) as (positions, branches):
        for window, mask in zip(model.window_sizes, tree_masks(positions, branches, model.window_sizes)):
            output = fa4(q, q, v, dropout_p=0, causal=True, window_size=window, use_packed=False)
            actual = output[0, :, 0, :length].float()
            expected = mask.float() / mask.sum(dim=-1, keepdim=True)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0.01)
            max_window_diff = max(max_window_diff, (actual - expected).abs().max().item())
    return {"branch_b_max_logit_abs": max_b_diff, "window_max_weight_abs": max_window_diff}


def score_examples(mode, cases, model, tokenizer, device, fa4):
    scores = {}
    packed_count = 0
    pad = tokenizer.get_bos_token_id()
    for label, idx, data, meta in cases:
        tokens, starts, ends = prepare_example(idx, tokenizer, data, meta, model.max_seq_len)
        packed = pack_choices(tokens, starts, ends) if mode == "C" else None
        if packed is not None:
            nll, _ = fa4.packed_choices(model, packed, device)
            packed_count += 1
        elif mode == "B-score":
            nll = separate_choices_with_prefix_scoring(model, tokens, starts, ends, pad, device)
        else:
            nll, _ = separate_choices(model, tokens, starts, ends, pad, device)
        scores[label, idx] = nll, first_minimum(nll)
    return scores, packed_count


def probe_kernel(mode, case, model, tokenizer, device, fa4):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        _, packed = score_examples(mode, [case], model, tokenizer, device, fa4)
        torch.cuda.synchronize(device)
    if mode == "C" and packed != 1:
        raise RuntimeError("The FA4 mask_mod profiler probe must run a packed example")
    symbols = ("flashattnfwdsm90", "flash_fwd_kernel") if mode == "A" else ("flashattentionforwardsm90",)
    names = sorted({event.name for event in trace.events() if event.device_type == DeviceType.CUDA})
    matched = [name for name in names if any(symbol in name.lower() for symbol in symbols)]
    if not matched:
        raise RuntimeError(f"No expected native attention kernel for {mode}. CUDA events: {names}")
    return matched


def measure(mode, cases, probe, model, tokenizer, device, fa4):
    device_sync = torch.cuda.synchronize
    before_compiles, before_seconds = fa4.compiles, fa4.compile_seconds
    device_sync(device)
    dist.barrier()
    start = perf_counter()
    score_examples(mode, cases, model, tokenizer, device, fa4)
    device_sync(device)
    warmup_seconds = perf_counter() - start
    kernels = probe_kernel(mode, probe, model, tokenizer, device, fa4)
    compile_stats = {
        "compiles": fa4.compiles - before_compiles,
        "compilation_seconds": fa4.compile_seconds - before_seconds,
    }
    before_compiles = fa4.compiles
    torch.cuda.reset_peak_memory_stats(device)
    device_sync(device)
    dist.barrier()
    start = perf_counter()
    scores, packed = score_examples(mode, cases, model, tokenizer, device, fa4)
    device_sync(device)
    elapsed = perf_counter() - start
    if fa4.compiles != before_compiles:
        raise RuntimeError(f"FA4 compiled during the timed {mode} sweep")
    return scores, {
        "rank": dist.get_rank(), "gpu": torch.cuda.get_device_name(device),
        "elapsed_seconds": elapsed, "warmup_seconds": warmup_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "packed_questions": packed, "kernel_symbols": kernels, **compile_stats,
    }


def margin(nll):
    if len(nll) < 2 or not all(math.isfinite(loss) for loss in nll):
        return None
    best, second = sorted(nll)[:2]
    return second - best


def compare(reference, candidate):
    assert reference.keys() == candidate.keys()
    result = dict(max_nll_abs=0.0, outside_tolerance=0, paired_nan=0, flips=[])
    for (label, idx), (ref_nll, ref_winner) in reference.items():
        nll, winner = candidate[label, idx]
        assert len(nll) == len(ref_nll)
        for before, after in zip(ref_nll, nll):
            if math.isnan(before) and math.isnan(after):
                result["paired_nan"] += 1
                continue
            if not math.isfinite(before) or not math.isfinite(after):
                result["outside_tolerance"] += 1
                continue
            diff = abs(before - after)
            result["outside_tolerance"] += int(diff > BF16_NLL_ATOL)
            result["max_nll_abs"] = max(result["max_nll_abs"], diff)
        if ref_winner != winner:
            result["flips"].append(dict(
                task=label, example_index=idx, reference_winner=ref_winner, candidate_winner=winner,
                reference_margin=margin(ref_nll), candidate_margin=margin(nll),
            ))
    return result


@torch.inference_mode()
def run(args, context):
    device = context.device
    model, tokenizer, checkpoint_data, _ = build_model(
        model_name=args.model_name, run_name=args.run_name, step=args.step,
        model_cachedir=args.model_dir, phase="eval", comp_info=context, attn_impl="fa3",
    )
    del checkpoint_data
    if model.config.positional_encoding != "rope":
        raise ValueError("The benchmark requires a RoPE checkpoint")
    data_path, tasks, _ = load_core_bundle()
    cases, unique_count = [], 0
    for task in tasks:
        if task["icl_task_type"] == "multiple_choice":
            data, meta = load_core_task(data_path, task, args.max_per_task)
            unique_count += len(data)
            cases.extend((task["label"], idx, data, meta) for idx in rank_example_indices(len(data)))
    packed_probe = next((
        (label, idx, data, meta) for label, idx, data, meta in cases
        if pack_choices(*prepare_example(idx, tokenizer, data, meta, model.max_seq_len)) is not None
    ), None)
    if packed_probe is None:
        raise RuntimeError("Each rank needs at least one eligible packed question for its kernel probe")

    fixture_model = small_model(device, head_dim=64)
    fa4 = NativeFA4(model.window_sizes + fixture_model.window_sizes, device)
    dispatch = [block.attention.flash_attn for block in model.blocks]
    fa3_calls = [entry.flash_attn_func for entry in dispatch]
    fa3_kernel = require_flash_attn3()
    fa4.cute.compile = fa4.timed_compile
    records, scores = {}, {}
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preflight = check_native_fa4(fixture_model, fa4, device)
        preflight.update(compilation_seconds=fa4.compile_seconds, compiles=fa4.compiles)
        print(f"Rank {context.rank} FA4 preflight passed: {preflight}", flush=True)
        del fixture_model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for mode in ARMS:
                for entry, fa3_call in zip(dispatch, fa3_calls):
                    entry.flash_attn_func = fa3_call if mode == "A" else fa4
                scores[mode], records[mode] = measure(mode, cases, packed_probe, model, tokenizer, device, fa4)
    finally:
        fa4.cute.compile = fa4.original_compile
        for entry, fa3_call in zip(dispatch, fa3_calls):
            entry.flash_attn_func = fa3_call

    parity = {f"{mode}_vs_A": compare(scores["A"], scores[mode]) for mode in ARMS[1:]}
    parity["C_vs_B-score"] = compare(scores["B-score"], scores["C"])
    per_rank = [None] * context.world_size
    dist.all_gather_object(per_rank, dict(ids=list(scores["A"]), measurements=records, parity=parity, preflight=preflight))
    ids = [example for rank in per_rank for example in rank["ids"]]
    if len(ids) != len(set(ids)) or len(ids) != unique_count:
        raise RuntimeError("CORE sharding did not cover each selected question exactly once")
    if context.rank == 0:
        throughput = {
            mode: unique_count / max(rank["measurements"][mode]["elapsed_seconds"] for rank in per_rank)
            for mode in ARMS
        }
        print(json.dumps({
            "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": "bfloat16",
            "checkpoint": vars(args), "model_config": model.config.model_dump(mode="json"),
            "layer_windows": model.window_sizes,
            "fa3_source": fa3_kernel.__file__, "fa4_source": fa4.origin,
            "fa4_repo": FA4_REPO, "fa4_revision": FA4_REVISION, "unique_questions": unique_count,
            "nll_atol": BF16_NLL_ATOL, "nll_rtol": 0, "questions_per_second": throughput,
            "speedup_over_A": {mode: value / throughput["A"] for mode, value in throughput.items()},
            "per_rank": per_rank,
        }, indent=2))
    if any(result["outside_tolerance"] for rank in per_rank for result in rank["parity"].values()):
        raise SystemExit("Per-choice BF16 NLL exceeded tolerance")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CORE multiple-choice scoring on two H100s.")
    parser.add_argument("--model-dir", required=True, help="GPT-Lab checkpoint cache root")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--max-per-task", type=int, default=64)
    args = parser.parse_args()
    if args.max_per_task < 11 or args.step < 0:
        parser.error("use at least 11 examples per task and an explicit nonnegative checkpoint step")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2 or not torch.cuda.is_available():
        parser.error("launch on two H100 GPUs with torchrun --standalone --nproc-per-node=2")

    context = init_dist_groups("cuda")
    try:
        if "H100" not in torch.cuda.get_device_name(context.device) or context.compute_dtype != torch.bfloat16:
            raise RuntimeError("Each rank requires an H100 and BF16 compute")
        run(args, context)
    finally:
        cleanup_dist_groups()


if __name__ == "__main__":
    main()
