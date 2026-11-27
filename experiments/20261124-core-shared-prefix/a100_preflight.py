"""Validate matched attention backends on the one-A100 CORE experiment."""
from contextlib import contextmanager, nullcontext
from functools import partial
from importlib import import_module, util
from pathlib import Path
from statistics import median
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark import check_native_fa4, compare, probe_kernels
from benchmark_a100 import ARMS, WIDTHS, attempt, model_command, profile_shapes, summarize_trial, validate, write_json
from batched_scoring import check_wide_fixture
from core_timing import paired
from fa4 import FA4_REPO, FA4_REVISION, NativeFA4
from fixture_model import small_model
from packing_pair import comparison_plans, plan_counts, score_plan


REPLAY = {("commonsense_qa", 6), ("commonsense_qa", 7), ("bigbench_language_identification", 13)}


def load_fa4_source():
    from huggingface_hub import snapshot_download

    # The locked Hub kernels client conflicts with Transformers' import-time registration.
    snapshot = Path(snapshot_download(FA4_REPO, revision=FA4_REVISION, allow_patterns="build/torch-cuda/**"))
    package = snapshot / "build/torch-cuda"
    spec = util.spec_from_file_location("_gptlab_core_sm80_fa4", package / "__init__.py",
                                       submodule_search_locations=[str(package)])
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def matched_sdpa(model, backend):
    if hasattr(model, "blocks"):
        flash_attn = import_module("gpt_lab.model.flash_attn")
        fallback = flash_attn._sdpa_fallback

        def repeat_kv(q, k, v, **kwargs):
            repeat = q.size(-2) // k.size(-2)
            if repeat > 1:
                k = k.repeat_interleave(repeat, dim=-2)
                v = v.repeat_interleave(repeat, dim=-2)
            return fallback(q, k, v, **(kwargs | {"use_gqa": False}))

        gqa = patch.object(flash_attn, "_sdpa_fallback", repeat_kv)
    else:
        from transformers.integrations import sdpa_attention

        gqa = patch.object(sdpa_attention, "use_gqa_in_sdpa", return_value=False)
    with gqa, sdpa_kernel(backend):
        yield


@contextmanager
def native_attention(model, fa4):
    if hasattr(model, "blocks"):
        originals = [block.attention.flash_attn.flash_attn_func for block in model.blocks]
        try:
            for block in model.blocks:
                block.attention.flash_attn.flash_attn_func = fa4
            yield
        finally:
            for block, function in zip(model.blocks, originals):
                block.attention.flash_attn.flash_attn_func = function
    else:
        from transformers import AttentionInterface

        def forward(module, q, k, v, attention_mask, *, dropout, scaling, **kwargs):
            if attention_mask is not None:
                raise ValueError("Native FA4 owns attention visibility")
            return fa4(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout,
                       causal=True, window_size=(-1, 0), use_packed=False, softmax_scale=scaling), None

        name = "gptlab_core_fa4_sm80"
        AttentionInterface.register(name, forward)
        original = model.config._attn_implementation
        model.config._attn_implementation = name
        try:
            yield
        finally:
            model.config._attn_implementation = original


def option_records(scores):
    return [{"task": task, "index": i, "option_nll": values, "winner": winner}
            for (task, i), (values, winner) in scores.items()]


def replay_questions(examples, model, pad, device, *, fa4=None):
    groups = [[i] for i, ex in enumerate(examples) if (ex.label, ex.index) in REPLAY]
    return {
        arm: score_plan([(arm == "packed", group) for group in groups], examples, model, pad,
                        device, fa4=fa4, width_buckets=WIDTHS)
        for arm in ARMS
    }


def validated_sweep(args, device, model, tokenizer, configs, examples, baselines, reference, task_reference, identity):
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise RuntimeError("This validation requires SM80")
    pad = tokenizer.get_bos_token_id()
    report = {**identity, "fa4_revision": FA4_REVISION, "backends": {}, "sizes": None}
    worker_args = model_command(args)

    def save():
        write_json(args.output / "validated-report.json", report)

    replay_reference = {key: value for key, value in reference.items() if key in REPLAY}
    report["replay_master"] = option_records(replay_reference)

    def replay(fa4=None):
        scores = replay_questions(examples, model, pad, device, fa4=fa4)
        return {
            arm: {"options": option_records(scores[arm]), "vs_master": compare(replay_reference, scores[arm]),
                  "vs_same_backend_separate": compare(scores["separate"], scores[arm])}
            for arm in ARMS
        }

    with torch.autocast("cuda", dtype=torch.bfloat16):
        report["replay_default_sdpa"] = attempt(
            replay, report, "replay_default_sdpa",
        )
        save()
        fixture = small_model(device, head_dim=64)
        for name in ("matched_efficient", "matched_math", "native_fa4"):
            if name == "matched_math" and report["backends"]["matched_efficient"].get("predictions_agree"):
                continue
            record = report["backends"][name] = {"sizes": {}}
            fa4 = None
            backend = SDPBackend.EFFICIENT_ATTENTION if name == "matched_efficient" else SDPBackend.MATH
            try:
                if name == "native_fa4":
                    fa4 = NativeFA4((*model.window_sizes, *fixture.window_sizes), device, kernel=load_fa4_source())
                    record["source"] = fa4.origin
                    with patch.object(fa4.cute, "compile", fa4.timed_compile):
                        record["short_window_probe"] = check_native_fa4(fixture, fa4, device)
                        record["isolation_fixture"] = check_wide_fixture(fixture, 256, device, fa4=fa4)
                else:
                    with matched_sdpa(fixture, backend):
                        record["isolation_fixture"] = check_wide_fixture(fixture, 256, device)
            except (RuntimeError, AssertionError, ImportError, OSError, ValueError, AttributeError, TypeError) as error:
                record["fixture_error"] = repr(error)
                save()
                continue

            attention = partial(native_attention, model, fa4) if fa4 else partial(matched_sdpa, model, backend)
            compiler = patch.object(fa4.cute, "compile", fa4.timed_compile) if fa4 else nullcontext()

            def run(plan):
                return score_plan(plan, examples, model, pad, device, fa4=fa4, width_buckets=WIDTHS)

            with attention(), compiler:
                record["replay"] = attempt(lambda: replay(fa4), record, "replay")
                save()
                if "execution_error" in record or "allocation_error" in record:
                    continue
                first = configs[1]
                try:
                    if fa4:
                        record["kernel"] = {
                            arm: probe_kernels(lambda: run(first[arm][:1]), device, ("flashattentionforwardsm80",))
                            for arm in ARMS
                        }
                    else:
                        record["kernel"] = {arm: profile_shapes(first[arm][:1], examples, run) for arm in ARMS}
                        operators = [record["kernel"][arm]["sdpa_operators"] for arm in ARMS]
                        expected = ("_scaled_dot_product_efficient_attention" if name == "matched_efficient"
                                    else "_scaled_dot_product_attention_math")
                        assert operators[0] == operators[1] and all(expected in names[0] for names in operators), operators
                except (RuntimeError, AssertionError, IndexError) as error:
                    record["kernel_error"] = repr(error)
                    save()
                    continue
                for size, plans in configs.items():
                    entry = record["sizes"][size] = {arm: plan_counts(plans[arm], arm) for arm in ARMS}
                    for arm in ARMS:
                        case = entry[arm]
                        scores = attempt(lambda: run(plans[arm]), case, "parity_warmup")
                        if scores is not None:
                            case["parity"] = validate(reference, scores, examples, baselines, task_reference)
                            failures = case["parity"]["flips"]
                            case["predictions_agree"] = (
                                case["parity"]["passed"] and not failures and case["parity"]["task_scores_exact"]
                            )
                            if failures:
                                changed = {(flip["task"], flip["example_index"]) for flip in failures}
                                case["option_nlls_for_flips"] = option_records({key: scores[key] for key in sorted(changed)})
                    save()
                eligible_cases = [case for entry in record["sizes"].values() for case in entry.values()]
                record["predictions_agree"] = all(case.get("predictions_agree", False) for case in eligible_cases)
                if fa4:
                    record["warmup_compilation"] = {"seconds": fa4.compile_seconds, "compiles": fa4.compiles}
                save()
                if not any(case.get("predictions_agree") for case in eligible_cases):
                    record["timing_skipped"] = "no configuration passed validation against pinned master"
                    save()
                    continue

                baseline = {}
                for size, plans in configs.items():
                    entry = record["sizes"][size]
                    for case in entry.values():
                        case["trials"] = []
                    for i in range(3):
                        for arm in (ARMS if i % 2 == 0 else ARMS[::-1]):
                            case = entry[arm]
                            if not case.get("predictions_agree") or "execution_error" in case or "allocation_error" in case:
                                continue
                            sample = attempt(lambda: summarize_trial(lambda: run(plans[arm])), case, "timing")
                            if sample is not None:
                                case["trials"].append(sample)
                    for arm in ARMS:
                        case = entry[arm]
                        if case["trials"]:
                            elapsed = median(sample["seconds"] for sample in case["trials"])
                            case.update(median_seconds=elapsed, questions_per_second=len(examples) / elapsed,
                                        peak_allocated_bytes=max(sample["peak_allocated_bytes"] for sample in case["trials"]))
                            if size == 1:
                                baseline[arm] = elapsed
                            case["batching_speedup_vs_size_1"] = baseline[arm] / elapsed if arm in baseline else None
                    if all(len(entry[arm]["trials"]) == 3 for arm in ARMS):
                        entry["median_paired_packing_speedup"] = median(
                            a["seconds"] / b["seconds"] for a, b in zip(*(entry[arm]["trials"] for arm in ARMS))
                        )
                    save()
                if fa4:
                    record["eligible_sweep_extra_compiles"] = fa4.compiles - record["warmup_compilation"]["compiles"]
                    save()
            directory = args.output / name
            directory.mkdir(exist_ok=True)
            task_args = SimpleNamespace(output=directory, configuration="both", repeats=3)
            with patch.object(fa4.cute, "compile", fa4.timed_compile) if fa4 else nullcontext():
                record["all_tasks"] = attempt(
                    lambda: paired(task_args, model, tokenizer, device, extra_worker_args=worker_args,
                                   fa4=fa4, require_predictions=True, width_buckets=WIDTHS, backend=attention),
                    record, "all_tasks",
                )
                if fa4:
                    record["total_compilation"] = {"seconds": fa4.compile_seconds, "compiles": fa4.compiles}
                save()
