"""Separate CPU stage, MPS interval, and Metal trace instrumentation for CORE."""
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from ctypes import CDLL, c_int, c_uint64
from functools import wraps
from time import perf_counter
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import batched_scoring
import packing_pair
from batching import load_examples
from core_timing import planned_winners, timed
from gpt_lab.evaluate import core
from gpt_lab.model import layers


@contextmanager
def metal_markers(library):
    marks = CDLL(str(library.resolve()))
    marks.core_signposts_init()
    marks.core_signpost_begin.argtypes = (c_int,)
    marks.core_signpost_begin.restype = c_uint64
    marks.core_signpost_end.argtypes = (c_int, c_uint64)

    def annotate(phase, fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            token = marks.core_signpost_begin(phase)
            try:
                return fn(*args, **kwargs)
            finally:
                marks.core_signpost_end(phase, token)
        return wrapped

    with patch.object(packing_pair, "collate_group", annotate(1, packing_pair.collate_group)), \
         patch.object(packing_pair, "score_group", annotate(2, packing_pair.score_group)):
        token = marks.core_signpost_begin(0)
        try:
            yield
        finally:
            marks.core_signpost_end(0, token)


class Measurements(TorchDispatchMode):
    def __init__(self, rope):
        super().__init__()
        self.rope_ptr = rope.data_ptr()
        self.cpu = defaultdict(float)
        self.counts = defaultdict(int)
        self.events = defaultdict(list)
        self.mask_base = None
        self.mask_peak = 0

    def measure(self, name, fn, *, device=False):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            start = perf_counter()
            if device:
                before, after = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
                before.record()
            result = fn(*args, **kwargs)
            if device:
                after.record()
                self.events[name].append((before, after))
            self.cpu[name] += perf_counter() - start
            self.counts[name] += 1
            return result
        return wrapped

    def track_mask(self, fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            self.mask_base = torch.mps.current_allocated_memory()
            try:
                return fn(*args, **kwargs)
            finally:
                self.mask_base = None
        return wrapped

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        run = func
        if func is torch.ops.aten.index_select.default and args[0].data_ptr() == self.rope_ptr:
            run = self.measure("rope_table_gather", func, device=True)
        result = run(*args, **(kwargs or {}))
        if self.mask_base is not None:
            self.mask_peak = max(self.mask_peak, torch.mps.current_allocated_memory() - self.mask_base)
        return result

    def report(self):
        return {
            name: {"host_seconds": value, "calls": self.counts[name],
                   **({"mps_interval_seconds": sum(a.elapsed_time(b) for a, b in self.events[name]) / 1000}
                      if name in self.events else {})}
            for name, value in self.cpu.items()
        }


def components(model, tokenizer, device):
    measured = Measurements(model.pe_cache)
    pad = tokenizer.get_bos_token_id()
    with ExitStack() as stack:
        for prefix in ("mc", "schema", "lm"):
            for operation, name in (("render_prompts", "rendering"), ("batch_sequences", "tokenization_and_spans")):
                target = f"{operation}_{prefix}"
                stack.enter_context(patch.object(core, target, measured.measure(name, getattr(core, target))))
        all_examples, _ = measured.measure("prompt_preparation_and_packing", load_examples)(tokenizer)
    examples = [ex for ex in all_examples if ex.task_type == "multiple_choice"]
    plan = measured.measure("grouping", packing_pair.comparison_plans)(examples, 4)["all"]["packed"]
    shapes = []
    for group in plan:
        packed, indices, batch = packing_pair.collate_group(group, examples, pad, model.max_seq_len, ())
        lengths = ([len(examples[i].packed.tokens) for i in indices] if packed
                   else [len(row) for i in indices for row in examples[i].prepared[0]])
        shapes.append({"packed": packed, "rows": batch.tokens.size(0), "width": batch.tokens.size(1),
                       "padding_tokens": batch.tokens.numel() - sum(lengths),
                       "max_logical_position": int(batch.positions.max()) if packed else max(lengths) - 1})

    def run():
        return planned_winners(plan, examples, model, pad, device, prefetch=True, defer=True)

    warm = run()
    stack_width = batched_scoring.stack_at_width
    position_stack = measured.measure("cpu_position_ids", stack_width)

    def track_position_stack(rows, pad_id, width):
        # The frozen fixture uses BOS 256 and reserves zero for padded positions.
        return (position_stack if pad_id == 0 else stack_width)(rows, pad_id, width)

    with ExitStack() as stack:
        patches = [(packing_pair, "collate_group", "collation", False),
                   (layers, "apply_rope", "qk_rotation", True),
                   (model.lm_head, "forward", "vocab_projection", True),
                   (model, "loss_fn", "fp32_cross_entropy", True),
                   (batched_scoring, "reduce_options", "option_reduction", True),
                   (torch.Tensor, "cpu", "host_copy_and_wait", False),
                   (torch.Tensor, "tolist", "host_tolist", False)]
        for target, attr, label, gpu in patches:
            stack.enter_context(patch.object(target, attr, measured.measure(label, getattr(target, attr), device=gpu)))
        for block in model.blocks:
            attention = block.attention.flash_attn
            stack.enter_context(patch.object(attention, "flash_attn_func",
                                             measured.measure("attention", attention.flash_attn_func, device=True)))
        stack.enter_context(patch.object(batched_scoring, "stack_at_width", track_position_stack))
        stack.enter_context(patch.object(batched_scoring, "tree_masks", measured.track_mask(
            measured.measure("dense_branch_masks", batched_scoring.tree_masks, device=True))))
        stack.enter_context(measured)
        elapsed, actual = timed(run, device)
    assert warm == actual
    return {"instrumented_seconds": elapsed, "stages": measured.report(), "batch_shapes": shapes,
            "sampled_peak_mask_tensor_bytes_above_call_entry": measured.mask_peak}
