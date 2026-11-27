from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from time import perf_counter

import torch

from gpt_lab.evaluate.core import forward_rows
from scoring import packed_scores


FA4_REPO = "kernels-community/flash-attn4"
FA4_REVISION = "a3944dc87df804a42f0248d16c780fed90f84508"  # Hub v0


class NativeFA4:
    def __init__(self, windows, device, *, kernel=None):
        import cutlass
        import cutlass.cute as cute

        if kernel is None:
            from kernels import get_kernel

            kernel = get_kernel(FA4_REPO, revision=FA4_REVISION)
        if FA4_REVISION not in Path(kernel.__file__).parts:
            raise RuntimeError(f"Unexpected FA4 source path: {kernel.__file__}")
        self.origin = kernel.__file__
        self.separate = kernel.flash_attn_func
        interface = import_module(self.separate.__module__)
        self.forward = interface._flash_attn_fwd
        scalar_to_ssa = interface.utils.scalar_to_ssa

        @cute.jit
        def branch_mask(batch, head, q, k, seqlen_info, aux):
            positions, branches, window = aux
            row = batch[0]
            q_idx = cutlass.min(q[0], seqlen_info.seqlen_q - 1)
            k_idx = cutlass.min(k[0], seqlen_info.seqlen_k - 1)
            q_pos = scalar_to_ssa(positions[row, q_idx], cutlass.Int32)
            k_pos = scalar_to_ssa(positions[row, k_idx], cutlass.Int32)
            q_branch = scalar_to_ssa(branches[row, q_idx], cutlass.Int32)
            k_branch = scalar_to_ssa(branches[row, k_idx], cutlass.Int32)
            left = scalar_to_ssa(window[0], cutlass.Int32)
            zero = scalar_to_ssa(0, cutlass.Int32)
            distance = q_pos - k_pos
            return (k_branch >= zero) & (distance >= zero) & ((k_branch == zero) | (k_branch == q_branch)) & (
                (left < zero) | (distance <= left)
            )

        self.mask = branch_mask
        self.windows = {
            left: torch.tensor([left], dtype=torch.int32, device=device)
            for left in {left for left, _ in windows}
        }
        self.coords = None
        self.compile_seconds = 0.0
        self.compiles = 0
        self.cute = cute
        self.original_compile = cute.compile

    def timed_compile(self, *args, **kwargs):
        start = perf_counter()
        try:
            return self.original_compile(*args, **kwargs)
        finally:
            self.compile_seconds += perf_counter() - start
            self.compiles += 1

    def __call__(self, q, k, v, *, dropout_p, causal, window_size, use_packed, softmax_scale=None):
        assert causal and dropout_p == 0 and not use_packed
        q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
        left, right = window_size
        assert right == 0
        if self.coords is None:
            return self.separate(q, k, v, causal=True, softmax_scale=softmax_scale,
                                 window_size=(None if left < 0 else left, 0))[0]
        return self.forward(
            q, k, v, causal=False, mask_mod=self.mask, softmax_scale=softmax_scale,
            aux_tensors=[*self.coords, self.windows[left]],
        )[0]

    @contextmanager
    def batched_layout(self, positions, branches):
        self.coords = positions, branches
        try:
            yield
        finally:
            self.coords = None

    @contextmanager
    def packed_layout(self, packed, device):
        positions = torch.tensor(packed.positions, dtype=torch.int32, device=device)
        branches = torch.tensor(packed.branches, dtype=torch.int32, device=device)
        with self.batched_layout(positions[None], branches[None]):
            yield positions, branches

    def packed_choices(self, model, packed, device, *, capture_logits=False):
        row = torch.tensor([packed.tokens], device=device)
        with self.packed_layout(packed, device) as (positions, _):
            outputs = forward_rows(model, row, position_ids=positions.long())
        return packed_scores(outputs, row, packed, capture_logits=capture_logits)
