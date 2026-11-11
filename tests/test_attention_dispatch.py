from importlib import import_module
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("case", ["packed", "missing_packed", "gqa", "unequal_lengths"])
def test_attention_selects_layout_and_preserves_outputs_and_gradients(monkeypatch, backend, case):
    attention = import_module("gpt_lab.model.flash_attn")
    torch.manual_seed(42)
    heads = 1 if case == "gqa" else 2
    key_length = 5 if case == "unequal_lengths" else 4
    inputs = (
        torch.randn(2, 4, 2, 8, requires_grad=True),
        torch.randn(2, key_length, heads, 8, requires_grad=True),
        torch.randn(2, key_length, heads, 8, requires_grad=True),
    )
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    calls = []

    def reference(q, k, v, **kwargs):
        return attention._sdpa_fallback(q, k, v, use_gqa=q.size(-2) != k.size(-2), **kwargs)

    # These substitute only the CUDA entrypoints; arithmetic uses real CPU SDPA.
    def separate(q, k, v, **kwargs):
        calls.append(("separate", kwargs))
        return reference(q, k, v, **kwargs)

    def packed(qkv, **kwargs):
        assert qkv.shape == (2, 4, 3, 2, 8)
        assert qkv.is_contiguous()
        calls.append(("packed", kwargs))
        return reference(*qkv.unbind(dim=2), **kwargs)

    kernel = SimpleNamespace(flash_attn_func=separate)
    if case != "missing_packed":
        kernel.flash_attn_qkvpacked_func = packed
    monkeypatch.setattr(attention, "require_flash_attn2" if backend == "fa2" else "require_flash_attn3",
                        lambda: kernel)
    kwargs = dict(softmax_scale=0.25, causal=True, window_size=(1, 0), deterministic=True)

    output = attention.flash_attn_func(*inputs, backend=backend, **kwargs)
    expected = reference(*reference_inputs, **kwargs)

    expected_kwargs = kwargs | ({"dropout_p": 0.0, "alibi_slopes": None} if backend == "fa2" else {})
    assert calls == [("packed" if case == "packed" else "separate", expected_kwargs)]
    torch.testing.assert_close(output, expected)
    gradients = torch.autograd.grad(output.square().sum(), inputs)
    expected_gradients = torch.autograd.grad(expected.square().sum(), reference_inputs)
    for actual, wanted in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, wanted)
