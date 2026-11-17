"""Separate Q/K preprocessing against the previous joint arithmetic on CPU."""
import copy
import sys
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from gpt_lab.model.layers import CausalSelfAttention
from gpt_lab.model.utils import apply_rope, precompute_rope
from gpt_lab.utils.schemas import TransformerConfig


def stub_kernel(monkeypatch, calls, packed=True, backend="fa3"):
    dispatch = import_module("gpt_lab.model.flash_attn")

    def separate(q, k, v, **kwargs):
        calls.append("separate")
        return dispatch._sdpa_fallback(q, k, v, use_gqa=q.size(-2) != k.size(-2), **kwargs)

    def qkvpacked(qkv, *, num_heads_q=None, **kwargs):
        if num_heads_q is None:
            calls.append("packed5d")
            q, k, v = qkv.unbind(2)
        else:
            calls.append("packed4d")
            assert qkv.ndim == 4 and qkv.is_contiguous()
            hkv = (qkv.size(-2) - num_heads_q) // 2
            q, k, v = qkv.split((num_heads_q, hkv, hkv), dim=-2)
        return dispatch._sdpa_fallback(q, k, v, use_gqa=q.size(-2) != k.size(-2), **kwargs)

    def kvcache(q, k_cache, v_cache, k, v, **kwargs):
        calls.append("kvcache")
        return dispatch._sdpa_fallback(q, k, v, causal=True)

    kernel = SimpleNamespace(flash_attn_func=separate, flash_attn_with_kvcache=kvcache)
    if packed:
        kernel.flash_attn_qkvpacked_func = qkvpacked
    if "/" in backend:
        monkeypatch.setattr(import_module("gpt_lab.utils.import_utils"), "_attention_kernels", {})
        monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=lambda *args, **kw: kernel))
    else:
        monkeypatch.setattr(dispatch, "require_flash_attn3" if backend == "fa3" else "require_flash_attn2", lambda: kernel)


def joint_reference(attention, x, values, rope, window):
    b, t, _ = x.shape
    qkv = attention.qkv_proj(x).view(b, t, attention.n_heads + 2 * attention.n_kv_heads, attention.d_head)
    qk, v = qkv.split((attention.n_heads + attention.n_kv_heads, attention.n_kv_heads), -2)
    if rope is not None:
        qk = apply_rope(qk, rope)
    if attention.norm_before_attn:
        qk = attention.norm(qk)
    if values is not None:
        gate = 3 * torch.sigmoid(attention.ve_gate(x[..., :attention.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * values.view_as(v)
    q, k = qk.split((attention.n_heads, attention.n_kv_heads), -2)
    # The legacy 4-D packed call keeps cat's promoted dtype without CUDA AMP.
    dtype = torch.promote_types(q.dtype, v.dtype) if (attention.training and
        attention.use_packed_attn_kernel and attention._packed_fa3) else q.dtype
    q, k, v = (tensor.to(dtype) for tensor in (q, k, v))
    dispatch = import_module("gpt_lab.model.flash_attn")
    out = dispatch._sdpa_fallback(q, k, v, causal=True, window_size=window,
                                  use_gqa=attention.n_heads != attention.n_kv_heads)
    return attention.o_proj(out.reshape(b, t, attention.d_model))


def test_qk_preprocessing_uses_head_views_and_rope_before_norm(monkeypatch):
    layers = import_module("gpt_lab.model.layers")
    cfg = TransformerConfig(d_model=16, d_ffn=32, n_heads=2, n_kv_heads=1,
                            d_head=8, n_layers=1, attn_impl="sdpa", norm_before_attn=True)
    attention = CausalSelfAttention(cfg).cpu()
    calls = []
    projected = []
    attention.qkv_proj.register_forward_hook(lambda module, args, output: projected.append(output))
    original_norm = attention.norm

    def rope(x, cache):
        assert x.untyped_storage().data_ptr() == projected[0].untyped_storage().data_ptr()
        calls.append(("rope", x.size(-2)))
        return apply_rope(x, cache)

    def norm(x):
        calls.append(("norm", x.size(-2)))
        return original_norm(x)

    monkeypatch.setattr(layers, "apply_rope", rope)
    monkeypatch.setattr(attention, "norm", norm)
    attention(torch.randn(2, 7, 16), rope_cache=precompute_rope(7, 8, device=torch.device("cpu")))
    assert len(projected) == 1
    assert calls == [("rope", 2), ("rope", 1), ("norm", 2), ("norm", 1)]


@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("backend", ["fa2", "fa3", "example/attention"])
@pytest.mark.parametrize("use_packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("normalization,rope_enabled,norm_enabled,values_enabled", [
    ("rms", True, True, True),
    ("layer", True, True, True),
    ("rms", False, False, True),
    ("rms", True, False, False),
    ("rms", False, True, False),
])
def test_separate_qk_outputs_and_gradients(monkeypatch, kv_heads, backend, use_packed, dtype, normalization,
                                      rope_enabled, norm_enabled, values_enabled):
    calls = []
    stub_kernel(monkeypatch, calls, backend=backend)
    torch.manual_seed(42)
    config = TransformerConfig(d_model=16, d_ffn=32, n_heads=2, n_kv_heads=kv_heads,
        d_head=8, n_layers=1, attn_impl=backend, normalization=normalization,
        norm_before_attn=norm_enabled, norm_eps=3e-4, use_packed_attn_kernel=use_packed)
    attention = CausalSelfAttention(config).to(device="cpu", dtype=dtype).train()
    reference = copy.deepcopy(attention)
    x = torch.randn(2, 7, 16, dtype=dtype, requires_grad=True)
    values = torch.randn(2, 7, kv_heads * 8, dtype=torch.float32, requires_grad=True) if values_enabled else None
    rx = x.detach().clone().requires_grad_()
    rv = values.detach().clone().requires_grad_() if values_enabled else None
    rope = precompute_rope(7, 8, device=torch.device("cpu")) if rope_enabled else None
    output, _ = attention(x, value_embeds=values, rope_cache=rope, window_size=(2, 0))
    expected = joint_reference(reference, rx, rv, rope, (2, 0))
    expected_call = "packed4d" if backend == "fa3" else "packed5d" if kv_heads == 2 else "separate"
    assert calls == [expected_call if use_packed else "separate"]
    torch.testing.assert_close(output, expected)
    dy = torch.randn_like(output)
    output.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(x.grad, rx.grad)
    if values_enabled:
        torch.testing.assert_close(values.grad, rv.grad)
        assert attention.ve_gate.weight.grad.norm() > 0
    for (name, parameter), (_, baseline) in zip(attention.named_parameters(), reference.named_parameters()):
        if baseline.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, baseline.grad, msg=name)


@pytest.mark.parametrize("use_packed", [False, True])
@pytest.mark.parametrize("backend,training,packed,cache,expected_call", [
    ("fa3", False, True, False, "separate"),
    ("fa3", False, True, True, "kvcache"),
    ("fa3", True, False, False, "separate"),
    ("fa3", True, True, True, "packed5d"),
    ("fa2", True, True, False, "packed5d"),
    ("fa2", True, False, False, "separate"),
])
def test_layout_for_inference_and_other_kernel_capabilities(monkeypatch, backend, training,
                                                          packed, cache, expected_call, use_packed):
    calls = []
    stub_kernel(monkeypatch, calls, packed=packed, backend=backend)
    cfg = TransformerConfig(d_model=16, d_ffn=32, n_heads=2, d_head=8, n_layers=1,
                            attn_impl=backend, norm_before_attn=True, use_packed_attn_kernel=use_packed)
    attention = CausalSelfAttention(cfg).cpu().train(training)
    x, values = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
    rope = precompute_rope(7, 8, device=torch.device("cpu"))
    kv_cache = None
    if cache:
        kv_cache = SimpleNamespace(check_sizes=lambda *args: None,
            layer=lambda index: (torch.zeros(2, 8, 2, 8), torch.zeros(2, 8, 2, 8)),
            seqlens=torch.zeros(2, dtype=torch.int32))
    output, _ = attention(x, value_embeds=values, rope_cache=rope, kv_cache=kv_cache)
    expected = joint_reference(attention, x, values, rope, None)
    torch.testing.assert_close(output, expected)
    assert calls == ["separate" if expected_call == "packed5d" and not use_packed else expected_call]


@pytest.mark.parametrize("use_packed", [False, True])
def test_separate_preprocessing_compiles_as_one_graph(monkeypatch, use_packed):
    calls = []
    stub_kernel(monkeypatch, calls)
    cfg = TransformerConfig(d_model=16, d_ffn=32, n_heads=2, n_kv_heads=1,
        d_head=8, n_layers=1, attn_impl="fa3", norm_before_attn=True, use_packed_attn_kernel=use_packed)
    attention = CausalSelfAttention(cfg).cpu().train()
    x, values = torch.randn(2, 7, 16), torch.randn(2, 7, 8)
    rope = precompute_rope(7, 8, device=torch.device("cpu"))
    expected = attention(x, value_embeds=values, rope_cache=rope)[0]
    compiled = torch.compile(attention, backend="eager", fullgraph=True)
    output = compiled(x, value_embeds=values, rope_cache=rope)[0]
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("backend", ["fa2", "fa3", "example/attention"])
@pytest.mark.parametrize("use_packed", [False, True])
def test_model_resolves_dispatch_before_forward(monkeypatch, backend, use_packed):
    calls = []
    stub_kernel(monkeypatch, calls, backend=backend)
    cfg = TransformerConfig(d_model=16, d_ffn=32, n_heads=2, d_head=8, n_layers=1,
                            attn_impl=backend, use_packed_attn_kernel=use_packed)
    attention = CausalSelfAttention(cfg).cpu()

    def unexpected(*args, **kwargs):
        pytest.fail("Regular forwards must use the bound attention callable")

    dispatch = import_module("gpt_lab.model.flash_attn")
    monkeypatch.setattr(dispatch, "_initialize_flash_attention", unexpected)
    monkeypatch.setattr(dispatch, "require_attention_kernel", unexpected)
    monkeypatch.setattr(dispatch, "require_flash_attn3", unexpected)
    monkeypatch.setattr(dispatch, "require_flash_attn2", unexpected)
    monkeypatch.setattr(dispatch, "_load_flash_attn_functions", unexpected)
    monkeypatch.setattr(dispatch, "flash_attn_func", unexpected)
    monkeypatch.setattr(dispatch, "flash_attn_qkvpacked_func", unexpected)
    monkeypatch.setattr(dispatch, "flash_attn_with_kvcache", unexpected)
    x = torch.randn(2, 7, 16)
    first = attention(x)[0]
    attention.eval()
    torch.testing.assert_close(attention(x)[0], first)
    attention.train()
    torch.testing.assert_close(attention(x)[0], first)
    attention.eval()
    cache = SimpleNamespace(check_sizes=lambda *args: None,
        layer=lambda index: (torch.zeros(2, 8, 2, 8), torch.zeros(2, 8, 2, 8)),
        seqlens=torch.zeros(2, dtype=torch.int32))
    torch.testing.assert_close(attention(x, kv_cache=cache)[0], first)
    training_call = ("packed4d" if backend == "fa3" else "packed5d") if use_packed else "separate"
    assert calls == [training_call, "separate", training_call, "kvcache"]


def test_sdpa_and_cache_are_selected_before_forward(monkeypatch):
    cfg = TransformerConfig(d_model=16, d_ffn=32, n_heads=2, n_kv_heads=1,
                            d_head=8, n_layers=1, attn_impl="sdpa")
    attention = CausalSelfAttention(cfg).cpu().eval()
    x = torch.randn(2, 7, 16)
    expected = attention(x)[0]

    def unexpected(*args, **kwargs):
        pytest.fail("SDPA execution must not select or load a backend")

    dispatch = import_module("gpt_lab.model.flash_attn")
    monkeypatch.setattr(dispatch, "_load_flash_attn_functions", unexpected)
    monkeypatch.setattr(dispatch, "_initialize_flash_attention", unexpected)
    monkeypatch.setattr(dispatch, "require_attention_kernel", unexpected)
    torch.testing.assert_close(attention(x)[0], expected)
    cache = SimpleNamespace(check_sizes=lambda *args: None,
        layer=lambda index: (torch.zeros(2, 8, 1, 8), torch.zeros(2, 8, 1, 8)),
        seqlens=torch.zeros(2, dtype=torch.int32))
    torch.testing.assert_close(attention(x, kv_cache=cache)[0], expected)
