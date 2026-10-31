"""CPU checks for explicit kernel dispatch and packed QKV semantics."""

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch

from gpt_lab.cli.parser import build_parser
from gpt_lab.model import layers
from gpt_lab.model.auto import AutoGPTConfig
from gpt_lab.model.utils import KVCache, precompute_rope
from gpt_lab.utils.schemas import TransformerConfig

attention = importlib.import_module("gpt_lab.model.flash_attn")


def config(**overrides):
    return TransformerConfig(vocab_size=64, d_model=32, n_heads=2, d_head=16,
                             n_layers=2, max_context=8, **overrides)


@pytest.mark.parametrize("backend", ["sdpa", "fa2", "fa3"])
def test_configuration_and_dispatch(monkeypatch, backend):
    args = build_parser().parse_args(["train", "auto", "--attn-impl", backend])
    assert args.attn_impl == backend
    assert AutoGPTConfig(attn_impl=backend).build_transformer_config(depth=2, vocab_size=64).attn_impl == backend
    assert config(attn_impl=backend, n_kv_heads=1).attn_impl == backend
    calls = []

    def kernel(q, k, v, **kwargs):
        calls.append(kwargs)
        assert q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
        assert q.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
        if backend == "fa3":
            assert "dropout_p" not in kwargs
            assert "alibi_slopes" not in kwargs
        return q

    fake = SimpleNamespace(flash_attn_func=kernel)
    monkeypatch.setattr(attention, "_flash_attn", fake)
    monkeypatch.setattr(attention, "_flash_attn2", fake)
    monkeypatch.setattr(attention, "flash_attention_is_installed", True)
    packed = torch.randn(2, 4, 6, 16)
    q, k, v = packed.split(2, dim=-2)
    result = attention.flash_attn_func(q, k, v, causal=True, backend=backend)
    if backend == "sdpa":
        assert not calls  # Explicit SDPA must ignore installed FA kernels.
        expected = attention._sdpa_fallback(q, k, v, causal=True)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    else:
        assert len(calls) == 1
        assert result is q


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_requested_backend_failure_never_falls_back(monkeypatch, backend):
    monkeypatch.setattr(attention, "_flash_attn", None)
    monkeypatch.setattr(attention, "_flash_attn2", None)
    monkeypatch.setattr(attention, "_flash_attn_load_error", ImportError("FA3 sentinel"))
    monkeypatch.setattr(attention, "_flash_attn2_load_error", ImportError("FA2 sentinel"))
    with pytest.raises(RuntimeError, match="refusing SDPA fallback") as failure:
        layers.CausalSelfAttention(config(attn_impl=backend), layer_idx=0)
    assert isinstance(failure.value.__cause__, ImportError)


@pytest.mark.parametrize("transformed", [False, True])
def test_packed_projection_and_view_split_loss_and_gradients(transformed):
    torch.manual_seed(17)
    actual = layers.CausalSelfAttention(config(norm_before_attn=transformed), layer_idx=0)
    reference = copy.deepcopy(actual)
    x = torch.randn(2, 4, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    # The separate-kernel path must retain one packed projection and view slices.
    packed = actual.qkv_proj(x).view(2, 4, 6, 16)
    for view in actual.unfused(packed):
        assert view.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    q, k, v = reference.qkv_proj(y).view(2, 4, 3, 2, 16).unbind(dim=2)
    q, k = q.clone(), k.clone()
    rope = precompute_rope(4, 16, device="cpu") if transformed else None
    if transformed:
        q, k = layers.apply_rope(q, rope), layers.apply_rope(k, rope)
        q, k = reference.norm(q), reference.norm(k)
        q, k, v = torch.cat([q, k, v], dim=-2).split(2, dim=-2)
    expected = attention._sdpa_fallback(q.clone(), k.clone(), v.clone(), causal=True)
    expected = reference.o_proj(expected.contiguous().view(2, 4, 32))
    output, _ = actual(x, rope_cache=rope)
    loss, reference_loss = output.square().mean(), expected.square().mean()
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    assert actual.state_dict().keys() == reference.state_dict().keys()
    for a, b in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_sdpa_cached_decoding_matches_full_sequence():
    torch.manual_seed(18)
    cfg = config()
    layer = layers.CausalSelfAttention(cfg, layer_idx=0).eval()
    x = torch.randn(1, 4, 32)
    cache = KVCache(config=cfg)
    with torch.no_grad():
        expected, _ = layer(x)
        parts = []
        for position in range(4):
            part, _ = layer(x[:, position:position+1], kv_cache=cache)
            parts.append(part)
            cache.advance()
    torch.testing.assert_close(torch.cat(parts, dim=1), expected)
    assert cache.cur_pos == 4
