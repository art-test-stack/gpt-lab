from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from gpt_lab.model.layers import CausalSelfAttention
from gpt_lab.utils.schemas import TransformerConfig


@pytest.mark.parametrize("backend", ["sdpa", "fa2", "fa3"])
def test_value_residual_without_rope_or_qk_norm(monkeypatch, backend):
    dispatch = import_module("gpt_lab.model.flash_attn")

    def packed(qkv, **kwargs):
        return dispatch._sdpa_fallback(*qkv.unbind(dim=2), **kwargs)

    if backend != "sdpa":
        kernel = SimpleNamespace(flash_attn_qkvpacked_func=packed)
        loader = "require_flash_attn2" if backend == "fa2" else "require_flash_attn3"
        monkeypatch.setattr(dispatch, loader, lambda: kernel)
        monkeypatch.setattr(import_module("gpt_lab.model.layers"), loader, lambda: kernel)
    config = TransformerConfig(
        d_model=16, d_ffn=32, n_heads=2, d_head=8, n_layers=1,
        norm_before_attn=False, attn_impl=backend, dropout=0.0,
    )
    attention = CausalSelfAttention(config).to("cpu").eval()
    with torch.no_grad():
        attention.qkv_proj.weight.zero_()
        attention.o_proj.weight.copy_(torch.eye(config.d_model))
        attention.ve_gate.weight.zero_()
    x = torch.ones(2, 4, config.d_model)
    positions = torch.arange(1, 5, dtype=x.dtype).view(1, 4, 1)
    values = positions.expand_as(x).clone().requires_grad_()

    output, _ = attention(x, value_embeds=values)

    # Zero Q/K gives causal prefix means; a zero gate projection gives 3/2.
    expected = 1.5 * values.cumsum(dim=1) / positions
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    expected_value_grad = (1.5 / positions).flip(1).cumsum(dim=1).flip(1).expand_as(values)
    torch.testing.assert_close(values.grad, expected_value_grad)
    # Each gate weight sees 2 batches * 8 channels * (1 + 1.5 + 2 + 2.5) * 3/4.
    torch.testing.assert_close(attention.ve_gate.weight.grad,
                               torch.full_like(attention.ve_gate.weight, 84.0))
