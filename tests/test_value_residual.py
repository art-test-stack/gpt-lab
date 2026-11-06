import copy
from importlib import import_module

import torch

from gpt_lab.model.layers import CausalSelfAttention
from gpt_lab.utils.schemas import TransformerConfig


def test_fused_value_residual_matches_sdpa_outputs_and_gradients(monkeypatch):
    # Exercise real packed attention through its CPU fallback, without a GPU kernel.
    monkeypatch.setattr(import_module("gpt_lab.model.flash_attn"), "flash_attention_is_installed", False)
    torch.manual_seed(42)
    config = TransformerConfig(
        d_model=16, d_ffn=32, n_heads=2, d_head=8, n_layers=1,
        norm_before_attn=False, attn_impl="sdpa", dropout=0.0,
    )
    reference = CausalSelfAttention(config).to("cpu").eval()
    packed = copy.deepcopy(reference)
    packed.attn_impl = "fused"
    x = torch.randn(2, 4, config.d_model)
    values = torch.randn_like(x, requires_grad=True)
    reference_values = values.detach().clone().requires_grad_()

    output, _ = packed(x, value_embeds=values)
    expected, _ = reference(x, value_embeds=reference_values)

    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    expected.square().sum().backward()
    assert values.grad is not None and values.grad.abs().sum() > 0
    assert packed.ve_gate.weight.grad is not None and packed.ve_gate.weight.grad.abs().sum() > 0
    torch.testing.assert_close(values.grad, reference_values.grad)
    torch.testing.assert_close(packed.ve_gate.weight.grad, reference.ve_gate.weight.grad)
