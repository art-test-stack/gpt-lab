import pytest
import torch

from gpt_lab.model.layers import DecoderLayer
from gpt_lab.utils.schemas import TransformerConfig


@pytest.mark.parametrize("activation", ["swiglu", "relu_squared"])
@pytest.mark.parametrize("training", [True, False])
def test_decoder_dropout_masks_attention_branch_and_preserves_skip(activation, training):
    config = TransformerConfig(
        d_model=16, d_ffn=32, n_heads=2, d_head=8, n_layers=2,
        act_func=activation, attn_impl="sdpa", dropout=1.0,
    )
    block = DecoderLayer(config).to("cpu").train(training)
    # Isolate output dropout from attention-probability dropout.
    block.attention.dropout_rate = 0.0
    with torch.no_grad():
        block.attention.qkv_proj.weight.fill_(1 / config.d_model)
        block.attention.o_proj.weight.copy_(torch.eye(config.d_model))
    x = torch.ones(2, 3, config.d_model)
    attention_output, _ = block.attention(block.norm(x))
    torch.testing.assert_close(attention_output, torch.ones_like(x))
    h = x if training else x + torch.ones_like(x)
    expected = h + block.ffn(block.norm(h))

    output, _ = block(x)

    torch.testing.assert_close(output, expected)
