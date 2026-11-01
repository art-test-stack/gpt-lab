"""CPU safeguard tests. Fake conversions verify wiring, never FP8 arithmetic."""

import builtins
import copy
from types import SimpleNamespace

import pytest
import torch

from gpt_lab.cli.parser import build_parser
from gpt_lab.model import fp8
from gpt_lab.model.gpt import DenseTransformer
from gpt_lab.model.layers import Linear
from gpt_lab.train.trainer import Trainer
from gpt_lab.utils.distributed import ComputeContext
from gpt_lab.utils.schemas import TrainerConfig, TransformerConfig


def model(activation="swiglu"):
    result = DenseTransformer(TransformerConfig(vocab_size=64, d_model=32, d_ffn=128,
        n_layers=2, n_heads=2, d_head=16, max_context=8, act_func=activation))
    result.init_weights()
    return result


def test_bf16_default_never_imports_torchao(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("torchao"):
            pytest.fail("default BF16 path imported optional backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    candidate = model()
    reference = copy.deepcopy(candidate)
    reference._enforce_precision(torch.bfloat16)
    original_ids = {id(p) for p in candidate.parameters()}
    config = TrainerConfig(comp_info=ComputeContext(torch.device("cpu"), compute_dtype=torch.bfloat16))
    trainer = Trainer(model=candidate, tokenizer=None, train_loader=None, val_loader=None, config=config,
                      checkpoint_manager=SimpleNamespace(source_dir=tmp_path))
    assert not trainer.config.fp8_ffn_forward
    assert {id(p) for p in candidate.parameters()} == original_ids
    x = torch.randint(64, (2, 8))
    y = torch.randint(64, (2, 8))
    losses = []
    for item in (reference, candidate):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = item.training_loss(x, y)
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for p, q in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert not build_parser().parse_args(["train", "auto"]).fp8_ffn_forward
    assert build_parser().parse_args(["train", "auto", "--fp8-ffn-forward"]).fp8_ffn_forward


def test_unsupported_devices_reject_without_import(monkeypatch):
    candidate = model()
    ids = [id(p) for p in candidate.parameters()]
    with pytest.raises(RuntimeError, match="Hopper"):
        fp8.enable_fp8_ffn_forward(candidate, torch.bfloat16)
    assert ids == [id(p) for p in candidate.parameters()]
    with pytest.raises(ValueError, match="BF16"):
        fp8.check_fp8_support(torch.device("cuda"), torch.float32)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    with pytest.raises(RuntimeError, match="A100"):
        fp8.check_fp8_support(torch.device("cuda"), torch.bfloat16)


class Float8Proxy(Linear):
    """BF16 test double, not an emulated or native FP8 kernel."""

    @classmethod
    def from_linear(cls, original):
        result = cls(original.in_features, original.out_features, device="meta")
        result.weight = original.weight
        result.bias = original.bias
        result.train(original.training)
        return result


@pytest.mark.parametrize("activation", ["swiglu", "relu_squared"])
def test_conversion_identity_current_weights_and_state_keys(monkeypatch, activation):
    monkeypatch.setattr(fp8, "check_fp8_support", lambda device, dtype: None)
    monkeypatch.setattr(fp8, "_float8_linear_class", lambda: Float8Proxy)
    candidate = model(activation)
    original = dict(candidate.named_parameters())
    state = copy.deepcopy(candidate.state_dict())
    context = ComputeContext(torch.device("cpu"), compute_dtype=torch.bfloat16)
    optimizer = candidate.build_optimizer(TrainerConfig(comp_info=context))
    assert fp8.enable_fp8_ffn_forward(candidate, torch.bfloat16) == 4
    assert all(parameter is original[name] for name, parameter in candidate.named_parameters())
    assert set(state) == set(candidate.state_dict())
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in candidate.parameters()}
    assert type(candidate.lm_head) is Linear
    assert all(type(block.attention.qkv_proj) is Linear for block in candidate.blocks)
    linear = candidate.blocks[0].ffn.w_1
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    before = linear(x)
    with torch.no_grad():
        original["blocks.0.ffn.w_1.weight"].zero_()
    assert not torch.equal(before, linear(x))
    assert torch.count_nonzero(linear(x)) == 0
    candidate.load_state_dict(state, strict=True)
    torch.testing.assert_close(linear(x), before, rtol=0, atol=0)
    saved = TrainerConfig(fp8_ffn_forward=True).model_dump_json()
    assert TrainerConfig.model_validate_json(saved).fp8_ffn_forward
    with pytest.raises(ValueError, match="unconverted"):
        fp8.enable_fp8_ffn_forward(candidate, torch.bfloat16)


def test_missing_extra_and_wrong_architecture_fail_before_conversion(monkeypatch):
    monkeypatch.setattr(fp8, "check_fp8_support", lambda device, dtype: None)
    original_import = builtins.__import__
    def without_torchao(name, *args, **kwargs):
        if name.startswith("torchao"):
            raise ImportError("optional backend unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_torchao)
    candidate = model()
    original = [block.ffn.w_1 for block in candidate.blocks]
    with pytest.raises(ImportError, match="extra fp8"):
        fp8.enable_fp8_ffn_forward(candidate, torch.bfloat16)
    assert original == [block.ffn.w_1 for block in candidate.blocks]
    wrong = model()
    wrong.blocks[0].ffn = torch.nn.Identity()
    with pytest.raises(ValueError, match="native SwiGLU or squared-ReLU"):
        fp8.enable_fp8_ffn_forward(wrong, torch.bfloat16)


def test_eval_fallback_preserves_cast_parameters_and_mode():
    candidate = torch.nn.Sequential(Float8Proxy(32, 64, device="cpu")).eval()
    original = candidate[0]
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    expected = original(x)
    with pytest.raises(RuntimeError, match="test failure"):
        with fp8.DisableFP8(candidate):
            assert type(candidate[0]) is Linear
            assert not candidate[0].training
            assert candidate[0].weight is original.weight
            torch.testing.assert_close(candidate(x), expected, rtol=0, atol=0)
            raise RuntimeError("test failure")
    assert candidate[0] is original
