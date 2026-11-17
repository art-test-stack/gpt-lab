import torch
from gpt_lab.utils.distributed import ComputeContext
import math

import pytest
import json
from pathlib import Path
from types import SimpleNamespace

from gpt_lab.model.checkpoint import build_meta_model
from gpt_lab.utils.schemas import OptimizerConfig, TrainerConfig, TransformerConfig


@pytest.mark.parametrize('custom', [False, True])
def test_optimizer_profile_is_scaled_once_without_mutation(tmp_path, custom):
    path = None
    if custom:
        path = tmp_path / 'optimizer.yaml'
        path.write_text('head:\n  lr: 0.2\ntransformer:\n  weight_decay: 0.05\n')
    profile = OptimizerConfig.from_file(path)
    snapshot = profile.model_dump()
    config = TrainerConfig(
        optim_config=profile, comp_info=ComputeContext(device=torch.device("cpu")),
        batch_lr_scale=4.0, weight_decay_scale=3.0, n_steps=8,
    )
    model = build_meta_model(TransformerConfig(
        vocab_size=300, max_context=4, d_model=16, d_ffn=32,
        n_heads=2, n_kv_heads=2, n_layers=2, d_head=8, window_pattern='L',
    )).to_empty(device='cpu')
    optimizer = model.build_optimizer(config)
    bases = profile.group_configs()
    for group in optimizer.param_groups:
        name = 'transformer' if group['name'].startswith('block.') else group['name']
        width = math.sqrt(768 / 16) if name in ('embeddings', 'head', 'value_embeds') else 1.0
        assert group['lr'] == pytest.approx(bases[name]['lr'] * 4 * width)
        decay = 3.0 if name == 'transformer' else 1.0
        assert group['weight_decay'] == pytest.approx(bases[name]['weight_decay'] * decay)
    assert profile.model_dump() == snapshot
    assert config.weight_decay_schedule(0) == 1.0
    assert config.weight_decay_schedule(8) == 0.0
    optimizer.update_hyperparams(lrm=0.5, weight_decay=config.weight_decay_schedule(0))
    for group in optimizer.param_groups:
        assert group['lr'] == pytest.approx(group['initial_lr'] * 0.5)
        assert group['weight_decay'] == group['initial_weight_decay']
    restored = TrainerConfig.model_validate_json(config.model_dump_json())
    assert restored.optim_config.model_dump() == snapshot


def test_normuon_override_profile():
    profile = OptimizerConfig.from_file("configs/normuon.yaml")
    groups = profile.group_configs()
    assert groups["transformer"]["opt"] == "normuon"
    assert groups["transformer"]["beta"] == 0.95
    assert groups["embeddings"]["opt"] == groups["head"]["opt"] == "adamw"


@pytest.mark.parametrize("mode", ["joint", "projection", "head"])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("opt_name", ["muon", "normuon"])
def test_qkv_optimizer_profile_keeps_one_projection_per_layer(tmp_path, monkeypatch, mode, kv_heads, opt_name):
    path = tmp_path / "optimizer.yaml"
    path.write_text(f"transformer:\n  opt: {opt_name}\n  muon_qkv: {mode}\n")
    profile = OptimizerConfig.from_file(path)
    config = TrainerConfig(optim_config=profile, comp_info=ComputeContext(device=torch.device("cpu")))
    model = build_meta_model(TransformerConfig(
        vocab_size=300, max_context=4, d_model=16, d_ffn=(2 + 2 * kv_heads) * 8,
        n_heads=2, n_kv_heads=kv_heads, n_layers=2, d_head=8, window_pattern="L",
        attn_impl="sdpa",
    )).to_empty(device="cpu")
    model.init_weights()
    model.precompute_pos_enc()
    with torch.no_grad():
        for block in model.blocks:
            block.attention.o_proj.weight.normal_(std=0.1)
    qkv_params = {block.attention.qkv_proj.weight for block in model.blocks}
    names = dict(model.named_parameters())
    assert {name for name in names if "qkv_proj" in name} == {
        f"blocks.{i}.attention.qkv_proj.weight" for i in range(2)
    }
    assert all(param.shape == ((2 + 2 * kv_heads) * 8, 16) for param in qkv_params)
    projection_calls = []
    linear = torch.nn.functional.linear

    def observe_linear(x, weight, bias=None):
        if weight in qkv_params:
            projection_calls.append(weight)
        return linear(x, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", observe_linear)
    ids = torch.randint(0, 300, (2, 4))
    before = model(ids, ids)
    before.loss.backward()
    gradients = {name: p.grad.clone() if p.grad is not None else None for name, p in names.items()}
    assert len(projection_calls) == 2
    model.zero_grad(set_to_none=True)
    projection_calls.clear()

    optimizer = model.build_optimizer(config)
    assert dict(model.named_parameters()).keys() == names.keys()
    assert all(names[name] is p for name, p in model.named_parameters())
    assert not optimizer.state
    after = model(ids, ids)
    after.loss.backward()
    torch.testing.assert_close(after.logits, before.logits)
    torch.testing.assert_close(after.loss, before.loss)
    assert len(projection_calls) == 2
    for name, p in model.named_parameters():
        if gradients[name] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, gradients[name])

    splits = ((16, kv_heads * 8, kv_heads * 8) if mode == "projection"
              else (8,) * (2 + 2 * kv_heads) if mode == "head" else ())
    for group in optimizer.param_groups:
        params = set(group["params"])
        if params & qkv_params:
            assert group.get("qkv_split_sizes", ()) == splits
            if splits:
                assert params == qkv_params
            else:
                # The FFN up weights share the packed QKV weight's shape.
                assert model.blocks[0].ffn.w_1.weight in params
        else:
            assert "qkv_split_sizes" not in group
    assert profile.group_configs()["transformer"]["muon_qkv"] == mode


@pytest.mark.parametrize('contents', [
    'batch_size: 64',
    'transformer:\n  n_steps: 8',
    'head:\n  lr: -0.1',
    'head:\n  lr: bad',
    'head:\n  weight_decay: .nan',
    'head:\n  opt: unknown',
    'head:\n  muon_qkv: head',
    'transformer:\n  muon_qkv: fused',
])
def test_optimizer_file_rejects_conflicting_or_invalid_parameters(tmp_path, contents):
    path = tmp_path / 'invalid.yaml'
    path.write_text(contents)
    with pytest.raises(ValueError):
        OptimizerConfig.from_file(path)


def test_legacy_trainer_keeps_saved_decay_schedule():
    config = TrainerConfig(weight_decay_scale=3.0, n_steps=8)
    assert config.optim_config is None
    assert config.weight_decay_schedule(0) == 3.0
    # Old pickled TrainerConfig instances do not contain newly added fields.
    config.__dict__.pop('optim_config')
    assert config.weight_decay_schedule(0) == 3.0


def test_nanochat_muon_momentum_schedule_and_optimizer_application():
    config = TrainerConfig(n_steps=4364, lr_warmdown_ratio=0.65)
    # The scheduler emits a multiplier; the optimizer owns the base momentum.
    base = 0.95
    warmdown_start = 4364 - round(0.65 * 4364)
    for step, expected in (
        (0, 0.85),
        (200, 0.91),
        (399, 0.85 + 399 / 400 * 0.12),
        (400, 0.97),
        (warmdown_start, 0.97),
        (4364, 0.90),
    ):
        assert base * config.muon_momentum_schedule(step) == pytest.approx(expected)

    profile = OptimizerConfig.from_file("configs/normuon.yaml")
    configured = TrainerConfig(n_steps=4364, optim_config=profile)
    normuon_base = profile.group_configs()["transformer"]["momentum"]
    assert normuon_base * configured.muon_momentum_schedule(0) == pytest.approx(0.85)
    assert normuon_base * configured.muon_momentum_schedule(4364) == pytest.approx(0.90)
