from gpt_lab.utils.distributed import ComputeContext
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from gpt_lab.model.checkpoint import build_meta_model, build_model, capture_rng_state, load_checkpoint, load_meta_config, save_checkpoint
from gpt_lab.utils.schemas import CheckpointState, MetaConfig, TrainerConfig, TrainerState, TransformerConfig


def _config(**overrides):
    values = dict(
        vocab_size=300, max_context=4, d_model=16, d_ffn=32,
        n_heads=2, n_kv_heads=2, n_layers=4, d_head=8,
        window_pattern="L", attn_impl="sdpa",
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _assert_sharing(model):
    config = model.config
    values = list(model.value_embeds.values())
    assert list(model.value_embeds) == ["1", "3"]
    share_values = config.tie_value_embeddings or config.tie_all_embeddings
    share_head = config.tie_word_embeddings or config.tie_all_embeddings
    assert (values[0] is values[1]) == share_values
    assert (model.lm_head.weight is model.embeds.weight) == share_head
    assert all((ve.weight is model.embeds.weight) == config.tie_all_embeddings for ve in values)


@pytest.mark.parametrize("flags", [
    {},
    {"tie_value_embeddings": True},
    {"tie_word_embeddings": True},
    {"tie_value_embeddings": True, "tie_word_embeddings": True},
    {"tie_all_embeddings": True},
    {"tie_all_embeddings": True, "tie_value_embeddings": True, "tie_word_embeddings": True},
])
def test_embedding_tying_lifecycle(flags, tmp_path):
    config = _config(**flags)
    model = build_meta_model(config)
    _assert_sharing(model)
    model.to_empty(device="cpu")
    _assert_sharing(model)
    model.init_weights()
    assert model.embeds.weight.std().item() == pytest.approx(0.8, rel=0.15)
    model.double().float()
    _assert_sharing(model)

    counts = model.n_params_per_layer()
    table_size = model.embeds.weight.numel()
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["lm_head"] == (0 if config.tie_word_embeddings or config.tie_all_embeddings else table_size)
    expected_value_tables = 0 if config.tie_all_embeddings else 1 if config.tie_value_embeddings else 2
    assert counts["value_embeds"] == expected_value_tables * table_size
    assert model.n_non_vocab_params() == counts["blocks"] + counts["residuals"]
    assert model.n_scaling_params() == counts["blocks"] + table_size
    untied = build_meta_model(_config())
    assert model.estimate_flops() == untied.estimate_flops()

    optimizer = model.build_optimizer(TrainerConfig(dist_info=ComputeContext(device=torch.device("cpu"))))
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len({id(p) for p in optimized}) == len(list(model.parameters()))
    assert {id(p) for p in optimized} == {id(p) for p in model.parameters()}
    embedding_group = next(group for group in optimizer.param_groups if group["name"] == "embeddings")
    assert any(p is model.embeds.weight for p in embedding_group["params"])

    # Nonzero output projections let value tables receive gradients on the first pass.
    with torch.no_grad():
        for block in model.blocks:
            block.attention.o_proj.weight.fill_(0.01)
    ids = torch.randint(0, config.vocab_size, (2, config.max_context))
    labels = torch.randint(0, config.vocab_size, ids.shape)
    output = model(ids, labels)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    for ve in model.value_embeds.values():
        assert ve.weight.grad is not None
        assert torch.isfinite(ve.weight.grad).all()
        assert ve.weight.grad.abs().sum() > 0

    dist_info = ComputeContext(device=torch.device("cpu"))
    save_checkpoint(model, tmp_path, step=7, trainer_state=TrainerState(step=7),
                    checkpoint_state=CheckpointState(), dist_info=dist_info)
    weights_path = tmp_path / "checkpoint_step_000007" / "model.safetensors"
    assert not weights_path.with_suffix(".pt").exists()
    assert set(load_file(weights_path)) == set(model.state_dict())
    state = load_checkpoint(tmp_path, step=7, load_optim=False, load_trainer=False,
                            load_scaler=False, load_rng=False, dist_info=dist_info).model_state
    restored = build_meta_model(config).to_empty(device="cpu")
    restored.precompute_pos_enc()
    restored.load_state_dict(state, strict=True, assign=True)
    _assert_sharing(restored)
    torch.testing.assert_close(restored(ids, labels).logits, output.logits, rtol=0, atol=0)
    restored.load_state_dict(state, strict=True)
    _assert_sharing(restored)


def test_full_tying_rejects_incompatible_value_width():
    with pytest.raises(ValueError, match="value and input embedding widths must match"):
        _config(n_kv_heads=1, tie_all_embeddings=True)


def test_frozen_configs_still_initialize_derived_fields():
    _config()
    config = _config(n_kv_heads=1, attn_impl="fused", window_pattern=None)
    assert config.use_gqa
    assert config.attn_impl == "sdpa"
    assert config.window_pattern == "L"
    assert config.window_size == 1
    assert config.attention_dropout == 0.0
    with pytest.raises(ValueError, match="frozen"):
        config.d_model = 32


@pytest.mark.parametrize("flags", [
    {"tie_value_embeddings": True},
    {"tie_word_embeddings": True},
    {"tie_value_embeddings": True, "tie_word_embeddings": True},
])
def test_partial_tying_supports_gqa(flags):
    model = build_meta_model(_config(n_kv_heads=1, **flags)).to_empty(device="cpu")
    model.init_weights()
    _assert_sharing(model)
    assert next(iter(model.value_embeds.values())).weight.shape == (320, 8)
    ids = torch.randint(0, 300, (1, 4))
    assert torch.isfinite(model(ids, ids).loss)


@pytest.mark.parametrize("version,missing_fields,expected_tie", [
    ("0.0.9", False, False),
    ("0.1.0", True, False),
    ("0.1.0", False, True),
    ("0.1.1", True, True),
    ("0.1.1", False, True),
    ("0.2.0", True, True),
    (None, True, False),
    (None, False, True),
])
def test_checkpoint_metadata_preserves_tying_mode(tmp_path, version, missing_fields, expected_tie):
    run_dir = tmp_path / "demo" / "run"
    meta = MetaConfig(
        name="demo", run_name="run", dirname=run_dir,
        model_cfg=_config(tie_word_embeddings=True),
        base_train={},
    )
    assert meta.version == "0.1.1"
    metadata = meta.model_dump(mode="json")
    if version is None:
        metadata.pop("version")
    else:
        metadata["version"] = version
    if missing_fields:
        metadata["model_cfg"].pop("tie_value_embeddings")
        metadata["model_cfg"].pop("tie_all_embeddings")
    path = run_dir / "meta.json"
    path.write_text(json.dumps(json.dumps(metadata)))
    saved = path.read_text()
    restored = load_meta_config("demo", run_name="run", model_cachedir=tmp_path)
    assert restored.model_cfg.tie_word_embeddings == expected_tie
    assert not restored.model_cfg.tie_value_embeddings
    assert not restored.model_cfg.tie_all_embeddings
    assert restored.version == (version or "0.1.1")
    assert restored.git_info == metadata["git_info"]
    assert path.read_text() == saved


@pytest.mark.parametrize("extension", ["pt", "pth"])
def test_legacy_checkpoint_loads_distinct_embedding_weights(tmp_path, monkeypatch, extension):
    config = _config()
    model = build_meta_model(config).to_empty(device="cpu")
    model.init_weights()
    model.eval()
    with torch.no_grad():
        for index, table in enumerate([model.embeds, model.lm_head, *model.value_embeds.values()]):
            table.weight.fill_(0.1 * (index + 1))
    run_dir = tmp_path / "demo" / "run"
    meta = MetaConfig(name="demo", run_name="run", dirname=run_dir, model_cfg=config, base_train={})
    metadata = meta.model_dump(mode="json")
    metadata["version"] = "0.1.0"
    metadata["model_cfg"]["tie_word_embeddings"] = True
    metadata["model_cfg"].pop("tie_value_embeddings")
    metadata["model_cfg"].pop("tie_all_embeddings")
    (run_dir / "meta.json").write_text(json.dumps(json.dumps(metadata)))
    step_dir = run_dir / "base" / "checkpoint_step_000007"
    step_dir.mkdir(parents=True)
    torch.save(model.state_dict(), step_dir / f"model.{extension}")
    original = (step_dir / f"model.{extension}").read_bytes()
    torch.save(capture_rng_state(), step_dir / "rng_state.pt")
    (step_dir / "trainer_state.json").write_text(TrainerState(step=7).model_dump_json())
    monkeypatch.setattr("gpt_lab.model.checkpoint.build_tokenizer", lambda _: SimpleNamespace(vocab_size=config.vocab_size))

    restored, _, _, _ = build_model(
        "demo", run_name="run", model_cachedir=tmp_path, step=7, phase="eval",
        dist_info=ComputeContext(device=torch.device("cpu")),
    )
    assert not restored.config.tie_word_embeddings
    assert (step_dir / "model.safetensors").exists()
    assert (step_dir / f"model.{extension}").read_bytes() == original
    _assert_sharing(restored)
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], weight, rtol=0, atol=0)
    ids = torch.tensor([[0, 1, 2, 3]])
    torch.testing.assert_close(restored(ids).logits, model(ids).logits, rtol=0, atol=0)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Converted checkpoints must use safetensors"))
    reloaded, _, _, _ = build_model(
        "demo", run_name="run", model_cachedir=tmp_path, step=7, phase="eval",
        dist_info=ComputeContext(device=torch.device("cpu")),
    )
    torch.testing.assert_close(reloaded(ids).logits, model(ids).logits, rtol=0, atol=0)
