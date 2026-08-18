from pathlib import Path

import pytest

from gpt_lab.model.auto import AutoGPTConfig
from gpt_lab.model.checkpoint import build_meta_model
from gpt_lab.utils.schemas import TokenizerConfig, TransformerConfig
from gpt_lab.utils.special_tokens import SpecialTokens


class _DummyTokenizer:
    def __init__(self, vocab_size: int, special_tokens: SpecialTokens):
        self.vocab_size = vocab_size
        self.config = TokenizerConfig(
            name="dummy",
            source="dummy",
            vocab_size=vocab_size,
            pat_str="gpt2",
            special_tokens=special_tokens,
        )


def _auto_config(tmp_path: Path, **overrides) -> AutoGPTConfig:
    values = {
        "name": "test-model",
        "run_name": "test-run",
        "dirname": tmp_path,
        "tokenizer_model": "auto",
        "train_tokenizer": False,
        "vocab_size": -1,
        "depth": 3,
        "aspect_ratio": 70,
        "d_head": 64,
        "max_seq_len": 8,
        "n_steps": 1,
        "total_batch_size": 64,
        "n_acc_steps": 1,
        "device_batch_size": 1,
        "dist_info": {"WORLD_SIZE": 1, "LOCAL_RANK": 0},
    }
    values.update(overrides)
    return AutoGPTConfig(**values)


@pytest.mark.fast
def test_non_vocab_parameter_count_is_vocab_independent():
    common = {
        "tf_type": "dense",
        "max_context": 8,
        "d_model": 128,
        "d_ffn": 512,
        "n_layers": 2,
        "n_heads": 2,
        "n_kv_heads": 2,
        "d_head": 64,
    }
    small = build_meta_model(TransformerConfig(vocab_size=257, **common))
    large = build_meta_model(TransformerConfig(vocab_size=1025, **common))

    assert small.n_params != large.n_params
    assert small.n_non_vocab_params() == large.n_non_vocab_params()


@pytest.mark.fast
def test_auto_config_uses_real_architecture_and_loaded_vocab(
    monkeypatch,
    tmp_path,
):
    import gpt_lab.model.auto as model_auto

    special_tokens = SpecialTokens()
    tokenizer = _DummyTokenizer(32_001, special_tokens)
    calls = {}

    def fake_resolve(name, vocab_size, configured_special_tokens):
        calls["resolve"] = (name, vocab_size, configured_special_tokens)
        return "selected-tokenizer"

    def fake_load(name, *, special_tokens, tokenizer_dir):
        calls["load"] = (name, special_tokens, tokenizer_dir)
        return tokenizer

    monkeypatch.setattr(model_auto, "resolve_tokenizer", fake_resolve)
    monkeypatch.setattr(model_auto, "load_tokenizer", fake_load)

    config = _auto_config(
        tmp_path,
        special_tokens=special_tokens,
        tokenizer_dir=tmp_path / "tokenizers",
    )
    generated = config.generate_gpt_config("cpu")

    resolved_name, estimated_size, resolved_special_tokens = calls["resolve"]
    assert resolved_name == "auto"
    assert estimated_size >= 256 + len(special_tokens)
    assert resolved_special_tokens == special_tokens
    assert calls["load"] == (
        "selected-tokenizer",
        special_tokens,
        tmp_path / "tokenizers",
    )

    # depth * aspect_ratio is 210, so the actual model width must be rounded
    # to the configured head dimension before computing the scaling estimate.
    assert generated.model_cfg.d_model == 256
    assert generated.model_cfg.n_heads == 4
    assert generated.model_cfg.d_head == 64
    assert generated.model_cfg.vocab_size == tokenizer.vocab_size
    assert generated.tokenizer_cfg.vocab_size == tokenizer.vocab_size


@pytest.mark.fast
def test_explicit_tokenizer_vocab_mismatch_is_rejected(monkeypatch, tmp_path):
    import gpt_lab.model.auto as model_auto

    special_tokens = SpecialTokens()
    tokenizer = _DummyTokenizer(32_001, special_tokens)
    monkeypatch.setattr(
        model_auto,
        "load_tokenizer",
        lambda *_args, **_kwargs: tokenizer,
    )

    config = _auto_config(
        tmp_path,
        tokenizer_model="gpt2",
        vocab_size=32_000,
        special_tokens=special_tokens,
    )
    with pytest.raises(ValueError, match="does not match tokenizer"):
        config.generate_gpt_config("cpu")


@pytest.mark.fast
def test_auto_config_forwards_tokenizer_training_options(monkeypatch, tmp_path):
    import gpt_lab.model.auto as model_auto

    special_tokens = SpecialTokens(bos="<start>")
    tokenizer = _DummyTokenizer(4096, special_tokens)
    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return tokenizer

    monkeypatch.setattr(model_auto, "train_new_tokenizer", fake_train)

    config = _auto_config(
        tmp_path,
        tokenizer_model="auto",
        train_tokenizer=True,
        vocab_size=4096,
        pat_str="nanochat",
        special_tokens=special_tokens,
        tokenizer_dir=tmp_path / "tokenizers",
    )
    generated = config.generate_gpt_config("cpu")

    assert captured["vocab_size"] == 4096
    assert captured["pat_str"] == "nanochat"
    assert captured["special_tokens"] == special_tokens
    assert captured["tokenizer_dir"] == tmp_path / "tokenizers"
    assert generated.model_cfg.vocab_size == tokenizer.vocab_size
    assert generated.tokenizer_cfg.special_tokens == special_tokens


@pytest.mark.fast
def test_training_rejects_an_explicit_tokenizer_model(tmp_path):
    with pytest.raises(
        ValueError,
        match="tokenizer_model must be None or 'auto'",
    ):
        _auto_config(
            tmp_path,
            tokenizer_model="gpt2",
            train_tokenizer=True,
            vocab_size=4096,
        )


@pytest.mark.fast
def test_deprecated_kv_head_alias_maps_to_head_count(tmp_path):
    with pytest.warns(DeprecationWarning, match="use n_kv_heads"):
        config = _auto_config(tmp_path, d_kv_head=2)

    assert config.n_kv_heads == 2
