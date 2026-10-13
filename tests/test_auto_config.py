import torch
from gpt_lab.utils.distributed import DistributedContext
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
        "dist_info": DistributedContext(device=torch.device("cpu")),
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
    assert generated.base_train["device_batch_size"] == 1


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


@pytest.mark.parametrize("flag", ["tie_value_embeddings", "tie_word_embeddings", "tie_all_embeddings"])
def test_auto_config_forwards_embedding_tying(tmp_path, flag):
    config = _auto_config(tmp_path, **{flag: True})
    for depth in (config.depth, config.reference_depth()):
        model_cfg = config.build_transformer_config(depth, 4096)
        assert getattr(model_cfg, flag)
        assert model_cfg.model_dump()[flag]


def test_cli_batch_and_horizon_override_automatic_sizing(monkeypatch, tmp_path):
    from gpt_lab.cli.parser import build_parser
    from gpt_lab.workflows.train_base import build_auto_config

    args = build_parser().parse_args([
        'train', '--model-name', 'test', '--run-name', 'run',
        '--model-dir', str(tmp_path), '--depth', '3', '--aspect-ratio', '64',
        '--d-head', '64', '--vocab-size', '4096', '--max-seq-len', '8',
        '--device-batch-size', '1', '--total-batch-size', '64', '--num-steps', '7',
    ])
    def reject_automatic_sizing(*_args):
        pytest.fail('Explicit CLI value must bypass automatic sizing')

    monkeypatch.setattr(AutoGPTConfig, 'compute_batch_size', reject_automatic_sizing)
    monkeypatch.setattr(AutoGPTConfig, 'compute_vocab_size', reject_automatic_sizing)
    monkeypatch.setattr('gpt_lab.model.auto.resolve_tokenizer', lambda *_args: 'dummy')
    monkeypatch.setattr(
        'gpt_lab.model.auto.load_tokenizer',
        lambda *_args, **kwargs: _DummyTokenizer(4096, kwargs['special_tokens']),
    )
    generated = build_auto_config(vars(args), 'cpu', DistributedContext(device=torch.device("cpu")))
    assert generated.model_cfg.n_layers == 3
    assert generated.model_cfg.vocab_size == 4096
    assert generated.base_train['total_batch_size'] == 64
    assert generated.base_train['n_steps'] == 7
    assert generated.base_train['n_acc_steps'] == 8
    assert generated.base_train['n_total_tokens'] == 448


@pytest.mark.fast
def test_default_scaling_methods(tmp_path):
    from gpt_lab.tokenizer.auto import compute_optimal_vocab_size, round_vocab_size

    config = _auto_config(tmp_path, n_acc_steps=-1)
    assert config.compute_model_width(3) == 256
    assert config.compute_ffn_width(256) == 1024
    assert config.reference_depth() == 12
    assert config.reference_batch_size() == 2**19
    assert config.compute_target_tokens(100) == 1100.0
    assert config.compute_batch_size(1600, 100, 512) == 2048
    assert config.compute_learning_rate_scale(64, 256) == 0.5
    assert config.compute_weight_decay_scale(64, 256, 1600, 100) == 0.03125
    assert config.compute_accumulation_steps(64) == 8
    assert config.compute_vocab_size(100_000, 256) == round_vocab_size(
        compute_optimal_vocab_size(n_non_vocab_params=100_000, d_model=256),
        multiple=128,
        minimum=256 + len(config.special_tokens),
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"n_steps": 7, "target_flops": 1024}, 7),
        ({"n_steps": -1, "target_flops": 1024}, 8),
        ({"n_steps": -1}, 4),
        ({"n_steps": -1, "target_param_data_ratio": -1}, None),
    ],
)
def test_training_steps_preserve_target_priority(tmp_path, overrides, expected):
    config = _auto_config(tmp_path, **overrides)
    if expected is None:
        with pytest.raises(ValueError, match="positive value"):
            config.compute_training_steps(256, 64, 2)
    else:
        assert config.compute_training_steps(256, 64, 2) == expected


@pytest.mark.fast
@pytest.mark.parametrize("explicit_sizes", [False, True])
def test_generation_uses_overridden_rules(monkeypatch, tmp_path, explicit_sizes):
    calls = []

    class CustomAutoConfig(AutoGPTConfig):
        def compute_model_width(self, depth):
            calls.append(("width", depth))
            return depth * self.d_head

        def compute_ffn_width(self, d_model):
            return 3 * d_model

        def compute_vocab_size(self, n_non_vocab_params, d_model):
            calls.append(("vocab", d_model))
            return 4096

        def reference_depth(self):
            return 2

        def reference_batch_size(self):
            return 128

        def compute_batch_size(self, target_tokens, reference_tokens, reference_batch_size):
            calls.append(("batch", reference_batch_size))
            return 64

    config = CustomAutoConfig(**_auto_config(
        tmp_path,
        n_steps=-1,
        n_acc_steps=-1,
        vocab_size=4096 if explicit_sizes else -1,
        total_batch_size=64 if explicit_sizes else -1,
    ).model_dump())
    tokenizer = _DummyTokenizer(4096, config.special_tokens)
    monkeypatch.setattr("gpt_lab.model.auto.resolve_tokenizer", lambda *_args: "dummy")
    monkeypatch.setattr("gpt_lab.model.auto.load_tokenizer", lambda *_args, **_kwargs: tokenizer)

    generated = config.generate_gpt_config("cpu")
    assert generated.model_cfg.d_model == 192
    assert generated.model_cfg.d_ffn == 576
    assert generated.model_cfg.n_heads == 3
    assert generated.model_cfg.vocab_size == 4096
    assert ("width", 3) in calls
    assert ("width", 2) in calls
    assert (("vocab", 192) in calls) is (not explicit_sizes)
    assert (("batch", 128) in calls) is (not explicit_sizes)

    # Unchanged rules still use the overridden architecture and reference batch.
    model = build_meta_model(generated.model_cfg)
    reference = config.build_meta_model_from_depth(2, 4096)
    target_tokens = int(11 * model.n_scaling_params())
    reference_tokens = 11 * reference.n_scaling_params()
    training = generated.base_train
    assert training["target_tokens"] == target_tokens
    assert training["total_batch_size"] == 64
    assert training["n_steps"] == target_tokens // 64
    assert training["n_total_tokens"] == 64 * (target_tokens // 64)
    assert training["n_acc_steps"] == 8
    assert training["n_flops_per_token"] == model.estimate_flops()
    assert training["batch_lr_scale"] == pytest.approx((64 / 128) ** 0.5)
    assert training["weight_decay_scale"] == pytest.approx(
        (64 / 128) ** 0.5 * reference_tokens / target_tokens
    )
