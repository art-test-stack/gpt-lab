import torch
from gpt_lab.utils.distributed import ComputeContext
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
        "comp_info": ComputeContext(device=torch.device("cpu")),
    }
    values.update(overrides)
    return AutoGPTConfig(**values)


@pytest.mark.parametrize("kv_heads, expected_parameters", [
    (None, 370_704_408),
    (3, 349_470_744),
])
def test_llama_preset_with_gpt2_and_gqa_override(monkeypatch, tmp_path, kv_heads, expected_parameters):
    from gpt_lab.cli.parser import build_parser
    from gpt_lab.model.layers import Linear
    from gpt_lab.workflows.train_base import build_auto_config

    # Linear's explicit default device otherwise bypasses the existing meta context.
    monkeypatch.setattr(Linear.__init__, "__defaults__", (False, torch.device("meta"), None))
    monkeypatch.setattr(
        AutoGPTConfig, "compute_vocab_size",
        lambda *_args: pytest.fail("An explicit GPT-2 tokenizer must bypass vocabulary sizing"),
    )
    loaded = []

    def load(name, *, special_tokens, tokenizer_dir):
        loaded.append(name)
        return _DummyTokenizer(50_257, special_tokens)

    monkeypatch.setattr("gpt_lab.model.auto.load_tokenizer", load)
    args = build_parser().parse_args([
        "train", "--architecture-config", str(Path(__file__).resolve().parents[1] / "configs/llama3.yaml"),
        "--tokenizer-model", "gpt2", "--model-name", "preset", "--run-name", "test",
        "--model-dir", str(tmp_path), "--max-seq-len", "8", "--device-batch-size", "1",
        "--total-batch-size", "64", "--num-steps", "1",
    ] + (["--n-kv-heads", str(kv_heads)] if kv_heads is not None else []))
    generated = build_auto_config(vars(args), "meta", ComputeContext(device=torch.device("cpu")))
    config = generated.model_cfg
    assert loaded == ["gpt2"]
    assert (config.n_layers, config.d_model, config.d_ffn) == (12, 1152, 4608)
    assert (config.n_heads, config.n_kv_heads) == (9, kv_heads or 9)
    assert config.act_func == "swiglu"
    assert config.disable_value_embeds and config.freeze_res_connections
    assert (config.res_h, config.res_x0) == (1.0, 0.0)
    model = build_meta_model(config)
    assert all(p.device.type == "meta" for p in model.parameters())
    assert model.n_params == sum(p.numel() for p in model.parameters()) == expected_parameters
    assert sum(p.numel() for p in model.parameters() if not p.requires_grad) == 24
    assert generated.base_train["target_tokens"] == int(11 * model.n_scaling_params())
    assert generated.base_train["n_flops_per_token"] == model.estimate_flops()


def test_llama_d16_mha_training_sizing_on_four_devices(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from gpt_lab.model.layers import Linear

    monkeypatch.setattr(Linear.__init__, "__defaults__", (False, torch.device("meta"), None))
    config = AutoGPTConfig(
        architecture_config=Path(__file__).resolve().parents[1] / "configs/llama3.yaml",
        depth=16, tokenizer_model="gpt2", target_param_data_ratio=15,
        device_batch_size=16, dirname=tmp_path, run_name="sizing",
        comp_info=ComputeContext(
            device=torch.device("cpu"),
            mesh=SimpleNamespace(size=lambda: 4, get_rank=lambda: 0),
        ),
    )
    model = config.build_meta_model_from_depth(config.depth, 50_257)
    reference = config.build_meta_model_from_depth(config.reference_depth(), 50_257)
    assert all(p.device.type == "meta" for m in (model, reference) for p in m.parameters())
    assert (model.config.n_layers, model.config.d_model, model.config.d_ffn) == (16, 1536, 6144)
    assert (model.config.n_heads, model.config.n_kv_heads) == (12, 12)
    assert (reference.config.n_heads, reference.config.n_kv_heads) == (9, 9)
    assert model.n_params == 758_513_696
    assert model.n_scaling_params() == 681_246_720
    training = config.build_training_config(model, reference)
    assert (training["n_steps"], training["n_acc_steps"], training["total_batch_size"]) == (19490, 4, 524288)


def test_architecture_yaml_overrides_apply_before_scaling(monkeypatch, tmp_path):
    path = tmp_path / "architecture.yaml"
    path.write_text(
        "act_func: swiglu\nd_head: 16\nn_kv_heads: 2\n"
        "use_packed_attn_kernel: true\nwindow_pattern: L\n"
    )
    config = _auto_config(
        tmp_path, architecture_config=path, n_kv_heads=None,
        use_packed_attn_kernel=False,
    )
    assert config.act_func == "swiglu"
    assert config.d_head == 64  # Explicit helper setting overrides the file.
    assert config.n_kv_heads is None
    assert config.use_packed_attn_kernel is False

    observed = []
    original = AutoGPTConfig.build_transformer_config

    def capture(self, depth, vocab_size):
        model_cfg = original(self, depth, vocab_size)
        observed.append(model_cfg)
        return model_cfg

    monkeypatch.setattr(AutoGPTConfig, "build_transformer_config", capture)
    monkeypatch.setattr("gpt_lab.model.auto.resolve_tokenizer", lambda *_args: "dummy")
    monkeypatch.setattr(
        "gpt_lab.model.auto.load_tokenizer",
        lambda *_args, **kwargs: _DummyTokenizer(4096, kwargs["special_tokens"]),
    )
    generated = config.generate_gpt_config("cpu")
    assert [cfg.n_layers for cfg in observed] == [3, config.reference_depth(), 3]
    for cfg in observed:
        assert cfg.act_func == "swiglu"
        assert cfg.d_head == 64
        assert cfg.n_kv_heads == cfg.n_heads
        assert cfg.window_pattern == "L"
        assert cfg.use_packed_attn_kernel is False
    model = build_meta_model(generated.model_cfg)
    assert generated.base_train["target_tokens"] == int(11 * model.n_scaling_params())
    assert generated.base_train["n_flops_per_token"] == model.estimate_flops()

    # Resolved configs survive removal of the source file.
    saved = config.model_dump_json()
    assert "architecture_config" not in config.model_dump()
    path.unlink()
    restored = AutoGPTConfig.model_validate_json(saved)
    assert restored.build_transformer_config(3, 4096) == generated.model_cfg


@pytest.mark.parametrize("filename, contents, error", [
    ("architecture.json", "{}", "must be a YAML file"),
    ("architecture.yaml", "act_func: [", "Invalid architecture YAML"),
    ("architecture.yaml", "- swiglu", "must map field names"),
    ("architecture.yaml", "typo: true", "Unknown architecture fields"),
    ("architecture.yaml", "n_steps: 10", "Unknown architecture fields"),
    ("architecture.yaml", "act_func: unknown", "act_func"),
])
def test_invalid_architecture_yaml_is_rejected(tmp_path, filename, contents, error):
    path = tmp_path / filename
    path.write_text(contents)
    with pytest.raises(ValueError, match=error):
        _auto_config(tmp_path, architecture_config=path)


@pytest.mark.parametrize("flags, expected", [
    ([], True),
    (["--disable-value-embeds", "--freeze-res-connections"], True),
    (["--disable-value-embeds", "true", "--freeze-res-connections", "true"], True),
    (["--disable-value-embeds", "false", "--freeze-res-connections", "false"], False),
])
def test_cli_architecture_defaults_and_boolean_overrides(monkeypatch, tmp_path, flags, expected):
    from gpt_lab.cli.parser import build_parser
    from gpt_lab.workflows.train_base import build_auto_config

    path = tmp_path / "architecture.yaml"
    path.write_text(
        "depth: 4\nd_head: 64\nact_func: swiglu\ntie_embeddings: embs;head\n"
        "disable_value_embeds: true\nfreeze_res_connections: true\nres_h: 1.0\nres_x0: 0.0\n"
    )
    # Stop at the resolved configuration; model behavior is checked separately.
    monkeypatch.setattr(AutoGPTConfig, "generate_gpt_config", lambda self, device: self)
    args = build_parser().parse_args(["train", "--architecture-config", str(path), "--depth", "6"] + flags)
    config = build_auto_config(vars(args), "cpu", ComputeContext(device=torch.device("cpu")))
    assert config.depth == 6
    assert config.d_head == 64
    assert config.act_func == "swiglu"
    assert config.disable_value_embeds is expected
    assert config.freeze_res_connections is expected
    model_cfg = config.build_transformer_config(config.depth, 4096)
    assert model_cfg.res_h == 1.0
    assert model_cfg.res_x0 == 0.0


def test_training_architecture_defaults_without_a_preset(monkeypatch):
    from gpt_lab.cli.parser import build_parser
    from gpt_lab.workflows.train_base import build_auto_config

    monkeypatch.setattr(AutoGPTConfig, "generate_gpt_config", lambda self, device: self)
    args = build_parser().parse_args(["train"])
    config = build_auto_config(vars(args), "cpu", ComputeContext(device=torch.device("cpu")))
    assert (config.depth, config.d_head, config.aspect_ratio, config.max_seq_len) == (12, 128, 64, 2048)
    assert config.act_func == "relu_squared"
    assert config.tie_embeddings == "embs;value_embs;head"
    assert not config.disable_value_embeds
    assert not config.freeze_res_connections


@pytest.mark.parametrize("flag", ["--disable-value-embeds", "--freeze-res-connections"])
def test_cli_rejects_invalid_boolean(flag):
    from gpt_lab.cli.parser import build_parser

    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["train", flag, "maybe"])
    assert error.value.code == 2


@pytest.mark.parametrize("config_class", [TransformerConfig, AutoGPTConfig])
def test_legacy_fused_attention_is_rejected(config_class):
    with pytest.raises(ValueError, match="attn_impl"):
        config_class.model_validate({"attn_impl": "fused"})


@pytest.mark.parametrize("config_class", [TransformerConfig, AutoGPTConfig])
def test_custom_attention_kernel_config_roundtrip(config_class, tmp_path):
    config = (_auto_config(tmp_path, attn_impl="example/attention") if config_class is AutoGPTConfig
              else config_class(attn_impl="example/attention"))
    assert config_class.model_validate_json(config.model_dump_json()).attn_impl == "example/attention"


@pytest.mark.parametrize("config_class", [TransformerConfig, AutoGPTConfig])
def test_packed_attention_config_defaults_and_roundtrip(config_class, tmp_path):
    config = _auto_config(tmp_path) if config_class is AutoGPTConfig else config_class()
    assert config.use_packed_attn_kernel is False
    config = config_class.model_validate(config.model_dump() | {"use_packed_attn_kernel": True})
    assert config_class.model_validate_json(config.model_dump_json()).use_packed_attn_kernel is True
    data = config.model_dump(exclude={"use_packed_attn_kernel"})
    assert config_class.model_validate(data).use_packed_attn_kernel is False


@pytest.mark.parametrize("name", ["typo", "owner/", "/kernel", "owner/kernel@v2"])
def test_invalid_attention_kernel_config(name):
    with pytest.raises(ValueError, match="attn_impl"):
        TransformerConfig(attn_impl=name)


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


@pytest.mark.parametrize("mode", ["value_embs:", "embs,head", "embs,head,value_embs:"])
def test_auto_config_forwards_embedding_tying(tmp_path, mode):
    config = _auto_config(tmp_path, tie_embeddings=mode)
    for depth in (config.depth, config.reference_depth()):
        model_cfg = config.build_transformer_config(depth, 4096)
        assert model_cfg.tie_embeddings == mode
        assert model_cfg.model_dump()["tie_embeddings"] == mode


@pytest.mark.parametrize("use_packed", [False, True])
def test_cli_batch_and_horizon_override_automatic_sizing(monkeypatch, tmp_path, use_packed):
    from gpt_lab.cli.parser import build_parser
    from gpt_lab.workflows.train_base import build_auto_config

    args = build_parser().parse_args([
        'train', '--model-name', 'test', '--run-name', 'run',
        '--model-dir', str(tmp_path), '--depth', '3', '--aspect-ratio', '64',
        '--d-head', '64', '--vocab-size', '4096', '--max-seq-len', '8',
        '--device-batch-size', '1', '--total-batch-size', '64', '--num-steps', '7',
    ] + (['--use-packed-attn-kernel'] if use_packed else []))
    def reject_automatic_sizing(*_args):
        pytest.fail('Explicit CLI value must bypass automatic sizing')

    monkeypatch.setattr(AutoGPTConfig, 'compute_batch_size', reject_automatic_sizing)
    monkeypatch.setattr(AutoGPTConfig, 'compute_vocab_size', reject_automatic_sizing)
    monkeypatch.setattr('gpt_lab.model.auto.resolve_tokenizer', lambda *_args: 'dummy')
    monkeypatch.setattr(
        'gpt_lab.model.auto.load_tokenizer',
        lambda *_args, **kwargs: _DummyTokenizer(4096, kwargs['special_tokens']),
    )
    generated = build_auto_config(vars(args), 'cpu', ComputeContext(device=torch.device("cpu")))
    assert generated.model_cfg.n_layers == 3
    assert generated.model_cfg.use_packed_attn_kernel is use_packed
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
