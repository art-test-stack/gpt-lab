import pytest
import random
import string
from pathlib import Path

from gpt_lab.tokenizer.serialization import (
    load_mergeable_ranks,
    save_mergeable_ranks,
    validate_mergeable_ranks,
    validate_no_special_token_overlap,
)
from gpt_lab.tokenizer.truncation import parse_truncated_name, truncated_from_pretrained
from gpt_lab.utils.special_tokens import SpecialTokens
from gpt_lab.tokenizer import Tokenizer, TokenizerConfig
from gpt_lab.tokenizer import hf as tokenizer_hf
import gpt_lab.tokenizer.auto as tokenizer_auto


def make_dummy_dataset(size: int, max_seq_len: int):
    """Creates a dummy dataset of random token sequences for testing purposes.
    
    Args:
        size (int): The number of samples in the dataset (in MB).
        max_seq_len (int): The maximum sequence length for each sample.
    Returns:
        Iterator[str]: An Iterator of random strings representing token sequences.
    """
    num_samples = size * 1024 * 1024 // max_seq_len  # Approximate number of samples based on size and sequence length

    for _ in range(num_samples):
        seq_len = random.randint(1, max_seq_len)
        sample = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=seq_len))
        yield sample


@pytest.fixture(scope="module")
def dummy_small():
    return make_dummy_dataset(size=10, max_seq_len=16)

@pytest.fixture(scope="module")
def dummy_large():
    return make_dummy_dataset(size=100, max_seq_len=16)

@pytest.mark.fast
def test_tokenizer(dummy_small, dummy_large):
    special_tokens = SpecialTokens()
    config = TokenizerConfig(name="gpt2", vocab_size=-1, max_context=16, pat_str="gpt2", source="tiktoken", special_tokens=special_tokens)
    tokenizer = Tokenizer.from_config(config=config)
    
    # Test small dataset
    for sample in dummy_small:
        tokens = tokenizer.encode(sample)
        assert len(tokens) <= 16, "Tokenized sequence exceeds max context length"
        decoded = tokenizer.decode(tokens)
        assert decoded == sample, "Decoded text does not match original sample"
    
    # Test large dataset
    for sample in dummy_large:
        tokens = tokenizer.encode(sample)
        assert len(tokens) <= 16, "Tokenized sequence exceeds max context length"
        decoded = tokenizer.decode(tokens)
        assert decoded == sample, "Decoded text does not match original sample"

@pytest.mark.fast
def test_train_tokenizer(dummy_small):
    special_tokens = SpecialTokens()
    corpus = list(dummy_small)  # Convert generator to list for multiple iterations
    config = TokenizerConfig(name="gpt2", vocab_size=-1, max_context=16, pat_str="gpt2", source="tiktoken", special_tokens=special_tokens)
    tokenizer = Tokenizer.from_config(config=config)
    
    # Simulate training by encoding and decoding the same sample multiple times
    for sample in dummy_small:
        tokens = tokenizer.encode(sample)
        decoded = tokenizer.decode(tokens)
        assert decoded == sample, "Decoded text does not match original sample after training simulation"
    import warnings
    warnings.warn("This is dummy training test. TODO: Implement actual training logic and test it properly.")

@pytest.mark.fast
def test_parse_truncated_name_valid_and_invalid():
    assert parse_truncated_name("gpt2_truncated_1024") == ("gpt2", 1024)
    assert parse_truncated_name("foo/bar_truncated_32000") == ("foo/bar", 32000)

    assert parse_truncated_name("gpt2") is None
    assert parse_truncated_name("gpt2_truncated_x") is None
    assert parse_truncated_name("gpt2_truncated_") is None


@pytest.mark.fast
def test_save_load_mergeable_ranks_roundtrip_and_fingerprint_stable(tmp_path: Path):
    path_a = tmp_path / "tok_a.msgpack"
    path_b = tmp_path / "tok_b.msgpack"

    # Same logical mapping, different insertion order.
    ranks_a = {b"b": 1, b"a": 0, b"ab": 2}
    ranks_b = {b"ab": 2, b"a": 0, b"b": 1}

    fp_a = save_mergeable_ranks(path_a, ranks_a)
    fp_b = save_mergeable_ranks(path_b, ranks_b)

    assert fp_a == fp_b

    loaded = load_mergeable_ranks(path_a)
    assert loaded == {b"a": 0, b"b": 1, b"ab": 2}


@pytest.mark.fast
def test_load_mergeable_ranks_rejects_invalid_payloads(tmp_path: Path):
    path = tmp_path / "invalid.msgpack"

    # Missing mergeable_ranks
    path.write_bytes(__import__("msgpack").packb({"version": 1}, use_bin_type=True))
    with pytest.raises(ValueError, match="Missing 'mergeable_ranks'"):
        load_mergeable_ranks(path)

    # Version must be int
    path.write_bytes(
        __import__("msgpack").packb({"version": "1", "mergeable_ranks": {}}, use_bin_type=True)
    )
    with pytest.raises(ValueError, match="version' must be an integer"):
        load_mergeable_ranks(path)

    # Unsupported version
    path.write_bytes(
        __import__("msgpack").packb({"version": 2, "mergeable_ranks": {}}, use_bin_type=True)
    )
    with pytest.raises(ValueError, match="Unsupported tokenizer file version"):
        load_mergeable_ranks(path)


@pytest.mark.fast
def test_validate_mergeable_ranks_and_special_overlap_errors():
    with pytest.raises(AssertionError, match="cannot have empty"):
        validate_mergeable_ranks({})

    with pytest.raises(AssertionError, match="non-empty bytes"):
        validate_mergeable_ranks({"a": 0})  # type: ignore[arg-type]

    with pytest.raises(AssertionError, match="start at 0"):
        validate_mergeable_ranks({b"a": 1})

    with pytest.raises(AssertionError, match="not contiguous"):
        validate_mergeable_ranks({b"a": 0, b"b": 2})

    with pytest.raises(AssertionError, match="overlap"):
        validate_no_special_token_overlap({b"<|bos|>": 0}, {"<|bos|>": 0})


@pytest.mark.fast
def test_truncated_from_pretrained_rejects_vocab_below_byte_tokens(monkeypatch):
    # Avoid disk lookups and base-tokenizer loading for this branch test.
    monkeypatch.setattr(
        "gpt_lab.tokenizer.tokenizer.Tokenizer.from_disk",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    dummy_base = type("DummyBase", (), {"vocab_size": 300})()
    monkeypatch.setattr(
        "gpt_lab.tokenizer.tokenizer.Tokenizer.from_pretrained",
        lambda *_args, **_kwargs: dummy_base,
    )

    with pytest.raises(ValueError, match="must retain all 256 byte-level tokens"):
        truncated_from_pretrained(
            base_name="gpt2",
            new_vocab_size=len(SpecialTokens().list()) + 255,
            source="tiktoken",
            special_tokens=SpecialTokens(),
        )


@pytest.mark.fast
def test_truncated_from_pretrained_returns_base_when_target_not_smaller(monkeypatch):
    class DummyBase:
        def __init__(self):
            self.vocab_size = 300
            self.mergeable_ranks = {bytes([i]): i for i in range(256)}
            self.config = type("Cfg", (), {"source": "tiktoken", "pat_str": "x", "special_tokens": SpecialTokens()})()

    dummy_base = DummyBase()

    monkeypatch.setattr(
        "gpt_lab.tokenizer.tokenizer.Tokenizer.from_disk",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(
        "gpt_lab.tokenizer.tokenizer.Tokenizer.from_pretrained",
        lambda *_args, **_kwargs: dummy_base,
    )

    out = truncated_from_pretrained(
        base_name="gpt2",
        new_vocab_size=dummy_base.vocab_size,
        source="tiktoken",
        special_tokens=SpecialTokens(),
    )
    assert out is dummy_base


@pytest.mark.fast
def test_train_huggingface_from_iterator_requires_tokenizers(monkeypatch):
    monkeypatch.setattr(tokenizer_hf, "HFTokenizer", None)

    cfg = type(
        "DummyCfg",
        (),
        {
            "vocab_size": 300,
            "pat_str": "gpt2",
            "show_progress": False,
            "special_tokens": type("DummyST", (), {"list": lambda self: ["<|bos|>"]})(),
        },
    )()

    with pytest.raises(RuntimeError, match="tokenizers library is required"):
        tokenizer_hf.train_huggingface_from_iterator(["hello"], cfg)


@pytest.mark.fast
def test_hf_wrapper_encode_decode_and_special_tokens():
    class DummyAdded:
        def __init__(self, content):
            self.content = content

    class DummyEncodeOut:
        def __init__(self, ids):
            self.ids = ids

    class DummyMain:
        def get_added_tokens_decoder(self):
            return {0: DummyAdded("<s>"), 1: DummyAdded("</s>")}

        def encode(self, text, add_special_tokens=False):
            _ = add_special_tokens
            return DummyEncodeOut([len(text)])

        def decode(self, ids, skip_special_tokens=False):
            _ = skip_special_tokens
            return f"decoded-{sum(ids)}"

    cfg = type("Cfg", (), {"name": "dummy", "source": "huggingface", "vocab_size": 10, "pat_str": None})()
    wrapper = tokenizer_hf.HuggingFaceTokenizerWrapper(DummyMain(), cfg)

    assert wrapper.special_tokens == ["<s>", "</s>"]
    assert wrapper.encode("abcd") == [4]
    assert wrapper.decode([1, 2, 3]) == "decoded-6"

@pytest.mark.fast
def test_compute_optimal_vocab_size_with_explicit_tokenizer_model(monkeypatch):
    class DummyTokenizer:
        vocab_size = 777

    monkeypatch.setattr(
        tokenizer_auto.Tokenizer,
        "from_pretrained",
        lambda name: DummyTokenizer(),
    )

    out = tokenizer_auto.compute_optimal_vocab_size(
        depth=4,
        aspect_ratio=16,
        train_tokenizer=False,
        tokenizer_model="gpt2",
        special_tokens=SpecialTokens(),
    )
    assert out == 777


@pytest.mark.fast
def test_compute_optimal_vocab_size_raises_when_too_small(monkeypatch):
    class DummyMetaModel:
        n_params = 1

    monkeypatch.setattr(tokenizer_auto, "build_meta_model", lambda _cfg: DummyMetaModel())

    with pytest.raises(ValueError, match="<256"):
        tokenizer_auto.compute_optimal_vocab_size(
            depth=2,
            aspect_ratio=8,
            train_tokenizer=False,
            tokenizer_model=None,
            special_tokens=SpecialTokens(),
            get_closest=lambda _x: ("tiny", 128),
        )


@pytest.mark.fast
def test_resolve_tokenizer_explicit_or_auto(monkeypatch):
    monkeypatch.setattr(
        tokenizer_auto,
        "get_closest_tokenizer_size",
        lambda _vocab_size: ("cl100k_base", 100000),
    )

    assert tokenizer_auto.resolve_tokenizer("gpt2", 32000, SpecialTokens()) == "gpt2"
    assert tokenizer_auto.resolve_tokenizer(None, 32000, SpecialTokens()) == "cl100k_base"
    assert tokenizer_auto.resolve_tokenizer("auto", 32000, SpecialTokens()) == "cl100k_base"


@pytest.mark.fast
def test_build_or_load_tokenizer_notrain_uses_pretrained(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        tokenizer_auto.Tokenizer,
        "from_pretrained",
        lambda _name: sentinel,
    )

    out = tokenizer_auto.build_or_load_tokenizer(
        tname="gpt2",
        vocab_size=32000,
        train_tokenizer=False,
        base_name="unused",
        pat_str="gpt2",
        special_tokens=SpecialTokens(),
        data_dir="unused",
        random_seed=42,
    )
    assert out is sentinel


@pytest.mark.fast
def test_build_or_load_tokenizer_training_path(monkeypatch):
    created_cfg = {}

    class FakeTrainerCfg:
        def __init__(self, **kwargs):
            created_cfg.update(kwargs)

    class FakeCorpus:
        def iterator(self):
            return iter(["abc", "def"])

    sentinel = object()

    monkeypatch.setattr(tokenizer_auto, "TokenizerTrainerConfig", FakeTrainerCfg)
    monkeypatch.setattr(
        tokenizer_auto.TokenizerCorpus,
        "from_sources",
        lambda **_kwargs: FakeCorpus(),
    )
    monkeypatch.setattr(
        tokenizer_auto.Tokenizer,
        "train_from_iterator",
        lambda text_iterator, config: sentinel,
    )

    out = tokenizer_auto.build_or_load_tokenizer(
        tname=None,
        vocab_size=4096,
        train_tokenizer=True,
        base_name="my_tok",
        pat_str="gpt2",
        special_tokens=SpecialTokens(),
        data_dir="/tmp/corpus",
        random_seed=7,
        dirname="/tmp/tokdir",
    )

    assert out is sentinel
    assert created_cfg["name"] == "my_tok"
    assert created_cfg["vocab_size"] == 4096
    assert created_cfg["dirname"] == "/tmp/tokdir"
