import pytest
import random
import string
from pathlib import Path
import json

from gpt_lab.tokenizer.serialization import (
    load_mergeable_ranks,
    save_mergeable_ranks,
    validate_mergeable_ranks,
    validate_no_special_token_overlap,
)
from gpt_lab.tokenizer.truncation import parse_truncated_name, truncated_from_pretrained
from gpt_lab.utils.special_tokens import SpecialTokens
from gpt_lab.tokenizer import Tokenizer, TokenizerConfig
from gpt_lab.utils.schemas import TokenizerTrainerConfig
from gpt_lab.tokenizer import hf as tokenizer_hf
from gpt_lab.tokenizer.corpus import TokenizerCorpus
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
    config = TokenizerConfig(name="gpt2", vocab_size=-1, pat_str="gpt2", source="tiktoken", special_tokens=special_tokens)
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
    config = TokenizerConfig(name="gpt2", vocab_size=-1, pat_str="gpt2", source="tiktoken", special_tokens=special_tokens)
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
def test_compute_optimal_vocab_size_matches_reference_fit():
    # Golden value from the Approach-1 coefficients published with
    # Tao et al. (2024), evaluated at N_nv=7B and d_model=4096.
    out = tokenizer_auto.compute_optimal_vocab_size(
        n_non_vocab_params=7_000_000_000,
        d_model=4096,
    )
    assert out == pytest.approx(62_280.97134786828)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("n_non_vocab_params", "d_model", "message"),
    [
        (0, 4096, "n_non_vocab_params must be positive"),
        (1_000_000, 0, "d_model must be positive"),
    ],
)
def test_compute_optimal_vocab_size_validates_inputs(
    n_non_vocab_params,
    d_model,
    message,
):
    with pytest.raises(ValueError, match=message):
        tokenizer_auto.compute_optimal_vocab_size(
            n_non_vocab_params=n_non_vocab_params,
            d_model=d_model,
        )


@pytest.mark.fast
def test_round_vocab_size():
    assert tokenizer_auto.round_vocab_size(25_998) == 25_984
    assert tokenizer_auto.round_vocab_size(10, minimum=257) == 257


@pytest.mark.fast
def test_resolve_tokenizer_explicit_or_auto(monkeypatch):
    requested_sizes = []

    def fake_closest(vocab_size):
        requested_sizes.append(vocab_size)
        return "cl100k_base", 100_000

    monkeypatch.setattr(
        tokenizer_auto,
        "get_closest_tokenizer_size",
        fake_closest,
    )

    special_tokens = SpecialTokens()
    total_vocab_size = 32_000 + len(special_tokens)
    assert (
        tokenizer_auto.resolve_tokenizer(
            "gpt2",
            total_vocab_size,
            special_tokens,
        )
        == "gpt2"
    )
    assert (
        tokenizer_auto.resolve_tokenizer(None, total_vocab_size, special_tokens)
        == "cl100k_base"
    )
    assert (
        tokenizer_auto.resolve_tokenizer("auto", total_vocab_size, special_tokens)
        == "cl100k_base"
    )
    assert requested_sizes == [32_000, 32_000]


@pytest.mark.fast
def test_load_tokenizer_uses_pretrained_when_local_config_is_absent(
    monkeypatch,
    tmp_path,
):
    sentinel = object()
    monkeypatch.setattr(
        "gpt_lab.tokenizer.auto.TokenizerConfig.from_directory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("missing")
        ),
    )
    calls = []

    def fake_from_pretrained(name, special_tokens):
        calls.append((name, special_tokens))
        return sentinel

    monkeypatch.setattr(
        "gpt_lab.tokenizer.auto.Tokenizer.from_pretrained",
        fake_from_pretrained,
    )

    special_tokens = SpecialTokens()
    out = tokenizer_auto.load_tokenizer(
        "gpt2",
        special_tokens=special_tokens,
        tokenizer_dir=tmp_path,
    )
    assert out is sentinel
    assert calls == [("gpt2", special_tokens)]


@pytest.mark.fast
def test_load_tokenizer_prefers_valid_local_tokenizer(monkeypatch, tmp_path):
    special_tokens = SpecialTokens()
    local_config = TokenizerConfig(
        name="local",
        dirname=tmp_path,
        source="local",
        vocab_size=257,
        pat_str="gpt2",
        special_tokens=special_tokens,
    )
    sentinel = object()
    monkeypatch.setattr(
        "gpt_lab.tokenizer.auto.TokenizerConfig.from_directory",
        lambda *_args, **_kwargs: local_config,
    )
    monkeypatch.setattr(
        "gpt_lab.tokenizer.auto.Tokenizer.from_disk",
        lambda *_args, **_kwargs: sentinel,
    )
    monkeypatch.setattr(
        "gpt_lab.tokenizer.auto.Tokenizer.from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("pretrained fallback was used"),
    )

    out = tokenizer_auto.load_tokenizer(
        "local",
        special_tokens=SpecialTokens(),
        tokenizer_dir=tmp_path,
    )
    assert out is sentinel


@pytest.mark.fast
def test_build_or_load_tokenizer_training_path(monkeypatch, tmp_path):
    captured = {}

    class FakeCorpus:
        def iterator(self):
            return iter(["abc", "def"])

    class DummyTokenizer:
        vocab_size = 4096

    monkeypatch.setattr(
        "gpt_lab.tokenizer.corpus.TokenizerCorpus.from_sources",
        lambda **kwargs: captured.setdefault("corpus_kwargs", kwargs)
        and FakeCorpus(),
    )

    def fake_train(text_iterator, config):
        captured["documents"] = list(text_iterator)
        captured["config"] = config
        return DummyTokenizer()

    monkeypatch.setattr(
        "gpt_lab.tokenizer.tokenizer.Tokenizer.train_from_iterator",
        fake_train,
    )
    monkeypatch.setattr(
        tokenizer_auto,
        "resolve_tokenizer",
        lambda *_args, **_kwargs: pytest.fail(
            "training must not resolve a cached tokenizer"
        ),
    )

    trainer_config = TokenizerTrainerConfig(
        source="huggingface",
        max_bytes=12_345,
        bytes_per_doc=321,
        show_progress=False,
    )
    out = tokenizer_auto.build_or_load_tokenizer(
        name=None,
        vocab_size=4096,
        train_tokenizer=True,
        base_name="my_tok",
        pat_str="gpt2",
        special_tokens=SpecialTokens(),
        data_dir="/tmp/corpus",
        random_seed=7,
        dirname=tmp_path,
        trainer_config=trainer_config,
    )

    assert isinstance(out, DummyTokenizer)
    assert captured["documents"] == ["abc", "def"]
    assert captured["corpus_kwargs"] == {
        "corpus_dir": "/tmp/corpus",
        "random_seed": 7,
        "max_bytes": 12_345,
        "bytes_per_doc": 321,
    }
    config = captured["config"]
    assert config.name == "my_tok"
    assert config.source == "local"
    assert config.dirname == tmp_path / "my_tok"
    assert config.trainer == trainer_config


@pytest.mark.fast
def test_local_config_reads_saved_msgpack_vocabulary(tmp_path):
    special_tokens = SpecialTokens()
    ranks = {bytes([index]): index for index in range(256)}
    tokenizer_dir = tmp_path / "local_tok"
    tokenizer_dir.mkdir()
    save_mergeable_ranks(tokenizer_dir / "mergeable_ranks.msgpack", ranks)

    config = TokenizerConfig(
        name="local_tok",
        dirname=tmp_path,
        source="local",
        vocab_size=len(ranks) + len(special_tokens),
        pat_str="gpt2",
        special_tokens=special_tokens,
    )

    assert config.get_mergeable_ranks() == ranks
    tokenizer = Tokenizer.from_config(config)
    assert tokenizer.vocab_size == len(ranks) + len(special_tokens)
    config.save_to_directory()

    reloaded = tokenizer_auto.load_tokenizer(
        "local_tok",
        special_tokens=special_tokens,
        tokenizer_dir=tmp_path,
    )
    assert reloaded.mergeable_ranks == ranks
    assert reloaded.vocab_size == tokenizer.vocab_size


@pytest.mark.fast
def test_train_huggingface_from_iterator_requires_tokenizers(monkeypatch):
    import gpt_lab.tokenizer.hf as tokenizer_hf
    monkeypatch.setattr(tokenizer_hf, "HFTokenizer", None)

    special_tokens = SpecialTokens()
    trainer_cfg = TokenizerTrainerConfig(
        source="huggingface",
        show_progress=False,
    )
    cfg = TokenizerConfig(
        name="test",
        vocab_size=300,
        pat_str="gpt2",
        special_tokens=special_tokens,
        source="huggingface",
        trainer=trainer_cfg,
    )

    with pytest.raises(Exception):
        tokenizer_hf.train_huggingface_from_iterator(["hello"], cfg)


@pytest.mark.fast
def test_hf_wrapper_encode_decode_and_special_tokens():
    import gpt_lab.tokenizer.hf as tokenizer_hf

    class DummyAdded:
        def __init__(self, content):
            self.content = content

    class DummyEncodeOut:
        def __init__(self, ids):
            self.ids = ids

    class DummyMain:
        def get_vocab_size(self, with_added_tokens=True):
            assert with_added_tokens
            return 10

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
    assert wrapper.vocab_size == 10
    assert wrapper.encode("abcd") == [4]
    assert wrapper.decode([1, 2, 3]) == "decoded-6"


@pytest.mark.fast
def test_tokenizer_forwards_special_tokens_to_huggingface(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_from_pretrained(name, special_tokens):
        captured["args"] = (name, special_tokens)
        return sentinel

    monkeypatch.setattr(
        tokenizer_hf.HuggingFaceTokenizerWrapper,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    special_tokens = SpecialTokens()
    result = Tokenizer.from_pretrained(
        "org/model",
        source="huggingface",
        special_tokens=special_tokens,
    )

    assert result is sentinel
    assert captured["args"] == ("org/model", special_tokens)

    config = TokenizerConfig(
        name="org/model",
        source="huggingface",
        vocab_size=1000,
        pat_str=None,
        special_tokens=special_tokens,
    )
    assert Tokenizer.from_config(config) is sentinel
    assert captured["args"] == ("org/model", special_tokens)


@pytest.mark.fast
def test_tokenizer_auto_source_loads_tiktoken_without_scoping_error():
    special_tokens = SpecialTokens()

    tokenizer = Tokenizer.from_pretrained(
        "gpt2",
        special_tokens=special_tokens,
    )

    assert tokenizer.config.source == "tiktoken"
    assert tokenizer.vocab_size == len(tokenizer.mergeable_ranks) + len(
        special_tokens
    )


@pytest.mark.fast
def test_train_huggingface_from_iterator_with_mock_tokenizers(monkeypatch):
    import gpt_lab.tokenizer.hf as tokenizer_hf

    # Create dummy components that mimic the API used in hf.train_huggingface_from_iterator
    class DummyTokenizer:
        def __init__(self, *args, **kwargs):
            self._merges = [["a", "b"], ["Ġx", "y"]]
            self.normalizer = None
            self.pre_tokenizer = None
            self.decoder = None
            self.post_processor = None

        def train_from_iterator(self, iterator, trainer=None):
            # no-op: we already have _merges
            return None

        def to_str(self):
            return json.dumps({"model": {"merges": self._merges}})

        def get_vocab_size(self):
            return 123

        def save(self, path):
            Path(path).write_text(self.to_str())

    class DummyBPE:
        def __init__(self, *args, **kwargs):
            pass

    class DummyRegex:
        def __init__(self, pat):
            self.pat = pat

    class DummyPreTokenizers:
        class Split:
            def __init__(self, pattern=None, behavior=None, invert=None):
                self.pattern = pattern

        class ByteLevel:
            @staticmethod
            def alphabet():
                return [0, 1, 2]

            def __init__(self, add_prefix_space=False, use_regex=False):
                pass

        @staticmethod
        def Sequence(items):
            return items

    class DummyDecoders:
        class ByteLevel:
            def __init__(self):
                pass

    class DummyBpeTrainer:
        def __init__(self, *args, **kwargs):
            self.show_progress = kwargs.get("show_progress", False)

    # Monkeypatch the tokenizer_hf module-level names to our dummies
    monkeypatch.setattr(tokenizer_hf, "HFTokenizer", DummyTokenizer)
    monkeypatch.setattr(tokenizer_hf, "BPE", DummyBPE)
    monkeypatch.setattr(tokenizer_hf, "Regex", DummyRegex)
    monkeypatch.setattr(tokenizer_hf, "pre_tokenizers", DummyPreTokenizers)
    monkeypatch.setattr(tokenizer_hf, "decoders", DummyDecoders)
    monkeypatch.setattr(tokenizer_hf, "BpeTrainer", DummyBpeTrainer)

    # Create a proper TokenizerConfig with trainer
    special_tokens = SpecialTokens()
    trainer_cfg = TokenizerTrainerConfig(
        source="huggingface",
        show_progress=False,
    )
    cfg = TokenizerConfig(
        name="test",
        vocab_size=300,
        pat_str="\\w+",
        special_tokens=special_tokens,
        source="huggingface",
        trainer=trainer_cfg,
    )

    out = tokenizer_hf.train_huggingface_from_iterator(["hello world"], cfg)

    # Expect mergeable ranks to be a dict mapping bytes to ints and include single-byte entries
    assert isinstance(out, dict)
    # Check that merged pairs produced exact byte keys and ranks
    # Dummy merges: ["a","b"] -> b"ab" with rank 256, ["Ġx","y"] -> b" xy" with rank 257
    assert out.get(b"ab") == 256
    assert out.get(" xy".encode("utf-8")) == 257

    # All single-byte tokens should be present and map to their own integer values
    for i in range(256):
        assert out.get(bytes([i])) == i

    # Total entries should equal 256 single-bytes + 2 merges
    assert len(out) == 256 + 2
