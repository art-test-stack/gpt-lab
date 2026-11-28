"""Native text parity using GPT-2 from an explicitly configured local cache."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gpt_lab.data import DistDataLoader, build_dataloader
from gpt_lab.rust_tools import dataloader as native
from gpt_lab.tokenizer import Tokenizer
from gpt_lab.utils.schemas import DataLoaderState, TokenizerConfig


@pytest.fixture(scope="module")
def tokenizer(native_extension):
    cache = os.environ.get("TIKTOKEN_CACHE_DIR")
    keys = ("6c7ea1a7e38e3a7f062df639a5b80947f075ffe6", "6d1cbeee0f20b3d9449abfede4726ed8212e3aee")
    if not cache or not all((Path(cache) / key).is_file() for key in keys):
        if os.environ.get("GPT_LAB_REQUIRE_NATIVE") == "1":
            pytest.fail("Preload GPT-2 into TIKTOKEN_CACHE_DIR before running native tests")
        pytest.skip("Set TIKTOKEN_CACHE_DIR to an existing GPT-2 cache")
    return Tokenizer.from_pretrained("gpt2", source="tiktoken")


@pytest.fixture
def byte_tokenizer():
    config = TokenizerConfig(name="byte-test", vocab_size=257, pat_str="gpt2")
    return Tokenizer({bytes([i]): i for i in range(256)}, config.special_tokens.list(), config)


@pytest.fixture
def text_shards(tmp_path):
    dataset = tmp_path / "native-text"
    dataset.mkdir()
    for shard in range(5):
        texts = [f"shard {shard} document {index}: {text}" if text else "" for index, text in enumerate([
            "a", "b", " long document" * 29, "", "café 東京 🙂 <|bos|> <|endoftext|>",
            "x y", " retained suffix" * 23, "short",
        ] * 18)]
        pq.write_table(pa.table({"text": texts}), dataset / f"shard_{shard:05d}.parquet",
                       row_group_size=36 if shard == 4 else 72, compression="zstd")
    return tmp_path


def make_loader(root, tokenizer, backend, rank, *, split="train", state=None,
                packing_strategy="rolling-16"):
    selector = {} if backend is None else {"dataloader_backend": backend}
    return build_dataloader(
        name="native-text", cachedir=root, max_shards=4, tokenizer=tokenizer,
        batch_size=2, seq_len=17, tokenizer_threads=2, tokenizer_batch_size=64,
        split=split, resume_state=state, data_seed=19, **selector,
        packing_strategy=packing_strategy,
        comp_info=SimpleNamespace(rank=rank, world_size=2, device=torch.device("cpu")),
    )


def assert_equal(expected, actual):
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])
    assert expected[2].model_dump_json() == actual[2].model_dump_json()


@pytest.mark.parametrize("compression", ["gzip", "brotli", "lz4"])
def test_default_native_loader_reads_cached_parquet_codecs(tmp_path, byte_tokenizer, native_extension, compression):
    path = tmp_path / "codec"
    path.mkdir()
    pq.write_table(pa.table({"text": ["café", "長い文", "short", "tail"] * 4}),
                   path / "shard_00000.parquet", compression=compression, row_group_size=4)
    args = dict(name=path.name, cachedir=tmp_path, max_shards=0, split="val",
                batch_size=1, seq_len=7, tokenizer=byte_tokenizer,
                comp_info=SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu")))
    default = build_dataloader(**args)()
    python = build_dataloader(**args, dataloader_backend="python")()
    assert default._fill_batch is not DistDataLoader._pack_python
    batches = 0
    for actual, expected in zip(default, python, strict=True):
        assert_equal(expected, actual)
        batches += 1
    assert batches > 0


def test_missing_extension_preserves_python_default_and_strict_native_requests(text_shards, byte_tokenizer, monkeypatch):
    monkeypatch.setattr(native, "_EXTENSION", "_gpt_lab_missing_test_extension")
    args = dict(name="native-text", cachedir=text_shards, max_shards=4, split="train",
                batch_size=2, seq_len=17, tokenizer=byte_tokenizer,
                comp_info=SimpleNamespace(rank=0, world_size=2, device=torch.device("cpu")))
    default = build_dataloader(**args)
    assert default._fill_batch is DistDataLoader._pack_python
    assert_equal(next(default), next(build_dataloader(**args, dataloader_backend="python")))
    block = build_dataloader(**args, packing_strategy="block-best-fit")
    assert block._fill_batch is DistDataLoader._pack_python
    for selector in ({"dataloader_backend": "rust"}, {"dataloader_backend": "rust-packing"}):
        with pytest.raises(ImportError, match="rust-dataloader"):
            build_dataloader(**args, **selector)


def test_default_uses_only_rank_owned_cached_shards(tmp_path, byte_tokenizer, native_extension):
    path = tmp_path / "partial"
    path.mkdir()
    # Seed 0 gives train order (0, 1). Rank 1's shard still needs downloading.
    pq.write_table(pa.table({"text": ["cached"] * 8}), path / "shard_00000.parquet")
    args = dict(name=path.name, cachedir=tmp_path, max_shards=2, base_url="https://example.invalid",
                data_seed=0, split="train", batch_size=1, seq_len=3, tokenizer=byte_tokenizer)
    for rank in (0, 1):
        context = SimpleNamespace(rank=rank, world_size=2, device=torch.device("cpu"))
        default = build_dataloader(**args, comp_info=context)
        assert (default._fill_batch is not DistDataLoader._pack_python) == (rank == 0)
    with pytest.raises(FileNotFoundError, match="shard_00001"):
        build_dataloader(**args, comp_info=context, dataloader_backend="rust")


@pytest.mark.parametrize("strategy", ["rolling-16", "block-best-fit"])
@pytest.mark.parametrize("rank", [0, 1])
def test_training_parity_and_resume(text_shards, tokenizer, rank, strategy):
    def build(backend, state=None):
        return make_loader(text_shards, tokenizer, backend, rank, state=state, packing_strategy=strategy)

    python, rust = build("python"), build("rust")
    comparisons = []
    if strategy == "rolling-16":
        default = build(None)
        assert default._fill_batch is not DistDataLoader._pack_python
        comparisons.extend((default, build("rust-packing")))
    resumed_epochs = set()
    seen = set()
    spanned = [False, False]
    previous = None
    drained = False
    for batch in range(1, 1101):
        expected = next(python)
        actual = next(rust)
        try:
            assert_equal(expected, actual)
            for loader in comparisons:
                assert_equal(expected, next(loader))
        except AssertionError as exc:
            raise AssertionError(f"{strategy}, rank {rank}, batch {batch}: {exc}") from exc
        state = expected[2]
        seen.add((state.epoch, state.global_shard_idx, state.row_group_idx))
        if (state.epoch <= 2 and state.epoch not in resumed_epochs
                and state.lookahead_refs and state.pending_document_ref):
            assert state.document_token_offset > 0
            comparisons.extend((
                build("rust", DataLoaderState.model_validate_json(state.model_dump_json())),
                build("python", DataLoaderState.model_validate_json(actual[2].model_dump_json())),
            ))
            resumed_epochs.add(state.epoch)
        if strategy == "block-best-fit":
            refs = state.lookahead_refs
            spanned[0] |= len({ref[0] for ref in refs}) > 1
            spanned[1] |= len({ref[:2] for ref in refs}) > 1
            if previous is not None:
                drained |= (
                    (state.epoch, state.global_shard_idx, state.row_group_idx, state.offset_in_row_group)
                    == (previous.epoch, previous.global_shard_idx, previous.row_group_idx, previous.offset_in_row_group)
                    and 0 < len(refs) < len(previous.lookahead_refs)
                )
        previous = state
    assert resumed_epochs == {1, 2}
    assert len({entry[0] for entry in seen}) > 1
    assert len({entry[1] for entry in seen}) == 2
    assert len({entry[2] for entry in seen}) == 2
    if strategy == "block-best-fit":
        assert drained and all(spanned)


def test_resume_frozen_master_rolling_checkpoint(text_shards, tokenizer):
    fixture = Path(__file__).with_name("fixtures") / "rolling16_master.json"
    for case in json.loads(fixture.read_text())["cases"]:
        saved = DataLoaderState.model_validate(case["state"])
        assert saved.lookahead_refs and saved.pending_document_ref and saved.document_token_offset
        expected = (torch.tensor(case["next_x"]), torch.tensor(case["next_y"]),
                    DataLoaderState.model_validate(case["next_state"]))
        for backend in ("python", "rust"):
            uninterrupted = make_loader(text_shards, tokenizer, backend, case["rank"])
            for _ in range(case["batch"]):
                _, _, state = next(uninterrupted)
            assert state.model_dump_json() == saved.model_dump_json()
            restored = make_loader(text_shards, tokenizer, backend, case["rank"], state=saved)
            assert_equal(expected, next(restored))


@pytest.mark.parametrize("rank", [0, 1])
def test_native_text_finite_validation(text_shards, tokenizer, rank):
    loaders = [make_loader(text_shards, tokenizer, backend, rank, split="val")()
               for backend in ("python", "rust-packing", "rust")]
    groups = set()
    while True:
        reference = next(loaders[0], None)
        packed, native = next(loaders[1], None), next(loaders[2], None)
        if reference is None:
            assert packed is None and native is None
            break
        assert_equal(reference, packed)
        assert_equal(reference, native)
        groups.add(reference[2].row_group_idx)
    assert groups == {rank, rank + 2}


@pytest.mark.parametrize("source,target", [("rolling-16", "block-best-fit"),
                                           ("block-best-fit", "rolling-16")])
def test_block_best_fit_rejects_other_strategy_checkpoint(text_shards, tokenizer, source, target):
    loader = make_loader(text_shards, tokenizer, "rust", 0, packing_strategy=source)
    saved = DataLoaderState.model_validate_json(next(loader)[2].model_dump_json())
    with pytest.raises(ValueError, match="Incompatible training data checkpoint"):
        make_loader(text_shards, tokenizer, "rust", 0, state=saved, packing_strategy=target)


def test_block_best_fit_is_training_only(text_shards, tokenizer):
    with pytest.raises(ValueError, match="train"):
        make_loader(text_shards, tokenizer, "rust", 0, split="val", packing_strategy="block-best-fit")()


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_exact_pair_nine_six_four(tmp_path, tokenizer, backend):
    dataset = tmp_path / "exact-pair"
    dataset.mkdir()
    texts = [" a" * 8, " b" * 5, " c" * 3]
    documents = [tokenizer(text, prepend_bos=True) for text in texts]
    assert list(map(len, documents)) == [9, 6, 4]
    pq.write_table(pa.table({"text": texts}), dataset / "shard_00000.parquet")
    loader = build_dataloader(
        name=dataset.name, cachedir=tmp_path, max_shards=1, tokenizer=tokenizer,
        batch_size=1, seq_len=9, dataloader_backend=backend, packing_strategy="block-best-fit",
        comp_info=SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu")),
    )
    x, y, state = next(loader)
    row = documents[1] + documents[2]
    assert x.tolist() == [row[:-1]]
    assert y.tolist() == [row[1:]]
    assert state.lookahead_refs == ((0, 0, 0),)
    assert state.pending_document_ref is None
