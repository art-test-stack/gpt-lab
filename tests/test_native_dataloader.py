"""Native text parity using GPT-2 from an explicitly configured local cache."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gpt_lab.data import build_dataloader
from gpt_lab.tokenizer import Tokenizer
from gpt_lab.utils.schemas import DataLoaderState


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
    return build_dataloader(
        name="native-text", cachedir=root, max_shards=4, tokenizer=tokenizer,
        batch_size=2, seq_len=17, tokenizer_threads=2, tokenizer_batch_size=64,
        dataloader_backend=backend, split=split, resume_state=state, data_seed=19,
        packing_strategy=packing_strategy,
        comp_info=SimpleNamespace(rank=rank, world_size=2, device=torch.device("cpu")),
    )


def assert_equal(expected, actual):
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])
    assert expected[2] == actual[2]


@pytest.mark.parametrize("rank", [0, 1])
def test_native_text_batches_and_cross_backend_resume(text_shards, tokenizer, rank):
    loaders = [make_loader(text_shards, tokenizer, backend, rank)
               for backend in ("python", "rust-packing", "rust")]
    resumed = []
    seen = set()
    for _ in range(1100):
        reference = next(loaders[0])
        packed, native = next(loaders[1]), next(loaders[2])
        assert_equal(reference, packed)
        assert_equal(reference, native)
        for restored in resumed:
            assert_equal(reference, next(restored))
        state = reference[2]
        seen.add((state.epoch, state.global_shard_idx, state.row_group_idx))
        if not resumed and state.lookahead_refs and state.pending_document_ref:
            assert state.document_token_offset > 0
            for source, target in ((reference, "rust"), (native, "python")):
                saved = DataLoaderState.model_validate_json(source[2].model_dump_json())
                resumed.append(make_loader(text_shards, tokenizer, target, rank, state=saved))
    assert resumed
    assert len({entry[0] for entry in seen}) > 1
    assert len({entry[1] for entry in seen}) == 2
    assert len({entry[2] for entry in seen}) == 2


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


@pytest.mark.parametrize("rank", [0, 1])
def test_block_best_fit_matches_drained_python_blocks_and_resume(text_shards, tokenizer, rank):
    native = make_loader(text_shards, tokenizer, "rust", rank, packing_strategy="block-best-fit")
    reference = make_loader(text_shards, tokenizer, "python", rank)
    reference.lookahead_size = 64
    reference.signature = native.signature
    fill = reference._fill_lookahead

    def fill_empty_block():
        if not reference.lookahead:
            fill()

    reference._fill_lookahead = fill_empty_block
    greedy_take = reference._take_document

    def take_exact_fit(space):
        fill_empty_block()
        lengths = [len(document) for document, _ in reference.lookahead]
        for index, length in enumerate(lengths):
            if length == space:
                return reference.lookahead.pop(index)
        for first, length in enumerate(lengths):
            for other in lengths[first + 1:]:
                if length + other == space:
                    return reference.lookahead.pop(first)
        return greedy_take(space)

    reference._take_document = take_exact_fit
    resumed = []
    seen = set()
    previous = None
    drained_without_refill = False
    block_crossed_group = block_crossed_shard = False
    saved_epochs = set()
    for _ in range(1100):
        expected, actual = next(reference), next(native)
        assert_equal(expected, actual)
        for restored in resumed:
            assert_equal(expected, next(restored))
        state = actual[2]
        seen.add((state.epoch, state.global_shard_idx, state.row_group_idx))
        refs = state.lookahead_refs + ((state.pending_document_ref,) if state.pending_document_ref else ())
        block_crossed_group |= len({ref[:2] for ref in refs}) > 1
        block_crossed_shard |= len({ref[0] for ref in refs}) > 1
        if previous is not None and state.lookahead_refs:
            same_cursor = all(getattr(state, field) == getattr(previous, field) for field in (
                "epoch", "global_shard_idx", "row_group_idx", "offset_in_row_group"))
            drained_without_refill |= same_cursor and len(state.lookahead_refs) < len(previous.lookahead_refs)
        if (len(state.lookahead_refs) > 15 and state.pending_document_ref
                and state.document_token_offset > 0 and state.epoch not in saved_epochs
                and len(saved_epochs) < 2):
            saved = DataLoaderState.model_validate_json(state.model_dump_json())
            resumed.append(make_loader(text_shards, tokenizer, "rust", rank, state=saved,
                                       packing_strategy="block-best-fit"))
            saved_epochs.add(state.epoch)
        previous = state
    assert len(saved_epochs) == 2
    assert drained_without_refill
    assert block_crossed_group and block_crossed_shard
    assert len({entry[0] for entry in seen}) > 1
    assert len({entry[1] for entry in seen}) == 2
    assert len({entry[2] for entry in seen}) == 2


@pytest.mark.parametrize("source,target", [("rolling-16", "block-best-fit"),
                                           ("block-best-fit", "rolling-16")])
def test_block_best_fit_rejects_other_strategy_checkpoint(text_shards, tokenizer, source, target):
    loader = make_loader(text_shards, tokenizer, "rust", 0, packing_strategy=source)
    saved = DataLoaderState.model_validate_json(next(loader)[2].model_dump_json())
    with pytest.raises(ValueError, match="Incompatible training data checkpoint"):
        make_loader(text_shards, tokenizer, "rust", 0, state=saved, packing_strategy=target)


def test_exact_pair_rejects_greedy_block_checkpoint(text_shards, tokenizer, monkeypatch):
    old_signatures = []
    sha256 = hashlib.sha256

    def capture_greedy_signature(payload=b""):
        if b'"block-best-fit-exact-pair"' in payload:
            old_signatures.append(sha256(payload.replace(
                b'"block-best-fit-exact-pair"', b'"block-best-fit"',
            )).hexdigest())
        return sha256(payload)

    with monkeypatch.context() as patch:
        patch.setattr("gpt_lab.data.loader.hashlib.sha256", capture_greedy_signature)
        loader = make_loader(text_shards, tokenizer, "rust", 0, packing_strategy="block-best-fit")
    assert len(old_signatures) == 1
    saved = next(loader)[2].model_copy(update={"packing_signature": old_signatures[0]})
    with pytest.raises(ValueError, match="Incompatible training data checkpoint"):
        make_loader(text_shards, tokenizer, "rust", 0, state=saved, packing_strategy="block-best-fit")


def test_block_best_fit_is_training_only(text_shards, tokenizer):
    with pytest.raises(ValueError, match="train"):
        make_loader(text_shards, tokenizer, "rust", 0, split="val", packing_strategy="block-best-fit")()


def test_exact_pair_nine_six_four(tmp_path, tokenizer):
    dataset = tmp_path / "exact-pair"
    dataset.mkdir()
    texts = [" a" * 8, " b" * 5, " c" * 3]
    documents = [tokenizer(text, prepend_bos=True) for text in texts]
    assert list(map(len, documents)) == [9, 6, 4]
    pq.write_table(pa.table({"text": texts}), dataset / "shard_00000.parquet")
    loader = build_dataloader(
        name=dataset.name, cachedir=tmp_path, max_shards=1, tokenizer=tokenizer,
        batch_size=1, seq_len=9, dataloader_backend="rust", packing_strategy="block-best-fit",
        comp_info=SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu")),
    )
    x, y, state = next(loader)
    row = documents[1] + documents[2]
    assert x.tolist() == [row[:-1]]
    assert y.tolist() == [row[1:]]
    assert state.lookahead_refs == ((0, 0, 0),)
    assert state.pending_document_ref is None


@pytest.mark.parametrize("rank", [0, 1])
def test_rust_training_strategies_share_validation_batches(text_shards, tokenizer, rank):
    validation = []
    for strategy in ("rolling-16", "block-best-fit"):
        training = make_loader(text_shards, tokenizer, "rust", rank, packing_strategy=strategy)
        next(training)
        # The workflow passes the strategy only to training, leaving validation at rolling-16.
        validation.append(make_loader(text_shards, tokenizer, "rust", rank, split="val")())
    batches = 0
    while True:
        left, right = (next(loader, None) for loader in validation)
        if left is None:
            assert right is None
            break
        assert_equal(left, right)
        batches += 1
    assert batches > 0
