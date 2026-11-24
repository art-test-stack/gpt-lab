"""Native text parity using GPT-2 from an explicitly configured local cache."""
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gpt_lab.data.loader import build_dataloader
from gpt_lab.tokenizer import Tokenizer
from gpt_lab.utils.schemas import DataLoaderState


@pytest.fixture(scope="module")
def tokenizer():
    cache = os.environ.get("TIKTOKEN_CACHE_DIR")
    keys = ("6c7ea1a7e38e3a7f062df639a5b80947f075ffe6", "6d1cbeee0f20b3d9449abfede4726ed8212e3aee")
    if not cache or not all((Path(cache) / key).is_file() for key in keys):
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


def make_loader(root, tokenizer, backend, rank, *, split="train", state=None):
    return build_dataloader(
        name="native-text", cachedir=root, max_shards=4, tokenizer=tokenizer,
        batch_size=2, seq_len=17, tokenizer_threads=2, tokenizer_batch_size=64,
        dataloader_backend=backend, split=split, resume_state=state, data_seed=19,
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
