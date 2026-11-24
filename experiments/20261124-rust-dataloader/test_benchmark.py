"""The selected benchmark shard must not redefine the source manifest."""
import importlib.util
from contextlib import closing
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gpt_lab.data.sharder import ShardManager
from gpt_lab.tokenizer import Tokenizer


@pytest.mark.parametrize("metadata", [None, b'{ "name": "source", "base_url": "", "column_name": "text", "max_shards": 2 }\n'])
@pytest.mark.parametrize("backend", ["baseline", "python", "rust"])
def test_selected_shard_preserves_source_metadata(tmp_path, monkeypatch, metadata, backend):
    spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).with_name("benchmark.py"))
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        pq.write_table(pa.table({"text": ["1 2 3 4"]}), source / f"shard_{index:05d}.parquet")
    meta = source / "meta.json"
    if metadata is not None:
        meta.write_bytes(metadata)

    class Encoder:
        def get_bos_token_id(self):
            return 0

        def __call__(self, text, **kwargs):
            return [0, *map(int, text.split())]

    monkeypatch.setattr(Tokenizer, "from_pretrained", lambda *args, **kwargs: Encoder())
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(benchmark, "SCRATCH", tmp_path / "benchmark-scratch")
    monkeypatch.setattr(benchmark, "BATCH_SIZE", 1)
    monkeypatch.setattr(benchmark, "CONTEXT", 3)
    result = benchmark.measure(SimpleNamespace(corpus="climbmix", backend=backend,
                                              shard=source / "shard_00001.parquet", threads=1))
    assert result["batches"] == 1
    if metadata is None:
        assert not meta.exists()
    else:
        assert meta.read_bytes() == metadata
    context = SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu"))
    train = ShardManager("source", cachedir=tmp_path, split="train", comp_info=context)
    validation = ShardManager("source", cachedir=tmp_path, split="val", comp_info=context)
    with closing(train.iterate()) as chunks:
        assert {state.global_shard_idx for _, state in islice(chunks, 2)} == {0, 1}
    assert [state.global_shard_idx for _, state in validation.iterate()] == [2]
