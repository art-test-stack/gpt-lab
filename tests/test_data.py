from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gpt_lab.data.loader import _document_batches, build_dataloader
from gpt_lab.data.sharder import ShardManager
from gpt_lab.utils.schemas import DataLoaderState


DIST = {"RANK": 0, "WORLD_SIZE": 1, "DEVICE": "cpu"}


def _write_shards(root: Path):
    dataset = root / "tiny"
    dataset.mkdir()
    pq.write_table(
        pa.table({"text": [f"doc-{i}" for i in range(6)]}),
        dataset / "shard_00000.parquet",
        row_group_size=6,
    )
    pq.write_table(
        pa.table({"text": ["validation"]}),
        dataset / "shard_00001.parquet",
    )


def test_shard_manager_resume_continues_after_saved_batch(tmp_path):
    _write_shards(tmp_path)
    manager = ShardManager("tiny", cachedir=tmp_path, dist_info=DIST)
    state = DataLoaderState(shard_idx=0, row_group_idx=0, offset_in_row_group=2)

    texts, resumed = next(manager.iterate(start_state=state, batch_size=4))

    assert texts == ["doc-3"]
    assert resumed.offset_in_row_group == 3


def test_nanochat_loader_defaults_to_cache_and_prepends_bos(tmp_path):
    _write_shards(tmp_path)

    class Tokenizer:
        def encode(self, texts, prepend_bos=False, **_):
            assert prepend_bos
            return [[99, len(text)] for text in texts]

        def get_bos_token_id(self):
            return 99

    loader = build_dataloader(
        "tiny", 1, 2, tokenizer=Tokenizer(), cachedir=tmp_path,
        dist_info=DIST, use_nanochat=True, buffer_size=1,
    )

    inputs, _, _ = next(loader)
    assert inputs[0, 0].item() == 99


def test_remote_probe_returns_last_shard_index():
    manager = object.__new__(ShardManager)
    manager.base_url = "https://example.test"
    manager.ddp_rank = 1

    class Session:
        def head(self, url, timeout):
            index = int(url.rsplit("_", 1)[1].split(".", 1)[0])
            return type("Response", (), {"status_code": 200 if index < 5 else 404})()

    manager._session = Session()
    assert manager.get_num_remote_shards() == 4


def test_nanochat_loader_rejects_dataset_without_train_shard(tmp_path):
    pq.write_table(pa.table({"text": ["validation"]}), tmp_path / "shard_00000.parquet")

    try:
        next(_document_batches("train", None, 1, tmp_path))
    except ValueError as error:
        assert "two parquet shards" in str(error)
    else:
        raise AssertionError("single-shard dataset should be rejected")
