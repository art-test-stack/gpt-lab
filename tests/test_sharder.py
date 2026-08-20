import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gpt_lab.data.sharder import ShardManager


def _metadata_manager(tmp_path):
    manager = ShardManager.__new__(ShardManager)
    manager.ds_path = tmp_path
    return manager


@pytest.mark.fast
def test_metadata_write_never_exposes_truncated_destination(tmp_path, monkeypatch):
    manager = _metadata_manager(tmp_path)
    old_metadata = {"max_shards": 8}
    new_metadata = {"max_shards": 16}
    manager.save_metadata(old_metadata)

    dump_started = threading.Event()
    allow_dump = threading.Event()
    writer_errors = []
    original_dump = json.dump

    def blocking_dump(metadata, file_obj):
        dump_started.set()
        if not allow_dump.wait(timeout=5):
            raise TimeoutError("test did not release metadata writer")
        return original_dump(metadata, file_obj)

    monkeypatch.setattr("gpt_lab.data.sharder.json.dump", blocking_dump)

    def write_metadata():
        try:
            manager.save_metadata(new_metadata)
        except BaseException as exc:  # Propagate failures from the writer thread.
            writer_errors.append(exc)

    writer = threading.Thread(target=write_metadata)
    writer.start()
    assert dump_started.wait(timeout=5)

    try:
        # The old implementation had already truncated meta.json at this point,
        # so this read raised JSONDecodeError on distributed startup.
        assert manager.load_metadata() == old_metadata
    finally:
        allow_dump.set()
        writer.join(timeout=5)

    assert not writer.is_alive()
    assert writer_errors == []
    assert manager.load_metadata() == new_metadata
    assert list(tmp_path.glob(".meta.json.*.tmp")) == []


@pytest.mark.fast
def test_empty_legacy_metadata_is_regenerated(tmp_path):
    manager = _metadata_manager(tmp_path)
    (tmp_path / "meta.json").write_text("", encoding="utf-8")

    assert manager.load_metadata() == {}


@pytest.mark.fast
def test_training_workers_retrieve_disjoint_shards(tmp_path):
    dataset_name = "synthetic"
    dataset_path = tmp_path / dataset_name
    dataset_path.mkdir()

    world_size = 8
    num_train_shards = 16

    for shard_idx in range(num_train_shards):
        pq.write_table(
            pa.table({"text": [f"shard-{shard_idx}"]}),
            dataset_path / f"shard_{shard_idx:05d}.parquet",
        )

    (dataset_path / "meta.json").write_text(
        json.dumps({
            "name": dataset_name,
            "base_url": "",
            "column_name": "text",
            "max_shards": num_train_shards,
        }),
        encoding="utf-8",
    )

    def read_worker(rank):
        manager = ShardManager(
            name=dataset_name,
            cachedir=tmp_path,
            split="train",
            max_shards=num_train_shards,
            dist_info={"RANK": rank, "WORLD_SIZE": world_size},
        )
        expected_shard_ids = set(range(rank, num_train_shards, world_size))
        assigned_shard_ids = {
            int(path.stem.split("_")[1])
            for path in manager.shard_paths
        }

        # Check assignment before consuming the infinite iterator so a
        # regression cannot leave the test waiting for data forever.
        assert assigned_shard_ids == expected_shard_ids

        retrieved_texts = set()
        retrieved_shard_ids = set()
        iterator = manager.iterate(batch_size=1)
        try:
            for _ in range(len(manager.shard_paths)):
                texts, state = next(iterator)
                assert len(texts) == 1
                retrieved_texts.update(texts)
                retrieved_shard_ids.add(state.global_shard_idx)
        finally:
            iterator.close()

        return retrieved_texts, retrieved_shard_ids

    with ThreadPoolExecutor(max_workers=world_size) as executor:
        worker_results = list(executor.map(read_worker, range(world_size)))

    expected_texts = {
        f"shard-{shard_idx}"
        for shard_idx in range(num_train_shards)
    }
    all_retrieved_texts = set()

    for rank, (retrieved_texts, retrieved_shard_ids) in enumerate(worker_results):
        expected_shard_ids = set(range(rank, num_train_shards, world_size))
        assert retrieved_shard_ids == expected_shard_ids
        assert retrieved_texts == {
            f"shard-{shard_idx}"
            for shard_idx in expected_shard_ids
        }

        for other_rank in range(rank):
            other_texts, other_shard_ids = worker_results[other_rank]
            assert retrieved_texts.isdisjoint(other_texts)
            assert retrieved_shard_ids.isdisjoint(other_shard_ids)

        all_retrieved_texts.update(retrieved_texts)

    assert all_retrieved_texts == expected_texts

