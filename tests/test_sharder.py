import json
import threading

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
