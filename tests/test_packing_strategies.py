import inspect

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gpt_lab.data.loader import DistDataLoader, build_dataloader
from gpt_lab.utils.schemas import DataLoaderState
from gpt_lab.utils.distributed import ComputeContext


BOS = 99


class ResumableDocuments:
    split = "train"

    def __init__(self, documents, start_state=None):
        self.documents = documents
        self.start_state = start_state

    def __iter__(self):
        start = self.start_state.offset_in_row_group if self.start_state else 0
        for index, document in enumerate(self.documents[start:], start):
            yield torch.tensor(document), DataLoaderState(offset_in_row_group=index + 1)


class IdentityTokenizer:
    def get_bos_token_id(self):
        return BOS

    def __call__(self, tokens, **_):
        return tokens


def make_loader(documents, *, strategy="bos_aligned", batch_size=2, seq_len=4, state=None, buffer_docs=3):
    dataset = ResumableDocuments(documents, state)
    return DistDataLoader(
        dataset,
        batch_size=batch_size,
        seq_len=seq_len,
        device="cpu",
        packing_strategy=strategy,
        bos_token_id=BOS,
        resume_state=state,
        best_fit_buffer_docs=buffer_docs,
    )


def clone_batch(batch):
    inputs, targets, state = batch
    return inputs.clone(), targets.clone(), state


def source_rows(batch):
    inputs, targets, _ = batch
    rows = []
    for row_inputs, row_targets in zip(inputs, targets):
        rows.append([int(row_inputs[0]), *row_targets.tolist()])
    return rows


def drain(loader):
    batches = []
    while True:
        try:
            batches.append(clone_batch(next(loader)))
        except StopIteration:
            return batches


def test_bos_aligned_rows_start_with_bos_and_batch_size_gt_one():
    loader = make_loader([[BOS, *range(1, 18)], [BOS, 20], [BOS, 21, 22]], batch_size=3)
    for inputs, targets, _ in drain(loader):
        assert inputs.shape == targets.shape == (3, 4)
        assert torch.all(inputs[:, 0] == BOS)


def test_bos_aligned_emits_each_source_content_once_and_counts_synthetic_bos():
    documents = [
        [BOS, 1, 2],                    # shorter than row capacity
        [BOS, 3, 4, 5, 6],             # equal to row capacity
        [BOS, 7, 8, 9, 10, 11, 12, 13, 14],  # longer than row capacity
    ]
    loader = make_loader(documents)
    rows = [row for batch in drain(loader) for row in source_rows(batch)]
    observed_content = [token for row in rows for token in row if token != BOS]
    expected_content = [token for document in documents for token in document[1:]]

    assert observed_content == expected_content
    assert loader.packing_stats.source_tokens_read == sum(map(len, documents))
    assert loader.packing_stats.source_tokens_emitted == sum(map(len, documents))
    assert loader.packing_stats.source_bos_tokens_emitted == len(documents)
    assert loader.packing_stats.synthetic_bos_tokens_inserted > 0
    assert loader.packing_stats.destructive_cropped_tokens == 0


def test_bos_aligned_packs_multiple_short_documents_in_one_row():
    loader = make_loader([[BOS, 1], [BOS, 2], [BOS, 3]], batch_size=1, seq_len=5)
    batch = clone_batch(next(loader))
    assert source_rows(batch)[0] == [BOS, 1, BOS, 2, BOS, 3]


def test_bos_aligned_long_document_spans_rows_and_batches_without_loss():
    document = [BOS, *range(1, 25)]
    loader = make_loader([document], batch_size=2, seq_len=4)
    batches = drain(loader)
    rows = [row for batch in batches for row in source_rows(batch)]

    assert len(batches) > 1
    assert [token for row in rows for token in row if token != BOS] == document[1:]
    assert loader.packing_stats.synthetic_bos_tokens_inserted == len(rows) - 1


def test_validation_iteration_is_deterministic():
    documents = [[BOS, 1, 2], [BOS, 3], [BOS, *range(4, 14)]]
    first = drain(make_loader(documents))
    second = drain(make_loader(documents))
    assert len(first) == len(second)
    for (x1, y1, _), (x2, y2, _) in zip(first, second):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


@pytest.mark.parametrize("strategy", ["stream", "bos_aligned", "bos_bestfit_crop"])
def test_checkpoint_resume_continues_at_exact_next_batch(strategy):
    documents = [[BOS, *range(i, i + size)] for i, size in zip(range(1, 400, 20), [2, 4, 19, 1] * 5)]
    loader = make_loader(documents, strategy=strategy, batch_size=2, seq_len=4)
    _, _, state = clone_batch(next(loader))
    restored_state = DataLoaderState.model_validate_json(state.model_dump_json())
    resumed = make_loader(documents, strategy=strategy, batch_size=2, seq_len=4, state=restored_state)
    for _ in range(3):
        expected_inputs, expected_targets, expected_state = clone_batch(next(loader))
        actual_inputs, actual_targets, actual_state = clone_batch(next(resumed))
        assert torch.equal(actual_inputs, expected_inputs)
        assert torch.equal(actual_targets, expected_targets)
        assert actual_state.model_dump() == expected_state.model_dump()


def test_default_strategy_remains_stream():
    assert inspect.signature(build_dataloader).parameters["packing_strategy"].default == "stream"
    loader = make_loader([[BOS, *range(20)]], strategy="stream")
    assert loader.packing_strategy == "stream"


@pytest.mark.parametrize("strategy", ["stream", "bos_aligned", "bos_bestfit_crop"])
def test_incomplete_batches_are_not_emitted(strategy):
    loader = make_loader([[BOS, 1]], strategy=strategy, batch_size=1, seq_len=4)
    with pytest.raises(StopIteration):
        next(loader)


@pytest.mark.parametrize("strategy", ["stream", "bos_aligned", "bos_bestfit_crop"])
def test_build_dataloader_validation_and_shard_resume(tmp_path, strategy):
    dataset = tmp_path / "packing"
    dataset.mkdir()
    documents = [[BOS, *range(i, i + 7)] for i in range(0, 42, 7)]
    table = pa.table({"text": documents})
    pq.write_table(table, dataset / "shard_00000.parquet", row_group_size=3)
    pq.write_table(table, dataset / "shard_00001.parquet", row_group_size=3)
    common = dict(
        name="packing",
        tokenizer=IdentityTokenizer(),
        cachedir=tmp_path,
        batch_size=2,
        seq_len=4,
        max_shards=1,
        packing_strategy=strategy,
        best_fit_buffer_docs=3,
        comp_info=ComputeContext(device=torch.device("cpu")),
    )

    loader = build_dataloader(split="train", **common)
    _, _, state = clone_batch(next(loader))
    expected = clone_batch(next(loader))[:2]
    state = DataLoaderState.model_validate_json(state.model_dump_json())
    resumed = build_dataloader(split="train", resume_state=state, **common)
    actual = clone_batch(next(resumed))[:2]
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))

    validation = build_dataloader(split="val", **common)
    first, second = clone_batch(next(validation()))[:2], clone_batch(next(validation()))[:2]
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_bestfit_real_bos_shifted_targets_and_suffix_accounting():
    docs = [[BOS, 31, 32, 33], [BOS, 11, 12], [BOS, 21, 22, 23, 24, 25, 26, 27]]
    loader = make_loader(docs, strategy="bos_bestfit_crop", batch_size=1)
    x, y, state = clone_batch(next(loader))
    assert source_rows((x, y, state)) == [[BOS, 31, 32, 33, BOS]]
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert loader.packing_stats.destructive_cropped_tokens == 2
    assert loader.packing_stats.synthetic_bos_tokens_inserted == 0
    assert state.packing_doc_buffer == [docs[2]]
    x, y, state = clone_batch(next(loader))
    assert source_rows((x, y, state)) == [[BOS, 21, 22, 23, 24]]
    assert loader.packing_stats.destructive_cropped_tokens == 5
    assert loader.packing_stats.source_tokens_emitted + loader.packing_stats.destructive_cropped_tokens == sum(map(len, docs))


@pytest.mark.parametrize("strategy", ["bos_aligned", "bos_bestfit_crop"])
def test_bos_policies_reject_missing_real_document_bos(strategy):
    with pytest.raises(ValueError, match="real BOS"):
        next(make_loader([[12, 13, 14, 15, 16]], strategy=strategy, batch_size=1))


@pytest.mark.parametrize("override", [dict(seq_len=3), dict(batch_size=1), dict(buffer_docs=2), dict(strategy="stream")])
def test_resume_rejects_incompatible_packing_state(override):
    docs = [[BOS, *range(i, i + 8)] for i in range(0, 100, 10)]
    _, _, state = next(make_loader(docs, strategy="bos_bestfit_crop"))
    kwargs = dict(strategy="bos_bestfit_crop", state=state) | override
    with pytest.raises(ValueError, match="[Ii]ncompatible|Cannot resume"):
        make_loader(docs, **kwargs)


def test_output_storage_is_reused_after_consumption():
    loader = make_loader([[BOS, *range(100)]], strategy="stream")
    x, y, _ = next(loader)
    first = x.clone()
    x2, y2, _ = next(loader)
    assert x.data_ptr() == x2.data_ptr()
    assert y.data_ptr() == y2.data_ptr()
    assert not torch.equal(first, x2)



def test_legacy_state_warns_and_versionless_draft_packing_is_rejected():
    docs = [[BOS, *range(20)]]
    with pytest.warns(UserWarning, match="exact token continuation is unavailable"):
        make_loader(docs, strategy="stream", state=DataLoaderState())
    with pytest.raises(ValueError, match="versioned state"):
        make_loader(docs, strategy="stream", state=DataLoaderState(packing_strategy="stream"))
