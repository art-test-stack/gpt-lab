import inspect

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from gpt_lab.data.loader import DistDataLoader, build_dataloader
from gpt_lab.model.loss import build_loss
from gpt_lab.utils.schemas import DataLoaderState


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


def make_loader(documents, *, strategy="bos_aligned", batch_size=2, seq_len=4, state=None):
    dataset = ResumableDocuments(documents, state)
    return DistDataLoader(
        dataset,
        batch_size=batch_size,
        seq_len=seq_len,
        device="cpu",
        packing_strategy=strategy,
        bos_token_id=BOS,
        resume_state=state,
    )


def clone_batch(batch):
    inputs, targets, state = batch
    return inputs.clone(), targets.clone(), state


def source_rows(batch):
    inputs, targets, _ = batch
    rows = []
    for row_inputs, row_targets in zip(inputs, targets):
        valid_targets = row_targets[row_targets != DistDataLoader.IGNORE_INDEX]
        if len(valid_targets):
            rows.append([int(row_inputs[0]), *valid_targets.tolist()])
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
        [BOS, 7, 8, 9, 10, 11, 12, 13],  # longer than row capacity
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
    document = [BOS, *range(1, 31)]
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


@pytest.mark.parametrize("strategy", ["stream", "bos_aligned"])
def test_checkpoint_resume_continues_at_exact_next_batch(strategy):
    documents = [[BOS, *range(1, 10)], [BOS, *range(10, 30)]]
    loader = make_loader(documents, strategy=strategy, batch_size=2, seq_len=4)
    _, _, state = clone_batch(next(loader))
    expected_inputs, expected_targets, _ = clone_batch(next(loader))

    resumed = make_loader(documents, strategy=strategy, batch_size=2, seq_len=4, state=state)
    actual_inputs, actual_targets, _ = clone_batch(next(resumed))
    assert torch.equal(actual_inputs, expected_inputs)
    assert torch.equal(actual_targets, expected_targets)


def test_default_strategy_remains_stream():
    assert inspect.signature(build_dataloader).parameters["packing_strategy"].default == "stream"
    loader = make_loader([[BOS, *range(20)]], strategy="stream")
    assert loader.packing_strategy == "stream"


def test_padding_targets_are_ignored_by_model_loss():
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]]])
    labels = torch.tensor([[0, 1, DistDataLoader.IGNORE_INDEX]])
    loss = build_loss()(logits, labels)
    expected = F.cross_entropy(logits[:, :2].reshape(-1, 2), labels[:, :2].reshape(-1))
    assert torch.allclose(loss, expected)


def test_build_dataloader_validation_and_shard_resume(tmp_path):
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
        packing_strategy="bos_aligned",
        dist_info={"RANK": 0, "WORLD_SIZE": 1, "DEVICE": "cpu"},
    )

    loader = build_dataloader(split="train", **common)
    _, _, state = clone_batch(next(loader))
    expected = clone_batch(next(loader))[:2]
    resumed = build_dataloader(split="train", resume_state=state, **common)
    actual = clone_batch(next(resumed))[:2]
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))

    validation = build_dataloader(split="val", **common)
    first, second = clone_batch(next(validation()))[:2], clone_batch(next(validation()))[:2]
    assert all(torch.equal(left, right) for left, right in zip(first, second))
