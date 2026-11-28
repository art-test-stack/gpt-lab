"""Offline parity checks using real Parquet cursors and logical CPU ranks."""
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gpt_lab.data import build_dataloader
from gpt_lab.utils.schemas import DataLoaderState


class ScalarTokenizer:
    def get_bos_token_id(self):
        return 0

    def __call__(self, text, *, prepend_bos, threads):
        assert prepend_bos and threads == 2
        return [0, *map(int, text.split())]


class BatchTokenizer(ScalarTokenizer):
    supports_exact_batch_encoding = True

    def __init__(self):
        self.batch_lengths = []

    def encode_batch(self, texts, *, prepend_bos, num_threads):
        assert num_threads == 2
        self.batch_lengths.append(len(texts))
        return [self(text, prepend_bos=prepend_bos, threads=num_threads) for text in texts]


@pytest.fixture
def shards(tmp_path):
    path = tmp_path / "parity"
    path.mkdir()
    # Equal lengths exercise tie order, long documents exercise retained suffixes.
    lengths = [2, 2, 47, 1, 7, 3, 29, 5] * 18
    for shard in range(5):
        documents = [[10000 * shard + 100 * index + token + 1 for token in range(length)]
                     for index, length in enumerate(lengths)]
        pq.write_table(pa.table({"text": [" ".join(map(str, doc)) for doc in documents],
                                 "tokens": documents}),
                       path / f"shard_{shard:05d}.parquet", row_group_size=72)
    return tmp_path


def loader(shards, batch, *, rank=0, split="train", state=None, tokenizer=None, column="text", backend="python"):
    return build_dataloader(
        name="parity", cachedir=shards, max_shards=4, split=split,
        batch_size=2, seq_len=17, tokenizer=tokenizer, column=column,
        tokenizer_threads=2, tokenizer_batch_size=batch, resume_state=state, packing_backend=backend,
        comp_info=SimpleNamespace(rank=rank, world_size=2, device=torch.device("cpu")),
        data_seed=19,
    )


def equal(left, right):
    x, y, state = left
    other_x, other_y, other_state = right
    assert torch.equal(x, other_x)
    assert torch.equal(y, other_y)
    assert state == other_state


def test_batch_padding_adapter_uses_scalar_encoding(shards):
    class PaddingTokenizer(BatchTokenizer):
        def encode_batch(self, texts, *, prepend_bos, num_threads):
            documents = super().encode_batch(texts, prepend_bos=prepend_bos, num_threads=num_threads)
            width = max(map(len, documents))
            return [document + [99999] * (width - len(document)) for document in documents]

    scalar = loader(shards, 1, tokenizer=PaddingTokenizer())
    tokenizer = PaddingTokenizer()
    requested_batching = loader(shards, 64, tokenizer=tokenizer)
    for _ in range(3):
        equal(next(scalar), next(requested_batching))
    assert not tokenizer.batch_lengths


@pytest.mark.parametrize("rank", [0, 1])
def test_batches_and_resume_across_boundaries(shards, rank, native_extension):
    scalar = loader(shards, 1, rank=rank, tokenizer=BatchTokenizer())
    tokenizer = BatchTokenizer()
    batched = loader(shards, 64, rank=rank, tokenizer=tokenizer)
    rust = loader(shards, 64, rank=rank, tokenizer=BatchTokenizer(), backend="rust")
    resumed = []
    seen = set()
    for _ in range(350):
        reference = next(scalar)
        actual, accelerated = next(batched), next(rust)
        equal(reference, actual)
        equal(reference, accelerated)
        for restored in resumed:
            equal(reference, next(restored))
        state = reference[2]
        seen.add((state.epoch, state.global_shard_idx, state.row_group_idx))
        if not resumed and state.lookahead_refs and state.pending_document_ref:
            assert state.document_token_offset > 0
            assert state.offset_in_row_group < 64
            assert tokenizer.batch_lengths == [64]
            # Serialize each source backend's state, then resume on the other backend.
            for source, backend in ((actual, "rust"), (accelerated, "python")):
                saved = DataLoaderState.model_validate_json(source[2].model_dump_json())
                resumed.append(loader(shards, 64, rank=rank, state=saved,
                                      tokenizer=BatchTokenizer(), backend=backend))
    assert resumed
    assert len({entry[0] for entry in seen}) > 1
    assert len({entry[1] for entry in seen}) == 2
    assert len({entry[2] for entry in seen}) == 2
    assert max(tokenizer.batch_lengths) == 64


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("kind", ["batch", "scalar", "pretokenized"])
def test_finite_validation_and_flexible_inputs(shards, rank, kind, native_extension):
    tokenizer = {"batch": BatchTokenizer, "scalar": ScalarTokenizer}.get(kind)
    kwargs = dict(rank=rank, split="val", column="tokens" if kind == "pretokenized" else "text")
    left = loader(shards, 1, tokenizer=tokenizer() if tokenizer else None, **kwargs)()
    right = loader(shards, 64, tokenizer=tokenizer() if tokenizer else None, **kwargs)()
    rust = loader(shards, 64, tokenizer=tokenizer() if tokenizer else None, backend="rust", **kwargs)()
    count = 0
    while True:
        reference = next(left, None)
        actual = next(right, None)
        accelerated = next(rust, None)
        if reference is None:
            assert actual is None and accelerated is None
            break
        equal(reference, actual)
        equal(reference, accelerated)
        assert reference[0].shape == (2, 17)
        count += 1
    assert count > 0


def test_validation_drops_incomplete_final_batch(tmp_path, native_extension):
    path = tmp_path / "tail"
    path.mkdir()
    # No BOS and one document: 25 tokens fill one 2x9 batch, leaving 7 tokens.
    pq.write_table(pa.table({"tokens": [list(range(25))]}), path / "shard_00000.parquet")
    context = SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu"))
    for backend in ("python", "rust"):
        factory = build_dataloader(name="tail", cachedir=tmp_path, max_shards=0,
                                   split="val", tokenizer=None, column="tokens",
                                   batch_size=2, seq_len=8, tokenizer_batch_size=64,
                                   comp_info=context, packing_backend=backend)
        iterator = factory()
        x, y, _ = next(iterator)
        assert x.tolist() == [list(range(8)), list(range(9, 17))]
        assert y.tolist() == [list(range(1, 9)), list(range(10, 18))]
        assert next(iterator, None) is None


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_checkpoint_recovery_after_refill_error(tmp_path, backend, request):
    if backend == "rust":
        request.getfixturevalue("native_extension")
    path = tmp_path / "recovery"
    path.mkdir()
    pq.write_table(pa.table({"tokens": [list(range(6)), list(range(10, 20))]}),
                   path / "shard_00000.parquet")
    context = SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu"))

    def make_loader(state=None):
        return build_dataloader(
            name="recovery", cachedir=tmp_path, max_shards=1, split="train",
            tokenizer=None, column="tokens", batch_size=1, seq_len=3,
            comp_info=context, packing_backend=backend, resume_state=state,
        )

    uninterrupted = make_loader()
    failed = make_loader()
    reference = next(uninterrupted)
    committed = next(failed)
    equal(reference, committed)
    saved = DataLoaderState.model_validate_json(committed[2].model_dump_json())
    assert saved.pending_document_ref == (0, 0, 0)
    assert saved.document_token_offset == 4

    def broken_source():
        raise OSError("refill failed")
        yield  # Keep this a document iterator whose error escapes next().

    failed.iterator = broken_source()
    with pytest.raises(OSError, match="refill failed"):
        next(failed)
    del failed

    recovered = make_loader(saved)
    expected = next(uninterrupted)
    actual = next(recovered)
    assert actual[0].tolist() == [[4, 5, 10]]
    assert actual[1].tolist() == [[5, 10, 11]]
    equal(expected, actual)
