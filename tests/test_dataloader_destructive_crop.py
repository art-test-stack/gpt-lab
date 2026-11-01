from itertools import cycle
from types import SimpleNamespace

import torch

import gpt_lab.data.loader as loader_module
from gpt_lab.data.loader import DistDataLoader, PackingStats
from scripts.benchmark.dataloaders import (
    Counters,
    IdentityTokenizer,
    TorchPackedDataset,
    benchmark,
)


class IdentifiableDocuments:
    split = "train"

    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for document in cycle(self.documents):
            yield torch.tensor(document), None


def packed_tokens(inputs, targets):
    return [*inputs.flatten().tolist(), targets.flatten().tolist()[-1]]


def test_gpt_lab_fifo_retains_long_document_tail():
    stats = PackingStats()
    loader = DistDataLoader(
        IdentifiableDocuments(([1, 2, 3, 4, 5, 6, 7, 8], [20, 21, 22])),
        batch_size=1,
        seq_len=4,
        device="cpu",
        buffer_size=1,
        packing_stats=stats,
    )

    first_inputs, first_targets, _ = next(loader)
    first_inputs, first_targets = first_inputs.clone(), first_targets.clone()
    second_inputs, second_targets, _ = next(loader)

    assert packed_tokens(first_inputs, first_targets) == [1, 2, 3, 4, 5]
    assert packed_tokens(second_inputs, second_targets) == [5, 6, 7, 8, 20]
    assert stats.destructive_cropped_tokens == 0


def test_pytorch_stream_retains_long_document_tail():
    dataset = TorchPackedDataset(
        path=None,
        tokenizer=IdentityTokenizer(0, 100),
        batch_size=1,
        seq_len=4,
        tokenized=True,
        packing="stream",
        buffer_docs=1,
    )
    stream = dataset._stream(iter(([1, 2, 3, 4, 5, 6, 7, 8], [20, 21, 22])))

    first_inputs, first_targets, first_stats = next(stream)
    second_inputs, second_targets, second_stats = next(stream)

    assert packed_tokens(first_inputs, first_targets) == [1, 2, 3, 4, 5]
    assert packed_tokens(second_inputs, second_targets) == [5, 6, 7, 8, 20]
    assert first_stats["destructive_cropped_tokens"] == 0
    assert second_stats["destructive_cropped_tokens"] == 0


def test_nanochat_bestfit_counts_awkward_suffix_and_long_document(monkeypatch):
    document_batches = iter(
        (
            ([[0, 31, 32, 33], [0, 11, 12], [0, 21, 22, 23, 24, 25, 26, 27]], (0, 0, 1)),
            ([[0, 41, 42, 43, 44, 45, 46, 47], [0, 51, 52, 53, 54, 55, 56, 57, 58]], (0, 1, 1)),
        )
    )
    monkeypatch.setattr(loader_module, "_document_batches", lambda *args, **kwargs: document_batches)
    stats = PackingStats()
    loader = loader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        IdentityTokenizer(0, 100),
        B=1,
        T=4,
        split="train",
        device="cpu",
        buffer_size=3,
        packing_stats=stats,
    )

    inputs, targets, _ = next(loader)

    assert packed_tokens(inputs, targets) == [0, 31, 32, 33, 0]
    assert stats.destructive_cropped_tokens == 2

    long_batches = cycle(
        [([[0, 61, 62, 63, 64, 65, 66, 67], [0, 71, 72, 73, 74, 75, 76, 77, 78]], (0, 0, 1))]
    )
    monkeypatch.setattr(loader_module, "_document_batches", lambda *args, **kwargs: long_batches)
    long_stats = PackingStats()
    long_loader = loader_module.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        IdentityTokenizer(0, 100),
        B=1,
        T=4,
        split="train",
        device="cpu",
        buffer_size=2,
        packing_stats=long_stats,
    )

    long_inputs, long_targets, _ = next(long_loader)

    assert packed_tokens(long_inputs, long_targets) == [0, 61, 62, 63, 64]
    assert long_stats.destructive_cropped_tokens == 3


def test_pytorch_bestfit_counts_awkward_suffix_and_long_document():
    dataset = TorchPackedDataset(
        path=None,
        tokenizer=IdentityTokenizer(0, 100),
        batch_size=1,
        seq_len=4,
        tokenized=True,
        packing="bestfit",
        buffer_docs=3,
    )
    awkward = dataset._bestfit(
        cycle(([0, 31, 32, 33], [0, 11, 12], [0, 21, 22, 23, 24, 25, 26, 27]))
    )

    inputs, targets, stats = next(awkward)

    assert packed_tokens(inputs, targets) == [0, 31, 32, 33, 0]
    assert stats["destructive_cropped_tokens"] == 2

    long_dataset = TorchPackedDataset(
        path=None,
        tokenizer=IdentityTokenizer(0, 100),
        batch_size=1,
        seq_len=4,
        tokenized=True,
        packing="bestfit",
        buffer_docs=2,
    )
    long = long_dataset._bestfit(
        iter(([0, 61, 62, 63, 64, 65, 66, 67], [0, 71, 72, 73, 74, 75, 76, 77, 78]))
    )

    long_inputs, long_targets, long_stats = next(long)

    assert packed_tokens(long_inputs, long_targets) == [0, 61, 62, 63, 64]
    assert long_stats["destructive_cropped_tokens"] == 3


def test_benchmark_resets_destructive_crop_counter_after_warmup():
    def factory(counters: Counters):
        def batches():
            for discarded in (7, 3):
                counters.destructive_cropped_tokens += discarded
                counters.source_tokens_read += 5 + discarded
                counters.new_source_tokens_advanced += 5 + discarded
                counters.skipped_adjacent_transitions += 1 + discarded
                data = torch.tensor([[0, 1, 2, 3, 4]])
                yield data[:, :-1], data[:, 1:], None

        return batches()

    args = SimpleNamespace(
        warmup_batches=1,
        batches=1,
        batch_size=1,
        seq_len=4,
        device=torch.device("cpu"),
    )

    result = benchmark(
        "nanochat",
        factory,
        "pretokenized",
        "identity",
        0,
        None,
        "none",
        "none",
        None,
        None,
        args,
        0.0,
    )

    assert result.destructively_cropped_tokens == 3
