from argparse import Namespace
from pathlib import Path

import pytest
import torch

from gpt_lab.data.loader import (
    DistDataLoader,
    PackingStats,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from gpt_lab.utils.schemas import DataLoaderState
from scripts.benchmark.dataloaders import (
    Counters,
    IdentityTokenizer,
    LoaderSpec,
    TorchPackedDataset,
    aggregate,
    benchmark,
    benchmark_trial,
    check_accounting,
    write_split,
)


class DeterministicDataset:
    def __init__(self, documents, split="train"):
        self.documents = documents
        self.split = split

    def __iter__(self):
        state = DataLoaderState()
        for document in self.documents:
            yield torch.tensor(document, dtype=torch.long), state


def loader_batches(kind, documents, batch_size=2, seq_len=3):
    if kind == "gpt_lab":
        loader = DistDataLoader(
            DeterministicDataset(documents),
            batch_size=batch_size,
            seq_len=seq_len,
            device="cpu",
            buffer_size=batch_size * seq_len,
        )
        return loader

    dataset = TorchPackedDataset(
        Path("."), None, batch_size, seq_len, True, "stream", 1
    )
    return dataset._stream(iter(documents))


def take_batches(loader, count):
    batches = []
    for _ in range(count):
        inputs, targets = next(loader)[:2]
        batches.append((inputs.clone(), targets.clone()))
    return batches


def flatten_batches(batches):
    windows = [torch.cat((inputs.flatten(), targets.flatten()[-1:])) for inputs, targets in batches]
    return torch.cat((windows[0], *[window[1:] for window in windows[1:]]))


@pytest.mark.parametrize("kind", ["gpt_lab", "pytorch"])
def test_flat_stream_preserves_every_transition_across_rows_and_batches(kind):
    loader = loader_batches(kind, [list(range(30))])
    batches = take_batches(loader, 3)

    for inputs, targets in batches:
        assert inputs.shape == targets.shape == (2, 3)
        flat = torch.cat((inputs.flatten(), targets.flatten()[-1:]))
        assert torch.equal(targets.flatten(), flat[1:])
        assert targets[0, -1] == inputs[1, 0]

    for (_, targets), (inputs, _) in zip(batches, batches[1:]):
        assert inputs.flatten()[0] == targets.flatten()[-1]

    observed = flatten_batches(batches)
    assert torch.equal(observed, torch.arange(19))


@pytest.mark.parametrize("kind", ["gpt_lab", "pytorch"])
def test_flat_stream_keeps_document_tails_and_bos_tokens(kind):
    bos = 99
    documents = [
        [bos, *range(0, 10)],
        [bos, *range(10, 20)],
        [bos, *range(20, 30)],
    ]
    expected = torch.tensor([token for document in documents for token in document])
    loader = loader_batches(kind, documents)
    batches = take_batches(loader, 5)
    observed = flatten_batches(batches)

    assert torch.equal(observed, expected[: len(observed)])
    assert observed.tolist().count(bos) == expected[: len(observed)].tolist().count(bos)


@pytest.mark.parametrize("kind", ["gpt_lab", "pytorch"])
def test_fifo_accounting_has_no_crop_or_skipped_transitions(kind):
    documents = [list(range(30))]
    if kind == "gpt_lab":
        stats = PackingStats()
        loader = DistDataLoader(
            DeterministicDataset(documents), 2, 3, device="cpu", buffer_size=6,
            packing_stats=stats,
        )
        buffered_start = stats.buffered_source_tokens
        batches = take_batches(loader, 2)
        buffered_delta = stats.buffered_source_tokens - buffered_start
        values = vars(stats)
    else:
        dataset = TorchPackedDataset(Path("."), None, 2, 3, True, "stream", 1)
        loader = dataset._stream(iter(documents))
        batch_stats = []
        batches = []
        for _ in range(2):
            inputs, targets, current = next(loader)
            batches.append((inputs, targets))
            batch_stats.append(current)
        values = {
            key: sum(row[key] for row in batch_stats)
            for key in (
                "source_tokens_read", "new_source_tokens_advanced",
                "destructive_cropped_tokens", "skipped_adjacent_transitions",
                "synthetic_bos_tokens_inserted",
            )
        }
        buffered_delta = sum(row["buffered_source_tokens_delta"] for row in batch_stats)

    targets = sum(target.numel() for _, target in batches)
    assert values["destructive_cropped_tokens"] == 0
    assert values["skipped_adjacent_transitions"] == 0
    assert targets == 2 * 2 * 3
    check_accounting(
        source_tokens_read=values["source_tokens_read"],
        new_source_tokens_advanced=values["new_source_tokens_advanced"],
        buffered_source_tokens_delta=buffered_delta,
        target_positions_emitted=targets,
        skipped_adjacent_transitions=values["skipped_adjacent_transitions"],
        synthetic_bos_tokens_inserted=values["synthetic_bos_tokens_inserted"],
    )


def test_bos_bestfit_reports_exact_discarded_suffix(tmp_path):
    bos = 99
    write_split(tmp_path, [[bos, 1, 2, 3, 4, 5]], row_group_size=1)
    stats = PackingStats(debug_discarded_suffixes=[])
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        IdentityTokenizer(bos, 128), B=1, T=3, split="train", device="cpu",
        tokenizer_batch_size=1, buffer_size=1, base_path=tmp_path,
        packing_stats=stats,
    )

    inputs, targets, _ = next(loader)

    assert inputs.tolist() == [[bos, 1, 2]]
    assert targets.tolist() == [[1, 2, 3]]
    assert stats.destructive_cropped_tokens == 2
    assert stats.debug_discarded_suffixes == [(4, 5)]
    assert stats.skipped_adjacent_transitions == 3  # two crop skips + one BOS row boundary
    assert stats.intentional_bos_boundaries == 1


def test_synthetic_bos_is_not_counted_as_source():
    # Four source tokens advance; one real adjacency is segmented and replaced
    # by one synthetic-BOS target. The synthetic token never enters source read.
    check_accounting(
        source_tokens_read=4,
        new_source_tokens_advanced=4,
        buffered_source_tokens_delta=0,
        target_positions_emitted=4,
        skipped_adjacent_transitions=1,
        synthetic_bos_tokens_inserted=1,
    )


def test_benchmark_excludes_warmup_counters():
    def factory(counters: Counters):
        def batches():
            while True:
                counters.source_tokens_read += 4
                counters.new_source_tokens_advanced += 4
                tokens = torch.tensor([[99, 1, 2, 3]])
                yield tokens, tokens, {}
        return batches()

    args = Namespace(
        warmup_batches=1, batches=2, device=torch.device("cpu"),
        batch_size=1, seq_len=4, torch_packing="stream",
    )
    result = benchmark(
        "fake", factory, "pretokenized", "identity", 99, None,
        "none", "none", None, None, args, 0.0,
    )

    assert result.source_tokens_read == 8
    assert result.new_source_tokens_advanced == 8
    assert result.target_positions_emitted == 8


def test_benchmark_trial_reports_memory_pressure():
    def factory():
        def batches():
            while True:
                tokens = torch.tensor([[99, 1, 2, 3]])
                yield tokens, tokens, {
                    "source_tokens_read": 4,
                    "new_source_tokens_advanced": 4,
                    "destructive_cropped_tokens": 0,
                    "buffered_source_tokens_delta": 0,
                    "skipped_adjacent_transitions": 0,
                    "synthetic_bos_tokens_inserted": 0,
                    "intentional_bos_boundaries": 0,
                    "actual_buffered_tokens": 0,
                    "actual_buffered_documents": 0,
                }
        return batches()

    spec = LoaderSpec("fake", "fake", "flat_stream", "stream_packing", "test", factory)
    args = Namespace(
        warmup_batches=1, batches=2, device=torch.device("cpu"),
        batch_size=1, seq_len=4, trials=1, best_fit_buffer_docs=1,
        correctness_batches=1,
    )

    trial = benchmark_trial(spec, args)
    _, _, _, memory = trial
    result = aggregate(spec, [trial], 0.5, "pretokenized", "identity", 0.0, args)

    assert memory["host_rss_peak_mib"] > 0
    assert memory["host_rss_peak_delta_mib"] >= 0
    assert memory["accelerator_peak_allocated_mib"] is None
    assert memory["accelerator_peak_delta_mib"] is None
    assert result.host_rss_peak_mib == memory["host_rss_peak_mib"]
    assert result.accelerator_peak_allocated_mib is None
