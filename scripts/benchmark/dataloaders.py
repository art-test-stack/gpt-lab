"""Compare GPT-Lab packing strategies on deterministic tokenized documents."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gpt_lab.data.loader import (
    DistDataLoader,
    PackingStats,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from gpt_lab.tokenizer import Tokenizer, TokenizerConfig
from gpt_lab.utils.default import DATA_DIR


STREAM = "flat_stream"
BEST_FIT = "bos_aligned_best_fit"
STREAM_GROUP = "stream_packing"
BEST_FIT_GROUP = "bos_aligned_best_fit"


@dataclass
class Counters(PackingStats):
    search_ns: int = 0
    searches: int = 0


@dataclass
class Result:
    loader: str
    throughput_tokens_s: float
    mean_latency_ms: float
    std_latency_ms: float
    source_tokens_read: int
    new_source_tokens_advanced: int
    target_positions_emitted: int
    active_target_positions: int
    destructively_cropped_tokens: int
    skipped_adjacent_transitions: int
    synthetic_bos_tokens_inserted: int
    intentional_bos_boundaries: int
    buffered_source_tokens_delta: int
    padding_tokens: int
    source_token_utilization: float
    compute_utilization: float
    target_supervision_utilization: float
    source_transition_coverage: float
    destructive_crop_rate: float
    bos_row_alignment: float

    @property
    def destructive_cropped_tokens(self):
        return self.destructively_cropped_tokens


class IdentityTokenizer:
    def __init__(self, bos_id: int, vocab_size: int):
        self.bos_id, self.vocab_size = bos_id, vocab_size

    def get_bos_token_id(self):
        return self.bos_id

    def __call__(self, tokens, **_):
        return list(tokens)

    def encode(self, rows, **_):
        return [list(map(int, row)) for row in rows]


def write_split(path: Path, documents: Sequence[Any], row_group_size: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": list(documents)})
    pq.write_table(table, path / "shard_00000.parquet", row_group_size=row_group_size)
    pq.write_table(table.slice(0, min(len(table), row_group_size)), path / "shard_00001.parquet", row_group_size=row_group_size)


def check_accounting(*, source_tokens_read, new_source_tokens_advanced,
                     buffered_source_tokens_delta, target_positions_emitted,
                     skipped_adjacent_transitions, synthetic_bos_tokens_inserted,
                     active_target_positions=None):
    """Check source conservation independently from next-token supervision."""
    assert source_tokens_read == new_source_tokens_advanced + buffered_source_tokens_delta
    represented = new_source_tokens_advanced - skipped_adjacent_transitions
    active = target_positions_emitted if active_target_positions is None else active_target_positions
    assert active == represented + synthetic_bos_tokens_inserted
    assert active <= target_positions_emitted


class TorchPackedDataset:
    """Tiny reference implementation used by the comparison tests."""

    def __init__(self, path, tokenizer, batch_size, seq_len, tokenized, packing, buffer_docs):
        self.path, self.tokenizer = path, tokenizer
        self.batch_size, self.seq_len = batch_size, seq_len
        self.packing, self.buffer_docs = packing, buffer_docs

    def _stream(self, documents: Iterable[Sequence[int]]):
        documents, pending, carry = iter(documents), [], []
        while True:
            needed = self.batch_size * self.seq_len + 1
            read = 0
            buffered_before = len(pending) + len(carry)
            while len(carry) + len(pending) < needed:
                document = list(next(documents))
                pending.extend(document)
                read += len(document)
            take = needed - len(carry)
            window = carry + pending[:take]
            del pending[:take]
            carry = window[-1:]
            data = torch.tensor(window)
            yield (
                data[:-1].view(self.batch_size, self.seq_len),
                data[1:].view(self.batch_size, self.seq_len),
                {
                    "source_tokens_read": read,
                    "new_source_tokens_advanced": needed - 1,
                    "destructive_cropped_tokens": 0,
                    "skipped_adjacent_transitions": 0,
                    "synthetic_bos_tokens_inserted": 0,
                    "intentional_bos_boundaries": 0,
                    "buffered_source_tokens_delta": len(pending) + 1 - buffered_before,
                    "actual_buffered_tokens": len(pending) + 1,
                    "actual_buffered_documents": None,
                },
            )

    def _bestfit(self, documents: Iterable[Sequence[int]]):
        documents, buffer = iter(documents), []
        capacity = self.seq_len + 1
        while True:
            rows = torch.empty((self.batch_size, capacity), dtype=torch.long)
            read = advanced = cropped = 0
            for row_index in range(self.batch_size):
                pos = 0
                while pos < capacity:
                    while len(buffer) < self.buffer_docs:
                        document = list(next(documents))
                        buffer.append(document)
                        read += len(document)
                    remaining = capacity - pos
                    index = max(
                        (i for i, doc in enumerate(buffer) if len(doc) <= remaining),
                        key=lambda i: len(buffer[i]),
                        default=-1,
                    )
                    if index < 0:
                        index = min(range(len(buffer)), key=lambda i: len(buffer[i]))
                    document = buffer.pop(index)
                    take = min(len(document), remaining)
                    rows[row_index, pos:pos + take] = torch.tensor(document[:take])
                    pos += take
                    advanced += len(document)
                    cropped += len(document) - take
            yield rows[:, :-1], rows[:, 1:], {
                "source_tokens_read": read,
                "new_source_tokens_advanced": advanced,
                "destructive_cropped_tokens": cropped,
                "skipped_adjacent_transitions": self.batch_size + cropped,
                "synthetic_bos_tokens_inserted": 0,
                "intentional_bos_boundaries": self.batch_size,
                "buffered_source_tokens_delta": read - advanced,
                "actual_buffered_tokens": sum(map(len, buffer)),
                "actual_buffered_documents": len(buffer),
            }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(loader_name: str, factory: Callable[[Counters], Iterator],
              tokenization: str, tokenizer_name: str, bos_id: int,
              eos_id, workload: str, model_target: str, model, torch_workers,
              args, nano_search_us: float) -> Result:
    del tokenization, tokenizer_name, eos_id, workload, model_target, torch_workers, nano_search_us
    counters, latencies = Counters(), []
    loader = factory(counters)

    def accumulate(batch_stats):
        if not isinstance(batch_stats, dict):
            return
        for key, value in batch_stats.items():
            if key == "buffered_source_tokens_delta":
                counters.buffered_source_tokens += value
            elif hasattr(counters, key):
                setattr(counters, key, getattr(counters, key) + value)

    for _ in range(args.warmup_batches):
        inputs, _, batch_stats = next(loader)
        accumulate(batch_stats)
        if model is not None:
            model(inputs)
    buffered = counters.buffered_source_tokens
    counters.reset()
    counters.buffered_source_tokens = buffered

    bos_rows = total_rows = active_targets = 0
    for _ in range(args.batches):
        _sync(args.device)
        started = time.perf_counter()
        inputs, labels, batch_stats = next(loader)
        accumulate(batch_stats)
        if model is not None:
            model(inputs)
        _sync(args.device)
        latencies.append(time.perf_counter() - started)
        bos_rows += int((inputs[:, 0] == bos_id).sum())
        total_rows += inputs.shape[0]
        active_targets += int((labels != DistDataLoader.IGNORE_INDEX).sum())

    target_positions = args.batches * args.batch_size * args.seq_len
    buffered_delta = counters.buffered_source_tokens - buffered
    check_accounting(
        source_tokens_read=counters.source_tokens_read,
        new_source_tokens_advanced=counters.new_source_tokens_advanced,
        buffered_source_tokens_delta=buffered_delta,
        target_positions_emitted=target_positions,
        active_target_positions=active_targets,
        skipped_adjacent_transitions=counters.skipped_adjacent_transitions,
        synthetic_bos_tokens_inserted=counters.synthetic_bos_tokens_inserted,
    )
    represented = counters.new_source_tokens_advanced - counters.skipped_adjacent_transitions
    advanced = max(counters.new_source_tokens_advanced, 1)
    return Result(
        loader_name, target_positions / sum(latencies), statistics.mean(latencies) * 1000,
        statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0.0,
        counters.source_tokens_read, counters.new_source_tokens_advanced, target_positions,
        active_targets, counters.destructive_cropped_tokens,
        counters.skipped_adjacent_transitions, counters.synthetic_bos_tokens_inserted,
        counters.intentional_bos_boundaries, buffered_delta, counters.padding_tokens,
        (counters.new_source_tokens_advanced - counters.destructive_cropped_tokens) / advanced,
        active_targets / max(target_positions, 1),
        represented / max(target_positions, 1), represented / advanced,
        counters.destructive_cropped_tokens / advanced,
        bos_rows / max(total_rows, 1),
    )


class SyntheticDocuments:
    split, start_state = "train", None

    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for document in cycle(self.documents):
            yield torch.tensor(document), None


@dataclass
class ComparisonResult:
    comparison_group: str
    implementation: str
    implementation_id: str
    packing_policy: str
    provenance: str
    tokenization: str
    tokenizer: str
    device: str
    transfer_policy: str
    batch_size: int
    sequence_length: int
    trials: int
    batches_per_trial: int
    total_measured_batches: int
    throughput_median_tokens_s: float
    throughput_p50_tokens_s: float
    throughput_p95_tokens_s: float
    throughput_mean_tokens_s: float
    throughput_std_tokens_s: float
    latency_median_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    destructive_crop_policy: str
    destructively_cropped_tokens: int
    destructive_crop_rate: float
    bos_row_alignment: float
    buffer_budget_tokens: int | None
    buffer_budget_documents: int | None
    actual_buffered_tokens_mean: float | None
    actual_buffered_tokens_min: int | None
    actual_buffered_tokens_max: int | None
    actual_buffered_documents_mean: float | None
    source_tokens_read: int
    new_source_tokens_advanced: int
    target_positions_emitted: int
    active_target_positions: int
    skipped_adjacent_transitions: int
    synthetic_bos_tokens_inserted: int
    intentional_bos_boundaries: int
    buffered_source_tokens_delta: int
    source_token_utilization: float
    compute_utilization: float
    target_supervision_utilization: float
    source_transition_coverage: float
    correctness_status: str
    correctness_batches: int
    pretokenization_seconds: float

    @property
    def destructive_cropped_tokens(self):
        return self.destructively_cropped_tokens


@dataclass
class LoaderSpec:
    implementation: str
    implementation_id: str
    packing_policy: str
    comparison_group: str
    provenance: str
    factory: Callable[[], Iterator]


class BenchmarkTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self):
        return int(self.tokenizer.vocab_size)

    def get_bos_token_id(self):
        return int(self.tokenizer.get_bos_token_id())

    def encode_one(self, text):
        return list(self.tokenizer.encode(text, prepend_bos=True))

    def encode(self, texts, prepend=None, num_threads=1, **_):
        try:
            rows = self.tokenizer.encode(list(texts), num_threads=num_threads)
        except TypeError:
            rows = [self.tokenizer.encode(text) for text in texts]
        return [([prepend] if prepend is not None else []) + list(row) for row in rows]


class CorpusDocuments:
    split, start_state = "train", None

    def __init__(self, raw, tokens, tokenizer, on_the_fly):
        self.raw, self.tokens = raw, tokens
        self.tokenizer, self.on_the_fly = tokenizer, on_the_fly

    def __iter__(self):
        for index in cycle(range(len(self.raw))):
            row = self.tokenizer.encode_one(self.raw[index]) if self.on_the_fly else self.tokens[index]
            yield torch.tensor(row, dtype=torch.long), None


class TransferIterator:
    def __init__(self, iterator, device):
        self.iterator, self.device = iter(iterator), torch.device(device)
        self.use_cuda = torch.device(device).type == "cuda"

    def __iter__(self):
        return self

    def __next__(self):
        inputs, targets, stats = next(self.iterator)
        if self.use_cuda:
            inputs, targets = inputs.pin_memory(), targets.pin_memory()
        return (
            inputs.to(self.device, non_blocking=self.use_cuda),
            targets.to(self.device, non_blocking=self.use_cuda),
            stats,
        )


class StatsAdapter:
    """Expose per-batch deltas from production aggregate PackingStats."""

    FIELDS = (
        "source_tokens_read", "new_source_tokens_advanced",
        "destructive_cropped_tokens", "skipped_adjacent_transitions",
        "synthetic_bos_tokens_inserted", "intentional_bos_boundaries",
    )

    def __init__(self, iterator, stats):
        self.iterator, self.stats = iter(iterator), stats

    def __iter__(self):
        return self

    def __next__(self):
        before = {field: getattr(self.stats, field) for field in self.FIELDS}
        buffered_before = self.stats.buffered_source_tokens
        inputs, targets, _ = next(self.iterator)
        values = {field: getattr(self.stats, field) - before[field] for field in self.FIELDS}
        values["buffered_source_tokens_delta"] = self.stats.buffered_source_tokens - buffered_before
        values["actual_buffered_tokens"] = self.stats.buffered_source_tokens
        values["actual_buffered_documents"] = None
        return inputs, targets, values


def nanochat_flat_stream(documents, batch_size, seq_len, device):
    documents, token_buffer = iter(documents), []
    device, use_cuda = torch.device(device), torch.device(device).type == "cuda"
    advance = batch_size * seq_len
    while True:
        buffered_before, source_read = len(token_buffer), 0
        while len(token_buffer) < advance + 1:
            document = list(next(documents))
            token_buffer.extend(document)
            source_read += len(document)
        values = token_buffer[:advance + 1]
        token_buffer = token_buffer[advance:]
        scratch = torch.tensor(values, dtype=torch.long, pin_memory=use_cuda)
        stats = {
            "source_tokens_read": source_read,
            "new_source_tokens_advanced": advance,
            "destructive_cropped_tokens": 0,
            "skipped_adjacent_transitions": 0,
            "synthetic_bos_tokens_inserted": 0,
            "intentional_bos_boundaries": 0,
            "buffered_source_tokens_delta": len(token_buffer) - buffered_before,
            "actual_buffered_tokens": len(token_buffer),
            "actual_buffered_documents": None,
        }
        yield (
            scratch[:-1].view(batch_size, seq_len).to(device, non_blocking=use_cuda),
            scratch[1:].view(batch_size, seq_len).to(device, non_blocking=use_cuda),
            stats,
        )


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (index - low)


def read_documents(path, column, limit):
    documents = []
    for shard in sorted(path.glob("*.parquet")):
        parquet = pq.ParquetFile(shard)
        if column not in parquet.schema_arrow.names:
            raise ValueError(f"{column!r} is missing from {shard}")
        for row_group in range(parquet.num_row_groups):
            documents.extend(parquet.read_row_group(row_group, columns=[column]).column(0).to_pylist())
            if len(documents) >= limit:
                return documents[:limit]
    if not documents or not all(isinstance(value, str) for value in documents):
        raise ValueError("The selected corpus must contain text documents")
    return documents


def find_dataset(path):
    path = path.expanduser().resolve()
    if any(path.glob("*.parquet")):
        return path
    candidates = sorted({item.parent for item in path.rglob("*.parquet")}) if path.exists() else []
    if len(candidates) != 1:
        raise ValueError(f"Expected one Parquet dataset under {path}; found {len(candidates)}")
    return candidates[0]


def token_rows(raw, tokens, tokenizer, on_the_fly):
    for index in cycle(range(len(raw))):
        yield tokenizer.encode_one(raw[index]) if on_the_fly else list(tokens[index])


def build_specs(raw, tokens, tokenizer, on_the_fly, raw_path, token_path, policy, selected, args):
    group, specs = (STREAM_GROUP if policy == STREAM else BEST_FIT_GROUP), []
    rows = lambda: token_rows(raw, tokens, tokenizer, on_the_fly)
    if policy == STREAM and "gpt_lab_stream" in selected:
        def gpt_lab():
            stats = PackingStats()
            loader = DistDataLoader(
                CorpusDocuments(raw, tokens, tokenizer, on_the_fly),
                args.batch_size, args.seq_len, device=args.device,
                packing_strategy="stream", bos_token_id=tokenizer.get_bos_token_id(),
                packing_stats=stats,
            )
            return StatsAdapter(loader, stats)
        specs.append(LoaderSpec("GPT-Lab DistDataLoader", "gpt_lab_stream", policy, group, "gpt_lab.data.loader.DistDataLoader", gpt_lab))
    if "custom_pytorch" in selected:
        def custom():
            dataset = TorchPackedDataset(None, None, args.batch_size, args.seq_len, True, "stream" if policy == STREAM else "bestfit", args.best_fit_buffer_docs)
            iterator = dataset._stream(rows()) if policy == STREAM else dataset._bestfit(rows())
            return TransferIterator(iterator, args.device)
        specs.append(LoaderSpec("Custom PyTorch packer", "custom_pytorch", policy, group, "benchmark-local implementation; not PyTorch generally", custom))
    nano_id = "nanochat_stream" if policy == STREAM else "nanochat_best_fit"
    if nano_id in selected:
        if policy == STREAM:
            factory = lambda: nanochat_flat_stream(rows(), args.batch_size, args.seq_len, args.device)
            name = "nanochat flat stream (adapted)"
        else:
            def factory():
                stats = PackingStats()
                path = raw_path if on_the_fly else token_path
                active_tokenizer = tokenizer if on_the_fly else IdentityTokenizer(tokenizer.get_bos_token_id(), tokenizer.vocab_size)
                loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
                    active_tokenizer, B=args.batch_size, T=args.seq_len, split="train",
                    tokenizer_threads=1, tokenizer_batch_size=1, device=args.device,
                    buffer_size=args.best_fit_buffer_docs, base_path=path, packing_stats=stats,
                )
                return StatsAdapter(loader, stats)
            name = "nanochat BOS best-fit (adapted)"
        specs.append(LoaderSpec(name, nano_id, policy, group, "vendored/adapted from nanochat 3c3a3d7; no upstream runtime import", factory))
    return specs


def correctness_gate(specs, documents, bos, args):
    reference_dataset = TorchPackedDataset(None, None, args.batch_size, args.seq_len, True, "stream" if specs[0].packing_policy == STREAM else "bestfit", args.best_fit_buffer_docs)
    reference = reference_dataset._stream(cycle(documents)) if specs[0].packing_policy == STREAM else reference_dataset._bestfit(cycle(documents))
    expected = [(x.clone(), y.clone(), stats) for x, y, stats in (next(reference) for _ in range(args.correctness_batches))]
    alignments = {}
    for spec in specs:
        loader, previous, bos_rows, total_rows = spec.factory(), None, 0, 0
        for batch_index, (expected_inputs, expected_targets, expected_stats) in enumerate(expected):
            inputs, targets, stats = next(loader)
            _sync(args.device)
            inputs, targets = inputs.detach().cpu().clone(), targets.detach().cpu().clone()
            label = f"{spec.implementation} correctness batch {batch_index}"
            if inputs.shape != (args.batch_size, args.seq_len) or targets.shape != inputs.shape:
                raise RuntimeError(f"{label}: invalid shapes")
            if not torch.equal(inputs, expected_inputs) or not torch.equal(targets, expected_targets):
                raise RuntimeError(f"{label}: differs from independent {spec.packing_policy} reference")
            if spec.packing_policy == STREAM:
                if not torch.equal(inputs.flatten()[1:], targets.flatten()[:-1]):
                    raise RuntimeError(f"{label}: invalid flat shift")
                if previous is not None and inputs.flatten()[0] != previous.flatten()[-1]:
                    raise RuntimeError(f"{label}: invalid one-token carry")
                if stats["destructive_cropped_tokens"]:
                    raise RuntimeError(f"{label}: stream destructively cropped tokens")
                if stats["skipped_adjacent_transitions"]:
                    raise RuntimeError(f"{label}: stream skipped adjacent transitions")
            else:
                if not torch.equal(inputs[:, 1:], targets[:, :-1]) or not bool(torch.all(inputs[:, 0] == bos)):
                    raise RuntimeError(f"{label}: invalid BOS alignment or row shift")
                if stats["destructive_cropped_tokens"] != expected_stats["destructive_cropped_tokens"]:
                    raise RuntimeError(f"{label}: crop accounting differs from reference")
            try:
                check_accounting(
                    source_tokens_read=stats["source_tokens_read"],
                    new_source_tokens_advanced=stats["new_source_tokens_advanced"],
                    buffered_source_tokens_delta=stats["buffered_source_tokens_delta"],
                    target_positions_emitted=args.batch_size * args.seq_len,
                    skipped_adjacent_transitions=stats["skipped_adjacent_transitions"],
                    synthetic_bos_tokens_inserted=stats["synthetic_bos_tokens_inserted"],
                )
            except AssertionError as error:
                raise RuntimeError(f"{label}: aggregate accounting failed") from error
            bos_rows += int((inputs[:, 0] == bos).sum()); total_rows += inputs.shape[0]
            previous = targets
        alignments[spec.implementation_id] = bos_rows / total_rows
    return alignments


def benchmark_trial(spec, args):
    loader = spec.factory()
    for _ in range(args.warmup_batches):
        next(loader)
    _sync(args.device)
    latencies, stats = [], []
    for _ in range(args.batches):
        _sync(args.device); started = time.perf_counter()
        _, _, batch_stats = next(loader)
        _sync(args.device); latencies.append((time.perf_counter() - started) * 1000)
        stats.append(batch_stats)
    tokens = args.batches * args.batch_size * args.seq_len
    for batch_stats in stats:
        check_accounting(
            source_tokens_read=batch_stats["source_tokens_read"],
            new_source_tokens_advanced=batch_stats["new_source_tokens_advanced"],
            buffered_source_tokens_delta=batch_stats["buffered_source_tokens_delta"],
            target_positions_emitted=args.batch_size * args.seq_len,
            skipped_adjacent_transitions=batch_stats["skipped_adjacent_transitions"],
            synthetic_bos_tokens_inserted=batch_stats["synthetic_bos_tokens_inserted"],
        )
    return tokens / (sum(latencies) / 1000), latencies, stats


def aggregate(spec, trials, alignment, mode, tokenizer_name, pretokenization_seconds, args):
    throughputs = [trial[0] for trial in trials]
    latencies = [latency for _, values, _ in trials for latency in values]
    stats = [value for _, _, values in trials for value in values]
    token_samples = [row["actual_buffered_tokens"] for row in stats if row.get("actual_buffered_tokens") is not None]
    document_samples = [row["actual_buffered_documents"] for row in stats if row.get("actual_buffered_documents") is not None]
    cropped = sum(row["destructive_cropped_tokens"] for row in stats)
    advanced = sum(row["new_source_tokens_advanced"] for row in stats)
    source_read = sum(row["source_tokens_read"] for row in stats)
    target_positions = len(stats) * args.batch_size * args.seq_len
    skipped = sum(row["skipped_adjacent_transitions"] for row in stats)
    synthetic = sum(row["synthetic_bos_tokens_inserted"] for row in stats)
    intentional = sum(row["intentional_bos_boundaries"] for row in stats)
    buffered_delta = sum(row["buffered_source_tokens_delta"] for row in stats)
    represented = advanced - skipped
    check_accounting(
        source_tokens_read=source_read,
        new_source_tokens_advanced=advanced,
        buffered_source_tokens_delta=buffered_delta,
        target_positions_emitted=target_positions,
        skipped_adjacent_transitions=skipped,
        synthetic_bos_tokens_inserted=synthetic,
    )
    return ComparisonResult(
        spec.comparison_group, spec.implementation, spec.implementation_id,
        spec.packing_policy, spec.provenance, mode, tokenizer_name, str(args.device),
        "pinned CPU + non-blocking CUDA copy; regular copy otherwise",
        args.batch_size, args.seq_len, args.trials, args.batches, args.trials * args.batches,
        statistics.median(throughputs), percentile(throughputs, .5), percentile(throughputs, .95),
        statistics.mean(throughputs), statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        statistics.median(latencies), percentile(latencies, .5), percentile(latencies, .95),
        statistics.mean(latencies), statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "none" if spec.packing_policy == STREAM else "discard selected document remainder",
        cropped, cropped / max(advanced, 1), alignment,
        args.batch_size * args.seq_len + 1 if spec.packing_policy == STREAM else None,
        args.best_fit_buffer_docs if spec.packing_policy == BEST_FIT else None,
        statistics.mean(token_samples) if token_samples else None,
        min(token_samples) if token_samples else None, max(token_samples) if token_samples else None,
        statistics.mean(document_samples) if document_samples else None,
        source_read, advanced, target_positions, target_positions,
        skipped, synthetic, intentional, buffered_delta,
        (advanced - cropped) / max(advanced, 1), 1.0,
        represented / max(target_positions, 1), represented / max(advanced, 1),
        "passed", args.correctness_batches, pretokenization_seconds,
    )


def write_outputs(results, metadata, output, plots, html_report):
    rows = [asdict(result) for result in results]
    (output / "results.json").write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2))
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    images = []
    if plots:
        try:
            import matplotlib.pyplot as plt
            for policy in (STREAM, BEST_FIT):
                selected = [row for row in results if row.packing_policy == policy]
                if not selected: continue
                labels = [f"{row.implementation}\n{row.tokenization}" for row in selected]
                figure, axes = plt.subplots(1, 4, figsize=(max(18, len(selected) * 2.6), 5))
                axes[0].bar(labels, [row.throughput_median_tokens_s for row in selected])
                axes[1].bar(labels, [row.latency_p50_ms for row in selected], yerr=[row.latency_p95_ms - row.latency_p50_ms for row in selected], capsize=4)
                axes[0].set_ylabel("tokens/s (trial median)"); axes[1].set_ylabel("latency p50, whisker to p95 (ms)")
                x, width = list(range(len(selected))), 0.25
                for offset, field_name, label in (
                    (-1, "source_token_utilization", "source preserved"),
                    (0, "source_transition_coverage", "source transitions covered"),
                    (1, "target_supervision_utilization", "targets supervising source"),
                ):
                    axes[2].bar(
                        [value + offset * width for value in x],
                        [getattr(row, field_name) for row in selected],
                        width,
                        label=label,
                    )
                axes[2].set_xticks(x, labels); axes[2].set_ylim(0, 1.05)
                axes[2].set_ylabel("utilization fraction"); axes[2].legend(fontsize=8)
                axes[3].bar(labels, [row.destructive_crop_rate for row in selected])
                axes[3].set_ylim(0, 1.05)
                axes[3].set_ylabel("destructive crop rate")
                for axis in axes: axis.tick_params(axis="x", labelrotation=20); axis.grid(axis="y", alpha=.25)
                figure.suptitle(f"Matched policy: {policy}"); figure.tight_layout()
                path = output / f"{policy}.png"; figure.savefig(path, dpi=150); plt.close(figure); images.append(path)
        except ImportError:
            print("matplotlib unavailable; skipping plots", file=sys.stderr)
    if html_report:
        columns = [
            "comparison_group", "packing_policy", "implementation", "tokenization",
            "throughput_median_tokens_s", "latency_p50_ms", "latency_p95_ms",
            "source_tokens_read", "new_source_tokens_advanced", "target_positions_emitted",
            "destructively_cropped_tokens", "skipped_adjacent_transitions",
            "synthetic_bos_tokens_inserted", "intentional_bos_boundaries",
            "source_token_utilization", "compute_utilization",
            "target_supervision_utilization", "source_transition_coverage",
            "destructive_crop_rate", "bos_row_alignment", "actual_buffered_tokens_mean",
        ]
        heading = "".join(f"<th>{column}</th>" for column in columns)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(asdict(row)[column]))}</td>" for column in columns) + "</tr>" for row in results)
        pictures = "".join(f'<img src="{path.name}" alt="{path.stem}">' for path in images)
        (output / "report.html").write_text(f"<h1>Policy-matched dataloaders</h1><pre>{html.escape(json.dumps(metadata, indent=2))}</pre><table><tr>{heading}</tr>{body}</table>{pictures}")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, default=DATA_DIR); parser.add_argument("--column", default="text")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results") / datetime.now().strftime("dataloaders-%Y%m%d-%H%M%S"))
    parser.add_argument("--tokenizers", default="gpt2"); parser.add_argument("--tokenization", choices=("on-the-fly", "pretokenized", "both"), default="both")
    parser.add_argument("--groups", default=f"{STREAM_GROUP},{BEST_FIT_GROUP}")
    parser.add_argument("--implementations", default="gpt_lab_stream,custom_pytorch,nanochat_stream,nanochat_best_fit")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batches", type=int, default=300); parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup-batches", type=int, default=20); parser.add_argument("--correctness-batches", type=int, default=3)
    parser.add_argument("--max-docs", type=int, default=4096); parser.add_argument("--row-group-size", type=int, default=256)
    parser.add_argument("--best-fit-buffer-docs", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true"); parser.add_argument("--html", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.max_docs, args.batches, args.trials, args.warmup_batches = min(args.max_docs, 256), 10, 2, 3
        args.best_fit_buffer_docs = min(args.best_fit_buffer_docs, 64)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA unavailable")
    if args.device.type == "mps" and not torch.backends.mps.is_available(): parser.error("MPS unavailable")
    if min(args.batches, args.trials, args.warmup_batches, args.correctness_batches) < 1: parser.error("timing counts must be positive")
    return args


def load_tokenizer(name):
    try:
        return Tokenizer.from_pretrained(name)
    except UnboundLocalError:
        return Tokenizer.from_config(TokenizerConfig(name=name, source="tiktoken"))


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = find_dataset(args.dataset_path); raw = read_documents(dataset, args.column, args.max_docs)
    selected, groups = set(filter(None, args.implementations.split(","))), set(filter(None, args.groups.split(",")))
    known = {"gpt_lab_stream", "custom_pytorch", "nanochat_stream", "nanochat_best_fit"}
    if selected - known: raise ValueError(f"Unknown implementations: {sorted(selected - known)}")
    if groups - {STREAM_GROUP, BEST_FIT_GROUP}: raise ValueError(f"Unknown groups: {sorted(groups)}")
    modes = ("on-the-fly", "pretokenized") if args.tokenization == "both" else (args.tokenization,)
    results, execution_orders, pretokenization = [], [], {}
    for tokenizer_name in filter(None, args.tokenizers.split(",")):
        tokenizer = BenchmarkTokenizer(load_tokenizer(tokenizer_name)); bos = tokenizer.get_bos_token_id()
        started = time.perf_counter(); tokens = tokenizer.encode(raw, prepend=bos); elapsed = time.perf_counter() - started
        pretokenization[tokenizer_name] = {"seconds": elapsed, "documents": len(raw), "tokens": sum(map(len, tokens))}
        corpus = args.output_dir / "corpus" / tokenizer_name.replace("/", "_")
        raw_path, token_path = corpus / "raw", corpus / "pretokenized"
        write_split(raw_path, raw, args.row_group_size); write_split(token_path, tokens, args.row_group_size)
        for mode in modes:
            for policy, group in ((STREAM, STREAM_GROUP), (BEST_FIT, BEST_FIT_GROUP)):
                if group not in groups: continue
                specs = build_specs(raw, tokens, tokenizer, mode == "on-the-fly", raw_path, token_path, policy, selected, args)
                if len(specs) < 2: continue
                print(f"Correctness gate: {group} / {mode} / {tokenizer_name}")
                alignments = correctness_gate(specs, tokens, bos, args)
                trials = {spec.implementation_id: [] for spec in specs}
                for trial in range(args.trials):
                    offset = trial % len(specs); order = [*specs[offset:], *specs[:offset]]
                    execution_orders.append({"tokenizer": tokenizer_name, "tokenization": mode, "packing_policy": policy, "trial": trial, "order": [spec.implementation_id for spec in order]})
                    for spec in order:
                        print(f"Trial {trial + 1}/{args.trials}: {spec.implementation} / {policy} / {mode}")
                        trials[spec.implementation_id].append(benchmark_trial(spec, args))
                results.extend(aggregate(spec, trials[spec.implementation_id], alignments[spec.implementation_id], mode, tokenizer_name, elapsed, args) for spec in specs)
    if not results: raise RuntimeError("No selected matched group contains at least two implementations")
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(), "command": sys.argv,
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "device": str(args.device),
        "corpus": {"path": str(dataset), "documents": len(raw), "order": "sorted shards/row groups/rows, repeated exactly"},
        "config": {"batch_size": args.batch_size, "sequence_length": args.seq_len, "trials": args.trials, "batches_per_trial": args.batches, "warmup_batches": args.warmup_batches, "correctness_batches": args.correctness_batches, "best_fit_buffer_documents": args.best_fit_buffer_docs, "tokenization": args.tokenization, "tokenizers": args.tokenizers},
        "pretokenization": pretokenization,
        "matched_groups": {STREAM_GROUP: ["GPT-Lab", "Custom PyTorch", "nanochat adapted"], BEST_FIT_GROUP: ["Custom PyTorch", "nanochat adapted"]},
        "gpt_lab_best_fit": "Unavailable: GPT-Lab bos_aligned retains suffixes and is not destructive best-fit, so it is excluded.",
        "invalid_previous_comparisons": ["stream versus destructive BOS-best-fit", "B*(T+1) row stream versus flat B*T+1 carry stream", "loader timing obscured by model execution"],
        "timing": {"scope": "loader-only next() and device transfer", "excluded": ["construction", "correctness", "warmup", "pretokenization", "model execution"], "accelerator_sync": "before/after each measured batch", "rotated_orders": execution_orders},
        "metric_definitions": {
            "source_tokens_read": "Tokenized source tokens entering the packer during measured batches, including original document BOS tokens; excludes warmup and synthetic BOS.",
            "new_source_tokens_advanced": "Unique source-stream tokens permanently advanced during measured batches, including discarded tokens and excluding a retained carry token.",
            "target_positions_emitted": "Language-model target tensor positions produced; exactly trials * batches * B * T.",
            "active_target_positions": "Emitted target positions that are not padding/ignore positions.",
            "destructively_cropped_tokens": (
                "Original tokenized source tokens permanently discarded by the packing algorithm "
                "and never emitted or retained for a future batch. Excludes the shifted-batch carry, "
                "padding, synthetic BOS, temporarily buffered tails, and separately reported skipped "
                "adjacent transitions."
            ),
            "skipped_adjacent_transitions": "Adjacent source-token pairs not emitted as (input, target), including crop loss and intentional row segmentation at BOS.",
            "synthetic_bos_tokens_inserted": "Continuation BOS tokens absent from the source; never counted as source read or advanced.",
            "intentional_bos_boundaries": "Subset of skipped transitions deliberately segmented because a row starts at an original source BOS.",
            "buffered_source_tokens_delta": "Final minus initial buffered source tokens over measured batches, including retained carry tokens.",
            "bos_row_alignment": "Fraction of rows whose first input token is BOS.",
            "source_token_utilization": "(advanced - destructively cropped) / advanced; source preservation.",
            "compute_utilization": "Active target positions / emitted target positions; 1.0 for these full, unpadded benchmark policies.",
            "target_supervision_utilization": "Represented adjacent source transitions / emitted target positions; excludes synthetic and padding supervision.",
            "source_transition_coverage": "Represented adjacent source transitions / advanced source tokens.",
            "destructive_crop_rate": "Destructively cropped source tokens / advanced source tokens.",
        },
        "accounting_invariants": {
            "source_balance": "source_tokens_read = new_source_tokens_advanced + buffered_source_tokens_delta",
            "supervision_balance": "active_target_positions = new_source_tokens_advanced - skipped_adjacent_transitions + synthetic_bos_tokens_inserted",
            "target_capacity": "target_positions_emitted = trials * batches * B * T",
            "fifo": "Flat B*T+1 stream retains one carry and has zero destructive crop and zero skipped adjacent transitions.",
        },
        "instrumentation": "Packing paths emit aggregate integer deltas only. Exact discarded suffix provenance is enabled only by debug tests and is outside throughput timing.",
        "interpretation": "Implementation-level, policy-matched results; not a global framework ranking.",
    }
    write_outputs(results, metadata, args.output_dir, not args.no_plots, args.html)
    for result in results:
        print(
            f"{result.packing_policy:24} {result.implementation:34} {result.tokenization:12} "
            f"{result.throughput_median_tokens_s:12,.0f} tok/s  "
            f"source={result.source_token_utilization:.2%} "
            f"coverage={result.source_transition_coverage:.2%} "
            f"supervision={result.target_supervision_utilization:.2%} "
            f"crop={result.destructive_crop_rate:.2%} "
            f"skips={result.skipped_adjacent_transitions:,} "
            f"synth_BOS={result.synthetic_bos_tokens_inserted:,} "
            f"BOS_rows={result.bos_row_alignment:.2%}"
        )
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
