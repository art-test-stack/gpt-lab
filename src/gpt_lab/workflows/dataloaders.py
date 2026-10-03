"""Offline adapters around the production loaders, plus the original timing report.

Synthetic on-the-fly conversion parses serialized integer tokens; it is not a
measurement of natural-language BPE throughput. Real mode uses the original
uint32 .bin / uint64 .idx format (offsets include the final token boundary).
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import statistics
import time

import numpy as np
import torch

from gpt_lab.data.loader import DistDataLoader, tokenizing_distributed_data_loader_with_state_bos_bestfit
from gpt_lab.utils.schemas import DataLoaderState


def make_synthetic_bin_idx(num_docs=10_000, min_len=32, max_len=1024, vocab=50_257, seed=42):
    """
    Build in-memory fake .bin / .idx buffers that look like PretokenizedDataset files.
    Returns (tokens_np, offsets_np) – same dtypes as the real files.
    """
    rng = np.random.default_rng(seed)
    lengths = rng.integers(min_len, max_len + 1, size=num_docs)
    tokens = rng.integers(1, vocab, size=int(lengths.sum()), dtype=np.uint32)
    offsets = np.concatenate([[0], lengths.cumsum()]).astype(np.uint64)
    return tokens, offsets


class SyntheticPretokenizedDataset:
    """Drop-in replacement for PretokenizedDataset that lives in RAM."""

    def __init__(self, tokens, offsets):
        self.tokens = tokens
        self.offsets = offsets
        self.num_docs = len(offsets) - 1

    def __len__(self):
        return self.num_docs

    def get_doc(self, idx):
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1]) if idx + 1 < self.num_docs else len(self.tokens)
        return torch.from_numpy(self.tokens[start:end].astype(np.int64))



def benchmark_loader(name, loader, B, T, num_batches, bos_id, device):
    latencies = []
    last_stats = {}

    # warm-up
    for _ in range(2):
        next(loader)

    bos_rows = 0
    total_rows = 0

    for _ in range(num_batches):
        synchronize(device)
        t0 = time.perf_counter()
        inputs, targets, stats = next(loader)
        synchronize(device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        last_stats = stats if isinstance(stats, dict) else {}

        # check BOS alignment
        bos_rows  += int((inputs[:, 0] == bos_id).sum().item())
        total_rows += B

    return {
        "name": name,
        "mean_latency_ms": statistics.mean(latencies),
        "std_latency_ms":  statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "throughput_tok_s": (B * T * 1000) / statistics.mean(latencies),
        "crop_rate": last_stats["cropped_tokens"] / max(last_stats.get("total_tokens", 1), 1) if "cropped_tokens" in last_stats else None,
        "mean_search_us": statistics.mean(last_stats["search_times_us"]) if last_stats.get("search_times_us") else None,
        "bos_alignment": bos_rows / max(total_rows, 1),
    }


def print_results(results: list[dict]):
    keys = [
        ("mean_latency_ms",   "Latency mean (ms)",      ".2f"),
        ("std_latency_ms",    "Latency std  (ms)",      ".2f"),
        ("throughput_tok_s",  "Throughput (tok/s)",     ",.0f"),
        ("crop_rate",         "Crop rate",              ".2%"),
        ("mean_search_us",    "Search time mean (μs)",  ".3f"),
        ("bos_alignment",     "BOS alignment",          ".2%"),
    ]

    col_w = 32
    name_w = 42

    header = f"{'Metric':<{col_w}}" + "".join(f"{r['name']:<{name_w}}" for r in results)
    print()
    print("=" * (col_w + name_w * len(results)))
    print(header)
    print("-" * (col_w + name_w * len(results)))

    for key, label, fmt in keys:
        row = f"{label:<{col_w}}"
        for r in results:
            val = r.get(key)
            if val is None:
                row += f"{'N/A':<{name_w}}"
            else:
                row += f"{format(val, fmt):<{name_w}}"
        print(row)

    print("=" * (col_w + name_w * len(results)))
    print()


def synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


class IntegerTokenizer:
    """Offline conversion fixture; no pretrained tokenizer downloads."""
    def __init__(self, bos_id):
        self.bos_id = bos_id

    def get_bos_token_id(self):
        return self.bos_id

    def encode(self, texts, prepend=None, **kwargs):
        return [[self.bos_id] + [int(token) for token in text.split()] for text in texts]


class Documents:
    def __init__(self, dataset, bos_id, texts=None):
        self.dataset = dataset
        self.bos_id = bos_id
        self.texts = texts

    def __iter__(self):
        epoch = 1
        while True:
            for index in range(len(self.dataset)):
                if self.texts is None:
                    doc = self.dataset.get_doc(index)
                else:
                    doc = torch.tensor([int(token) for token in self.texts[index].split()], dtype=torch.long)
                tokens = torch.cat((torch.tensor([self.bos_id]), doc))
                yield tokens, DataLoaderState(epoch=epoch, row_offset=index)
            epoch += 1


def load_dataset(args):
    if args.mode == 'synthetic':
        tokens, offsets = make_synthetic_bin_idx(num_docs=args.num_docs)
    else:
        tokens = np.memmap(args.bin, dtype=np.uint32, mode='r')
        offsets = np.memmap(args.idx, dtype=np.uint64, mode='r')
        if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(tokens):
            raise ValueError('.idx must contain document boundaries starting at 0 and ending at the .bin token count')
        if np.any(offsets[1:] <= offsets[:-1]):
            raise ValueError('.idx document boundaries must be strictly increasing')
    return SyntheticPretokenizedDataset(tokens, offsets)


def run(args):
    dataset = load_dataset(args)
    selected = {
        'all': ('glab-tokenized', 'glab-on-the-fly', 'nanochat'),
        'glab': ('glab-tokenized', 'glab-on-the-fly'),
    }.get(args.loader, (args.loader,))
    print(f'Dataloader benchmark: mode={args.mode}, B={args.batch_size}, T={args.seq_len}, device={args.device}')
    print('On-the-fly variants parse integer text; results do not measure BPE tokenization.')
    print('Buffer size is a token budget for gpt-lab and a document count for nanochat.')
    texts = None
    if any(name != 'glab-tokenized' for name in selected):
        texts = [' '.join(map(str, dataset.tokens[int(dataset.offsets[i]):int(dataset.offsets[i + 1])]))
                 for i in range(len(dataset))]
    results = []
    with TemporaryDirectory(prefix='gpt_lab_benchmark_') as temporary:
        for name in selected:
            start = time.perf_counter()
            if name == 'nanochat':
                import pyarrow as pa
                import pyarrow.parquet as pq
                # Reuse the production nanochat iterator without monkeypatching
                # its globals. It reserves the final Parquet shard for validation.
                path = Path(temporary)
                pq.write_table(pa.table({'text': texts}), path / 'shard_00000.parquet')
                pq.write_table(pa.table({'text': texts[:1]}), path / 'shard_00001.parquet')
                loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
                    IntegerTokenizer(args.bos_id), args.batch_size, args.seq_len, 'train',
                    device=args.device, buffer_size=args.buffer_size, base_path=path,
                )
            else:
                documents = Documents(dataset, args.bos_id, texts if name == 'glab-on-the-fly' else None)
                loader = DistDataLoader(documents, batch_size=args.batch_size, seq_len=args.seq_len,
                                        buffer_size=args.buffer_size, device=args.device)
            initialization_time = time.perf_counter() - start
            result = benchmark_loader(name, loader, args.batch_size, args.seq_len,
                                      args.num_batches, args.bos_id, args.device)
            result['initialization_time_s'] = initialization_time
            results.append(result)
    print_results(results)
    return results
