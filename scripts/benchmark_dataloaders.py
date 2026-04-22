"""
benchmark_dataloaders.py (Sonnet 4.6 generated)
========================
Compares PackedDataLoader (binary/pretokenized) vs the nanochat
tokenizing_distributed_data_loader_with_state_bos_bestfit loader.

Metrics reported
----------------
  - Throughput          tokens/sec (inputs only, i.e. B*T per batch)
  - Batch latency       ms per batch (mean ± std)
  - Packing efficiency  fraction of non-padding tokens (always 1.0 for both,
                        but we measure tokens-per-row to verify)
  - Crop rate           fraction of tokens discarded due to cropping
  - BOS alignment       fraction of rows that begin with the BOS token id
  - Buffer search time  time spent inside the best-fit search loop (us)

Usage
-----
  # Quick synthetic test (no real data needed)
  uv run python -m scripts.benchmark_dataloaders --mode synthetic

  # Run for different buffer sizes
  for buf in 100 500 1000 2000; do
    uv run python -m scripts.benchmark_dataloaders --buffer_size $buf
done

  # Test against real data
  uv run python -m scripts.benchmark_dataloaders --mode real \
      --bin path/to/data.bin \
      --idx path/to/data.idx \
      --parquet_dir path/to/parquets/

  # Run only one loader
  uv run python -m scripts.benchmark_dataloaders --mode synthetic --loader packed
  uv run python -m scripts.benchmark_dataloaders --mode synthetic --loader nanochat
"""

import argparse
import time
import statistics
import sys
from collections import defaultdict

import torch
import numpy as np

from gpt_lab.data.loader import build_dataloader

# ─────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────

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
        self.num_docs = len(offsets)

    def __len__(self):
        return self.num_docs

    def get_doc(self, idx):
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1]) if idx + 1 < self.num_docs else len(self.tokens)
        return torch.from_numpy(self.tokens[start:end].astype(np.int64))


# ─────────────────────────────────────────────
# HuggingFace dataloader
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
# PyTorch-based dataloader
# from https://github.com/mddunlap924/PyTorch-LLM/blob/4bb378dcf6352c538b13f94ad5c325de5961f568/src/dataloading/preprocess.py
# ─────────────────────────────────────────────

from pathlib import Path
import pandas as pd


# Load Data
class LoadData:
    """ 
    Load CSV Data Files
    (Expand this class to other datasets suitable for your needs)
    """

    def __init__(self, base_dir: str):
        """
        :param base_dir: Directory data files are stored
        """
        self.base_dir = Path(base_dir)


    def load(self, filename: str) -> pd.DataFrame:
        """
        Pandas Read CSV filename 
        :param filename: Name of File to Load
        :return: Data returned as a Pandas DataFrame
        """
        return pd.read_csv(self.base_dir / filename,
                           low_memory=False)
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader


class CustomTextCollator:
    """
    Data Collator used for a classification task. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, tokenizer, tokenizer_cfg):

        # Tokenizer to be used inside the class.
        self.tokenizer = tokenizer

        # Tokenizer configuration
        self.tok_cfg = tokenizer_cfg

        # Check max sequence length.
        self.max_sequence_len = tokenizer_cfg.max_length
        return


    def __call__(self, sequences):
        """
        This function allows the class objects to be used as a function call.
        Since the PyTorch DataLoader needs a collator function, this 
        class can be used as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holds the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]

        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        # https://huggingface.co/docs/transformers/pad_truncation
        inputs = self.tokenizer(text=texts,
                                return_tensors=self.tok_cfg.return_tensors,
                                padding=self.tok_cfg.padding,
                                truncation=self.tok_cfg.truncation,
                                max_length=self.max_sequence_len,
                                add_special_tokens=self.tok_cfg.add_special_tokens,
                                )
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels': torch.tensor(labels, dtype=torch.long)})
        return inputs


class TrainDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tok,
                 tok_cfg,
                 X_cols: list[str],
                 label: str,
                 encoder):
        self.df = df
        self.tokenizer = tok
        self.tokenizer_cfg = tok_cfg
        self.X_cols = X_cols
        self.label = label
        self.encoder = encoder


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        # Extract all source fields into a list
        text = []
        for col in self.X_cols:
            if col == 'ZIP code':
                feature = f'Zip code {self.df[col].iloc[idx]}'
            elif col == 'Sub-issue':
                feature = f'{self.df[col].iloc[idx]}'
            elif col == 'Consumer complaint narrative':
                feature = self.df[col].iloc[idx]
            text.append(feature)

        # Combine the fields using special SEP token
        text = '[SEP]'.join(text)
        # Extract all source fields into a list
        # text = self.df['Consumer complaint narrative'].iloc[idx]

        # Convert text labels into labels (e.g., if 18 classes then labels are 0-17)
        label_text = self.df[self.label].iloc[idx]
        label = self.encoder.transform([label_text])[0]
        return {'text': text, 'label': label}


class TestDataset(Dataset):
    def __init__(self, df, tokenizer, tokenizer_cfg):
        self.tokenizer = tokenizer
        self.tokenizer_cfg = tokenizer_cfg
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(tokenizer=self.tokenizer,
                               cfg=self.tokenizer_cfg,
                               text=self.texts[item])
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.float)
        return {'input_ids': input_ids}


def get_ds_dl(df,
              cfg,
              tokenizer,
              encoder,
              collator):
    "Get the PyTorch Dataset (ds) and Dataloader (dl)"
    # Dataset 
    ds = TrainDataset(df=df,
                      tok=tokenizer,
                      tok_cfg=cfg.tokenizer,
                      X_cols=cfg.data_info.source_fields,
                      label=cfg.data_info.target,
                      encoder=encoder)

    # Dataloader
    dl = DataLoader(ds,
                    batch_size=cfg.batch_size,
                    collate_fn=collator,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    )
    return ds, dl



# ─────────────────────────────────────────────
# nanochat dataloader
# copied for easy comparision from 
# https://github.com/karpathy/nanochat/blob/324e69c45d3606095adb6b409078647145165454/nanochat/dataloader.py
# ─────────────────────────────────────────────
"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import torch
import pyarrow.parquet as pq

from gpt_lab.utils.distributed import get_dist_info # replaced 'from nanochat.common import get_dist_info'
# L317-353 replaced 'from nanochat.dataset import list_parquet_files'
import os
from gpt_lab.utils.common import DATA_DIR
base_dir = DATA_DIR
def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir

    # Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    # This code will eventually be deleted.
    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            print()
            print("    python -m nanochat.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            print()
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            print("=" * 80)
            print()
        # attempt a fallback to the legacy data directory
        data_dir = os.path.join(base_dir, "base_data")

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets

# ─────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────

def benchmark_loader(name, loader, B, T, num_batches, bos_id, device):
    latencies = []
    last_stats = {}

    # warm-up
    for _ in range(2):
        next(loader)

    bos_rows = 0
    total_rows = 0

    for _ in range(num_batches):
        t0 = time.perf_counter()
        inputs, targets, stats = next(loader)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        last_stats = stats

        # check BOS alignment
        bos_rows  += int((inputs[:, 0] == bos_id).sum().item())
        total_rows += B

    return {
        "name": name,
        "mean_latency_ms": statistics.mean(latencies),
        "std_latency_ms":  statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "throughput_tok_s": (B * T * 1000) / statistics.mean(latencies),
        "crop_rate": last_stats.get("cropped_tokens", 0) / max(last_stats.get("total_tokens", 1), 1),
        "mean_search_us": statistics.mean(last_stats["search_times_us"]) if last_stats.get("search_times_us") else 0.0,
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

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark PackedDataLoader vs nanochat best-fit")
    parser.add_argument("--mode",        choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--loader",      choices=["glib", "glib-on-the-fly", "glib-tokenized", "nanochat", "all"], default="all")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--seq_len",     type=int, default=2048)
    parser.add_argument("--num_batches", type=int, default=20,
                        help="Number of batches to time (after 2 warm-up batches)")
    parser.add_argument("--buffer_size", type=int, default=1000,
                        help="Document buffer size for both loaders")
    parser.add_argument("--num_docs",    type=int, default=20_000,
                        help="Number of synthetic documents")
    parser.add_argument("--device",      default="cpu",
                        help="torch device, e.g. cpu or cuda")
    parser.add_argument("--bos_id",      type=int, default=1,
                        help="BOS token id (for nanochat loader)")
    # real-data paths
    parser.add_argument("--bin",  default=None, help="Path to pretokenized .bin file")
    parser.add_argument("--idx",  default=None, help="Path to pretokenized .idx file")
    args = parser.parse_args()

    print(f"\n{'─'*60}")
    print(f"  Dataloader benchmark")
    print(f"  mode={args.mode}  B={args.batch_size}  T={args.seq_len}")
    print(f"  batches={args.num_batches}  buffer={args.buffer_size}  device={args.device}")
    print(f"{'─'*60}\n")

    if args.mode == "synthetic":
        print(f"Generating {args.num_docs:,} synthetic documents …")
        tokens, offsets = make_synthetic_bin_idx(num_docs=args.num_docs)
        dataset = SyntheticPretokenizedDataset(tokens, offsets)
        print(f"  vocab=50257  len range=[32, 1024]  total tokens={len(tokens):,}\n")
    else:
        if args.bin is None or args.idx is None:
            sys.exit("--mode real requires --bin and --idx arguments.")
        # import real dataset class
        try:
            from gpt_lab.data.loader import PretokenizedDataset   # adjust import path as needed
        except ImportError:
            sys.exit("Could not import PretokenizedDataset. "
                     "Make sure dataloader.py is on sys.path.")
        dataset = PretokenizedDataset(args.bin, args.idx)
        print(f"Loaded dataset: {len(dataset):,} docs\n")

    results = []

    if args.loader in ("glib", "both"):
        print("Running DataLoader with tokenized data…")
        t0 = time.perf_counter()
        glib_loader = PackedDataLoader(
            dataset, batch_size=args.batch_size, seq_len=args.seq_len,
            buffer_size=args.buffer_size, device=args.device,
        )
        init_time = time.perf_counter() - t0
        print(f"  initialization time: {init_time:.2f}s")
        r = benchmark_loader("DataLoader (O log N) - tokenized data", glib_loader, args.batch_size, args.seq_len,
                             args.num_batches, args.buffer_size, args.bos_id, args.device)
        r["initialization_time_s"] = init_time
        results.append(r)
        print("  done.\n")
    
    if args.loader in ("glib", "both") :
        print("Running DataLoader with data tokenization on-flight…")
        t0 = time.perf_counter()
        glib_loader = DataLoader(

        )
        r = benchmark_loader("DataLoader (O log N) - tokenization on-flight", glib_loader, args.batch_size, args.seq_len,
                             args.num_batches, args.buffer_size, args.bos_id, args.device)
        init_time = time.perf_counter() - t0
        r["initialization_time_s"] = init_time
        results.append(r)
        print("  done.\n")

    if args.loader in ("nanochat", "both"):
        print("Running nanochat BOS best-fit …")
        r = benchmark_loader("nanochat BOS best-fit (O N)", dataset, args.batch_size, args.seq_len,
                               args.num_batches, args.buffer_size,
                               args.bos_id, args.device)
        results.append(r)
        print("  done.\n")

    print_results(results)

    if len(results) == 2:
        ratio = results[0]["mean_latency_ms"] / results[1]["mean_latency_ms"]
        faster = results[0]["name"] if ratio < 1 else results[1]["name"]
        slower = results[1]["name"] if ratio < 1 else results[0]["name"]
        print(f"  {faster} is {abs(1 - ratio):.1%} {'faster' if ratio < 1 else 'slower'} "
              f"than {slower} on average.\n")


if __name__ == "__main__":
    main()
