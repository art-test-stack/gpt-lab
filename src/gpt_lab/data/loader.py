from collections import deque
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple, Union, Literal

import torch

from gpt_lab.utils.default import DATA_DIR
from gpt_lab.utils.distributed import get_dist_info
from gpt_lab.data.sharder import ShardManager, ShardIterationState

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardedDataset:
    """
    Streams tokenized documents from sharded parquet files.

    Each document is tokenized on the fly and yielded as a 1-D LongTensor
    alongside its shard state. No packing or cross-document logic here —
    that is the sole responsibility of DistDataLoader.
    """

    def __init__(
        self,
        name: str,
        tokenizer: Optional[Callable] = None,
        split: str = "train",
        start_state: Optional[ShardIterationState] = None,
        base_url: Optional[str] = None,
        column: str = "text",
        shard_limit: Optional[int] = None,
        max_shards: Optional[int] = None,
        cachedir: Union[str, Path] = DATA_DIR,
        dist_info: Optional[dict] = None,
    ):
        if isinstance(cachedir, str):
            cachedir = Path(cachedir)
        if dist_info is None:
            dist_info = get_dist_info()

        self.sm = ShardManager(
            name=name,
            cachedir=cachedir,
            split=split,
            base_url=base_url or "",
            column_name=column,
            shard_limit=shard_limit,
            max_shards=max_shards,
            dist_info=dist_info,
        )
        self.tokenizer = tokenizer
        self.split = split
        self.start_state = start_state

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ShardIterationState]]:
        for texts, state in self.sm.iterate(start_state=self.start_state):
            for txt in texts:
                tokens = (
                    self.tokenizer(txt, prepend_bos=True)
                    if self.tokenizer is not None
                    else txt
                )
                # Materialise to a tensor exactly once, here.
                # Everything downstream receives a LongTensor and never
                # needs to call torch.tensor() again (no extra allocations).
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens, dtype=torch.long)
                yield tokens, state


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

class DistDataLoader:
    """
    Packs tokenized documents into fixed-shape (B, T) input/target tensors.

    Design notes
    ------------
    * Buffer is a deque of (Tensor, state) pairs — O(1) popleft and head
      update, versus O(n) for list.pop(0).

    * Buffer is sized in **tokens**, not in documents.  The old heuristic
      `buffer_size = B * T * 16` was meant to be a token budget but was
      interpreted as a document count, which could queue hundreds of
      thousands of documents and consume gigabytes of memory before the
      first batch.

    * Documents are separated by an explicit EOS token so the model always
      sees clean boundaries: …doc_N_last [EOS] [BOS] doc_N+1_first…
      Without this, cross-document targets are silently trained on,
      which is especially harmful for value-embedding architectures where
      BOS is the per-document anchor.

    * Partial-document tail is kept as a tensor *view* (tokens[take:]),
      not a Python list slice, so no allocation occurs when a document
      spans multiple batches.

    * Output tensors (self.inputs, self.targets) are pre-allocated once
      and reused every step.
    """

    def __init__(
        self,
        dataset: ShardedDataset,
        batch_size: int,
        seq_len: int,
        device: str = "cuda",
        buffer_size: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        self.dataset = dataset
        self.B = batch_size
        self.T = seq_len
        self.device = torch.device(device)
        self.eos_token_id = eos_token_id

        # Token budget for the pre-fetch buffer.
        # Default: ~16 full batches so the GPU is never starved.
        self.token_buffer_size = buffer_size or (batch_size * seq_len * 16)

        self.iterator = iter(dataset)
        self.buffer: deque[Tuple[torch.Tensor, ShardIterationState]] = deque()
        self._buffered_tokens: int = 0
        self.last_state: Optional[ShardIterationState] = None

        total = batch_size * (seq_len + 1)

        # Pinned CPU staging buffer → non-blocking async copy to GPU.
        self.cpu = torch.empty(
            total, dtype=torch.long,
            pin_memory=(self.device.type == "cuda"),
        )
        self.gpu = torch.empty(total, dtype=torch.long, device=self.device)

        # Single contiguous allocation; inputs and targets are non-overlapping
        # views into it.  Avoids two separate allocations per forward pass.
        _out = torch.empty(
            2 * batch_size * seq_len, dtype=torch.long, device=self.device
        )
        self.inputs  = _out[: batch_size * seq_len].view(batch_size, seq_len)
        self.targets = _out[batch_size * seq_len :].view(batch_size, seq_len)

    def _refill(self) -> None:
        """Pull documents from the dataset until the token budget is met."""
        while self._buffered_tokens < self.token_buffer_size:
            try:
                tokens, state = next(self.iterator)
            except StopIteration:
                if getattr(self.dataset, "split", None) == "val":
                    # Validation set is finite — wrap around silently.
                    self.iterator = iter(self.dataset)
                    continue
                # Training iterator (ShardManager.iterate) is infinite;
                # reaching here means something went wrong upstream.
                break

            # Append EOS so the model sees an explicit document boundary.
            # The next document's BOS (prepended by the tokenizer) will
            # immediately follow, giving the sequence:
            #   …last_tok [EOS] [BOS] first_tok…
            if self.eos_token_id is not None:
                tokens = torch.cat([tokens, tokens.new_tensor([self.eos_token_id])])

            self.buffer.append((tokens, state))
            self._buffered_tokens += len(tokens)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, ShardIterationState]:
        self._refill()
        B, T = self.B, self.T
        total = B * (T + 1)
        pos = 0

        while pos < total:
            if not self.buffer:
                # Mid-batch top-up (rare: only if a single doc > token_buffer_size).
                self._refill()
                if not self.buffer:
                    raise RuntimeError(
                        "DistDataLoader buffer is empty mid-batch. "
                        "The training ShardManager iterator should be infinite — "
                        "check shard availability and ShardManager.iterate()."
                    )

            # O(1) peek at deque head — no pop yet.
            tokens, state = self.buffer[0]
            self.last_state = state

            remaining = total - pos
            take = min(len(tokens), remaining)

            # Zero-copy write into the pinned CPU buffer.
            # tokens[:take] is a tensor view (no new allocation).
            self.cpu[pos : pos + take].copy_(tokens[:take])
            pos += take
            self._buffered_tokens -= take

            if take < len(tokens):
                # Document spans into the next batch.
                # Update the head in place with a view of the remainder —
                # deque[0] access and assignment are both O(1).
                self.buffer[0] = (tokens[take:], state)
            else:
                # Document fully consumed — O(1) removal.
                self.buffer.popleft()

        # Async host-to-device transfer.
        self.gpu.copy_(self.cpu, non_blocking=(self.device.type == "cuda"))

        data = self.gpu.view(B, T + 1)
        self.inputs.copy_(data[:, :-1])
        self.targets.copy_(data[:, 1:])

        return self.inputs, self.targets, self.last_state

# just to compare
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

from gpt_lab.utils.distributed import get_dist_info
from gpt_lab.data.sharder import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size, base_path=None):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    dist_info = get_dist_info()
    ddp_rank = dist_info["RANK"]
    ddp_world_size = dist_info["WORLD_SIZE"]

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(base_path)
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
    buffer_size=1000, base_path=None,
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
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size, base_path=base_path)
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


def build_dataloader(
    name: str,
    batch_size: int,
    seq_len: int,
    tokenizer: Optional[Callable] = None,
    column: str = "text",
    split: str = "train",
    base_url: Optional[str] = None,
    shard_limit: Optional[int] = None,
    max_shards: Optional[int] = None,
    buffer_size: Optional[int] = None,
    cachedir: Optional[Union[str, Path]] = None,
    start_state: Optional[ShardIterationState] = None,
    dist_info: Optional[dict] = None,
    use_nanochat: bool = False,
) -> DistDataLoader:
    if dist_info is None:
        dist_info = get_dist_info()
    if use_nanochat:
        # This is the original dataloader from nanochat, from which I derived the pipeline for gpt-lab.
        # So it is based on the same underlying data loading and on-the-fly tokenization
        dataloader = lambda: tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer=tokenizer,
            B=batch_size,
            T=seq_len,
            split=split,
            tokenizer_threads=4,
            tokenizer_batch_size=128,
            device=dist_info["DEVICE"],
            resume_state_dict=start_state._asdict() if start_state is not None else None,
            buffer_size=buffer_size or (batch_size * seq_len * 16),
            base_path=DATA_DIR / name
        )
    else:
        ds = ShardedDataset(
            name=name,
            tokenizer=tokenizer,
            split=split,
            column=column,
            base_url=base_url,
            shard_limit=shard_limit,
            max_shards=max_shards,
            cachedir=cachedir,
            start_state=start_state,
            dist_info=dist_info,
        )
        dataloader = lambda: DistDataLoader(
            ds,
            batch_size=batch_size,
            seq_len=seq_len,
            device=dist_info["DEVICE"],
            buffer_size=buffer_size,
        )
    if split == "val":
        return dataloader
    else:
        return dataloader()