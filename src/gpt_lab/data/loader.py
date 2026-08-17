from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple, Union

import torch

from gpt_lab.utils.default import DATA_DIR
from gpt_lab.utils.distributed import get_dist_info
from gpt_lab.utils.schemas import DataLoaderState
from gpt_lab.utils.types import PackingStrategy
from gpt_lab.data.sharder import ShardManager


@dataclass
class PackingStats:
    """Optional aggregate packing counters; omitted in normal training.

    ``source_tokens_read`` counts tokenized source tokens entering the packer,
    including document BOS tokens. ``new_source_tokens_advanced`` counts unique
    source-stream tokens permanently advanced, excluding a retained FIFO carry.
    Cropped tokens are source tokens advanced without being emitted. Skipped
    transitions include destructive suffix loss and deliberate row segmentation
    at an original BOS. Synthetic BOS tokens are never source tokens.
    """

    destructive_cropped_tokens: int = 0
    source_tokens_read: int = 0
    new_source_tokens_advanced: int = 0
    skipped_adjacent_transitions: int = 0
    synthetic_bos_tokens_inserted: int = 0
    intentional_bos_boundaries: int = 0
    buffered_source_tokens: int = 0
    source_tokens_emitted: int = 0
    source_bos_tokens_emitted: int = 0
    rows: int = 0
    batches: int = 0
    debug_discarded_suffixes: Optional[list[tuple[int, ...]]] = None

    def reset(self) -> None:
        self.destructive_cropped_tokens = 0
        self.source_tokens_read = 0
        self.new_source_tokens_advanced = 0
        self.skipped_adjacent_transitions = 0
        self.synthetic_bos_tokens_inserted = 0
        self.intentional_bos_boundaries = 0
        self.buffered_source_tokens = 0
        self.source_tokens_emitted = 0
        self.source_bos_tokens_emitted = 0
        self.rows = 0
        self.batches = 0
        if self.debug_discarded_suffixes is not None:
            self.debug_discarded_suffixes.clear()

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
        start_state: Optional[DataLoaderState] = None,
        base_url: Optional[str] = None,
        column: str = "text",
        shard_limit: Optional[int] = None,
        max_shards: Optional[int] = None,
        cachedir: Union[str, Path] = DATA_DIR,
        dist_info: Optional[dict] = None,
        tokenizer_threads: int = 4,
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
            dist_info=dist_info
        )
        self.tokenizer = tokenizer
        self.split = split
        self.start_state = start_state
        self.tokenizer_threads = tokenizer_threads

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, DataLoaderState]]:
        # One document per source state makes checkpoint resume exact: no
        # untracked documents remain suspended inside this generator.
        for texts, state in self.sm.iterate(start_state=self.start_state, batch_size=1):
            for txt in texts:
                tokens = (
                    self.tokenizer(txt, prepend_bos=True, threads=self.tokenizer_threads)
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

    ``stream`` (the default) is a corrected flat B*T+1 stream with one-token
    carry between batches. ``bos_aligned`` starts every row with BOS and keeps
    every continuation suffix, inserting a synthetic BOS before it.

    BOS alignment does not isolate documents in attention. Documents packed in
    the same row can attend to earlier documents unless the model is separately
    given a segment-aware attention mask.
    """

    def __init__(
        self,
        dataset: ShardedDataset,
        batch_size: int,
        seq_len: int,
        device: str = "cuda",
        buffer_size: Optional[int] = None,
        packing_strategy: PackingStrategy = "stream",
        bos_token_id: Optional[int] = None,
        resume_state: Optional[DataLoaderState] = None,
        packing_stats: Optional[PackingStats] = None,
    ):
        if packing_strategy not in ("stream", "bos_aligned"):
            raise ValueError(f"DistDataLoader does not implement {packing_strategy!r}")
        if batch_size < 1 or seq_len < 1:
            raise ValueError("batch_size and seq_len must be positive")
        if packing_strategy == "bos_aligned" and bos_token_id is None:
            raise ValueError("bos_aligned packing requires bos_token_id")
        self.dataset = dataset
        self.B = batch_size
        self.T = seq_len
        self.device = torch.device(device)
        self.packing_strategy = packing_strategy
        self.bos_token_id = bos_token_id
        self.packing_stats = packing_stats or PackingStats()
        self.last_batch_stats = PackingStats()

        # Retained for API compatibility. On-demand reads make a checkpoint
        # exact without serializing a multi-batch prefetch queue.
        self.token_buffer_size = buffer_size or (batch_size * seq_len * 16)

        self.iterator = iter(dataset)
        self.buffer: deque[Tuple[torch.Tensor, Optional[DataLoaderState], bool]] = deque()
        self.last_state: Optional[DataLoaderState] = None
        self._carry_token: Optional[int] = None
        self._continuation_pending = False
        self._finished = False

        state = resume_state or getattr(dataset, "start_state", None)
        if state is not None and state.packing_strategy is not None:
            if state.packing_strategy != packing_strategy:
                raise ValueError(
                    f"Cannot resume {state.packing_strategy!r} state with {packing_strategy!r} packing"
                )
            self.last_state = state
            if state.pending_tokens:
                self.buffer.append((
                    torch.tensor(state.pending_tokens, dtype=torch.long),
                    state,
                    state.pending_is_document_start,
                ))
            self._carry_token = state.carry_token
            self._continuation_pending = state.continuation_pending
            saved = {
                key: value for key, value in state.packing_stats.items()
                if key in PackingStats.__dataclass_fields__
            }
            self.packing_stats = packing_stats or PackingStats(**saved)

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
        self.inputs  = _out[:batch_size * seq_len].view(batch_size, seq_len)
        self.targets = _out[batch_size * seq_len:].view(batch_size, seq_len)

    def _pull_document(self) -> bool:
        if self.buffer:
            return True
        try:
            tokens, state = next(self.iterator)
        except StopIteration:
            return False
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        self.last_state = state
        self.buffer.append((tokens, state, True))
        self.packing_stats.source_tokens_read += len(tokens)
        self.packing_stats.buffered_source_tokens += len(tokens)
        self.last_batch_stats.source_tokens_read += len(tokens)
        return True

    def _copy_source(self, destination: torch.Tensor, limit: int) -> int:
        if not self._pull_document():
            return 0
        tokens, state, is_start = self.buffer[0]
        take = min(len(tokens), limit)
        destination[:take].copy_(tokens[:take])
        for stats in (self.packing_stats, self.last_batch_stats):
            stats.source_tokens_emitted += take
            stats.new_source_tokens_advanced += take
            if is_start and take:
                stats.source_bos_tokens_emitted += 1
        self.packing_stats.buffered_source_tokens -= take
        if take == len(tokens):
            self.buffer.popleft()
        else:
            self.buffer[0] = (tokens[take:], state, False)
        return take

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, DataLoaderState]:
        if self._finished:
            raise StopIteration
        self.last_batch_stats = PackingStats()
        if self.packing_strategy == "stream":
            self._next_stream()
        else:
            self._next_bos_aligned()
        for stats in (self.packing_stats, self.last_batch_stats):
            stats.batches += 1
            stats.rows += self.B
        return self.inputs, self.targets, self._state()

    def _next_stream(self) -> None:
        total = self.B * self.T + 1
        pos = 0
        had_carry = self._carry_token is not None
        if had_carry:
            self.packing_stats.buffered_source_tokens -= 1
            self.cpu[0] = self._carry_token
            pos = 1
        while pos < total:
            take = self._copy_source(self.cpu[pos:total], total - pos)
            if not take:
                break
            pos += take
        if pos < total:
            self._finished = True
            raise StopIteration
        if not had_carry:
            # The first source token seeds the flat window; only the following
            # B*T tokens advance the stream and contribute target positions.
            for stats in (self.packing_stats, self.last_batch_stats):
                stats.new_source_tokens_advanced -= 1
        self.gpu[:total].copy_(self.cpu[:total], non_blocking=(self.device.type == "cuda"))
        self.inputs.copy_(self.gpu[:total - 1].view(self.B, self.T))
        self.targets.copy_(self.gpu[1:total].view(self.B, self.T))
        self._carry_token = int(self.cpu[total - 1])
        self.packing_stats.buffered_source_tokens += 1

    def _next_bos_aligned(self) -> None:
        B, capacity = self.B, self.T + 1
        rows = self.cpu[:B * capacity].view(B, capacity)
        for row_index in range(B):
            pos = 0
            if self._continuation_pending:
                rows[row_index, 0] = self.bos_token_id
                for stats in (self.packing_stats, self.last_batch_stats):
                    stats.synthetic_bos_tokens_inserted += 1
                    stats.skipped_adjacent_transitions += 1
                self._continuation_pending = False
                pos = 1
            while pos < capacity:
                if not self._pull_document():
                    self._finished = True
                    raise StopIteration
                tokens, _, is_start = self.buffer[0]
                if is_start and (not len(tokens) or int(tokens[0]) != self.bos_token_id):
                    raise ValueError("bos_aligned requires every source document to begin with BOS")
                if pos == 0 and is_start:
                    for stats in (self.packing_stats, self.last_batch_stats):
                        stats.intentional_bos_boundaries += 1
                        stats.skipped_adjacent_transitions += 1
                take = self._copy_source(rows[row_index, pos:capacity], capacity - pos)
                pos += take
                if self.buffer and pos == capacity:
                    self._continuation_pending = True
        self.gpu.copy_(self.cpu, non_blocking=(self.device.type == "cuda"))
        data = self.gpu.view(B, capacity)
        self.inputs.copy_(data[:, :-1])
        self.targets.copy_(data[:, 1:])

    def _state(self) -> DataLoaderState:
        base = self.last_state or DataLoaderState()
        pending_tokens: list[int] = []
        pending_is_start = False
        if self.buffer:
            tokens, _, pending_is_start = self.buffer[0]
            pending_tokens = tokens.tolist()
        values = base.model_dump(exclude={
            "packing_strategy", "pending_tokens", "pending_is_document_start",
            "carry_token", "continuation_pending", "packing_stats",
        })
        return DataLoaderState(
            **values,
            packing_strategy=self.packing_strategy,
            pending_tokens=pending_tokens,
            pending_is_document_start=pending_is_start,
            carry_token=self._carry_token,
            continuation_pending=self._continuation_pending,
            packing_stats=asdict(self.packing_stats),
        )

# just to compare
"""
Distributed dataloaders for pretraining.

BOS-aligned destructive best-fit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

This legacy strategy may lose substantial source content to cropping. Starting
a row with BOS does not isolate documents packed later in that row: causal
attention still crosses document boundaries without segment-aware masks.

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
    packing_stats: Optional[PackingStats] = None,
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
            if packing_stats is not None:
                packing_stats.source_tokens_read += len(tokens)
                packing_stats.buffered_source_tokens += len(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = torch.device(device).type == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            if packing_stats is not None:
                # The transition into this row's original BOS is deliberately
                # segmented, rather than lost accidentally or made synthetic.
                packing_stats.intentional_bos_boundaries += 1
                packing_stats.skipped_adjacent_transitions += 1
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
                    take = min(len(doc), remaining)
                    row_buffer[row_idx, pos:pos + take] = torch.tensor(doc[:take], dtype=torch.long)
                    if packing_stats is not None:
                        discarded = len(doc) - take
                        packing_stats.destructive_cropped_tokens += discarded
                        packing_stats.skipped_adjacent_transitions += discarded
                        if packing_stats.debug_discarded_suffixes is not None and discarded:
                            packing_stats.debug_discarded_suffixes.append(tuple(doc[take:]))
                    pos += take

                if packing_stats is not None:
                    packing_stats.new_source_tokens_advanced += len(doc)
                    packing_stats.buffered_source_tokens -= len(doc)

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        # state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        # replaced the line above with the line below to return a state object consistent with the rest of the codebase
        # offset in row group is not tracked in nanochat dataloader
        state_dict = DataLoaderState(shard_idx=pq_idx, row_group_idx=rg_idx, epoch=epoch)

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
    datadir: Optional[Union[str, Path]] = None,
    resume_state: Optional[DataLoaderState] = None,
    dist_info: Optional[dict] = None,
    tokenizer_threads: int = 4,
    packing_strategy: PackingStrategy = "stream",
    bos_token_id: Optional[int] = None,
    use_nanochat: Optional[bool] = None,
    packing_stats: Optional[PackingStats] = None,
) -> DistDataLoader:
    if dist_info is None:
        dist_info = get_dist_info()
    if use_nanochat:
        if packing_strategy != "stream":
            raise ValueError("Use packing_strategy instead of combining it with use_nanochat")
        packing_strategy = "bos_bestfit_crop"
    if packing_strategy == "bos_bestfit_crop":
        # This is the original dataloader from nanochat, from which I derived the pipeline for gpt-lab.
        # So it is based on the same underlying data loading and on-the-fly tokenization
        if resume_state is not None:
            resume_state_dict = dict(
                pq_idx=resume_state.shard_idx,
                rg_idx=resume_state.row_group_idx,
                epoch=resume_state.epoch
            )
        else:
            resume_state_dict = None
        dataloader = lambda: tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer=tokenizer,
            B=batch_size,
            T=seq_len,
            split=split,
            tokenizer_threads=tokenizer_threads,
            tokenizer_batch_size=128,
            device=dist_info["DEVICE"],
            resume_state_dict=resume_state_dict,
            buffer_size=buffer_size or 1000,
            base_path=(Path(datadir) if datadir is not None else DATA_DIR) / name,
            packing_stats=packing_stats,
        )
    elif packing_strategy in ("stream", "bos_aligned"):
        if bos_token_id is None and tokenizer is not None and hasattr(tokenizer, "get_bos_token_id"):
            bos_token_id = tokenizer.get_bos_token_id()
        ds = ShardedDataset(
            name=name,
            tokenizer=tokenizer,
            split=split,
            column=column,
            base_url=base_url,
            shard_limit=shard_limit,
            max_shards=max_shards,
            cachedir=cachedir or DATA_DIR,
            start_state=resume_state,
            dist_info=dist_info,
            tokenizer_threads=tokenizer_threads,
        )
        dataloader = lambda: DistDataLoader(
            ds,
            batch_size=batch_size,
            seq_len=seq_len,
            device=dist_info["DEVICE"],
            buffer_size=buffer_size,
            packing_strategy=packing_strategy,
            bos_token_id=bos_token_id,
            resume_state=resume_state,
            packing_stats=packing_stats,
        )
    else:
        raise ValueError(f"Unknown packing strategy: {packing_strategy!r}")
    if split == "val":
        return dataloader
    else:
        return dataloader()
