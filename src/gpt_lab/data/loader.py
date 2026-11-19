from array import array
from bisect import bisect_right, insort
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple, Union

import torch

from gpt_lab.utils.default import DATA_DIR
from gpt_lab.utils.distributed import ComputeContext, get_comp_info
from gpt_lab.utils.schemas import DataLoaderState
from gpt_lab.data.sharder import ShardManager

_TRAIN_LOOKAHEAD_DOCS = 128

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardedDataset:
    """
    Streams tokenized documents from sharded parquet files.

    Each document is tokenized on the fly and yielded alongside its shard state.
    Validation uses tensors and training keeps token lists for the checkpoint.
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
        comp_info: Optional[ComputeContext] = None,
        tokenizer_threads: int = 4,
        data_seed: Optional[int] = None,
    ):
        if isinstance(cachedir, str):
            cachedir = Path(cachedir)
        if comp_info is None:
            comp_info = get_comp_info()

        self.sm = ShardManager(
            name=name,
            cachedir=cachedir,
            split=split,
            base_url=base_url or "",
            column_name=column,
            shard_limit=shard_limit,
            max_shards=max_shards,
            comp_info=comp_info,
            data_seed=data_seed,
            start_state=start_state,
        )
        self.tokenizer = tokenizer
        self.split = split
        self.start_state = start_state
        self.tokenizer_threads = tokenizer_threads

    def __iter__(self) -> Iterator[Tuple[Union[torch.Tensor, list[int]], DataLoaderState]]:
        batch_size = 1 if self.split == "train" else 128
        bos = self.tokenizer.get_bos_token_id() if self.split == "train" else None
        for texts, state in self.sm.iterate(start_state=self.start_state, batch_size=batch_size):
            for txt in texts:
                tokens = (
                    self.tokenizer(txt, prepend_bos=True, threads=self.tokenizer_threads)
                    if self.tokenizer is not None
                    else txt
                )
                if bos is not None:
                    if isinstance(tokens, torch.Tensor):
                        if tokens.ndim != 1:
                            raise ValueError("Source document must be a 1-D token sequence")
                        tokens = tokens.tolist()
                    if not len(tokens) or tokens[0] != bos:
                        raise ValueError("Source document must begin with BOS")
                elif not isinstance(tokens, torch.Tensor):
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
    * Training uses a bounded best-fit lookahead. Checkpoints retain the
      buffer in read order, the pending suffix, and the next raw cursor.

    * The validation buffer is a deque sized in tokens.

    * Documents are separated by an explicit BOS token so the model always
      sees clean boundaries: …doc_N_last [BOS] doc_N+1_first…
      Training rows also start with BOS when continuing a document.

    * Training keeps an offset into the current document even when the
      document spans multiple batches.

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
        self.initial_state = None
        if getattr(dataset, "split", None) == "train":
            self.bos_token_id = dataset.tokenizer.get_bos_token_id()
            sm = dataset.sm
            shape = dict(
                packing_batch_size=batch_size, packing_seq_len=seq_len,
                packing_bos_token_id=self.bos_token_id,
                packing_lookahead_size=_TRAIN_LOOKAHEAD_DOCS,
            )
            self.initial_state = dataset.start_state
            if self.initial_state is None:
                self.initial_state = DataLoaderState(
                    cursor_version=1, packing_state_version=2,
                    manifest_id=sm.manifest_id, ordered_shard_ids=sm.ordered_shard_ids,
                    data_seed=sm.data_seed, packing_rank=sm.ddp_rank, packing_world_size=sm.world_size,
                    **shape,
                )
            else:
                state = self.initial_state
                if any(getattr(state, name) != value for name, value in shape.items()):
                    raise ValueError("Incompatible packing configuration in training data checkpoint")
                pending = state.pending_document
                offset = state.document_token_offset
                if (state.continuation_pending != (pending is not None)
                        or (pending is None and offset != 0)
                        or (pending is not None and not (0 < offset < len(pending) and pending[0] == self.bos_token_id))):
                    raise ValueError("Incompatible continuation state in training data checkpoint")
                if (len(state.lookahead_documents) > _TRAIN_LOOKAHEAD_DOCS
                        or any(not doc or doc[0] != self.bos_token_id for doc in state.lookahead_documents)):
                    raise ValueError("Invalid lookahead buffer in training data checkpoint")
                if not (0 <= state.shard_idx <= len(sm.shard_idx) and
                        min(state.row_group_idx, state.offset_in_row_group) >= 0):
                    raise ValueError("Invalid next-unread training data cursor")
            self._source_cursor = self.initial_state
            self._document_offset = self.initial_state.document_token_offset
            self._pending_document = self.initial_state.pending_document
            self._document_tensor = (
                torch.frombuffer(array("q", self._pending_document), dtype=torch.long)
                if self._pending_document is not None else None
            )
            self._lookahead: list[tuple[int, int, list[int]]] = []
            self._read_index = 0
            for document in self.initial_state.lookahead_documents:
                self._buffer_document(document)
            self._source_exhausted = False
            self._exhausted = False
            self._training_counts = Counter()
            self._io_offsets = {}

        # Pre-fetch budget for the validation stream.
        self.token_buffer_size = buffer_size or (batch_size * seq_len * 16)

        self.iterator = iter(dataset)
        self.buffer: deque[Tuple[torch.Tensor, DataLoaderState]] = deque()
        self._buffered_tokens: int = 0
        self.last_state: Optional[DataLoaderState] = None

        total = batch_size * (seq_len + 1)

        # Pinned CPU staging reused after each blocking device copy.
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
                break

            self.buffer.append((tokens, state))
            self._buffered_tokens += len(tokens)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, DataLoaderState]:
        if self.initial_state is not None:
            return self._next_best_fit()

        self._refill()
        B, T = self.B, self.T
        total = B * (T + 1)
        pos = 0
        while pos < total:
            if not self.buffer:
                self._refill()
                if not self.buffer:
                    raise RuntimeError("DistDataLoader buffer is empty mid-batch")
            tokens, self.last_state = self.buffer[0]
            take = min(len(tokens), total - pos)
            self.cpu[pos : pos + take].copy_(tokens[:take])
            pos += take
            self._buffered_tokens -= take
            if take < len(tokens):
                self.buffer[0] = (tokens[take:], self.last_state)
            else:
                self.buffer.popleft()
        return self._output(self.last_state)

    def _buffer_document(self, document: list[int]) -> None:
        # The rightmost entry of a given length is the earliest to be read.
        insort(self._lookahead, (len(document), -self._read_index, document))
        self._read_index += 1

    def _take_best_fit_document(self, space: int) -> Optional[list[int]]:
        while len(self._lookahead) < _TRAIN_LOOKAHEAD_DOCS and not self._source_exhausted:
            result = next(self.iterator, None)
            if result is None:
                self._source_exhausted = True
                break
            document, self._source_cursor = result
            self._buffer_document(document)
        if not self._lookahead:
            return None
        index = bisect_right(self._lookahead, (space, 1)) - 1
        if index < 0:
            shortest = self._lookahead[0][0]
            index = bisect_right(self._lookahead, (shortest, 1)) - 1
        return self._lookahead.pop(index)[2]

    def allocate_training_buffers(self, count):
        return [(torch.empty_like(self.inputs), torch.empty_like(self.targets)) for _ in range(count)]

    def next_full_batch(self, output):
        """Fill a caller-owned training slot, or return None on exhaustion."""
        try:
            return self._next_best_fit(output)
        except StopIteration:
            return None

    def record_training_update(self, states):
        for state in states:
            self._training_counts.update(
                target_positions=self.B * self.T,
                bos_starting_rows=self.B,
                source_content_tokens=state.source_content_tokens,
                document_bos_tokens=state.document_bos_tokens,
                continuation_bos_tokens=state.continuation_bos_tokens,
            )

    def take_training_metrics(self):
        metrics = dict(self._training_counts, discarded_content_tokens=0)
        self._training_counts.clear()
        for name in ("opened_shards", "downloaded_shards", "read_retries", "download_retries"):
            events = getattr(self.dataset.sm, name)
            metrics[name] = events[self._io_offsets.get(name, 0):]
            self._io_offsets[name] = len(events)
        return metrics

    def _next_best_fit(self, output=None):
        if self._exhausted:
            raise StopIteration
        capacity = self.T + 1
        rows = self.cpu.view(self.B, capacity)
        rows[:, 0].fill_(self.bos_token_id)
        content = document_bos = continuation_bos = padding = 0
        # The final target already consumed its token before the continuation.
        for row_idx, row in enumerate(rows):
            pos = 0
            if self._document_offset:
                continuation_bos += 1
                pos = 1
            while pos < capacity:
                if self._pending_document is None:
                    document = self._take_best_fit_document(capacity - pos)
                    if document is None:
                        self._exhausted = True
                        if row_idx == 0 and pos == 0:
                            raise StopIteration
                        row[pos:].fill_(-1)
                        rows[row_idx + 1:].fill_(-1)
                        padding = (self.B - row_idx - 1) * self.T + self.T - max(0, pos - 1)
                        break
                    self._pending_document = document
                    self._document_tensor = torch.frombuffer(array("q", document), dtype=torch.long)
                offset = self._document_offset
                length = len(self._pending_document)
                take = min(length - offset, capacity - pos)
                row[pos:pos + take].copy_(self._document_tensor[offset:offset + take])
                pos += take
                document_bos += offset == 0
                content += take - (offset == 0)
                self._document_offset += take
                if self._document_offset == length:
                    self._pending_document = None
                    self._document_tensor = None
                    self._document_offset = 0
            if self._exhausted:
                break

        if output is not None and padding:
            return None

        cursor = self._source_cursor
        self.last_state = self.initial_state.model_copy(update=dict(
            shard_idx=cursor.shard_idx, global_shard_idx=cursor.global_shard_idx,
            row_group_idx=cursor.row_group_idx, offset_in_row_group=cursor.offset_in_row_group,
            lookahead_documents=tuple(entry[2] for entry in sorted(self._lookahead, key=lambda entry: -entry[1])),
            pending_document=self._pending_document,
            document_token_offset=self._document_offset, continuation_pending=self._document_offset > 0,
            source_content_tokens=content, document_bos_tokens=document_bos,
            continuation_bos_tokens=continuation_bos, padding_positions=padding,
        ))
        return self._output(self.last_state, output)

    def _output(self, state, output=None):
        # Blocking host-to-device transfer.
        self.gpu.copy_(self.cpu, non_blocking=False)
        data = self.gpu.view(self.B, self.T + 1)
        inputs, targets = (self.inputs, self.targets) if output is None else output
        inputs.copy_(data[:, :-1])
        targets.copy_(data[:, 1:])
        if state is not None and state.padding_positions:
            inputs.masked_fill_(inputs == -1, self.bos_token_id)
        return inputs, targets, state

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

from gpt_lab.utils.distributed import get_comp_info
from gpt_lab.data.sharder import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size, base_path=None):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    comp_info = get_comp_info()
    ddp_rank = comp_info.rank
    ddp_world_size = comp_info.world_size

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
    comp_info: Optional[ComputeContext] = None,
    tokenizer_threads: int = 4,
    data_seed: Optional[int] = None,
) -> DistDataLoader:
    if comp_info is None:
        comp_info = get_comp_info()
    ds = ShardedDataset(
        name=name, tokenizer=tokenizer, split=split, column=column, base_url=base_url,
        shard_limit=shard_limit, max_shards=max_shards, cachedir=cachedir or datadir or DATA_DIR,
        start_state=resume_state, comp_info=comp_info, tokenizer_threads=tokenizer_threads,
        data_seed=data_seed,
    )

    def dataloader():
        return DistDataLoader(
            ds,
            batch_size=batch_size,
            seq_len=seq_len,
            device=comp_info.device,
            buffer_size=buffer_size,
        )
    if split == "val":
        return dataloader
    return dataloader()
