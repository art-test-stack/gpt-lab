import hashlib
import json
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple, Union

import torch

from gpt_lab.utils.default import DATA_DIR, RANDOM_SEED
from gpt_lab.utils.distributed import ComputeContext, get_comp_info
from gpt_lab.utils.schemas import DataLoaderState, DocumentRef, TokenizerConfig
from gpt_lab.data.sharder import ShardManager

_LOOKAHEAD_DOCUMENTS = 16


class ShardedDataset:
    """Read and tokenize documents alongside the cursor immediately after each one."""

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
        data_seed: int = RANDOM_SEED,
    ):
        if split == "train" and start_state is not None and start_state.ordered_shard_ids is None:
            raise ValueError("Training data checkpoint predates saved shard order and document references")
        self.sm = ShardManager(
            name=name,
            cachedir=cachedir,
            split=split,
            base_url=base_url,
            column_name=column,
            shard_limit=shard_limit,
            max_shards=max_shards,
            comp_info=comp_info,
            data_seed=data_seed,
            ordered_shard_ids=start_state.ordered_shard_ids if start_state is not None else None,
        )
        self.name = name
        self.tokenizer = tokenizer
        self.bos_token_id = getattr(tokenizer, "get_bos_token_id", lambda: None)()
        self.split = split
        self.start_state = start_state
        self.tokenizer_threads = tokenizer_threads

    def tokenize(self, text) -> list[int]:
        tokens = (
            self.tokenizer(text, prepend_bos=True, threads=self.tokenizer_threads)
            if self.tokenizer is not None
            else text
        )
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim != 1:
                raise ValueError("Source document must be a 1-D token sequence")
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            tokens = list(tokens)
        if self.bos_token_id is not None and (not tokens or tokens[0] != self.bos_token_id):
            raise ValueError("Source document must begin with BOS")
        return tokens

    def __iter__(self) -> Iterator[Tuple[list[int], DataLoaderState]]:
        for (text,), cursor in self.sm.iterate(start_state=self.start_state, batch_size=1):
            tokens = self.tokenize(text)
            if tokens:
                yield tokens, cursor


class DistDataLoader:
    """Pack each row without discarding document tails, starting with BOS when available."""

    def __init__(
        self,
        dataset: ShardedDataset,
        batch_size: int,
        seq_len: int,
        device: str = "cuda",
        buffer_size: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        *,
        include_partial: bool = False,
    ):
        if batch_size < 1 or seq_len < 1:
            raise ValueError("Batch size and sequence length must be positive")
        self.B = batch_size
        self.T = seq_len
        self.device = torch.device(device)
        self.include_partial = include_partial
        self.bos_token_id = getattr(dataset, "bos_token_id", getattr(dataset, "bos_id", None))
        self.lookahead_size = _LOOKAHEAD_DOCUMENTS if self.bos_token_id is not None else 1
        split = getattr(dataset, "split", None)
        start_state = getattr(dataset, "start_state", None)

        self.signature = None
        self.ordered_shard_ids = None
        if split == "train":
            sm = dataset.sm
            self.ordered_shard_ids = sm.ordered_shard_ids
            config = getattr(dataset.tokenizer, "config", None)
            tokenizer = type(dataset.tokenizer)
            signature_data = (
                2,  # Checkpoint format and packing algorithm version.
                dataset.name, sm.base_url, sm.column_name, sm.max_shards, sm.train_shards,
                self.ordered_shard_ids, sm.ddp_rank, sm.world_size,
                batch_size, seq_len, self.lookahead_size, self.bos_token_id,
                tokenizer.__module__, tokenizer.__qualname__,
                config.model_dump(mode="json", exclude={"dirname", "trainer", "save_token_bytes"})
                if isinstance(config, TokenizerConfig) else None,
            )
            self.signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()

        self.cursor = start_state or DataLoaderState()
        self.lookahead = []
        self.document = None
        self.document_ref = self.cursor.pending_document_ref
        self.offset = self.cursor.document_token_offset
        if start_state is not None:
            if split == "train" and self.cursor.packing_signature != self.signature:
                raise ValueError("Incompatible training data checkpoint (loader, dataset, topology, tokenizer or batch shape)")
            refs = self.cursor.lookahead_refs
            if self.document_ref is not None:
                refs += (self.document_ref,)
            if len(self.cursor.lookahead_refs) >= self.lookahead_size or len(set(refs)) != len(refs):
                raise ValueError("Invalid packing buffer in data checkpoint")
            documents = {ref: dataset.tokenize(text) for ref, text in dataset.sm.read_documents(refs)}
            self.lookahead = [(documents[ref], ref) for ref in self.cursor.lookahead_refs]
            self.document = documents.get(self.document_ref)
            if (any(not doc for doc in documents.values())
                    or (self.document is None and self.offset != 0)
                    or (self.document is not None and not 0 < self.offset < len(self.document))):
                raise ValueError("Invalid packing buffer in data checkpoint")

        self.document_tensor = torch.tensor(self.document, dtype=torch.long) if self.document is not None else None
        self.iterator = iter(dataset)
        self.next_epoch_document = None
        self.cpu = torch.empty((batch_size, seq_len + 1), dtype=torch.long, pin_memory=self.device.type == "cuda")
        self.gpu = torch.empty((batch_size, seq_len + 1), dtype=torch.long, device=self.device)
        self.inputs = torch.empty((batch_size, seq_len), dtype=torch.long, device=self.device)
        self.targets = torch.empty_like(self.inputs)

    def __iter__(self):
        return self

    def _take_document(self, space: int) -> tuple[list[int] | torch.Tensor, Optional[DocumentRef]]:
        while len(self.lookahead) < self.lookahead_size:
            if self.next_epoch_document is None:
                self.next_epoch_document = next(self.iterator, None)
                if self.next_epoch_document is None:
                    break
            document, cursor = self.next_epoch_document
            # Keep the resume cursor before the new epoch until the old buffer drains.
            if self.lookahead and cursor.epoch != self.cursor.epoch:
                break
            ref = None
            if self.signature is not None:
                ref = (cursor.global_shard_idx, cursor.row_group_idx, cursor.offset_in_row_group - 1)
            self.lookahead.append((document, ref))
            self.cursor = cursor
            self.next_epoch_document = None

        if not self.lookahead:
            raise StopIteration
        best = None
        for index, (document, _) in enumerate(self.lookahead):
            if len(document) <= space and (best is None or len(document) > len(self.lookahead[best][0])):
                best = index
        if best is None:
            # Split the shortest document when none fits, retaining its suffix.
            best = min(range(len(self.lookahead)), key=lambda index: len(self.lookahead[index][0]))
        return self.lookahead.pop(best)

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, DataLoaderState]:
        capacity = self.T + 1
        rows = self.B
        for index, row in enumerate(self.cpu):
            pos = 0
            if self.offset and self.bos_token_id is not None:
                row[0] = self.bos_token_id
                pos = 1

            while pos < capacity:
                if self.document is None:
                    try:
                        self.document, self.document_ref = self._take_document(capacity - pos)
                    except StopIteration:
                        if not self.include_partial:
                            raise
                        break
                    self.document_tensor = torch.as_tensor(self.document, dtype=torch.long)
                take = min(len(self.document) - self.offset, capacity - pos)
                row[pos:pos + take].copy_(self.document_tensor[self.offset:self.offset + take])
                pos += take
                self.offset += take
                if self.offset == len(self.document):
                    self.document = None
                    self.document_ref = None
                    self.document_tensor = None
                    self.offset = 0

            if pos < capacity:
                rows = index + int(pos > 1)
                if not rows:
                    raise StopIteration
                if pos > 1:
                    # Targets use ignore_index; embedding inputs need a valid ID.
                    row[pos:].fill_(-1)
                break

        cpu, gpu, inputs, targets = self.cpu, self.gpu, self.inputs, self.targets
        if rows < self.B:
            cpu, gpu, inputs, targets = cpu[:rows], gpu[:rows], inputs[:rows], targets[:rows]
        gpu.copy_(cpu)
        inputs.copy_(gpu[:, :-1])
        targets.copy_(gpu[:, 1:])
        if 1 < pos < capacity:
            inputs[-1, pos:].zero_()
        state = self.cursor
        if self.signature is not None:
            state = state.model_copy(update=dict(
                packing_signature=self.signature,
                ordered_shard_ids=self.ordered_shard_ids,
                lookahead_refs=tuple(ref for _, ref in self.lookahead),
                pending_document_ref=self.document_ref,
                document_token_offset=self.offset,
            ))
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
    Iterate over Parquet document batches, repeating only for training.

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

    while True:
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
        if split == "val":
            return
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
    source_open = True
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        batch = next(batches, None)
        if batch is None:
            return False
        doc_batch, (pq_idx, rg_idx, epoch) = batch
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens if tokens and tokens[0] == bos_token else [bos_token, *tokens])
        return True

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
                while len(doc_buffer) < buffer_size and source_open:
                    source_open = refill_buffer()
                if not doc_buffer:
                    return

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
    use_nanochat: bool = False,
    data_seed: int = RANDOM_SEED,
) -> DistDataLoader:
    """Build a training iterator or a factory for independent validation iterators.

    The native validation factory accepts include_partial=True to retain its
    final batch with ignored target padding. The buffer_size argument configures
    the nanochat backend.
    """
    if comp_info is None:
        comp_info = get_comp_info()
    if use_nanochat:
        if resume_state is not None and resume_state.packing_signature is not None:
            raise ValueError("Cannot resume a native dataloader checkpoint with nanochat")
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
            device=comp_info.device,
            resume_state_dict=resume_state_dict,
            buffer_size=buffer_size or (batch_size * seq_len * 16),
            base_path=Path(datadir or cachedir or DATA_DIR) / name
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
            cachedir=cachedir or datadir or DATA_DIR,
            start_state=resume_state,
            comp_info=comp_info,
            tokenizer_threads=tokenizer_threads,
            data_seed=data_seed,
        )
        dataloader = lambda *, include_partial=False: DistDataLoader(
            ds,
            batch_size=batch_size,
            seq_len=seq_len,
            device=comp_info.device,
            buffer_size=buffer_size,
            include_partial=include_partial,
        )
    if split == "val":
        return dataloader
    else:
        return dataloader()
