from gpt_lib.utils.default import DATA_DIR
from pathlib import Path
from typing import List, Optional, Callable, Iterator, Union

import torch
import pyarrow.parquet as pq


def list_shards(data_dir: Path, limit: Optional[int] = None) -> List[Path]:
    shards = sorted(data_dir.glob("*.parquet"))
    if limit is not None:
        shards = shards[:limit]
    if not shards:
        raise RuntimeError(f"No shards found in {data_dir}")
    return shards


class StreamingParquetDataset:
    def __init__(
        self,
        dsname: str,
        column: str = "text",
        split: str = "train",
        tokenizer: Optional[Callable] = None,
        shard_limit: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        cachedir: Optional[Union[str, Path]] = None,
    ):
        if cachedir is None:
            cachedir = DATA_DIR 
        if isinstance(cachedir, str):
            cachedir = Path(cachedir)
        datadir = cachedir / dsname 
        all_shards = list_shards(Path(datadir), shard_limit)

        # Shard-level DDP split: each rank owns a disjdoint slice of files.
        # Avoids reading the whole dataset on every rank.
        # Falls back to doc-level interleaving if there are fewer shards than ranks.
        if len(all_shards) >= world_size:
            self.paths = all_shards[rank::world_size]
        else:
            # too few shards to split — keep all and interleave at doc level
            self.paths = all_shards
            self._doc_rank = rank
            self._doc_world_size = world_size

        self.column = column
        self.tokenizer = tokenizer
        self._shard_split = len(all_shards) >= world_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        doc_idx = 0
        for path in self.paths:
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx, columns=[self.column])
                data = rg.column(self.column)
                for i in range(len(data)):
                    # doc-level interleave only used when shards < world_size
                    if not self._shard_split:
                        if doc_idx % self._doc_world_size != self._doc_rank:
                            doc_idx += 1
                            continue
                    doc_idx += 1

                    x = data[i].as_py()

                    if self.tokenizer is not None:
                        x = self.tokenizer(x, prepend_bos=True)
                    yield torch.tensor(x, dtype=torch.long)


class DistDataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int,
        seq_len: int,
        device: str = "cuda",
        buffer_size: int = 512,
        min_buffer: int = 64,
    ):
        self.dataset = dataset
        self.B = batch_size
        self.T = seq_len
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.min_buffer = min_buffer  # refill threshold, not just at batch start

        self.buffer: list[torch.Tensor] = []
        self.iterator = iter(dataset)

        use_pinned = self.device.type == "cuda"
        self.cpu = torch.empty((batch_size, seq_len + 1), dtype=torch.long,
                                pin_memory=use_pinned)
        self.gpu = torch.empty((batch_size, seq_len + 1), dtype=torch.long,
                                device=self.device)

    def _refill(self):
        while len(self.buffer) < self.buffer_size:
            try:
                self.buffer.append(next(self.iterator))
            except StopIteration:
                if self.dataset.split == "val":
                    # val dataset loops itself, so this should never happen —
                    # but guard anyway
                    self.iterator = iter(self.dataset)
                else:
                    return

    def __iter__(self):
        return self

    def __next__(self):
        B, T = self.B, self.T
        for b in range(B):
            pos = 0
            while pos < T + 1:
                # Refill inside the loop so the buffer never runs dry mid-batch
                if len(self.buffer) < self.min_buffer:
                    self._refill()

                doc = self.buffer.pop()
                remaining = (T + 1) - pos
                take = min(len(doc), remaining)
                self.cpu[b, pos:pos + take] = doc[:take]

                if take < len(doc):
                    self.buffer.append(doc[take:])  # remainder back into buffer

                pos += take

        self.gpu.copy_(self.cpu, non_blocking=self.device.type == "cuda")
        return self.gpu[:, :-1], self.gpu[:, 1:], None


def build_dataloader(
    dsname: str,
    batch_size: int,
    seq_len: int,
    tokenizer: Optional[Callable] = None,
    split: str = "train",
    shard_limit: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    device: str = "cuda",
    buffer_size: Optional[int] = None,
    cachedir: Optional[Union[str, Path]] = None,
) -> DistDataLoader:
    
    if buffer_size is None:
        buffer_size = batch_size * seq_len * 16  # heuristic default
    ds = StreamingParquetDataset(
        dsname=dsname,
        tokenizer=tokenizer,
        split=split,
        shard_limit=shard_limit,
        rank=rank,
        world_size=world_size,
        cachedir=cachedir,
    )
    return DistDataLoader(
        ds,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        buffer_size=buffer_size,
    )