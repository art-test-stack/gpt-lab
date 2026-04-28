from gpt_lab.utils.default import DATA_DIR
from gpt_lab.utils.distributed import get_dist_info
from gpt_lab.data.sharder import ShardManager, ShardIterationState
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

class ShardedDataset:
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
        if cachedir is None:
            cachedir = DATA_DIR
        if isinstance(cachedir, str):
            cachedir = Path(cachedir)

        if not dist_info:
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

    def __iter__(self) -> Iterator[torch.Tensor]:
        iterator = self.sm.iterate(
            start_state=self.start_state,
        )
        for texts, state in iterator:
            # texts = list[str] from a row group
            for txt in texts:
                if self.tokenizer is not None:
                    tokens = self.tokenizer(txt, prepend_bos=True)
                else:
                    tokens = txt # assume already tokenized as list[int]
                yield tokens, state

class DistDataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int,
        seq_len: int,
        device: str = "cuda",
        buffer_size: int = 8192,
    ):
        self.dataset = dataset
        self.B = batch_size
        self.T = seq_len
        self.device = torch.device(device)

        self.iterator = iter(dataset)
        self.buffer = []
        total_tokens = batch_size * (seq_len + 1)

        self.cpu = torch.empty(total_tokens, dtype=torch.long, pin_memory=(self.device.type == "cuda"))
        self.gpu = torch.empty(total_tokens, dtype=torch.long, device=self.device)
        self.buffer_size = buffer_size
        self.gpu_buffer = torch.empty(2 * self.B * self.T, device=self.device, dtype=torch.long)
        self.inputs = self.gpu_buffer[:self.B*self.T].view(self.B, self.T)
        self.targets = self.gpu_buffer[self.B*self.T:].view(self.B, self.T)
        self.last_state = None

    def _refill(self):
        while len(self.buffer) < self.buffer_size:
            try:
                self.buffer.append(next(self.iterator))
            except StopIteration:
                if getattr(self.dataset, "split", None) == "val":
                    self.iterator = iter(self.dataset)
                    continue
                else:
                    return

    def __iter__(self):
        return self
    
    def __next__(self):
        self._refill()

        B, T = self.B, self.T
        total = B * (T + 1)

        pos = 0
        while pos < total:
            if len(self.buffer) == 0:
                self._refill()

            doc, state = self.buffer.pop(0)
            self.last_state = state

            remaining = total - pos
            take = min(len(doc), remaining)
            self.cpu[pos:pos + take] = torch.tensor(doc[:take], dtype=torch.long)

            if take < len(doc):
                self.buffer.append((doc[take:], state))
            pos += take

        self.gpu.copy_(self.cpu, non_blocking=(self.device.type == "cuda"))

        data = self.gpu.view(B, T + 1)

        self.inputs.copy_(data[:, :-1])
        self.targets.copy_(data[:, 1:])

        return self.inputs, self.targets, self.last_state

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
) -> DistDataLoader:
    
    if buffer_size is None:
        buffer_size = batch_size * seq_len * 16  # heuristic default
    ds = ShardedDataset(
        name=name,
        tokenizer=tokenizer,
        split=split,
        column=column,
        base_url=base_url, # no downloading in this function, just load local shards
        shard_limit=shard_limit,
        max_shards=max_shards,
        cachedir=cachedir,
        start_state=start_state,
        dist_info=dist_info,
    )
    return DistDataLoader(
        ds,
        batch_size=batch_size,
        seq_len=seq_len,
        device=dist_info['DEVICE'],
        buffer_size=buffer_size,
    )
