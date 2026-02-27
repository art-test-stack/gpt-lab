import os
import random
import shutil
from typing import Iterator, Tuple, Iterable, Dict, Union, Callable
from pathlib import Path
from datasets import load_dataset, Dataset, get_dataset_split_names
from transformers import AutoTokenizer
from torch.utils.data import DataLoader as TorchDataLoader
from gpt_lib.utils.default import DATA_DIR


def load_datasets(
         # { "path": str, "name": str (optional), "weight": float (optional), "hook": Callable (optional) } 
        sources: Iterable[Dict[str, Union[str, float, Callable]]],
        data_dir: Union[str,Path] = DATA_DIR,
        split: str = "train",
        streaming: bool = True,
        *args, **kwargs
    ) -> Dict[str, Iterable]:
    ds = dict()
    for src in sources:
        ds_name = src["path"]
        ds_split = src.get("split", split)
        ds_hook = src.get("hook", lambda x: x)
        ds[ds_name] = ds_hook(
            load_dataset(
                ds_name, 
                name=src.get("name", None),
                split=ds_split,
                streaming=streaming, 
                cache_dir=data_dir, 
                *args, **kwargs
            )
        )
        if "filter_fn" in src:
            ds[ds_name] = ds[ds_name].filter(src["filter_fn"])
    return ds

def weighted_sample_generator(streams, prng):
    """
    streams: list of (iterable, weight)
    yields items from one of the streams according to weights.
    Designed to keep reading from selected stream until exhausted (streams are long)
    For streaming HF datasets these are effectively infinite for training; we just sample.
    """
    # convert weights to cumulative thresholds
    total = sum(w for _, w in streams)
    cum = []
    acc = 0.0
    for _, w in streams:
        acc += w / total
        cum.append(acc)

    # create iterators
    iterators = [iter(s) for s, _ in streams]
    while True:
        p = prng.random()
        # pick which stream index
        idx = 0
        while p > cum[idx]:
            idx += 1
        try:
            yield next(iterators[idx])
        except StopIteration:
            # If a stream ends, remove it from selection.
            # For HF streaming this is unlikely; but handle gracefully.
            iterators.pop(idx)
            streams.pop(idx)
            cum = []
            total = sum(w for _, w in streams) if streams else 0
            acc = 0.0
            for _, w in streams:
                acc += w / total
                cum.append(acc)
            if not streams:
                break
    
def list_parquet_files(data_dir: Union[str, Path] = DATA_DIR) -> list:
    data_dir = Path(data_dir)
    parquet_files = []
    for split in ["train", "validation"]:
        split_dir = data_dir / split
        if split_dir.exists():
            parquet_files.extend(list(split_dir.glob("*.bin")))
    return parquet_files