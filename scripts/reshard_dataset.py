"""
Unified resharding script for large-scale datasets.

This script generalizes https://github.com/karpathy/nanochat/blob/master/dev/repackage_data_reference.py into a single pipeline that:
- Downloads datasets from Hugging Face
- Reshards them into Parquet shards
- Writes output to one of:
    • Hugging Face buckets (large-scale storage)
    • Hugging Face dataset repositories
    • Local filesystem (no upload)

Designed for datasets that may not fit in memory, with:
- Streaming and non-streaming support
- Concurrent shard writing and uploading
- Fault-tolerant retry logic

Typical usage with `hf jobs`:

```bash
hf jobs run \
  --secrets HF_TOKEN=$HF_TOKEN \
  --flavor cpu-upgrade \
  --timeout 4d \
  python:3.12 \
  bash -c "
set -e
apt-get update && apt-get install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/art-test-stack/gpt-lab.git
cd gpt-lab
export HF_HUB_DISABLE_PROGRESS_BARS=1
uv sync
```

# Bucket mode (recommended for very large datasets)
`uv run python -m scripts.reshard --mode bucket --repo-id ai-teststack/llm-factory-storage --streaming`

# Dataset repo mode
`uv run python -m scripts.reshard --mode dataset --repo-id ai-teststack/my-dataset`

# Local-only rewrite
`uv run python -m scripts.reshard --mode local`

# Clean bucket memory:

`hf buckets rm $HF_ID/$HF_BUCKET_NAME --recursive --include "*.parquet"`

# Notes:
- Bucket mode is optimized for scale (streaming + minimal local storage).
- Dataset mode is suited for publishing datasets on the Hub.
- Local mode is useful for preprocessing or debugging pipelines.
"""
import os
import time
import yaml
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pyarrow as pa
import pyarrow.parquet as pq

from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfApi, batch_bucket_files

from gpt_lab.utils.common import get_banner
from gpt_lab.utils.default import CACHE_DIR, DATA_DIR
from gpt_lab.utils.schemas import DatasetConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
)
logger = logging.getLogger("reshard")
logger.setLevel(logging.INFO)

def read_dataset_config(ds_name: str, config_path: str) -> DatasetConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    config_dict = cfg.get(ds_name, {})
    if not config_dict:
        raise ValueError(f"No config for dataset {ds_name}")

    return DatasetConfig(name=ds_name, **config_dict)

class BaseWriter:
    def write(self, shard_path: str, shard_name: str):
        raise NotImplementedError

class LocalWriter(BaseWriter):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, shard_path: str, shard_name: str):
        final_path = self.output_dir / shard_name
        os.rename(shard_path, final_path)

class BucketWriter(BaseWriter):
    def __init__(self, repo_id: str, token: str, max_retries=3, retry_timeout=10):
        self.repo_id = repo_id
        self.token = token
        self.max_retries = max_retries
        self.retry_timeout = retry_timeout

    def write(self, shard_path: str, shard_name: str):
        for attempt in range(self.max_retries):
            try:
                batch_bucket_files(
                    bucket_id=self.repo_id,
                    add=[(shard_path, shard_name)],
                    token=self.token,
                )
                return
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_timeout * (2 ** attempt))

class DatasetWriter(BaseWriter):
    def __init__(self, repo_id: str, token: str, max_retries=3, retry_timeout=10):
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.max_retries = max_retries
        self.retry_timeout = retry_timeout

    def write(self, shard_path: str, shard_name: str):
        for attempt in range(self.max_retries):
            try:
                self.api.upload_large_file(
                    path_or_fileobj=shard_path,
                    path_in_repo=shard_name,
                    repo_id=self.repo_id,
                    repo_type="dataset"
                )
                return
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_timeout * (2 ** attempt))


def reshard_dataset(
    ds_config: DatasetConfig,
    mode: str = "bucket",  # bucket | dataset | local
    repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    chars_per_shard: int = 256_000_000,
    row_group_size: int = 1024,
    max_in_flight: int = 8,
    max_retries: int = 3,
    retry_timeout: int = 10,
    log_every: int = 1,
):

    builder_kwargs = {"path": ds_config.hfkwargs.get("path")}
    if ds_config.hfkwargs.get("name"):
        builder_kwargs["name"] = ds_config.hfkwargs["name"]

    ds_builder = load_dataset_builder(**builder_kwargs)
    ds = load_dataset(**ds_config.hfkwargs, streaming=ds_config.streaming)

    if ds_config.shuffle:
        ds = ds.shuffle(seed=42)

    postprocess_fn = None
    if ds_config.postprocess:
        from tiktoken import encoding_for_model
        tokenizer = encoding_for_model("gpt2")
        postprocess_fn = lambda x: tokenizer.decode(x)

    if mode == "bucket":
        writer = BucketWriter(repo_id, hf_token, max_retries, retry_timeout)
        work_dir = CACHE_DIR / "bucket"
    elif mode == "dataset":
        writer = DatasetWriter(repo_id, hf_token, max_retries, retry_timeout)
        work_dir = CACHE_DIR / "dataset"
    else:
        writer = LocalWriter(DATA_DIR / ds_config.output_dir)
        work_dir = CACHE_DIR / "local"

    work_dir.mkdir(parents=True, exist_ok=True)


    if not ds_config.streaming:
        ndocs = len(ds)
    else:
        try:
            ndocs = ds_builder.info.splits[
                ds_config.hfkwargs.get("split", "train")
            ].num_examples
        except Exception:
            ndocs = -1

    shard_docs = []
    shard_chars = 0
    shard_idx = 0

    total_docs = 0
    total_time = 0

    failed = []
    lock = Lock()

    executor = ThreadPoolExecutor(max_workers=max_in_flight)
    futures = []

    def process_shard(idx: int, docs: List[str]):
        shard_name = f"shard_{idx:05d}.parquet"
        shard_path = str(work_dir / shard_name)

        table = pa.Table.from_pydict({"text": docs})
        pq.write_table(
            table,
            shard_path,
            row_group_size=row_group_size,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )

        try:
            writer.write(shard_path, shard_name)
        except Exception:
            logger.exception(f"Shard {idx} failed")
            with lock:
                failed.append(idx)
        finally:
            if os.path.exists(shard_path):
                os.remove(shard_path)

    t0 = time.time()

    logger.info(f"Start dataset {ds_config.name} (mode={mode})")

    for doc in ds:
        if ds_config.max_shards and shard_idx >= ds_config.max_shards:
            break

        text = doc[ds_config.column_name]
        if postprocess_fn:
            text = postprocess_fn(text)

        shard_docs.append(text)
        shard_chars += len(text)

        if shard_chars >= chars_per_shard and len(shard_docs) % row_group_size == 0:

            futures.append(
                executor.submit(process_shard, shard_idx, shard_docs.copy())
            )

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            total_docs += len(shard_docs)
            total_time += dt

            if shard_idx % log_every == 0:
                logger.info(
                    f"{shard_idx:6d} | docs={len(shard_docs):8d} | total={total_docs:10d} | dt={dt:6.2f}s"
                )

            shard_idx += 1
            shard_docs = []
            shard_chars = 0

            if len(futures) >= max_in_flight:
                futures[0].result()
                futures.pop(0)

    if shard_docs:
        futures.append(executor.submit(process_shard, shard_idx, shard_docs))

    for f in futures:
        f.result()
    executor.shutdown()

    for attempt in range(max_retries):
        if not failed:
            break

        logger.warning(f"Retry {len(failed)} shards")

        retry_failed = []
        for idx in failed:
            try:
                process_shard(idx, [])
            except Exception:
                retry_failed.append(idx)

        failed = retry_failed
        time.sleep(retry_timeout * (2 ** attempt))

    if failed:
        logger.warning(f"Permanent failures: {failed}")

    logger.info(f"Done. Total docs: {total_docs}")


if __name__ == "__main__":
    get_banner(to_print=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds-name", default="climbmix")
    parser.add_argument("--config-path", default="configs/data.yaml")
    parser.add_argument("--mode", choices=["bucket", "dataset", "local"], default="bucket")
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--chars-per-shard", type=int, default=256_000_000)
    parser.add_argument("--row-group-size", type=int, default=1024)
    parser.add_argument("--max-in-flight", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-timeout", type=int, default=10)

    args = parser.parse_args()

    config = read_dataset_config(args.ds_name, Path(args.config_path))

    if args.streaming:
        config.streaming = True
    if args.max_shards:
        config.max_shards = args.max_shards

    hf_token = os.environ.get("HF_TOKEN")

    reshard_dataset(
        config,
        mode=args.mode,
        repo_id=args.repo_id,
        hf_token=hf_token,
        chars_per_shard=args.chars_per_shard,
        row_group_size=args.row_group_size,
        max_in_flight=args.max_in_flight,
        max_retries=args.max_retries,
        retry_timeout=args.retry_timeout,
    )