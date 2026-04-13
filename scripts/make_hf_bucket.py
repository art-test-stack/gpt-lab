"""
Script very similar to `./make_hf_dataset.py` but to push datasets to huggingface bucket.
Designed for large datasets that may not fit in memory, with streaming and sharding support.
"""
from gpt_lib.utils.common import get_banner
from gpt_lib.utils.default import DATA_DIR, CACHE_DIR
from gpt_lib.utils.schemas import DatasetConfig

import os, yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import argparse, time
from typing import List, Tuple, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfApi, batch_bucket_files

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
)
logger = logging.getLogger("hf_bucket_upload")
logger.setLevel(logging.INFO)

for lib in ["datasets", "huggingface_hub", "httpx", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

def read_dataset_config(ds_name: str, config_path: str) -> DatasetConfig:
    with open(config_path, "r") as f:
        _cfg = yaml.safe_load(f)
    config_dict = _cfg.get(ds_name, {})
    if not config_dict:
        raise ValueError(f"No download config found for dataset {ds_name!r} in {config_path!r}.")
    return DatasetConfig(name=ds_name, **config_dict)


def download_ds_and_upload_bucket(
        ds_config: DatasetConfig, hf_token: str = None, 
        chars_per_shard: int = 256_000_000, row_gp_size: int = 1024, 
        upload_repo: str = None, upload_every: int = 1, max_retries: int = 3, retry_timeout: int = 10
    ):
    # Streaming mode for large datasets, but still process shards fully locally
    ds_builder_kwargs = {"path": ds_config.hfkwargs.get("path")}
    if ds_config.hfkwargs.get("name") is not None:
        ds_builder_kwargs["name"] = ds_config.hfkwargs.get("name")

    ds_builder = load_dataset_builder(**ds_builder_kwargs)
    ds = load_dataset(**ds_config.hfkwargs, streaming=ds_config.streaming)  # streaming=False for full download

    # Postprocess
    postprocess_fn = None
    if ds_config.postprocess:
        from tiktoken import encoding_for_model
        tokenizer = encoding_for_model("gpt2")
        postprocess_fn = lambda x: tokenizer.decode(x)

    if ds_config.shuffle:
        ds = ds.shuffle(seed=42)

    def _get_shard_path_and_bucket_path(shard_index):
        # NOTE: path in repo is simply shard name for simplicity for now
        shard_name = f"shard_{shard_index:05d}.parquet"
        shard_path = CACHE_DIR / "bucket" / shard_name
        bucket_path = ds_config.output_dir + shard_name
        return str(shard_path), bucket_path
    
    def upload_shards_to_hf_bucket(shard_idx: List[int]):
        shard_tuples = [_get_shard_path_and_bucket_path(s_idx) for s_idx in shard_idx]
        for attempt in range(max_retries):
            try:
                batch_bucket_files(
                    bucket_id=upload_repo,
                    add=shard_tuples,
                    token=hf_token,
                )
                for path, _ in shard_tuples:
                    if Path(path).exists():
                        os.remove(path)
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_timeout * (2 ** attempt))
    

    if ds_config.streaming:
        if ds_config.hfkwargs.get("path") == "nvidia/Nemotron-ClimbMix":
            ndocs = 553_240_576 # hardcoded for simplicity
        else:
            ndocs = ds_builder.info.splits[ds_config.hfkwargs.get("split", "train")].num_examples
        # Special case for nvidia/Nemotron-Corpus which has incorrect num_examples in streaming mode
    else:
        ndocs = len(ds)

    shard_docs = []
    shard_index = 0
    shard_chars = 0
    total_docs_written = 0
    total_time = 0

    shard_idx_to_upload = []
    failed_shard_idx = []

    executor = ThreadPoolExecutor(max_workers=4)
    failed_lock = Lock()
    futures = []

    def upload_wrapper(shard_idx):
        try:
            upload_shards_to_hf_bucket(shard_idx)
        except Exception as e:
            logger.exception(f"Failed to upload shard index {', '.join(map(str, shard_idx))}.")
            with failed_lock:
                failed_shard_idx.extend(shard_idx)

    t0 = time.time()

    logger.info(f"Starting download and processing of dataset {ds_config.name!r} ({ndocs:,} documents)...")
    logger.info(" Shard index | Shard docs | Shard chars | Total docs | Failed docs | Elapsed time | Est. remaining time (hr) ")

    for doc_idx, doc in enumerate(ds):
        if ds_config.max_shards is not None and shard_index >= ds_config.max_shards:
            break
        
        bucketdir = CACHE_DIR / "bucket"
        bucketdir.mkdir(parents=True, exist_ok=True)

        data = doc[ds_config.column_name]
        text = postprocess_fn(data) if postprocess_fn else data

        shard_docs.append(text)
        shard_chars += len(text)

        if (shard_chars >= chars_per_shard) and (len(shard_docs) % row_gp_size == 0):
            shard_path = bucketdir / f"shard_{shard_index:05d}.parquet"
            shard_table = pa.Table.from_pydict({"text": shard_docs})
            pq.write_table(
                shard_table,
                shard_path,
                row_group_size=row_gp_size,
                use_dictionary=True,
                compression="zstd",
                compression_level=3,
                write_statistics=False
            )

            # NOTE: this is not optimal as it uploads shards one by one (so safe for memory) but
            # 1. it does not take advantage of batch uploads to HF bucket which can be faster,  
            # 2. if there is connexion issue during upload, we lose the shard and have to start over (TODO: add retry logic)
            shard_idx_to_upload.append(shard_index)
            if (len(shard_idx_to_upload) >= upload_every or 
                (args.max_shards is not None and shard_index >= args.max_shards - 1)):
                futures.append(
                    executor.submit(upload_wrapper, shard_idx_to_upload.copy()))
                shard_idx_to_upload = []

            t1 = time.time()
            shard_time = t1 - t0
            t0 = t1
            total_docs_written += len(shard_docs)
            total_time += shard_time
            remaining_docs = ndocs - total_docs_written
            with failed_lock:
                n_failed = len(failed_shard_idx)
            avg_time_per_doc = total_time / total_docs_written if total_docs_written > 0 else 0
            est_remaining_time_hour = remaining_docs * avg_time_per_doc / 3600
            logger.info(f" {shard_index:11d} | {len(shard_docs):10,d} | {shard_chars:11,d} | {total_docs_written:10,d} | {n_failed:11,d} | {shard_time:11.2f}s | {est_remaining_time_hour:22.2f} ")
            shard_index += 1
            shard_docs = []
            shard_chars = 0

    # Write remaining docs
    if shard_docs:
        shard_path = bucketdir / f"shard_{shard_index:05d}.parquet"
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table, shard_path,
            row_group_size=row_gp_size,
            use_dictionary=True,
            compression="zstd",
            compression_level=3,
            write_statistics=False
        )
        futures.append(
            executor.submit(upload_wrapper, [shard_index])
        )

    for f in futures:
        f.result()
    
    executor.shutdown(wait=True)

    final_attempt = 0
    while failed_shard_idx and (final_attempt < max_retries):
        logger.warning(f"Retrying {len(failed_shard_idx)} failed shards...")

        still_failed = []

        for shard_idx in failed_shard_idx:
            try:
                upload_shards_to_hf_bucket([shard_idx])
                logger.info(f"Retry success for shard {shard_idx}")
            except Exception:
                logger.exception(f"Retry failed for shard {shard_idx}")
                still_failed.append(shard_idx)

        failed_shard_idx = list(set(still_failed))

        if failed_shard_idx:
            logger.warning(f"{len(failed_shard_idx)} shards still failed after retry attempt {final_attempt}.")
            time.sleep(retry_timeout * (2 ** final_attempt))

            final_attempt += 1
    
    if failed_shard_idx:
        logger.warning(f"{len(failed_shard_idx)} shards permanently failed: {', '.join(map(str, failed_shard_idx))}. Please check logs for details.")

    logger.info(f"Finished processing dataset {ds_config.name!r}. Total documents written: {total_docs_written:,}. Total time: {total_time/3600:.2f} hr.")

if __name__ == "__main__":
    get_banner(to_print=True)
    parser = argparse.ArgumentParser(description="Download datasets, process, and optionally upload to Hugging Face Hub.")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--ds-name", type=str, default="climbmix")
    parser.add_argument("--config-path", type=str, default="configs/data.yaml")
    parser.add_argument("--bucket-path", type=str, default="ai-testack/llm-factory-storage")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--chars-per-shard", type=int, default=256_000_000)
    parser.add_argument("--row-gp-size", type=int, default=1024)
    parser.add_argument("--upload-every", type=int, default=1, help="Number of shards to upload in batch to HF bucket. Higher means faster upload but more memory usage.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts for uploading a shard to HF bucket in case of failure.")
    parser.add_argument("--retry-timeout", type=int, default=10, help="Timeout in seconds between retry attempts for uploading a shard to HF bucket.")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path!r} not found.")

    config = read_dataset_config(args.ds_name, config_path)
    if args.max_shards is not None:
        config.max_shards = args.max_shards
    if args.streaming:
        config.streaming = True

    hf_token = os.environ.get("HF_TOKEN")
    download_ds_and_upload_bucket(config, hf_token=hf_token, 
        chars_per_shard=args.chars_per_shard, row_gp_size=args.row_gp_size, 
        upload_repo=args.bucket_path, upload_every=args.upload_every, 
        max_retries=args.max_retries, retry_timeout=args.retry_timeout)