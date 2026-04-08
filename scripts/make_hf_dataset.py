import yaml
import argparse, time
from datasets import load_dataset
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import os
from gpt_lib.utils.common import get_banner
from gpt_lib.utils.default import DATA_DIR
from gpt_lib.utils.schemas import DatasetConfig
from huggingface_hub import HfApi

def read_dataset_config(ds_name: str, config_path: str) -> DatasetConfig:
    with open(config_path, "r") as f:
        _cfg = yaml.safe_load(f)
    config_dict = _cfg.get(ds_name, {})
    if not config_dict:
        raise ValueError(f"No download config found for dataset {ds_name!r} in {config_path!r}.")
    return DatasetConfig(name=ds_name, **config_dict)

def download_and_upload_dataset(ds_config: DatasetConfig, hf_token: str = None, chars_per_shard: int = 250_000_000, row_gp_size: int = 1024):
    # Streaming mode for large datasets, but still process shards fully locally
    ds = load_dataset(**ds_config.hfkwargs, streaming=False)  # streaming=False for full download

    # Postprocess
    postprocess_fn = None
    if ds_config.postprocess:
        from tiktoken import encoding_for_model
        tokenizer = encoding_for_model("gpt2")
        postprocess_fn = lambda x: tokenizer.decode(x)

    if ds_config.shuffle:
        ds = ds.shuffle(seed=42)

    ndocs = len(ds)
    output_dir = DATA_DIR / f"{ds_config.output_dir}-base"
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_docs = []
    shard_index = 0
    shard_chars = 0
    total_docs_written = 0
    total_time = 0

    api = HfApi(token=hf_token) if hf_token else None

    t0 = time.time()
    print(f"Starting download and processing of dataset {ds_config.name!r} ({ndocs:,} documents)...")
    print(" Shard index | Shard docs | Shard chars | Total docs | Elapsed time | Est. remaining time (hr) ")

    for doc in ds:
        if ds_config.max_shards is not None and shard_index >= ds_config.max_shards:
            break

        data = doc[ds_config.column_name]
        text = postprocess_fn(data) if postprocess_fn else data

        shard_docs.append(text)
        shard_chars += len(text)

        if (shard_chars >= chars_per_shard) and (len(shard_docs) % row_gp_size == 0):
            shard_path = output_dir / f"shard_{shard_index:05d}.parquet"
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

            # Optional: upload shard immediately to HF Hub
            if hf_token and ds_config.upload_name:
                api.upload_large_file(
                    path_or_fileobj=shard_path,
                    path_in_repo=shard_path.name,
                    repo_id=f"ai-teststack/{ds_config.upload_name}",
                    repo_type="dataset"
                )
                shard_path.unlink()  # delete local shard to free space

            t1 = time.time()
            shard_time = t1 - t0
            t0 = t1
            total_docs_written += len(shard_docs)
            total_time += shard_time

            remaining_docs = ndocs - total_docs_written
            avg_time_per_doc = total_time / total_docs_written if total_docs_written > 0 else 0
            est_remaining_time_hour = remaining_docs * avg_time_per_doc / 3600
            shard_index += 1
            shard_docs = []
            shard_chars = 0
            print(f" {shard_index:11d} | {len(shard_docs):10d} | {shard_chars:11d} | {total_docs_written:10d} | {shard_time:12.2f}s | {est_remaining_time_hour:22.2f} ")

    # Write remaining docs
    if shard_docs:
        shard_path = output_dir / f"shard_{shard_index:05d}.parquet"
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(shard_table, shard_path)
        if hf_token and ds_config.upload_name:
            api.upload_large_file(
                path_or_fileobj=shard_path,
                path_in_repo=shard_path.name,
                repo_id=f"ai-teststack/{ds_config.upload_name}",
                repo_type="dataset"
            )
            shard_path.unlink()

    print(f"Finished processing dataset {ds_config.name!r}. Total documents written: {total_docs_written:,}. Total time: {total_time/3600:.2f} hr.")

if __name__ == "__main__":
    get_banner(to_print=True)
    parser = argparse.ArgumentParser(description="Download datasets, process, and optionally upload to Hugging Face Hub.")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--ds-name", type=str, default="climbmix")
    parser.add_argument("--config-path", type=str, default="configs/data.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path!r} not found.")

    config = read_dataset_config(args.ds_name, config_path)
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.max_shards is not None:
        config.max_shards = args.max_shards

    hf_token = os.environ.get("HF_TOKEN") if args.upload_to_hf else None
    download_and_upload_dataset(config, hf_token=hf_token)