import yaml
import argparse, time
from datasets import load_dataset
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from gpt_lib.utils.default import DATA_DIR
from gpt_lib.utils.schemas import DatasetConfig


def read_dataset_config(ds_name: str, config_path: str) -> DatasetConfig:
    with open(config_path, "r") as f:
        _cfg = yaml.safe_load(f)
    config_dict = _cfg.get(ds_name, {})
    if not config_dict:
        raise ValueError(f"No download config found for dataset {ds_name!r} in {config_path!r}.")
    return DatasetConfig(name=ds_name, **config_dict)

def download_dataset(ds_config: DatasetConfig):
    ds = load_dataset(**ds_config.hfkwargs)

    postprocess_fn = None
    if ds_config.postprocess:
        # TODO: this is a bit hacky but allows for flexible postprocessing 
        # functions defined in gpt_lib.data.postprocess module; we can expand
        # this later with more structured approach if needed
        # postprocess_fn = getattr(__import__("gpt_lib.data.postprocess", fromlist=[ds_config.postprocess]), ds_config.postprocess)
        # ds = ds.map(postprocess_fn, num_proc=ds_config.hfkwargs.get("num_proc", 1))
        from tiktoken import encoding_for_model
        tokenizer = encoding_for_model("gpt2")
        postprocess_fn = lambda x: tokenizer.decode(x)

    if ds_config.shuffle:
        ds = ds.shuffle(seed=42)

    ndocs = len(ds)

    output_dir = DATA_DIR / f"{ds_config.output_dir}-base"
    output_dir.mkdir(parents=True, exist_ok=False)

    chars_per_shard = 250_000_000
    row_gp_size = 1024
    shard_docs = []
    shard_index = 0
    shard_chars = 0
    total_docs_written = 0
    total_time = 0

    t0 = time.time()
    print(f"Starting download and processing of dataset {ds_config.name!r} with {ndocs:,} documents...")
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
    
    print(f"Finished processing dataset {ds_config.name!r}. Total documents written: {total_docs_written:,}. Total time: {total_time/3600:.2f} hr.")
    print(f"Shard table schema: {shard_table.schema}")
    # Write any remaining documents to a final shard
    if shard_docs:
        shard_path = output_dir / f"shard_{shard_index:05d}.parquet"
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(shard_table, shard_path)
    


if __name__ == "__main__":
    # Compare methods to download datasets in parallel with multiprocessing Pool, vs sequentially; also compare with huggingface-cli download command
    parser = argparse.ArgumentParser(description="Download datasets, process them, and upload to Hugging Face Hub.")
    parser.add_argument("--max-shards", type=int, default=None, help="Maximum number of shards to download (for testing).")
    parser.add_argument("--ds-name", type=str, default="climbmix", help="Name of the dataset to download (must be in config YAML).")
    parser.add_argument("--config-path", type=str, default="configs/data.yaml", help="Path to download config YAML file.")
    parser.add_argument("--upload-to-hf", action="store_true", help="Whether to upload the downloaded dataset to Hugging Face Hub.")
    parser.add_argument("--streaming", action="store_true", help="Whether to use streaming download from Hugging Face Datasets (use it for dev mode).")
    args = parser.parse_args()  

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path!r} not found.")
    
    config = read_dataset_config(args.ds_name, config_path)
    if args.max_shards is not None:
        config.max_shards = args.max_shards
    print(f"Starting download for dataset {args.ds_name!r} with config: {config}")
    # df_cfg = DownloadConfig()

    download_dataset(config)

    if args.upload_to_hf:
        if not config.output_dir or not config.upload_name:
            raise ValueError("'output_dir' and 'upload_name' must be specified in the config to upload to Hugging Face Hub." \
                            f"Please add these fields to the dataset config YAML at {config_path}." \
                            f"Got output_dir={config.output_dir!r}, upload_name={config.upload_name!r}.")
        from huggingface_hub import HfApi
        import os
        token = os.getenv("HF_TOKEN")
        api = HfApi(token=token)
        api.upload_large_folder(
            folder_path=DATA_DIR / config.output_dir,  # Path to the local folder containing the dataset
            repo_id=f"ai-teststack/{config.upload_name}",
            repo_type="dataset",
        )