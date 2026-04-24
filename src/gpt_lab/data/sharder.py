import os
import warnings
from pathlib import Path
from typing import Union, List, Optional, Tuple, Iterator, Dict
from dataclasses import dataclass
import time, json
 
import pyarrow.parquet as pq
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
from gpt_lab.utils.default import DATA_DIR
from gpt_lab.utils.distributed import get_dist_info
 
SHARD_FILENAME_TEMPLATE = "shard_{:05d}.parquet"
 
@dataclass
class ShardIterationState:
    """Tracks position in shard iteration for resumption."""
    shard_idx: int = 0
    global_shard_idx: Optional[int] = None # for debugging - we keep track of original shard idx
    row_group_idx: int = 0
    offset_in_row_group: int = 0
    epoch: int = 1


class ShardManager:
    """
    Manages downloading and iteration over data shards for a dataset. 
    Supports auto-downloading missing shards from a base URL, and can be used in distributed training settings.
    This class assumes that the dataset is split into multiple Parquet files (shards) and that each shard can be read independently. 
    These shards are expected to be named in the specific format 'shard_{:05d}.parquet' (e.g., shard_00000.parquet, shard_00001.parquet, etc.) 
    where the ids (integers) are contiguous starting from 0 until 'max_shards'.
    
    This class supposes a specific workflow where the last shard (i.e id = max_shards - 1) is reserved for validation and the rest (0, ..., shard_limit - 1) for training.

    > [!WARNING]
    > If 'base_url' is provided, it will attempt to download shards until it reaches 'max_shards'. If 'max_shards' is not set,
    > it will attempt to download every shard under the following format: {base_url}/shard_00000.parquet, {base_url}/shard_00001.parquet, etc. until it encounters a missing shard.
    > If shard names do not follow this format under the base_url, they just will be ignored.

    This class provides the following features:
        - filesystem view
        - download engine
        - streaming iterator
    """
    def __init__(
        self,
        name: str,
        cachedir: Optional[Union[str, Path]] = DATA_DIR,
        split: str = "train",
        base_url: Optional[str] = None,
        column_name: str = "text",
        shard_limit: Optional[int] = None, # control what is used for training
        max_shards: Optional[int] = None, # control what is downloaded
        # download settings
        refresh_interval: float = 5.0,
        download_poll_interval: float = 1.0,
        # dist settings
        dist_info: Optional[Dict] = None
    ):
        assert split in ["train", "val"], f"split must be 'train' or 'val'. Got {split=!r}."
        if cachedir is None:
            cachedir = DATA_DIR
        if isinstance(cachedir, str):
            cachedir = Path(cachedir).resolve()
        
        ds_path = cachedir / name
        if not ds_path.exists():
            ds_path.mkdir(parents=True, exist_ok=True)
        self.ds_path = ds_path

        metadata = self.load_metadata()
        max_shards = max_shards or metadata.get("max_shards", None)

        self.base_url = base_url
        self.column_name = column_name
        self.split = split

        if dist_info is None:
            dist_info = get_dist_info()
        self.dist_info = dist_info
        self.ddp_rank = dist_info.get("RANK", 0)
        self.world_size = dist_info.get("WORLD_SIZE", 1)

        self._session = None
        if self.base_url == "":
            self.base_url = None
        if self.base_url is not None:
            self._session = self._create_session()

        if shard_limit is not None and max_shards is not None:
            assert shard_limit <= max_shards, f"shard_limit ({shard_limit}) must be <= max_shards ({max_shards}). Got {shard_limit=}, {max_shards=}."
        assert shard_limit is None or shard_limit >= 2, f"shard_limit must be None or >= 2 for having at least one training shard and one validation shard. Got {shard_limit=}."
        self.shard_limit = shard_limit # shard limit is dynamic -> not in metadata
        self.max_shards = max_shards or self.get_num_remote_shards()
        self.target_shard = ((self.shard_limit or self.max_shards) if self.split == "train" else (self.max_shards or self.shard_limit)) - 1 # shards are 0, ..., max_shards-1
        self.shard_idx = list(range(self.target_shard)) if self.split == "train" else [self.target_shard] # always reserve last shard for val
        print("ShardManager initialized with the following settings:")
        print(f"  - Split: {self.split}")
        print(f"  - Base URL: {self.base_url}")
        print(f"  - Column Name: {self.column_name}")
        print(f"  - Shard Limit: {self.shard_limit}")
        print(f"  - Max Shards: {self.max_shards}")
        print(f"  - Target Shard: {self.target_shard}")
        print(f"  - Shard Indices: {self.shard_idx}")
        self.refresh_interval = refresh_interval
        self.download_poll_interval = download_poll_interval
        self._stop = False

        world_size = self.world_size
        assert self.max_shards is None or self.max_shards >= self.world_size, f"max_shards must be None or >= world_size. Got {max_shards=}, {world_size=}."
        assert self.shard_limit is None or self.shard_limit >= self.world_size, f"shard_limit must be None or >= world_size for proper DDP sharding. Got {shard_limit=}, {world_size=}."

        self.prepare_shards(warn=self.ddp_rank == 0)

        if self.ddp_rank == 0:
            self.save_metadata({
                "name": name,
                "base_url": base_url,
                "column_name": column_name,
                "max_shards": max_shards,
            })

    def save_metadata(self, metadata: dict):
        metadata_path = self.ds_path / "meta.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def load_metadata(self) -> dict:
        metadata_path = self.ds_path / "meta.json"
        if not metadata_path.exists():
            return {}
        with open(metadata_path, "r") as f:
            return json.load(f)
        
    def get_shard_path(self, shard_idx: int) -> Path:
        return self.ds_path / SHARD_FILENAME_TEMPLATE.format(shard_idx)

    @staticmethod
    def _create_session(
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        status_forcelist: Tuple[int, ...] = (500, 502, 503, 504),
    ):
        session = Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def get_num_remote_shards(self, max_probe: int = 1_000_000):
        if not self.base_url:
            return len(self.list_local_shards())
        if self.ddp_rank == 0:
            warnings.warn("Starting to probe number of remote shards. This may take a while if there are many shards and the server is slow to respond. " \
                          "\nYOUR REMOTE SHARDS NEED TO BE IN THE FORMAT 'shard_{:05d}.parquet'. " \
                          "\nBEST OPTION IS TO CANCEL THIS RUN AND SPECIFY max_shards.")
        def exists(i):
            filename = SHARD_FILENAME_TEMPLATE.format(i)
            url = f"{self.base_url}/{filename}"
            try:
                r = self._session.head(url, timeout=5)
                return r.status_code == 200
            except Exception:
                return False

        lo, hi = 0, 1
        # exponentially increase hi until we find a missing shard or reach max_probe 
        # ~ O(log N * response_time) 
        while hi < max_probe and exists(hi):
            lo = hi
            hi *= 2
        
        # binary search between lo and hi to find the exact number of shards
        # ~ O(log N * response_time)
        while lo < hi:
            mid = (lo + hi) // 2
            if exists(mid):
                lo = mid + 1
            else:
                hi = mid
        return lo
    
    def list_local_shards(self):
        return list(sorted(
            p for p in self.ds_path.iterdir()
            if p.suffix == ".parquet"
        ))

    def prepare_shards(self, warn: bool = False):
        local_shard_paths = self.list_local_shards()
        local_shard_idx = set([int(p.stem.split("_")[1]) for p in local_shard_paths])

        shards_to_download = []
        for idx in range(self.target_shard):
            if idx not in local_shard_idx:
                shards_to_download.append(idx)

        if shards_to_download and self.base_url is not None:
            self.download(shards_to_download, verbose=warn)
            
        self.shard_paths = [
            p for p in self.list_local_shards()
            if int(p.stem.split("_")[1]) in self.shard_idx
        ]
        if self.split == "train":
            assert len(self.shard_idx) == len(self.shard_paths), f"Expected {self.target_shard} shards but found {len(self.shard_idx)}. Got {self.shard_idx=}, {local_shard_paths=}."
        else:
            assert len(self.shard_paths) == 1 and self.shard_idx[0] == self.target_shard, f"Expected exactly 1 validation shard but found {len(self.shard_paths)}. Got {self.shard_paths=}."
    
    def download(
        self,
        shard_indices: List[int],
        max_retries: int = 5,
        verbose: bool = True,
    ):  
        if self.base_url is None: 
            raise ValueError("base_url must be set to download shards")
        for idx in shard_indices:
            filepath = self.get_shard_path(idx)
            filename = filepath.name
            if filepath.exists():
                continue

            url = f"{self.base_url}/{filename}"
            tmp = str(filepath) + ".tmp"
            for attempt in range(max_retries):
                try:
                    if verbose:
                        print(f"Downloading {filename}")

                    r = self._session.get(url, stream=True, timeout=30)
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    os.rename(tmp, filepath)
                    break

                except Exception:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    time.sleep(2 ** attempt)


    def iterate(
        self,
        start_state: Optional[ShardIterationState] = None,
        batch_size: int = 128,
    ) -> Iterator[Tuple[List[str], ShardIterationState]]:
        is_resuming = start_state is not None
        state = start_state or ShardIterationState()

        while True:
            while state.shard_idx < len(self.shard_paths):
                shard_path = self.shard_paths[state.shard_idx]
                state.global_shard_idx = int(shard_path.stem.split("_")[1]) # for debugging - we keep track of original shard idx

                pf = pq.ParquetFile(shard_path)

                if is_resuming:
                    base = state.row_group_idx // self.world_size
                    state.row_group_idx = (base + 1) * self.world_size + self.ddp_rank      
                    if state.row_group_idx >= pf.num_row_groups:
                        state.shard_idx += 1 # go to resuming shard id
                        state.row_group_idx = self.ddp_rank # start at the first row group for the next shard
                        state.offset_in_row_group = 0
                        continue
                else:
                    state.row_group_idx = self.ddp_rank

                while state.row_group_idx < pf.num_row_groups:
                    rg = pf.read_row_group(state.row_group_idx)
                    batch = rg.column(self.column_name).to_pylist()
                    for i in range(0, len(batch), batch_size):
                        if is_resuming and i < state.offset_in_row_group:
                            continue
                        if is_resuming:
                            is_resuming = False  # only do this once
                        yield batch[i:i+batch_size], ShardIterationState(
                            shard_idx=state.shard_idx,                            
                            global_shard_idx=state.global_shard_idx, # for debbugging - we keep track of original shard idx
                            row_group_idx=state.row_group_idx,
                            offset_in_row_group=i,
                            epoch=state.epoch,
                        )
                    state.offset_in_row_group = 0
                    state.row_group_idx += self.world_size
                state.shard_idx += 1
                state.row_group_idx = self.ddp_rank
    
            state.shard_idx = 0
            state.row_group_idx = self.ddp_rank
            state.offset_in_row_group = 0
            state.epoch += 1