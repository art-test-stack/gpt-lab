import json
import os
import random
import tempfile
import warnings
import errno
from contextlib import suppress
from pathlib import Path
from typing import Union, List, Optional, Tuple, Iterator
import time
 
import pyarrow.parquet as pq
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
from gpt_lab.utils.default import DATA_DIR, RANDOM_SEED
from gpt_lab.utils.schemas import DataLoaderState
from gpt_lab.utils.distributed import ComputeContext, get_comp_info
 
SHARD_FILENAME_TEMPLATE = "shard_{:05d}.parquet"

def list_parquet_files(data_dir=None):
    """Looks into a data dir and returns full paths to all parquet files."""
    if data_dir is None:
        data_dir = DATA_DIR
    if isinstance(data_dir, str):
        data_dir = Path(data_dir).resolve()

    return list(sorted(
        p for p in data_dir.iterdir()
        if p.suffix == ".parquet"
    ))

class ShardManager:
    """Read owned Parquet shards, downloading missing shards as needed.

    Shards are numbered from zero. The shard with ID max_shards is reserved
    for validation. Training owns shards by ID modulo world size and visits
    them in a fixed shuffled order.
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
        comp_info: Optional[ComputeContext] = None,
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
        if max_shards is None:
            max_shards = metadata.get("max_shards")
        self.base_url = base_url or ""
        self.column_name = column_name
        self.split = split

        self.comp_info = comp_info or get_comp_info()
        self.ddp_rank = self.comp_info.rank
        self.world_size = self.comp_info.world_size
        self._session = self._create_session() if self.base_url else None
        self.max_shards = max_shards if max_shards is not None else self.get_num_remote_shards()
        self.train_shards = shard_limit if shard_limit is not None else self.max_shards
        self.refresh_interval = refresh_interval
        self.download_poll_interval = download_poll_interval

        if shard_limit is not None and not 1 <= shard_limit <= self.max_shards:
            raise ValueError(f"Expected shard_limit between 1 and {self.max_shards}, got {shard_limit}")
        if split == "train":
            if self.train_shards < self.world_size:
                raise ValueError(f"Need at least {self.world_size} training shards in {self.ds_path}, found {self.train_shards}")
            self.shard_idx = list(range(self.ddp_rank, self.train_shards, self.world_size))
            random.Random(RANDOM_SEED).shuffle(self.shard_idx)
        else:
            if self.max_shards < 0:
                raise ValueError(f"No validation shard in {self.ds_path}")
            self.shard_idx = [self.max_shards]

        self.shard_paths = [self.get_shard_path(idx) for idx in self.shard_idx]
        if not self.base_url:
            missing = [p.name for p in self.shard_paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing {split} shards in {self.ds_path}: {missing}")

        expected_metadata = {
            "name": name,
            "base_url": self.base_url,
            "column_name": column_name,
            "max_shards": self.max_shards,
        }
        if metadata:
            mismatches = {
                key: (metadata.get(key), expected)
                for key, expected in expected_metadata.items()
                if metadata.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"Dataset metadata does not match configuration: {mismatches}")
        elif self.ddp_rank == 0:
            self.save_metadata(expected_metadata)

    def save_metadata(self, metadata: dict):
        metadata_path = self.ds_path / "meta.json"
        tmp_path = None
        try:
            # Write beside the destination so os.replace() is atomic.  A direct
            # open(..., "w") briefly exposes an empty file to other ranks while
            # they construct their train/validation ShardManagers.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.ds_path,
                prefix=f".{metadata_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = Path(f.name)
                json.dump(metadata, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, metadata_path)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def load_metadata(self) -> dict:
        metadata_path = self.ds_path / "meta.json"
        if not metadata_path.exists():
            return {}
        payload = metadata_path.read_text(encoding="utf-8")
        # Recover metadata files left empty by the previous non-atomic writer.
        # Rank 0 will regenerate the file after deriving values from the shards.
        if not payload.strip():
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid dataset metadata in {metadata_path}") from exc
        
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
            return len(self.list_local_shards()) - 1
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
        return lo - 1
    
    def list_local_shards(self):
        return list(sorted(
            p for p in self.ds_path.iterdir()
            if p.suffix == ".parquet"
        ))

    def download(
        self,
        shard_indices: List[int],
        max_retries: int = 5,
        verbose: bool = True,
    ):  
        if not self.base_url:
            raise ValueError("base_url must be set to download shards")
        for idx in shard_indices:
            filepath = self.get_shard_path(idx)
            filename = filepath.name
            if filepath.exists():
                continue

            url = f"{self.base_url}/{filename}"
            tmp = f"{filepath}.{self.ddp_rank}.tmp"
            for attempt in range(max_retries):
                try:
                    if verbose:
                        print(f"Downloading {filename}")

                    with self._session.get(url, stream=True, timeout=30) as response:
                        response.raise_for_status()
                        with open(tmp, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    os.rename(tmp, filepath)
                    break

                except Exception as exc:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    if attempt == max_retries - 1:
                        raise OSError(f"Failed to download shard {idx} on rank {self.ddp_rank}") from exc
                    time.sleep(2 ** attempt)

    def iterate(
        self,
        start_state: Optional[DataLoaderState] = None,
        batch_size: int = 128,
    ) -> Iterator[Tuple[List[str], DataLoaderState]]:
        """Yield document chunks with the cursor pointing after the chunk."""
        if batch_size < 1:
            raise ValueError("Shard iteration batch size must be positive")
        first_group = 0 if self.split == "train" else self.ddp_rank
        stride = 1 if self.split == "train" else self.world_size
        shard_start, group_start, offset, epoch = 0, first_group, 0, 1
        if start_state is not None:
            shard_start = start_state.shard_idx
            group_start = start_state.row_group_idx
            offset = start_state.offset_in_row_group
            epoch = start_state.epoch
            if (not 0 <= shard_start < len(self.shard_paths) or group_start < first_group
                    or (group_start - first_group) % stride or offset < 0 or epoch < 1):
                raise ValueError("Invalid data checkpoint cursor")

        while True:
            had_documents = False
            for shard_idx in range(shard_start, len(self.shard_paths)):
                path = self.shard_paths[shard_idx]
                global_idx = self.shard_idx[shard_idx]
                if not path.is_file() and self.base_url:
                    self.download([global_idx], verbose=self.ddp_rank == 0)
                pf = pq.ParquetFile(path)

                try:
                    if start_state is not None and shard_idx == shard_start and group_start >= pf.num_row_groups:
                        raise ValueError(f"Data checkpoint row group {group_start} is outside {path}")

                    for group_idx in range(group_start, pf.num_row_groups, stride):
                        for attempt in range(5):
                            try:
                                if pf is None:
                                    pf = pq.ParquetFile(path)
                                group = pf.read_row_group(group_idx, columns=[self.column_name])
                                break
                            except OSError as exc:
                                if exc.errno != errno.EIO or attempt == 4:
                                    raise OSError(
                                        f"Failed to read {path}, row group {group_idx}, "
                                        f"rank {self.ddp_rank}, attempt {attempt + 1}/5"
                                    ) from exc
                                if pf is not None:
                                    with suppress(Exception):
                                        pf.close()
                                    pf = None
                                delay = 2 ** attempt
                                warnings.warn(
                                    f"Transient Parquet I/O error in {path}, row group {group_idx}, "
                                    f"rank {self.ddp_rank}. Retrying in {delay}s"
                                )
                                time.sleep(delay)

                        texts = group.column(self.column_name).to_pylist()
                        if offset > len(texts):
                            raise ValueError(f"Data checkpoint offset {offset} is outside row group {group_idx} in {path}")
                        for i in range(offset, len(texts), batch_size):
                            chunk = texts[i:i + batch_size]
                            had_documents = True
                            yield chunk, DataLoaderState(
                                shard_idx=shard_idx,
                                global_shard_idx=global_idx,
                                row_group_idx=group_idx,
                                offset_in_row_group=i + len(chunk),
                                epoch=epoch,
                            )
                        offset = 0
                finally:
                    if pf is not None:
                        with suppress(Exception):
                            pf.close()

                group_start = first_group

            if not had_documents and start_state is None:
                raise ValueError(f"No {self.split} documents for rank {self.ddp_rank} in {self.ds_path}")
            start_state = None
            shard_start = 0
            epoch += 1
