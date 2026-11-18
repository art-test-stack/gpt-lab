import errno
import hashlib
import json
import os
import random
import tempfile
import time
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
 
import pyarrow.parquet as pq
import torch.distributed as dist
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
from gpt_lab.utils.default import DATA_DIR
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
    """
    Manages downloading and iteration over data shards for a dataset. 
    Supports auto-downloading missing shards from a base URL, and can be used in distributed training settings.
    This class assumes that the dataset is split into multiple Parquet files (shards) and that each shard can be read independently. 
    These shards are expected to be named in the specific format 'shard_{:05d}.parquet' (e.g., shard_00000.parquet, shard_00001.parquet, etc.) 
    where the IDs are contiguous from zero. `max_shards` is the last ID
    reserved for validation. Training takes one pass over owned shards from
    the preceding IDs, optionally limited to `shard_limit` training shards.
    Missing remote shards are downloaded when an assigned rank needs them.
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
        data_seed: Optional[int] = None,
        start_state: Optional[DataLoaderState] = None,
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
        max_shards = max_shards if max_shards is not None else metadata.get("max_shards")

        self.base_url = base_url or ""
        self.column_name = column_name
        self.split = split

        if comp_info is None:
            comp_info = get_comp_info()
        self.comp_info = comp_info
        self.ddp_rank = comp_info.rank
        self.world_size = comp_info.world_size

        self._session = self._create_session() if self.base_url else None
        self.max_shards = max_shards if max_shards is not None else self.get_num_remote_shards()
        if shard_limit is not None and not 1 <= shard_limit <= self.max_shards:
            raise ValueError(f"shard_limit must be between 1 and {self.max_shards}: got {shard_limit}")
        self.shard_limit = shard_limit
        self.refresh_interval = refresh_interval
        self.download_poll_interval = download_poll_interval
        self._stop = False

        expected_metadata = {
            "name": name, 
            "base_url": self.base_url,
            "column_name": column_name,
            "max_shards": self.max_shards,
        }

        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata and metadata.get(key) != expected
        }
        error = f"Dataset metadata does not match configuration: {mismatches}" if mismatches else None
        if self.split == "train":
            canonical_ids = list(range(self.shard_limit or self.max_shards))
            if not canonical_ids:
                raise ValueError(f"No training shards in {self.ds_path}")
            manifest = {**expected_metadata, "training_shard_ids": canonical_ids}
            self.manifest_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            if start_state is None:
                self.data_seed = 42 if data_seed is None else data_seed
                random.Random(self.data_seed).shuffle(canonical_ids)
                self.ordered_shard_ids = tuple(canonical_ids)
            else:
                self.data_seed = start_state.data_seed
                self.ordered_shard_ids = start_state.ordered_shard_ids
                if start_state.packing_state_version == 0 or start_state.cursor_version == 0:
                    error = "Legacy dataloader checkpoint cannot resume exactly"
                elif (start_state.packing_rank, start_state.packing_world_size) != (self.ddp_rank, self.world_size):
                    error = f"Incompatible training topology: saved rank/world size {start_state.packing_rank}/{start_state.packing_world_size}, current {self.ddp_rank}/{self.world_size}"
                elif start_state.manifest_id != self.manifest_id or sorted(self.ordered_shard_ids) != canonical_ids:
                    error = "Incompatible training shard manifest"
                elif self.data_seed is None or (data_seed is not None and data_seed != self.data_seed):
                    error = f"Incompatible training data seed: saved {self.data_seed}, requested {data_seed}"
            self.shard_idx = self.ordered_shard_ids[self.ddp_rank::self.world_size]
            if self.base_url == "" and error is None:
                start = 0 if start_state is None else start_state.shard_idx
                missing = [shard_id for shard_id in self.shard_idx[start:] if not self.get_shard_path(shard_id).is_file()]
                if missing:
                    error = f"Missing training shards in {self.ds_path}: {missing}"
            if self.comp_info.is_distributed:
                identity = (self.manifest_id, self.ordered_shard_ids, self.data_seed)
                identities = [None] * self.world_size
                dist.all_gather_object(identities, (identity, error))
                errors = [f"rank {rank}: {message}" for rank, (_, message) in enumerate(identities) if message]
                if errors:
                    raise ValueError("Cannot initialize training data. " + ". ".join(errors))
                if any(other != identity for other, _ in identities):
                    raise ValueError("Training shard manifests or data seeds differ across ranks")
            elif error:
                raise ValueError(error)
        else:
            if error:
                raise ValueError(error)
            self.shard_idx = (self.max_shards,)
        self.opened_shards: list[int] = []
        self.downloaded_shards: list[int] = []
        self.read_retries: list[int] = []
        self.download_retries: list[int] = []

        if self.ddp_rank == 0 and not metadata:
            self.save_metadata(expected_metadata)

        self.shard_paths = [self.get_shard_path(shard_id) for shard_id in self.shard_idx]

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
            allowed_methods={"HEAD"},
            connect=0,
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

                    with self._session.get(url, stream=True, timeout=30) as r:
                        r.raise_for_status()
                        with open(tmp, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    os.rename(tmp, filepath)
                    self.downloaded_shards.append(idx)
                    break

                except Exception as exc:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    if attempt + 1 == max_retries:
                        raise OSError(f"Failed to download shard {idx} on rank {self.ddp_rank}") from exc
                    self.download_retries.append(idx)
                    time.sleep(2 ** attempt)

    def iterate(
        self,
        start_state: Optional[DataLoaderState] = None,
        batch_size: int = 128,
    ) -> Iterator[Tuple[List[str], DataLoaderState]]:
        is_resuming = start_state is not None
        state = start_state.model_copy() if start_state is not None else DataLoaderState()
        max_read_retries = 5
        row_group_start = 0 if self.split == "train" else self.ddp_rank
        row_group_stride = 1 if self.split == "train" else self.world_size

        while True:
            while state.shard_idx < len(self.shard_paths):
                shard_path = self.shard_paths[state.shard_idx]
                state.global_shard_idx = self.shard_idx[state.shard_idx]
                if not shard_path.is_file() and self.base_url:
                    self.download([state.global_shard_idx], verbose=self.ddp_rank == 0)
                self.opened_shards.append(state.global_shard_idx)
                pf = pq.ParquetFile(shard_path)

                try:
                    if is_resuming and state.cursor_version == 0:
                        base = state.row_group_idx // row_group_stride
                        state.row_group_idx = (
                            (base + 1) * row_group_stride + row_group_start
                        )

                        if state.row_group_idx >= pf.num_row_groups:
                            state.shard_idx += 1
                            state.row_group_idx = row_group_start
                            state.offset_in_row_group = 0
                            is_resuming = False
                            continue
                    elif not is_resuming:
                        state.row_group_idx = row_group_start

                    while state.row_group_idx < pf.num_row_groups:
                        for attempt in range(max_read_retries):
                            try:
                                # `pf` is set to None after a failed read.
                                # Reopening here means open failures are retried too.
                                if pf is None:
                                    self.opened_shards.append(state.global_shard_idx)
                                    pf = pq.ParquetFile(shard_path)

                                rg = pf.read_row_group(
                                    state.row_group_idx,
                                    columns=[self.column_name],
                                )
                                break

                            except OSError as exc:
                                is_io_error = exc.errno == errno.EIO

                                if pf is not None:
                                    try:
                                        pf.close()
                                    except Exception:
                                        pass
                                    pf = None

                                if (
                                    not is_io_error
                                    or attempt == max_read_retries - 1
                                ):
                                    raise OSError(
                                        "Failed to read Parquet row group: "
                                        f"path={shard_path}, "
                                        f"global_shard="
                                        f"{state.global_shard_idx}, "
                                        f"row_group="
                                        f"{state.row_group_idx}, "
                                        f"rank={self.ddp_rank}, "
                                        f"attempt={attempt + 1}/"
                                        f"{max_read_retries}"
                                    ) from exc

                                delay = 2**attempt
                                self.read_retries.append(state.global_shard_idx)

                                warnings.warn(
                                    "Transient Parquet I/O error; "
                                    "reopening and retrying: "
                                    f"path={shard_path}, "
                                    f"global_shard="
                                    f"{state.global_shard_idx}, "
                                    f"row_group="
                                    f"{state.row_group_idx}, "
                                    f"rank={self.ddp_rank}, "
                                    f"retry_in={delay}s"
                                )

                                time.sleep(delay)

                        batch = rg.column(self.column_name).to_pylist()

                        offset = state.offset_in_row_group if is_resuming and state.cursor_version == 1 else 0
                        if not 0 <= offset <= len(batch):
                            raise ValueError(f"Dataloader cursor offset {offset} is outside row group of length {len(batch)}")
                        is_resuming = False
                        for i in range(offset, len(batch), batch_size):
                            chunk = batch[i:i + batch_size]
                            next_offset = i + len(chunk)
                            next_group = state.row_group_idx
                            next_shard = state.shard_idx
                            if next_offset == len(batch):
                                next_group += row_group_stride
                                next_offset = 0
                                if next_group >= pf.num_row_groups:
                                    next_shard += 1
                                    next_group = row_group_start
                            yield (
                                chunk,
                                DataLoaderState(
                                    shard_idx=next_shard,
                                    global_shard_idx=(
                                        state.global_shard_idx
                                    ),
                                    row_group_idx=next_group,
                                    offset_in_row_group=next_offset,
                                    epoch=state.epoch,
                                    cursor_version=1,
                                ),
                            )

                        state.offset_in_row_group = 0
                        state.row_group_idx += row_group_stride

                finally:
                    # Also executes for resume `continue`, exceptions and
                    # generator shutdown.
                    if pf is not None:
                        try:
                            pf.close()
                        except Exception:
                            pass

                state.shard_idx += 1
                state.row_group_idx = row_group_start
                state.offset_in_row_group = 0
                is_resuming = False

            if self.split == "train":
                return
            state.shard_idx = 0
            state.row_group_idx = row_group_start
            state.offset_in_row_group = 0
            state.epoch += 1
