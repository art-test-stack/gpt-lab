import os
import warnings
from pathlib import Path
from typing import Union, List, Optional, Tuple
from dataclasses import dataclass
import time
 
import pyarrow.parquet as pq
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
from gpt_lib.utils.default import DATA_DIR
 

SHARD_FILENAME_TEMPLATE = "shard_{:05d}.parquet"
 
@dataclass
class ShardIterationState:
    """Tracks position in shard iteration for resumption."""
    shard_idx: int = 0
    row_group_idx: int = 0
    epoch: int = 1
 
class ShardManager:
    def __init__(
        self,
        data_dir: Union[str, Path] = DATA_DIR,
        base_url: str = "https://example.com/dataset_shards",
        column_name: str = "text",
        name: str = "base",
    ):
        """
        Initialize ShardManager.
        
        Args:
            data_dir: Directory where shards are stored/downloaded
            base_url: Base URL for downloading shards
            column_name: Name of the text column in parquet files
            name: Name identifier for this dataset
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        
        data_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = data_dir
        self.base_url = base_url
        self.column_name = column_name
        self.name = name
        
        self.shard_paths: List[str] = []
        self.num_shards = 0
        
        # Session with retry strategy for downloads
        self._session = self._create_session()
 
    @staticmethod
    def _create_session(
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        status_forcelist: Tuple[int, ...] = (500, 502, 503, 504),
    ) -> Session:
        """Create a requests session with retry strategy."""
        session = Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
 
    def list_shards(self, warn: bool = True) -> List[str]:
        """
        List all available parquet shards in data directory.
        
        Args:
            warn: If True, warn when no shards found
            
        Returns:
            Sorted list of paths to parquet files
        """
        parquet_files = sorted(
            p for p in self.data_dir.iterdir()
            if p.suffix == ".parquet" and p.name != ".tmp" and not p.name.endswith(".tmp")
        )
        self.shard_paths = [str(p) for p in parquet_files]
        self.num_shards = len(self.shard_paths)

        if self.num_shards == 0 and warn:
            warnings.warn(f"No parquet files found in {self.data_dir!r}. You may need to download the dataset first.")
        
        return self.shard_paths
 
    def download(
        self,
        shard_indices: Optional[List[int]] = None,
        num_workers: int = 1,
        max_retries: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Download specified shards from base_url.
        
        Args:
            shard_indices: Indices to download. If None, downloads all.
            num_workers: Number of parallel download workers
            max_retries: Max retry attempts per shard
            verbose: Print download progress
            
        Returns:
            Dict with 'successful' and 'failed' counts
        """
        if shard_indices is None:
            shard_indices = list(range(self.num_shards))
        
        successful = 0
        failed = 0
        
        for idx in shard_indices:
            if self._download_single_shard(idx, max_retries, verbose):
                successful += 1
            else:
                failed += 1
        
        return {"successful": successful, "failed": failed}
 
    def _download_single_shard(
        self,
        shard_idx: int,
        max_retries: int = 5,
        verbose: bool = True,
    ) -> bool:
        """
        Download a single shard with retry logic.
        
        Args:
            shard_idx: Index of shard to download
            max_retries: Number of retry attempts
            verbose: Print progress
            
        Returns:
            True if successful, False otherwise
        """
        filename = SHARD_FILENAME_TEMPLATE.format(shard_idx)
        filepath = self.data_dir / filename
        
        # skip if already exists
        if filepath.exists():
            if verbose:
                print(f"Skipping {filename} (already exists)")
            return True
        
        url = f"{self.base_url}/{filename}"
        temp_path = str(filepath) + ".tmp"
        
        for attempt in range(1, max_retries + 1):
            try:
                if verbose:
                    print(f"Downloading {filename} (attempt {attempt}/{max_retries})...")
                
                response = self._session.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # write to temp file first
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # move to final location
                os.rename(temp_path, str(filepath))
                
                if verbose:
                    print(f"✓ Downloaded {filename}")
                return True
                
            except Exception as e:
                if verbose:
                    print(f"✗ Attempt {attempt} failed: {e}")
                
                # Clean up partial file
                for path in [temp_path, str(filepath)]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
                
                # Exponential backoff
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    if verbose:
                        print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        if verbose:
            print(f"✗ Failed to download {filename} after {max_retries} attempts")
        return False
 
    def iterate(
        self,
        split: str = "train",
        start_state: Optional[ShardIterationState] = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ):
        """
        Iterate through shards, yielding text batches.
        
        Args:
            split: Either "train" or "val". Val uses only the last shard.
            start_state: Resume from this iteration state
            ddp_rank: DDP rank for distributed iteration
            ddp_world_size: Total DDP ranks
            
        Yields:
            Tuple of (texts: List[str], state: ShardIterationState)
        """
        if split not in ["train", "val"]:
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        
        if not self.shard_paths:
            self.list_shards(warn=False)
        
        # val uses only last shard, train uses all but last
        shard_indices = list(range(self.num_shards - 1)) if split == "train" else [self.num_shards - 1]
        
        if not shard_indices:
            raise ValueError(f"No shards available for split={split}")
        
        # resume state
        start_state = start_state or ShardIterationState()
        epoch = start_state.epoch
        shard_idx = start_state.shard_idx if split == "train" else shard_indices[0]
        rg_idx = start_state.row_group_idx
        
        first_pass = True
        
        while True:  # infinite iterator for multi-epoch training
            for shard_idx in shard_indices:
                filepath = self.shard_paths[shard_idx]
                pf = pq.ParquetFile(filepath)
                
                # determine starting row group
                if first_pass and rg_idx > 0 and shard_idx == start_state.shard_idx:
                    # resume from specific row group
                    start_rg = rg_idx // ddp_world_size
                    start_rg += 1  # Advance to avoid repeating data
                    rg_idx = start_rg * ddp_world_size + ddp_rank
                else:
                    rg_idx = ddp_rank
                
                # iterate row groups with DDP sharding
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column(self.column_name).to_pylist()
                    
                    state = ShardIterationState(
                        shard_idx=shard_idx,
                        row_group_idx=rg_idx,
                        epoch=epoch,
                    )
                    yield texts, state
                    
                    rg_idx += ddp_world_size
            
            first_pass = False
            epoch += 1
            rg_idx = ddp_rank  # reset for next epoch
