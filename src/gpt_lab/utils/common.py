from pathlib import Path
import os, subprocess, logging
import random, re
from .default import CACHE_DIR, DATA_DIR
from filelock import FileLock
import urllib.request
from functools import lru_cache

@lru_cache()
def get_rank():
    return int(os.environ.get("RANK", 0))

@lru_cache()
def is_rank0():
    return get_rank() == 0

def format_value(value):
    """Format a value for display, adding commas to numbers and limiting floats to 4 decimal places."""
    if isinstance(value, bool):
        return "True" if value else "False"
    elif isinstance(value, int):
        return f"{value:,}"
    elif isinstance(value, float) and (abs(value) < 1e-3 or abs(value) >= 1e7):
        return f"{value:.2e}"
    elif isinstance(value, float) and abs(value) < 1e7:
        return f"{value:,.2f}"
    else:
        return str(value)

def print0(*values, **kwargs):
    """Print message only if on global rank 0"""
    if is_rank0():
        print(*values, **kwargs)

def print0_dict(title, info):
    lines = [f"{title}:"]
    for k, v in info.items():
        lines.append(f"\t{k:<25}: {format_value(v):<60}")
    print0("\n".join(lines), end="\n\n")


def slugify(text: str) -> str:
    """Convert text to a slug suitable for filenames and URLs."""
    return "".join(c if c.isalnum() else "_" for c in text.replace(" ", "-")).lower()

def get_repo_dir():
    if os.getenv("GPTLAB_BASE_DIR"):
        return Path(os.getenv("GPTLAB_BASE_DIR"))
    else:
        home_dir = Path.home()
        cache_dir = home_dir / ".gpt_lab"
        repo_dir = cache_dir / "gpt-lab"
        return repo_dir
    

def get_banner(to_print: bool = False) -> str:
    """Banner made with https://manytools.org/hacker-tools/ascii-banner/"""
    banner1 = r"""
  .-_'''-.   .-------. ,---------.              .---.        ____     _______    
 '_( )_   \  \  _(`)_ \\          \             | ,_|      .'  __ `. \  ____  \  
|(_ o _)|  ' | (_ o._)| `--.  ,---'           ,-./  )     /   '  \  \| |    \ |  
. (_,_)/___| |  (_,_) /    |   \  _ _    _ _  \  '_ '`)   |___|  /  || |____/ /  
|  |  .-----.|   '-.-'     :_ _: ( ' )--( ' )  > (_)  )      _.-`   ||   _ _ '.  
'  \  '-   .'|   |         (_I_)(_{;}_)(_{;}_)(  .  .-'   .'   _    ||  ( ' )  \ 
 \  `-'`   | |   |        (_(=)_)(_,_)--(_,_)  `-'`-'|___ |  _( )_  || (_{;}_) | 
  \        / /   )         (_I_)                |        \\ (_ o _) /|  (_,_)  / 
   `'-...-'  `---'         '---'                `--------` '.(_,_).' /_______.'                                                                  
"""
    banner2 = r"""
      ___           ___           ___           ___       ___           ___     
     /\  \         /\  \         /\  \         /\__\     /\  \         /\  \    
    /::\  \       /::\  \        \:\  \       /:/  /    /::\  \       /::\  \   
   /:/\:\  \     /:/\:\  \        \:\  \     /:/  /    /:/\:\  \     /:/\:\  \  
  /:/  \:\  \   /::\~\:\  \       /::\  \   /:/  /    /::\~\:\  \   /::\~\:\__\ 
 /:/__/_\:\__\ /:/\:\ \:\__\     /:/\:\__\ /:/__/    /:/\:\ \:\__\ /:/\:\ \:|__|
 \:\  /\ \/__/ \/__\:\/:/  /    /:/  \/__/ \:\  \    \/__\:\/:/  / \:\~\:\/:/  /
  \:\ \:\__\        \::/  /    /:/  /       \:\  \        \::/  /   \:\ \::/  / 
   \:\/:/  /         \/__/     \/__/         \:\  \       /:/  /     \:\/:/  /  
    \::/  /                                   \:\__\     /:/  /       \::/__/   
     \/__/                                     \/__/     \/__/         ~~      
"""
    banner = random.choice([banner1, banner2])
    if to_print:
        print0(banner)
    return banner

def run_command(cmd):
    """Run a shell command and return output, or None if it fails."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        # Return stdout if we got output (even if some files in xargs failed)
        if result.stdout.strip():
            return result.stdout.strip()
        if result.returncode == 0:
            return ""
        return None
    except:
        return None

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = DATA_DIR / "core"
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path
