import argparse
from pathlib import Path
import shutil
from gpt_lib.utils.default import CACHE_DIR, TOKENIZERS_FOLDER, DATA_DIR

paths = {
    "tokenizer": TOKENIZERS_FOLDER,
    "data": DATA_DIR,
    "corpus": DATA_DIR / "corpus",
    "models": CACHE_DIR / "models",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear GPT library cache.")
    parser.add_argument("--cache_dir", type=str, default=str(CACHE_DIR), help="Path to the cache directory. Default is the library's default cache directory.")
    # parser.add_argument("--force", action="store_true", help="Force clear the cache without confirmation.")
    parser.add_argument("--subdirs", nargs="*", default=[], help="Specific subdirectories to clear within the cache directory. If not provided, the entire cache directory will be cleared.", choices=["tokenizer", "data", "corpus", "models", "all"])
    args = parser.parse_args()

    if len(args.subdirs) == 0:
        print("No subdirectories specified. Clearing entire cache directory.")
        args.subdirs = ["all"]
    cache_dir = Path(args.cache_dir)
    if cache_dir.exists() and cache_dir.is_dir():
        print(f"Clearing cache directory: {cache_dir}")

        if not args.force:
            confirm = input("Are you sure you want to clear the cache? This action cannot be undone. (y/N): ")
            if confirm.lower() != "y":
                print("Cache clearing cancelled.")
                exit(1)
        if args.subdirs:
            for subdir in args.subdirs:
                subdir_path = cache_dir / subdir
                if subdir_path.exists() and subdir_path.is_dir():
                    shutil.rmtree(subdir_path)
                    print(f"Cleared subdirectory: {subdir_path}")
                else:
                    print(f"Subdirectory does not exist: {subdir_path}")
        else:
            shutil.rmtree(cache_dir)
        print("Cache cleared successfully.")
    else:
        print(f"Cache directory does not exist: {cache_dir}")