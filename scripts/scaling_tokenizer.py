from gpt_lib.tokenizer.tokenizer import Tokenizer
from gpt_lib.utils.schemas import TokenizerTrainerConfig
from gpt_lib.tokenizer.corpus import TokenizerCorpus
from gpt_lib.utils.default import PAT_STR_GPT2, PAT_STR_GPT4, PAT_STR_punct, PAT_STR_cl100k_base, PAT_STR_o200k_base, TOKENIZERS_FOLDER, DATA_DIR

from pathlib import Path
import argparse, pickle, zipfile
import time
from tqdm import tqdm
from collections import Counter

import regex as re

def enwik8_path():
    base_dir = DATA_DIR / "corpus/eval_enwik8"
    base_dir.mkdir(parents=True, exist_ok=True)
    # download and unzip enwik8 to cache directory
    enwik8_url = "https://mattmahoney.net/dc/enwik8.zip"
    enwik8_local_path = base_dir.joinpath("enwik8")
    enwik8_local_path_zip = base_dir.joinpath("enwik8.zip")
    if not enwik8_local_path.exists():
        print(f"Downloading enwik8 to {enwik8_local_path_zip}")
        import requests
        response = requests.get(enwik8_url)
        with open(enwik8_local_path_zip, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(enwik8_local_path_zip, "r") as zip_ref:
            zip_ref.extractall(base_dir)
        print(f"Unzipped enwik8 to {enwik8_local_path}")
        enwik8_local_path_zip.unlink()
        print(f"Removed {enwik8_local_path_zip}")
    else:
        print(f"Using existing enwik8 at {enwik8_local_path}")
    return enwik8_local_path

enwik8_path = enwik8_path()

def enwik8_loader():
    with open(enwik8_path, "r", encoding="utf-8") as f:
        return f.read(10**7).split("\n") 

eval_configs = {
    "enwik8": dict(
        loader_fn=enwik8_loader,
    ),
    "HuggingFaceFW/fineweb-edu": dict(
        split="train" # no test or validation split available
    ),
    "HuggingFaceTB/finemath": dict(
        split="train",
        name=["finemath-3plus"]
    ),
    "ronantakizawa/github-top-code": dict(
        filter_fn=lambda x: x["file_language"] == "Python" # filter for python files only
    ),
    "HuggingFaceFW/fineweb-2": dict(
        name=["fra_Latn", "jpn_Jpan", "kor_Hang", "arb_Arab"],
    )
}
eval_sets = []
# prepare config to match the expected input of TokenizerCorpus.from_sources
for ds_name, ds_config in eval_configs.items():
    _ds = dict(name=ds_name)
    _ds["split"] = ds_config.get("split", "test")
    _ds["loader_fn"] = ds_config.get("loader_fn", None)
    _ds["generator_source"] = dict(path=ds_name, weight=1.0)
    if "filter_fn" in ds_config:
        _ds["generator_source"]["filter_fn"] = ds_config["filter_fn"]
    if ds_config.get("name", []) == []:
        _ds["localdir"] = DATA_DIR / f"corpus/eval_{ds_name.replace('/', '_')}"
        _ds["metricname"] = f"{ds_name.split('/')[-1]}"
        eval_sets.append(_ds)
    else:
        for name in ds_config["name"]:
            _ds = _ds.copy()
            _ds["generator_source"] = _ds["generator_source"].copy() # have to copy to avoid mutating the original for the next iteration
            _ds["subset"] = name
            _ds["localdir"] = DATA_DIR / f"corpus/eval_{ds_name.replace('/', '_')}:{name}"
            _ds["generator_source"]["name"] = name
            _ds["metricname"] = f"{ds_name.split('/')[-1]}:{name}"
            eval_sets.append(_ds)

def get_eval_corpus(eval_set):
    return TokenizerCorpus.from_sources(
        corpus_dir=eval_set["localdir"],
        sources=[eval_set["generator_source"]],
        max_chars=500_000, # just for evaluation, we can use a subset of the data
        chars_per_doc=10_000,
        split=eval_set["split"],
        compressed=True,
        shard_size_chars=50_000,
        loader_fn=eval_set.get("loader_fn", None), # will overwrite for enwik8
    )


def eval_tokenizer(tokenizer):
    results = {}
    for eval_set in eval_sets:
        metrics = dict()
        counter = Counter()
        len_tokens = 0
        len_chars = 0
        corpus = get_eval_corpus(eval_set).iterator()
        t0 = time.time()
        for text in corpus:
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, disallowed_special=())
            counter.update(tokens)
            len_tokens += len(tokens)
            len_chars += len(text)
            decoded = tokenizer.decode(tokens)
            acc = decoded == text
            compression_ratio = len(tokens) / len(text) if len(text) > 0 else 0
            # maybe optimized
            char_by_token = [len(tokenizer.decode([tok])) for tok in tokens]
            char_by_token_avg = sum(char_by_token) / len(char_by_token) if len(char_by_token) > 0 else 0
            for key, value in [
                ("accuracy", acc), 
                ("compression_ratio", compression_ratio),
                ("nb_char_by_token_avg", char_by_token_avg)
                ]:
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        t1 = time.time()
        res = {key: sum(values) / len(values) for key, values in metrics.items()}
        res["nb_tokens"] = len_tokens
        res["nb_chars"] = len_chars
        res["token_counter"] = counter
        res["eval_time"] = t1 - t0
        results[eval_set["metricname"]] = res
    return results

def run_tokenizer_experiment(task):
    (
        vocab_size,
        p_str_name,
        p_str,
        max_char,
        corpus_path,
        corpus_charmax,
        seed,
        name,
    ) = task
    num_procs = 1  # IMPORTANT: avoid nested parallelism
    corpus = TokenizerCorpus.from_sources(
        corpus_dir=corpus_path,
        sources=None,
        max_chars=corpus_charmax,
        chars_per_doc=corpus_charmax // 10_000,
        random_seed=seed,
    )
    config = TokenizerTrainerConfig(
        max_chars=max_char,
        chars_per_doc=max_char // 1000,
        vocab_size=vocab_size,
        name=name,
        num_proc=num_procs,
        trainer="huggingface",
        dircorpus=corpus_path,
        pat_str=p_str,
    )
    t0 = time.time()
    tokenizer = Tokenizer.train_from_iterator(
        text_iterator=corpus.iterator(max_chars=max_char),
        config=config,
    )
    t1 = time.time()

    result = {
        "vocab_size": vocab_size,
        "pattern": p_str_name,
        "max_chars": max_char,
        "config": str(config),
        "training_time": t1 - t0,
        "corpus_size_mb": corpus_path.stat().st_size / 1e6,
    }

    for text in corpus.iterator(max_chars=max_char):
        result["nb_chars_trained"] = result.get("nb_chars_trained", 0) + len(text)
        result["nb_words_trained"] = result.get("nb_words_trained", 0) + len(text.split())
        result["nb_subwords_trained"] = result.get("nb_subwords_trained", 0) + len(
            re.findall(config.pat_str, text)
        )
        result["nb_tokens_trained"] = result.get("nb_tokens_trained", 0) + len(
            tokenizer.encode(text, disallowed_special=())
        )
    result["evaluation"] = eval_tokenizer(tokenizer)

    del tokenizer
    return result

char_per_doc = lambda max_char: max_char // 1000 # Default to 1000 documents if not specified, adjust as needed

def main():
    parser = argparse.ArgumentParser(description="Find the optimal corpus size for training a BPE tokenizer with different vocabulary sizes, and evaluate the trained tokenizers on a simple test set to analyze the trade-offs between corpus size, vocabulary size, training time, and tokenization quality.")
    parser.add_argument("--write-corpus", action="store_true", help="Flag to indicate training mode (write corpus). If not set, the script will attempt to load an existing corpus from disk.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-path", type=str, default=str(TOKENIZERS_FOLDER / 'scaling_tokenizer_results.pkl'), help="Path to store the results of the tokenizer evaluations.")
    args = parser.parse_args()
    import os
    num_procs = min(os.cpu_count(), 32) 

    # initiate results storage
    # create results path if it doesn't exist 
    # and backup existing file if it does to avoid overwriting/mixing previous results
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        backup_path = results_path
        i = 1
        while backup_path.exists():
            with open(backup_path, "rb") as f:
                results = pickle.load(f)
            if results == [] or results is None:
                print(f"Existing results file {backup_path} is empty. It will be overwritten with new results.")
                break
            new_name = results_path.stem + f"_{i}"
            backup_path = backup_path.with_stem(new_name)
            i += 1
        results_path.rename(backup_path)
        print(f"Existing results file found. Renamed to {backup_path} to avoid overwriting. New results will be stored in {results_path}.")

    # Initiate test set and evaluation functions

    # testing_sets = ["HuggingFaceFW/fineweb-edu", "HuggingFaceTB/finemath", "codeparrot/codeparrot-clean", ]
    # "HuggingFaceFW/fineweb-2" "subset=fra_Latn,jpn_Jpan,kor_Hang,arb_Arab"

    def store_results(result, path=results_path):
        try:
            with open(path, "rb") as f:
                results = pickle.load(f)
        except FileNotFoundError:
            results = []
        results.append(result)
        with open(path, "wb") as f:
            pickle.dump(results, f)
        results = []

    # Baselines: gpt2, cl100k_base, o200k_base
    from tiktoken import get_encoding

    baselines = ["gpt2", "cl100k_base", "o200k_base"]
    for baseline in baselines:
        enc = get_encoding(baseline)
        evaluation = eval_tokenizer(enc)
        result = dict(
            vocab_size=enc.n_vocab,
            pattern=baseline,
            max_chars=None,
            config=None,
            training_time=None,
            corpus_size_mb=None,
            evaluation=evaluation,
            baseline=baseline,
        )
        store_results(result)

    # Corpus size varying with different vocab_sizes and split patterns
    patterns = { "pat_str-gpt2": PAT_STR_GPT2, "pat_str-gpt4": PAT_STR_GPT4, "pat_str-punct": PAT_STR_punct, "pat_str-cl100k_base": PAT_STR_cl100k_base, "pat_str-o200k_base": PAT_STR_o200k_base }
    # patterns = { "PAT_STR_o200k_base": PAT_STR_o200k_base }
    # TODO: optimize by running the biggest vocab size and slice it on top-k merges for smaller vocabs
    vocab_sizes = [10_000, 20_000, 30_000, 50_000, 100_000, 200_000, 300_000, 500_000] 

    # vocab_sizes = list(reversed(vocab_sizes))
    _max_char_runs = 8
    max_chars = lambda vocab_size: [int(vocab_size * i * 500) for i in range(1, _max_char_runs+1)] # ~3.5 characters per token on average, adjust as needed based on your corpus
    # Two options: same name for all tokenizers -> overwrite / different names -> many tokenizers on disk, consider cleaning up after training or implementing a caching mechanism to avoid retraining the same tokenizer multiple times.
    # name = lambda vocab_size, max_char, p_str_name: f"ic1-tok-{int(vocab_size//1000)}k_maxchar-{max_char//1e6:.1f}M_pattern-{p_str_name}"
    name = "ic1-scaling-tok"
    print(f"Using {num_procs} processes for tokenizer training.")
    corpus_path = DATA_DIR / "corpus/scaling_tokenizer_corpus"
    results = []
    corpus_charmax = max(max_chars(max(vocab_sizes)))

    # Prepare run configurations
    tasks = []

    for vocab_size in vocab_sizes:
        for p_str_name, p_str in patterns.items():
            for max_char in max_chars(vocab_size):
                tasks.append(
                    (
                        vocab_size,
                        p_str_name,
                        p_str,
                        max_char,
                        corpus_path,
                        corpus_charmax,
                        args.seed,
                        name,
                    )
                )

    t_total_start = time.time()
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    mp.set_start_method("spawn", force=True)

    max_workers = min(os.cpu_count(), 8)  # be conservative

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_tokenizer_experiment, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizer experiments"):
            results.append(future.result())

    for result in results:
        store_results(result)

    print(f"Total time for all runs: {(time.time() - t_total_start)/3600:.2f} hours.")
    print(f"All runs completed. Results stored in {results_path}.")

if __name__ == "__main__":
    main()