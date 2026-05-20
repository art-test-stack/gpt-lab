"""
# How ByteLevel BPE Tokenization Scales? 

## Summary
Full recipe for training and scaling tokenizer with different corpus sizes, vocabulary sizes, patterns, 
and evaluating the trained tokenizers on a simple test set to analyze the trade-offs between:
- training corpus size, 
- vocabulary size, 
- split pattern,
- tokenization quality (compression ration, efficiency, etc.)
- and cross-language generalization (if we have multilingual evaluation sets).

There is similar study on studying the optimal corpus size for training a BPE tokenizer as:
- [1] in which they find that the returns diminish after 150GB of training data, for BPE tokenizers with 40,960, 64,000, 128,000, and 256,000 vocabulary sizes.

However, this study is focused on training a BPE tokenizer with a specific size. They conclude that over 150GB a tokenizer with 

Here, we want to analyze the trade-offs 
between corpus size, vocabulary size, and tokenization quality, and also compare with truncated versions of baseline 
tokenizers to see how much of the performance can be retained with a smaller vocabulary size.

This is mainly motivated by the following facts:
- Language model have been scaled up but tokenizers sizes have not been scaled up as much, and it is not clear how much the tokenizer performance can be improved by scaling up the tokenizer training corpus and vocabulary size.
- According to [2], Language model performance is sensitive to tokenizer size, and the optimal size is often larger than the commonly used 50k tokens, especially for larger models and more diverse corpora. 

## Usage

How to run it from root directory of the repo:

- Make a new scaling run with new corpus sizes:


[!NOTE]
Recommended: run with `--optim-config-path=configs/optim.yaml` argument.

## Aknowledgements:
This code is inspired by and has some code adapted from the following sources:
- The Hugging Face Tokenizers library (https://github.com/huggingface/tokenizers)
- The OpenAI tiktoken library (https://github.com/openai/tiktoken)
- nanochat tokenizer code (https://github.com/karpathy/nanochat) for the idea of using HF-training backend + tiktoken-inference backend for efficient training and evaluation of tokenizers.

## References:
1. Reddy, Varshini, et al. "How much is enough? the diminishing returns of tokenization training data." arXiv preprint arXiv:2502.20273 (2025).
2. Zouhar, Vilém, et al. "Tokenization and the noiseless channel." Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023.

Author: Arthur Testard (arthur.testard.pro@gmail.com)
Please cite this work if the code is helpful to you.
"""
if __name__ == "__main__":
    from gpt_lab.utils.logging import init_logger
    init_logger()
    from gpt_lab.utils.common import get_banner
    get_banner(to_print=True)

from gpt_lab.tokenizer.tokenizer import Tokenizer
from gpt_lab.utils.schemas import TokenizerTrainerConfig, TokenizerConfig
from gpt_lab.tokenizer.corpus import TokenizerCorpus
from gpt_lab.utils.default import PAT_STR, TOKENIZERS_FOLDER, DATA_DIR
from gpt_lab.utils.logging import log0

import math
from pathlib import Path
import argparse, pickle, zipfile
import time
from tqdm import tqdm
from collections import Counter

import regex as re

import logging
logger = logging.getLogger(__name__)

BASELINES = ["gpt2", "cl100k_base", "o200k_base"]

# easy
def print(*args, **kwargs):
    log0(" ".join(str(arg) for arg in args), **kwargs, logger=logger)

def load_all_results(path):
    results = []
    with open(path, "rb") as f:
        while True:
            try:
                results.extend(pickle.load(f))
            except EOFError:
                break
    return results

def renyi_entropy(counter, alpha=2.5, eps=1e-12):
    """
    Rényi entropy of order alpha as proposed by Zouhar et al. (2023) to measure the diversity of the corpus:
    $$H_{\alpha}(X) = (1 / (1 - \alpha)) * \log( \sum_{x \in \mathcal{X}} p(x)^{\alpha})$$
    where p(x) is the probability of token x in the corpus.
    - For $\alpha \to 0$, it corresponds to the logarithm of the support size (number of unique tokens).
    - For $\alpha \to 1$, it corresponds to the Shannon entropy (the limit as $\alpha$approaches 1).
    - For $\alpha \to 2$, it corresponds to the collision entropy, which is related to the probability that two randomly chosen tokens are the same.
    - For $\alpha \to \infty$, it corresponds to the min-entropy, which is related to the probability of the most likely token.
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probabilities = [count / total for count in counter.values()]
    if alpha == 1:
        return -sum(p * math.log(p) for p in probabilities)
    else:
        return (1 / (1 - alpha)) * math.log(sum(p ** alpha for p in probabilities))
    
def entropy_efficiency(counter, alpha=2.5, eps=1e-12):
    """
    Efficiency of the tokenizer as proposed by Zouhar et al. (2023), defined as the ratio of the Rényi entropy of the token distribution to the logarithm of the vocabulary size:
    $$\text{Efficiency} = \frac{H_{\alpha}(X)}{\log(|V|)}$$
    where $H_{\alpha}(X)$ is the Rényi entropy of order $\alpha$ and $|V|$ is the vocabulary size.
    This metric captures how well the tokenizer utilizes its vocabulary to represent the diversity of the corpus. A higher efficiency indicates that the tokenizer is effectively using its vocabulary to capture the variability in the data, while a lower efficiency may suggest that many tokens are underutilized or that the tokenizer is not capturing enough diversity.
    """
    vocab_size = len(counter)
    if vocab_size == 0:
        return 0.0
    renyi_ent = renyi_entropy(counter, alpha=alpha, eps=eps)
    return renyi_ent / math.log(vocab_size)
    
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
        # max_chars=500_000, # just for evaluation, we can use a subset of the data
        max_bytes=500_000 * 4, # just for evaluation, we can use a subset of the data, adjust as needed based on your corpus
        bytes_per_doc=4096 * 4,
        split=eval_set["split"],
        compressed=True,
        shard_size_bytes=50_000 * 4,
        loader_fn=eval_set.get("loader_fn", None), # will overwrite for enwik8
    )

def eval_tokenizer(tokenizer):
    results = {}
    for eval_set in eval_sets:
        metrics = dict()
        counter = Counter()
        len_tokens = 0
        len_chars = 0
        len_bytes = 0
        corpus = get_eval_corpus(eval_set).iterator()
        t0 = time.time()
        for text in corpus:
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, disallowed_special=())
            counter.update(tokens)
            len_tokens += len(tokens)
            len_chars += len(text)
            len_bytes += len(text.encode("utf-8"))
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
        # both are useless actually as we store the counter and can compute any metric we want from it, 
        # but let's keep them for now as they are easy to compute and can be a quick proxy
        res["renyi_entropy"] = renyi_entropy(counter)
        res["entropy_efficiency"] = entropy_efficiency(counter)
        res["nb_tokens"] = len_tokens
        res["nb_chars"] = len_chars
        res["nb_bytes"] = len_bytes
        res["token_counter"] = counter
        res["eval_time"] = t1 - t0
        results[eval_set["metricname"]] = res
    return results

def compare_with_truncated_baselines(target_vocab_size):
    comparisons = {}
    from tiktoken import get_encoding

    for baseline in BASELINES:
        baseline_vocab_size = get_encoding(baseline).n_vocab
        if baseline_vocab_size <= target_vocab_size:
            continue

        truncated_name = f"{baseline}_truncated_{target_vocab_size}"
        truncated_tokenizer = Tokenizer.from_pretrained(truncated_name)
        comparisons[baseline] = {
            "base_vocab_size": baseline_vocab_size,
            "truncated_name": truncated_name,
            "evaluation": eval_tokenizer(truncated_tokenizer),
        }

    return comparisons

def run_tokenizer_experiment(task):
    (
        vocab_size,
        p_str_name,
        p_str,
        max_bytes,
        corpus_path,
        corpus_bytemax,
        seed,
        name,
    ) = task
    num_procs = 1  # IMPORTANT: avoid nested parallelism
    corpus = TokenizerCorpus.from_sources(
        corpus_dir=corpus_path,
        sources=None,
        max_bytes=corpus_bytemax,
        bytes_per_doc=corpus_bytemax // 20_000,
        random_seed=seed,
    )
    trainer_config = TokenizerTrainerConfig(
        max_bytes=max_bytes,
        bytes_per_doc=max_bytes // 20_000,
        num_proc=num_procs,
        source="huggingface",
        dircorpus=corpus_path, # TODO: add CorpusConfig instead
        show_progress=False,
        to_save=False,
    )
    config = TokenizerConfig(
        name=name,
        vocab_size=vocab_size,
        pat_str=p_str,
        trainer=trainer_config,
        source="huggingface", # this is quite dummy
        save_token_bytes=False, # we will compute token bytes on the fly without saving to disk to avoid IO overhead, adjust as needed based on your use case and whether you want to inspect the token bytes files
        # special_tokens=SpecialTokens(), # using default special tokens, adjust as needed
    )
    t0 = time.time()
    tokenizer = Tokenizer.train_from_iterator(
        text_iterator=corpus.iterator(max_bytes=max_bytes),
        config=config,
    )
    t1 = time.time()

    result = {
        "vocab_size": vocab_size,
        "pattern": p_str_name,
        "max_bytes": max_bytes,
        "tokenizer_name": name,
        "config": str(config),
        "training_time": t1 - t0,
        "corpus_size_mb": corpus_path.stat().st_size / 1e6,
    }

    for text in corpus.iterator(max_bytes=max_bytes):
        result["nb_chars_trained"] = result.get("nb_chars_trained", 0) + len(text)
        result["nb_words_trained"] = result.get("nb_words_trained", 0) + len(text.split())
        result["nb_bytes_trained"] = result.get("nb_bytes_trained", 0) + len(text.encode("utf-8"))
        result["nb_subwords_trained"] = result.get("nb_subwords_trained", 0) + len(re.findall(config.pat_str, text))
        result["nb_tokens_trained"] = result.get("nb_tokens_trained", 0) + len(tokenizer.encode(text, disallowed_special=()))
        
    result["evaluation"] = eval_tokenizer(tokenizer)

    del tokenizer
    return result

byte_per_doc = lambda max_byte: max_byte // 10_000 # Default to 1000 documents if not specified, adjust as needed

def main():
    parser = argparse.ArgumentParser(description="Find the optimal corpus size for training a BPE tokenizer with different vocabulary sizes, and evaluate the trained tokenizers on a simple test set to analyze the trade-offs between corpus size, vocabulary size, training time, and tokenization quality.")
    # General arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Default is 42.")
    parser.add_argument("--results-path", type=str, default=str(TOKENIZERS_FOLDER / 'scaling_tokenizer_results.pkl'), help="Path to store the results of the tokenizer evaluations. Default to './.gpt_lab/tokenizers/scaling_tokenizer_results.pkl'. If a file already exists at this path, it will be renamed with a number suffix to avoid overwriting previous results.")
    # Tokenizers configuration arguments
    parser.add_argument("--vocab-sizes", type=str, default="50000,70000,100000,200000", help="Comma-separated list of vocabulary sizes to train tokenizers with.")
    parser.add_argument("--pat-strs", type=str, default=None, help="Comma-separated list of pattern string names to use for tokenizer training. If not specified, defaults to using the GPT-2 pattern string.")
    # Corpus configuration arguments
    parser.add_argument("--write-corpus", action="store_true", help="Flag to indicate training mode (write corpus). If not set, the script will attempt to load an existing corpus from disk.")
    parser.add_argument("--corpus-sizes-mb", type=str, default=None, help="Comma-separated list of corpus sizes in megabytes to use for tokenizer training. If not specified, defaults to a range of sizes based on the vocabulary size.")
    parser.add_argument("--compare-truncated-baselines", action="store_true", help="Whether to compare trained tokenizers with truncated versions of baseline tokenizers.")
    parser.add_argument("--corpus-temperature-alpha", type=float, default=None, help="Optional temperature parameter to control the randomness of the corpus generation. Higher values will result in a more diverse corpus, while lower values will make it more focused on the most common samples. This can be useful for testing how the tokenizer performs with different levels of corpus diversity.")
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
                log0(f"Existing results file {backup_path} is empty. It will be overwritten with new results.", logger=logger)
                break
            new_name = results_path.stem + f"_{i}"
            backup_path = backup_path.with_stem(new_name)
            i += 1
        results_path.rename(backup_path)
        log0(f"Existing results file found. Renamed to {backup_path!r} to avoid overwriting. New results will be stored in {results_path!r}.", logger=logger)

    def store_results(results_batch, path=results_path):
        with open(path, "ab") as f:
            pickle.dump(results_batch, f)
        
        # try:
        #     with open(path, "rb") as f:
        #         results = pickle.load(f)
        # except FileNotFoundError:
        #     results = []

        # results.extend(results_batch)

        # with open(path, "wb") as f:
        #     pickle.dump(results, f)
    # Initiate test set and evaluation functions

    from tiktoken import get_encoding

    results = load_all_results(results_path) if results_path.exists() else []
    if len(results) == 0:
        results = []
        for baseline in BASELINES:
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
            results.append(result)
        store_results(results)
    

    # Corpus size varying with different vocab_sizes and split patterns
    # patterns = { "pat_str-gpt2": PAT_STR_GPT2, "pat_str-gpt4": PAT_STR_GPT4, "pat_str-punct": PAT_STR_punct, "pat_str-cl100k_base": PAT_STR_cl100k_base, "pat_str-o200k_base": PAT_STR_o200k_base }
    patterns = { "pat_str-gpt2": PAT_STR["gpt2"] }
    # patterns = { "PAT_STR_o200k_base": PAT_STR_o200k_base }
    # TODO: optimize by running the biggest vocab size and slice it on top-k merges for smaller vocabs
    # vocab_sizes = [10_000, 20_000, 30_000, 50_000, 100_000, 200_000, 300_000, 500_000] 
    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",")] if args.vocab_sizes else [50_000, 70_000, 100_000, 200_000]
    

    _max_char_runs = 16  # adjust the divisor to control how many runs are done before storing results to disk, this is a trade-off between memory usage and frequency of saving intermediate results. With 3 processes, we can afford to do more runs before saving, but if you have more memory constraints, you might want to save more frequently by using a smaller divisor.
    max_bytes = lambda vocab_size: [int(vocab_size * i * 1024) for i in range(1, _max_char_runs+1, 2)] # ~3.5 characters per token on average, adjust as needed based on your corpus
    # Two options: same name for all tokenizers -> overwrite / different names -> many tokenizers on disk, consider cleaning up after training or implementing a caching mechanism to avoid retraining the same tokenizer multiple times.
    # name = lambda vocab_size, max_char, p_str_name: f"ic1-tok-{int(vocab_size//1000)}k_maxchar-{max_char//1e6:.1f}M_pattern-{p_str_name}"
    log0(f"Using {num_procs} processes for tokenizer training.")
    corpus_path = DATA_DIR / "corpus" / results_path.stem
    results = []
    corpus_bytemax = max(max_bytes(max(vocab_sizes)))

    if args.write_corpus:
        print(f"Writing corpus to {corpus_path} with max bytes {corpus_bytemax:,}...")
        corpus = TokenizerCorpus.write_from_sources(
            corpus_dir=corpus_path,
            max_bytes=corpus_bytemax,
            bytes_per_doc=byte_per_doc(corpus_bytemax),
            random_seed=args.seed,
            temperature_alpha=args.corpus_temperature_alpha,
        )
        print(f"Corpus written to {corpus_path}. Size: {sum(c.stat().st_size / 1e6 for c in corpus_path.glob('*.txt')):.2f} MB")
    # Prepare run configurations
    if not args.write_corpus:
        print(f"Using existing corpus at {corpus_path} with max bytes {corpus_bytemax:,} for tokenizer training.")
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path {corpus_path} does not exist. Please run the script with --write-corpus flag to create the corpus before training tokenizers.")
        corpus = TokenizerCorpus.from_sources(corpus_dir=corpus_path)
        corpus.show_stats()
    tasks = []

    for vocab_size in vocab_sizes:
        for p_str_name, p_str in patterns.items():
            for max_byte in max_bytes(vocab_size):
                tasks.append(
                    (
                        vocab_size,
                        p_str_name,
                        p_str,
                        max_byte,
                        corpus_path,
                        corpus_bytemax,
                        args.seed,
                        f"ic1-scaling-tok-{p_str_name}-v{vocab_size}-b{max_byte//1e6:.1f}M",
                    )
                )

    t_total_start = time.time()
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    mp.set_start_method("spawn", force=True)
    max_workers = min(os.cpu_count(), 4)  # be conservative
    results = []
    # tasks_chunks = [tasks[i:i + max_workers] for i in range(0, len(tasks), _max_char_runs)]
    # for chunk in tqdm(tasks_chunks, desc="Processing task chunks"):
    #     with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #         futures = [executor.submit(run_tokenizer_experiment, task) for task in chunk]
    #         for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizer experiments"):
    #             results.append(future.result())

    #     store_results(results)
    #     results = []  # Reset results list for next chunk
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_tokenizer_experiment, t) for t in tasks]

        buffer = []
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Tokenizer experiments")):
            result = future.result()
            buffer.append(result)
            results.append(result)

            # if len(buffer) >= _max_char_runs:
            store_results(buffer)
            buffer.clear()

        if buffer:
            store_results(buffer)

    if args.compare_truncated_baselines:
        comparison_records = []
        for entry in results:
            if entry.get("baseline") is not None:
                continue
            target_vocab_size = entry["vocab_size"]
            comparison_records.append({
                "comparison_for": entry.get("tokenizer_name"),
                "vocab_size": target_vocab_size,
                "pattern": entry.get("pattern"),
                "max_chars": entry.get("max_chars"),
                "max_bytes": entry.get("max_bytes"),
                "truncated_baseline_evaluations": compare_with_truncated_baselines(target_vocab_size),
            })
        if comparison_records:
            store_results(comparison_records)

    print(f"Total time for all runs: {(time.time() - t_total_start)/3600:.2f} hours.")
    print(f"All runs completed. Results stored in {results_path}.")

if __name__ == "__main__":
    main()