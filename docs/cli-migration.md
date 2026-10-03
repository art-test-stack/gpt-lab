# CLI migration inventory

Inspected against clean `master` at `97f507a` before editing. No `AGENTS.md` or
configured formatter/linter was present. This inventory records **original parser
defaults**, including defaults that contradicted old help text. `$CACHE` means
`GPTLAB_CACHE_DIR` or `~/.cache/gpt_lab`. New help renders effective defaults.

| Original script | New command |
| --- | --- |
| `scripts/train_tokenizer.py` | `gpt-lab tokenizer train` |
| `scripts/train_base.py` | `gpt-lab train base auto / resume` |
| `scripts/benchmark/dataloaders.py` | `gpt-lab benchmark dataloaders` |
| `scripts/benchmark/tokenizer_corpus_size.py` | `gpt-lab experiment tokenizer-scaling` |
| `scripts/reshard_dataset.py` | `gpt-lab data reshard` |
| `scripts/clear_gpt_lib_cache.py` | `gpt-lab cache clear` |
| `scripts/chat_console.py` | `gpt-lab chat console` |
| `scripts/chat_app.py` | `gpt-lab chat app` |

All listed scripts are now thin compatibility wrappers. Parsing never imports the
old implementation. Reusable orchestration was moved into the installed package.
The original schema, model, optimizer, data loader, and training-loop implementations
are unchanged. The necessary CLI-boundary corrections are described below and in
[the CLI guide](cli.md).

## `scripts/train_tokenizer.py`

Loads or generates a corpus; trains and caches a tokenizer; prints round-trip diagnostics. Requires the tokenizer backend and, for generation, source datasets. The script rejected imports and parsed global argv. Its obsolete trainer/config and character-based corpus call did not match the current API.

| Scope | Argument | Original default |
| --- | --- | --- |
| command | `--vocab-size` | `32000` |
| command | `--max-chars` | `-1` |
| command | `--chars-per-doc` | `-1` |
| command | `--name` | `'ic1_tokenizer'` |
| command | `--corpus-path` | `'./.data/corpus.txt'` |
| command | `--write-corpus` | `False` |
| command | `--fast` | `False` |
| command | `--num-proc` | `-1` |
| command | `--trainer` | `'tiktoken'` |
| command | `--corpus-seed` | `42` |

## `scripts/train_base.py`

Allocates a model, optionally resolves/trains a tokenizer, opens local shards, initializes CUDA/NCCL under torchrun, trains/evaluates, writes checkpoints, and connects the selected metrics board. Imports set the Torch allocator environment, initialized logging, and imported side-effectful defaults. Resume uses model/run discovery and checkpoint restoration. arch/cust were nonfunctional placeholders.

| Scope | Argument | Original default |
| --- | --- | --- |
| auto | `--tokenizer-model` | `None` |
| auto | `--vocab-size` | `-1` |
| auto | `--pat-str` | `None` |
| auto | `--train-tokenizer` | `False` |
| auto | `--truncate-tokenizer` | `False` |
| auto | `--depth` | `12` |
| auto | `--aspect-ratio` | `64` |
| auto | `--d-head` | `128` |
| auto | `--n-kv-heads`, `--d-kv-head` | `None` |
| auto | `--window-pattern` | `None` |
| auto | `--window-size` | `None` |
| auto | `--softcap` | `18.0` |
| auto | `--attn-softcap` | `None` |
| auto | `--attn-impl` | `'sdpa'` |
| auto | `--num-steps` | `-1` |
| auto | `--target-flops` | `-1.0` |
| auto | `--target-param-data-ratio` | `11.0` |
| auto | `--target-time` | `-1.0` |
| auto | `--fp8` | `False` |
| auto | `--n-acc-steps` | `-1` |
| auto | `--total-batch-size` | `-1` |
| auto | `--lr-embeddings` | `0.3` |
| auto | `--lr-transformer` | `0.02` |
| auto | `--lr-head` | `0.008` |
| auto | `--lr-residuals` | `0.5` |
| auto | `--warmup-steps` | `40` |
| auto | `--warmdown-ratio` | `0.65` |
| auto | `--weight-decay` | `0.28` |
| auto | `--final-lr-frac` | `0.05` |
| resume | `--checkpoint-step` | `None` |
| resume | `--checkpoint-dir` | `None` |
| auto + resume (also former arch/cust) | `--model-name` | `'ic1'` |
| auto + resume (also former arch/cust) | `--run-name` | `None` |
| auto + resume (also former arch/cust) | `--model-dir` | `$CACHE/models` |
| auto + resume (also former arch/cust) | `--max-seq-len` | `2048` |
| auto + resume (also former arch/cust) | `--random-seed` | `42` |
| auto + resume (also former arch/cust) | `--optim-config-path` | `None` |
| auto + resume (also former arch/cust) | `--device` | `'auto'` |
| auto + resume (also former arch/cust) | `--board` | `'wandb'` |
| auto + resume (also former arch/cust) | `--board-dir` | `None` |
| auto + resume (also former arch/cust) | `--ds-config-path` | `'configs/data.yaml'` |
| auto + resume (also former arch/cust) | `--ds-name` | `'climbmix-base'` |
| auto + resume (also former arch/cust) | `--save-on-best` | `False` |
| auto + resume (also former arch/cust) | `--eval-bpb-every` | `250` |
| auto + resume (also former arch/cust) | `--n-bpb-tokens` | `80 * 524288` |
| auto + resume (also former arch/cust) | `--eval-core-every` | `2000` |
| auto + resume (also former arch/cust) | `--n-core-tokens` | `500` |
| auto + resume (also former arch/cust) | `--sample-every` | `0` |
| auto + resume (also former arch/cust) | `--save-every` | `-1` |
| auto + resume (also former arch/cust) | `--log-every` | `250` |
| auto + resume (also former arch/cust) | `--monitor-grad-norms` | `False` |
| auto + resume (also former arch/cust) | `--device-batch-size` | `32` |
| auto + resume (also former arch/cust) | `--use-nanochat-dataloader` | `False` |

## `scripts/benchmark/dataloaders.py`

Intended to allocate synthetic token buffers or read local bin/idx files and time loaders. The file contained a literal invalid import (`from ... import ...`), incompatible loader construction, wrong benchmark function arity, and mismatched selector choices. The old docstring advertised nonexistent packed/parquet_dir options. No functioning benchmark behavior could be preserved.

| Scope | Argument | Original default |
| --- | --- | --- |
| command | `--mode` | `'synthetic'` |
| command | `--loader` | `'all'` |
| command | `--batch_size` | `8` |
| command | `--seq_len` | `2048` |
| command | `--num_batches` | `20` |
| command | `--buffer_size` | `1000` |
| command | `--num_docs` | `20000` |
| command | `--device` | `'cpu'` |
| command | `--bos_id` | `1` |
| command | `--bin` | `None` |
| command | `--idx` | `None` |

## `scripts/benchmark/tokenizer_corpus_size.py`

Downloads baseline tokenizers and evaluation datasets, optionally generates the corpus, writes metadata and pickle results, and uses spawn multiprocessing. It downloaded enwik8 at import time. Resume failed on an existing directory; corpus-dir and the computed process cap were ignored; missing corpus sizes reached max(None). Baseline comparison and board flags were accepted but unused.

| Scope | Argument | Original default |
| --- | --- | --- |
| command | `--seed` | `42` |
| command | `--num-procs` | `None` |
| command | `--baselines` | `gpt2,cl100k_base,o200k_base` |
| command | `--vocab-sizes` | `'50000,70000,100000,200000'` |
| command | `--pat-strs` | `None` |
| command | `--write-corpus` | `False` |
| command | `--corpus-dir` | `None` |
| command | `--corpus-sizes-gb` | `None` |
| command | `--compare-truncated-baselines` | `False` |
| command | `--corpus-temperature-alpha` | `None` |
| command | `--resume` | `False` |
| command | `--result-dir` | `$CACHE/tokenizers/scaling_tokenizer_results` |
| command | `--save-every` | `10` |
| command | `--board` | `'dummy'` |

## `scripts/reshard_dataset.py`

Reads YAML and HF_TOKEN, downloads datasets, decodes token columns if configured, writes Parquet shards concurrently in temporary storage, and writes locally or uploads to a bucket/dataset repository. Existing shard names can be overwritten. Requires datasets, pyarrow, YAML and Hugging Face Hub. Streaming/output/max-shards overrides retain their original conditional behavior.

| Scope | Argument | Original default |
| --- | --- | --- |
| command | `--ds-name` | `'climbmix'` |
| command | `--config-path` | `'configs/data.yaml'` |
| command | `--mode` | `'bucket'` |
| command | `--repo-id` | `None` |
| command | `--output-dir` | `None` |
| command | `--streaming` | `False` |
| command | `--max-shards` | `None` |
| command | `--chars-per-shard` | `256000000` |
| command | `--row-group-size` | `1024` |
| command | `--max-in-flight` | `8` |
| command | `--max-retries` | `3` |
| command | `--retry-timeout` | `10` |

## `scripts/clear_gpt_lib_cache.py`

Intended to prompt and recursively delete cache directories. The force argument was commented out but args.force was still read. The all branch tried to delete a literal all directory; tokenizer/corpus selectors ignored the intended path mapping. Importing defaults created the cache and probed Torch.

| Scope | Argument | Original default |
| --- | --- | --- |
| command | `--cache_dir` | `$CACHE` |
| command | `--subdirs` | `[]` |

## `scripts/chat_console.py`

Only an import stub, including nonexistent schema imports; no parser, loop or generation was implemented. The CLI supplies a bounded local prompt loop using the existing checkpoint loader and Engine.generate.

No original command-line options.

## `scripts/chat_app.py`

Loaded dotenv, built Gradio Blocks at module scope, imported an interface which queries Hugging Face, then launched the app. No arguments existed. Requires UI and remote-completion dependencies. The dotenv import spelling was invalid; the existing UI also uses component arguments newer than the pinned Gradio version.

No original command-line options.

## Necessary boundary fixes

- Corrected misleading help defaults without changing their parsed values.
- Removed nonfunctional arch/cust command registrations. Future stages use a module registry.
- Converted explicit positive/zero checkpoint steps to integers; the checkpoint API rejected positive strings. Wired checkpoint-dir into existing run discovery and kept -N/latest/best selection.
- Moved base training setup into a callable and added exception-safe board/distributed cleanup. Configuration/training algorithms are retained.
- Constructed current nested TokenizerConfig/TokenizerTrainerConfig objects, applied character limits without confusing Unicode characters and bytes, and accepted existing UTF-8 text files or saved corpus directories. Unsupported backend selections now fail before data access.
- Replaced the invalid benchmark orchestration with adapters for existing production loaders, retained old options/defaults, and made all advertised selectors execute. Removed unreachable pasted classification and duplicate loader code. Integer-text conversion is explicitly identified as a synthetic fixture. Missing measurements are N/A, not fabricated zeroes.
- Deferred enwik8 download to evaluation; repaired experiment directory resume/backups, corpus-dir forwarding, process cap, and missing-size validation. GiB sizes are converted to integer byte budgets so fractional GiB values work with the current corpus/config APIs. Wired the existing truncated-baseline comparison helper. Result payloads and experiment methodology otherwise remain the existing implementation.
- Restored the intended cache selector mapping and whole-cache deletion, exposed force, and retained confirmation by default. Subdirectory symlink targets are not recursively deleted.
- Added a real local console loop in place of the import stub; delayed app construction and corrected dotenv imports. Existing experimental UI limitations are documented, not redesigned.

## Execution requirements and preserved limitations

Tokenizer training currently supports only the huggingface backend although the
old tiktoken default and all choices remain accepted by the parser. The training
flags fp8, truncate-tokenizer, and ds-config-path, tokenizer fast flag, and
experiment board flag remain unused as before; help identifies them explicitly.
The CLI does not add missing model/backend implementations. Full training and
external datasets/services are never invoked by the focused regression suite.
