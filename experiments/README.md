# Experiments

Run these modules from the repository root with the installed project environment.
They are separate from the `gpt-lab` CLI; the old `scripts/benchmark/` entry points
have moved here. Use `--help` for all options and defaults.

## Dataloader benchmark

```bash
uv run --locked --extra cpu python -m experiments.dataloaders \
  --mode synthetic --loader all --batch-size 2 --seq-len 128 \
  --num-batches 5 --num-docs 100 --buffer-size 32
```

Synthetic mode runs offline. `glab` selects both gpt-lab loaders; `all` adds
nanochat. The on-the-fly fixture parses integer text, so it does not measure
natural-language BPE. Initialization and two warm-up batches are excluded.
`--buffer-size` means tokens for gpt-lab and documents for nanochat;
uninstrumented crop/search metrics report `N/A`.

Real mode requires `--bin tokens.bin --idx tokens.idx`: raw uint32 token IDs and
uint64 document offsets, including zero and the final token boundary. It downloads
no data. Hyphenated and underscore options work (`--batch-size`/`--batch_size`);
`--parquet_dir` and `--loader packed` are unsupported.

## Tokenizer scaling

```bash
uv run --locked --extra cpu python -m experiments.tokenizer_scaling \
  --num-procs 2 --vocab-sizes 20000,50000 --pat-strs gpt2,cl100k_base \
  --corpus-sizes-gb 0.1,0.5 --write-corpus --result-dir ./scaling-results
```

Downloads training/evaluation data and baselines. New sweeps require
`--corpus-sizes-gb` (GiB). `--resume --result-dir ./scaling-results` restores saved
sweep settings and skips completed `.pkl` results; otherwise existing results are
backed up to a numbered sibling. `--corpus-dir` overrides
`<cache>/data/corpus/<result-directory-name>`, where the cache root is
`GPTLAB_CACHE_DIR` (default: `~/.cache/gpt_lab`). `--compare-truncated-baselines`
enables comparison; `--board` is unused. Training/evaluation sources may overlap;
check before claiming held-out performance.

## Checks

```bash
uv run --locked --extra cpu --group dev pytest -q tests/experiments
```

Tests exercise the offline benchmark, argument parsing, result preservation,
and mocked scaling resume. They do not run a full tokenizer sweep.

## Preserved training experiments

- [Communication buckets and overlap](communication/README.md): explicit baseline/bucket/pipeline/hook choices and measured negative results.
- [Optimizer comparison](optimizer_comparison/README.md): existing Muon, NorMuon and NorMuon with ANVIL spectral maps; remaining GPU validation belongs to Agent A.
