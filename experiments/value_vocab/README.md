# Value-embedding experiment

`run.py` trains and evaluates **one configuration per execution**. It accepts
one `--vocab-size`, `--sharing`, and `--seed`; grid expansion lives in
`submit.py`. The runner uses the existing tokenizer truncation, model,
optimizer, trainer, and BPB APIs without production-code changes.

Tiny single-run smoke check (no tokenizer training):

```bash
OMP_NUM_THREADS=1 uv run --locked --extra cpu python -m experiments.value_vocab.run \
  --smoke --vocab-size 320 --sharing independent --seed 1 \
  --output experiments/results/single-smoke.jsonl
```

Plan one configuration against your local UTF-8 train/held-out files:

```bash
uv run --locked --extra cpu python -m experiments.value_vocab.run \
  --train-text /path/to/train.txt --eval-text /path/to/heldout.txt \
  --base-tokenizer cl100k_base --vocab-size 32000 \
  --sharing independent --seed 0 --train-flops 1e12 --dry-run \
  --output experiments/results/single-plan.jsonl
```

`--base-tokenizer` supports `gpt2`, `cl100k_base` (default), and `o200k_base`.
`Tokenizer.truncated_from_pretrained` retains a prefix of the base's merge ranks,
including all 256 byte tokens; requested sizes include the repository's special
tokens. Requests above the base size fail. Pretrained assets may download on
first use. `--help` lists the seven sharing modes and model options.

Generate independent job commands with distinct result files:

```bash
uv run --locked --extra cpu python -m experiments.value_vocab.submit \
  --vocab-sizes 300 320 --sharing fully-shared independent --seeds 0 1 \
  --result-dir experiments/results/my-sweep -- --smoke > jobs.sh
```

Arguments after `--` are common runner options, such as model shape, input files,
base tokenizer, device, and training budget. Sweep axes belong before `--`.
The generator prints shell-quoted commands; it does not train, create result
files, or submit to a scheduler. Each line runs a separate Python process when
executed. Run commands from the repository root in the installed project
environment. The submission backend is undecided; these commands can be handed
to a parallel scheduler without embedding scheduling logic in the trainer.
Allocate devices externally (for example, `CUDA_VISIBLE_DEVICES` per job), and
avoid sharing a GPU when measuring runtime. Each command sets a separate
`GPTLAB_CACHE_DIR` under the result directory to prevent concurrent tokenizer
cache writes; tiktoken's pretrained-asset cache is still shared.

Each run writes one JSONL record with vocabulary/sharing/seed, parameter counts,
model configuration, text hashes, estimated FLOPs/token, training steps/tokens,
and held-out BPB/loss. Held-out evaluation covers every text token, including the
short final chunk. Parameter counts assign shared storage to input, output, then
value tables; a tied table contributes zero additional parameters. Existing
outputs are never overwritten. Temporary training checkpoints use the system
temporary directory and are removed when that execution finishes.

The protocol remains provisional: `--train-flops` controls only the repository's
**dense-matmul/attention estimate**, excluding full-table optimizer work. It does
not hold actual compute fixed across sharing configurations. The small default
budget, unconditional final checkpoint, and cyclic training stream still need
the protocol changes discussed in review. Do not interpret these results as a
compute-optimal vocabulary measurement. This refactor only separates execution
from grid preparation; it adds no corrected scaling law or GPU timing protocol.
