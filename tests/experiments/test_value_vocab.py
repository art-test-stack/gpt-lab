import json
import math
import itertools
import shlex

import pytest
import torch

from experiments.value_vocab.run import (
    SHARING, CyclingBatches, evaluation_batches, parser, plan, run,
)
from experiments.value_vocab.submit import commands, parser as submit_parser
from gpt_lab.model.checkpoint import build_meta_model
from gpt_lab.tokenizer.tokenizer import Tokenizer
from gpt_lab.utils.schemas import TransformerConfig


def test_sharing_accounting_and_dense_flop_estimate():
    rows = []
    for vocab in [300, 600]:
        for mode in SHARING.values():
            model = build_meta_model(TransformerConfig(
                vocab_size=vocab, n_layers=4, d_model=16, d_ffn=64,
                n_heads=2, n_kv_heads=2, d_head=8, max_context=8,
                tie_embeddings=mode, window_pattern="L",
            ))
            row = plan(model, 3e6, 8)
            counts = model.n_params_per_layer()
            assert row["total_trainable_params"] == sum(p.numel() for p in model.parameters())
            assert row["n_non_vocab_params"] + sum(row[key] for key in (
                "input_embedding_params", "output_embedding_params", "value_embedding_params"
            )) == row["total_trainable_params"]
            assert row["n_non_vocab_params"] == counts["blocks"] + counts["residuals"]
            assert 0 <= row["compute_shortfall"] < row["flops_per_token"] * 8
            rows.append(row)
    assert len({row["n_non_vocab_params"] for row in rows}) == 1
    assert len({row["flops_per_token"] for row in rows[:7]}) == 1
    assert rows[7]["flops_per_token"] > rows[0]["flops_per_token"]
    assert rows[0]["value_embedding_params"] == rows[0]["output_embedding_params"] == 0
    assert rows[4]["value_embedding_params"] * 2 == rows[6]["value_embedding_params"]
    with pytest.raises(ValueError, match="cannot fund"):
        plan(model, 1, 8)


def test_batches_cover_tail_and_cycle():
    ids = [99, 1, 2, 3, 4, 5]
    batches = list(evaluation_batches(ids, 4, "cpu"))
    assert [y.tolist()[0] for _, y, _ in batches] == [[1, 2, 3, 4], [5]]
    assert sum(y.numel() for _, y, _ in batches) == len(ids) - 1
    train = iter(CyclingBatches(ids, 2, 2, "cpu"))
    x, y, _ = next(train)
    assert x.tolist() == [[99, 1], [2, 3]]
    assert y.tolist() == [[1, 2], [3, 4]]
    x, y, _ = next(train)
    assert x.tolist() == [[4, 5], [99, 1]]
    assert y.tolist() == [[5, 99], [1, 2]]


@pytest.mark.parametrize("vocab,sharing,seed", [
    (300, "fully-shared", 0), (300, "independent", 1),
    (320, "shared-values", 0), (320, "independent-values-tied-head", 1),
])
def test_single_run_and_dry_run(tmp_path, monkeypatch, vocab, sharing, seed):
    def no_training(*args, **kwargs):
        pytest.fail("the experiment must not train tokenizers")

    monkeypatch.setattr(Tokenizer, "train_from_iterator", no_training)
    output = tmp_path / "results.jsonl"
    argv = ["--smoke", "--vocab-size", str(vocab), "--sharing", sharing,
            "--seed", str(seed), "--output", str(output)]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run(parser().parse_args(argv))
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(rows) == 1
        assert (rows[0]["requested_vocab_size"], rows[0]["sharing"], rows[0]["seed"]) == (vocab, sharing, seed)
        for row in rows:
            assert row["actual_vocab_size"] == row["requested_vocab_size"]
            assert row["base_tokenizer"] == "cl100k_base"
            assert row["tokenizer_method"] == "merge-rank-prefix"
            assert row["training_tokens"] == row["planned_training_tokens"]
            assert row["training_steps"] == row["planned_training_steps"]
            assert row["training_flops"] <= row["requested_training_flops"]
            assert math.isfinite(row["eval_bpb"]) and row["eval_bpb"] > 0
            assert row["eval_loss"] == pytest.approx(row["eval_bpb"] * math.log(2) * row["eval_bytes"] / row["eval_tokens"])
        original = output.read_bytes()
        with pytest.raises(FileExistsError):
            run(parser().parse_args(argv))
        assert output.read_bytes() == original
        dry = tmp_path / "dry.jsonl"
        run(parser().parse_args([*argv[:-2], "--dry-run", "--output", str(dry)]))
        planned = [json.loads(line) for line in dry.read_text().splitlines()]
        assert [row["actual_vocab_size"] for row in planned] == [vocab]
        for row in planned:
            assert row["status"] == "planned"
            assert row["training_tokens"] == row["training_steps"] == 0
            assert row["eval_loss"] is row["eval_bpb"] is None
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("name", ["gpt2", "cl100k_base", "o200k_base"])
def test_pretrained_truncation_and_base_size_limit(name, tmp_path):
    base = Tokenizer.from_pretrained(name, source="tiktoken")
    truncated = Tokenizer.truncated_from_pretrained(name, 320, source="tiktoken")
    expected = dict(sorted(base.mergeable_ranks.items(), key=lambda item: item[1])[:319])
    assert truncated.mergeable_ranks == expected
    assert truncated.vocab_size == 320
    assert all(bytes([byte]) in truncated.mergeable_ranks for byte in range(256))
    text = "Hello, café! 東京 🐍\n"
    ids = truncated.encode(text)
    assert truncated.decode(ids) == text
    assert int(truncated.token_bytes[ids].sum()) == len(text.encode("utf-8"))
    output = tmp_path / "too-large.jsonl"
    with pytest.raises(ValueError, match="exceeds"):
        run(parser().parse_args(["--smoke", "--dry-run", "--base-tokenizer", name,
                                "--vocab-size", str(base.vocab_size + 1), "--output", str(output)]))
    assert not output.exists()


def test_job_commands_cover_grid_and_keep_paths_separate(tmp_path):
    args = submit_parser().parse_args([
        "--vocab-sizes", "300", "320", "--sharing", "fully-shared", "independent",
        "--seeds", "0", "1", "--result-dir", str(tmp_path / "results with spaces"),
        "--", "--train-text", "train text.txt", "--eval-text", "eval text.txt", "--dry-run",
    ])
    jobs = list(commands(args))
    parsed = []
    for command in jobs:
        assert shlex.split(shlex.join(command)) == command
        assert command[0] == "env" and command[2:5] == ["python", "-m", "experiments.value_vocab.run"]
        options = parser().parse_args(command[5:])
        assert str(options.train_text) == "train text.txt"
        assert str(options.eval_text) == "eval text.txt"
        assert options.dry_run
        parsed.append(options)
    assert {(job.vocab_size, job.sharing, job.seed) for job in parsed} == set(
        itertools.product([300, 320], ["fully-shared", "independent"], [0, 1])
    )
    assert len(jobs) == len({job.output for job in parsed}) == len({job[1] for job in jobs}) == 8
    assert not (tmp_path / "results with spaces").exists()


@pytest.mark.parametrize("extra", [["--seed=3"], ["--output", "out.jsonl"]])
def test_job_commands_reject_runner_overrides(extra):
    with pytest.raises(ValueError, match="submitter"):
        list(commands(submit_parser().parse_args(["--", *extra])))


def test_job_commands_reject_duplicate_axes():
    with pytest.raises(ValueError, match="duplicates"):
        list(commands(submit_parser().parse_args(["--vocab-sizes", "300", "300"])))
