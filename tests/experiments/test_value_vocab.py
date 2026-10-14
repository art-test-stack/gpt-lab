import json
import math
from pathlib import Path
import tempfile

import pytest
import torch

from experiments.value_vocab import (
    ROOT, SHARING, CyclingBatches, evaluation_batches, parser, plan, run,
)
from gpt_lab.model.checkpoint import build_meta_model
from gpt_lab.tokenizer.tokenizer import Tokenizer
from gpt_lab.utils.schemas import TransformerConfig


@pytest.fixture
def workdir():
    root = ROOT / ".tmp/codex/value-vocab-tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def test_sharing_accounting_and_fixed_compute():
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


def test_real_smoke_sweep_and_dry_run(workdir, monkeypatch):
    def no_training(*args, **kwargs):
        pytest.fail("the experiment must not train tokenizers")

    monkeypatch.setattr(Tokenizer, "train_from_iterator", no_training)
    output = workdir / "results.jsonl"
    argv = ["--smoke", "--vocab-sizes", "300", "320", "--seeds", "0", "1", "--output", str(output)]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run(parser().parse_args(argv))
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(rows) == 2 * len(SHARING) * 2
        assert len({(r["requested_vocab_size"], r["sharing"], r["seed"]) for r in rows}) == len(rows)
        assert len({r["eval_sha256"] for r in rows}) == 1
        assert len({r["eval_bytes"] for r in rows}) == 1
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
        dry = workdir / "dry.jsonl"
        run(parser().parse_args(["--smoke", "--dry-run", "--sharing", "fully-shared", "--seeds", "0", "--output", str(dry)]))
        planned = [json.loads(line) for line in dry.read_text().splitlines()]
        assert [row["actual_vocab_size"] for row in planned] == [512, 1024, 2048]
        for row in planned:
            assert row["status"] == "planned"
            assert row["training_tokens"] == row["training_steps"] == 0
            assert row["eval_loss"] is row["eval_bpb"] is None
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("name", ["gpt2", "cl100k_base", "o200k_base"])
def test_pretrained_truncation_and_base_size_limit(name, workdir):
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
    output = workdir / "too-large.jsonl"
    with pytest.raises(ValueError, match="exceeds"):
        run(parser().parse_args(["--smoke", "--dry-run", "--base-tokenizer", name,
                                "--vocab-sizes", str(base.vocab_size + 1), "--output", str(output)]))
    assert not output.exists()
