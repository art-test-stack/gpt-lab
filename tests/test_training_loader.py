from datetime import timedelta
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel

from gpt_lab.data import loader as loader_module
from gpt_lab.data.loader import build_dataloader
from gpt_lab.evaluate.bpb import compute_bpb
from gpt_lab.model.checkpoint import CheckpointManager, capture_rng_state, set_rng_state
from gpt_lab.train.trainer import Trainer, TrainingDataExhaustedError
from gpt_lab.utils.board import DummyBoard
from gpt_lab.utils.distributed import ComputeContext
from gpt_lab.utils.schemas import DataLoaderState, TrainerConfig
from tests.test_checkpoint import assert_rng_state_consistency


class IntegerTokenizer:
    def get_bos_token_id(self):
        return 0

    def __call__(self, text, *, prepend_bos, threads):
        return [0] + [int(token) for token in text.split()]


@pytest.mark.parametrize("tokenizer", [IntegerTokenizer(), None], ids=["best-fit", "pretokenized"])
def test_validation_rank_without_row_groups_exhausts(tmp_path, monkeypatch, tokenizer):
    dataset = tmp_path / "tokens"
    dataset.mkdir()
    pq.write_table(pa.table({"text": ["1 2 3 4"]}), dataset / "shard_00001.parquet")
    make_loader = build_dataloader(
        name=dataset.name, cachedir=tmp_path, max_shards=1, split="val",
        tokenizer=tokenizer, batch_size=1, seq_len=4,
        comp_info=SimpleNamespace(rank=1, world_size=2, device=torch.device("cpu")),
    )
    loader = make_loader()
    open_parquet = pq.ParquetFile

    def open_once(path):
        assert loader.dataset.sm.opened_shards == [1]
        return open_parquet(path)

    monkeypatch.setattr(pq, "ParquetFile", open_once)
    assert next(loader, None) is None
    assert next(loader, None) is None
    assert loader.dataset.sm.opened_shards == [1]


def _validation_worker(rank, tmp_path):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=(tmp_path / "rendezvous").as_uri(),
        timeout=timedelta(seconds=30),
    )
    try:
        context = ComputeContext(device=torch.device("cpu"), mesh=init_device_mesh("cpu", (2,), mesh_dim_names=("dp",)))
        make_loader = build_dataloader(
            name="tokens", cachedir=tmp_path, max_shards=1, split="val", tokenizer=IntegerTokenizer(),
            batch_size=1, seq_len=4, comp_info=context,
        )
        assert len(list(islice(make_loader(), 3))) == 2 - rank
        calls = []

        def model(x, y, reduction):
            calls.append((x.tolist(), y.tolist()))
            return torch.ones_like(y, dtype=torch.float32)

        model.get_device = lambda: context.device
        with pytest.raises(RuntimeError, match=r"Validation exhausted after 1 of 2 full batches on ranks \[1\]"):
            compute_bpb(model, make_loader(), 2, torch.ones(20), comp_info=context)
        first = 1 + 10 * rank
        assert calls == [([[0, first, first + 1, first + 2]], [[first, first + 1, first + 2, first + 3]])]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_distributed_validation_rejects_uneven_exhaustion(tmp_path):
    dataset = tmp_path / "tokens"
    dataset.mkdir()
    pq.write_table(pa.table({"text": ["1 2 3 4 5 6 7 8", "11 12 13 14 15 16"]}),
                   dataset / "shard_00001.parquet", row_group_size=1)
    mp.spawn(_validation_worker, args=(tmp_path,), nprocs=2, join=True)


@pytest.mark.fast
@pytest.mark.parametrize("batch_size", [1, 2])
def test_best_fit_training_rows(tmp_path, batch_size):
    dataset = tmp_path / "letters"
    dataset.mkdir()
    texts = ["1 2 3 4 5", "6 7", "8 9", "10 11 12 13", "14 15 16 17 18 19 20"]
    pq.write_table(pa.table({"text": texts}), dataset / "shard_00000.parquet")

    args = dict(
        name=dataset.name,
        cachedir=tmp_path,
        max_shards=1,
        tokenizer=IntegerTokenizer(),
        batch_size=batch_size,
        seq_len=4,
        comp_info=ComputeContext(device=torch.device("cpu")),
    )
    loader = build_dataloader(**args)
    inputs, targets, states = [], [], []
    for _ in range(6 // batch_size):
        x, y, state = next(loader)
        inputs.extend(x.tolist())
        targets.extend(y.tolist())
        states.append(state)

    assert inputs == [
        [0, 10, 11, 12], [0, 6, 7, 0], [0, 9, 0, 1],
        [0, 3, 4, 5], [0, 14, 15, 16], [0, 18, 19, 20],
    ]
    assert targets == [
        [10, 11, 12, 13], [6, 7, 0, 8], [9, 0, 1, 2],
        [3, 4, 5, 0], [14, 15, 16, 17], [18, 19, 20, -1],
    ]
    saved = DataLoaderState.model_validate_json(states[2 // batch_size - 1].model_dump_json())
    assert saved.lookahead_documents == ([0, 1, 2, 3, 4, 5], [0, 14, 15, 16, 17, 18, 19, 20])
    assert saved.pending_document == [0, 8, 9] and saved.document_token_offset == 2
    x, y, _ = next(build_dataloader(**args, resume_state=saved))
    assert x.tolist() == inputs[2:2 + batch_size]
    assert y.tolist() == targets[2:2 + batch_size]


def test_resume_keeps_saved_lookahead_size(tmp_path, monkeypatch):
    dataset = tmp_path / "tokens"
    dataset.mkdir()
    texts = [f"{i} {i}" for i in range(1, 151)]
    pq.write_table(pa.table({"text": texts}), dataset / "shard_00000.parquet")
    args = dict(name=dataset.name, cachedir=tmp_path, max_shards=1, tokenizer=IntegerTokenizer(),
                batch_size=1, seq_len=4, comp_info=ComputeContext(device=torch.device("cpu")))

    with monkeypatch.context() as previous:
        previous.setattr(loader_module, "_TRAIN_LOOKAHEAD_DOCS", 128)
        uninterrupted = build_dataloader(**args)
        _, _, saved = next(uninterrupted)
    _, _, fresh_state = next(build_dataloader(**args))
    assert fresh_state.packing_lookahead_size == 16
    assert len(fresh_state.lookahead_documents) <= 16 < len(saved.lookahead_documents)
    resumed = build_dataloader(**args, resume_state=saved)
    for _ in range(3):
        expected_x, expected_y, expected_state = next(uninterrupted)
        x, y, state = next(resumed)
        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)
        assert state == expected_state
        assert state.packing_lookahead_size == 128


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"packing_world_size": 2}, "Incompatible training topology"),
        ({"ordered_shard_ids": (1,)}, "Incompatible training shard manifest"),
        ({"packing_state_version": 0}, "Legacy dataloader checkpoint cannot resume exactly"),
        ({"packing_state_version": 1}, "Source-order dataloader checkpoint cannot resume exactly"),
        ({"packing_lookahead_size": 0}, "Invalid lookahead size"),
    ],
)
def test_incompatible_training_checkpoint_fails_before_open(tmp_path, monkeypatch, changes, message):
    dataset = tmp_path / "tokens"
    dataset.mkdir()
    pq.write_table(pa.table({"text": ["1 2 3 4"]}), dataset / "shard_00000.parquet")
    args = dict(name=dataset.name, cachedir=tmp_path, max_shards=1, tokenizer=IntegerTokenizer(),
                batch_size=1, seq_len=4, comp_info=ComputeContext(device=torch.device("cpu")))
    _, _, state = next(build_dataloader(**args))

    def forbidden_open(*args, **kwargs):
        pytest.fail("Incompatible checkpoint opened a training shard")

    monkeypatch.setattr(pq, "ParquetFile", forbidden_open)
    with pytest.raises(ValueError, match=message):
        build_dataloader(**args, resume_state=state.model_copy(update=changes))


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(256, 8)
        self.dropout = torch.nn.Dropout(0.35)
        self.lm_head = torch.nn.Linear(8, 256)
        self.batches = []

    def forward(self, x, y):
        self.batches.append((x.tolist(), y.tolist()))
        return self.lm_head(self.dropout(self.embed(x)))


class DelegatingTrainer(Trainer):
    def train_step(self, train_iter, batch, should_log_step):
        return super().train_step(train_iter, batch, should_log_step)


def _training_loader_worker(rank, tmp_path, order, texts_by_shard):
    # Keep the fixture small enough to hit both Parquet boundaries with unread input.
    loader_module._TRAIN_LOOKAHEAD_DOCS = 2
    torch.set_num_threads(1)
    torch.manual_seed(17)
    random.seed(rank + 100)
    torch.compile = lambda model, **kwargs: model
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=(tmp_path / "rendezvous").as_uri(),
        timeout=timedelta(seconds=45),
    )
    try:
        context = ComputeContext(device=torch.device("cpu"), mesh=init_device_mesh("cpu", (2,), mesh_dim_names=("dp",)))
        tokenizer = IntegerTokenizer()
        args = dict(name="tokens", cachedir=tmp_path, max_shards=5, tokenizer=tokenizer,
                    batch_size=1, seq_len=4, comp_info=context)
        config = TrainerConfig(
            comp_info=context, n_steps=4, stop_after_step=3, device_batch_size=1, n_acc_steps=2,
            total_batch_size=16, n_flops_per_token=10, eval_bpb_every=0, eval_core_every=0,
            sample_every=0, save_every=1, log_every=0,
        )
        original_open = pq.ParquetFile
        opens = []

        def record_open(path, *args, **kwargs):
            opens.append(int(Path(path).stem.split("_")[1]))
            return original_open(path, *args, **kwargs)

        pq.ParquetFile = record_open

        initial_rng = capture_rng_state()

        def make_trainer(run_name, checkpoint=None, cap=3, dataset="tokens", trainer_class=Trainer):
            set_rng_state(initial_rng)
            loader = build_dataloader(
                **(args | {"name": dataset}), data_seed=83 if checkpoint is None else None,
                resume_state=None if checkpoint is None else checkpoint.trainer_state.train_loader_state,
            )
            model = DistributedDataParallel(RecordingModel())
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
            optimizer.update_hyperparams = lambda **kwargs: None
            (tmp_path / "test" / run_name).mkdir(parents=True, exist_ok=True)
            manager = CheckpointManager(
                model_name="test", model_run=run_name, model_cachedir=tmp_path,
                comp_info=context, optimizer_state_layout="sharded",
            )
            if checkpoint is not None:
                model.load_state_dict(checkpoint.model_state)
                optimizer.load_state_dict(checkpoint.optimizer_state)
                set_rng_state(checkpoint.rng_state)
            return trainer_class(
                model, tokenizer, loader, None, config=config.model_copy(update={"stop_after_step": cap}),
                optimizer=optimizer, checkpoint_manager=manager, board=DummyBoard(),
                resume_state=None if checkpoint is None else checkpoint.trainer_state,
            )

        uninterrupted = make_trainer("uninterrupted")
        uninterrupted.train()
        expected = uninterrupted.model.module.batches
        owned = order[rank::2]
        initial_opens = opens[:]
        assert initial_opens == owned
        assert len(expected) == 6

        resumed_opens = []
        for step, (shard, group) in enumerate([(1, 1), (2, 0) if rank == 0 else (1, 2)]):
            opens.clear()
            saved = uninterrupted.ckpt_manager.load_states(step=step, load_scaler=False)
            state = saved.trainer_state.train_loader_state
            assert (state.shard_idx, state.row_group_idx, state.offset_in_row_group) == (shard, group, 0)
            assert state.continuation_pending
            assert state.document_token_offset == (1, 5)[step]
            assert len(state.pending_document) == (2, 6)[step]
            assert [len(doc) for doc in state.lookahead_documents] == [6]
            assert state.ordered_shard_ids == tuple(order)
            assert (state.data_seed, state.packing_rank, state.packing_world_size) == (83, rank, 2)
            restarted = make_trainer(f"resume-{step}", saved)
            restarted.train()
            assert restarted.model.module.batches == expected[(step + 1) * 2:]
            assert opens == owned[shard:]
            resumed_opens.append(opens[:])

        opens.clear()
        loader = build_dataloader(**args, data_seed=83)
        content = []
        document_starts = 0
        reference_batches = []
        for x, y, state in loader:
            assert x[0, 0].item() == tokenizer.get_bos_token_id()
            reference_batches.append((x.tolist(), y.tolist()))
            row = [*x[0].tolist(), y[0, -1].item()]
            tokens = [token for token in row if token > 0]
            assert len(tokens) == state.source_content_tokens
            content.extend(tokens)
            document_starts += state.document_bos_tokens
        documents = [text for shard in owned for text in texts_by_shard[shard]]
        assert sorted(content) == sorted(int(token) for text in documents for token in text.split())
        assert document_starts == len(documents)
        assert expected == reference_batches[:len(expected)]
        assert opens == owned
        (tmp_path / f"io-{rank}.json").write_text(json.dumps([initial_opens, resumed_opens, opens]))

        saved = uninterrupted.ckpt_manager.load_states(step=2, load_scaler=False)
        short = make_trainer("shortfall", cap=4, trainer_class=DelegatingTrainer)
        with pytest.raises(TrainingDataExhaustedError, match=r"ranks \[1\].*update 4 of 4"):
            short.train()
        assert short.state.completed_steps == 3
        assert short.model.module.batches == expected
        restored = short.ckpt_manager.load_states(step=2, load_scaler=False)
        assert restored.trainer_state.train_loader_state == saved.trainer_state.train_loader_state
        torch.testing.assert_close(restored.model_state, saved.model_state, rtol=0, atol=0)
        torch.testing.assert_close(restored.optimizer_state, saved.optimizer_state, rtol=0, atol=0)
        assert_rng_state_consistency(restored.rng_state, saved.rng_state)

        exact = make_trainer("exact-cap", dataset="exact")
        take = exact.train_loader.next_full_batch
        reads = 0

        def count_reads(output):
            nonlocal reads
            reads += 1
            return take(output)

        exact.train_loader.next_full_batch = count_reads
        exact.train()
        assert exact.state.completed_steps == 3
        assert reads == len(exact.model.module.batches) == 6
        if rank == 1:
            assert take(exact.train_loader.allocate_training_buffers(1)[0]) is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_distributed_training_loader_ownership_and_exact_checkpoint_resume(tmp_path):
    dataset = tmp_path / "tokens"
    dataset.mkdir()
    exact = tmp_path / "exact"
    exact.mkdir()
    order = list(range(5))
    random.Random(83).shuffle(order)
    texts_by_shard = []
    for shard_id in range(5):
        tokens = iter(range(shard_id * 32 + 1, (shard_id + 1) * 32))
        lengths = [1, 1, 1, 5]
        if shard_id == order[-2]:
            lengths.append(5)
        texts = [" ".join(str(next(tokens)) for _ in range(length)) for length in lengths]
        texts_by_shard.append(texts)
        pq.write_table(pa.table({"text": texts}), dataset / f"shard_{shard_id:05d}.parquet", row_group_size=2)
        exact_docs = [" ".join(str(token) for token in range(start, start + 4)) for start in
                      (shard_id * 32 + 1, shard_id * 32 + 5, shard_id * 32 + 9)]
        pq.write_table(pa.table({"text": exact_docs}), exact / f"shard_{shard_id:05d}.parquet")
    mp.spawn(_training_loader_worker, args=(tmp_path, order, texts_by_shard), nprocs=2, join=True)

    rank0, rank1 = [json.loads((tmp_path / f"io-{rank}.json").read_text()) for rank in range(2)]
    for rank0_opens, rank1_opens in zip([rank0[0], *rank0[1], rank0[2]], [rank1[0], *rank1[1], rank1[2]]):
        assert set(rank0_opens).isdisjoint(rank1_opens)
    assert sorted(rank0[2] + rank1[2]) == list(range(5))
