"""Standalone experiment parsers, offline loaders, and mocked scaling runs."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest


@pytest.mark.parametrize('module_name', ['dataloaders', 'tokenizer_scaling'])
def test_standalone_entry_point(module_name, monkeypatch, capsys):
    module = importlib.import_module(f'experiments.{module_name}')
    execute = Mock()
    monkeypatch.setattr(module, 'run', execute)
    with pytest.raises(SystemExit) as error:
        module.main(['--help'])
    assert error.value.code == 0
    assert f'python -m experiments.{module_name}' in capsys.readouterr().out
    execute.assert_not_called()
    assert module.main([]) == 0
    assert vars(execute.call_args.args[0]) == vars(module.build_parser().parse_args([]))


def test_experiment_defaults(monkeypatch, tmp_path):
    from experiments.dataloaders import build_parser as dataloader_parser
    scaling = importlib.import_module('experiments.tokenizer_scaling')
    monkeypatch.setenv('GPTLAB_CACHE_DIR', str(tmp_path / 'cache'))
    bench = dataloader_parser().parse_args([])
    assert (bench.mode, bench.loader, bench.batch_size, bench.seq_len, bench.num_batches, bench.buffer_size, bench.num_docs, bench.device, bench.bos_id) == ('synthetic', 'all', 8, 2048, 20, 1000, 20000, 'cpu', 1)
    exp = scaling.build_parser().parse_args([])
    assert (exp.vocab_sizes, exp.baselines, exp.num_procs, exp.corpus_sizes_gb, exp.save_every, exp.board) == ('50000,70000,100000,200000', 'gpt2,cl100k_base,o200k_base', None, None, 10, 'dummy')
    assert exp.result_dir == str(tmp_path / 'cache/tokenizers/scaling_tokenizer_results')


def test_benchmark_aliases():
    from experiments.dataloaders import build_parser
    parser = build_parser()
    old = parser.parse_args('--batch_size 2 --seq_len 32 --num_batches 3 --buffer_size 16 --num_docs 8 --bos_id 7'.split())
    new = parser.parse_args('--batch-size 2 --seq-len 32 --num-batches 3 --buffer-size 16 --num-docs 8 --bos-id 7'.split())
    assert old == new


def test_synthetic_and_real_benchmarks_use_production_loaders(tmp_path):
    import numpy as np
    from experiments.dataloaders import build_parser, run
    args = build_parser().parse_args('--loader all --num-docs 8 --num-batches 2 --seq-len 8 --batch-size 1 --buffer-size 4'.split())
    results = run(args)
    assert [row['name'] for row in results] == ['glab-tokenized', 'glab-on-the-fly', 'nanochat']
    assert all(row['throughput_tok_s'] > 0 for row in results)
    assert results[-1]['bos_alignment'] == 1
    assert all(row['crop_rate'] is None for row in results)
    bin_path, idx_path = tmp_path / 'tokens.bin', tmp_path / 'tokens.idx'
    np.array([2, 3, 4, 5, 6, 7], dtype=np.uint32).tofile(bin_path)
    np.array([0, 3, 6], dtype=np.uint64).tofile(idx_path)
    args.mode, args.bin, args.idx = 'real', str(bin_path), str(idx_path)
    assert len(run(args)) == 3


def test_scaling_resume_metadata_and_no_import_downloads(tmp_path, monkeypatch):
    # Import the moved experiment with unavailable training deps stubbed. None
    # of the helpers exercised here should invoke training/data/network APIs.
    monkeypatch.setitem(sys.modules, 'gpt_lab.tokenizer.tokenizer', SimpleNamespace(Tokenizer=Mock()))
    monkeypatch.setitem(sys.modules, 'gpt_lab.tokenizer.corpus', SimpleNamespace(TokenizerCorpus=Mock()))
    monkeypatch.setitem(sys.modules, 'regex', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'tqdm', SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable))
    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(get=Mock(side_effect=AssertionError('network access'))))
    name = 'experiments.tokenizer_scaling'
    # Remove afterwards so mocked imports cannot leak into the existing suite.
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    try:
        results = tmp_path / 'results'
        results.mkdir()
        (results / 'meta.json').write_text('{"corpus_sizes_gb":"1"}')
        module.prepare_result_dir(results, resume=True)
        assert (results / 'meta.json').exists()
        module.prepare_result_dir(results, resume=False)
        assert list(results.iterdir()) == []
        assert (tmp_path / 'results_1/meta.json').exists()
        with pytest.raises(ValueError, match='corpus-sizes-gb'):
            module.validate_settings({})
        module.validate_settings(dict(corpus_sizes_gb='0.01,0.1', vocab_sizes='512,1024'))
        buffer = [({'score': 1}, 'run')]
        module.store_buffered_results(buffer, results)
        assert module.load_all_result_paths(results) == ['run']
        assert buffer == []
    finally:
        sys.modules.pop(name, None)


def test_experiment_resume_forwards_saved_corpus_and_process_cap(tmp_path, monkeypatch):
    import concurrent.futures
    import json
    import multiprocessing
    from concurrent.futures import Future

    corpus = Mock()
    corpus_type = Mock()
    corpus_type.from_path.return_value = corpus
    monkeypatch.setitem(sys.modules, 'gpt_lab.tokenizer.tokenizer', SimpleNamespace(Tokenizer=Mock()))
    monkeypatch.setitem(sys.modules, 'gpt_lab.tokenizer.corpus', SimpleNamespace(TokenizerCorpus=corpus_type))
    monkeypatch.setitem(sys.modules, 'regex', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'tqdm', SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable))
    monkeypatch.setitem(sys.modules, 'tiktoken', SimpleNamespace(get_encoding=Mock()))
    monkeypatch.setattr(multiprocessing, 'set_start_method', Mock())
    monkeypatch.setattr('os.cpu_count', lambda: 128)
    future = Future()
    future.set_result({'vocab_size': 512})
    executor = MagicMock()
    executor.submit.return_value = future
    pool = MagicMock()
    pool.return_value.__enter__.return_value = executor
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', pool)
    module_name = 'experiments.tokenizer_scaling'
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    try:
        result_dir = tmp_path / 'results'
        result_dir.mkdir()
        corpus_dir = tmp_path / 'chosen-corpus'
        corpus_dir.mkdir()
        meta = dict(corpus_dir=str(corpus_dir), corpus_sizes_gb='0.01', vocab_sizes='512',
                    pat_strs='gpt2', baselines='gpt2', num_procs=None, seed=123)
        (result_dir / 'meta.json').write_text(json.dumps(meta))
        module.store_single_result({'baseline': 'gpt2'}, result_dir / 'gpt2.pkl')
        args = module.build_parser().parse_args(['--resume'])
        args.result_dir = str(result_dir)
        module.run(args)
        pool.assert_called_once_with(max_workers=32)
        corpus_type.from_path.assert_called_once_with(corpus_dir)
        task = executor.submit.call_args.args[1]
        assert task[5] == corpus_dir and task[-1]['seed'] == 123
        assert task[4] == int(0.01 * 1024**3) and isinstance(task[4], int)
        assert len(list(result_dir.glob('*.pkl'))) == 2
        assert json.loads((result_dir / 'meta.json').read_text()) == meta
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("save_checkpoint", [False, True], ids=["wandb-only", "checkpointed"])
def test_study_launch_retries_missing_shards_and_protects_existing_run(tmp_path, monkeypatch, capsys, save_checkpoint):
    import json
    import os
    from shutil import copytree

    import pyarrow as pa
    import pyarrow.parquet as pq

    from gpt_lab.model.flash_attn import load_flash_attn
    from gpt_lab.tokenizer import build_tokenizer
    from gpt_lab.utils.schemas import TokenizerConfig, TrainerState

    package = "experiments.20261118-rope-paired-muon"
    launcher = importlib.import_module(f"{package}.train")
    monkeypatch.setitem(sys.modules, "optimizer", importlib.import_module(f"{package}.optimizer"))
    experiment_trainer = importlib.import_module(f"{package}.trainer")
    monkeypatch.setitem(sys.modules, "trainer", experiment_trainer)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "WANDB_RESUME", "DEVELOPMENT", "PYTORCH_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GPTLAB_DTYPE", "bfloat16")
    monkeypatch.setattr("gpt_lab.data.loader.DATA_DIR", tmp_path / "data")
    tokenizer = build_tokenizer(TokenizerConfig(name="dummy", source="dummy", vocab_size=257,
                                                dirname=tmp_path / "tokenizer", save_token_bytes=False))
    monkeypatch.setattr("gpt_lab.model.auto.AutoGPTConfig.build_tokenizer", lambda self: tokenizer)
    monkeypatch.setattr("gpt_lab.model.layers.load_flash_attn", lambda backend: load_flash_attn("sdpa"))

    dataset = tmp_path / "data" / "prepared"
    dataset.mkdir(parents=True)
    (dataset / "meta.json").write_text(json.dumps({"name": "prepared", "base_url": "",
                                                   "column_name": "text", "max_shards": 1}))
    shard = pa.table({"text": ["A local prepared shard."]})
    pq.write_table(shard, dataset / "shard_00001.parquet")
    model_dir = tmp_path / "models"
    run_dir = model_dir / launcher.STUDY_MODEL / "real-s7"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text('"abandoned setup"')
    monkeypatch.setattr(sys, "argv", [str(Path(launcher.__file__)), "--pairing", "real", "auto",
                                     "--model-name", launcher.STUDY_MODEL, "--model-dir", str(model_dir),
                                     "--device", "cpu", "--ds-name", "prepared", "--board", "wandb",
                                     "--random-seed", "7",
                                     "--depth", "1", "--aspect-ratio", "16", "--d-head", "8",
                                     "--max-seq-len", "8", "--device-batch-size", "1", "--num-steps", "1",
                                     "--total-batch-size", "8", "--n-acc-steps", "1"])

    def reject_launch(message, *, resume_hint=False):
        with pytest.raises(SystemExit) as error:
            launcher.main()
        assert error.value.code == 2
        stderr = capsys.readouterr().err
        assert message in stderr
        assert ("Use resume --checkpoint-dir" in stderr) is resume_hint
        return stderr

    def open_wandb(**kwargs):
        assert os.environ["WANDB_RESUME"] == "never"
        return SimpleNamespace(disabled=False)

    wandb = SimpleNamespace(init=Mock(side_effect=open_wandb), finish=Mock())
    monkeypatch.setattr("gpt_lab.utils.board.wandb", wandb)

    def build_trainer(*, checkpoint_manager, model, optimizer, monitor_qk_norms, **kwargs):
        assert monitor_qk_norms is True
        assert optimizer.backend.strategies["muon"].sample_names

        def train():
            reject_launch("A launch is already active")
            if save_checkpoint:
                checkpoint_manager.save(0, model, optimizer=optimizer, trainer_state=TrainerState(
                    step=0, completed_steps=1, train_loader_state=None,
                ))

        return SimpleNamespace(train=train)

    monkeypatch.setattr(experiment_trainer, "ExperimentTrainer", build_trainer)
    reject_launch("Missing train shards")
    wandb.init.assert_not_called()
    assert not (run_dir / ".wandb-run").exists()

    pq.write_table(shard, dataset / "shard_00000.parquet")
    launcher.main()

    assert (run_dir / ".wandb-run").is_file()
    wandb.init.assert_called_once()
    assert wandb.init.call_args.kwargs["id"] == "real-s7"
    assert wandb.init.call_args.kwargs["config"]["random_seed"] == 7
    assert wandb.init.call_args.kwargs["config"]["data_seed"] == 42
    complete = run_dir / "base" / "checkpoint_step_000000"
    if save_checkpoint:
        partial = copytree(complete, complete.with_name("checkpoint_step_000001"))
        (partial / "optim.safetensors").write_bytes(b"incomplete")
    stderr = reject_launch("already contains run artifacts", resume_hint=save_checkpoint)
    if save_checkpoint:
        assert str(complete) in stderr
    wandb.init.assert_called_once()


@pytest.mark.parametrize("bpb_every,core_every", [(400, 2000), (0, 2000), (400, 0)])
def test_study_evaluation_resumes_saved_schedule(tmp_path, monkeypatch, bpb_every, core_every):
    import itertools
    import json

    import torch

    from gpt_lab.model.checkpoint import CheckpointManager
    from gpt_lab.utils.distributed import ComputeContext
    from gpt_lab.utils.schemas import TrainerConfig

    package = "experiments.20261118-rope-paired-muon"
    launcher = importlib.import_module(f"{package}.train")
    experiment = importlib.import_module(f"{package}.trainer")
    monkeypatch.setitem(sys.modules, "trainer", experiment)
    monkeypatch.setitem(sys.modules, "optimizer", importlib.import_module(f"{package}.optimizer"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("GPTLAB_DTYPE", "bfloat16")
    monkeypatch.setenv("DEVELOPMENT", "0")
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    for name in ("collect", "freeze", "disable"):
        monkeypatch.setattr(f"gpt_lab.train.trainer.gc.{name}", lambda: None)

    context = ComputeContext(
        device=torch.device("cpu"), compute_dtype=torch.float32,
        mesh=SimpleNamespace(size=lambda: 8, get_rank=lambda: 0),
    )
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(max_context=2048)
    tokens = torch.ones(1, 2)
    board = Mock()
    val_loader = Mock(return_value=())
    calls = []
    run_dir = tmp_path / launcher.STUDY_MODEL / "real-s42-stop9746"
    endpoint = run_dir / "base" / "checkpoint_step_019489"
    bpb_result = {"bpb": 1.5, "loss": 2.0}

    def bounded_bpb(model, batches, budget, **kwargs):
        val_loader.assert_called_with()
        calls.append(("bpb", trainer.state.completed_steps, budget))
        return bpb_result

    def core(model, tokenizer, device, *, max_per_task):
        calls.append(("core", trainer.state.completed_steps, max_per_task))
        return {"core/core": 0.6, "core/accuracy": 0.7, "core/max_per_task": max_per_task}

    def full_bpb(model, batches, budget, *, comp_info, token_bytes):
        assert endpoint.joinpath("model.safetensors").is_file()
        assert comp_info.world_size == 8 and budget is None and not model.training
        val_loader.assert_called_with(include_partial=True)
        calls.append(("full", trainer.state.completed_steps, budget))
        return bpb_result

    def execute(args, *, trainer_factory, configure_optimizer):
        nonlocal trainer
        manager = CheckpointManager(
            args.model_name, args.run_name, args.model_dir, comp_info=context, optimizer_state_layout="sharded",
        )
        saved = None
        if args.model_init == "auto":
            config = TrainerConfig(
                n_steps=19490, n_acc_steps=2, total_batch_size=524288, device_batch_size=16,
                comp_info=context, stop_after_step=args.stop_after_step, eval_bpb_every=args.eval_bpb_every,
                n_bpb_tokens=args.n_bpb_tokens, eval_core_every=args.eval_core_every, n_core_tokens=args.n_core_tokens,
                log_every=0, sample_every=0, save_on_best=False,
            )
            manager.save_training_config(config)
            (run_dir / "meta.json").touch()
        else:
            config = manager.load_training_config().model_copy(update={"stop_after_step": args.stop_after_step})
            saved = manager.load_states(load_model=False, load_optim=False, load_rng=False, load_scaler=False)
        trainer = trainer_factory(
            model=model, tokenizer=SimpleNamespace(token_bytes=torch.ones(2)),
            train_loader=itertools.repeat((tokens, tokens)), val_loader=val_loader, config=config,
            checkpoint_manager=manager, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), board=board,
            resume_state=saved.trainer_state if saved else None, best_state=saved.checkpoint_state if saved else None,
        )
        trainer.train_step = lambda _, batch, __: (batch, {})
        trainer.update_metrics = lambda *_: {}
        trainer.train()

    trainer = None
    monkeypatch.setattr(launcher, "run", execute)
    monkeypatch.setattr("gpt_lab.train.trainer.compute_bpb", bounded_bpb)
    monkeypatch.setattr("gpt_lab.train.trainer.evaluate_core", core)
    monkeypatch.setattr(experiment, "compute_bpb", full_bpb)
    command = [launcher.__file__, "--pairing", "real", "--no-monitor-qk-norms"]
    monkeypatch.setattr(sys, "argv", command + [
        "--study-eval", "auto", "--model-name", launcher.STUDY_MODEL, "--model-dir", str(tmp_path),
        "--board", "dummy", "--stop-after-step", "9746", "--eval-bpb-every", str(bpb_every),
        "--n-bpb-tokens", "4194304", "--eval-core-every", str(core_every), "--n-core-tokens", "500",
    ])
    launcher.main()
    assert trainer.state.completed_steps == 9746
    assert (run_dir / experiment.STUDY_EVAL_FILE).is_file()
    assert not trainer.endpoint_bpb_path.exists()

    monkeypatch.setattr(sys, "argv", command + [
        "resume", "--checkpoint-dir", str(run_dir), "--stop-after-step", "19490", "--board", "dummy",
    ])
    launcher.main()
    if bpb_every:
        assert json.loads(trainer.endpoint_bpb_path.read_text()) == {
            "eval/full_endpoint_update": 19490, "eval/bpb": 1.5, "eval/loss": 2.0,
        }
    else:
        assert not trainer.endpoint_bpb_path.exists()
    weights = endpoint / "model.safetensors"
    saved_mtime = weights.stat().st_mtime_ns
    launcher.main()
    assert weights.stat().st_mtime_ns == saved_mtime
    bpb_steps = range(400, 19490, 400) if bpb_every else ()
    core_steps = (4873, 9745, 14618) if core_every else ()
    expected = sorted(
        [("bpb", step, 16) for step in bpb_steps] + [("core", step, 500) for step in core_steps],
        key=lambda call: call[1],
    )
    if bpb_every:
        expected.append(("full", 19490, None))
    assert calls == expected
