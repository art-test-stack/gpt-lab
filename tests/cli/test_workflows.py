"""Bounded workflow checks. Training and external services are mocked."""

import argparse
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from gpt_lab.cli import main
from gpt_lab.cli.commands.base import resolve_checkpoint
from gpt_lab.cli.commands.cache import run as clear_cache
from gpt_lab.cli.parser import build_parser


def parse(command):
    return build_parser().parse_args(command.split())


@pytest.mark.parametrize('suffix,step', [('', None), ('/base', None), ('/base/checkpoint_step_000120', 120)])
def test_explicit_checkpoint_resolves_location_without_mutating_args(tmp_path, suffix, step):
    run = tmp_path / 'models/demo/run-1'
    checkpoint = run / 'base/checkpoint_step_000120'
    checkpoint.mkdir(parents=True)
    (run / 'meta.json').write_text('{}')
    args = parse('train base resume')
    args.checkpoint_dir = str(run) + suffix
    resolved = resolve_checkpoint(args)
    assert (resolved.model_dir, resolved.model_name, resolved.run_name, resolved.checkpoint_step) == (str(tmp_path / 'models'), 'demo', 'run-1', step)
    assert args.model_name == 'ic1'
    if step is not None:
        args.checkpoint_step = 7
        with pytest.raises(ValueError, match='conflicts'):
            resolve_checkpoint(args)


def test_missing_checkpoint_fails_before_execution(tmp_path):
    args = parse('train base resume')
    args.checkpoint_dir = str(tmp_path / 'missing')
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(args)


@pytest.mark.parametrize('subdirs,deleted', [([], ['tokenizers', 'data', 'models']), (['all'], ['tokenizers', 'data', 'models']), (['tokenizer'], ['tokenizers']), (['corpus'], ['data/corpus'])])
def test_cache_selection(tmp_path, subdirs, deleted):
    root = tmp_path / 'cache'
    for folder in ('tokenizers', 'data/corpus', 'models'):
        (root / folder).mkdir(parents=True)
        (root / folder / 'sentinel').write_text('keep unless selected')
    args = argparse.Namespace(cache_dir=str(root), subdirs=subdirs, force=True)
    assert clear_cache(args) == 0
    for folder in deleted:
        assert not (root / folder).exists()
    if subdirs == ['tokenizer']:
        assert (root / 'data/corpus/sentinel').exists()
    if subdirs == ['corpus']:
        assert (root / 'tokenizers/sentinel').exists()


def test_cache_cancel_and_symlink_safety(tmp_path, monkeypatch):
    root = tmp_path / 'cache'
    root.mkdir()
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'sentinel').touch()
    (root / 'models').symlink_to(external, target_is_directory=True)
    args = argparse.Namespace(cache_dir=str(root), subdirs=['models'], force=False)
    monkeypatch.setattr('builtins.input', lambda _: 'n')
    assert clear_cache(args) == 1
    assert (root / 'models').is_symlink()
    args.force = True
    assert clear_cache(args) == 0
    assert not (root / 'models').exists()
    assert (external / 'sentinel').exists()
    (root / 'data').symlink_to(external, target_is_directory=True)
    args.subdirs = ['corpus']
    with pytest.raises(ValueError, match='escapes'):
        clear_cache(args)


@pytest.fixture
def training_dependencies(monkeypatch):
    class Config(SimpleNamespace):
        def model_dump(self):
            return vars(self)

    model = MagicMock()
    model.embeds.weight.device.type = 'cpu'
    model.config.max_context = 64
    model.config.model_dump.return_value = {}
    model.to_empty.return_value = model
    tokenizer = Mock()
    meta = Config(name='demo', run_name='run-1', model_cfg=model.config, tokenizer_cfg=object(),
                  base_train={'device_batch_size': 3, 'n_steps': 10})
    saved_config = Config(dist_info={})
    state = SimpleNamespace(train_loader_state=object())
    checkpoint = SimpleNamespace(trainer_state=state, checkpoint_state=object(), optimizer_state=object(), scaler_state=object())
    manager = Mock()
    manager.load.return_value = model, tokenizer, checkpoint, saved_config
    factory = Mock(return_value=manager)
    auto = Mock()
    auto.return_value.generate_gpt_config.return_value = meta
    loader = Mock(side_effect=['train-loader', 'val-loader'])
    trainer = Mock()
    trainer_factory = Mock(return_value=trainer)
    board = Mock()
    board_factory = Mock(return_value=board)
    dist_info = {'RANK': 0, 'DEVICE': 'cpu', 'DEVICE_TYPE': 'cpu', 'IS_DDP_INITIALIZED': True, 'WORLD_SIZE': 2}
    init_dist = Mock(return_value=dist_info)
    cleanup = Mock()
    load_meta = Mock(return_value=meta)
    modules = {
        'gpt_lab.utils.common': dict(get_banner=Mock(), print0_dict=Mock()),
        'gpt_lab.utils.logging': dict(init_logger=Mock(), log0=Mock()),
        'gpt_lab.utils.system': dict(get_git_info=Mock(return_value={}), get_gpu_info=Mock(return_value={}), get_system_info=Mock(return_value={})),
        'gpt_lab.utils.distributed': dict(get_device_type=Mock(return_value='cpu'), init_dist_groups=init_dist, cleanup_dist_groups=cleanup, broadcast_model=Mock()),
        'gpt_lab.utils.schemas': dict(TrainerConfig=Config),
        'gpt_lab.tokenizer': dict(Tokenizer=SimpleNamespace(from_config=Mock(return_value=tokenizer))),
        'gpt_lab.model.auto': dict(AutoGPTConfig=auto),
        'gpt_lab.model.checkpoint': dict(CheckpointManager=factory, build_meta_model=Mock(return_value=model), load_meta_config=load_meta),
        'gpt_lab.data.loader': dict(build_dataloader=loader),
        'gpt_lab.utils.board': dict(Board=board_factory, DummyBoard=Mock()),
        'gpt_lab.train.trainer': dict(Trainer=trainer_factory),
    }
    for name, attrs in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))
    monkeypatch.setenv('PYTORCH_ALLOC_CONF', 'test-original')
    return SimpleNamespace(**locals())


@pytest.mark.parametrize('step', [None, 0, 120, -2, 'best'])
def test_resume_restores_all_states_through_existing_pipeline(training_dependencies, step):
    from gpt_lab.workflows.train_base import run
    deps = training_dependencies
    args = parse('train base resume --model-name demo --run-name run-1 --device cpu --board dummy')
    args.checkpoint_step = step
    run(args)
    deps.manager.load.assert_called_once_with(step='latest' if step is None else step, phase='train')
    deps.auto.assert_not_called()
    assert deps.factory.call_args.kwargs['mode'] == 'shard'
    assert deps.loader.call_args_list[0].kwargs['resume_state'] is deps.state.train_loader_state
    deps.model.build_optimizer.return_value.load_state_dict.assert_called_once_with(deps.checkpoint.optimizer_state)
    assert deps.trainer_factory.call_args.kwargs['config'] is deps.saved_config
    assert deps.trainer_factory.call_args.kwargs['resume_state'] is deps.state
    deps.trainer.scaler.load_state_dict.assert_called_once_with(deps.checkpoint.scaler_state)
    deps.trainer.train.assert_called_once()
    deps.board.close.assert_called_once()
    deps.cleanup.assert_called_once()
    assert '_handler' not in deps.board_factory.call_args.kwargs['config']


def test_auto_preserves_configuration_and_cleanup_on_failure(training_dependencies):
    from gpt_lab.workflows.train_base import run
    deps = training_dependencies
    deps.trainer.train.side_effect = RuntimeError('simulated failure')
    args = parse('train base auto --depth 4 --train-tokenizer --vocab-size 512 --n-kv-heads 2 --board dummy')
    with pytest.raises(RuntimeError, match='simulated failure'):
        run(args)
    cfg = deps.auto.call_args.kwargs
    assert (cfg['depth'], cfg['vocab_size'], cfg['n_kv_heads'], cfg['tokenizer_model']) == (4, 512, 2, 'auto')
    deps.manager.load.assert_not_called()
    deps.model.init_weights.assert_called_once()
    deps.board.close.assert_called_once()
    deps.cleanup.assert_called_once()


def test_torchrun_module_argument_forwarding():
    # Only parse torchrun's arguments; never call its launcher or rendezvous.
    from torch.distributed.run import parse_args
    args = parse_args('--standalone --nproc_per_node=2 -m gpt_lab.cli train base resume --checkpoint-step -2'.split())
    assert args.module
    assert args.training_script == 'gpt_lab.cli'
    assert args.training_script_args == ['train', 'base', 'resume', '--checkpoint-step', '-2']


def test_tokenizer_config_and_unicode_limits():
    from gpt_lab.workflows.train_tokenizer import build_config, limit_characters
    args = parse('tokenizer train --trainer huggingface --name example --vocab-size 512 --num-proc 2')
    config = build_config(args)
    assert (config.name, config.vocab_size, config.trainer.source, config.trainer.num_proc) == ('example', 512, 'huggingface', 2)
    assert list(limit_characters(['éあbc', 'déf'], 5, 3)) == ['éあb', 'dé']
    assert list(limit_characters(['hello'], 0, -1)) == []


def test_tokenizer_executes_with_local_text_and_mocked_trainer(tmp_path, monkeypatch):
    from gpt_lab.workflows.train_tokenizer import run
    path = tmp_path / 'corpus.txt'
    path.write_text('éあbc\ndéf\n')
    captured = {}
    def train(text_iterator, config):
        captured['texts'] = list(text_iterator)
        captured['config'] = config
        return SimpleNamespace(encode=lambda text: list(map(ord, text)), decode=lambda ids: ''.join(map(chr, ids)), mergeable_ranks={b'a': 0})
    monkeypatch.setitem(sys.modules, 'gpt_lab.tokenizer', SimpleNamespace(Tokenizer=SimpleNamespace(train_from_iterator=train)))
    monkeypatch.setitem(sys.modules, 'gpt_lab.tokenizer.corpus', SimpleNamespace(TokenizerCorpus=Mock()))
    args = parse('tokenizer train --trainer huggingface --max-chars 5 --chars-per-doc 3')
    args.corpus_path = str(path)
    run(args)
    assert captured['texts'] == ['éあb', 'dé']


def test_synthetic_and_real_benchmarks_use_production_loaders(tmp_path):
    import numpy as np
    from gpt_lab.workflows.dataloaders import run
    args = parse('benchmark dataloaders --loader all --num-docs 8 --num-batches 2 --seq-len 8 --batch-size 1 --buffer-size 4')
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
    name = 'gpt_lab.workflows.tokenizer_scaling'
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


def test_chat_app_builds_and_launches_only_when_executed(monkeypatch):
    from gpt_lab.workflows.chat import app
    application = MagicMock()
    blocks = MagicMock()
    blocks.return_value.__enter__.return_value = application
    load_dotenv = Mock()
    chat_interface = Mock()
    benchmark_interface = Mock()
    monkeypatch.setitem(sys.modules, 'dotenv', SimpleNamespace(load_dotenv=load_dotenv))
    monkeypatch.setitem(sys.modules, 'gradio', SimpleNamespace(Blocks=blocks, Tab=MagicMock(), Markdown=Mock()))
    monkeypatch.setitem(sys.modules, 'gpt_lab.interface.chat', SimpleNamespace(chatapp_interface=chat_interface))
    monkeypatch.setitem(sys.modules, 'gpt_lab.interface.benchmark', SimpleNamespace(benchmark_interface=benchmark_interface))
    app(parse('chat app'))
    load_dotenv.assert_called_once()
    chat_interface.assert_called_once()
    benchmark_interface.assert_called_once()
    application.launch.assert_called_once_with()


def test_console_loads_local_checkpoint_and_exits_without_generation(monkeypatch):
    from gpt_lab.workflows.chat import console
    model, tokenizer, engine = Mock(), Mock(), Mock()
    build_model = Mock(return_value=(model, tokenizer, None, None))
    engine_factory = Mock(return_value=engine)
    monkeypatch.setitem(sys.modules, 'gpt_lab.model.checkpoint', SimpleNamespace(build_model=build_model))
    monkeypatch.setitem(sys.modules, 'gpt_lab.model.wrapper', SimpleNamespace(Engine=engine_factory))
    monkeypatch.setattr('builtins.input', lambda _: '/exit')
    console(parse('chat console --model-name demo --checkpoint-step 0 --device cpu'))
    assert build_model.call_args.kwargs['model_name'] == 'demo'
    assert build_model.call_args.kwargs['step'] == 0
    assert build_model.call_args.kwargs['phase'] == 'eval'
    engine.generate.assert_not_called()


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
    module_name = 'gpt_lab.workflows.tokenizer_scaling'
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
        args = parse('experiment tokenizer-scaling --resume')
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


@pytest.mark.parametrize('run_name', ['base', 'checkpoint_step_custom'])
def test_explicit_run_directory_name_does_not_override_meta_marker(tmp_path, run_name):
    run = tmp_path / 'models/demo' / run_name
    run.mkdir(parents=True)
    (run / 'meta.json').write_text('{}')
    args = parse('train base resume')
    args.checkpoint_dir = str(run)
    assert resolve_checkpoint(args).run_name == run_name
