"""Bounded workflow checks. Model training and external services are mocked."""

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
    args = parse('train resume')
    args.checkpoint_dir = str(run) + suffix
    resolved = resolve_checkpoint(args)
    assert (resolved.model_dir, resolved.model_name, resolved.run_name, resolved.checkpoint_step) == (str(tmp_path / 'models'), 'demo', 'run-1', step)
    assert args.model_name == 'ic1'
    if step is not None:
        args.checkpoint_step = 7
        with pytest.raises(ValueError, match='conflicts'):
            resolve_checkpoint(args)


def test_missing_checkpoint_fails_before_execution(tmp_path):
    args = parse('train resume')
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

        def group_configs(self):
            return self.groups

    profile = Config(groups={})
    Config.from_file = Mock(return_value=profile)

    model = MagicMock()
    model.embeds.weight.device.type = 'cpu'
    model.config.max_context = 64
    model.config.model_dump.return_value = {}
    model.to_empty.return_value = model
    tokenizer = Mock()
    meta = Config(name='demo', run_name='run-1', model_cfg=model.config, tokenizer_cfg=object(),
                  base_train={'device_batch_size': 3, 'n_steps': 10})
    saved_config = Config(comp_info={})
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
    comp_info = SimpleNamespace(rank=0, world_size=2, device='cpu', is_distributed=True, device_type='cpu')
    init_dist = Mock(return_value=comp_info)
    cleanup = Mock()
    load_meta = Mock(return_value=meta)
    modules = {
        'gpt_lab.utils.common': dict(get_banner=Mock(), print0_dict=Mock()),
        'gpt_lab.utils.logging': dict(init_logger=Mock(), log0=Mock()),
        'gpt_lab.utils.system': dict(get_git_info=Mock(return_value={}), get_gpu_info=Mock(return_value={}), get_system_info=Mock(return_value={})),
        'gpt_lab.utils.distributed': dict(get_device_type=Mock(return_value='cpu'), init_dist_groups=init_dist, cleanup_dist_groups=cleanup, broadcast_model=Mock()),
        'gpt_lab.utils.schemas': dict(TrainerConfig=Config, OptimizerConfig=Config),
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
    from gpt_lab.cli.commands.base import run
    deps = training_dependencies
    args = parse('train resume --model-name demo --run-name run-1 --device cpu --board dummy')
    args.checkpoint_step = step
    run(args)
    deps.manager.load.assert_called_once_with(step='latest' if step is None else step, phase='train')
    deps.auto.assert_not_called()
    assert deps.factory.call_args.kwargs['optimizer_state_layout'] == 'sharded'
    assert deps.loader.call_args_list[0].kwargs['resume_state'] is deps.state.train_loader_state
    deps.model.build_optimizer.return_value.load_state_dict.assert_called_once_with(deps.checkpoint.optimizer_state)
    assert deps.trainer_factory.call_args.kwargs['config'] is deps.saved_config
    assert deps.trainer_factory.call_args.kwargs['resume_state'] is deps.state
    deps.trainer.scaler.load_state_dict.assert_called_once_with(deps.checkpoint.scaler_state)
    deps.trainer.train.assert_called_once()
    deps.board.close.assert_called_once()
    deps.cleanup.assert_called_once()
    assert '_handler' not in deps.board_factory.call_args.kwargs['config']


@pytest.mark.parametrize('mode', ['auto', 'resume'])
@pytest.mark.parametrize('backend', ['wandb', 'tensorboard', 'dummy'])
def test_training_forwards_logging_backend(training_dependencies, mode, backend):
    from gpt_lab.cli.commands.base import run
    deps = training_dependencies
    run(parse(f'train {mode} --device cpu --board {backend}'))
    assert deps.board_factory.call_args.kwargs['board_type'] == backend
    assert deps.board_factory.call_args.kwargs['resume'] is (mode == 'resume')
    assert deps.trainer_factory.call_args.kwargs['board'] is deps.board
    deps.board.close.assert_called_once()


def test_auto_preserves_configuration_and_cleanup_on_failure(training_dependencies):
    from gpt_lab.cli.commands.base import run
    deps = training_dependencies
    deps.trainer.train.side_effect = RuntimeError('simulated failure')
    args = parse('train --depth 4 --train-tokenizer --vocab-size 512 --n-kv-heads 2 --board dummy --tie-embeddings embs,head,h_embs:')
    with pytest.raises(RuntimeError, match='simulated failure'):
        run(args)
    cfg = deps.auto.call_args.kwargs
    assert (cfg['depth'], cfg['vocab_size'], cfg['n_kv_heads'], cfg['tokenizer_model']) == (4, 512, 2, 'auto')
    assert cfg['tie_embeddings'] == 'embs,head,h_embs:'
    deps.manager.load.assert_not_called()
    deps.model.init_weights.assert_called_once()
    deps.board.close.assert_called_once()
    deps.cleanup.assert_called_once()


def test_torchrun_module_argument_forwarding():
    # Only parse torchrun's arguments; never call its launcher or rendezvous.
    from torch.distributed.run import parse_args
    args = parse_args('--standalone --nproc_per_node=2 -m gpt_lab.cli train resume --checkpoint-step -2'.split())
    assert args.module
    assert args.training_script == 'gpt_lab.cli'
    assert args.training_script_args == ['train', 'resume', '--checkpoint-step', '-2']


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


def test_standalone_tokenizer_training_saves_loadable_local_config(tmp_path):
    from gpt_lab.tokenizer import Tokenizer, TokenizerConfig
    from gpt_lab.workflows.train_tokenizer import build_config

    args = parse('tokenizer train --trainer huggingface --name ic1_tokenizer --vocab-size 257 --num-proc 1')
    config = build_config(args)
    config.dirname = tmp_path / config.name
    config.trainer.show_progress = False
    sample = 'Hello, world! éあ'
    trained = Tokenizer.train_from_iterator(iter([sample]), config)

    saved_config = TokenizerConfig.from_directory(args.name, cachedir=tmp_path)
    assert saved_config.source == 'local'
    assert saved_config.trainer.source == 'huggingface'
    loaded = Tokenizer.from_config(saved_config)
    assert loaded.mergeable_ranks == trained.mergeable_ranks
    assert loaded.encode(sample) == trained.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


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


def test_console_generates_from_nonempty_prompt_on_cpu(monkeypatch, capsys):
    from gpt_lab.model.gpt import DenseTransformer
    from gpt_lab.tokenizer import Tokenizer, TokenizerConfig
    from gpt_lab.utils.schemas import TransformerConfig
    from gpt_lab.workflows.chat import console

    tokenizer_config = TokenizerConfig(
        name='console-test', source='local', vocab_size=257, save_token_bytes=False,
    )
    tokenizer = Tokenizer(
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens=tokenizer_config.special_tokens.list(), config=tokenizer_config,
    )
    model = DenseTransformer(TransformerConfig(
        vocab_size=tokenizer.vocab_size, max_context=16, d_model=16, d_ffn=32,
        n_heads=2, n_kv_heads=2, n_layers=1, d_head=8,
        window_pattern='L', attn_impl='sdpa',
    ), pad_vocab_size_to=1).to('cpu')
    model.init_weights()
    model.eval()
    generated = []
    model.register_forward_hook(
        lambda _, inputs, output: generated.append(output.logits[0, -1].argmax().item())
    )
    monkeypatch.setattr('gpt_lab.model.checkpoint.build_model', lambda **kwargs: (model, tokenizer, None, None))
    prompts = iter(['Hi', '/exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(prompts))

    console(parse('chat console --model-name demo --device cpu --max-tokens 2'))

    assert len(generated) == 2
    assert 'Model: ' + tokenizer.decode(generated) in capsys.readouterr().out


@pytest.mark.parametrize('run_name', ['base', 'checkpoint_step_custom'])
def test_explicit_run_directory_name_does_not_override_meta_marker(tmp_path, run_name):
    run = tmp_path / 'models/demo' / run_name
    run.mkdir(parents=True)
    (run / 'meta.json').write_text('{}')
    args = parse('train resume')
    args.checkpoint_dir = str(run)
    assert resolve_checkpoint(args).run_name == run_name


def test_training_is_callable_without_argparse(training_dependencies, monkeypatch):
    from gpt_lab.workflows.train_base import train_base
    deps = training_dependencies
    monkeypatch.setattr('gpt_lab.cli.parser.build_parser', Mock(side_effect=AssertionError('Python training must not parse CLI arguments')))
    train_base(optim_config=deps.profile, depth=4, total_batch_size=128, num_steps=7,
               device='cpu', board='dummy', random_seed=123)
    assert deps.auto.call_args.kwargs['depth'] == 4
    assert deps.auto.call_args.kwargs['total_batch_size'] == 128
    assert deps.auto.call_args.kwargs['n_steps'] == 7
    deps.init_dist.assert_called_once_with(device_type='cpu', random_seed=123)
    assert deps.model.build_optimizer.call_args.kwargs['optim_config'] is deps.profile
    deps.trainer.train.assert_called_once()
    deps.cleanup.assert_called_once()


@pytest.mark.parametrize('conflict', [False, True])
def test_python_resume_rejects_optimizer_profile_conflicts(training_dependencies, conflict):
    from gpt_lab.workflows.train_base import train_base
    deps = training_dependencies
    deps.saved_config.optim_config = deps.profile
    profile = deps.Config(groups={'changed': {}}) if conflict else deps.profile
    if conflict:
        with pytest.raises(ValueError, match='conflicts with the saved optimizer profile'):
            train_base(model_init='resume', optim_config=profile, device='cpu', board='dummy')
        deps.trainer.train.assert_not_called()
    else:
        train_base(model_init='resume', optim_config=profile, device='cpu', board='dummy')
        deps.trainer.train.assert_called_once()
    deps.cleanup.assert_called_once()
