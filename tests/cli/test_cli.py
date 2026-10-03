"""CLI regression tests: no training, datasets, distributed groups, or UI."""

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gpt_lab.cli import main
from gpt_lab.cli.parser import build_parser

ROOT = Path(__file__).resolve().parents[2]
LEAVES = [
    ('tokenizer train', 'tokenizer', 'run', 'train_tokenizer', 'run'),
    ('train base auto', 'base', 'run_auto', 'train_base', 'run'),
    ('train base resume', 'base', 'run_resume', 'train_base', 'run'),
    ('benchmark dataloaders', 'benchmark', 'run', 'dataloaders', 'run'),
    ('experiment tokenizer-scaling', 'experiment', 'run', 'tokenizer_scaling', 'run'),
    ('data reshard', 'data', 'run', 'reshard', 'run'),
    ('cache clear', 'cache', 'run', None, None),
    ('chat console', 'chat', 'run_console', 'chat', 'console'),
    ('chat app', 'chat', 'run_app', 'chat', 'app'),
]
WRAPPERS = {
    'scripts.train_tokenizer': 'tokenizer train',
    'scripts.train_base': 'train base',
    'scripts.benchmark.dataloaders': 'benchmark dataloaders',
    'scripts.benchmark.tokenizer_corpus_size': 'experiment tokenizer-scaling',
    'scripts.reshard_dataset': 'data reshard',
    'scripts.clear_gpt_lib_cache': 'cache clear',
    'scripts.chat_console': 'chat console',
    'scripts.chat_app': 'chat app',
}


@pytest.mark.parametrize('prefix', ['', 'tokenizer', 'train', 'benchmark', 'experiment', 'data', 'cache', 'chat', 'train base'] + [row[0] for row in LEAVES])
def test_help(prefix, capsys):
    with pytest.raises(SystemExit) as error:
        main(prefix.split() + ['--help'])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert 'usage: gpt-lab' in output
    assert '--help' in output
    if not prefix:
        for name in ('tokenizer', 'train', 'benchmark', 'experiment', 'data', 'cache', 'chat'):
            assert name in output
        assert 'sft' not in output and 'rl' not in output


@pytest.mark.parametrize('prefix,module,handler,workflow,target', LEAVES)
def test_dispatch_only_selected_handler(prefix, module, handler, workflow, target, monkeypatch):
    command = importlib.import_module(f'gpt_lab.cli.commands.{module}')
    run = Mock(return_value=0)
    monkeypatch.setattr(command, handler, run)
    assert main(prefix.split()) == 0
    run.assert_called_once()
    args = run.call_args.args[0]
    assert isinstance(args, argparse.Namespace)
    if prefix.startswith('train base'):
        assert args.model_init == prefix.split()[-1]


@pytest.mark.parametrize('prefix,module,handler,workflow,target', [row for row in LEAVES if row[3]])
def test_handlers_forward_to_packaged_workflow(prefix, module, handler, workflow, target, monkeypatch):
    execute = Mock()
    monkeypatch.setitem(sys.modules, f'gpt_lab.workflows.{workflow}', SimpleNamespace(**{target: execute}))
    args = build_parser().parse_args(prefix.split() + (['--mode', 'local'] if module == 'data' else []))
    command = importlib.import_module(f'gpt_lab.cli.commands.{module}')
    getattr(command, handler)(args)
    execute.assert_called_once_with(args)


def test_important_training_defaults():
    args = build_parser().parse_args(['train', 'base', 'auto'])
    expected = dict(model_init='auto', model_name='ic1', run_name=None, max_seq_len=2048,
                    random_seed=42, board='wandb', ds_name='climbmix-base', depth=12,
                    aspect_ratio=64, d_head=128, n_kv_heads=None, softcap=18.0,
                    attn_softcap=None, attn_impl='sdpa', device_batch_size=32,
                    vocab_size=-1, num_steps=-1, target_param_data_ratio=11.0,
                    n_acc_steps=-1, total_batch_size=-1, warmup_steps=40,
                    lr_embeddings=.3, lr_transformer=.02, lr_head=.008, lr_residuals=.5,
                    weight_decay=.28, warmdown_ratio=.65, final_lr_frac=.05,
                    eval_bpb_every=250, n_bpb_tokens=80 * 524288, eval_core_every=2000,
                    sample_every=0, save_every=-1, log_every=250)
    assert vars(args) | expected == vars(args)


def test_auto_parsing_and_deprecated_alias():
    args = build_parser().parse_args('train base auto --model-name test --depth 4 --d-kv-head 2 --train-tokenizer --vocab-size 512 --num-steps 10 --device cpu --board dummy --window-size 128'.split())
    assert (args.depth, args.n_kv_heads, args.vocab_size, args.num_steps) == (4, 2, 512, 10)
    assert args.train_tokenizer and args.window_size == '128'


@pytest.mark.parametrize('value,expected', [('120', 120), ('0', 0), ('-1', -1), ('-2', -2), ('latest', 'latest'), ('best', 'best')])
def test_resume_step_parsing(value, expected):
    args = build_parser().parse_args(['train', 'base', 'resume', '--checkpoint-step', value, '--checkpoint-dir', '/missing/run/base'])
    assert args.checkpoint_step == expected
    assert args.checkpoint_dir == '/missing/run/base'
    assert not hasattr(args, 'depth')


def test_other_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv('GPTLAB_CACHE_DIR', str(tmp_path / 'not-created'))
    parser = build_parser()
    tok = parser.parse_args('tokenizer train'.split())
    assert (tok.vocab_size, tok.max_chars, tok.chars_per_doc, tok.trainer, tok.num_proc) == (32000, -1, -1, 'tiktoken', -1)
    assert tok.corpus_path == './.data/corpus.txt' and tok.corpus_seed == 42
    bench = parser.parse_args('benchmark dataloaders'.split())
    assert (bench.mode, bench.loader, bench.batch_size, bench.seq_len, bench.num_batches, bench.buffer_size, bench.num_docs, bench.device, bench.bos_id) == ('synthetic', 'all', 8, 2048, 20, 1000, 20000, 'cpu', 1)
    exp = parser.parse_args('experiment tokenizer-scaling'.split())
    assert (exp.vocab_sizes, exp.baselines, exp.num_procs, exp.corpus_sizes_gb, exp.save_every, exp.board) == ('50000,70000,100000,200000', 'gpt2,cl100k_base,o200k_base', None, None, 10, 'dummy')
    assert exp.result_dir == str(tmp_path / 'not-created/tokenizers/scaling_tokenizer_results')
    data = parser.parse_args('data reshard'.split())
    assert (data.ds_name, data.mode, data.chars_per_shard, data.row_group_size, data.max_in_flight) == ('climbmix', 'bucket', 256000000, 1024, 8)
    assert not (tmp_path / 'not-created').exists()


def test_benchmark_aliases():
    parser = build_parser()
    old = parser.parse_args('benchmark dataloaders --batch_size 2 --seq_len 32 --num_batches 3 --buffer_size 16 --num_docs 8 --bos_id 7'.split())
    new = parser.parse_args('benchmark dataloaders --batch-size 2 --seq-len 32 --num-batches 3 --buffer-size 16 --num-docs 8 --bos-id 7'.split())
    assert old == new


@pytest.mark.parametrize('argv', ['', 'train', 'train base', 'train base invalid', 'train sft', 'train rl', 'benchmark unknown', 'chat', 'unknown', 'train base auto --wat', 'train base resume --checkpoint-step nope'])
def test_invalid_or_missing_subcommands(argv, capsys):
    with pytest.raises(SystemExit) as error:
        main(argv.split())
    assert error.value.code == 2
    assert 'error:' in capsys.readouterr().err


@pytest.mark.parametrize('module,prefix', WRAPPERS.items())
def test_wrappers_delegate(module, prefix, monkeypatch):
    wrapper = importlib.import_module(module)
    dispatch = Mock(return_value=7)
    monkeypatch.setattr(wrapper, 'cli_main', dispatch)
    assert wrapper.main(['--help']) == 7
    dispatch.assert_called_once_with(prefix.split() + ['--help'])


def test_imports_and_help_are_stdlib_only_and_do_not_touch_files(tmp_path):
    # -S removes site-packages: help must work even without Torch/Pydantic/UI.
    code = '''
import argparse, builtins, importlib, importlib.abc, os, pathlib, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','datasets','gradio','transformers','wandb','numpy','pydantic'} or fullname.startswith(('gpt_lab.utils', 'gpt_lab.workflows')):
            raise AssertionError('heavy import: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected filesystem or argument parsing operation')
original_parse = argparse.ArgumentParser.parse_args
argparse.ArgumentParser.parse_args = forbidden
sys.argv = ['unrelated-process', '--invalid-global-flag']
import gpt_lab.cli
for name in ['__main__', 'main', 'parser', 'options', 'commands.base', 'commands.training', 'commands.tokenizer', 'commands.benchmark', 'commands.experiment', 'commands.data', 'commands.cache', 'commands.chat']:
    importlib.import_module('gpt_lab.cli.' + name)
argparse.ArgumentParser.parse_args = original_parse
for name in ['mkdir', 'stat', 'lstat', 'resolve', 'open']:
    setattr(pathlib.Path, name, forbidden)
for prefix in ['', 'tokenizer', 'train base auto', 'train base resume', 'benchmark dataloaders', 'experiment tokenizer-scaling', 'data reshard', 'cache clear', 'chat console', 'chat app']:
    try:
        gpt_lab.cli.main(prefix.split() + ['--help'])
    except SystemExit as exc:
        assert exc.code == 0
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1', GPTLAB_CACHE_DIR=str(tmp_path / 'cache'))
    process = subprocess.run([sys.executable, '-S', '-c', textwrap.dedent(code)], env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert not (tmp_path / 'cache').exists()


def test_module_smoke_outside_repository(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-S', '-m', 'gpt_lab.cli', 'train', 'base', 'resume', '--help'], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--checkpoint-step' in result.stdout


def test_future_stage_registration_requires_no_root_change(monkeypatch):
    from gpt_lab.cli.commands import training
    def register(stages):
        stage = stages.add_parser('test-stage')
        stage.set_defaults(_handler='test:run')
    monkeypatch.setattr(training, 'TRAINING_STAGES', training.TRAINING_STAGES + (SimpleNamespace(register=register),))
    assert build_parser().parse_args('train test-stage'.split())._handler == 'test:run'
