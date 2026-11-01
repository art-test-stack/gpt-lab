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
    ('train', 'base', 'run', 'train_base', 'train_base'),
    ('train resume', 'base', 'run', 'train_base', 'train_base'),
    ('data reshard', 'data', 'run', 'reshard', 'run'),
    ('cache clear', 'cache', 'run', None, None),
    ('chat console', 'chat', 'run_console', 'chat', 'console'),
    ('chat app', 'chat', 'run_app', 'chat', 'app'),
]
WRAPPERS = {
    'scripts.train_tokenizer': 'tokenizer train',
    'scripts.train_base': 'train',
    'scripts.reshard_dataset': 'data reshard',
    'scripts.clear_gpt_lib_cache': 'cache clear',
    'scripts.chat_console': 'chat console',
    'scripts.chat_app': 'chat app',
}


@pytest.mark.parametrize('prefix', ['', 'tokenizer', 'train', 'data', 'cache', 'chat'] + [row[0] for row in LEAVES])
def test_help(prefix, capsys):
    with pytest.raises(SystemExit) as error:
        main(prefix.split() + ['--help'])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert 'usage: gpt-lab' in output
    assert '--help' in output
    if not prefix:
        for name in ('tokenizer', 'train', 'data', 'cache', 'chat'):
            assert name in output
        assert 'sft' not in output and 'rl' not in output
        assert '{tokenizer,train,data,cache,chat}' in output


@pytest.mark.parametrize('prefix,module,handler,workflow,target', LEAVES)
def test_dispatch_only_selected_handler(prefix, module, handler, workflow, target, monkeypatch):
    command = importlib.import_module(f'gpt_lab.cli.commands.{module}')
    run = Mock(return_value=0)
    monkeypatch.setattr(command, handler, run)
    assert main(prefix.split()) == 0
    run.assert_called_once()
    args = run.call_args.args[0]
    assert isinstance(args, argparse.Namespace)
    if prefix.startswith('train'):
        assert args.model_init == ('resume' if prefix.endswith('resume') else 'auto')


@pytest.mark.parametrize('prefix,module,handler,workflow,target', [row for row in LEAVES if row[3]])
def test_handlers_forward_to_packaged_workflow(prefix, module, handler, workflow, target, monkeypatch):
    execute = Mock()
    monkeypatch.setitem(sys.modules, f'gpt_lab.workflows.{workflow}', SimpleNamespace(**{target: execute}))
    args = build_parser().parse_args(prefix.split() + (['--mode', 'local'] if module == 'data' else []))
    command = importlib.import_module(f'gpt_lab.cli.commands.{module}')
    getattr(command, handler)(args)
    if module == 'base':
        execute.assert_called_once()
        keywords = execute.call_args.kwargs
        assert keywords['model_init'] == args.model_init
        assert keywords['model_name'] == args.model_name
        assert '_handler' not in keywords and 'optim_config_path' not in keywords
        if args.model_init == 'auto':
            assert keywords['optim_config'] is not None
    else:
        execute.assert_called_once_with(args)


def test_important_training_defaults():
    args = build_parser().parse_args(['train'])
    expected = dict(model_init='auto', model_name='ic1', run_name=None, max_seq_len=2048,
                    random_seed=42, board='wandb', ds_name='climbmix-base', depth=12,
                    aspect_ratio=64, d_head=128, n_kv_heads=None, softcap=18.0,
                    tie_embeddings="embs;value_embs;head",
                    attn_softcap=None, attn_impl='sdpa', device_batch_size=32,
                    vocab_size=-1, num_steps=-1, target_param_data_ratio=11.0,
                    n_acc_steps=-1, total_batch_size=-1, warmup_steps=40,
                    warmdown_ratio=.65, final_lr_frac=.05,
                    eval_bpb_every=250, n_bpb_tokens=80 * 524288, eval_core_every=2000,
                    sample_every=0, save_every=-1, log_every=250)
    assert vars(args) | expected == vars(args)


def test_auto_parsing_and_deprecated_alias():
    args = build_parser().parse_args('train --model-name test --depth 4 --d-kv-head 2 --train-tokenizer --vocab-size 512 --num-steps 10 --device cpu --board dummy --window-size 128'.split())
    assert (args.depth, args.n_kv_heads, args.vocab_size, args.num_steps) == (4, 2, 512, 10)
    assert args.train_tokenizer and args.window_size == '128'


def test_step_limit_flag():
    args = build_parser().parse_args(['train', '--stop-after-step', '5'])
    assert args.stop_after_step == 5
    assert not hasattr(args, 'stop_after_updates')


def test_step_limit_rejects_old_flag():
    with pytest.raises(SystemExit):
        build_parser().parse_args(['train', '--stop-after-updates', '5'])


@pytest.mark.parametrize('value,expected', [('120', 120), ('0', 0), ('-1', -1), ('-2', -2), ('latest', 'latest'), ('best', 'best')])
def test_resume_step_parsing(value, expected):
    args = build_parser().parse_args(['train', 'resume', '--checkpoint-step', value, '--checkpoint-dir', '/missing/run/base'])
    assert args.checkpoint_step == expected
    assert args.checkpoint_dir == '/missing/run/base'
    assert not hasattr(args, 'depth')
    assert not hasattr(args, 'max_seq_len')


@pytest.mark.parametrize('option', [
    '--depth 4', '--max-seq-len 128', '--d-head 64', '--n-kv-heads 2',
    '--vocab-size 4096', '--tokenizer-model gpt2', '--tie-embeddings embs,head',
])
def test_resume_rejects_architecture_options(option):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(('train resume ' + option).split())
    assert error.value.code == 2


def test_optional_auto_spelling_preserves_training_defaults():
    parser = build_parser()
    assert vars(parser.parse_args(['train'])) == vars(parser.parse_args(['train', 'auto']))


@pytest.mark.parametrize('option', [
    '--lr-embeddings', '--lr-transformer', '--lr-head', '--lr-residuals', '--weight-decay',
])
def test_optimizer_values_are_not_cli_overrides(option):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(['train', option, '0.1'])
    assert error.value.code == 2


def test_other_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv('GPTLAB_CACHE_DIR', str(tmp_path / 'not-created'))
    parser = build_parser()
    tok = parser.parse_args('tokenizer train'.split())
    assert (tok.vocab_size, tok.max_chars, tok.chars_per_doc, tok.trainer, tok.num_proc) == (32000, -1, -1, 'tiktoken', -1)
    assert tok.corpus_path == './.data/corpus.txt' and tok.corpus_seed == 42
    data = parser.parse_args('data reshard'.split())
    assert (data.ds_name, data.mode, data.chars_per_shard, data.row_group_size, data.max_in_flight) == ('climbmix', 'bucket', 256000000, 1024, 8)
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('argv', ['', 'train base', 'train base invalid', 'train sft', 'train rl', 'benchmark dataloaders', 'experiment tokenizer-scaling', 'chat', 'unknown', 'train --wat', 'train resume --checkpoint-step nope'])
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
        if fullname in {'gpt_lab.utils', 'gpt_lab.utils.paths'}:
            return None
        if fullname.split('.')[0] in {'torch','datasets','gradio','transformers','wandb','numpy','pydantic'} or fullname.startswith(('gpt_lab.utils', 'gpt_lab.workflows')):
            raise AssertionError('heavy import: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected filesystem or argument parsing operation')
original_parse = argparse.ArgumentParser.parse_args
argparse.ArgumentParser.parse_args = forbidden
sys.argv = ['unrelated-process', '--invalid-global-flag']
import gpt_lab.cli
for name in ['__main__', 'main', 'parser', 'options', 'commands.base', 'commands.training', 'commands.tokenizer', 'commands.data', 'commands.cache', 'commands.chat']:
    importlib.import_module('gpt_lab.cli.' + name)
argparse.ArgumentParser.parse_args = original_parse
for name in ['mkdir', 'stat', 'lstat', 'resolve', 'open']:
    setattr(pathlib.Path, name, forbidden)
for prefix in ['', 'tokenizer', 'train', 'train resume', 'data reshard', 'cache clear', 'chat console', 'chat app']:
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
    result = subprocess.run([sys.executable, '-S', '-m', 'gpt_lab.cli', 'train', 'resume', '--help'], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--checkpoint-step' in result.stdout
