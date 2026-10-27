import torch
from gpt_lab.utils.distributed import ComputeContext
import math

import pytest
import json
from pathlib import Path
from types import SimpleNamespace

from gpt_lab.model.checkpoint import build_meta_model
from gpt_lab.utils.schemas import OptimizerConfig, TrainerConfig, TransformerConfig


def test_pr35_horizon_changes_only_schedules_and_cap_does_not_change_them():
    from gpt_lab.cli.parser import build_parser
    from gpt_lab.model.auto import AutoGPTConfig
    from scripts.adapters.pr35_comparison import verify_comparison

    reference_path = Path(__file__).resolve().parents[1] / 'configs/pr35-comparison.json'
    reference = json.loads(reference_path.read_text())

    class FourRanks(ComputeContext):
        @property
        def world_size(self):
            return 4

    context = FourRanks(device=torch.device('cpu'), compute_dtype=torch.bfloat16)
    resolved = []
    groups = []
    for horizon in (945, 2317):
        args = build_parser().parse_args(['train', '--depth', '12', '--device-batch-size', '16',
            '--target-param-data-ratio', '8', '--attn-impl', 'fused', '--num-steps', str(horizon),
            '--stop-after-updates', '945'])
        auto = AutoGPTConfig(depth=args.depth, d_head=args.d_head, aspect_ratio=args.aspect_ratio,
            n_steps=args.num_steps, device_batch_size=args.device_batch_size,
            target_param_data_ratio=args.target_param_data_ratio, attn_impl=args.attn_impl, comp_info=context)
        model = auto.build_meta_model_from_depth(args.depth, reference['tokenizer']['vocab_size'])
        assert model.config.model_dump() == reference['model']
        assert model.embeds.weight.shape == (50304, 768)
        training = auto.build_training_config(model, model)
        assert training['total_batch_size'] == 524288 and training['n_acc_steps'] == 4
        resolved.append({k: v for k, v in training.items() if k not in ('n_steps', 'n_total_tokens')})
        config = TrainerConfig(comp_info=context, **training, stop_after_updates=args.stop_after_updates,
            lr_warmup_steps=300, lr_warmdown_ratio=0.65, final_lr_ratio=0.05,
            optim_config=OptimizerConfig.from_file())
        optimizer = model.build_optimizer(config)
        groups.append([{k: v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups])
        if horizon == 2317:
            report = verify_comparison(SimpleNamespace(model=model, config=config, optimizer=optimizer,
                state=SimpleNamespace(completed_updates=0),
                tokenizer=SimpleNamespace(config=SimpleNamespace(**reference['tokenizer']))), reference_path)
            assert report['execution_cap'] == 945
            uncapped = config.model_copy(update={'stop_after_updates': None})
            for step in reference['shared_update_indices']:
                expected_lr = (step + 1) / 300 if step < 300 else (1.0 if step <= 811 else 0.05 + 0.95 * (2317 - step) / 1506)
                assert config.lr_multiplier_schedule(step) == pytest.approx(expected_lr)
                assert config.muon_momentum_schedule(step) == pytest.approx(expected_lr)
                assert config.weight_decay_schedule(step) == pytest.approx(0.5 * (1 + math.cos(math.pi * step / 2317)))
                for name in ('lr_multiplier_schedule', 'muon_momentum_schedule', 'weight_decay_schedule'):
                    assert getattr(config, name)(step) == getattr(uncapped, name)(step)
    assert resolved[0] == resolved[1]
    assert groups[0] == groups[1]


@pytest.mark.parametrize('custom', [False, True])
def test_optimizer_profile_is_scaled_once_without_mutation(tmp_path, custom):
    path = None
    if custom:
        path = tmp_path / 'optimizer.yaml'
        path.write_text('head:\n  lr: 0.2\ntransformer:\n  weight_decay: 0.05\n')
    profile = OptimizerConfig.from_file(path)
    snapshot = profile.model_dump()
    config = TrainerConfig(
        optim_config=profile, comp_info=ComputeContext(device=torch.device("cpu")),
        batch_lr_scale=4.0, weight_decay_scale=3.0, n_steps=8,
    )
    model = build_meta_model(TransformerConfig(
        vocab_size=300, max_context=4, d_model=16, d_ffn=32,
        n_heads=2, n_kv_heads=2, n_layers=2, d_head=8, window_pattern='L',
    )).to_empty(device='cpu')
    optimizer = model.build_optimizer(config)
    bases = profile.group_configs()
    for group in optimizer.param_groups:
        name = 'transformer' if group['name'].startswith('block.') else group['name']
        width = math.sqrt(768 / 16) if name in ('embeddings', 'head', 'value_embeds') else 1.0
        assert group['lr'] == pytest.approx(bases[name]['lr'] * 4 * width)
        decay = 3.0 if name == 'transformer' else 1.0
        assert group['weight_decay'] == pytest.approx(bases[name]['weight_decay'] * decay)
    assert profile.model_dump() == snapshot
    assert config.weight_decay_schedule(0) == 1.0
    assert config.weight_decay_schedule(8) == 0.0
    optimizer.update_hyperparams(lrm=0.5, weight_decay=config.weight_decay_schedule(0))
    for group in optimizer.param_groups:
        assert group['lr'] == pytest.approx(group['initial_lr'] * 0.5)
        assert group['weight_decay'] == group['initial_weight_decay']
    restored = TrainerConfig.model_validate_json(config.model_dump_json())
    assert restored.optim_config.model_dump() == snapshot


@pytest.mark.parametrize('contents', [
    'batch_size: 64',
    'transformer:\n  n_steps: 8',
    'head:\n  lr: -0.1',
    'head:\n  lr: bad',
    'head:\n  weight_decay: .nan',
    'head:\n  opt: unknown',
])
def test_optimizer_file_rejects_conflicting_or_invalid_parameters(tmp_path, contents):
    path = tmp_path / 'invalid.yaml'
    path.write_text(contents)
    with pytest.raises(ValueError):
        OptimizerConfig.from_file(path)


def test_legacy_trainer_keeps_saved_decay_schedule():
    config = TrainerConfig(weight_decay_scale=3.0, n_steps=8)
    assert config.optim_config is None
    assert config.weight_decay_schedule(0) == 3.0
    # Old pickled TrainerConfig instances do not contain newly added fields.
    config.__dict__.pop('optim_config')
    assert config.weight_decay_schedule(0) == 3.0
