import torch
from gpt_lab.utils.distributed import ComputeContext
import math

import pytest

from gpt_lab.model.checkpoint import build_meta_model
from gpt_lab.utils.schemas import OptimizerConfig, TrainerConfig, TransformerConfig


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
