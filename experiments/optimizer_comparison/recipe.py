"""Frozen selected production arguments; no Modal or network work on import."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTION = json.loads((Path(__file__).with_name("selected-runs.json")).read_text())
TRACKING = dict(WANDB_ENTITY="art-test-stack-ntnu", WANDB_PROJECT="trainbase_ic1",
                WANDB_BASE_URL="https://api.forge.coreweave.com",
                WANDB_APP_URL="https://forge.coreweave.com/wandb", WANDB_MODE="online")

# Freeze unspecified settings from the consolidated CLI, not historical diagnostics.
COMMON = dict(ds_name="climbmix-base", tokenizer_model="gpt2", random_seed=42,
              device="cuda", depth=20, aspect_ratio=64, d_head=128,
              act_func="relu_squared", attn_impl="fa3", max_seq_len=2048,
              tie_embeddings="embs;value_embs;head", window_pattern="SSSL", window_size="512",
              softcap=18.0, device_batch_size=32, total_batch_size=524288,
              n_acc_steps=1, num_steps=-1, target_flops=-1, target_time=-1,
              target_param_data_ratio=10, warmup_steps=40, warmdown_ratio=0.65,
              final_lr_frac=0.05, cuda_graph_mode="disabled", fp8_ffn_forward=True,
              eval_bpb_every=400, n_bpb_tokens=41943040, eval_core_every=400,
              n_core_tokens=500, sample_every=0, save_every=0, save_on_best=True,
              log_every=10, board="wandb")


def arguments(variant, run_name, root):
    from runs.modal_trainbase import cli_defaults
    return [*cli_defaults(COMMON), "--model-name", variant["model_name"],
            "--optim-config-path", variant["optimizer_config"], "--run-name", run_name,
            "--model-dir", str(Path(root)/"models"), "--board-dir", str(Path(root)/"wandb")]


def model_and_horizon(device="meta"):
    import torch
    from gpt_lab.model.auto import AutoGPTConfig
    from gpt_lab.utils.distributed import ComputeContext
    # GPT2's padded output vocabulary; gate verifies this against the cached tokenizer.
    auto = AutoGPTConfig(depth=20, aspect_ratio=64, d_head=128, max_seq_len=2048,
                         act_func="relu_squared", attn_impl="fa3", target_param_data_ratio=10,
                         total_batch_size=524288, device_batch_size=32, n_acc_steps=1,
                         comp_info=ComputeContext(torch.device("cpu")),
                         special_tokens={}, run_name="selected-plan")
    config = auto.build_transformer_config(20, 50304)
    from gpt_lab.model.checkpoint import build_meta_model
    count_config = config if torch.cuda.is_available() else config.model_copy(update={"attn_impl": "sdpa"})
    if device == "meta":
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            model = build_meta_model(count_config)
    else:
        model = build_meta_model(count_config)
    total = sum(p.numel() for p in model.parameters())
    scaling = model.n_scaling_params()
    target = int(auto.compute_target_tokens(scaling))
    from torch._subclasses.fake_tensor import FakeTensorMode
    reference_config = auto.build_transformer_config(auto.reference_depth(), 50304).model_copy(update={"attn_impl": "sdpa"})
    with FakeTensorMode():
        reference = build_meta_model(reference_config)
    base_training = auto.build_training_config(model, reference)
    steps = base_training["n_steps"]
    report = dict(parameter_count=total, scaling_parameters=scaling, base_training=base_training,
                  scaling_definition="decoder blocks plus output-table elements, counting a tied output table once; excludes input/value embeddings and residual scalars",
                  target_tokens=target, rounding="floor(target_tokens / 524288)",
                  horizon=steps, rounded_tokens=steps*524288,
                  target_data_to_parameter_ratio=10,
                  d_model=config.d_model, d_ffn=config.d_ffn, vocabulary=50304,
                  window_size=config.window_size)
    if device != "meta":
        if not torch.cuda.is_available():
            raise RuntimeError("Selected model execution requires CUDA/FA3; CPU SDPA is used for parameter counting only")
        model = model.to_empty(device=device)
        model.precompute_pos_enc()
    return model, report


def storage_summary(model):
    import torch
    embeddings = {id(model.embeds.weight), *(id(v.weight) for v in model.value_embeds.values())}
    counts = {}
    for parameter in model.parameters():
        expected = torch.bfloat16 if id(parameter) in embeddings and parameter is not model.lm_head.weight else torch.float32
        assert parameter.dtype == expected, "Parameter storage differs from the existing tying-aware policy"
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0)+parameter.numel()
    return counts


def selected_run(settings, run_name, world_size):
    variant = next((v for v in SELECTION["variants"] if v["model_name"] == settings["model_name"]), None)
    if variant is None:
        return None
    for key, value in COMMON.items():
        if settings[key] != value:
            raise ValueError(f"Selected recipe mismatch for {key}: expected {value!r}, got {settings[key]!r}")
    if world_size != 8 or settings["stop_after_step"] is not None:
        raise ValueError("Selected runs require eight GPUs and no execution cap")
    if settings["optim_config_path"] != variant["optimizer_config"]:
        raise ValueError("Selected model name and optimizer profile differ")
    if not run_name.startswith(variant["model_name"] + "-"):
        raise ValueError("Selected run names must start with the model name followed by a shared comparison suffix")
    suffix = run_name[len(variant["model_name"]):]
    index = SELECTION["variants"].index(variant)
    previous = SELECTION["variants"][max(0, index - 1)]["model_name"]
    return dict(reference_run=SELECTION["variants"][0]["model_name"] + suffix,
                predecessor_run=previous + suffix)
