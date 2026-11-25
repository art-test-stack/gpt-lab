from pathlib import Path
from types import SimpleNamespace

from gpt_lab.utils.distributed import ComputeContext
import pytest
import torch

from gpt_lab.optim.factory import OptimizerFactory
from gpt_lab.optim.kernels.muon import muon_step_fused
from gpt_lab.utils.tensor_io import load_tensor_state, save_tensor_state
from torch.utils._python_dispatch import TorchDispatchMode
from gpt_lab.utils.schemas import TrainerConfig



@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float32])
def test_muon_restores_parameter_dtype_after_orthogonalization(compute_dtype):
    generator = torch.Generator().manual_seed(123)
    params = torch.randn(2, 3, 4, generator=generator)
    grads = torch.randn(params.shape, generator=generator)
    momentum = torch.zeros_like(params)
    variance = torch.zeros(2, 3, 1)
    initial_params = params.clone()
    matmul_dtypes, scaling_dtypes = [], []

    class ObserveDtypes(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.bmm.default:
                matmul_dtypes.append((args[0].dtype, args[1].dtype))
            if (func == torch.ops.aten.mul.Tensor
                    and isinstance(args[1], torch.Tensor)
                    and args[0].shape == params.shape and args[1].shape == variance.shape):
                scaling_dtypes.append((args[0].dtype, args[1].dtype))
            return func(*args, **(kwargs or {}))

    with ObserveDtypes():
        muon_step_fused.__wrapped__(
            grads, params, momentum, variance,
            *(torch.tensor(value) for value in (0.95, 0.02, 0.28, 0.9)),
            5, -1, compute_dtype,
        )

    assert matmul_dtypes == [(compute_dtype, compute_dtype)] * 15
    assert scaling_dtypes == [(torch.float32, torch.float32)]
    for tensor in (params, grads, momentum, variance):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    assert not torch.equal(params, initial_params)


@pytest.mark.parametrize("opt_name", ["muon", "normuon"])
def test_hyperparameter_schedules_preserve_group_specific_values(opt_name):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = OptimizerFactory(
        [
            dict(
                params=[param],
                opt=opt_name,
                lr=0.2,
                momentum=0.95,
                beta=0.9,
                ns_steps=5,
                weight_decay=0.28,
            )
        ],
        comp_info=ComputeContext(device=torch.device("cpu")),
    )

    optimizer.update_hyperparams(lrm=0.5, muon_momentum=0.8, weight_decay=0.25)

    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(0.1)
    assert group["momentum"] == pytest.approx(0.76)
    assert group["weight_decay"] == pytest.approx(0.07)


@pytest.mark.parametrize("opt_name", ["muon", "normuon"])
def test_legacy_checkpoint_gets_schedule_baselines(opt_name):
    param = torch.nn.Parameter(torch.zeros(1))
    config = dict(
        params=[param],
        opt=opt_name,
        lr=0.2,
        momentum=0.95,
        beta=0.9,
        ns_steps=5,
        weight_decay=0.28,
    )
    optimizer = OptimizerFactory([config], comp_info=ComputeContext(device=torch.device("cpu")))
    state = optimizer.state_dict()
    for key in ("initial_lr", "initial_momentum", "initial_weight_decay"):
        state["param_groups"][0].pop(key)

    optimizer.load_state_dict(state)
    optimizer.update_hyperparams(lrm=0.5, muon_momentum=0.8, weight_decay=0.25)

    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.07)


@pytest.mark.parametrize("opt_name", ["muon", "normuon"])
@pytest.mark.parametrize("splits", [(8, 8, 8), (8, 2, 2), (2,) * 12, (2,) * 6])
def test_split_qkv_matches_independent_matrices_after_resume(tmp_path, opt_name, splits):
    generator = torch.Generator().manual_seed(35)
    packed = [torch.nn.Parameter(torch.randn(sum(splits), 8, generator=generator)) for _ in range(2)]
    parts = [[torch.nn.Parameter(block.clone()) for block in p.detach().split(splits)] for p in packed]
    settings = dict(opt=opt_name, lr=0.02, momentum=0.95, beta=0.9, ns_steps=5, weight_decay=0.1)
    context = ComputeContext(device=torch.device("cpu"))
    group = dict(params=packed, qkv_split_sizes=splits, **settings)
    optimizer = OptimizerFactory([group], comp_info=context)
    reference = OptimizerFactory([dict(params=list(blocks), **settings) for blocks in zip(*parts)], comp_info=context)

    with torch.compiler.set_stance("force_eager"):
        for step in range(3):
            for p, matrices in zip(packed, parts):
                p.grad = torch.randn(p.shape, generator=generator)
                for part, grad in zip(matrices, p.grad.split(splits)):
                    part.grad = grad.clone()
            optimizer.step()
            reference.step()
            for p, matrices in zip(packed, parts):
                torch.testing.assert_close(p, torch.cat(matrices))

            state = optimizer.state[packed[0]]
            expected_momentum = torch.cat([
                reference.state[part]["momentum_buffer"] for part in parts[0]
            ], dim=1)
            torch.testing.assert_close(state["momentum_buffer"], expected_momentum)
            offset, block = 0, 0
            for key, variance in state.items():
                if key == "momentum_buffer":
                    continue
                assert key == f"second_momentum_buffer_{offset}"
                for actual in variance.unbind(1):
                    expected = reference.state[parts[0][block]]["second_momentum_buffer"]
                    torch.testing.assert_close(actual, expected)
                    offset += splits[block]
                    block += 1
            assert block == len(splits)

            if step == 0:
                path = tmp_path / "optimizer.safetensors"
                save_tensor_state(optimizer.state_dict(), path)
                saved = load_tensor_state(path)
                optimizer = OptimizerFactory([group], comp_info=context)
                optimizer.load_state_dict(saved)
                for other_layout in (None, (sum(splits) // 2,) * 2):
                    other_group = dict(params=packed, **settings)
                    if other_layout is not None:
                        other_group["qkv_split_sizes"] = other_layout
                    incompatible = OptimizerFactory([other_group], comp_info=context)
                    with pytest.raises(ValueError, match="different Muon QKV split layout"):
                        incompatible.load_state_dict(saved)


@pytest.mark.parametrize("arm", ["joint-real", "joint-rope", "joint-shuffled"])
def test_matching_qk_heads_share_one_polar_map_after_resume(tmp_path, monkeypatch, arm):
    from gpt_lab.optim.kernels.muon import _real_polar_express, paired_polar_express

    experiment = Path(__file__).resolve().parents[1] / "experiments" / "20261118-rope-paired-muon"
    monkeypatch.syspath_prepend(str(experiment))
    from optimizer import configure_experiment_muon, pairing_order

    generator = torch.Generator().manual_seed(35)
    context = ComputeContext(device=torch.device("cpu"), compute_dtype=torch.float32)
    lr, momentum, beta, decay = 0.02, 0.9, 0.8, 0.1
    n_heads, d_head, d_model = 2, 4, 8

    def configured_optimizer(params, pairing, n_kv_heads=n_heads):
        blocks = [
            SimpleNamespace(attention=SimpleNamespace(
                n_heads=n_heads, n_kv_heads=n_kv_heads, d_head=d_head,
                qkv_proj=SimpleNamespace(weight=p),
            ))
            for p in params
        ]
        model = SimpleNamespace(config=SimpleNamespace(positional_encoding="rope"), blocks=blocks)
        group = dict(params=params, opt="muon", lr=lr, momentum=momentum, beta=beta, ns_steps=5,
                     weight_decay=decay, qkv_split_sizes=(d_head,) * (n_heads + 2 * n_kv_heads))
        optimizer = OptimizerFactory([group], comp_info=context)
        assert configure_experiment_muon(model, optimizer, pairing=pairing) == {"pairing": pairing}
        return optimizer

    weights = torch.randn(2, 3 * n_heads, d_head, d_model, generator=generator)
    params = [torch.nn.Parameter(w.flatten(0, 1).clone()) for w in weights]
    first_moment = torch.zeros_like(weights)
    variance = torch.zeros_like(weights[:, :, :1, :])
    optimizer = configured_optimizer(params, arm)
    polar = _real_polar_express if arm == "joint-real" else paired_polar_express
    rows = torch.tensor(pairing_order(d_head, "shuffled")) if arm == "joint-shuffled" else torch.arange(d_head)

    with torch.compiler.set_stance("force_eager"):
        for step in range(2):
            grads = torch.randn(weights.shape, generator=generator)
            for p, grad in zip(params, grads, strict=True):
                p.grad = grad.flatten(0, 1)
            optimizer.step()

            first_moment.lerp_(grads, 1 - momentum)
            proposals = grads.lerp(first_moment, momentum)
            q, k, v = proposals.split(n_heads, dim=1)
            matched = torch.cat((q[..., rows, :], k[..., rows, :]), dim=-1)
            q, k = polar(matched, 5, context.compute_dtype)[..., rows.argsort(), :].chunk(2, dim=-1)
            raw = torch.cat((q, k, _real_polar_express(v, 5, context.compute_dtype)), dim=1)
            variance.lerp_(raw.square().mean(dim=-2, keepdim=True), 1 - beta)
            update = raw * variance.clamp_min(1e-10).rsqrt()
            update *= raw.norm(dim=(-2, -1), keepdim=True) / update.norm(dim=(-2, -1), keepdim=True)
            weights -= lr * (update + decay * weights * (update * weights >= 0))
            torch.testing.assert_close(torch.stack(params).reshape_as(weights), weights)

            if step == 0:
                path = tmp_path / "joint-qk.safetensors"
                save_tensor_state(optimizer.state_dict(), path)
                saved = load_tensor_state(path)
                optimizer = configured_optimizer(params, arm)
                optimizer.load_state_dict(saved)
                other_arm = "joint-real" if arm == "joint-rope" else "joint-rope"
                with pytest.raises(ValueError, match="different Muon Q/K pairing"):
                    configured_optimizer(params, other_arm).load_state_dict(saved)

    grouped_weight = torch.nn.Parameter(torch.zeros(4 * d_head, d_model))
    with pytest.raises(ValueError, match="equal query and key head counts"):
        configured_optimizer([grouped_weight], arm, n_kv_heads=1)


@pytest.mark.parametrize("opt_name", ["muon", "normuon"])
def test_joint_update_and_checkpoint_keep_legacy_layout(tmp_path, opt_name):
    from gpt_lab.optim.kernels.normuon import normuon_step_fused

    generator = torch.Generator().manual_seed(50)
    params = [torch.nn.Parameter(torch.randn(12, 4, generator=generator)) for _ in range(2)]
    weights = torch.stack(params).detach()
    momentum = torch.zeros_like(weights)
    variance = torch.zeros(2, 12, 1)
    settings = dict(opt=opt_name, lr=0.02, momentum=0.95, beta=0.9, ns_steps=5, weight_decay=0.1)
    context = ComputeContext(device=torch.device("cpu"))
    optimizer = OptimizerFactory([dict(params=params, **settings)], comp_info=context)
    kernel = muon_step_fused if opt_name == "muon" else normuon_step_fused
    lr = settings["lr"] * (3 ** 0.5 if opt_name == "muon" else 1)

    with torch.compiler.set_stance("force_eager"):
        for step in range(3):
            for p in params:
                p.grad = torch.randn(p.shape, generator=generator)
            kernel.__wrapped__(
                torch.stack([p.grad for p in params]), weights, momentum, variance,
                *(torch.tensor(value) for value in (0.95, lr, 0.1, 0.9)), 5, -1, torch.float32,
            )
            optimizer.step()
            torch.testing.assert_close(torch.stack(params), weights)
            state = optimizer.state[params[0]]
            assert set(state) == {"momentum_buffer", "second_momentum_buffer"}
            torch.testing.assert_close(state["momentum_buffer"], momentum)
            torch.testing.assert_close(state["second_momentum_buffer"], variance)
            assert "qkv_split_sizes" not in optimizer.param_groups[0]
            if step == 0:
                path = tmp_path / "joint.safetensors"
                save_tensor_state(optimizer.state_dict(), path)
                optimizer = OptimizerFactory([dict(params=params, **settings)], comp_info=context)
                optimizer.load_state_dict(load_tensor_state(path))


@pytest.mark.parametrize("opt_name", ["muon", "normuon"])
@pytest.mark.parametrize("splits,muon_calls", [
    (None, ["compiled"]),
    ((8, 2, 2), ["eager", "eager"]),
    ((2,) * 6, ["compiled"]),
])
def test_qkv_batches_share_storage_and_limit_cpu_fallback(monkeypatch, opt_name, splits, muon_calls):
    params = [torch.nn.Parameter(torch.randn(12, 8)) for _ in range(2)]
    group = dict(params=params, opt=opt_name, lr=0.02, momentum=0.95, beta=0.9, ns_steps=5, weight_decay=0.1)
    if splits is not None:
        group["qkv_split_sizes"] = splits
    optimizer = OptimizerFactory([group], comp_info=ComputeContext(device=torch.device("cpu")))
    strategy = optimizer.backend.strategies[opt_name]
    kernel = strategy.step_kernel.__wrapped__
    calls, storage = [], []

    def observe(kind, *args, **kwargs):
        calls.append(kind)
        storage.append(tuple(t.untyped_storage().data_ptr() for t in args[:3]))
        kernel(*args, **kwargs)

    def eager(*args, **kwargs):
        observe("eager", *args, **kwargs)

    def compiled(*args, **kwargs):
        observe("compiled", *args, **kwargs)

    compiled.__wrapped__ = eager
    monkeypatch.setattr(type(strategy), "step_kernel", staticmethod(compiled))
    for p in params:
        p.grad = torch.randn_like(p)
    optimizer.step()

    assert calls == (muon_calls if opt_name == "muon" else ["compiled"] * len(muon_calls))
    assert all(pointers == storage[0] for pointers in storage)
    assert storage[0][2] == optimizer.state[params[0]]["momentum_buffer"].untyped_storage().data_ptr()


def _reference_normuon_update(grad, momentum, second_moment, beta, beta2, steps):
    # Unbatched official reference; MIT (c) 2025 zichongli5, LICENSES/NorMuon-MIT.txt.
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta)
    x = update.bfloat16()
    if grad.size(0) > grad.size(1):
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * a @ a
        x = 3.4445 * x + b @ x
    if grad.size(0) > grad.size(1):
        x = x.T
    update = x.to(grad.dtype)
    norm = update.norm()
    second_moment.lerp_(update.square().mean(dim=-1, keepdim=True), 1 - beta2)
    update.mul_(1 / (second_moment.sqrt() + 1e-10))
    update.mul_(norm / (update.norm() + 1e-10))
    return update * max(1, grad.size(0) / grad.size(1)) ** 0.5


@pytest.mark.parametrize("shape", [(4, 2), (2, 4)])
def test_normuon_matches_reference_updates_and_checkpoint(shape):
    parameters = [torch.nn.Parameter(torch.linspace(-0.3, 0.7, 8).reshape(shape) + i)
                  for i in range(2)]
    expected = [p.detach().clone() for p in parameters]
    moments = [torch.zeros_like(p) for p in parameters]
    variances = [torch.zeros(shape[0], 1) for _ in parameters]
    config = dict(opt="normuon", lr=0.02, momentum=0.95, beta=0.95,
                  ns_steps=5, weight_decay=0.1)
    context = ComputeContext(device=torch.device("cpu"))
    optimizer = OptimizerFactory([dict(params=parameters, **config)], comp_info=context)
    with torch.compiler.set_stance("force_eager"):
        for step in range(3):
            for i, parameter in enumerate(parameters):
                gradient = torch.sin(torch.arange(8).reshape(shape) + i + step)
                parameter.grad = gradient
                update = _reference_normuon_update(gradient, moments[i], variances[i], 0.95, 0.95, 5)
                expected[i].mul_(1 - config["lr"] * config["weight_decay"])
                expected[i].add_(update, alpha=-config["lr"])
            optimizer.step()
            for parameter, reference in zip(parameters, expected):
                torch.testing.assert_close(parameter, reference)
            state = optimizer.state[parameters[0]]
            torch.testing.assert_close(state["momentum_buffer"], torch.stack(moments))
            torch.testing.assert_close(state["second_momentum_buffer"], torch.stack(variances))
            if step == 0:
                resumed = OptimizerFactory([dict(params=parameters, **config)], comp_info=context)
                resumed.load_state_dict(optimizer.state_dict())
                optimizer = resumed


@pytest.mark.parametrize("key,value", [("beta", 1.0), ("momentum", -0.1), ("lr", float("nan")),
                                       ("ns_steps", 0), ("ns_steps", 1.5)])
def test_normuon_rejects_invalid_hyperparameters(key, value):
    group = dict(params=[torch.nn.Parameter(torch.zeros(2, 3))], opt="normuon",
                 lr=0.02, momentum=0.95, beta=0.95, ns_steps=5, weight_decay=0.1)
    group[key] = value
    with pytest.raises(ValueError, match="NorMuon"):
        OptimizerFactory([group], comp_info=ComputeContext(device=torch.device("cpu")))
