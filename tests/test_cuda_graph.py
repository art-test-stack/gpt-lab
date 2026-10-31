import copy
import importlib.util
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from gpt_lab.train.cuda_graph import CapturedForwardBackward
from gpt_lab.train.trainer import Trainer
from gpt_lab.utils.distributed import ComputeContext
from gpt_lab.utils.schemas import TrainerConfig


spec = importlib.util.spec_from_file_location(
    "cuda_graph_equivalence", Path(__file__).parents[1] / "scripts/benchmark/cuda_graph_equivalence.py"
)
equivalence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(equivalence)


def test_nested_equivalence_reports_tensor_and_counter_mismatches():
    errors, maxima = [], {}
    equivalence.compare_values(
        {"momentum": torch.tensor([1., 2.]), "step": 2},
        {"momentum": torch.tensor([1., 3.]), "step": 3},
        "optimizer", errors, maxima, rtol=1e-5, atol=1e-6,
    )
    assert len(errors) == 2
    assert maxima["optimizer"] == 1
    with pytest.raises(RuntimeError, match="Real CUDA is required"):
        equivalence.run_check("configs/normuon.yaml", ComputeContext(device=torch.device("cpu")), ".")


def test_single_microstep_cli_also_checks_accumulation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(equivalence.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(equivalence, "init_dist_groups", lambda *args, **kwargs:
                        SimpleNamespace(rank=0, world_size=8))
    monkeypatch.setattr(equivalence, "cleanup_dist_groups", lambda: None)
    def check(profile, context, output, *, n_acc_steps, attn_impl):
        calls.append(n_acc_steps)
        return dict(passed=True, rank=0, world_size=8, n_acc_steps=n_acc_steps)
    monkeypatch.setattr(equivalence, "run_check", check)
    monkeypatch.setattr(sys, "argv", ["check", "--optim-config-path", "configs/normuon.yaml",
                                    "--output-dir", str(tmp_path), "--n-acc-steps", "1"])
    equivalence.main()
    assert calls == [1, 2]
    result = json.loads((tmp_path / "rank_0.json").read_text())
    assert [check["n_acc_steps"] for check in result["accumulation_checks"]] == [1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA capture")
@pytest.mark.parametrize("n_acc_steps", [1, 2])
def test_production_normuon_compiled_capture_equivalence(tmp_path, n_acc_steps):
    from gpt_lab.utils.distributed import get_comp_info
    torch.manual_seed(17)
    result = equivalence.run_check("configs/normuon.yaml", get_comp_info("cuda"), tmp_path,
                                   n_acc_steps=n_acc_steps, updates=6, attn_impl="fused")
    assert result["passed"]
    assert {"loss", "gradients", "parameters", "optimizer"} <= result["max_absolute_errors"].keys()


def test_capture_is_opt_in_and_requires_cuda():
    config = TrainerConfig(comp_info=ComputeContext(device=torch.device("cpu")))
    assert config.cuda_graph_mode == "disabled"
    config = config.model_copy(update={"cuda_graph_mode": "forward_backward"})
    with pytest.raises(ValueError, match="requires CUDA"):
        Trainer(torch.nn.Linear(2, 2), None, None, None, config=config)
    with pytest.raises(ValueError, match="input/label tensors"):
        CapturedForwardBackward(torch.nn.Linear(2, 2), None, (torch.ones(2), torch.ones(2)), 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA capture")
@pytest.mark.parametrize("n_acc_steps", [1, 3])
def test_compiled_capture_matches_multiple_updates_and_resume(n_acc_steps):
    torch.manual_seed(17)
    device = torch.device("cuda", torch.cuda.current_device())
    eager = torch.nn.Linear(4, 3).to(device)
    replayed = copy.deepcopy(eager)
    eager_opt = torch.optim.AdamW(eager.parameters(), lr=0.01)
    replay_opt = torch.optim.AdamW(replayed.parameters(), lr=0.01)

    def loss_fn(model):
        return torch.compile(
            lambda batch: (model(batch[0]) - batch[1]).square().mean(),
            dynamic=False, options={"triton.cudagraphs": False},
        )

    compiled_loss = loss_fn(replayed)
    reference_loss = loss_fn(eager)
    batch = (torch.randn(2, 4, device=device), torch.randn(2, 3, device=device))
    before = [p.detach().clone() for p in replayed.parameters()]
    rng_before = torch.cuda.get_rng_state(device)
    graph = CapturedForwardBackward(replayed, compiled_loss, batch, n_acc_steps)
    assert torch.equal(torch.cuda.get_rng_state(device), rng_before)
    for p, value in zip(replayed.parameters(), before):
        torch.testing.assert_close(p, value)
        torch.testing.assert_close(p.grad, torch.zeros_like(p))
    grad_addresses = [p.grad.data_ptr() for p in replayed.parameters()]

    for step in range(5):
        eager_opt.zero_grad(set_to_none=False)
        replay_opt.zero_grad(set_to_none=False)
        for _ in range(n_acc_steps):
            # Reused loader storage must be copied before the next replay.
            batch[0].normal_()
            batch[1].normal_()
            expected = reference_loss(batch) / n_acc_steps
            expected.backward()
            actual = graph.replay(batch)
            torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=1e-6)
        for a, b in zip(eager.parameters(), replayed.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)
        for opt in (eager_opt, replay_opt):
            opt.param_groups[0]["lr"] = 0.01 / (step + 1)
            opt.step()
        for a, b in zip(eager.parameters(), replayed.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            for key in eager_opt.state[a]:
                torch.testing.assert_close(eager_opt.state[a][key], replay_opt.state[b][key], rtol=1e-5, atol=1e-6)
        assert [p.grad.data_ptr() for p in replayed.parameters()] == grad_addresses
        if step == 2:
            # Evaluation is uncaptured; resume reconstructs a graph over restored storage.
            replayed.eval()
            with torch.no_grad():
                torch.testing.assert_close(replayed(batch[0]), eager(batch[0]))
            resumed = copy.deepcopy(replayed)
            resumed.train()
            replay_opt_state = copy.deepcopy(replay_opt.state_dict())
            replayed = resumed
            replay_opt = torch.optim.AdamW(replayed.parameters(), lr=0.01)
            replay_opt.load_state_dict(replay_opt_state)
            replay_opt.zero_grad(set_to_none=True)
            compiled_loss = loss_fn(replayed)
            graph = CapturedForwardBackward(replayed, compiled_loss, batch, n_acc_steps)
            grad_addresses = [p.grad.data_ptr() for p in replayed.parameters()]
    with pytest.raises(ValueError, match="shape, dtype, and device"):
        graph.replay((batch[0][:1], batch[1][:1]))
