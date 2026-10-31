import copy

import pytest
import torch

from gpt_lab.train.cuda_graph import CapturedForwardBackward
from gpt_lab.train.trainer import Trainer
from gpt_lab.utils.distributed import ComputeContext
from gpt_lab.utils.schemas import TrainerConfig


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
            expected = (eager(batch[0]) - batch[1]).square().mean() / n_acc_steps
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
