import torch
import torch.distributed as dist
from torch import Tensor

from gpt_lib.optim.adamw import adamw_step_fused
from gpt_lib.optim.muon import muon_step_fused


class _BaseFactory(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._init_scalars()

    def _init_scalars(self):
        self._adamw_step_t = torch.tensor(0.0, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, device="cpu")
        self._muon_beta_t = torch.tensor(0.0, device="cpu")

    def _step_adamw_local(self, group):
        for p in group['params']:
            if p.grad is None:
                continue

            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            state['step'] += 1

            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            adamw_step_fused(
                p, p.grad,
                state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t,
                self._adamw_beta1_t, self._adamw_beta2_t,
                self._adamw_eps_t, self._adamw_wd_t
            )

    def _step_muon_local(self, group):
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            return

        p0 = params[0]
        state = self.state[p0]

        shape = p0.shape
        device = p0.device
        dtype = p0.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, device=device, dtype=dtype)

        if "second_momentum_buffer" not in state:
            state["second_momentum_buffer"] = torch.zeros(len(params), shape[-2], 1, device=device, dtype=dtype)

        grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group['momentum'])
        self._muon_lr_t.fill_(group['lr'])
        self._muon_wd_t.fill_(group['weight_decay'])
        self._muon_beta_t.fill_(group['beta'])

        muon_step_fused(
            grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta_t,
            group['ns_steps'],
            -1
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

# =========================
# BACKENDS
# =========================

class _LocalBackend:
    def __init__(self, opt):
        self.opt = opt

    def step(self, *args, **kwargs):
        for group in self.opt.param_groups:
            if group['type'] == 'adamw':
                self.opt._step_adamw_local(group)
            else:
                self.opt._step_muon_local(group)

class _DistributedBackend:
    def __init__(self, opt):
        self.opt = opt

    def step(self, *args, **kwargs):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        for group in self.opt.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # fallback to local compute after sync
        for group in self.opt.param_groups:
            if group['type'] == 'adamw':
                self.opt._step_adamw_local(group)
            else:
                self.opt._step_muon_local(group)

# =========================
# UNIFIED OPTIMIZER
# =========================

class OptimizerFactory(_BaseFactory):
    def __init__(self, param_groups, distributed=False):
        super().__init__(param_groups)

        if distributed:
            self.backend = _DistributedBackend(self)
        else:
            self.backend = _LocalBackend(self)

    @torch.no_grad()
    def step(self, *args, **kwargs):
        self.backend.step(*args, **kwargs)