"""Reference NorMuon updates with the existing whole-matrix sharding."""
from gpt_lab.optim.kernels.normuon import normuon_step_fused
from gpt_lab.optim.strategies.muon import MuonStrategy


class NorMuonStrategy(MuonStrategy):
    step_kernel = staticmethod(normuon_step_fused)

    @staticmethod
    def _normalization_dim(shape) -> int:
        if len(shape) != 2:
            raise ValueError("NorMuon groups require same-shaped 2D parameters")
        return -1

    @staticmethod
    def _learning_rate(group: dict, shape) -> float:
        return group["lr"]
