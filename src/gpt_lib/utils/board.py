from typing import Literal
from pathlib import Path
try:
    import wandb
except:
    wandb = None
try:
    from torch.utils import tensorboard as tb
except:
    tb = None
from gpt_lib.utils.default import BOARD_DIR
import warnings


class DummyBoard:
    def __init__(self) -> None:
        pass

    def log(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwds):
        pass
 

class Board:
    def __init__(
            self, 
            board_type: Literal["wandb", "trackio", "tensorboard", "dummy"],
            project: str | None = None,
            run: str | None = None,
            config: dict | None = None,
            board_dir: str | Path | None = None,
        ) -> None:
        self.board_type = board_type
        if board_dir is None:
            board_dir = BOARD_DIR
        config = config
        if self.board_type == "wandb":
            if wandb is None:
                raise ImportError("wandb is not installed. Please install wandb to use the wandb board.")
            self.main = wandb.init(
                project=project,
                name=run,
                dir=board_dir, # TODO
                config=config,
            )
            
        elif self.board_type == "trackio":
            import trackio
            self.main = trackio.init(
                project=project,
                name=run,
                dir=board_dir, # TODO
                config=config,
            )
        elif self.board_type == "tensorboard":
            if tb is None:
                raise ImportError("tensorboard is not installed. Please install tensorboard to use the tensorboard board.")
            self.main = tb.SummaryWriter(
                log_dir=board_dir
            ) # TODO
            print("config: ", config)
            for key, value in config.items():
                import torch
                if not type(value) in [str, int, float, bool, torch.Tensor]:
                    config[key] = str(value)
                    
            self.main.add_hparams(config, {}, run_name=f"{project}.{run}") # log config as hparams
        else:
            self.main = DummyBoard()

    def log(self, data: dict, step: int | None = None) -> None:
        self.main.log(data, step)

    def close(self):
        if self.board_type == "tensorboard":
            self.main.flush()
            self.main.close()
        elif self.board_type == "wandb":
            wandb.finish()
        