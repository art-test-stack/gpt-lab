from gpt_lib.model.trainer import Trainer
from gpt_lib.model.arg import ModelArgs
from gpt_lib.model.gpt import MichelTransformer
from gpt_lib.train.optimizer import AdamW

from gpt_lib.data.datasets.dataset import Dataset
from gpt_lib.data.datasets.wikipedia import WikipediaDataset
# from michelgpt.data.tokenizer.models import HGFBPETokenizer as Tokenizer
from gpt_lib.tokenizer.tok import TikTokenizer as Tokenizer
from gpt_lib.utils import get_logger, rank_log, verify_min_gpu_count
from gpt_lib.utils.default import *

import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)
from torch.distributed.tensor.device_mesh import init_device_mesh

from typing import Callable


# ---- GPU check ------------
_min_gpu_count = 4

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

class FSDPTrainer:
    
    def __init__(
            self,
            model: MichelTransformer = MichelTransformer(),
            tokenizer: Tokenizer = Tokenizer(), 
            optimizer: optim.Optimizer | Callable = None, 
            padding_token: int = 1,
            device: torch.device = DEVICE
        ):
        print(f"GPU COUNT = {torch.cuda.device_count()}")
        tp_size = 2
        self.logger = get_logger()

        self._world_size = int(os.environ["WORLD_SIZE"])

        # device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(self._world_size,))
        # self._rank = device_mesh.get_rank()
        self._rank = int(os.environ["RANK"])

        print(f"{self._world_size}")
        print(f"Starting PyTorch 2D (FSDP + TP) example on rank {self._rank}.")
        assert (
            self._world_size % tp_size == 0
        ), f"World size {self._world_size} needs to be divisible by TP size {tp_size}"


        # create a sharding plan based on the given world_size.
        dp_size = self._world_size // tp_size

        # Create a device mesh with 2 dimensions.
        # First dim is the data parallel dimension
        # Second dim is the tensor parallel dimension.
        device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

        # rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")
        tp_mesh = device_mesh["tp"]
        dp_mesh = device_mesh["dp"]

        # For TP, input needs to be same across all TP ranks.
        # while for SP, input can be different across all ranks.
        # We will use dp_rank for setting the random seed
        # to mimic the behavior of the dataloader.
        self.dp_rank = dp_mesh.get_local_rank()

        # create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
        simple_config = Settings()

        model = MichelTransformer(simple_config).cuda()
        model.init_weights()

        model = parallelize_module(
            model,
            tp_mesh,
            {
                "embedding": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                # "norm": SequenceParallel(),
                "model_head": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate()
                ),
            }
        )

        for layer_id, transformer_block in enumerate(model.decoder_stack.layers):
            layer_tp_plan = {
                # "attention_norm": SequenceParallel(),
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.w_q": ColwiseParallel(),
                "attention.w_k": ColwiseParallel(),
                "attention.w_v": ColwiseParallel(),
                # "attention.w_o": RowwiseParallel(output_layouts=Shard(1)),
                # "dropout": SequenceParallel(),
                "layer_norm": SequenceParallel(),
                "ffn": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "ffn.w_1": ColwiseParallel(),
                "ffn.w_2": RowwiseParallel(output_layouts=Shard(1)),
                # "ffn.w_2": ColwiseParallel(),
                "ffn.layer_norm": SequenceParallel(),
                # "ffn.layer_norm": RowwiseParallel(output_layouts=Shard(1)),
            }
            attn_layer = transformer_block.attention
            attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
            attn_layer.d_head = attn_layer.d_head // tp_mesh.size()

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan
            )

        # Init FSDP using the dp device mesh
        self.model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)

        self.optimizer = AdamW(self.model.parameters(), foreach=True)
        rank_log(self._rank, self.logger, f"Model after parallelization {self.model=}\n")
        self.trainer = Trainer(
            model=self.model, 
            tokenizer=tokenizer, 
            optimizer=optimizer, 
            padding_token=padding_token, 
            device=device
        )

    def fit(self):
        # Create an optimizer for the parallelized and sharded model.

        # Training loop:
        # Perform a num of iterations of forward/backward
        # and optimizations for the sharded module.
        dataset = WikipediaDataset()
        self.trainer.fit(dataset=dataset)
        rank_log(self._rank, self.logger, "\nStarting 2D training...")

        # seeding with dp_rank to ensure identical inputs for TP groups
        
        # rank_log(self._rank, self.logger, f"2D iter {i} complete")

        rank_log(self._rank, self.logger, "2D training successfully completed!")