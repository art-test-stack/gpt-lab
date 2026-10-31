"""Optional torchao implementation; imported only by explicit FP8 conversion.

Pinned to torchao 0.15.0 because its low-level training API is not public.
"""

from dataclasses import replace

import torch
from torchao.float8 import Float8LinearConfig
from torchao.float8.config import CastConfig, Float8GemmConfig, ScalingType
from torchao.float8.float8_linear import Float8Linear, matmul_with_hp_or_float8_args


class Float8FFNLinear(Float8Linear):
    """FP8 forward GEMM; BF16 input/weight-gradient GEMMs and FP32 weights."""

    @classmethod
    def from_linear(cls, linear):
        high_precision = CastConfig(scaling_type=ScalingType.DISABLED,
                                    target_dtype=torch.float8_e4m3fn)
        config = replace(
            Float8LinearConfig.from_recipe_name("rowwise"),
            cast_config_grad_output=high_precision,
            cast_config_weight_for_grad_input=high_precision,
            cast_config_input_for_grad_weight=high_precision,
            cast_config_grad_output_for_grad_weight=high_precision,
            gemm_config_output=Float8GemmConfig(use_fast_accum=False),
        )
        result = cls.from_float(linear, config=config)
        result.train(linear.training)
        return result

    def forward(self, x):
        if x.device.type != "cuda":
            raise RuntimeError("Native FP8 requires CUDA")
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        if dtype != torch.bfloat16:
            raise RuntimeError("FP8 FFN requires BF16 activations")
        if x.ndim < 2 or x.shape[-1] != self.in_features or x.numel() // self.in_features % 16:
            raise ValueError("FP8 FFN requires a matching last dimension and a flattened token count divisible by 16")
        x = x.to(torch.bfloat16)
        # Match gpt-lab's temporary weight cast; stored parameters stay FP32.
        weight = self.weight.to(x.dtype)
        output = matmul_with_hp_or_float8_args.apply(x, weight.t(), self.linear_mm_config, self.config)
        return output if self.bias is None else output + self.bias.to(output.dtype)
