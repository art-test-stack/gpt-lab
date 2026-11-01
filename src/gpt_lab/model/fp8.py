from functools import cache

import torch

from gpt_lab.model.layers import Linear, SquaredReLUFeedForward, SwigLUFeedForward


def check_fp8_support(device, compute_dtype):
    """Initial native target is Hopper SM90 with BF16 activations."""
    if compute_dtype != torch.bfloat16:
        raise ValueError("FP8 FFN requires BF16 compute")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FP8 FFN requires a native Hopper CUDA GPU; use BF16 on CPU/A100")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("FP8 FFN initially supports Hopper SM90 (H100/H200); use BF16 on A100")


def enable_fp8_ffn_forward(model, compute_dtype):
    """Convert native SwiGLU or squared-ReLU FFN matrices, preserving Parameter objects and keys.

    Scaling is dynamic and stateless; checkpoints need no quantization history.
    """
    check_fp8_support(next(model.parameters()).device, compute_dtype)
    if any(type(block.ffn) not in (SwigLUFeedForward, SquaredReLUFeedForward) for block in model.blocks):
        raise ValueError("FP8 requires native SwiGLU or squared-ReLU FFNs; architecture is never changed automatically")
    locations = [(block.ffn, name) for block in model.blocks for name in ("w_1", "w_2")]
    for parent, name in locations:
        linear = getattr(parent, name)
        if type(linear) is not Linear or linear.weight.dtype != torch.float32:
            raise ValueError("Expected unconverted FFN Linear modules with FP32 parameter storage")
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError("FP8 FFN matrix dimensions must be divisible by 16; automatic padding is disabled")
    try:
        Float8FFNLinear = _float8_linear_class()
    except ImportError as error:
        raise ImportError("Install the optional backend: uv sync --locked --extra gpu --extra fp8") from error
    converted = [(parent, name, Float8FFNLinear.from_linear(getattr(parent, name))) for parent, name in locations]
    for parent, name, module in converted:
        setattr(parent, name, module)
    return len(converted)


@cache
def _float8_linear_class():
    """Load the torchao 0.15.0 adapter only for explicit FP8 conversion."""
    from dataclasses import replace

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

    return Float8FFNLinear


class DisableFP8:
    def __init__(self, model):
        self.model = model
        self.fp8_locations = None
        self._active = False

    def _find_modules(self):
        locations = []
        for name, module in self.model.named_modules():
            if 'Float8' in type(module).__name__:
                if '.' in name:
                    parent_name, attr_name = name.rsplit('.', 1)
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model
                    attr_name = name
                locations.append((parent, attr_name, module))
        return locations

    def __enter__(self, *args, **kwargs):
        if self._active:
            raise RuntimeError("DisableFP8 is not reentrant")

        if self.fp8_locations is None:
            self.fp8_locations = self._find_modules()

        if not self.fp8_locations:
            self._active = True
            return self.model

        for parent, attr_name, fp8_module in self.fp8_locations:
            linear = Linear(
                fp8_module.in_features,
                fp8_module.out_features,
                bias=fp8_module.bias is not None,
                device="meta",
                dtype=fp8_module.weight.dtype,
            )
            linear.weight = fp8_module.weight
            if fp8_module.bias is not None:
                linear.bias = fp8_module.bias
            linear.train(fp8_module.training)

            setattr(parent, attr_name, linear)

        self._active = True
        return self.model

    def __exit__(self, exc_type, exc, tb):
        if not self._active:
            return

        for parent, attr_name, fp8_module in self.fp8_locations:
            setattr(parent, attr_name, fp8_module)

        self._active = False
