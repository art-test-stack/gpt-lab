import pytest
import torch

from gpt_lab.model.utils import apply_rope, precompute_rope


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pairwise_split", [True, False])
def test_rope_stack_preserves_slice_assignment_outputs_and_gradients(dtype, cache_dtype, pairwise_split):
    torch.manual_seed(42)
    # A head slice has the same kind of row gaps as the packed projection's Q/K view.
    x = torch.randn(2, 7, 9, 8, dtype=dtype)[..., :6, :].requires_grad_()
    cache = precompute_rope(7, 8, dtype=cache_dtype, device=torch.device("cpu")).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    reference_cache = cache.detach().clone().requires_grad_()
    sin = reference_cache[..., 0][None, :, None, :]
    cos = reference_cache[..., 1][None, :, None, :]
    x1, x2 = (reference_x[..., ::2], reference_x[..., 1::2]) if pairwise_split else reference_x.chunk(2, -1)
    expected = torch.empty_like(reference_x)
    expected[..., ::2] = x1 * cos - x2 * sin
    expected[..., 1::2] = x1 * sin + x2 * cos

    actual = apply_rope(x, cache, pairwise_split=pairwise_split)

    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dy = torch.randn_like(actual)
    actual.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(cache.grad, reference_cache.grad)
