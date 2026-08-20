import pytest
from gpt_lab.model.layers import (
    apply_rms_norm,
    scaled_dot_product_attention,
)
from gpt_lab.model.utils import (
    SelfAttentionMask,
    apply_rope,
    precompute_rope, 
)
from gpt_lab.utils.schemas import GPTConfig
import torch
import torch.nn.functional as F
import time
import sys
from types import SimpleNamespace

from gpt_lab.model.flash_attn import (
    _sdpa_fallback,
    flash_attn_qkvpacked_func,
)


@pytest.mark.fast
def test_rms_norm():
    x = torch.rand(4, 16, 32)
    eps = 1e-8
    t0 = time.time()
    rms_normed_x1 = apply_rms_norm(x, eps=eps, torch_impl=False)
    custom_time = time.time() - t0
    t0 = time.time()
    rms_normed_x2 = F.rms_norm(x, normalized_shape=(x.size(-1),), eps=eps)
    torch_time = time.time() - t0

    assert torch.allclose(rms_normed_x1, rms_normed_x2), "RMSNorm output does not match expected output"

class TestSelfAttentionMask:
    pad_idx = 0
    max_context = 8
    mask_generator = SelfAttentionMask(pad_idx=pad_idx, max_context=max_context)
    expected_mask = torch.tensor([[[
        [0,  -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [0, 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [0, 0, 0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [0, 0, 0, 0     , -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [0, 0, 0, 0     , 0     , -float('inf'), -float('inf'), -float('inf')],
        [0, 0, 0, 0     , 0     , 0     , -float('inf'), -float('inf')],          
        [0, 0, 0, 0     , 0     , 0     , 0     , -float('inf')],
        [0, 0, 0, 0     , 0     , 0     , 0     , 0]]]]
    )

    @pytest.mark.fast
    def test_self_attention_mask(self):

        input_ids = torch.arange(1, self.max_context + 1).unsqueeze(0)
        attn_mask = self.mask_generator(input_ids, mask_pad_token=False, to_bool=False, is_causal=True)
        
        assert attn_mask is None, "Attention mask should be None when mask_pad_token is False."
 
    @pytest.mark.fast
    def test_self_attention_mask_with_padding(self):
        input_ids = torch.arange(1, self.max_context - 2).tolist() + [self.pad_idx] * 3
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        attn_mask = self.mask_generator(input_ids, to_bool=False)
        expected_mask_with_padding = self.expected_mask.clone()
        expected_mask_with_padding[0, 0, :, -3:] = -float('inf')

        assert torch.equal(attn_mask, expected_mask_with_padding), "Attention mask does not match expected mask for padding case."

class TestSDPA:
    B, H, S, E = 4, 4, 16, 8
    query = torch.rand(B, H, S, E)
    key = torch.rand(B, H, S, E)
    value = torch.rand(B, H, S, E)

    @pytest.mark.fast
    def test_sdpa_not_causal(self):
        B, H, S, E = self.B, self.H, self.S, self.E
        query = self.query
        key = self.key
        value = self.value

        sdpa, _ = scaled_dot_product_attention(query, key, value)

        reference_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        self.common_assertions(sdpa, reference_output, (B, H, S, E))

    @pytest.mark.fast
    def test_sdpa_causal(self):
        B, H, S, E = self.B, self.H, self.S, self.E
        query = self.query
        key = self.key
        value = self.value

        sdpa, _ = scaled_dot_product_attention(query, key, value, is_causal=True)

        ref_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )

        self.common_assertions(sdpa, ref_output, (B, H, S, E))
    
    @pytest.mark.fast
    def test_sdpa_with_mask(self):
        B, H, S, E = self.B, self.H, self.S, self.E
        query = self.query
        key = self.key
        value = self.value

        attn_mask = SelfAttentionMask(pad_idx=-100, max_context=self.S).get(
            seq=torch.randint(0, 10, (B, S)), mask_pad_token=False, to_bool=False, is_causal=True
        )

        sdpa, _ = scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

        ref_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        self.common_assertions(sdpa, ref_output, (B, H, S, E))

    def common_assertions(self, output: torch.Tensor, ref: torch.Tensor, shape):
        assert output.shape == shape, f"Unexpected SDPA output shape: {output.shape}. Expected: {shape}. Got: {output.shape}"
        assert torch.isfinite(output).all(), "SDPA output contains non-finite values."
        assert torch.allclose(output, ref), "Custom SDPA output does not match reference output."


@pytest.mark.fast
def test_flash_fallback_local_attention_is_causal_and_uses_keep_mask():
    q = torch.zeros(1, 4, 1, 2)
    k = torch.zeros_like(q)
    v = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1).expand(-1, -1, -1, 2)
    qkv = torch.stack((q, k, v), dim=2)

    output = flash_attn_qkvpacked_func(
        qkv,
        causal=True,
        window_size=(1, 0),
    )

    allowed = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, True, True, False],
            [False, False, True, True],
        ]
    )
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=allowed,
    ).transpose(1, 2)

    assert torch.allclose(output, expected)


@pytest.mark.fast
def test_flash_fallback_aligns_cached_queries_to_bottom_right():
    q = torch.zeros(1, 2, 1, 2)
    k = torch.zeros(1, 5, 1, 2)
    v = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1).expand(-1, -1, -1, 2)

    output = _sdpa_fallback(
        q,
        k,
        v,
        causal=True,
        window_size=(2, 0),
    )

    allowed = torch.tensor(
        [
            [False, True, True, True, False],
            [False, False, True, True, True],
        ]
    )
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=allowed,
    ).transpose(1, 2)

    assert torch.allclose(output, expected)


@pytest.mark.fast
def test_missing_flash_kernel_does_not_recurse(monkeypatch):
    import gpt_lab.utils.import_utils as import_utils

    calls = 0

    def fail_get_kernel(_name):
        nonlocal calls
        calls += 1
        raise RuntimeError("missing kernel")

    monkeypatch.setattr(import_utils, "is_torch_cuda_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "kernels",
        SimpleNamespace(get_kernel=fail_get_kernel),
    )
    import_utils.is_flash_attn3_available_from_kernel.cache_clear()
    try:
        with pytest.warns(UserWarning, match="Falling back to PyTorch SDPA"):
            assert not import_utils.is_flash_attn3_available_from_kernel()
        assert calls == 1
    finally:
        import_utils.is_flash_attn3_available_from_kernel.cache_clear()


class TestRoPE:
    _bs = 4
    seq_len = 16
    d_head = 8
    n_head = 4

    @pytest.mark.fast
    def test_precompute_rope(self):
        d_head = self.d_head
        seq_len = self.seq_len
        with pytest.raises(AssertionError) as excinfo:
            precompute_rope(seq_len, d_head + 1, device='cpu')  # d_head must be even
        assert excinfo.type is AssertionError

        rope_cache = precompute_rope(seq_len, d_head, device='cpu')
        assert rope_cache.shape == (seq_len, d_head // 2, 2), f"Unexpected rope_cache shape: {rope_cache.shape}"

    @pytest.mark.fast
    def test_rope(self):
        _bs = self._bs
        n_head = self.n_head
        d_head = self.d_head
        seq_len = self.seq_len

        rope_cache = precompute_rope(seq_len, d_head, device='cpu')
        assert rope_cache.shape == (seq_len, d_head // 2, 2), f"Unexpected rope_cache shape: {rope_cache.shape}"

        x = torch.randn(_bs, seq_len, d_head)
        with pytest.raises(AssertionError) as excinfo:
            apply_rope(x, rope_cache)  # x must have shape (B, n_heads, seq_len, d_head)
        assert excinfo.type is AssertionError

        x = torch.randn(_bs, seq_len, n_head, d_head) 
        x_rope = apply_rope(x, rope_cache)
        assert x_rope.shape == x.shape, f"Output shape mismatch: {x_rope.shape} != {x.shape}"
