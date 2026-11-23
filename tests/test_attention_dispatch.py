from importlib import import_module, reload
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace, ModuleType
import copy
import subprocess
import sys
import textwrap

import pytest
import torch


@pytest.fixture(autouse=True)
def clear_kernel_cache():
    from gpt_lab.utils.import_utils import _attention_kernels
    _attention_kernels.clear()
    yield
    _attention_kernels.clear()


@pytest.mark.parametrize("mode", ["sdpa", "fa3"])
def test_public_attention_compiles_before_any_eager_call(mode):
    code = """
        from importlib import import_module
        from types import SimpleNamespace
        import sys
        import torch

        attention = import_module("gpt_lab.model.flash_attn")
        assert not attention._flash_attn_initialized
        mode = sys.argv[1]
        attention.is_torch_cuda_available = lambda: mode == "fa3"
        if mode == "fa3":
            torch.cuda.get_device_capability = lambda: (9, 0)
            attention.get_attention_kernel = lambda backend: SimpleNamespace(flash_attn_func=attention._sdpa_func)
        backend = "sdpa" if mode == "sdpa" else None
        q = torch.randn(1, 4, 2, 8)
        compiled = torch.compile(
            lambda q: attention.flash_attn_func(q, q, q, backend=backend),
            backend="eager", fullgraph=True,
        )
        output = compiled(q)
        assert attention._flash_attn_initialized == (mode != "sdpa")
        assert (attention._flash_attn is not None) == (mode == "fa3")
        torch.testing.assert_close(output, attention._sdpa_func(q, q, q))
    """
    process = subprocess.run([sys.executable, "-c", textwrap.dedent(code), mode],
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("backend", ["fa2", "fa3", "example/attention"])
@pytest.mark.parametrize("case", ["default", "packed", "missing_packed", "gqa", "unequal_lengths"])
def test_attention_selects_layout_and_preserves_outputs_and_gradients(monkeypatch, backend, case):
    attention = import_module("gpt_lab.model.flash_attn")
    torch.manual_seed(42)
    heads = 1 if case == "gqa" else 2
    key_length = 5 if case == "unequal_lengths" else 4
    inputs = (
        torch.randn(2, 4, 2, 8, requires_grad=True),
        torch.randn(2, key_length, heads, 8, requires_grad=True),
        torch.randn(2, key_length, heads, 8, requires_grad=True),
    )
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    calls = []

    def reference(q, k, v, **kwargs):
        return attention._sdpa_fallback(q, k, v, use_gqa=q.size(-2) != k.size(-2), **kwargs)

    # These substitute only the CUDA entrypoints; arithmetic uses real CPU SDPA.
    def separate(q, k, v, **kwargs):
        calls.append(("separate", kwargs))
        return reference(q, k, v, **kwargs)

    def packed(qkv, **kwargs):
        assert qkv.shape == (2, 4, 3, 2, 8)
        assert qkv.is_contiguous()
        calls.append(("packed", kwargs))
        return reference(*qkv.unbind(dim=2), **kwargs)

    kernel = SimpleNamespace(flash_attn_func=separate)
    if case != "missing_packed":
        kernel.flash_attn_qkvpacked_func = packed
    if "/" in backend:
        def get_kernel(name, **kwargs):
            assert name == backend
            assert kwargs == {"revision": "main", "trust_remote_code": [backend]}
            return kernel
        monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=get_kernel))
    else:
        monkeypatch.setattr(attention, "require_flash_attn2" if backend == "fa2" else "require_flash_attn3",
                            lambda: kernel)
    kwargs = dict(softmax_scale=0.25, causal=True, window_size=(1, 0), deterministic=True)

    options = {} if case == "default" else {"use_packed": True}
    output = attention.flash_attn_func(*inputs, backend=backend, **options, **kwargs)
    expected = reference(*reference_inputs, **kwargs)

    expected_kwargs = kwargs | ({"dropout_p": 0.0, "alibi_slopes": None} if backend != "fa3" else {})
    assert calls == [("packed" if case == "packed" else "separate", expected_kwargs)]
    torch.testing.assert_close(output, expected)
    gradients = torch.autograd.grad(output.square().sum(), inputs)
    expected_gradients = torch.autograd.grad(expected.square().sum(), reference_inputs)
    for actual, wanted in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, wanted)


@pytest.mark.parametrize("backend,repo,version", [
    ("fa3", "kernels-community/flash-attn3", 2),
    ("fa2", "kernels-community/flash-attn2", 4),
])
def test_builtin_kernel_versions(monkeypatch, backend, repo, version):
    from gpt_lab.utils.import_utils import get_attention_kernel
    calls = []
    kernel = SimpleNamespace(flash_attn_func=lambda *args: None)

    def get_kernel(name, **kwargs):
        calls.append((name, kwargs))
        return kernel

    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=get_kernel))
    assert get_attention_kernel(backend) is kernel
    assert get_attention_kernel(backend) is kernel
    assert calls == [(repo, {"version": version})]


@pytest.mark.parametrize("use_packed", [False, True])
def test_custom_kernel_with_fa3_signature(monkeypatch, use_packed):
    attention = import_module("gpt_lab.model.flash_attn")
    calls = []

    # FA3 does not accept FA2's dropout_p or alibi_slopes, even at their defaults.
    def separate(q, k, v, *, softmax_scale=None, causal=False,
                 window_size=(-1, -1), deterministic=False):
        calls.append("separate")
        return attention._sdpa_fallback(q, k, v, causal=causal, window_size=window_size,
                                        softmax_scale=softmax_scale)

    def packed(qkv, *, softmax_scale=None, causal=False,
               window_size=(-1, -1), deterministic=False):
        calls.append("packed")
        return attention._sdpa_fallback(*qkv.unbind(2), causal=causal,
                                        window_size=window_size, softmax_scale=softmax_scale)

    kernel = SimpleNamespace(flash_attn_func=separate)
    if use_packed:
        kernel.flash_attn_qkvpacked_func = packed
    loads = []
    def get_kernel(*args, **kwargs):
        loads.append(args)
        return kernel
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=get_kernel))
    inputs = tuple(torch.randn(1, 4, 2, 8, requires_grad=True) for _ in range(3))
    output = attention.flash_attn_func(*inputs, backend="example/fa3", causal=True, use_packed=True)
    expected = attention._sdpa_fallback(*inputs, causal=True)
    torch.testing.assert_close(output, expected)
    assert calls == ["packed" if use_packed else "separate"]
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(output.sum(), inputs))
    compiled = torch.compile(
        lambda q, k, v: attention.flash_attn_func(q, k, v, backend="example/fa3", causal=True, use_packed=True),
        backend="eager", fullgraph=True,
    )
    torch.testing.assert_close(compiled(*inputs), expected)
    assert len(loads) == 1
    with pytest.raises(ValueError, match="does not support dropout_p"):
        attention.flash_attn_func(*inputs, backend="example/fa3", dropout_p=0.1)
    with pytest.raises(ValueError, match="does not support alibi_slopes"):
        attention.flash_attn_func(*inputs, backend="example/fa3", alibi_slopes=torch.ones(2))


def test_custom_kernel_load_failure_has_no_sdpa_fallback(monkeypatch):
    attention = import_module("gpt_lab.model.flash_attn")

    def fail(*args, **kwargs):
        raise OSError("missing build")

    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=fail))
    inputs = tuple(torch.randn(1, 4, 2, 8) for _ in range(3))
    with pytest.raises(RuntimeError, match="example/missing.*failed to load") as error:
        attention.flash_attn_func(*inputs, backend="example/missing")
    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.parametrize("backend", ["fa2", "sdpa", "torch", "raw", "owner/kernel", "unknown"])
def test_public_packed_helper_rejects_unsupported_backends_before_loading(monkeypatch, backend):
    attention = import_module("gpt_lab.model.flash_attn")

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported packed backends must be rejected before loading")

    monkeypatch.setattr(attention, "_load_flash_attn_functions", unexpected)
    qkv = torch.randn(1, 4, 3, 2, 8)
    with pytest.raises(ValueError, match=f"Unsupported packed attention backend: {backend}"):
        attention.flash_attn_qkvpacked_func(qkv, backend=backend)


@pytest.mark.parametrize("entrypoint", ["public", "bound"])
@pytest.mark.parametrize("backend,fa3_loaded,window,expected", [
    (None, True, (512, 0), "fa2"),
    (None, True, (1, 0), "fa3"),
    (None, False, (512, 0), "fa2"),
    (None, False, (1, 0), "sdpa"),
    ("fa3", True, (512, 0), "fa3"),
])
def test_implicit_packed_dispatch_preserves_fa2_priority(monkeypatch, entrypoint, backend,
                                                        fa3_loaded, window, expected):
    attention = import_module("gpt_lab.model.flash_attn")
    calls = []
    output = torch.empty(1, 4, 2, 8, dtype=torch.bfloat16)

    def record(name):
        def packed(*args, **kwargs):
            calls.append(name)
            return output
        return packed

    fa2 = SimpleNamespace(flash_attn_qkvpacked_func=record("fa2"))
    fa3 = SimpleNamespace(flash_attn_func=lambda *args, **kwargs: output,
                          flash_attn_qkvpacked_func=record("fa3"))
    monkeypatch.setattr(attention, "_flash_attn_initialized", True)
    monkeypatch.setattr(attention, "_flash_attn", fa3 if fa3_loaded else None)
    monkeypatch.setattr(attention, "_flash_attn2", fa2)
    monkeypatch.setattr(attention, "_sdpa_qkvpacked_func", record("sdpa"))
    tensor = torch.empty(1, 4, 3, 2, 8, dtype=torch.bfloat16)
    # Simulate CUDA metadata while keeping dtype conversions on CPU.
    qkv = SimpleNamespace(is_cuda=True, dtype=tensor.dtype, size=tensor.size, to=tensor.to)
    options = dict(causal=True, window_size=window, softmax_scale=0.25, deterministic=True)

    if entrypoint == "public":
        actual = attention.flash_attn_qkvpacked_func(qkv, backend=backend, **options)
    else:
        actual = attention.load_flash_attn(backend).flash_attn_qkvpacked_func(qkv, **options)
    assert actual is output
    assert calls == [expected]


@pytest.mark.parametrize("fa3_loaded", [False, True])
@pytest.mark.parametrize("packed_entrypoint", ["missing", "not_callable"])
def test_implicit_separate_and_cache_allow_fa2_without_packed(monkeypatch, fa3_loaded, packed_entrypoint):
    attention = import_module("gpt_lab.model.flash_attn")
    fa2 = SimpleNamespace(flash_attn_func=attention._sdpa_fallback)
    if packed_entrypoint == "not_callable":
        fa2.flash_attn_qkvpacked_func = object()
    fa3 = SimpleNamespace(flash_attn_func=attention._sdpa_fallback,
        flash_attn_qkvpacked_func=attention._sdpa_qkvpacked_func,
        flash_attn_with_kvcache=attention._sdpa_with_kvcache)
    monkeypatch.setattr(attention, "_flash_attn_initialized", True)
    monkeypatch.setattr(attention, "_flash_attn", fa3 if fa3_loaded else None)
    monkeypatch.setattr(attention, "_flash_attn2", fa2)
    q, k, v = torch.randn(3, 1, 4, 2, 8, dtype=torch.bfloat16).unbind(0)
    expected = attention._sdpa_func(q, k, v, causal=True)

    torch.testing.assert_close(attention.flash_attn_func(q, k, v, causal=True), expected)
    k_cache, v_cache = torch.zeros_like(k), torch.zeros_like(v)
    torch.testing.assert_close(attention.flash_attn_with_kvcache(q, k_cache, v_cache, k, v,
        cache_seqlens=torch.zeros(1, dtype=torch.int32), causal=True), expected)
    tensor = torch.stack((q, k, v), dim=2)
    qkv = SimpleNamespace(is_cuda=True, dtype=tensor.dtype, size=tensor.size,
                          to=tensor.to, unbind=tensor.unbind)
    torch.testing.assert_close(
        attention.flash_attn_qkvpacked_func(qkv, causal=True, window_size=(512, 0)), expected)


def test_custom_kernel_kvcache_dispatch(monkeypatch):
    attention = import_module("gpt_lab.model.flash_attn")
    q = torch.randn(1, 1, 2, 8)
    cache = torch.zeros(1, 4, 2, 8)
    calls = []

    def with_kvcache(*args, **kwargs):
        calls.append((args, kwargs))
        return q

    kernel = SimpleNamespace(flash_attn_func=lambda *args, **kw: q,
                             flash_attn_with_kvcache=with_kvcache)
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=lambda *args, **kw: kernel))
    assert attention.flash_attn_with_kvcache(q, cache, cache, backend="example/cache") is q
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0][0], (q, cache, cache)))
    del kernel.flash_attn_with_kvcache
    with pytest.raises(RuntimeError, match="does not support KV-cache decoding"):
        attention.flash_attn_with_kvcache(q, cache, cache, backend="example/cache")


@pytest.mark.parametrize("backend,capability,warning", [
    ("fa3", 9, "FlashAttention 3 backend initialization failed"),
    ("fa2", 8, "FlashAttention 2 unavailable"),
])
def test_loader_logs_warning_and_preserves_strict_error(monkeypatch, caplog, backend, capability, warning):
    attention = import_module("gpt_lab.model.flash_attn")
    calls = []
    failure = OSError("missing build")
    emit = caplog.handler.emit

    def check_availability(record):
        assert attention._flash_attn_initialized
        assert not attention._flash_attn_init_lock.locked()
        assert not attention.flash_attention_is_installed
        assert not attention.flash_attn.flash_attention_is_installed
        emit(record)

    def fail(name):
        calls.append(name)
        raise failure

    for name in ("_flash_attn", "_flash_attn2", "_flash_attn_load_error", "_flash_attn2_load_error"):
        monkeypatch.setattr(attention, name, None)
    monkeypatch.setattr(attention, "_flash_attn_initialized", False)
    monkeypatch.setattr(attention, "is_torch_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (capability, 0))
    monkeypatch.setattr(attention, "get_attention_kernel", fail)
    monkeypatch.setattr(caplog.handler, "emit", check_availability)
    with caplog.at_level("WARNING", logger=attention.__name__):
        with pytest.raises(RuntimeError, match="refusing SDPA fallback") as error:
            attention.load_flash_attn(backend)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert warning in caplog.records[0].message
    assert error.value.__cause__ is failure
    assert calls == [backend]


def test_loader_returns_bound_namespace_and_initializes_once(monkeypatch):
    attention = import_module("gpt_lab.model.flash_attn")
    loads = []
    kernel = ModuleType("native_attention")
    kernel.flash_attn_func = lambda q, k, v, **kwargs: q

    def load(name):
        loads.append(name)
        return kernel

    monkeypatch.setattr(attention, "_flash_attn", None)
    monkeypatch.setattr(attention, "_flash_attn_initialized", False)
    monkeypatch.setattr(attention, "is_torch_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(attention, "get_attention_kernel", load)
    bound = attention.load_flash_attn("fa3")
    assert isinstance(bound, SimpleNamespace)
    assert bound.flash_attention_is_installed and attention.flash_attn.flash_attention_is_installed
    q = torch.randn(1, 4, 2, 8)
    assert bound.flash_attn_func(q, q, q) is q
    assert copy.deepcopy(bound).flash_attn_func(q, q, q) is q
    attention.load_flash_attn("fa3")
    assert loads == ["fa3"]


@pytest.mark.parametrize("caller", ["strict", "implicit", "namespace_flag", "module_flag"])
def test_concurrent_first_use_waits_for_fa3_initialization(monkeypatch, caller):
    attention = import_module("gpt_lab.model.flash_attn")
    loading, release, caller_started = Event(), Event(), Event()
    loads = []
    kernel = SimpleNamespace(flash_attn_func=lambda q, k, v, **kwargs: q)

    def load(name):
        loads.append(name)
        loading.set()
        assert release.wait(timeout=5)
        return kernel

    def concurrent_caller():
        caller_started.set()
        if caller == "namespace_flag":
            return attention.flash_attn.flash_attention_is_installed
        if caller == "module_flag":
            return attention.flash_attention_is_installed
        return attention.load_flash_attn(None) if caller == "implicit" else attention.require_flash_attn3()

    monkeypatch.setattr(attention, "_flash_attn", None)
    monkeypatch.setattr(attention, "_flash_attn_initialized", False)
    monkeypatch.setattr(attention, "is_torch_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(attention, "get_attention_kernel", load)
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(attention.require_flash_attn3)
        try:
            assert loading.wait(timeout=5)
            second = executor.submit(concurrent_caller)
            assert caller_started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
            assert not attention._flash_attn_initialized
            assert attention._flash_attn is None
        finally:
            release.set()
        assert first.result(timeout=5) is kernel
        result = second.result(timeout=5)

    if caller.endswith("_flag"):
        assert result is True
    elif caller == "implicit":
        assert result.flash_attention_is_installed
        q = torch.randn(1, 4, 2, 8)
        assert result.flash_attn_func(q, q, q) is q
    else:
        assert result is kernel
    assert attention._flash_attn_initialized
    assert attention.flash_attention_is_installed
    assert attention.flash_attn.flash_attention_is_installed
    assert loads == ["fa3"]


def test_kvcache_retains_implicit_fa3_selection(monkeypatch):
    attention = import_module("gpt_lab.model.flash_attn")
    q = torch.randn(1, 1, 2, 8)
    cache = torch.zeros(1, 4, 2, 8)
    calls = []

    def native(*args, **kwargs):
        calls.append(args)
        return q

    monkeypatch.setattr(attention, "_flash_attn_initialized", True)
    monkeypatch.setattr(attention, "_flash_attn", SimpleNamespace(
        flash_attn_func=lambda *args, **kwargs: q, flash_attn_with_kvcache=native))
    assert attention.flash_attn_with_kvcache(q, cache, cache) is q
    assert len(calls) == 1


def test_backend_loading_waits_for_the_helper(monkeypatch):
    attention = import_module("gpt_lab.model.flash_attn")
    imports = import_module("gpt_lab.utils.import_utils")
    calls = []

    def load(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(imports, "get_attention_kernel", load)
            patch.setattr(imports, "is_torch_cuda_available", lambda: True)
            patch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
            reload(attention)
            assert not calls
    finally:
        reload(attention)


@pytest.mark.parametrize("surface", ["namespace", "module"])
@pytest.mark.parametrize("outcome", ["available", "missing", "cpu"])
def test_public_availability_flag_initializes_on_first_read(monkeypatch, caplog, surface, outcome):
    attention = import_module("gpt_lab.model.flash_attn")
    public = attention.flash_attn if surface == "namespace" else attention
    loads = []
    kernel = SimpleNamespace(flash_attn_func=lambda q, k, v, **kwargs: q)

    def load(name):
        loads.append(name)
        if outcome == "missing":
            raise OSError("missing build")
        return kernel

    monkeypatch.setattr(attention, "_flash_attn", None)
    monkeypatch.setattr(attention, "_flash_attn_load_error", None)
    monkeypatch.setattr(attention, "_flash_attn_initialized", False)
    monkeypatch.setattr(attention, "is_torch_cuda_available", lambda: outcome != "cpu")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(attention, "get_attention_kernel", load)
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    assert not loads

    with caplog.at_level("WARNING", logger=attention.__name__):
        installed = public.flash_attention_is_installed
    expected = outcome == "available"
    assert installed is expected
    assert attention._flash_attn_initialized
    assert public.flash_attention_is_installed is expected
    assert attention.flash_attention_is_installed is expected
    assert attention.flash_attn.flash_attention_is_installed is expected
    if installed:
        assert attention.require_flash_attn3() is kernel
    assert loads == ([] if outcome == "cpu" else ["fa3"])
    assert len(caplog.records) == int(outcome == "missing")
    if outcome == "missing":
        assert caplog.records[0].levelname == "WARNING"
        assert "FlashAttention 3 backend initialization failed" in caplog.records[0].message


@pytest.mark.parametrize("backend", ["torch", "raw"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_legacy_attention_configs_support_cached_model_evaluation(backend, dtype):
    from gpt_lab.model.gpt import DenseTransformer
    from gpt_lab.model.utils import KVCache
    from gpt_lab.utils.schemas import TransformerConfig

    config = TransformerConfig(vocab_size=64, max_context=4, d_model=16, d_ffn=32,
        n_heads=2, d_head=8, n_layers=1, attn_impl=backend, window_pattern="L")
    model = DenseTransformer(config).to(device="cpu", dtype=dtype).eval()
    model.init_weights()
    # Keep attention's contribution nonzero so incorrect cache contents affect logits.
    torch.nn.init.normal_(model.blocks[0].attention.o_proj.weight, std=0.1)
    reference = DenseTransformer(config.model_copy(update={"attn_impl": "sdpa"}))
    reference.to(device="cpu", dtype=dtype).eval()
    reference.load_state_dict(model.state_dict())
    cache = KVCache(config=config)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))

    for chunk in (slice(0, 3), slice(3, 4)):
        actual = model(input_ids[:, chunk], past_key_values=cache)
        expected = reference(input_ids[:, :chunk.stop])
        torch.testing.assert_close(actual.logits.to(dtype), expected.logits.to(dtype))
        assert cache.cache.dtype == dtype
        assert cache.device == cache.seqlens.device == input_ids.device
        assert cache.seqlens.dtype == torch.int32
        cache.advance()
        assert cache.cur_pos == chunk.stop


@pytest.mark.parametrize("backend", ["torch", "raw"])
@torch.no_grad()
def test_legacy_attention_configs_support_layer_weight_inspection(backend):
    from gpt_lab.model.layers import CausalSelfAttention
    from gpt_lab.utils.schemas import TransformerConfig

    config = TransformerConfig(d_model=16, d_ffn=32, n_heads=2, d_head=8,
                              n_layers=1, attn_impl=backend)
    attention = CausalSelfAttention(config).cpu().eval()
    reference = CausalSelfAttention(config.model_copy(update={"attn_impl": "sdpa"})).cpu().eval()
    reference.load_state_dict(attention.state_dict())
    x = torch.randn(2, 3, 16)

    output, weights = attention(x, return_attn_weights=True)
    expected_output, expected_weights = reference(x, return_attn_weights=True)
    assert weights is not None and torch.isfinite(weights).all()
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(weights, expected_weights)
    for training in (True, False):
        attention.train(training)
        with pytest.raises(ValueError, match=f"Attention implementation {backend} is not supported"):
            attention(x)
