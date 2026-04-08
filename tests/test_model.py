import pytest
import torch
from gpt_lib.model.gpt import GPTModel
from gpt_lib.model.utils import KVCache

from gpt_lib.utils.schemas import (
    GPTConfig, 
    LossConfig, 
    TokenizerConfig, 
    TransformerConfig,
    GenerationConfig,
    ModelOutput,
)
import tempfile, warnings

# TODO: use fixtures for config and model initialization
# @pytest.fixture(scope="module")
# def model_config():
#     model_name = "test-model"
#     pad_token_id = 0
#     tmpdirname = tempfile.mkdtemp()
#     tokenizer_config = TokenizerConfig(
#         vocab_size=1000,
#         max_context=16,
#         name="simple-tokenizer",
#         source="dummy"
#     )
#     model_config = TransformerConfig(
#         vocab_size=1000,
#         pad_id=pad_token_id,
#         max_context=16,
#         d_model=16,
#         d_ffn=64,
#         n_heads=4,
#         n_layers=4,
#         d_head=4,
#         dropout=0.1
#     )
#     loss_config = LossConfig(
#         loss_fn="cross_entropy",
#         ignore_index=pad_token_id,
#         kwargs={"reduction": "mean"}
#     )
#     config = GPTConfig(
#         name=model_name,
#         tokenizer=tokenizer_config,
#         model=model_config,
#         loss=loss_config,
#         dirname=tmpdirname
#     )
#     return config


def assert_single_warning(fn, warning_type, message_substring=None):
    """
    Execute fn() and assert that exactly one warning of type `warning_type`
    is emitted. Optionally check that its message contains `message_substring`.
    """
    with pytest.warns(warning_type) as record:
        result = fn()

    assert len(record) == 1, (
        f"Expected exactly one {warning_type.__name__} "
        f"but got {len(record)} warnings."
    )
    assert record[0].category is warning_type, (
        f"Expected warning category {warning_type.__name__}, "
        f"got {record[0].category} instead."
    )
    if message_substring is not None:
        assert message_substring in str(record[0].message), (
            f"Expected warning message to contain '{message_substring}', "
            f"got '{record[0].message}'."
        )
    return result

class TestGPTModel:
    model_name = "test-model"
    pad_token_id = 0
    tmpdirname = tempfile.mkdtemp()
    tokenizer_config = TokenizerConfig(
        vocab_size=1000,
        name="simple-tokenizer",
        source="dummy"
    )
    model_config = TransformerConfig(
        vocab_size=1000,
        pad_id=pad_token_id,
        max_context=16,
        d_model=16,
        d_ffn=64,
        n_heads=4,
        n_layers=4,
        d_head=4,
        dropout=0.1
    )
    loss_config = LossConfig(
        loss_fn="cross_entropy",
        ignore_index=pad_token_id,
        kwargs={"reduction": "mean"}
    )
    config = GPTConfig(
        name=model_name,
        tokenizer=tokenizer_config,
        model=model_config,
        loss=loss_config,
        dirname=tmpdirname
    )

    # TODO: re-enable this test after implementing model saving and loading functionality
    @pytest.mark.fast
    def test_model_loading_saving(self):

        warnings.warn(f"This test has to be re-enabled after refounding model engine and generation logic.")
        assert True
        return
        # self.config.to_file(mode="pickle")

        # loaded_config = GPTConfig.from_file(model_name=self.model_name, model_dir=self.tmpdirname)

        # for key, value in self.config.__dict__.items():
        #     assert key in loaded_config.__dict__, f"Key {key} missing in loaded config"
        #     assert getattr(loaded_config, key) is not None, f"Key {key} is None in loaded config"
        #     assert getattr(loaded_config, key) == value, f"Value for key {key} does not match: {getattr(loaded_config, key)} != {value}"
        # assert loaded_config == self.config, "Loaded config does not match the original"
        
        # model = assert_single_warning(
        #     lambda: GPTModel.from_scratch(config=self.config),
        #     UserWarning,
        #     message_substring="DummyTokenizer",
        # )
        # assert model.main.embeds.weight.device != torch.device("meta"), "Model initialized on meta device"
        # # TODO: change that; this is a temporary check
        # model = model.main
        # model.save_checkpoint(ckpt_version="test-1", keep_vars=True)

        # loaded_model = assert_single_warning(
        #     lambda: GPTModel.load(model_name=self.model_name, ckpt_version="test-1", model_dir=self.tmpdirname),
        #     UserWarning,
        #     message_substring="DummyTokenizer",
        # )
        # # TODO: change that; this is a temporary check
        # loaded_model = loaded_model.main
        # assert loaded_model.config == self.config, "Loaded model config does not match the original"
        # assert loaded_model.main.state_dict().keys() == model.main.state_dict().keys(), "Loaded model state dict keys do not match the original"
        # assert all(torch.equal(loaded_model.main.state_dict()[k], model.main.state_dict()[k]) for k in model.main.state_dict().keys()), "Loaded model state dict values do not match the original"
    
    @pytest.mark.fast
    def test_model_forward(self):
        warnings.warn(f"This test has to be re-enabled after refounding model engine and generation logic.")
        assert True
        return
        # config = self.config
        # model = assert_single_warning(
        #     lambda: GPTModel.from_scratch(config=self.config),
        #     UserWarning,
        #     message_substring="DummyTokenizer",
        # )
        # model.main.eval()
        # max_context = config.model.max_context
        # vocab_size = config.model.vocab_size
        # batch_size = 4
        # # Dummy input ids and labels
        # input_ids = torch.randint(0, vocab_size, (batch_size, max_context), device=model.device)
        # labels = torch.randint(0, vocab_size, (batch_size, max_context), device=model.device)

        # generation_config = GenerationConfig(
        #     max_length=20,
        #     temperature=1.0,
        #     top_k=0,
        #     top_p=1.0,
        #     repetition_penalty=1.0,
        #     do_sample=True,
        #     num_return_sequences=1,
        #     stream=False
        # )
        # with torch.no_grad():
        #     output = model.forward(input_ids=input_ids, labels=labels, **generation_config.__dict__)
        #     logits = model(
        #         input_ids=input_ids,
        #         return_attentions=False,
        #         # log_prob=False,
        #         # temperature=generation_config.temperature
        #     ).logits

        # assert isinstance(output, ModelOutput), "Output is not an instance of ModelCompletionOutput"
        # assert logits.size(0) == batch_size, "Logits batch size does not match input batch size"
        # assert logits.size(1) == max_context, "Logits sequence length does not match input sequence length"
        # assert logits.size(2) == vocab_size, "Logits vocab size does not match model vocab size"
        # assert not torch.isnan(logits).any(), "Logits contain NaN values"
        # assert torch.isfinite(logits).all(), "Logits contain non-finite values"
        # assert (logits == output.logits).all(), "Logits from forward method do not match logits from __call__ method"

    @pytest.mark.fast
    def test_model_generation(self):
        warnings.warn(f"This test has to be re-enabled after refounding model engine and generation logic.")
        assert True
        return
        # config = self.config
        # model = assert_single_warning(
        #     lambda: GPTModel.from_scratch(config=self.config),
        #     UserWarning,
        #     message_substring="DummyTokenizer",
        # )
        # model.main.eval()
        # max_context = config.model.max_context
        # vocab_size = config.model.vocab_size
        # batch_size = 2
        # # Dummy input ids
        # input_ids = torch.randint(0, vocab_size, (batch_size, max_context), device=model.device)

        # generation_config = GenerationConfig(
        #     max_length=10,
        #     temperature=1.0,
        #     top_k=0,
        #     top_p=1.0,
        #     repetition_penalty=1.0,
        #     do_sample=False,
        #     num_return_sequences=1,
        #     stream=False,
        #     use_cache=False
        # )

        # import time
        # t1 = time.time()
        # with torch.no_grad(), pytest.warns(UserWarning) as record:
        #     results = model.generate(
        #         input_ids=input_ids,
        #         generation_config=generation_config
        #     )
        # t_no_cache = time.time() - t1
        # assert len(record) == 1, f"Expected one warning when 'use_cache' is set to False in generation results. Got {len(record)} warnings instead."
        # assert record[0].category is UserWarning, f"Expected a warning when 'use_cache' is set to False in generation results. Got {record[0].category} instead."
        # assert "use_cache" in str(record[0].message)

        # self._test_model_generation_results(results, batch_size)
        
        # generation_config.use_cache = True
        # t2 = time.time()
        # with torch.no_grad():
        #     results_cache = model.generate(
        #         input_ids=input_ids,
        #         generation_config=generation_config
        #     )
        # t_with_cache = time.time() - t2

        # self._test_model_generation_results(results_cache, batch_size)
        # assert t_with_cache < t_no_cache, f"Generation with cache is not faster than without cache. Got t with cache {t_with_cache} and t no cache {t_no_cache}"

    def _test_model_generation_results(self, results, batch_size):
        assert isinstance(results, list), "Generation results is not a list"
        assert batch_size == len(results), "Number of generated sequences does not match batch size"
        assert any(len(r) > 0 for r in results), "No generated sequences"

    @pytest.mark.fast
    def test_kv_cache_initialization(self):
        L, B, T, H, D = ( 
            self.config.model.n_layers,
            0,
            0,
            self.config.model.n_heads,
            self.config.model.d_head
        )
        kv_state = KVCache(
            config=self.config.model,
        )
        assert kv_state.shape == (L, 2, B, T, H, D), f"KV cache shape mismatch: {(L, 2, B, 1, H, D)}. Got {kv_state.shape}"

    @pytest.mark.fast
    def test_kv_cache_update(self):
        """
        Testing KV cache update functionality.

        Logic is that the KV cache should must handle sequential updates correctly (without restricting to seqlen or batch size -> dynamics).

        The cache shape is (n_layers, 2, batch_size, >=cur_pos, n_heads, d_head)

        We simulate prefill with a fake context -> check for kv_cache shape and value
        Then, we simulate decoding by adding one token at a time and checking the cache content.
            - we remove an element from the batch -> simulating beam search pruning
            - we increase the batch size -> simulating beam search expansion
            - we reset the cache -> simulating new sequence generation
            - we restart the generation from a given position -> simulating generation with past
        """
        L, B, T, H, D = ( 
            self.config.model.n_layers,
            8, # bs
            16, # sl (context)
            self.config.model.n_heads,
            self.config.model.d_head
        )
        kv_state = KVCache(config=self.config.model)
        from gpt_lib.utils.default import DEVICE
        kv_state.check_sizes(B, T, DEVICE, torch.float32)

        k = torch.randn(L, B, T, H, D)
        v = torch.randn(L, B, T, H, D)
        # prefill
        for layer_idx in range(L):
            kv_state.update(k[layer_idx], v[layer_idx], layer_idx)
        kv_state.advance()
        assert kv_state.shape == (L, 2, B, T, H, D), f"Key-Value cache shape mismatch: {(L, 2, B, T, H, D)}. Got: {kv_state.shape}"

        k_memory = k.clone()
        v_memory = v.clone()
        # decoding for 16 tokens
        dec_len = 16
        for i in range(dec_len):
            k = torch.randn(L, B, 1, H, D)
            v = torch.randn(L, B, 1, H, D)
            for layer_idx in range(L):
                kv_state.update(k[layer_idx], v[layer_idx], layer_idx)
            
            kv_state.advance()
            assert kv_state.cur_pos == T+i+1, f"KV cache cur pos expected {T+i+1} got {kv_state.cur_pos}."
            assert kv_state.shape[-3] == T+i+1, f"KV cache tensor is not expending while decoding. Got size ({kv_state.shape[-3]}) at position (-3) while expecting ({T+i+1})"
            # accumulate memory to check later
            k_memory = torch.cat([k_memory, k.clone()], dim=2)
            v_memory = torch.cat([v_memory, v.clone()], dim=2)

        # check if the cache content matches the accumulated memory
        assert kv_state.cur_pos == T+dec_len, f"Current position mismatch: {kv_state.cur_pos} != {T+dec_len}"
        
        for layer_idx in range(L):
            k_cache, v_cache = kv_state.layer(layer_idx)
            assert k_cache.shape == (B, T+dec_len, H, D), f"Key cache shape mismatch at layer {layer_idx}: {(B, T+dec_len, H, D)}. Got: {k_cache.shape}"
            assert v_cache.shape == (B, T+dec_len, H, D), f"Value cache shape mismatch at layer {layer_idx}: {(B, T+dec_len, H, D)}. Got: {v_cache.shape}"
            assert k_cache.shape == k_memory[layer_idx].shape, f"Key cache and Key memory do not match at layer {layer_idx}. Got {k_cache.shape} and {k_memory[layer_idx].shape}"
            assert (k_cache == k_memory[layer_idx]).all(), f"Got key cache mismatching k memory at layer {layer_idx}."
            assert (v_cache == v_memory[layer_idx]).all(), f"Got value cache mismatching v memory at layer {layer_idx}."

