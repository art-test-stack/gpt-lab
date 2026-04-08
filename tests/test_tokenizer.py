import pytest
import random
import string

from gpt_lib.tokenizer import Tokenizer, TokenizerConfig
from gpt_lib.utils.special_tokens import SpecialTokens

def make_dummy_dataset(size: int, max_seq_len: int):
    """Creates a dummy dataset of random token sequences for testing purposes.
    
    Args:
        size (int): The number of samples in the dataset (in MB).
        max_seq_len (int): The maximum sequence length for each sample.
    Returns:
        Iterator[str]: An Iterator of random strings representing token sequences.
    """
    num_samples = size * 1024 * 1024 // max_seq_len  # Approximate number of samples based on size and sequence length

    for _ in range(num_samples):
        seq_len = random.randint(1, max_seq_len)
        sample = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=seq_len))
        yield sample


@pytest.fixture(scope="module")
def dummy_small():
    return make_dummy_dataset(size=10, max_seq_len=16)

@pytest.fixture(scope="module")
def dummy_large():
    return make_dummy_dataset(size=100, max_seq_len=16)

@pytest.mark.fast
def test_tokenizer(dummy_small, dummy_large):
    special_tokens = SpecialTokens()
    config = TokenizerConfig(name="gpt2", vocab_size=-1, max_context=16, pat_str="gpt2", source="tiktoken", special_tokens=special_tokens)
    tokenizer = Tokenizer.from_pretrained(config=config)
    
    # Test small dataset
    for sample in dummy_small:
        tokens = tokenizer.encode(sample)
        assert len(tokens) <= 16, "Tokenized sequence exceeds max context length"
        decoded = tokenizer.decode(tokens)
        assert decoded == sample, "Decoded text does not match original sample"
    
    # Test large dataset
    for sample in dummy_large:
        tokens = tokenizer.encode(sample)
        assert len(tokens) <= 16, "Tokenized sequence exceeds max context length"
        decoded = tokenizer.decode(tokens)
        assert decoded == sample, "Decoded text does not match original sample"

@pytest.mark.fast
def test_train_tokenizer(dummy_small):
    special_tokens = SpecialTokens()
    corpus = list(dummy_small)  # Convert generator to list for multiple iterations
    config = TokenizerConfig(name="gpt2", vocab_size=-1, max_context=16, pat_str="gpt2", source="tiktoken", special_tokens=special_tokens)
    tokenizer = Tokenizer.from_pretrained(config=config)
    
    # Simulate training by encoding and decoding the same sample multiple times
    for sample in dummy_small:
        tokens = tokenizer.encode(sample)
        decoded = tokenizer.decode(tokens)
        assert decoded == sample, "Decoded text does not match original sample after training simulation"
    import warnings
    warnings.warn("This is dummy training test. TODO: Implement actual training logic and test it properly.")