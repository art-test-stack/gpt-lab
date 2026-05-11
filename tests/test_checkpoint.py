import pytest
import torch
import numpy as np
import random
from gpt_lab.model.checkpoint import capture_rng_state, set_rng_state

def numpy_rng_equal(s1, s2):
    if s1[0] != s2[0]:  # algorithm name
        return False
    if not np.array_equal(s1[1], s2[1]):  # state array
        return False
    return s1[2:] == s2[2:] 

def assert_rng_state_consistency(rng_state1, rng_state2, raise_msg="RNG states do not match."):
    assert torch.equal(rng_state1["torch"], rng_state2["torch"]), f"{raise_msg} Torch RNG states do not match."
    assert numpy_rng_equal(rng_state1["numpy"], rng_state2["numpy"]), f"{raise_msg} Numpy RNG states do not match."
    assert rng_state1["python"] == rng_state2["python"], f"{raise_msg} Python random RNG states do not match."
    if torch.cuda.is_available():
        for s1, s2 in zip(rng_state1["cuda"], rng_state2["cuda"]):
            assert torch.equal(s1, s2), f"{raise_msg} CUDA RNG states do not match."

@pytest.mark.fast
def test_checkpoint_rng_state_capture():
    # Capture the RNG state before saving the checkpoint
    rng_state_before = capture_rng_state()

    # Simulate saving and loading a checkpoint (you can replace this with actual checkpoint code)
    # For this test, we'll just set the RNG state back to the captured state
    set_rng_state(rng_state_before)

    # Capture the RNG state after loading the checkpoint
    rng_state_after = capture_rng_state()

    # Assert that the RNG states are the same
    assert_rng_state_consistency(rng_state_before, rng_state_after, "RNG state was not preserved across checkpoint save/load.")

@pytest.mark.fast
def test_checkpoint_rng_state_with_seed():
    random_seed = [42, 1234, 9999]
    for seed in random_seed:
        # Set the RNG state to a specific seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Capture the RNG state after setting it
        rng_state_set = capture_rng_state()

        # Set the RNG state again to the same seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Capture the RNG state again
        rng_state_set_again = capture_rng_state()

        # Assert that the RNG states are the same
        assert_rng_state_consistency(rng_state_set, rng_state_set_again, f"RNG state was not consistent for seed {seed}.")

@pytest.mark.fast
def test_checkpoint_rng_state_with_random_values():
    # Set a random seed and capture the RNG state
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng_state_before = capture_rng_state()

    # Generate some random values
    torch.rand(5)
    np.random.rand(5)
    random.random()

    # Capture the RNG state after generating random values
    rng_state_after = capture_rng_state()

    # Assert that the RNG states are different after generating random values
    with pytest.raises(AssertionError):
        assert_rng_state_consistency(rng_state_before, rng_state_after, "RNG state should change after generating random values.")

    # Now set the RNG state back to the captured state
    set_rng_state(rng_state_before)

    # Capture the RNG state again
    rng_state_restored = capture_rng_state()

    # Assert that the restored RNG state matches the original captured state
    assert_rng_state_consistency(rng_state_before, rng_state_restored, "RNG state was not correctly restored after setting it back.")

# TODO: Add tests for checkpoint saving and loading that also every states