import pytest

from gpt_lab.utils.schemas import OptimizerConfig


def test_explicit_muon_profile_preserves_the_bundled_default():
    explicit = OptimizerConfig.from_file("experiments/optimizer_comparison/configs/muon.yaml")
    assert explicit.group_configs() == OptimizerConfig.from_file().group_configs()


def test_anvil_spectral_profiles_are_matched():
    baseline = OptimizerConfig.from_file("configs/normuon.yaml").group_configs()
    for variant, spectral in (("a_normuon", False), ("b_spectral", True)):
        groups = OptimizerConfig.from_file(f"experiments/optimizer_comparison/configs/{variant}.yaml").group_configs()
        assert groups["transformer"].pop("anvil_spectral") is spectral
        assert groups == baseline


def test_anvil_spectral_requires_boolean(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text('transformer:\n  opt: normuon\n  anvil_spectral: "false"\n')
    with pytest.raises(ValueError, match="anvil_spectral must be a boolean"):
        OptimizerConfig.from_file(path)
