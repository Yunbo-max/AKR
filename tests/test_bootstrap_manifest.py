from animal_omni.bootstrap import paired_accuracy_delta_ci
from animal_omni.manifest import deterministic_split


def test_paired_bootstrap_detects_uniform_improvement():
    y = ["a"] * 30
    result = paired_accuracy_delta_ci(y, ["a"] * 30, ["b"] * 30, samples=500, seed=1)
    assert result == {"delta": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_group_split_is_deterministic():
    a = deterministic_split("recording-7", 42, 0.7, 0.15)
    b = deterministic_split("recording-7", 42, 0.7, 0.15)
    assert a == b

