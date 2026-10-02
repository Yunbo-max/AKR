import numpy as np

from animal_omni.geometry import gradient_spectrum
from animal_omni.metrics import classification_metrics, normalize_label


def test_strict_label_parser_and_metrics():
    labels = ["Infant Cry", "Phee", "Seep"]
    assert normalize_label("Phee.", labels) == "Phee"
    assert normalize_label("Phee or Seep", labels) is None
    m = classification_metrics(labels, ["Infant Cry", "Phee", None], labels)
    assert m["accuracy"] == 2 / 3


def test_parser_accepts_dataset_punctuation_variants():
    labels = ["Atlantic_Spotted_Dolphin", "Beluga,_White_Whale", "Killer_Whale", "False_Killer_Whale"]
    assert normalize_label("Atlantic spotted dolphin", labels) == "Atlantic_Spotted_Dolphin"
    assert normalize_label("Beluga, White Whale", labels) == "Beluga,_White_Whale"
    assert normalize_label("False Killer Whale", labels) == "False_Killer_Whale"
    assert normalize_label("Beluga, White Whale or Killer Whale", labels) is None


def test_low_rank_gradient_energy():
    rng = np.random.default_rng(2)
    direction = rng.normal(size=40)
    coefficients = rng.normal(size=(20, 1))
    g = coefficients * direction + 1e-6 * rng.normal(size=(20, 40))
    result = gradient_spectrum(g, [1, 2, 4])
    assert result["explained_energy"]["1"] > 0.999
    fast = gradient_spectrum(g, [1, 2, 4], return_basis=False)
    assert fast["basis"] is None
    assert np.allclose(result["singular_values"], fast["singular_values"])
