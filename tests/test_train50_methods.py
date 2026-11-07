import json

import numpy as np
import pytest

from akr_final.train50_methods import fit_correction


@pytest.fixture
def support():
    x = np.array([[-2., 0.], [-1., 1.], [0., -1.], [2., 0.]])
    y = np.array([0, 0, 0, 1])
    g = np.array([[1., 0., 2.], [3., 0., 2.], [2., 3., 2.], [10., 4., 6.]])
    return x, g, y


def test_support_prior_centering_and_uniform_distinction(support, monkeypatch):
    x, g, y = support
    model = fit_correction(x, g, y, 2, {"kind": "soft"})
    np.testing.assert_allclose(model.priors @ model.prototypes, 0., atol=1e-14)
    assert np.linalg.norm(model.prototypes.mean(0)) > 1
    balanced = fit_correction(x, g, [0, 0, 1, 1], 2, {"kind": "soft"})
    np.testing.assert_allclose(balanced.prototypes.mean(0), 0., atol=1e-14)
    np.testing.assert_allclose(model.gradient_mean, g.mean(0))
    assert model.metadata["class_counts"] == [3, 1]
    assert model.metadata["class_priors"] == [.75, .25]
    monkeypatch.setattr(model.readout.mapping, "predict", lambda q: np.tile(np.log(model.priors), (len(q), 1)))
    np.testing.assert_allclose(model.predict(x), 0., atol=1e-14)


def test_soft_hard_and_uncentered_predictions(support):
    x, g, y = support
    soft = fit_correction(x, g, y, 2, {"kind": "soft", "temperature": .2})
    hard = fit_correction(x, g, y, 2, {"kind": "hard", "temperature": .2})
    probs = soft.probabilities(x)
    np.testing.assert_allclose(probs.sum(1), 1.)
    np.testing.assert_allclose(soft.predict(x), probs @ soft.prototypes)
    np.testing.assert_allclose(hard.predict(x), hard.prototypes[probs.argmax(1)])
    raw = fit_correction(x, g, y, 2, {"kind": "soft", "centered": False, "temperature": .2})
    np.testing.assert_allclose(raw.predict(x), soft.predict(x) + g.mean(0))
    assert soft.predict(x[0]).shape == (1, 3)
    assert soft.predict(np.empty((0, 2))).shape == (0, 3)


def test_stable_softmax(support):
    x, g, y = support
    model = fit_correction(x, g, y, 2, {"kind": "soft", "temperature": 1e-12})
    probs = model.probabilities(x)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(1), 1.)


def test_original_delegates_and_shuffled_pair_deranges(support, monkeypatch):
    from akr_final import predictability
    calls = []
    class Stub:
        rank = 2
        def predict(self, q):
            return np.repeat(np.array([[4., 5., 6.]]), len(q), axis=0)
    def fit(x, g, rank, alpha):
        calls.append((x.copy(), g.copy(), rank, alpha))
        return Stub()
    monkeypatch.setattr(predictability, "fit_repository", fit)
    x, g, y = support
    original = fit_correction(x, g, y, 2, {"kind": "original", "rank": 3, "ridge": 7})
    np.testing.assert_array_equal(calls[0][0], x)
    np.testing.assert_array_equal(calls[0][1], g)
    assert calls[0][2:] == (3, 7.)
    np.testing.assert_array_equal(original.predict(x), [[4., 5., 6.]] * len(x))
    a = fit_correction(x, g, y, 2, {"kind": "shuffled_pair"}, seed=8)
    b = fit_correction(x, g, y, 2, {"kind": "shuffled_pair"}, seed=8)
    perm = np.array(a.metadata["support_permutation"])
    assert np.all(perm != np.arange(len(x)))
    np.testing.assert_array_equal(calls[1][0], x[perm])
    np.testing.assert_array_equal(calls[1][1], g)
    assert a.metadata == b.metadata


def test_exact_repository_original_parity(support):
    pytest.importorskip("sklearn")
    from akr_final.predictability import fit_repository
    x, g, y = support
    model = fit_correction(x, g, y, 2, {"kind": "original", "rank": 20, "ridge": 3})
    reference = fit_repository(x, g, rank=20, alpha=3)
    np.testing.assert_array_equal(model.predict(x), reference.predict(x))
    assert model.metadata["actual_rank"] == reference.rank


def test_fixed_random_and_shuffled_prototypes(support):
    x, g, y = support
    fixed = fit_correction(x, g, y, 2, {"kind": "fixed"})
    np.testing.assert_allclose(fixed.predict(x), np.repeat(g.mean(0)[None], len(x), axis=0))
    soft = fit_correction(x, g, y, 2, {"kind": "soft"})
    random = fit_correction(x, g, y, 2, {"kind": "random"}, seed=3)
    np.testing.assert_allclose(np.linalg.norm(random.predict(x), axis=1), np.linalg.norm(soft.predict(x), axis=1))
    np.testing.assert_allclose(random.predict(x), np.vstack([random.predict(q) for q in x]))
    np.testing.assert_array_equal(random.predict(x), fit_correction(x, g, y, 2, {"kind": "random"}, seed=3).predict(x))
    assert not np.allclose(random.predict(x), fit_correction(x, g, y, 2, {"kind": "random"}, seed=4).predict(x))
    shuffled = fit_correction(x, g, y, 2, {"kind": "shuffled_prototypes"}, seed=6)
    np.testing.assert_allclose(shuffled.predict(x), soft.probabilities(x) @ soft.prototypes[[1, 0]])
    for model in (fixed, random, shuffled):
        json.dumps(model.metadata, allow_nan=False)
        assert model.metadata["rank_relevant"] is False


@pytest.mark.parametrize("spec", [
    {"kind": "unknown"}, {"kind": "soft", "ridge": 0},
    {"kind": "soft", "ridge": float("nan")}, {"kind": "soft", "temperature": 0},
    {"kind": "soft", "temperature": float("inf")}, {"kind": "original", "rank": -1},
    {"kind": "original", "rank": 1.5}, {"kind": "soft", "centered": "false"},
])
def test_invalid_specs(support, spec):
    with pytest.raises(ValueError):
        fit_correction(*support, 2, spec)


@pytest.mark.parametrize("labels,nclasses", [([0, 0, 0, 0], 2), ([0, 0, 0, 2], 2),
    ([0., .1, 0., 1.], 2), ([[0], [0], [0], [1]], 2), ([0, 0, 1], 2),
    ([0, 0, 0, 1], 2.5)])
def test_invalid_labels(support, labels, nclasses):
    x, g, _ = support
    with pytest.raises(ValueError):
        fit_correction(x, g, labels, nclasses, {"kind": "soft"})


def test_invalid_arrays_and_queries(support):
    x, g, y = support
    for bad_x, bad_g in ((x[:, :0], g), (x, g[:, :0]), (x[:-1], g),
                         (np.full_like(x, np.nan), g), (x, np.full_like(g, np.inf))):
        with pytest.raises(ValueError):
            fit_correction(bad_x, bad_g, y, 2, {"kind": "soft"})
    model = fit_correction(x, g, y, 2, {"kind": "soft"})
    for q in ([1., 2., 3.], [[1., np.nan]], np.zeros((1, 1, 2))):
        with pytest.raises(ValueError):
            model.predict(q)


def test_single_class_and_zero_random_norm():
    x = np.array([[0.], [1.]])
    g = np.ones((2, 3))
    model = fit_correction(x, g, [0, 0], 1, {"kind": "random"})
    np.testing.assert_array_equal(model.predict(x), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="two classes"):
        fit_correction(x, g, [0, 0], 1, {"kind": "shuffled_prototypes"})


def test_original_rank_zero_and_metadata(support):
    pytest.importorskip("sklearn")
    x, g, y = support
    model = fit_correction(x, g, y, 2, {"kind": "original", "rank": 0})
    np.testing.assert_allclose(model.predict(x), np.repeat(g.mean(0)[None, :], len(x), axis=0))
    assert model.metadata["actual_rank"] == 0
    json.dumps(model.metadata, allow_nan=False)
    assert model.predict(np.empty((0, 2))).shape == (0, 3)
    with pytest.raises(ValueError, match="no class probability"):
        model.probabilities(x)


@pytest.mark.parametrize("seed", [-1, 1.2, True])
def test_invalid_seed(support, seed):
    with pytest.raises(ValueError, match="seed"):
        fit_correction(*support, 2, {"kind": "soft"}, seed=seed)
