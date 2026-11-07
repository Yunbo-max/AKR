"""Support-only correction maps for the Train-50 cases.

Class routes use NumPy and the existing ``RidgeReadout``. ``original`` and
``shuffled_pair`` intentionally call ``predictability.fit_repository`` and thus
retain the production router's sklearn dependency, precision, SVD and rank rules.
No correction map accepts query labels or fits anything at query time.

These maps return flat gradient vectors, before K/V injection. The random control
matches each query's *global flat-vector L2 norm* to its soft centered correction.
It does not match per-layer or per-K/V-block norms; injection normalization/caps
must be applied separately and identically to the methods being compared.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np

from akr_closing.repair import RidgeReadout
from akr_final.train50_evidence import class_probabilities


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain finite real numbers")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite real numbers") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


@dataclass
class Correction:
    """A fitted support-only map; 1-D queries become a one-row result.

    ``probabilities`` is available for hard/soft/random/shuffled-prototype routes
    and always returns the underlying ridge-softmax probabilities. Hard routing
    takes their argmax in ``predict``. Empty batches retain their 2-D shape.
    ``prototypes`` contains already centered (or explicitly raw) class targets.
    """
    kind: str
    n_features: int
    gradient_mean: np.ndarray
    metadata: dict
    readout: RidgeReadout | None = None
    prototypes: np.ndarray | None = None
    priors: np.ndarray | None = None
    temperature: float = 1.
    repository: object | None = None
    random_direction: np.ndarray | None = None

    def _queries(self, x):
        x = _array(x, "query features")
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"query features must have shape [n, {self.n_features}]")
        return x

    def probabilities(self, x):
        x = self._queries(x)
        if self.readout is None:
            raise ValueError(f"{self.kind} has no class probability readout")
        if not len(x):
            return np.empty((0, len(self.priors)), dtype=np.float64)
        scores = self.readout.mapping.predict(x)
        return class_probabilities(scores, self.temperature)

    def predict(self, x):
        x = self._queries(x)
        if not len(x):
            return np.empty((0, len(self.gradient_mean)), dtype=np.float64)
        if self.repository is not None:
            result = self.repository.predict(x)
        elif self.kind == "fixed":
            result = np.repeat(self.gradient_mean[None, :], len(x), axis=0)
        else:
            weights = self.probabilities(x)
            if self.kind == "hard":
                result = self.prototypes[weights.argmax(axis=1)]
            else:
                result = weights @ self.prototypes
                if self.kind == "random":
                    result = np.linalg.norm(result, axis=1, keepdims=True) * self.random_direction
        result = np.asarray(result)
        if result.shape != (len(x), len(self.gradient_mean)) or not np.isfinite(result).all():
            raise ValueError("correction prediction has invalid shape or nonfinite values")
        return result


def fit_correction(x, gradients, labels, n_classes, spec, seed=0):
    """Fit a correction from aligned support arrays and integer class IDs.

    ``spec`` is a mapping with ``kind`` and optional ``ridge=10``, ``rank=4``,
    ``temperature=1`` and ``centered=True``. Every class in ``range(n_classes)``
    must occur. Rank is relevant only to original/shuffled-pair continuous maps;
    the repository retains its dimensional clipping, including null singular
    directions. The fixed control always returns the raw support-gradient mean.
    Random always uses the centered soft norm even if ``centered=False`` was
    requested; this effective setting is recorded. Prototype shuffling permutes
    targets *after* centering, so unequal-prior cancellation need not survive it.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be a mapping")
    kind = spec.get("kind")
    kinds = {"original", "hard", "soft", "fixed", "random", "shuffled_pair", "shuffled_prototypes"}
    if not isinstance(kind, str) or kind not in kinds:
        raise ValueError(f"kind must be one of {sorted(kinds)}")
    n_classes = _integer(n_classes, "n_classes", 1)
    seed = _integer(seed, "seed", 0)
    rank = _integer(spec.get("rank", 4), "rank", 0)
    ridge = _positive(spec.get("ridge", 10.), "ridge")
    temperature = _positive(spec.get("temperature", 1.), "temperature")
    centered = spec.get("centered", True)
    if not isinstance(centered, (bool, np.bool_)):
        raise ValueError("centered must be a boolean")
    x = _array(x, "support features")
    g = _array(gradients, "support gradients")
    y = _array(labels, "support labels")
    if (x.ndim != 2 or g.ndim != 2 or not len(x) or len(x) != len(g)
            or not x.shape[1] or not g.shape[1]):
        raise ValueError("support features and gradients must be nonempty aligned 2-D arrays")
    if (y.ndim != 1 or len(y) != len(x) or not np.equal(y, np.floor(y)).all()
            or (y < 0).any() or (y >= n_classes).any()):
        raise ValueError("support labels must be aligned integer class IDs in range(n_classes)")
    y = y.astype(np.int64)
    counts = np.bincount(y, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError("every class must be in support")
    priors = counts / len(x)
    mean = g.mean(axis=0)
    if not np.isfinite(mean).all():
        raise ValueError("support gradient mean is nonfinite")
    continuous = kind in {"original", "shuffled_pair"}
    metadata = {
        "kind": kind, "type": "continuous_centered_svd_ridge" if continuous else kind,
        "n_support": int(len(x)), "n_features": int(x.shape[1]),
        "gradient_dimension": int(g.shape[1]), "n_classes": n_classes,
        "class_counts": counts.tolist(), "class_priors": priors.tolist(),
        "ridge": ridge, "requested_rank": rank, "rank_relevant": continuous,
        "temperature": temperature, "requested_centered": bool(centered),
        "seed": seed, "query_labels_used": False,
    }
    model = Correction(kind, x.shape[1], mean, metadata, priors=priors, temperature=temperature)
    if continuous:
        from akr_final import predictability
        train_x = x
        if kind == "shuffled_pair":
            permutation = predictability.derangement(len(x), seed)
            train_x = x[permutation]
            metadata["support_permutation"] = permutation.tolist()
        model.repository = predictability.fit_repository(train_x, g, rank=rank, alpha=ridge)
        metadata.update(actual_rank=int(model.repository.rank), centered=True,
                        centering="repository_mean_plus_centered_regression",
                        implementation="akr_final.predictability.fit_repository",
                        seeding="deterministic support derangement" if kind == "shuffled_pair" else "unused")
        return model
    if kind == "fixed":
        metadata.update(centered=False, centering="raw_support_mean", seeding="unused")
        return model
    model.readout = RidgeReadout.fit(x, y, n_classes, alpha=ridge)
    raw_prototypes = np.stack([g[y == c].mean(axis=0) for c in range(n_classes)])
    effective_centered = bool(centered) or kind == "random"
    model.prototypes = raw_prototypes - mean if effective_centered else raw_prototypes
    metadata.update(centered=effective_centered,
                    centering="support_prior_weighted_class_mean" if effective_centered else "none",
                    readout="akr_closing.repair.RidgeReadout", seeding="unused")
    if kind == "shuffled_prototypes":
        if n_classes < 2:
            raise ValueError("prototype shuffling requires at least two classes")
        from akr_final.predictability import derangement
        permutation = derangement(n_classes, seed)
        model.prototypes = model.prototypes[permutation]
        metadata.update(prototype_permutation=permutation.tolist(),
                        seeding="deterministic class derangement",
                        shuffle_order="after_centering")
    if kind == "random":
        direction = np.random.default_rng(seed).standard_normal(g.shape[1])
        model.random_direction = direction / np.linalg.norm(direction)
        metadata.update(seeding="fixed seeded Gaussian direction; query/order/batch independent",
                        norm_match="per_query_global_flat_L2_of_soft_centered_prediction",
                        per_block_norm_matched=False)
    return model
