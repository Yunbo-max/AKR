from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def flatten_gradient(gradient: dict) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Flatten a pooled layer/KV gradient using a stable key order."""
    keys = sorted(gradient)
    vector = np.concatenate([gradient[key].float().cpu().numpy() for key in keys])
    return vector.astype(np.float32, copy=False), keys


def unflatten_gradient(vector: np.ndarray, keys: list[tuple[int, str]], width: int):
    """Restore a flat vector to the pooled-gradient dictionary format."""
    import torch

    expected = len(keys) * width
    if vector.size != expected:
        raise ValueError(f"expected {expected} values, got {vector.size}")
    return {
        key: torch.from_numpy(vector[i * width:(i + 1) * width].copy())
        for i, key in enumerate(keys)
    }


def flatten_token_gradient(gradient: dict) -> tuple[np.ndarray, list[tuple[int, str]], int, int]:
    """Flatten a same-length tokenwise layer/KV gradient field."""
    keys = sorted(gradient)
    shapes = {tuple(gradient[key].shape) for key in keys}
    if len(shapes) != 1:
        raise ValueError(f"tokenwise gradient shapes differ: {shapes}")
    token_count, width = next(iter(shapes))
    vector = np.concatenate([
        gradient[key].float().cpu().numpy().reshape(-1) for key in keys
    ])
    return vector.astype(np.float32, copy=False), keys, token_count, width


def unflatten_token_gradient(
    vector: np.ndarray, keys: list[tuple[int, str]], token_count: int, width: int
):
    """Restore a flat vector to ordered [audio_token, projection] fields."""
    import torch

    stride = token_count * width
    expected = len(keys) * stride
    if vector.size != expected:
        raise ValueError(f"expected {expected} values, got {vector.size}")
    return {
        key: torch.from_numpy(
            vector[i * stride:(i + 1) * stride].reshape(token_count, width).copy()
        )
        for i, key in enumerate(keys)
    }


@dataclass
class ConditionalGradientRouter:
    """Low-rank linear router fit only from labeled support examples.

    Gradient PCA supplies U, while ridge regression maps a frozen query
    representation phi(h) to centered gradient coefficients.  The support mean
    is retained as the fixed/mean-KV baseline.
    """

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    gradient_mean: np.ndarray
    basis: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    rank: int
    alpha: float

    @classmethod
    def fit(cls, features: np.ndarray, gradients: np.ndarray, *, rank: int, alpha: float):
        from sklearn.linear_model import Ridge

        if features.ndim != 2 or gradients.ndim != 2 or len(features) != len(gradients):
            raise ValueError("features and gradients must be aligned 2-D arrays")
        feature_mean = features.mean(0)
        feature_scale = features.std(0)
        feature_scale[feature_scale < 1e-6] = 1.0
        z = (features - feature_mean) / feature_scale
        gradient_mean = gradients.mean(0)
        centered = gradients - gradient_mean
        # SVD over support rows is inexpensive even though the flattened KV
        # target is wide. Rank cannot exceed the centered support rank.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        effective_rank = max(0, min(rank, len(features) - 1, len(vt)))
        basis = vt[:effective_rank].astype(np.float32)
        if effective_rank:
            targets = centered @ basis.T
            model = Ridge(alpha=alpha).fit(z, targets)
            coefficients = np.asarray(model.coef_, dtype=np.float32)
            if coefficients.ndim == 1:
                coefficients = coefficients[None, :]
            intercept = np.atleast_1d(model.intercept_).astype(np.float32)
        else:
            coefficients = np.empty((0, features.shape[1]), dtype=np.float32)
            intercept = np.empty(0, dtype=np.float32)
        return cls(feature_mean, feature_scale, gradient_mean.astype(np.float32),
                   basis, coefficients, intercept, effective_rank, alpha)

    def predict(self, feature: np.ndarray) -> np.ndarray:
        if self.rank == 0:
            return self.gradient_mean.copy()
        z = (feature - self.feature_mean) / self.feature_scale
        scores = self.coefficients @ z + self.intercept
        return self.gradient_mean + scores @ self.basis

    def fixed_mean(self) -> np.ndarray:
        return self.gradient_mean.copy()


@dataclass
class ConditionalTokenGradientRouter(ConditionalGradientRouter):
    """Memory-efficient router for very wide token-preserving gradient fields."""

    @classmethod
    def fit(cls, features: np.ndarray, gradients: np.ndarray, *, rank: int, alpha: float):
        from sklearn.linear_model import Ridge

        if features.ndim != 2 or gradients.ndim != 2 or len(features) != len(gradients):
            raise ValueError("features and gradients must be aligned 2-D arrays")
        feature_mean = features.mean(0)
        feature_scale = features.std(0)
        feature_scale[feature_scale < 1e-6] = 1.0
        z = (features - feature_mean) / feature_scale
        gradient_mean = gradients.mean(0)
        centered = gradients - gradient_mean
        # For tokenwise fields p can exceed one million while support n is only
        # tens. Eigendecompose the n x n Gram matrix and recover the requested
        # right singular vectors without constructing a full SVD workspace.
        gram = centered @ centered.T
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        tolerance = max(float(values[0]) if len(values) else 0.0, 1.0) * 1e-10
        available = int(np.sum(values > tolerance))
        effective_rank = max(0, min(rank, len(features) - 1, available))
        if effective_rank:
            singular = np.sqrt(np.maximum(values[:effective_rank], 0.0))
            basis = ((vectors[:, :effective_rank].T @ centered) /
                     singular[:, None]).astype(np.float32)
            targets = centered @ basis.T
            model = Ridge(alpha=alpha).fit(z, targets)
            coefficients = np.asarray(model.coef_, dtype=np.float32)
            if coefficients.ndim == 1:
                coefficients = coefficients[None, :]
            intercept = np.atleast_1d(model.intercept_).astype(np.float32)
        else:
            basis = np.empty((0, gradients.shape[1]), dtype=np.float32)
            coefficients = np.empty((0, features.shape[1]), dtype=np.float32)
            intercept = np.empty(0, dtype=np.float32)
        return cls(feature_mean, feature_scale, gradient_mean.astype(np.float32),
                   basis, coefficients, intercept, effective_rank, alpha)


@dataclass
class ConditionalLocalTokenRouter:
    """Predict time-aligned KV gradients from local and global query states.

    Absolute token-field transfer assumes that support and query events are
    temporally aligned. This router instead learns a shared local mapping: each
    query token receives coefficients conditioned on its own hidden state and on
    a low-dimensional global event representation.
    """

    local_mean: np.ndarray
    local_components: np.ndarray
    global_mean: np.ndarray
    global_components: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    gradient_means: dict
    gradient_bases: dict
    coefficients: np.ndarray
    intercept: np.ndarray
    keys: list
    gradient_rank: int
    alpha: float

    @staticmethod
    def _components(matrix: np.ndarray, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.utils.extmath import randomized_svd

        mean = matrix.mean(0).astype(np.float32)
        centered = matrix.astype(np.float32, copy=False) - mean
        effective = min(rank, centered.shape[0] - 1, centered.shape[1])
        if effective <= 0:
            return mean, np.empty((0, centered.shape[1]), dtype=np.float32)
        _, _, vt = randomized_svd(
            centered, n_components=effective, n_iter=3, random_state=seed
        )
        return mean, vt.astype(np.float32)

    @classmethod
    def fit(
        cls, token_features: np.ndarray, global_features: np.ndarray,
        gradients: list[dict], *, local_rank: int = 64, global_rank: int = 16,
        gradient_rank: int = 4, alpha: float = 10.0, seed: int = 20250813,
    ):
        from sklearn.linear_model import Ridge
        from sklearn.utils.extmath import randomized_svd

        if token_features.ndim != 3 or global_features.ndim != 2:
            raise ValueError("token_features must be [events,tokens,width] and globals [events,width]")
        events, tokens, width = token_features.shape
        if global_features.shape != (events, width) or len(gradients) != events:
            raise ValueError("support features and gradients must be event-aligned")
        keys = sorted(gradients[0])
        expected = {(tokens, next(iter(gradients[0].values())).shape[1])}
        for gradient in gradients:
            if sorted(gradient) != keys or {tuple(gradient[key].shape) for key in keys} != expected:
                raise ValueError("all local-token gradient fields must share keys and shapes")

        local = token_features.reshape(events * tokens, width).astype(np.float32)
        local_mean, local_components = cls._components(local, local_rank, seed)
        local_scores = (local - local_mean) @ local_components.T
        global_mean, global_components = cls._components(
            global_features.astype(np.float32), global_rank, seed + 1
        )
        global_scores = (global_features.astype(np.float32) - global_mean) @ global_components.T
        global_scores = np.repeat(global_scores, tokens, axis=0)
        features = np.concatenate([local_scores, global_scores], axis=1).astype(np.float32)
        feature_mean = features.mean(0)
        feature_scale = features.std(0)
        feature_scale[feature_scale < 1e-6] = 1.0
        z = (features - feature_mean) / feature_scale

        gradient_means, gradient_bases, targets = {}, {}, []
        actual_rank = None
        for index, key in enumerate(keys):
            field = np.concatenate([
                gradient[key].float().cpu().numpy() for gradient in gradients
            ], axis=0).astype(np.float32)
            mean = field.mean(0).astype(np.float32)
            centered = field - mean
            effective = min(gradient_rank, centered.shape[0] - 1, centered.shape[1])
            u, singular, vt = randomized_svd(
                centered, n_components=effective, n_iter=3,
                random_state=seed + 2 + index,
            )
            gradient_means[key] = mean
            gradient_bases[key] = vt.astype(np.float32)
            targets.append((u * singular).astype(np.float32))
            actual_rank = effective if actual_rank is None else min(actual_rank, effective)
        target = np.concatenate(targets, axis=1)
        model = Ridge(alpha=alpha).fit(z, target)
        coefficients = np.asarray(model.coef_, dtype=np.float32)
        if coefficients.ndim == 1:
            coefficients = coefficients[None, :]
        intercept = np.atleast_1d(model.intercept_).astype(np.float32)
        return cls(local_mean, local_components, global_mean, global_components,
                   feature_mean, feature_scale, gradient_means, gradient_bases,
                   coefficients, intercept, keys, int(actual_rank or 0), alpha)

    def predict(self, token_features: np.ndarray, global_feature: np.ndarray) -> dict:
        local = (token_features.astype(np.float32) - self.local_mean) @ self.local_components.T
        global_score = ((global_feature.astype(np.float32) - self.global_mean) @
                        self.global_components.T)
        repeated = np.repeat(global_score[None, :], len(local), axis=0)
        features = np.concatenate([local, repeated], axis=1)
        z = (features - self.feature_mean) / self.feature_scale
        scores = z @ self.coefficients.T + self.intercept
        fields, offset = {}, 0
        for key in self.keys:
            basis = self.gradient_bases[key]
            rank = len(basis)
            fields[key] = self.gradient_means[key] + scores[:, offset:offset + rank] @ basis
            offset += rank
        return fields

    def fixed_mean(self, token_count: int) -> dict:
        return {
            key: np.repeat(mean[None, :], token_count, axis=0)
            for key, mean in self.gradient_means.items()
        }


def broadcast_audio_delta(pooled: dict, audio_mask, eta: float) -> dict:
    """Broadcast a fixed-width pooled direction only over prefill audio tokens."""
    import torch

    mask = audio_mask.detach().cpu().bool()
    if mask.ndim != 2 or mask.shape[0] != 1:
        raise ValueError("audio_mask must have shape [1, sequence]")
    deltas = {}
    for key, vector in pooled.items():
        delta = torch.zeros((*mask.shape, vector.numel()), dtype=torch.float32)
        delta[mask] = vector.float().mul(eta)
        deltas[key] = delta
    return deltas


def place_audio_token_delta(tokenwise: dict, audio_mask, eta: float) -> dict:
    """Place an ordered predicted gradient field at prefill audio positions."""
    import torch

    mask = audio_mask.detach().cpu().bool()
    if mask.ndim != 2 or mask.shape[0] != 1:
        raise ValueError("audio_mask must have shape [1, sequence]")
    token_count = int(mask.sum())
    deltas = {}
    for key, field in tokenwise.items():
        if field.ndim != 2 or field.shape[0] != token_count:
            raise ValueError(
                f"token count mismatch for {key}: {tuple(field.shape)} vs {token_count}"
            )
        delta = torch.zeros((*mask.shape, field.shape[1]), dtype=torch.float32)
        delta[mask] = field.float().mul(eta)
        deltas[key] = delta
    return deltas
