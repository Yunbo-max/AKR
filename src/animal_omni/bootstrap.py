from __future__ import annotations

import numpy as np


def paired_accuracy_delta_ci(
    targets: list[str],
    predictions_a: list[str | None],
    predictions_b: list[str | None],
    *,
    samples: int = 10_000,
    seed: int = 20250813,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired event bootstrap for accuracy(A)-accuracy(B)."""
    n = len(targets)
    if not n or len(predictions_a) != n or len(predictions_b) != n:
        raise ValueError("all inputs must have the same non-zero length")
    a = np.fromiter((y == p for y, p in zip(targets, predictions_a)), dtype=float)
    b = np.fromiter((y == p for y, p in zip(targets, predictions_b)), dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, n, size=(size, n))
        deltas[start : start + size] = (a[indices] - b[indices]).mean(axis=1)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(a.mean() - b.mean()), "ci_low": float(lo), "ci_high": float(hi)}

