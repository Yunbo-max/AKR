from __future__ import annotations

import numpy as np


def gradient_spectrum(
    gradients: np.ndarray, ranks: list[int], *, return_basis: bool = True
) -> dict:
    """SVD over flattened per-example KV gradients [examples, parameters]."""
    g = np.asarray(gradients, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] < 2:
        raise ValueError("gradients must be [examples, parameters] with >=2 examples")
    if return_basis:
        _, singular_values, vh = np.linalg.svd(g, full_matrices=False)
    else:
        # Geometry summaries need only singular values.  The row Gram matrix is
        # much smaller when examples << flattened KV width and avoids computing
        # thousands of right-singular-vector entries that are discarded.
        eigenvalues = np.linalg.eigvalsh(g @ g.T)
        singular_values = np.sqrt(np.clip(eigenvalues[::-1], 0, None))
        vh = None
    energy = singular_values**2
    total = energy.sum()
    explained = {str(r): float(energy[: min(r, len(energy))].sum() / total) for r in ranks}
    normalized = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    cosine = normalized @ normalized.T
    return {"singular_values": singular_values, "explained_energy": explained, "basis": vh, "cosine": cosine}
