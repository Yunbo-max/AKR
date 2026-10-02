#!/usr/bin/env python3
"""Layerwise SVD and cosine summaries for saved pooled Oracle gradients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from animal_omni.geometry import gradient_spectrum


def effective_rank(matrix: np.ndarray) -> float:
    """Entropy effective rank of a centered sample-by-feature matrix."""
    if not len(matrix) or not np.any(matrix):
        return 0.0
    # Samples are far fewer than gradient dimensions.  The row Gram spectrum
    # is identical and avoids repeatedly constructing wide SVD workspaces.
    energy = np.clip(np.linalg.eigvalsh(matrix @ matrix.T), 0, None)
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    probability = energy[energy > 0] / total
    return float(np.exp(-(probability * np.log(probability)).sum()))


def class_geometry(matrix: np.ndarray, labels: np.ndarray) -> dict:
    global_centered = matrix - matrix.mean(0, keepdims=True)
    within = np.empty_like(matrix)
    between_trace = 0.0
    class_effective_ranks = {}
    grand = matrix.mean(0)
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        centered = matrix[mask] - matrix[mask].mean(0, keepdims=True)
        within[mask] = centered
        between_trace += float(mask.sum()) * float(np.square(matrix[mask].mean(0) - grand).sum())
        class_effective_ranks[label] = effective_rank(centered)
    within_trace = float(np.square(within).sum())
    return {
        "global_effective_rank_centered": effective_rank(global_centered),
        "within_effective_rank_pooled": effective_rank(within),
        "within_effective_rank_by_class": class_effective_ranks,
        "between_trace": between_trace,
        "within_trace": within_trace,
        "between_to_within_trace_ratio": between_trace / max(within_trace, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1,2,4,8,16,32])
    args = parser.parse_args()
    files = sorted(args.gradient_dir.glob("*.pt"))
    if len(files) < 2: raise SystemExit("need at least two gradient files")
    records = [torch.load(path, map_location="cpu", weights_only=True) for path in files]
    labels = np.array([str(record.get("target", "")) for record in records])
    keys = sorted(records[0]["pooled_audio_gradient"])
    output = {"n": len(records), "labels": {
        label: int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))
    }, "projections": {}, "layers": {}}

    def summarize(matrix: np.ndarray) -> dict:
        spectrum = gradient_spectrum(matrix, args.ranks, return_basis=False)
        cosine = spectrum.pop("cosine"); spectrum.pop("basis")
        off_diagonal = ~np.eye(len(cosine), dtype=bool)
        same_label = labels[:, None] == labels[None, :]
        same_values = cosine[off_diagonal & same_label]
        different_values = cosine[off_diagonal & ~same_label]
        neighbor = cosine.copy(); np.fill_diagonal(neighbor, -np.inf)
        nearest = neighbor.argmax(axis=1)
        return {
            "singular_values": spectrum["singular_values"].tolist(),
            "explained_energy": spectrum["explained_energy"],
            "mean_off_diagonal_cosine": float((cosine.sum()-np.trace(cosine))/(cosine.size-len(cosine))),
            "off_diagonal_cosine_quantiles": {
                "q10": float(np.quantile(cosine[off_diagonal], .1)),
                "q50": float(np.quantile(cosine[off_diagonal], .5)),
                "q90": float(np.quantile(cosine[off_diagonal], .9)),
            },
            "same_label_mean_cosine": float(same_values.mean()) if len(same_values) else None,
            "different_label_mean_cosine": float(different_values.mean()) if len(different_values) else None,
            "same_minus_different_cosine": (
                float(same_values.mean() - different_values.mean())
                if len(same_values) and len(different_values) else None
            ),
            "nearest_cosine_label_accuracy": float(np.mean(labels[nearest] == labels)),
            **class_geometry(matrix, labels),
        }

    for layer, kind in keys:
        matrix = torch.stack([
            record["pooled_audio_gradient"][(layer, kind)] for record in records
        ]).numpy()
        output["projections"][f"{layer}_{kind}"] = summarize(matrix)
    for layer in sorted({layer for layer, _ in keys}):
        matrix = np.concatenate([
            torch.stack([
                record["pooled_audio_gradient"][(layer, kind)] for record in records
            ]).numpy()
            for kind in ("k", "v") if (layer, kind) in keys
        ], axis=1)
        output["layers"][str(layer)] = summarize(matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__": main()
