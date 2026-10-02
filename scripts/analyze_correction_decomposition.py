#!/usr/bin/env python3
"""Decompose saved corrective KV gradients into class-shared and residual parts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def effective_rank(matrix: np.ndarray) -> float:
    if len(matrix) < 2 or not np.any(matrix):
        return 0.0
    gram = matrix @ matrix.T
    spectrum = np.clip(np.linalg.eigvalsh(gram), 0, None)
    if float(spectrum.sum()) <= 0:
        return 0.0
    probability = spectrum[spectrum > 0] / spectrum.sum()
    return float(np.exp(-(probability * np.log(probability)).sum()))


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def summarize(matrix: np.ndarray, labels: np.ndarray) -> dict:
    classes = sorted(set(labels.tolist()))
    centroids = {label: matrix[labels == label].mean(0) for label in classes}
    reconstruction = np.stack([centroids[label] for label in labels])
    residual = matrix - reconstruction
    grand = matrix.mean(0, keepdims=True)
    centered = matrix - grand
    total_centered = float(np.square(centered).sum())
    within = float(np.square(residual).sum())
    between = max(total_centered - within, 0.0)
    total_raw = float(np.square(matrix).sum())

    loo_prediction = []
    loo_available = all(int(np.sum(labels == label)) >= 2 for label in classes)
    if loo_available:
        for index, row in enumerate(matrix):
            candidate_centroids = []
            for label in classes:
                selected = matrix[(labels == label) & (np.arange(len(labels)) != index)]
                candidate_centroids.append(selected.mean(0))
            candidates = np.stack(candidate_centroids)
            similarities = cosine_rows(candidates, np.repeat(row[None], len(classes), axis=0))
            loo_prediction.append(classes[int(np.argmax(similarities))])

    by_class_rank = {}
    for label in classes:
        selected = matrix[labels == label]
        by_class_rank[label] = effective_rank(selected - selected.mean(0, keepdims=True))

    residual_relative_norm = np.linalg.norm(residual, axis=1) / np.maximum(
        np.linalg.norm(matrix, axis=1), 1e-12
    )
    return {
        "global_effective_rank_centered": effective_rank(centered),
        "within_effective_rank_pooled": effective_rank(residual),
        "within_effective_rank_by_class": by_class_rank,
        "class_centroid_fraction_raw_energy": 1.0 - within / max(total_raw, 1e-12),
        "class_specific_fraction_centered_variance": between / max(total_centered, 1e-12),
        "between_to_within_trace_ratio": between / max(within, 1e-12),
        "mean_cosine_to_class_centroid": float(cosine_rows(matrix, reconstruction).mean()),
        "median_cosine_to_class_centroid": float(np.median(cosine_rows(matrix, reconstruction))),
        "mean_relative_residual_norm": float(residual_relative_norm.mean()),
        "median_relative_residual_norm": float(np.median(residual_relative_norm)),
        "leave_one_out_centroid_accuracy": (
            float(np.mean(np.asarray(loo_prediction) == labels)) if loo_available else None
        ),
    }


def load(path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    records = [
        torch.load(file, map_location="cpu", weights_only=True)
        for file in sorted(path.glob("*.pt"))
    ]
    if len(records) < 2:
        raise RuntimeError(f"need at least two gradient files in {path}")
    labels = np.asarray([str(record["target"]) for record in records])
    keys = sorted(records[0]["pooled_audio_gradient"])
    if any(sorted(record["pooled_audio_gradient"]) != keys for record in records):
        raise RuntimeError(f"inconsistent gradient keys in {path}")
    layers = {}
    for layer in sorted({key[0] for key in keys}):
        layers[layer] = np.concatenate([
            torch.stack([
                record["pooled_audio_gradient"][(layer, kind)] for record in records
            ]).float().numpy()
            for kind in ("k", "v") if (layer, kind) in keys
        ], axis=1)
    return labels, layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", action="append", required=True,
        help="repeat name=gradient_directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "definition": (
            "g_i = mu_{y_i} + residual_i; class-specific fraction is between-class "
            "centered variance divided by total centered variance"
        ),
        "datasets": {},
    }
    for item in args.dataset:
        name, separator, directory = item.partition("=")
        if not separator or name in output["datasets"]:
            raise ValueError(f"invalid or duplicate dataset {item!r}")
        labels, layers = load(Path(directory))
        layer_results = {str(layer): summarize(matrix, labels) for layer, matrix in layers.items()}
        best_layer = max(
            layers,
            key=lambda layer: layer_results[str(layer)]["class_specific_fraction_centered_variance"],
        )
        output["datasets"][name] = {
            "gradient_directory": directory,
            "n": len(labels),
            "label_counts": {
                label: int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))
            },
            "best_layer": int(best_layer),
            "best_layer_metrics": layer_results[str(best_layer)],
            "median_across_layers": {
                key: float(np.median([metrics[key] for metrics in layer_results.values()]))
                for key in (
                    "global_effective_rank_centered",
                    "within_effective_rank_pooled",
                    "class_centroid_fraction_raw_energy",
                    "class_specific_fraction_centered_variance",
                    "between_to_within_trace_ratio",
                    "median_cosine_to_class_centroid",
                    "median_relative_residual_norm",
                )
            },
            "layers": layer_results,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {"best_layer": data["best_layer"], **data["best_layer_metrics"]}
        for name, data in output["datasets"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
