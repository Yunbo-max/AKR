#!/usr/bin/env python3
"""Compare corrective-gradient geometry across matched frequency conditions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def effective_rank(matrix: np.ndarray) -> float:
    centered = matrix - matrix.mean(0, keepdims=True)
    spectrum = np.clip(np.linalg.eigvalsh(centered @ centered.T), 0, None)
    if float(spectrum.sum()) <= 0:
        return 0.0
    probability = spectrum[spectrum > 0] / spectrum.sum()
    return float(np.exp(-(probability * np.log(probability)).sum()))


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return normalized @ normalized.T


def within_condition(matrix: np.ndarray, labels: np.ndarray) -> dict:
    cosine = cosine_matrix(matrix)
    off = ~np.eye(len(matrix), dtype=bool)
    same = labels[:, None] == labels[None, :]
    same_values = cosine[off & same]
    different_values = cosine[off & ~same]
    grand = matrix.mean(0)
    within_trace = 0.0
    between_trace = 0.0
    for label in sorted(set(labels.tolist())):
        selected = matrix[labels == label]
        center = selected.mean(0)
        within_trace += float(np.square(selected - center).sum())
        between_trace += len(selected) * float(np.square(center - grand).sum())
    neighbor = cosine.copy()
    np.fill_diagonal(neighbor, -np.inf)
    nearest = neighbor.argmax(1)
    return {
        "same_label_mean_cosine": float(same_values.mean()),
        "different_label_mean_cosine": float(different_values.mean()),
        "same_minus_different_cosine": float(same_values.mean() - different_values.mean()),
        "nearest_cosine_label_accuracy": float(np.mean(labels[nearest] == labels)),
        "effective_rank_centered": effective_rank(matrix),
        "between_to_within_trace_ratio": between_trace / max(within_trace, 1e-12),
        "median_gradient_norm": float(np.median(np.linalg.norm(matrix, axis=1))),
    }


def read_directory(path: Path) -> tuple[list[str], np.ndarray, dict[int, np.ndarray]]:
    records = [
        torch.load(file, map_location="cpu", weights_only=True)
        for file in sorted(path.glob("*.pt"))
    ]
    if len(records) < 2:
        raise RuntimeError(f"need at least two gradients in {path}")
    records.sort(key=lambda record: str(record["event_id"]))
    event_ids = [str(record["event_id"]) for record in records]
    labels = np.asarray([str(record["target"]) for record in records])
    keys = sorted(records[0]["pooled_audio_gradient"])
    if any(sorted(record["pooled_audio_gradient"]) != keys for record in records):
        raise RuntimeError(f"gradient keys differ within {path}")
    layers = {}
    for layer in sorted({key[0] for key in keys}):
        layers[layer] = np.concatenate([
            torch.stack([
                record["pooled_audio_gradient"][(layer, kind)] for record in records
            ]).numpy()
            for kind in ("k", "v")
        ], axis=1)
    return event_ids, labels, layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition-dir", action="append", required=True,
        help="repeat condition=gradient_directory",
    )
    parser.add_argument("--anchor", default="full_0-8k")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {}
    for item in args.condition_dir:
        condition, separator, directory = item.partition("=")
        if not separator or condition in paths:
            raise ValueError(f"invalid or duplicate --condition-dir {item!r}")
        paths[condition] = Path(directory)
    if args.anchor not in paths:
        raise ValueError("anchor is not present in --condition-dir")

    loaded = {condition: read_directory(path) for condition, path in paths.items()}
    anchor_events, anchor_labels, anchor_layers = loaded[args.anchor]
    for condition, (events, labels, layers) in loaded.items():
        if events != anchor_events or not np.array_equal(labels, anchor_labels):
            raise RuntimeError(f"support IDs/labels differ for {condition}")
        if set(layers) != set(anchor_layers):
            raise RuntimeError(f"layer set differs for {condition}")

    output = {
        "protocol": "same registered K=2/class support at every frequency condition",
        "n_support": len(anchor_events),
        "event_ids": anchor_events,
        "labels": {label: int(np.sum(anchor_labels == label))
                   for label in sorted(set(anchor_labels.tolist()))},
        "anchor": args.anchor,
        "conditions": list(paths),
        "layers": {},
        "condition_summary": {},
    }
    for layer in sorted(anchor_layers):
        layer_output = {}
        anchor = anchor_layers[layer]
        anchor_norm = np.maximum(np.linalg.norm(anchor, axis=1), 1e-12)
        for condition, (_, _, layers) in loaded.items():
            matrix = layers[layer]
            current_norm = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
            paired_cosine = np.sum(anchor * matrix, axis=1) / (anchor_norm * current_norm)
            layer_output[condition] = {
                **within_condition(matrix, anchor_labels),
                "paired_to_anchor_cosine_mean": float(paired_cosine.mean()),
                "paired_to_anchor_cosine_median": float(np.median(paired_cosine)),
            }
        output["layers"][str(layer)] = layer_output

    for condition in paths:
        rows = [output["layers"][str(layer)][condition] for layer in sorted(anchor_layers)]
        best_layer = max(
            sorted(anchor_layers),
            key=lambda layer: output["layers"][str(layer)][condition][
                "same_minus_different_cosine"
            ],
        )
        output["condition_summary"][condition] = {
            "mean_layer_class_separation": float(np.mean([
                row["same_minus_different_cosine"] for row in rows
            ])),
            "median_layer_effective_rank": float(np.median([
                row["effective_rank_centered"] for row in rows
            ])),
            "mean_paired_to_anchor_cosine": float(np.mean([
                row["paired_to_anchor_cosine_mean"] for row in rows
            ])),
            "best_separation_layer": int(best_layer),
            "best_layer_class_separation": output["layers"][str(best_layer)][condition][
                "same_minus_different_cosine"
            ],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["condition_summary"], indent=2))


if __name__ == "__main__":
    main()
