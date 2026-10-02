#!/usr/bin/env python3
"""Evaluate few-shot readouts with exactly the registered K-per-class support."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from animal_omni.metrics import classification_metrics


def load(root: Path, event_id: str, layer: int) -> np.ndarray:
    with np.load(root / f"{event_id}.npz", allow_pickle=False) as record:
        return record["representation"][layer].astype(np.float32)


def cosine_centroid(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray,
                    labels: list[str]) -> np.ndarray:
    train_x = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
    query_x = query_x / np.maximum(np.linalg.norm(query_x, axis=1, keepdims=True), 1e-12)
    centroids = np.stack([train_x[train_y == label].mean(0) for label in labels])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    return np.asarray(labels)[np.argmax(query_x @ centroids.T, axis=1)]


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--feature-layer", type=int, default=28)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    by_event = {row["event_id"]: row for row in rows}
    split = json.loads(args.split.read_text())
    labels = split["labels"]
    query_events = split["query_events"]
    query_x = np.stack([load(args.representation_dir, event, args.feature_layer)
                        for event in query_events])
    query_y = np.asarray([by_event[event]["label"] for event in query_events])
    output, summaries = [], {}
    for k_text, support_events in split["support_sets"].items():
        support_x = np.stack([load(args.representation_dir, event, args.feature_layer)
                              for event in support_events])
        support_y = np.asarray([by_event[event]["label"] for event in support_events])
        predictions = {
            "nearest_centroid": cosine_centroid(support_x, support_y, query_x, labels),
            "linear_probe": make_pipeline(
                StandardScaler(), RidgeClassifier(alpha=args.ridge_alpha)
            ).fit(support_x, support_y).predict(query_x),
        }
        for method, predicted in predictions.items():
            for event, target, prediction in zip(query_events, query_y, predicted):
                output.append({
                    "event_id": event, "target": target,
                    "support_k_per_class": int(k_text),
                    "support_k_total": len(support_events), "method": method,
                    "feature_layer": args.feature_layer, "prediction": prediction,
                    "correct": str(prediction == target).lower(),
                })
            summaries[f"K={k_text}/{method}"] = {
                "support_k_per_class": int(k_text), "support_k_total": len(support_events),
                "method": method, "n_query": len(query_events),
                **classification_metrics(query_y.tolist(), predicted.tolist(), labels),
            }
    atomic_csv(args.output, output)
    payload = {
        "manifest": str(args.manifest), "split": str(args.split),
        "representation_dir": str(args.representation_dir),
        "feature_layer": args.feature_layer, "ridge_alpha": args.ridge_alpha,
        "protocol": "same nested K-per-class support; no query-label model selection",
        "results": summaries,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
