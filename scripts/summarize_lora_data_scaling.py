#!/usr/bin/env python3
"""Matched training-prefix LoRA versus frozen-ridge scaling on Dogs validation."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from animal_omni.metrics import classification_metrics


LABELS = ["Farley", "Freid", "Keri", "Louie", "Luke", "Mac",
          "Roodie", "Rudy", "Siggy", "Zoe"]


def representation(root: Path, event: str, layer: int) -> np.ndarray:
    with np.load(root / f"{event}.npz", allow_pickle=False) as payload:
        return payload["representation"][layer].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True,
                        help="repeat n=prediction.csv")
    parser.add_argument("--feature-layer", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20250813)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    train = [row for row in rows if row["split"] == "train"]
    valid = [row for row in rows if row["split"] == "valid"]
    random.Random(args.seed).shuffle(train)
    valid_x = np.stack([
        representation(args.representation_dir, row["event_id"], args.feature_layer)
        for row in valid
    ])
    valid_y = np.asarray([row["label"] for row in valid])
    cells = []
    for item in args.prediction:
        n_text, separator, filename = item.partition("=")
        if not separator:
            raise ValueError(f"invalid --prediction {item!r}")
        n = int(n_text); prefix = train[:n]
        train_x = np.stack([
            representation(args.representation_dir, row["event_id"], args.feature_layer)
            for row in prefix
        ])
        train_y = np.asarray([row["label"] for row in prefix])
        ridge_prediction = make_pipeline(
            StandardScaler(), RidgeClassifier(alpha=args.ridge_alpha)
        ).fit(train_x, train_y).predict(valid_x)
        lora_rows = list(csv.DictReader(Path(filename).open(newline="", encoding="utf-8")))
        if len(lora_rows) != len(valid) or {row["event_id"] for row in lora_rows} != {
            row["event_id"] for row in valid
        }:
            raise RuntimeError(f"incomplete or mismatched LoRA validation file {filename}")
        by_event = {row["event_id"]: row for row in lora_rows}
        lora_prediction = [by_event[row["event_id"]]["prediction"] or None for row in valid]
        cells.append({
            "n_train_prefix": n,
            "label_coverage": len(set(train_y.tolist())),
            "train_label_counts": dict(Counter(train_y.tolist())),
            "lora": {
                **classification_metrics(valid_y.tolist(), lora_prediction, LABELS),
                "invalid_response_rate": float(np.mean([value is None for value in lora_prediction])),
                "prediction_counts": dict(Counter(value or "" for value in lora_prediction)),
            },
            "ridge": classification_metrics(valid_y.tolist(), ridge_prediction.tolist(), LABELS),
            "ridge_minus_lora_accuracy": float(np.mean(ridge_prediction == valid_y) - np.mean([
                prediction == target for prediction, target in zip(lora_prediction, valid_y)
            ])),
        })
    payload = {
        "protocol": "same deterministic shuffled training prefix for LoRA checkpoint and frozen layer-10 ridge; complete official validation; no test selection",
        "seed": args.seed, "feature_layer": args.feature_layer,
        "ridge_alpha": args.ridge_alpha, "n_valid": len(valid), "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
