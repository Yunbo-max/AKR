#!/usr/bin/env python3
"""Train grouped probes on full states and test paired frequency conditions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_directory(path: Path) -> dict[str, dict]:
    records = {}
    for file in sorted(path.glob("*.npz")):
        with np.load(file, allow_pickle=False) as record:
            event_id = str(record["event_id"])
            records[event_id] = {
                "representation": record["representation"].astype(np.float32),
                "label": str(record["label"]),
                "recording_id": str(record["recording_id"]),
            }
    return records


def macro_f1(target, prediction) -> float:
    return float(f1_score(target, prediction, average="macro", zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", action="append", required=True,
                        help="CONDITION=DIR; full_0-8k must be present")
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1, 10, 100])
    args = parser.parse_args()

    directories = {}
    for item in args.representation:
        condition, raw_path = item.split("=", 1)
        directories[condition] = Path(raw_path)
    if "full_0-8k" not in directories:
        raise SystemExit("full_0-8k representation directory is required")
    records = {condition: load_directory(path) for condition, path in directories.items()}
    event_ids = sorted(records["full_0-8k"])
    if any(set(value) != set(event_ids) for value in records.values()):
        raise SystemExit("all condition directories must contain identical event IDs")
    full_records = records["full_0-8k"]
    y = np.array([full_records[event_id]["label"] for event_id in event_ids])
    groups = np.array([full_records[event_id]["recording_id"] for event_id in event_ids])
    x = {
        condition: np.stack([value[event_id]["representation"] for event_id in event_ids])
        for condition, value in records.items()
    }
    layers = range(x["full_0-8k"].shape[1])
    outer = StratifiedGroupKFold(args.outer_folds, shuffle=True, random_state=20250813)
    predictions = {condition: np.empty(len(y), dtype=object) for condition in records}
    selections = []
    for fold, (train, test) in enumerate(outer.split(x["full_0-8k"][:, 0], y, groups), 1):
        inner = StratifiedGroupKFold(args.inner_folds, shuffle=True, random_state=20250813 + fold)
        candidates = []
        for layer in layers:
            for alpha in args.alphas:
                values = []
                for inner_train_rel, validation_rel in inner.split(
                    x["full_0-8k"][train, layer], y[train], groups[train]
                ):
                    inner_train = train[inner_train_rel]
                    validation = train[validation_rel]
                    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
                    model.fit(x["full_0-8k"][inner_train, layer], y[inner_train])
                    values.append(macro_f1(y[validation], model.predict(
                        x["full_0-8k"][validation, layer]
                    )))
                candidates.append((float(np.mean(values)), layer, alpha))
        inner_score, layer, alpha = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
        model.fit(x["full_0-8k"][train, layer], y[train])
        for condition in records:
            predictions[condition][test] = model.predict(x[condition][test, layer])
        selections.append({
            "fold": fold, "layer": layer, "alpha": alpha,
            "full_inner_macro_f1": inner_score,
            "n_train": len(train), "n_test": len(test),
        })

    rows = []
    summaries = []
    for condition, prediction in predictions.items():
        summaries.append({
            "condition": condition,
            "n": len(y),
            "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": macro_f1(y, prediction),
        })
        rows.extend({
            "event_id": event_id,
            "recording_id": group,
            "condition": condition,
            "target": target,
            "prediction": predicted,
            "correct": str(target == predicted).lower(),
        } for event_id, group, target, predicted in zip(event_ids, groups, y, prediction))
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.output_predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "protocol": "nested_recording_group_oof_train_full_apply_paired_conditions",
        "selection_uses": "full condition inner folds only",
        "conditions": summaries,
        "selections": selections,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
