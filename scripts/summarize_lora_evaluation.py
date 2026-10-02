#!/usr/bin/env python3
"""Summarize a complete deterministic Thinker-LoRA prediction CSV."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

from animal_omni.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    labels = config["dataset"]["labels"]
    manifest = [
        row for row in csv.DictReader(
            args.manifest.open(newline="", encoding="utf-8")
        ) if row["split"] == args.split
    ]
    rows = list(csv.DictReader(
        args.predictions.open(newline="", encoding="utf-8")
    ))
    expected = {row["event_id"]: row for row in manifest}
    observed = {row["event_id"]: row for row in rows}
    if len(observed) != len(rows) or set(observed) != set(expected):
        raise RuntimeError(
            f"prediction coverage mismatch: expected={len(expected)} "
            f"unique_observed={len(observed)} rows={len(rows)}"
        )
    ordered = [observed[row["event_id"]] for row in manifest]
    if any(row["target"] != expected[row["event_id"]]["label"] for row in ordered):
        raise RuntimeError("prediction targets do not match manifest")
    predictions = [row["prediction"] or None for row in ordered]
    summary = {
        **classification_metrics(
            [row["target"] for row in ordered], predictions, labels
        ),
        "n": len(ordered),
        "invalid": sum(prediction is None for prediction in predictions),
        "invalid_response_rate": sum(prediction is None for prediction in predictions) / len(ordered),
        "prediction_counts": dict(Counter(row["prediction"] for row in ordered)),
        "split": args.split,
        "protocol": args.protocol,
        "coverage_verified_against_manifest": True,
        "query_labels_used_for": "post_hoc_scoring_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
