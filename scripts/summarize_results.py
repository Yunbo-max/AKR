#!/usr/bin/env python3
"""Aggregate condition metrics and paired deltas from completed predictions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

from animal_omni.bootstrap import paired_accuracy_delta_ci
from animal_omni.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    labels = cfg["dataset"]["labels"]
    with args.predictions.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    if "full_0-8k" not in grouped:
        raise SystemExit("full_0-8k predictions are required")
    full = {r["event_id"]: r for r in grouped["full_0-8k"]}
    summaries = []
    paired = {}
    for condition, condition_rows in sorted(grouped.items()):
        targets = [r["target"] for r in condition_rows]
        predictions = [r["prediction"] or None for r in condition_rows]
        metrics = classification_metrics(targets, predictions, labels)
        summaries.append({"condition": condition, "n": len(condition_rows), **metrics})
        current = {r["event_id"]: r for r in condition_rows}
        common = sorted(full.keys() & current.keys())
        if condition != "full_0-8k" and common:
            # Importance is Acc(full)-Acc(condition), so bootstrap A=full.
            paired[condition] = {
                "n": len(common),
                **paired_accuracy_delta_ci(
                    [full[k]["target"] for k in common],
                    [full[k]["prediction"] or None for k in common],
                    [current[k]["prediction"] or None for k in common],
                    samples=args.bootstrap_samples, seed=int(cfg.get("seed", 20250813)),
                ),
            }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader(); writer.writerows(summaries)
    args.output_json.write_text(json.dumps({"conditions": summaries, "paired_full_minus_condition": paired}, indent=2))


if __name__ == "__main__":
    main()
