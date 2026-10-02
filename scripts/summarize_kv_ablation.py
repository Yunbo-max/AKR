#!/usr/bin/env python3
"""Select a conditional-KV layer/rank using validation macro-F1 only."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

from animal_omni.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = yaml.safe_load(args.config.read_text())["dataset"]["labels"]
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["support_k"]), int(row["feature_layer"]),
                 int(row["requested_rank"]), row["method"])].append(row)
    summaries = []
    for (support_k, layer, rank, method), selected in sorted(grouped.items()):
        metrics = classification_metrics(
            [row["target"] for row in selected],
            [row["prediction"] or None for row in selected],
            labels,
        )
        summaries.append({
            "feature_layer": layer,
            "support_k": support_k,
            "requested_rank": rank,
            "method": method,
            "actual_rank": int(selected[0]["actual_rank"]),
            "n": len(selected),
            **metrics,
        })
    conditional = [row for row in summaries if row["method"] == "conditional"]
    best = max(
        conditional,
        key=lambda row: (
            row["macro_f1"], row["accuracy"],
            -row["requested_rank"], -row["feature_layer"], -row["support_k"],
        ),
    )
    result = {
        "selection_split": "validation",
        "selection_metric": "macro_f1_then_accuracy_then_smaller_rank_then_earlier_layer_then_smaller_k",
        "best": best,
        "candidates": summaries,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
