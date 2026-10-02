#!/usr/bin/env python3
"""Summarize complete matched-norm token-KV validation cells."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from animal_omni.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], float(row["relative_alpha"]))].append(row)
    cells, incomplete = [], []
    for (method, alpha), selected in sorted(grouped.items()):
        if len(selected) != args.expected_n or len({r["event_id"] for r in selected}) != args.expected_n:
            incomplete.append({"method": method, "relative_alpha": alpha, "n": len(selected)})
            continue
        metrics = classification_metrics(
            [row["target"] for row in selected],
            [row["prediction"] or None for row in selected], args.labels,
        )
        cells.append({
            "method": method, "relative_alpha": alpha, "n": len(selected), **metrics,
            "achieved_ratio_mean": sum(float(r["applied_ratio_mean"]) for r in selected) / len(selected),
        })
    payload = {"complete_cells": cells, "incomplete_cells": incomplete}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
