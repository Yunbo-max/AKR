#!/usr/bin/env python3
"""Freeze one factorized rank from the rank-selection subset."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from animal_omni.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], {})[row["event_id"]] = row
    if any(len(cell) != args.expected_n for cell in grouped.values()):
        raise RuntimeError("rank-selection cells are incomplete")
    candidates = sorted(method for method in grouped if "factorized" in method)
    cells = []
    for method in sorted(grouped):
        selected = list(grouped[method].values())
        metric = classification_metrics(
            [row["target_output"] for row in selected],
            [row["prediction"] or None for row in selected], list("ABCDEFGHIJ")
        )
        cells.append({"method": method, **metric,
                      "invalid_response_rate": sum(not row["prediction"] for row in selected) / len(selected),
                      "prediction_counts": dict(Counter(row["prediction"] for row in selected))})
    lookup = {cell["method"]: cell for cell in cells}
    selected_method = max(candidates, key=lambda method: (
        lookup[method]["accuracy"], lookup[method]["macro_f1"],
        -int(re.search(r"r(\d+)$", method).group(1)),
    ))
    payload = {
        "selection_rule": "highest accuracy, then macro-F1, then lower rank at alpha=.01",
        "selected_method": selected_method,
        "selected_rank": int(re.search(r"r(\d+)$", selected_method).group(1)),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
