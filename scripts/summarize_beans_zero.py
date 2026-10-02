#!/usr/bin/env python3
"""Summarize canonical exact match for BEANS-Zero components."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def target_contained(row: dict[str, str]) -> bool:
    """Target-aware diagnostic only; never replaces canonical exact match."""
    target = row["canonical_target"].strip()
    prediction = row["canonical_prediction"].strip()
    return bool(target) and f" {target} " in f" {prediction} "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset_name"]].append(row)
    result = {}
    for component, selected in sorted(grouped.items()):
        correct = sum(row["exact_match"] == "true" for row in selected)
        contained = sum(target_contained(row) for row in selected)
        result[component] = {
            "n": len(selected), "exact_match": correct / len(selected), "correct": correct,
            "target_contained": contained / len(selected),
            "contained_correct": contained,
        }
    correct = sum(row["exact_match"] == "true" for row in rows)
    contained = sum(target_contained(row) for row in rows)
    result["overall"] = {
        "n": len(rows), "exact_match": correct / len(rows), "correct": correct,
        "target_contained": contained / len(rows),
        "contained_correct": contained,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
