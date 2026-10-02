#!/usr/bin/env python3
"""Summarize complete tokenwise-KV validation groups and select eta."""
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
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--conditional-method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = yaml.safe_load(args.config.read_text())["dataset"]["labels"]
    rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], float(row["eta"]))].append(row)
    candidates, incomplete = [], []
    for (method, eta), selected in sorted(grouped.items()):
        event_ids = {row["event_id"] for row in selected}
        if len(selected) != args.expected_n or len(event_ids) != args.expected_n:
            incomplete.append({"method": method, "eta": eta, "n": len(selected),
                               "unique_events": len(event_ids)})
            continue
        candidates.append({
            "method": method, "eta": eta, "n": len(selected),
            **classification_metrics(
                [row["target"] for row in selected],
                [row["prediction"] or None for row in selected], labels,
            ),
        })
    conditional = [row for row in candidates if row["method"] == args.conditional_method]
    if not conditional:
        raise SystemExit(f"no complete groups for {args.conditional_method}")
    best = max(conditional, key=lambda row: (row["macro_f1"], row["accuracy"], -row["eta"]))
    result = {
        "selection_split": "validation", "expected_n": args.expected_n,
        "selection_metric": "macro_f1_then_accuracy_then_smaller_eta",
        "best": best, "complete_candidates": candidates,
        "incomplete_groups_excluded": incomplete,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
