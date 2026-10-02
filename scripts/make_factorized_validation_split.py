#!/usr/bin/env python3
"""Create a deterministic stratified rank-selection/confirmation validation split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", default="lp_0-1000")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--selection-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in csv.DictReader(args.manifest.open(newline="", encoding="utf-8"))
            if row["condition"] == args.condition and row["split"] == args.split]
    labels = sorted({row["label"] for row in rows})
    selection = []
    for label in labels:
        candidates = sorted(
            [row for row in rows if row["label"] == label],
            key=lambda row: hashlib.sha256(
                f"{args.seed}:{label}:{row['event_id']}".encode()
            ).hexdigest(),
        )
        if len(candidates) < args.selection_per_class:
            raise RuntimeError(f"{label} has fewer than requested selection examples")
        selection.extend(row["event_id"] for row in candidates[:args.selection_per_class])
    selected = set(selection)
    confirmation = [row["event_id"] for row in rows if row["event_id"] not in selected]
    payload = {
        "protocol": "stratified rank selection; confirmation labels used only for final scoring",
        "seed": args.seed, "condition": args.condition, "source_split": args.split,
        "selection_per_class": args.selection_per_class,
        "selection": selection, "confirmation": confirmation,
        "selection_label_counts": {
            label: sum(row["event_id"] in selected and row["label"] == label for row in rows)
            for label in labels
        },
        "confirmation_label_counts": {
            label: sum(row["event_id"] not in selected and row["label"] == label for row in rows)
            for label in labels
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
