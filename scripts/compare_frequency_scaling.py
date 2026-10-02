#!/usr/bin/env python3
"""Paired bootstrap comparison of 7B and 3B condition-wise accuracy."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from animal_omni.bootstrap import paired_accuracy_delta_ci


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    small, large = defaultdict(dict), defaultdict(dict)
    for row in read(args.small):
        small[row["condition"]][row["event_id"]] = row
    for row in read(args.large):
        large[row["condition"]][row["event_id"]] = row
    summaries = []
    for condition in sorted(set(small) & set(large)):
        event_ids = sorted(set(small[condition]) & set(large[condition]))
        paired = paired_accuracy_delta_ci(
            [large[condition][event_id]["target"] for event_id in event_ids],
            [large[condition][event_id]["prediction"] or None for event_id in event_ids],
            [small[condition][event_id]["prediction"] or None for event_id in event_ids],
            samples=args.bootstrap_samples,
            seed=20250813,
        )
        summaries.append({
            "condition": condition,
            "n": len(event_ids),
            "small_accuracy": sum(
                small[condition][event_id]["correct"] == "true" for event_id in event_ids
            ) / len(event_ids),
            "large_accuracy": sum(
                large[condition][event_id]["correct"] == "true" for event_id in event_ids
            ) / len(event_ids),
            "large_minus_small": paired,
        })
    result = {"paired_by_event": True, "conditions": summaries}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
