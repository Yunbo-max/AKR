#!/usr/bin/env python3
"""Materialize a compact manifest from a registered equal-support split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--support-k-per-class", type=int, required=True)
    parser.add_argument("--condition")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if args.condition is not None:
        rows = [row for row in rows if row.get("condition") == args.condition]
    by_event = {row["event_id"]: row for row in rows}
    split = json.loads(args.split.read_text())
    support = split["support_sets"][str(args.support_k_per_class)]
    query = split["query_events"]
    overlap = set(support) & set(query)
    if overlap:
        raise ValueError(f"support/query overlap: {sorted(overlap)[:5]}")
    output = []
    for role, events in (("train", support), ("valid", query)):
        for event in events:
            row = dict(by_event[event]); row["source_split"] = row.get("split", "")
            row["split"] = role
            row["equal_support_k_per_class"] = args.support_k_per_class
            row["equal_support_role"] = "labeled_support" if role == "train" else "untouched_query"
            output.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0])
        writer.writeheader(); writer.writerows(output)
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output), "support": len(support), "query": len(query),
        "support_k_per_class": args.support_k_per_class, "condition": args.condition,
    }, indent=2))


if __name__ == "__main__":
    main()
