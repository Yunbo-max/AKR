#!/usr/bin/env python3
"""Create nested K-per-class support and untouched query partitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def order(rows: list[dict], seed: int, label: str) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}:{label}:{row['event_id']}".encode()
    ).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", help="optional manifest condition filter")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--support-k-per-class", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--support-source-split")
    parser.add_argument("--query-split")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--group-column")
    parser.add_argument("--query-groups-from", type=Path)
    parser.add_argument("--support-events-from", type=Path,
                        help="reuse an existing JSON support_order exactly")
    parser.add_argument("--limit-query", type=int)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if args.condition is not None:
        rows = [row for row in rows if row.get("condition") == args.condition]
    by_event = {row["event_id"]: row for row in rows}
    if len(by_event) != len(rows):
        raise ValueError("event_id must be unique")
    if args.query_groups_from:
        if not args.group_column:
            raise ValueError("--group-column is required with --query-groups-from")
        source = json.loads(args.query_groups_from.read_text())
        query_groups = set(source["query_groups"])
        query = [row for row in rows if row[args.group_column] in query_groups]
        support_pool = [row for row in rows if row[args.group_column] not in query_groups]
        partition = "recording-group-disjoint"
    else:
        if not args.support_source_split or not args.query_split:
            raise ValueError("fixed splits require --support-source-split and --query-split")
        support_pool = [row for row in rows if row[args.split_column] == args.support_source_split]
        query = [row for row in rows if row[args.split_column] == args.query_split]
        query_groups = set()
        partition = "official-fixed-split"
    if args.limit_query is not None:
        query = query[:args.limit_query]
    maximum = max(args.support_k_per_class)
    support_by_label = {}
    registered = None
    if args.support_events_from:
        source = json.loads(args.support_events_from.read_text())
        registered = [by_event[event] for event in source["support_order"]]
    for label in args.labels:
        if registered is not None:
            candidates = [row for row in registered if row["label"] == label]
        else:
            candidates = order([row for row in support_pool if row["label"] == label], args.seed, label)
        if len(candidates) < maximum:
            raise ValueError(f"{label}: {len(candidates)} support rows, need {maximum}")
        # Prefer distinct recordings within each label when a group is available.
        if args.group_column:
            distinct, deferred, seen = [], [], set()
            for row in candidates:
                group = row[args.group_column]
                (distinct if group not in seen else deferred).append(row)
                seen.add(group)
            candidates = distinct + deferred
        support_by_label[label] = [row["event_id"] for row in candidates[:maximum]]
    support_sets = {
        str(k): [event for label in args.labels for event in support_by_label[label][:k]]
        for k in args.support_k_per_class
    }
    payload = {
        "seed": args.seed, "manifest": str(args.manifest), "condition": args.condition,
        "labels": args.labels,
        "partition": partition, "group_column": args.group_column,
        "query_groups": sorted(query_groups),
        "support_source_split": args.support_source_split,
        "support_events_from": str(args.support_events_from) if args.support_events_from else None,
        "query_split": args.query_split,
        "support_k_per_class": args.support_k_per_class,
        "support_by_label": support_by_label, "support_sets": support_sets,
        "query_events": [row["event_id"] for row in query],
        "query_label_counts": {
            label: sum(row["label"] == label for row in query) for label in args.labels
        },
        "query_labels_used_for": "post_hoc_scoring_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
