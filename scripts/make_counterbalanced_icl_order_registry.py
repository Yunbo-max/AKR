#!/usr/bin/env python3
"""Assign support-order rotations approximately evenly within each query class."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {
        row["event_id"]: row
        for row in csv.DictReader(args.manifest.open(newline="", encoding="utf-8"))
    }
    split = json.loads(args.split.read_text())
    labels = split["labels"]
    by_label = defaultdict(list)
    for event in split["query_events"]:
        by_label[rows[event]["label"]].append(event)
    rng = np.random.default_rng(args.seed)
    assignments = {}
    global_counts = Counter()
    for label in labels:
        events = sorted(by_label[label])
        rng.shuffle(events)
        candidates = []
        for offset in range(len(labels)):
            proposed = global_counts.copy()
            proposed.update((index + offset) % len(labels) for index in range(len(events)))
            values = [proposed.get(index, 0) for index in range(len(labels))]
            candidates.append((max(values) - min(values), sum(value * value for value in values), offset))
        offset = min(candidates)[2]
        for index, event in enumerate(events):
            rotation = int((index + offset) % len(labels))
            assignments[event] = rotation
            global_counts[rotation] += 1
    if set(assignments) != set(split["query_events"]):
        raise RuntimeError("registry does not cover the frozen query set")
    rotation_counts = Counter(assignments.values())
    within_target = {
        label: dict(Counter(assignments[event] for event in by_label[label]))
        for label in labels
    }
    payload = {
        "seed": args.seed,
        "manifest": str(args.manifest),
        "split": str(args.split),
        "labels": labels,
        "n_query": len(assignments),
        "assignment_uses_query_label_for": (
            "experimental blocking only; never exposed to model or used for model selection"
        ),
        "rotation_definition": (
            "cyclic class-order shift; rotations distributed within each target class "
            "with count difference at most one"
        ),
        "rotation_by_query": assignments,
        "rotation_counts": dict(sorted(rotation_counts.items())),
        "rotation_counts_by_target": within_target,
    }
    for counts in within_target.values():
        values = [counts.get(index, 0) for index in range(len(labels))]
        if max(values) - min(values) > 1:
            raise RuntimeError("within-target rotation blocking is imbalanced")
    values = [rotation_counts.get(index, 0) for index in range(len(labels))]
    if max(values) - min(values) > 1:
        raise RuntimeError("aggregate rotation blocking is imbalanced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
