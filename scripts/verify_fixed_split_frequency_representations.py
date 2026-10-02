#!/usr/bin/env python3
"""Verify complete Qwen fixed-split representations at every low-pass cutoff."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONDITIONS = ["full_0-8k", "lp_0-1000", "lp_0-2000", "lp_0-4000", "lp_0-6000", "lp_0-8000"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", choices=["dogs", "watkins"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = list(csv.DictReader(
        (root / f"data/manifests/beans_{args.dataset}_protocol.csv").open(
            newline="", encoding="utf-8"
        )
    ))
    expected = {row["event_id"]: row for row in manifest}
    directories = {
        "full_0-8k": root / f"results/reps_beans_{args.dataset}_7b",
        "lp_0-1000": root / f"results/reps_beans_{args.dataset}_lp1_7b",
        **{
            f"lp_0-{cutoff}000": (
                root / f"results/reps_beans_{args.dataset}_frequency_extra_7b/lp_0-{cutoff}000"
            )
            for cutoff in (2, 4, 6, 8)
        },
    }
    output = {
        "dataset": args.dataset,
        "expected_total": len(expected),
        "expected_split_counts": {
            split: sum(row["split"] == split for row in manifest)
            for split in ("train", "valid", "test")
        },
        "conditions": {},
        "passed": True,
    }
    for condition in CONDITIONS:
        directory = directories[condition]
        files = sorted(directory.glob("*.npz"))
        events = {path.stem for path in files}
        metadata_errors = 0
        representation_errors = 0
        split_counts = {split: 0 for split in ("train", "valid", "test")}
        for path in files:
            if path.stem not in expected:
                metadata_errors += 1
                continue
            row = expected[path.stem]
            try:
                with np.load(path, allow_pickle=False) as payload:
                    representation = payload["representation"]
                    if (representation.shape != (29, 3584)
                            or representation.dtype != np.float16
                            or not np.isfinite(representation).all()):
                        representation_errors += 1
                    expected_metadata = {
                        "event_id": row["event_id"], "label": row["label"],
                        "split": row["split"], "condition": condition,
                        "model_id": "Qwen/Qwen2.5-Omni-7B",
                    }
                    metadata_errors += sum(
                        key not in payload.files or str(payload[key]) != value
                        for key, value in expected_metadata.items()
                    )
                    split_counts[row["split"]] += 1
            except Exception:
                representation_errors += 1
        passed = len(files) == len(expected) and events == set(expected)
        passed &= metadata_errors == 0 and representation_errors == 0
        passed &= split_counts == output["expected_split_counts"]
        output["passed"] &= passed
        output["conditions"][condition] = {
            "directory": str(directory.relative_to(root)),
            "files": len(files), "unique_events": len(events),
            "split_counts": split_counts, "metadata_errors": metadata_errors,
            "representation_errors": representation_errors, "passed": passed,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
