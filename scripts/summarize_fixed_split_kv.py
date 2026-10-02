#!/usr/bin/env python3
"""Summarize fixed-split KV predictions against a paired degraded baseline."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

from animal_omni.bootstrap import paired_accuracy_delta_ci
from animal_omni.metrics import classification_metrics


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-condition", default="lp_0-1000")
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    labels = cfg["dataset"]["labels"]
    seed = int(cfg.get("seed", 20250813))
    baseline_rows = [
        row for row in read_rows(args.baseline)
        if row.get("condition") == args.baseline_condition
    ]
    if not baseline_rows:
        raise SystemExit(f"no baseline rows for {args.baseline_condition}")
    baseline = {row["event_id"]: row for row in baseline_rows}
    baseline_metrics = classification_metrics(
        [row["target"] for row in baseline_rows],
        [row["prediction"] or None for row in baseline_rows],
        labels,
    )

    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(args.adapted):
        grouped[(int(row["support_k"]), row["method"])].append(row)

    summaries = []
    for (support_k, method), rows in sorted(grouped.items()):
        if len(rows) != len(baseline):
            raise SystemExit(
                f"incomplete group K={support_k} {method}: "
                f"{len(rows)} adapted versus {len(baseline)} baseline"
            )
        adapted = {row["event_id"]: row for row in rows}
        if adapted.keys() != baseline.keys():
            raise SystemExit(f"event mismatch for K={support_k} {method}")
        event_ids = sorted(baseline)
        metrics = classification_metrics(
            [baseline[event_id]["target"] for event_id in event_ids],
            [adapted[event_id]["prediction"] or None for event_id in event_ids],
            labels,
        )
        paired = paired_accuracy_delta_ci(
            [baseline[event_id]["target"] for event_id in event_ids],
            [adapted[event_id]["prediction"] or None for event_id in event_ids],
            [baseline[event_id]["prediction"] or None for event_id in event_ids],
            samples=args.bootstrap_samples,
            seed=seed,
        )
        summaries.append({
            "support_k": support_k,
            "method": method,
            "n": len(rows),
            **metrics,
            "accuracy_minus_baseline": paired["delta"],
            "accuracy_minus_baseline_ci_low": paired["ci_low"],
            "accuracy_minus_baseline_ci_high": paired["ci_high"],
        })

    result = {
        "baseline_condition": args.baseline_condition,
        "baseline_n": len(baseline_rows),
        "baseline": baseline_metrics,
        "adapted": summaries,
        "query_label_free": True,
        "selection_note": "eta must be selected on validation before test summarization",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
