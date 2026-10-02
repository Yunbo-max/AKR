#!/usr/bin/env python3
"""Summarize fixed-split conditional-KV runs without pooling overlapping seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--eta", type=float, default=300.)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    manifest = {r["event_id"]: r for r in csv.DictReader(args.manifest.open())
                if r["condition"] == "lp_0-1000"}
    baseline = {r["event_id"]: r for r in csv.DictReader(args.baseline.open())
                if r["condition"] == "lp_0-1000"}
    rows = []
    for path in args.runs:
        predictions = list(csv.DictReader(path.open()))
        split = json.loads(path.with_name(path.stem + "_split.json").read_text())
        events = split["query_events"]
        target = [manifest[event]["label"] for event in events]
        record = {"seed": split["seed"], "n": len(events),
                  "baseline_accuracy": accuracy_score(target, [baseline[e]["prediction"] for e in events]),
                  "baseline_macro_f1": f1_score(target, [baseline[e]["prediction"] for e in events], average="macro", zero_division=0)}
        for method in ("fixed_mean", "conditional"):
            selected = {r["event_id"]: r["prediction"] for r in predictions
                        if r["method"] == method and float(r["eta"]) == args.eta}
            predicted = [selected[event] for event in events]
            record[f"{method}_accuracy"] = accuracy_score(target, predicted)
            record[f"{method}_macro_f1"] = f1_score(target, predicted, average="macro", zero_division=0)
        record["conditional_minus_baseline"] = record["conditional_accuracy"] - record["baseline_accuracy"]
        record["conditional_minus_fixed"] = record["conditional_accuracy"] - record["fixed_mean_accuracy"]
        rows.append(record)
    keys = [k for k in rows[0] if k not in {"seed", "n"}]
    aggregate = {key: {"mean": float(np.mean([r[key] for r in rows])),
                       "sample_sd": float(np.std([r[key] for r in rows], ddof=1))}
                 for key in keys}
    output = {"eta": args.eta, "protocol": "mean_and_sample_sd_across_recording_splits_not_pooled",
              "runs": rows, "aggregate": aggregate}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
