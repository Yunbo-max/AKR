#!/usr/bin/env python3
"""Paired summary for native versus transferred pooled KV across prompts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from animal_omni.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    output = {"protocol": "same A-J support gradients and router; only query prompt changes",
              "prompts": []}
    labels = list("ABCDEFGHIJ")
    for path in args.predictions:
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        by_method = {}
        for row in rows:
            by_method.setdefault(row["method"], {})[row["event_id"]] = row
        required = {"native", "probe_class_pooled"}
        if set(by_method) != required:
            raise RuntimeError(f"{path} methods={sorted(by_method)}")
        if any(len(cell) != args.expected_n for cell in by_method.values()):
            raise RuntimeError(f"incomplete prompt cells in {path}")
        events = sorted(by_method["native"])
        if set(events) != set(by_method["probe_class_pooled"]):
            raise RuntimeError(f"unpaired events in {path}")
        cells = {}
        correctness = {}
        for method, mapped in by_method.items():
            targets = [mapped[event]["target_output"] for event in events]
            predictions = [mapped[event]["prediction"] or None for event in events]
            cells[method] = classification_metrics(targets, predictions, labels)
            cells[method]["invalid_response_rate"] = float(np.mean([
                not mapped[event]["prediction"] for event in events
            ]))
            correctness[method] = np.asarray([
                mapped[event]["prediction"] == mapped[event]["target_output"]
                for event in events
            ])
        delta = correctness["probe_class_pooled"].astype(float) - correctness["native"].astype(float)
        indices = rng.integers(0, len(events), size=(args.draws, len(events)))
        bootstrap = delta[indices].mean(1)
        repair_only = int(np.sum(correctness["probe_class_pooled"] & ~correctness["native"]))
        native_only = int(np.sum(~correctness["probe_class_pooled"] & correctness["native"]))
        discordant = repair_only + native_only
        output["prompts"].append({
            "name": rows[0]["prompt_name"], "predictions": str(path), "n": len(events),
            "native": cells["native"], "pooled_alpha_03": cells["probe_class_pooled"],
            "paired_accuracy_gain": float(delta.mean()),
            "paired_bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
            "repair_only_correct": repair_only, "native_only_correct": native_only,
            "mcnemar_exact_two_sided_p": (
                float(binomtest(repair_only, discordant, .5).pvalue) if discordant else 1.0
            ),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
