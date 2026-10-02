#!/usr/bin/env python3
"""Summarize the preregistered factorized token-by-feature KV validation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from animal_omni.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], float(row["relative_alpha"])), {})[
            row["event_id"]
        ] = row
    if any(len(cell) != args.expected_n for cell in grouped.values()):
        raise RuntimeError("all factorized method/alpha cells must be complete")
    labels = list("ABCDEFGHIJ")
    cells = []
    correctness = {}
    for (method, alpha), mapped in sorted(grouped.items()):
        events = sorted(mapped)
        targets = [mapped[event]["target_output"] for event in events]
        predictions = [mapped[event]["prediction"] or None for event in events]
        metric = classification_metrics(targets, predictions, labels)
        metric.update({
            "method": method, "relative_alpha": alpha, "n": len(events),
            "invalid_response_rate": float(np.mean([not mapped[event]["prediction"] for event in events])),
            "prediction_counts": dict(Counter(mapped[event]["prediction"] for event in events)),
            "applied_ratio_mean": float(np.mean([float(mapped[event]["applied_ratio_mean"]) for event in events])),
        })
        cells.append(metric)
        correctness[(method, alpha)] = (events, np.asarray([
            mapped[event]["prediction"] == mapped[event]["target_output"] for event in events
        ]))

    rng = np.random.default_rng(args.seed)
    comparisons = []
    factorized = sorted({method for method, _ in grouped if "factorized" in method})
    alphas = sorted({alpha for _, alpha in grouped})

    def compare(method_a: str, method_b: str, alpha: float) -> dict:
        events_a, a = correctness[(method_a, alpha)]
        events_b, b = correctness[(method_b, alpha)]
        if events_a != events_b:
            raise RuntimeError("paired event IDs differ")
        delta = a.astype(float) - b.astype(float)
        indices = rng.integers(0, len(a), size=(args.draws, len(a)))
        bootstrap = delta[indices].mean(1)
        a_only = int(np.sum(a & ~b)); b_only = int(np.sum(~a & b))
        discordant = a_only + b_only
        return {
            "method_a": method_a, "method_b": method_b, "relative_alpha": alpha,
            "accuracy_a": float(a.mean()), "accuracy_b": float(b.mean()),
            "paired_difference": float(delta.mean()),
            "paired_bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
            "a_only_correct": a_only, "b_only_correct": b_only,
            "mcnemar_exact_two_sided_p": (
                float(binomtest(a_only, discordant, .5).pvalue) if discordant else 1.0
            ),
        }

    for method in factorized:
        for alpha in alphas:
            comparisons.append(compare(method, "probe_class_pooled", alpha))
            comparisons.append(compare(method, "probe_class_tokenwise", alpha))

    cell_lookup = {(cell["method"], cell["relative_alpha"]): cell for cell in cells}
    stable_ranks = []
    for method in factorized:
        beats_pooled = [alpha for alpha in alphas if
                        cell_lookup[(method, alpha)]["accuracy"] >
                        cell_lookup[("probe_class_pooled", alpha)]["accuracy"]]
        beats_tokenwise = [alpha for alpha in alphas if
                           cell_lookup[(method, alpha)]["accuracy"] >
                           cell_lookup[("probe_class_tokenwise", alpha)]["accuracy"]]
        low_invalid = [alpha for alpha in alphas if
                       cell_lookup[(method, alpha)]["invalid_response_rate"] < .10]
        if len(beats_pooled) >= 2 and len(beats_tokenwise) >= 2 and len(low_invalid) == len(alphas):
            stable_ranks.append(method)
    payload = {
        "protocol": "Dogs A-J official validation only; same K=2/class support/router; per-layer relative Frobenius norm matched",
        "registered_gate": "one fixed rank strictly beats pooled and full tokenwise at >=2/3 alphas and has <10% invalid at all alphas",
        "gate_passed": bool(stable_ranks), "passing_methods": stable_ranks,
        "test_action": "allowed_once" if stable_ranks else "no_test",
        "cells": cells, "paired_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
