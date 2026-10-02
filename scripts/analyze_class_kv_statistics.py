#!/usr/bin/env python3
"""Paired uncertainty for a completed probe-routed class-KV validation run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    cells: dict[tuple[str, float], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["method"], float(row["relative_alpha"]))
        cells.setdefault(key, {})[row["event_id"]] = row
    if any(len(cell) != args.expected_n for cell in cells.values()):
        raise RuntimeError("all method/alpha cells must be complete before paired analysis")
    rng = np.random.default_rng(args.seed)

    def compare(alpha: float, method_a: str, method_b: str) -> dict:
        a, b = cells[(method_a, alpha)], cells[(method_b, alpha)]
        if set(a) != set(b):
            raise RuntimeError("paired event sets differ")
        events = sorted(a)

        def correct(row: dict[str, str]) -> bool:
            target = row.get("target_output") or row["target"]
            return row["prediction"] == target

        ac = np.asarray([correct(a[event]) for event in events])
        bc = np.asarray([correct(b[event]) for event in events])
        delta = ac.astype(float) - bc.astype(float)
        indices = rng.integers(0, len(events), size=(args.draws, len(events)))
        bootstrap = delta[indices].mean(1)
        a_only = int(np.sum(ac & ~bc)); b_only = int(np.sum(~ac & bc))
        discordant = a_only + b_only
        return {
            "relative_alpha": alpha,
            "method_a": method_a, "method_b": method_b,
            "n": len(events), "accuracy_a": float(ac.mean()),
            "accuracy_b": float(bc.mean()),
            "paired_difference_a_minus_b": float(delta.mean()),
            "bootstrap_95_ci": [
                float(value) for value in np.quantile(bootstrap, [.025, .975])
            ],
            "a_only_correct": a_only, "b_only_correct": b_only,
            "mcnemar_exact_two_sided_p": (
                float(binomtest(a_only, discordant, .5).pvalue)
                if discordant else 1.0
            ),
        }

    alphas = sorted(alpha for method, alpha in cells if method == "probe_class_tokenwise")
    comparisons = []
    for alpha in alphas:
        comparisons.extend([
            compare(alpha, "probe_class_tokenwise", "probe_class_pooled"),
            compare(alpha, "probe_class_tokenwise", "probe_class_permuted"),
            compare(alpha, "probe_class_pooled", "probe_class_permuted"),
        ])
    payload = {
        "predictions": str(args.predictions), "bootstrap_draws": args.draws,
        "seed": args.seed,
        "inference": "paired example bootstrap; exact two-sided discordant-pair binomial test",
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
