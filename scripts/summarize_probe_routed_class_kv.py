#!/usr/bin/env python3
"""Summarize and apply the frozen class-dictionary validation gate."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from animal_omni.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], float(row["relative_alpha"]))].append(row)
    cells, incomplete = [], []
    for (method, alpha), selected in sorted(grouped.items()):
        if len(selected) != args.expected_n or len({row["event_id"] for row in selected}) != args.expected_n:
            incomplete.append({"method": method, "relative_alpha": alpha, "n": len(selected)})
            continue
        cell = {
            "method": method, "relative_alpha": alpha, "n": len(selected),
            **classification_metrics(
                [row.get("target_output") or row["target"] for row in selected],
                [row["prediction"] or None for row in selected], args.labels,
            ),
            "router_accuracy": sum(row["router_correct"] == "true" for row in selected) / len(selected),
            "prediction_matches_routed_label": sum(
                row["prediction_matches_routed_label"] == "true" for row in selected
            ) / len(selected),
            "invalid_response_rate": sum(not row["prediction"] for row in selected) / len(selected),
            "prediction_counts": dict(Counter(row["prediction"] for row in selected)),
            "achieved_ratio_mean": sum(float(row["applied_ratio_mean"])
                                       for row in selected) / len(selected),
        }
        cells.append(cell)
    lookup = {(cell["method"], cell["relative_alpha"]): cell for cell in cells}
    alphas = sorted({cell["relative_alpha"] for cell in cells})
    tokenwise_wins = [
        alpha for alpha in alphas
        if ("probe_class_tokenwise", alpha) in lookup
        and ("probe_class_pooled", alpha) in lookup
        and lookup[("probe_class_tokenwise", alpha)]["accuracy"] >
            lookup[("probe_class_pooled", alpha)]["accuracy"]
    ]
    tokenwise = [cell for cell in cells if cell["method"] == "probe_class_tokenwise"]
    selected = max(tokenwise, key=lambda cell: (
        cell["accuracy"], cell["macro_f1"], -cell["relative_alpha"]
    )) if tokenwise else None
    beats_permuted = bool(selected) and (
        ("probe_class_permuted", selected["relative_alpha"]) in lookup and
        selected["accuracy"] >
        lookup[("probe_class_permuted", selected["relative_alpha"])]["accuracy"]
    )
    gate = {
        "registered_rule": (
            "ordered tokenwise beats pooled at >=2 alphas and beats token-permuted "
            "at selected alpha; ties select lower norm"
        ),
        "tokenwise_beats_pooled_alphas": tokenwise_wins,
        "selected_tokenwise_alpha": selected["relative_alpha"] if selected else None,
        "selected_tokenwise_beats_permuted": beats_permuted,
        "passed": len(tokenwise_wins) >= 2 and beats_permuted,
        "test_action": "allowed_once" if len(tokenwise_wins) >= 2 and beats_permuted
                       else "no_test",
    }
    payload = {"complete_cells": cells, "incomplete_cells": incomplete, "gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
