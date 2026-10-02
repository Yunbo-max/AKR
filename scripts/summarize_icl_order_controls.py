#!/usr/bin/env python3
"""Summarize positional copying across counterbalanced audio-ICL orders."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import binomtest


SPECS = (
    ("Marm K2 blocked", "marmaudio_icl_order_control_k2_blocked_7b.csv",
     "marmaudio_icl_order_registry_seed20260814.json", 75),
    ("Marm K2 interleaved", "marmaudio_icl_order_control_k2_interleaved_7b.csv",
     "marmaudio_icl_order_registry_seed20260814.json", 75),
    ("Marm K8 interleaved", "marmaudio_icl_order_control_k8_interleaved_7b.csv",
     "marmaudio_icl_order_registry_seed20260814.json", 75),
    ("Dogs K2 interleaved", "beans_dogs_lp1_icl_order_control_k2_interleaved_7b.csv",
     "beans_dogs_lp1_icl_order_registry_seed20260814.json", 139),
    ("Watkins K1 interleaved", "beans_watkins_icl_order_control_k1_interleaved_7b.csv",
     "beans_watkins_icl_order_registry_seed20260814.json", 339),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    results = args.root.resolve() / "results"
    cells = []
    for name, csv_name, registry_name, expected in SPECS:
        path, registry_path = results / csv_name, results / registry_name
        if not path.exists() or not registry_path.exists():
            continue
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        if len(rows) != expected or len({row["event_id"] for row in rows}) != expected:
            continue
        registry = json.loads(registry_path.read_text())
        labels = registry["labels"]
        positions = []
        first_copies = 0
        last_copies = 0
        for row in rows:
            rotation = int(row["rotation"])
            order = labels[rotation:] + labels[:rotation]
            prediction = row["prediction"]
            first_copies += prediction == order[0]
            last_copies += prediction == order[-1]
            positions.append(order.index(prediction) if prediction in order else None)
        valid_positions = [position for position in positions if position is not None]
        position_counts = Counter(valid_positions)
        cells.append({
            "name": name,
            "n_query": expected,
            "n_labels": len(labels),
            "valid_prediction_rate": len(valid_positions) / expected,
            "first_support_class_copy_rate": first_copies / expected,
            "last_support_class_copy_rate": last_copies / expected,
            "first_copy_exact_binomial_p_vs_uniform_label_prior": float(
                binomtest(first_copies, expected, 1 / len(labels), alternative="greater").pvalue
            ),
            "last_copy_exact_binomial_p_vs_uniform_label_prior": float(
                binomtest(last_copies, expected, 1 / len(labels), alternative="greater").pvalue
            ),
            "prediction_position_counts": {
                str(index): position_counts.get(index, 0) for index in range(len(labels))
            },
        })
    payload = {
        "protocol": (
            "support class-order rotations are blocked within target class and "
            "approximately balanced globally; positions are post-hoc diagnostics"
        ),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if cells:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
        x = list(range(len(cells)))
        width = .35
        axis.bar(
            [value - width / 2 for value in x],
            [100 * cell["first_support_class_copy_rate"] for cell in cells],
            width, label="first support class",
        )
        axis.bar(
            [value + width / 2 for value in x],
            [100 * cell["last_support_class_copy_rate"] for cell in cells],
            width, label="last support class",
        )
        axis.set_xticks(x, [cell["name"] for cell in cells], rotation=18, ha="right")
        axis.set_ylabel("prediction-copy rate (%)")
        axis.set_title("Counterbalanced audio ICL exposes positional copying")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=.2)
        axis.legend(frameon=False)
        fig.savefig(args.figure, dpi=240)
        plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
