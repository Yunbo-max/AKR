#!/usr/bin/env python3
"""Build a source-condition by target-condition decoder-transfer matrix.

For every source frequency condition, layer and ridge strength are selected on
that source's validation set only.  The decoder is then refit on the source
train+validation split and evaluated unchanged on the paired test examples from
all target conditions.  No target-condition label is used for selection or fit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fixed_split_probe_cross_condition import CONDITIONS, directories, load_layer


def summary_path(results: Path, dataset: str, condition: str) -> Path:
    if condition == "full_0-8k":
        return results / f"beans_{dataset}_probe_7b_summary.json"
    cutoff = int(condition.removeprefix("lp_0-")) // 1000
    return results / f"beans_{dataset}_probe_lp{cutoff}_7b_summary.json"


def short(condition: str) -> str:
    if condition == "full_0-8k":
        return "full"
    return f"{int(condition.removeprefix('lp_0-')) // 1000}k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", choices=("dogs", "watkins"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    results = args.root.resolve() / "results"
    condition_dirs = directories(results, args.dataset)
    rows: list[dict[str, object]] = []
    matrix = np.empty((len(CONDITIONS), len(CONDITIONS)), dtype=float)
    selections = {}
    canonical_targets = None

    for source_index, source in enumerate(CONDITIONS):
        selected = json.loads(summary_path(results, args.dataset, source).read_text())
        layer = int(selected["selected_layer"])
        alpha = float(selected["alpha"])
        selections[source] = {
            "selected_layer": layer,
            "alpha": alpha,
            "validation_macro_f1": float(selected["validation_macro_f1"]),
        }
        source_records = load_layer(condition_dirs[source], layer)
        fit = [record for record in source_records if record[2] in {"train", "valid"}]
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha)).fit(
            np.stack([record[3] for record in fit]),
            np.asarray([record[1] for record in fit]),
        )
        for target_index, target in enumerate(CONDITIONS):
            test = load_layer(condition_dirs[target], layer, split="test")
            event_ids = [record[0] for record in test]
            targets = [record[1] for record in test]
            event_targets = dict(zip(event_ids, targets))
            if len(event_targets) != len(test):
                raise RuntimeError(f"duplicate test examples for {target}")
            if canonical_targets is None:
                canonical_targets = event_targets
            elif event_targets != canonical_targets:
                raise RuntimeError(f"unpaired test examples for {target}")
            predictions = model.predict(np.stack([record[3] for record in test]))
            accuracy = float(accuracy_score(targets, predictions))
            macro_f1 = float(f1_score(
                targets, predictions, average="macro", zero_division=0
            ))
            matrix[source_index, target_index] = accuracy
            rows.append({
                "selection_mode": "source_selected",
                "source_condition": source,
                "target_condition": target,
                "selected_layer": layer,
                "ridge_alpha": alpha,
                "n_fit": len(fit),
                "n_test": len(test),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            })

    # The diagonal independently reproduces each registered condition-specific
    # probe.  Failing this assertion signals a representation/protocol mismatch.
    diagonal_checks = {}
    for index, condition in enumerate(CONDITIONS):
        registered = json.loads(summary_path(results, args.dataset, condition).read_text())
        expected = float(registered["test_accuracy"])
        observed = float(matrix[index, index])
        diagonal_checks[condition] = {
            "matrix_accuracy": observed,
            "registered_accuracy": expected,
            "absolute_difference": abs(observed - expected),
        }
        if not np.isclose(observed, expected, atol=1e-12):
            raise RuntimeError(
                f"diagonal mismatch for {condition}: {observed} != {expected}"
            )

    # Matched control: hold the representation level and ridge strength fixed at
    # the full-input validation choice for every source condition.  Only the
    # source-condition fitting examples change.
    full_selected = json.loads(
        summary_path(results, args.dataset, "full_0-8k").read_text()
    )
    shared_layer = int(full_selected["selected_layer"])
    shared_alpha = float(full_selected["alpha"])
    shared_matrix = np.empty_like(matrix)
    for source_index, source in enumerate(CONDITIONS):
        source_records = load_layer(condition_dirs[source], shared_layer)
        fit = [record for record in source_records if record[2] in {"train", "valid"}]
        model = make_pipeline(
            StandardScaler(), RidgeClassifier(alpha=shared_alpha)
        ).fit(
            np.stack([record[3] for record in fit]),
            np.asarray([record[1] for record in fit]),
        )
        for target_index, target in enumerate(CONDITIONS):
            test = load_layer(condition_dirs[target], shared_layer, split="test")
            targets = [record[1] for record in test]
            predictions = model.predict(np.stack([record[3] for record in test]))
            accuracy = float(accuracy_score(targets, predictions))
            macro_f1 = float(f1_score(
                targets, predictions, average="macro", zero_division=0
            ))
            shared_matrix[source_index, target_index] = accuracy
            rows.append({
                "selection_mode": "full_shared",
                "source_condition": source,
                "target_condition": target,
                "selected_layer": shared_layer,
                "ridge_alpha": shared_alpha,
                "n_fit": len(fit),
                "n_test": len(test),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "dataset": args.dataset,
        "protocol": (
            "for each source condition select layer/alpha on source validation; "
            "refit source train+valid; transfer unchanged to every paired target test"
        ),
        "conditions": list(CONDITIONS),
        "n_test": len(canonical_targets or {}),
        "source_selections": selections,
        "accuracy_matrix": matrix.tolist(),
        "source_selected_accuracy_matrix": matrix.tolist(),
        "diagonal_reproduction": diagonal_checks,
        "full_shared_hyperparameters": {
            "selected_layer": shared_layer,
            "alpha": shared_alpha,
        },
        "full_shared_accuracy_matrix": shared_matrix.tolist(),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.55), constrained_layout=True)
    labels = [short(condition) for condition in CONDITIONS]
    images = []
    for axis, values, title in zip(
        axes,
        (matrix, shared_matrix),
        ("source-selected readout", "full-selected layer/alpha fixed"),
    ):
        images.append(axis.imshow(values * 100, vmin=0, vmax=100, cmap="viridis"))
        axis.set_xticks(range(len(labels)), labels)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("target test condition")
        axis.set_title(title)
        for row_index in range(len(CONDITIONS)):
            for column_index in range(len(CONDITIONS)):
                value = 100 * values[row_index, column_index]
                color = "white" if value < 52 else "black"
                axis.text(column_index, row_index, f"{value:.1f}", ha="center",
                          va="center", color=color, fontsize=7.5)
    axes[0].set_ylabel("source train/validation condition")
    fig.suptitle(f"{args.dataset.title()}: frozen decoder transfer (%)")
    fig.colorbar(images[-1], ax=axes, label="accuracy (%)", shrink=.88)
    fig.savefig(args.output_figure, dpi=240)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
