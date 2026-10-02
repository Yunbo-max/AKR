#!/usr/bin/env python3
"""Combine full official fixed-split probe results across low-pass cutoffs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


CONDITIONS = ["full_0-8k", "lp_0-1000", "lp_0-2000", "lp_0-4000", "lp_0-6000", "lp_0-8000"]


def read_correctness(path: Path, condition: str | None = None) -> dict[str, tuple[str, bool]]:
    rows: dict[str, tuple[str, bool]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if condition is not None and row.get("condition") != condition:
                continue
            event_id = row["event_id"]
            if event_id in rows:
                raise RuntimeError(f"duplicate event {event_id} in {path} ({condition})")
            rows[event_id] = (row["target"], row["correct"].strip().lower() == "true")
    return rows


def paired_delta(
    left: dict[str, tuple[str, bool]],
    right: dict[str, tuple[str, bool]],
    *,
    seed: int,
    draws: int = 10_000,
) -> dict[str, float | int]:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:5]
        missing_right = sorted(set(left) - set(right))[:5]
        raise RuntimeError(
            f"paired event mismatch: missing_left={missing_left}, missing_right={missing_right}"
        )
    event_ids = sorted(left)
    for event_id in event_ids:
        if left[event_id][0] != right[event_id][0]:
            raise RuntimeError(f"target mismatch for {event_id}")
    left_correct = np.asarray([left[event_id][1] for event_id in event_ids], dtype=float)
    right_correct = np.asarray([right[event_id][1] for event_id in event_ids], dtype=float)
    difference = left_correct - right_correct
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=float)
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        indices = rng.integers(0, len(event_ids), size=(stop - start, len(event_ids)))
        bootstrap[start:stop] = difference[indices].mean(axis=1)
    left_only = int(np.sum((left_correct == 1) & (right_correct == 0)))
    right_only = int(np.sum((left_correct == 0) & (right_correct == 1)))
    discordant = left_only + right_only
    if discordant == 0:
        mcnemar_p = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        mcnemar_p = min(1.0, 2.0 * tail / (2.0 ** discordant))
    return {
        "n": len(event_ids),
        "delta_accuracy": float(difference.mean()),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "mcnemar_exact_p": float(mcnemar_p),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", choices=["dogs", "watkins"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = args.root.resolve() / "results"
    paths = [results / f"beans_{args.dataset}_probe_7b_summary.json"] + [
        results / f"beans_{args.dataset}_probe_lp{cutoff}_7b_summary.json"
        for cutoff in (1, 2, 4, 6, 8)
    ]
    cells = []
    for condition, path in zip(CONDITIONS, paths):
        value = json.loads(path.read_text())
        cells.append({"condition": condition, **value})
    split_sizes = {(cell["n_train"], cell["n_valid"], cell["n_test"]) for cell in cells}
    if len(split_sizes) != 1:
        raise RuntimeError(f"frequency cells use different split sizes: {sorted(split_sizes)}")
    generation = json.loads(
        (results / f"beans_{args.dataset}_frequency_qwen7b_summary.json").read_text()
    )
    generation_by_condition = {
        row["condition"]: row for row in generation["conditions"]
    }
    native_predictions = results / f"beans_{args.dataset}_frequency_qwen7b.csv"
    probe_prediction_paths = [results / f"beans_{args.dataset}_probe_7b_test.csv"] + [
        results / f"beans_{args.dataset}_probe_lp{cutoff}_7b_test.csv"
        for cutoff in (1, 2, 4, 6, 8)
    ]
    full_probe = read_correctness(probe_prediction_paths[0])
    for index, (cell, prediction_path) in enumerate(zip(cells, probe_prediction_paths)):
        native = generation_by_condition[cell["condition"]]
        cell["native_generation_accuracy"] = native["accuracy"]
        cell["probe_minus_native_accuracy"] = cell["test_accuracy"] - native["accuracy"]
        probe_correctness = read_correctness(prediction_path)
        native_correctness = read_correctness(native_predictions, cell["condition"])
        cell["paired_probe_minus_native"] = paired_delta(
            probe_correctness,
            native_correctness,
            seed=20260814 + 100 * (1 if args.dataset == "watkins" else 0) + index,
        )
        cell["paired_full_probe_minus_condition_probe"] = paired_delta(
            full_probe,
            probe_correctness,
            seed=20260914 + 100 * (1 if args.dataset == "watkins" else 0) + index,
        )
    probe = np.asarray([cell["test_accuracy"] for cell in cells])
    native = np.asarray([cell["native_generation_accuracy"] for cell in cells])
    correlation = None if np.std(native) == 0 or np.std(probe) == 0 else float(
        np.corrcoef(probe, native)[0, 1]
    )
    output = {
        "dataset": f"BEANS {args.dataset.title()}",
        "protocol": "each frequency uses the complete official train/valid/test split; layer and ridge alpha selected only on that condition's validation split; refit train+valid; test once",
        "split_sizes": dict(zip(("train", "valid", "test"), next(iter(split_sizes)))),
        "conditions": cells,
        "probe_native_accuracy_correlation": correlation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
