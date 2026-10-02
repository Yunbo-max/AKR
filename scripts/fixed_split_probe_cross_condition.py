#!/usr/bin/env python3
"""Fit one full-input probe and transfer it unchanged across low-pass conditions."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONDITIONS = ("full_0-8k", "lp_0-1000", "lp_0-2000", "lp_0-4000", "lp_0-6000", "lp_0-8000")


def directories(results: Path, dataset: str) -> dict[str, Path]:
    return {
        "full_0-8k": results / f"reps_beans_{dataset}_7b",
        "lp_0-1000": results / f"reps_beans_{dataset}_lp1_7b",
        **{
            f"lp_0-{cutoff * 1000}": (
                results / f"reps_beans_{dataset}_frequency_extra_7b" / f"lp_0-{cutoff * 1000}"
            )
            for cutoff in (2, 4, 6, 8)
        },
    }


def load_layer(directory: Path, layer: int, split: str | None = None):
    records = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as record:
            record_split = str(record["split"])
            if split is not None and record_split != split:
                continue
            records.append((
                str(record["event_id"]), str(record["label"]), record_split,
                record["representation"][layer].astype(np.float32),
            ))
    if not records:
        raise RuntimeError(f"no representations in {directory} for split={split}")
    return records


def paired_correctness_delta(
    left: list[bool], right: list[bool], rng: np.random.Generator,
    draws: int = 10_000,
) -> dict[str, float | int]:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    difference = left_bool.astype(float) - right_bool.astype(float)
    bootstrap = np.empty(draws, dtype=float)
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        bootstrap[start:stop] = difference[indices].mean(axis=1)
    left_only = int(np.sum(left_bool & ~right_bool))
    right_only = int(np.sum(~left_bool & right_bool))
    discordant = left_only + right_only
    exact_p = 1.0 if discordant == 0 else min(
        1.0,
        2.0 * sum(math.comb(discordant, k)
                  for k in range(min(left_only, right_only) + 1)) / (2.0 ** discordant),
    )
    return {
        "n": len(difference), "delta": float(difference.mean()),
        "ci_low": float(np.quantile(bootstrap, .025)),
        "ci_high": float(np.quantile(bootstrap, .975)),
        "bootstrap_draws": draws,
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "mcnemar_exact_p": float(exact_p),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", choices=("dogs", "watkins"), required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    results = args.root.resolve() / "results"
    selected = json.loads(
        (results / f"beans_{args.dataset}_probe_7b_summary.json").read_text()
    )
    layer, alpha = int(selected["selected_layer"]), float(selected["alpha"])
    condition_dirs = directories(results, args.dataset)
    full = load_layer(condition_dirs["full_0-8k"], layer)
    train_valid = [record for record in full if record[2] in {"train", "valid"}]
    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha)).fit(
        np.stack([record[3] for record in train_valid]),
        np.asarray([record[1] for record in train_valid]),
    )
    output_rows = []
    correctness = {}
    condition_cells = []
    event_targets = None
    for condition in CONDITIONS:
        test = load_layer(condition_dirs[condition], layer, split="test")
        event_ids = [record[0] for record in test]
        targets = [record[1] for record in test]
        if len(event_ids) != len(set(event_ids)):
            raise RuntimeError(f"duplicate test events for {condition}")
        current_targets = dict(zip(event_ids, targets))
        if event_targets is None:
            event_targets = current_targets
        elif current_targets != event_targets:
            raise RuntimeError(f"test event/target mismatch for {condition}")
        predictions = model.predict(np.stack([record[3] for record in test]))
        correctness[condition] = {
            event: prediction == target
            for event, target, prediction in zip(event_ids, targets, predictions)
        }
        output_rows.extend({
            "event_id": event, "condition": condition, "target": target,
            "prediction": prediction, "correct": str(prediction == target).lower(),
        } for event, target, prediction in zip(event_ids, targets, predictions))
        condition_cells.append({
            "condition": condition, "n": len(test),
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        })
    rng = np.random.default_rng(20260814 + (1 if args.dataset == "watkins" else 0))
    paired = {}
    full_events = sorted(correctness["full_0-8k"])
    full_values = [correctness["full_0-8k"][event] for event in full_events]
    for condition in CONDITIONS[1:]:
        paired[condition] = paired_correctness_delta(
            full_values,
            [correctness[condition][event] for event in full_events],
            rng,
        )
    matched_minus_transfer = {}
    for condition, cutoff in zip(CONDITIONS, (None, 1, 2, 4, 6, 8)):
        matched_path = (
            results / f"beans_{args.dataset}_probe_7b_test.csv"
            if cutoff is None else
            results / f"beans_{args.dataset}_probe_lp{cutoff}_7b_test.csv"
        )
        with matched_path.open(newline="", encoding="utf-8") as handle:
            matched_rows = list(csv.DictReader(handle))
        matched = {
            row["event_id"]: row["correct"].lower() == "true" for row in matched_rows
        }
        if set(matched) != set(full_events):
            raise RuntimeError(f"unpaired matched probe for {condition}")
        matched_minus_transfer[condition] = paired_correctness_delta(
            [matched[event] for event in full_events],
            [correctness[condition][event] for event in full_events],
            rng,
        )
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.output_predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0])
        writer.writeheader(); writer.writerows(output_rows)
    payload = {
        "dataset": args.dataset,
        "protocol": "select layer/alpha on full validation; fit full train+valid once; transfer the unchanged decoder to every condition test",
        "selected_layer": layer, "alpha": alpha,
        "n_train_valid": len(train_valid),
        "conditions": condition_cells,
        "paired_full_minus_condition": paired,
        "paired_condition_specific_minus_fulltrained_transfer": matched_minus_transfer,
    }
    args.output_summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
