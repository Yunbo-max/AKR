#!/usr/bin/env python3
"""Summarize the fully counterbalanced MarmAudio A--F candidate panel."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read_cell(path: Path, k: int) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if int(row["support_k_per_class"]) == k]
    result = {row["event_id"]: row for row in rows}
    if len(rows) != 75 or len(result) != 75:
        raise RuntimeError(f"expected 75 unique K={k} rows in {path}, got {len(rows)}")
    return result


def correct(row: dict[str, str]) -> bool:
    return row["prediction"] == row["target_output"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [("identity", args.root / "results/marmaudio_arbitrary_labels_candidate_k0_k1_7b.csv")]
    files += [
        (f"shift{shift}", args.root / f"results/marmaudio_arbitrary_labels_candidate_shift{shift}_k0_k1_7b.csv")
        for shift in range(1, 6)
    ]
    mappings = []
    delta_by_event: dict[str, list[float]] = {}
    association_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for name, path in files:
        k0, k1 = read_cell(path, 0), read_cell(path, 1)
        if set(k0) != set(k1):
            raise RuntimeError(f"K0/K1 event mismatch for {name}")
        events = sorted(k0)
        c0 = np.asarray([correct(k0[event]) for event in events])
        c1 = np.asarray([correct(k1[event]) for event in events])
        association_arrays.append((
            np.asarray([k0[event]["prediction"] for event in events]),
            np.asarray([k1[event]["prediction"] for event in events]),
            np.asarray([k1[event]["target_output"] for event in events]),
        ))
        a_only = int(np.sum(c1 & ~c0))
        b_only = int(np.sum(~c1 & c0))
        discordant = a_only + b_only
        for index, event in enumerate(events):
            delta_by_event.setdefault(event, []).append(float(c1[index]) - float(c0[index]))
        mappings.append({
            "mapping": name,
            "k0_accuracy": float(c0.mean()),
            "k1_accuracy": float(c1.mean()),
            "k1_minus_k0": float((c1.astype(float) - c0.astype(float)).mean()),
            "k1_only_correct": a_only,
            "k0_only_correct": b_only,
            "mcnemar_exact_two_sided_p": (
                float(binomtest(a_only, discordant, .5).pvalue) if discordant else 1.0
            ),
            "k0_prediction_counts": dict(Counter(row["prediction"] for row in k0.values())),
            "k1_prediction_counts": dict(Counter(row["prediction"] for row in k1.values())),
        })
    if any(len(values) != 6 for values in delta_by_event.values()):
        raise RuntimeError("each query must occur in all six mappings")
    event_delta = np.asarray([np.mean(delta_by_event[event]) for event in sorted(delta_by_event)])
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(event_delta), size=(args.draws, len(event_delta)))
    bootstrap = event_delta[indices].mean(1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(args.draws, len(event_delta)))
    signflip = (signs * event_delta).mean(1)
    observed = float(event_delta.mean())
    # Use one shared query permutation across all mappings per draw. This keeps
    # the repeated-query structure and every mapping's output prior intact while
    # testing whether predictions are attached to the correct acoustic query.
    permutations = np.argsort(
        rng.random((args.draws, len(event_delta))), axis=1
    )
    k0_null = np.zeros(args.draws, dtype=float)
    k1_null = np.zeros(args.draws, dtype=float)
    for k0_predictions, k1_predictions, targets in association_arrays:
        k0_null += (k0_predictions[permutations] == targets[None, :]).mean(axis=1)
        k1_null += (k1_predictions[permutations] == targets[None, :]).mean(axis=1)
    k0_null /= len(association_arrays)
    k1_null /= len(association_arrays)
    observed_k0 = float(np.mean([row["k0_accuracy"] for row in mappings]))
    observed_k1 = float(np.mean([row["k1_accuracy"] for row in mappings]))
    payload = {
        "design": "six cyclic A--F mappings; every acoustic class is paired once with every letter",
        "n_mappings": 6,
        "n_unique_queries": len(event_delta),
        "n_query_mapping_pairs": len(event_delta) * 6,
        "mappings": mappings,
        "macro_k0_accuracy": observed_k0,
        "macro_k1_accuracy": observed_k1,
        "clustered_k1_minus_k0": observed,
        "query_cluster_bootstrap_95_ci": [
            float(value) for value in np.quantile(bootstrap, [.025, .975])
        ],
        "query_cluster_signflip_two_sided_p": float(
            (1 + np.sum(np.abs(signflip) >= abs(observed))) / (args.draws + 1)
        ),
        "audio_label_association_null": {
            "procedure": "shared query permutation across six mappings; output frequencies fixed",
            "k0_null_95_interval": [
                float(value) for value in np.quantile(k0_null, [.025, .975])
            ],
            "k0_one_sided_p": float(
                (1 + np.sum(k0_null >= observed_k0)) / (args.draws + 1)
            ),
            "k1_null_95_interval": [
                float(value) for value in np.quantile(k1_null, [.025, .975])
            ],
            "k1_one_sided_p": float(
                (1 + np.sum(k1_null >= observed_k1)) / (args.draws + 1)
            ),
        },
        "draws": args.draws,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
