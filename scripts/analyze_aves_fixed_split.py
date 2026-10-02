#!/usr/bin/env python3
"""Paired AVES-bio versus frozen-Qwen probe analysis on official BEANS tests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["event_id"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate events in {path}")
    return result


def compare(
    dataset: str,
    qwen: dict[str, dict[str, str]],
    aves: dict[str, dict[str, str]],
    rng: np.random.Generator,
    draws: int,
) -> dict:
    if set(qwen) != set(aves):
        raise RuntimeError(f"{dataset} event mismatch: Qwen={len(qwen)} AVES={len(aves)}")
    events = sorted(qwen)
    if any(qwen[event]["target"] != aves[event]["target"] for event in events):
        raise RuntimeError(f"{dataset} target mismatch")
    qwen_correct = np.asarray([
        qwen[event]["prediction"] == qwen[event]["target"] for event in events
    ])
    aves_correct = np.asarray([
        aves[event]["prediction"] == aves[event]["target"] for event in events
    ])
    delta = qwen_correct.astype(float) - aves_correct.astype(float)
    indices = rng.integers(0, len(events), size=(draws, len(events)))
    bootstrap = delta[indices].mean(1)
    qwen_only = int(np.sum(qwen_correct & ~aves_correct))
    aves_only = int(np.sum(~qwen_correct & aves_correct))
    discordant = qwen_only + aves_only
    return {
        "dataset": dataset,
        "n": len(events),
        "qwen_probe_accuracy": float(qwen_correct.mean()),
        "aves_bio_probe_accuracy": float(aves_correct.mean()),
        "qwen_minus_aves": float(delta.mean()),
        "paired_bootstrap_95_ci": [
            float(value) for value in np.quantile(bootstrap, [.025, .975])
        ],
        "qwen_only_correct": qwen_only,
        "aves_only_correct": aves_only,
        "mcnemar_exact_two_sided_p": (
            float(binomtest(qwen_only, discordant, .5).pvalue) if discordant else 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    comparisons = []
    for dataset in ("dogs", "watkins"):
        comparisons.append(compare(
            dataset,
            read(args.root / f"results/beans_{dataset}_probe_7b_test.csv"),
            read(args.root / f"results/predictions_beans_{dataset}_aves_bio_fixed.csv"),
            rng,
            args.draws,
        ))
    payload = {
        "protocol": "paired official fixed test; no test-label model selection",
        "bootstrap_draws": args.draws,
        "seed": args.seed,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
