#!/usr/bin/env python3
"""Paired fixed-test comparisons for the full-train Dogs Thinker LoRA."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read(path: Path, condition: str | None = None) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if condition is not None:
        rows = [row for row in rows if row.get("condition") == condition]
    output = {row["event_id"]: row for row in rows}
    if len(output) != len(rows):
        raise RuntimeError(f"duplicate events in {path}")
    return output


def compare(
    name: str, lora: dict[str, dict[str, str]], reference: dict[str, dict[str, str]],
    rng: np.random.Generator, draws: int,
) -> dict:
    if set(lora) != set(reference):
        raise RuntimeError(f"{name} event mismatch")
    events = sorted(lora)
    if any(lora[event]["target"] != reference[event]["target"] for event in events):
        raise RuntimeError(f"{name} target mismatch")
    a = np.asarray([lora[event]["correct"] == "true" for event in events])
    b = np.asarray([
        reference[event].get("correct", str(
            reference[event]["prediction"] == reference[event]["target"]
        ).lower()) == "true"
        for event in events
    ])
    delta = a.astype(float) - b.astype(float)
    indices = rng.integers(0, len(events), size=(draws, len(events)))
    bootstrap = delta[indices].mean(1)
    a_only = int(np.sum(a & ~b)); b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    return {
        "reference": name, "n": len(events),
        "lora_accuracy": float(a.mean()), "reference_accuracy": float(b.mean()),
        "lora_minus_reference": float(delta.mean()),
        "paired_bootstrap_95_ci": [
            float(value) for value in np.quantile(bootstrap, [.025, .975])
        ],
        "lora_only_correct": a_only, "reference_only_correct": b_only,
        "mcnemar_exact_two_sided_p": (
            float(binomtest(a_only, discordant, .5).pvalue) if discordant else 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = args.root.resolve() / "results"
    lora = read(results / "beans_dogs_lora_full_7b.csv")
    references = {
        "native_generation": read(
            results / "beans_dogs_frequency_qwen7b.csv", "full_0-8k"
        ),
        "frozen_qwen_probe": read(results / "beans_dogs_probe_7b_test.csv"),
        "aves_bio_probe": read(results / "predictions_beans_dogs_aves_bio_fixed.csv"),
    }
    rng = np.random.default_rng(args.seed)
    output = {
        "protocol": "paired official Dogs fixed test; LoRA trained on all 415 official train examples; test generated once after the locked one-epoch adapter",
        "bootstrap_draws": args.draws, "seed": args.seed,
        "comparisons": [
            compare(name, lora, reference, rng, args.draws)
            for name, reference in references.items()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
