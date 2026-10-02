#!/usr/bin/env python3
"""Summarize prompt-order, audio-prior, and RMS-matched MarmAudio controls."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from animal_omni.metrics import classification_metrics
from animal_omni.bootstrap import paired_accuracy_delta_ci


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metrics(rows: list[dict[str, str]], labels: list[str]) -> dict:
    result = classification_metrics(
        [row["target"] for row in rows],
        [row["prediction"] or None for row in rows],
        labels,
    )
    result["n"] = len(rows)
    result["prediction_counts"] = dict(Counter(row["prediction"] or "<invalid>" for row in rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--canonical-controls", type=Path, required=True)
    parser.add_argument("--reversed", type=Path, required=True)
    parser.add_argument("--permuted", type=Path, required=True)
    parser.add_argument("--authoritative-frequency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = yaml.safe_load(args.config.read_text())["dataset"]["labels"]
    controls = read(args.canonical_controls)
    grouped = defaultdict(list)
    for row in controls:
        grouped[row["condition"]].append(row)
    canonical = {row["event_id"]: row for row in grouped["full_0-8k"]}
    prompt_results = {"canonical": metrics(grouped["full_0-8k"], labels)}
    for name, path in (("reversed", args.reversed), ("seeded_permutation", args.permuted)):
        rows = read(path)
        prompt_results[name] = metrics(rows, labels)
        prompt_results[name]["agreement_with_canonical"] = sum(
            row["prediction"] == canonical[row["event_id"]]["prediction"] for row in rows
        ) / len(rows)
        prompt_results[name]["accuracy_minus_canonical"] = paired_accuracy_delta_ci(
            [row["target"] for row in rows],
            [row["prediction"] or None for row in rows],
            [canonical[row["event_id"]]["prediction"] or None for row in rows],
            seed=20250813,
        )

    manifest_rows = read(args.control_manifest)
    source_event = {
        row["event_id"]: row["control_source_event_id"]
        for row in manifest_rows if row["condition"] == "shuffled_audio"
    }
    core_label = {
        row["event_id"]: row["label"]
        for row in manifest_rows if row["condition"] == "full_0-8k"
    }
    shuffled = grouped["shuffled_audio"]
    shuffled_source_accuracy = sum(
        row["prediction"] == core_label[source_event[row["event_id"]]] for row in shuffled
    ) / len(shuffled)

    core_ids = set(canonical)
    authoritative = [row for row in read(args.authoritative_frequency) if row["event_id"] in core_ids]
    authoritative_by_condition = defaultdict(list)
    for row in authoritative:
        authoritative_by_condition[row["condition"]].append(row)
    rms = {}
    for cutoff in (1000, 2000, 4000):
        original_condition = f"lp_0-{cutoff}"
        matched_condition = f"{original_condition}_rms_matched"
        original = metrics(authoritative_by_condition[original_condition], labels)
        matched = metrics(grouped[matched_condition], labels)
        original_by_event = {
            row["event_id"]: row for row in authoritative_by_condition[original_condition]
        }
        matched_rows = grouped[matched_condition]
        rms[str(cutoff)] = {
            "original_lowpass": original,
            "rms_matched_lowpass": matched,
            "matched_minus_original_accuracy": matched["accuracy"] - original["accuracy"],
            "matched_minus_original_accuracy_paired_ci": paired_accuracy_delta_ci(
                [row["target"] for row in matched_rows],
                [row["prediction"] or None for row in matched_rows],
                [original_by_event[row["event_id"]]["prediction"] or None for row in matched_rows],
                seed=20250813,
            ),
        }
    result = {
        "prompt_order": prompt_results,
        "audio_controls": {
            "shuffled_target": metrics(shuffled, labels),
            "shuffled_source_label_accuracy": shuffled_source_accuracy,
            "silence": metrics(grouped["silence"], labels),
        },
        "rms_matching": rms,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
