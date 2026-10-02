#!/usr/bin/env python3
"""Build balanced prompt-prior, shuffled-audio, silence, and RMS controls."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20250813)
    parser.add_argument("--rms-cutoffs", type=int, nargs="+", default=[1000, 2000, 4000])
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_event = defaultdict(dict)
    for row in rows:
        by_event[row["event_id"]][row["condition"]] = row
    full = [conditions["full_0-8k"] for conditions in by_event.values()]
    by_label = defaultdict(list)
    for row in full:
        by_label[row["label"]].append(row)
    rng = np.random.default_rng(args.seed)
    core = []
    for label in sorted(by_label):
        candidates = sorted(by_label[label], key=lambda row: row["event_id"])
        indices = rng.permutation(len(candidates))[:args.per_class]
        core.extend(candidates[index] for index in indices)
    core.sort(key=lambda row: (row["label"], row["event_id"]))
    write_manifest(args.core_manifest, core)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(by_label)
    core_by_label = {label: [row for row in core if row["label"] == label] for label in labels}
    controls = [
        {**row, "control_source_event_id": row["event_id"]}
        for row in core
    ]
    rms_gains = defaultdict(list)
    for label_index, label in enumerate(labels):
        source_label = labels[(label_index + 1) % len(labels)]
        for item_index, row in enumerate(core_by_label[label]):
            source = core_by_label[source_label][item_index]
            controls.append({
                **row,
                "audio_path": source["audio_path"],
                "condition": "shuffled_audio",
                "control_source_event_id": source["event_id"],
            })

            full_audio, sample_rate = sf.read(row["audio_path"], dtype="float32")
            silence_path = args.output_dir / "silence" / f'{row["event_id"]}.wav'
            silence_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(silence_path, np.zeros_like(full_audio), sample_rate, subtype="FLOAT")
            controls.append({
                **row,
                "audio_path": str(silence_path.resolve()),
                "condition": "silence",
                "control_source_event_id": row["event_id"],
            })

            full_rms = float(np.sqrt(np.mean(np.square(full_audio, dtype=np.float64))))
            for cutoff in args.rms_cutoffs:
                source_condition = f"lp_0-{cutoff}"
                degraded_row = by_event[row["event_id"]][source_condition]
                degraded, degraded_rate = sf.read(degraded_row["audio_path"], dtype="float32")
                if degraded_rate != sample_rate or degraded.shape != full_audio.shape:
                    raise ValueError(f"paired shape/rate mismatch for {row['event_id']} {source_condition}")
                degraded_rms = float(np.sqrt(np.mean(np.square(degraded, dtype=np.float64))))
                gain = full_rms / degraded_rms if degraded_rms > 1e-10 else 0.0
                matched = degraded * np.float32(gain)
                condition = f"{source_condition}_rms_matched"
                output = args.output_dir / condition / f'{row["event_id"]}.wav'
                output.parent.mkdir(parents=True, exist_ok=True)
                sf.write(output, matched, sample_rate, subtype="FLOAT")
                controls.append({
                    **row,
                    "audio_path": str(output.resolve()),
                    "condition": condition,
                    "control_source_event_id": row["event_id"],
                })
                rms_gains[condition].append(gain)

    controls.sort(key=lambda row: (row["event_id"], row["condition"]))
    write_manifest(args.control_manifest, controls)
    summary = {
        "seed": args.seed,
        "per_class": args.per_class,
        "n_core": len(core),
        "n_controls": len(controls),
        "labels": labels,
        "audio_shuffle": "cyclic next-label pairing at matched within-label index",
        "rms_gain": {
            condition: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for condition, values in rms_gains.items()
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
