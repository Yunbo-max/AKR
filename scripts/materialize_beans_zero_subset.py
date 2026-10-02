#!/usr/bin/env python3
"""Stream and materialize balanced BEANS-Zero component subsets."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from scipy.signal import resample_poly


DEFAULT_TASKS = [
    "call-type", "zf-indiv", "watkins",
    "unseen-species-cmn", "unseen-species-sci", "unseen-species-tax",
    "unseen-genus-cmn", "unseen-genus-sci", "unseen-genus-tax",
    "unseen-family-cmn", "unseen-family-sci", "unseen-family-tax",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=25,
                        help="maximum examples per task; 0 scans every matching example")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--target-rate", type=int, default=None)
    parser.add_argument("--duration-cap", type=float, default=None)
    parser.add_argument("--minimum-duration", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()
    rows = []
    if args.resume and args.output_manifest.exists():
        rows = list(csv.DictReader(args.output_manifest.open(newline="", encoding="utf-8")))
    complete = {row["event_id"] for row in rows}
    counts = Counter(row["dataset_name"] for row in rows)

    def checkpoint() -> None:
        if not rows:
            return
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_manifest.with_suffix(args.output_manifest.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        temporary.replace(args.output_manifest)

    stream = load_dataset("EarthSpeciesProject/BEANS-Zero", "BEANS-Zero",
                          split="test", streaming=True)
    for sample in stream:
        task = sample["dataset_name"]
        event_id = str(sample["id"])
        if task not in args.tasks or event_id in complete:
            continue
        if args.per_task and counts[task] >= args.per_task:
            continue
        metadata = json.loads(sample["metadata"])
        sample_rate = int(metadata.get("sample_rate", metadata.get("sampling_rate")))
        output = args.output_dir / task / f"{event_id}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        audio = np.asarray(sample["audio"], dtype=np.float32)
        if args.target_rate is not None and sample_rate != args.target_rate:
            divisor = gcd(sample_rate, args.target_rate)
            audio = resample_poly(audio, args.target_rate // divisor,
                                  sample_rate // divisor).astype(np.float32)
            sample_rate = args.target_rate
        if args.duration_cap is not None:
            audio = audio[:round(args.duration_cap * sample_rate)]
        if args.minimum_duration is not None:
            minimum = round(args.minimum_duration * sample_rate)
            audio = np.pad(audio, (0, max(0, minimum - len(audio))))
        sf.write(output, audio, sample_rate, subtype="FLOAT")
        rows.append({
            "event_id": event_id, "audio_path": str(output.resolve()),
            "label": sample["output"], "instruction_text": sample["instruction_text"],
            "dataset_name": task, "source_dataset": sample["source_dataset"],
            "license": sample["license"], "file_name": sample["file_name"],
            "metadata": sample["metadata"], "split": "test",
            "processed_sample_rate": sample_rate,
            "processed_duration": len(audio) / sample_rate,
            "duration_protocol": (
                f"target_rate={args.target_rate};cap={args.duration_cap};"
                f"minimum={args.minimum_duration}"
            ),
        })
        complete.add(event_id)
        counts[task] += 1
        if len(rows) % args.checkpoint_every == 0:
            checkpoint()
            print(f"materialized={len(rows)} counts={dict(counts)}", flush=True)
        if args.per_task and all(counts[task] >= args.per_task for task in args.tasks):
            break
    checkpoint()
    print(dict(counts), flush=True)
    # pyarrow streaming can abort during interpreter teardown in this pinned
    # environment; all files are flushed, so bypass only that buggy destructor.
    os._exit(0)


if __name__ == "__main__": main()
