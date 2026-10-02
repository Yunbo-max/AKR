#!/usr/bin/env python3
"""Evaluate a saved Thinker LoRA adapter with deterministic generation.

The prediction CSV is checkpointed atomically after every query.  ``--resume``
skips completed event IDs, so an interrupted full fixed-split evaluation never
needs to regenerate (or silently discard) earlier decisions.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import yaml
from peft import PeftModel

from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    labels = config["dataset"]["labels"]
    rows = [
        row for row in csv.DictReader(
            args.manifest.open(newline="", encoding="utf-8")
        ) if row["split"] == args.split
    ]
    expected = {row["event_id"] for row in rows}
    output = list(csv.DictReader(args.output.open(newline="", encoding="utf-8"))) \
        if args.resume and args.output.exists() else []
    completed = {row["event_id"] for row in output}
    if not completed <= expected:
        raise RuntimeError("existing output contains events outside requested split")
    if len(completed) != len(output):
        raise RuntimeError("existing output contains duplicate event IDs")
    if completed == expected:
        print(f"evaluation already complete: {len(completed)}/{len(expected)}", flush=True)
        return

    runner = QwenThinkerRunner(args.model_id)
    logging.disable(logging.CRITICAL)
    runner.model.thinker = PeftModel.from_pretrained(runner.model.thinker, args.adapter)
    for index, row in enumerate(rows, 1):
        if row["event_id"] in completed:
            continue
        raw = runner.predict(
            row["audio_path"], config["evaluation"]["prompt"],
            max_new_tokens=config["evaluation"]["max_new_tokens"],
        )
        prediction = normalize_label(raw, labels)
        output.append({
            "event_id": row["event_id"],
            "split": row["split"],
            "target": row["label"],
            "raw_prediction": raw,
            "prediction": prediction or "",
            "correct": str(prediction == row["label"]).lower(),
        })
        completed.add(row["event_id"])
        atomic_csv(args.output, output)
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"[{len(completed)}/{len(rows)}] {raw!r}", flush=True)


if __name__ == "__main__":
    main()
