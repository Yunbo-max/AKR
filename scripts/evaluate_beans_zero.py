#!/usr/bin/env python3
"""Deterministic Thinker-only BEANS-Zero evaluation using official instructions."""
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

from animal_omni.qwen_runner import QwenThinkerRunner


def canonical(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def write(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    source = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    results = list(csv.DictReader(args.output.open())) if args.resume and args.output.exists() else []
    complete = {row["event_id"] for row in results}
    pending = [row for row in source if row["event_id"] not in complete]
    if not pending: print("all examples complete"); return
    runner = QwenThinkerRunner(args.model_id)
    # Qwen resets the root logger during model construction and otherwise emits
    # the same Talker-related warning once per Thinker-only example. Silence it
    # after construction; this does not alter prompts, decoding, or model state.
    logging.disable(logging.CRITICAL)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(pending, 1):
        raw = runner.predict(row["audio_path"], row["instruction_text"],
                             max_new_tokens=args.max_new_tokens)
        prediction, target = canonical(raw), canonical(row["label"])
        results.append({
            "model_id": args.model_id, "event_id": row["event_id"],
            "dataset_name": row["dataset_name"], "instruction_text": row["instruction_text"],
            "target": row["label"], "raw_prediction": raw,
            "canonical_prediction": prediction, "canonical_target": target,
            "exact_match": str(prediction == target).lower(), "batch_size": 1,
        })
        write(args.output, results)
        if index == 1 or index % 25 == 0 or index == len(pending):
            print(f"[{index}/{len(pending)}] {row['dataset_name']} -> {raw!r}", flush=True)


if __name__ == "__main__": main()
