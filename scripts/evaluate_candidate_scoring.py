#!/usr/bin/env python3
"""Evaluate native candidate likelihoods under fixed prompt controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import yaml

from animal_omni.metrics import classification_metrics
from animal_omni.qwen_runner import QwenThinkerRunner


def atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def balanced_gate(rows: list[dict], labels: list[str], per_class: int, seed: int) -> list[dict]:
    selected = []
    for label in labels:
        candidates = [row for row in rows if row["label"] == label]
        candidates.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row['event_id']}".encode()
        ).hexdigest())
        if len(candidates) < per_class:
            raise ValueError(f"{label} has only {len(candidates)} rows, need {per_class}")
        selected.extend(candidates[:per_class])
    return sorted(selected, key=lambda row: row["event_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--prompt-names", nargs="+")
    parser.add_argument("--gate-per-class", type=int, default=0,
                        help="0 evaluates all manifest rows")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    config = yaml.safe_load(args.config.read_text())
    labels = list(config["labels"])
    manifest = args.manifest or Path(config["manifest"])
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    if args.gate_per_class:
        rows = balanced_gate(rows, labels, args.gate_per_class, int(config["seed"]))
    if args.limit is not None:
        rows = rows[:args.limit]
    prompts = config["prompts"]
    prompt_names = args.prompt_names or list(prompts)
    unknown = set(prompt_names) - set(prompts)
    if unknown:
        raise ValueError(f"unknown prompts: {sorted(unknown)}")
    output = list(csv.DictReader(args.output.open(newline="", encoding="utf-8"))) \
        if args.resume and args.output.exists() else []
    complete = {(row["event_id"], row["prompt_name"]) for row in output}
    runner = QwenThinkerRunner(args.model_id or config["model_id"])
    total = len(rows) * len(prompt_names)
    for row in rows:
        for prompt_name in prompt_names:
            identity = (row["event_id"], prompt_name)
            if identity in complete:
                continue
            scores = runner.score_candidates(
                row["audio_path"], prompts[prompt_name], labels,
                candidate_batch_size=int(config["candidate_scoring"]["candidate_batch_size"]),
            )
            by_label = {str(record["candidate"]): record for record in scores}
            prediction_mean = max(labels, key=lambda label: by_label[label]["mean_token_logprob"])
            prediction_sum = max(labels, key=lambda label: by_label[label]["sequence_logprob"])
            record = {
                "event_id": row["event_id"], "recording_id": row.get("recording_id", ""),
                "target": row["label"], "prompt_name": prompt_name,
                "prediction_mean": prediction_mean, "prediction_sum": prediction_sum,
                "correct_mean": str(prediction_mean == row["label"]).lower(),
                "correct_sum": str(prediction_sum == row["label"]).lower(),
                "candidate_scores_json": json.dumps(scores, separators=(",", ":")),
            }
            output.append(record); complete.add(identity)
            atomic_write(args.output, output)
            print(f"[{len(output)}/{total}] {row['event_id']} {prompt_name} "
                  f"mean={prediction_mean} sum={prediction_sum}", flush=True)
    summaries = {}
    for prompt_name in prompt_names:
        selected = [row for row in output if row["prompt_name"] == prompt_name and
                    row["event_id"] in {item["event_id"] for item in rows}]
        summaries[prompt_name] = {
            "n": len(selected),
            "mean_normalized": classification_metrics(
                [row["target"] for row in selected],
                [row["prediction_mean"] for row in selected], labels,
            ),
            "sequence_sum": classification_metrics(
                [row["target"] for row in selected],
                [row["prediction_sum"] for row in selected], labels,
            ),
        }
    payload = {
        "model_id": args.model_id or config["model_id"], "manifest": str(manifest),
        "gate_per_class": args.gate_per_class, "prompts": prompt_names,
        "candidate_scoring": config["candidate_scoring"], "summaries": summaries,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
