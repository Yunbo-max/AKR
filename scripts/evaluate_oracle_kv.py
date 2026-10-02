#!/usr/bin/env python3
"""Oracle pre-RoPE KV recovery on preselected eligible failures."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
import yaml

from animal_omni.kv_hooks import KVDeltaHooks, label_kv_gradients, pooled_audio_gradient
from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--intervention-manifest", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split")
    parser.add_argument("--gradients-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    cfg = yaml.safe_load(args.config.read_text())
    labels = cfg["dataset"]["labels"]

    with args.predictions.open(newline="", encoding="utf-8") as f:
        predictions = list(csv.DictReader(f))
    by_event = {}
    for row in predictions:
        by_event.setdefault(row["event_id"], {})[row["condition"]] = row
    eligible = [event_id for event_id, values in by_event.items()
                if values.get("full_0-8k", {}).get("correct") == "true"
                and values.get(args.condition, {}).get("correct") != "true"]
    if args.limit is not None: eligible = eligible[:args.limit]

    with args.intervention_manifest.open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    if args.split:
        allowed = {r["event_id"] for r in manifest_rows if r.get("split") == args.split}
        eligible = [event_id for event_id in eligible if event_id in allowed]
    manifest = {(r["event_id"], r["condition"]): r for r in manifest_rows}
    results = []
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as f: results = list(csv.DictReader(f))
    completed = {r["event_id"] for r in results}

    runner = QwenThinkerRunner(args.model_id)
    for parameter in runner.model.parameters(): parameter.requires_grad_(False)
    audio_token_id = runner.model.thinker.config.audio_token_id
    args.gradient_dir.mkdir(parents=True, exist_ok=True)
    for index, event_id in enumerate([x for x in eligible if x not in completed], 1):
        row = manifest[(event_id, args.condition)]
        inputs = runner.teacher_forced_inputs(row["audio_path"], cfg["evaluation"]["prompt"], row["label"])
        audio_mask = inputs["input_ids"].eq(audio_token_id)
        prompt_length = int((inputs["labels"] == -100).sum())
        base_loss, directions = label_kv_gradients(runner.model.thinker, inputs)
        pooled = pooled_audio_gradient(directions, audio_mask)
        torch.save({"event_id": event_id, "condition": args.condition, "target": row["label"],
                    "pooled_audio_gradient": pooled}, args.gradient_dir / f"{event_id}.pt")
        record = {"event_id": event_id, "condition": args.condition, "target": row["label"],
                  "baseline_prediction": by_event[event_id][args.condition]["prediction"],
                  "base_loss": base_loss}
        if args.gradients_only:
            results.append(record); atomic_write(args.output, results)
            print(f"[{index}] {event_id} {row['label']} gradient_saved", flush=True)
            continue
        for eta in cfg["kv"]["oracle_etas"]:
            deltas = {key: value[:, :prompt_length, :].mul(eta) for key, value in directions.items()}
            with KVDeltaHooks(runner.model.thinker, deltas):
                raw = runner.predict(row["audio_path"], cfg["evaluation"]["prompt"],
                                     max_new_tokens=cfg["evaluation"]["max_new_tokens"])
            parsed = normalize_label(raw, labels)
            suffix = str(eta).replace(".", "p")
            record[f"prediction_eta_{suffix}"] = parsed or ""
            record[f"correct_eta_{suffix}"] = str(parsed == row["label"]).lower()
        results.append(record); atomic_write(args.output, results)
        print(f"[{index}] {event_id} {row['label']} recovered={any(v == 'true' for k,v in record.items() if k.startswith('correct_'))}", flush=True)


if __name__ == "__main__": main()
