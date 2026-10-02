#!/usr/bin/env python3
"""Deterministic Thinker-only evaluation with append-safe result persistence."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import yaml

from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--event-id")
    parser.add_argument("--condition")
    parser.add_argument("--split")
    parser.add_argument("--model-id")
    parser.add_argument("--prompt", help="fixed prompt override for prompt-sensitivity controls")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed-from", type=Path)
    parser.add_argument("--seed-condition")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--order-by-condition", action="store_true",
                        help="evaluate complete conditions sequentially for earlier resumable summaries")
    args = parser.parse_args()
    # qwen-omni-utils emits one irrelevant Talker warning per text-only sample.
    logging.getLogger().setLevel(logging.ERROR)
    cfg = yaml.safe_load(args.config.read_text())
    labels = cfg["dataset"]["labels"]
    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.event_id:
        rows = [r for r in rows if r["event_id"] == args.event_id]
    if args.condition:
        rows = [r for r in rows if r.get("condition", "full_0-8k") == args.condition]
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    if args.order_by_condition:
        rows.sort(key=lambda r: (r.get("condition", "full_0-8k"), r["event_id"]))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no rows selected")

    model_id = args.model_id or cfg["model"]["id"]
    prompt = args.prompt or cfg["evaluation"]["prompt"]
    selected_keys = {(row["event_id"], row.get("condition", "full_0-8k")) for row in rows}
    existing = []
    if args.seed_from and not args.output.exists():
        with args.seed_from.open(newline="", encoding="utf-8") as f:
            seed_rows = list(csv.DictReader(f))
        for row in seed_rows:
            if (row["model_id"] != model_id
                    or (row["event_id"], row["condition"]) not in selected_keys
                    or (args.seed_condition and row["condition"] != args.seed_condition)):
                continue
            existing.append({**row, "batch_size": "1"})
        print(f"seeded {len(existing)} prior batch=1 rows", flush=True)
    completed = {(r["model_id"], r["event_id"], r["condition"]) for r in existing}
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        completed = {(r["model_id"], r["event_id"], r["condition"]) for r in existing}
    rows = [r for r in rows if (model_id, r["event_id"], r.get("condition", "full_0-8k")) not in completed]
    if not rows:
        if existing and not args.output.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=existing[0].keys())
                writer.writeheader(); writer.writerows(existing)
        print("all selected rows already completed")
        return
    runner = QwenThinkerRunner(model_id)
    output_rows = existing
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        raw_predictions = runner.predict_batch(
            [row["audio_path"] for row in batch], prompt,
            max_new_tokens=cfg["evaluation"]["max_new_tokens"],
        ) if len(batch) > 1 else [runner.predict(
            batch[0]["audio_path"], prompt,
            max_new_tokens=cfg["evaluation"]["max_new_tokens"],
        )]
        for offset, (row, raw) in enumerate(zip(batch, raw_predictions), 1):
            parsed = normalize_label(raw, labels)
            output_rows.append({
                "model_id": model_id, "event_id": row["event_id"],
                "condition": row.get("condition", "full_0-8k"), "target": row["label"],
                "raw_prediction": raw, "prediction": parsed or "", "correct": str(parsed == row["label"]).lower(),
                "batch_size": str(args.batch_size),
            })
            progress = start + offset
            if progress == 1 or progress == len(rows) or progress % args.log_every == 0:
                print(f"[{progress}/{len(rows)}] {row['event_id']} {row.get('condition', 'full_0-8k')} -> {raw!r}", flush=True)
        # Atomic rewrite after every example makes long GPU jobs resumable.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
            writer.writeheader(); writer.writerows(output_rows)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
