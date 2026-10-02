#!/usr/bin/env python3
"""Extract layerwise audio-token representations with resumable per-event files."""
from __future__ import annotations

import argparse
import csv
import logging
import json
from pathlib import Path

import numpy as np
import yaml

from animal_omni.qwen_runner import QwenThinkerRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", default="full_0-8k")
    parser.add_argument("--conditions", nargs="+",
                        help="extract several conditions in one model load; creates condition subdirectories")
    parser.add_argument("--split")
    parser.add_argument("--event-ids-from", type=Path,
                        help="JSON containing support_order or query_events")
    parser.add_argument("--fixed-kv-support", type=Path,
                        help="JSON support split; extract its support plus validation/query rows")
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--query-split", default="test")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tokenwise-layer", type=int,
                        help="also save ordered audio-token states at this representation level")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    cfg = yaml.safe_load(args.config.read_text())
    conditions = set(args.conditions or [args.condition])
    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("condition", "full_0-8k") in conditions]
    if args.split: rows = [r for r in rows if r.get("split") == args.split]
    if args.event_ids_from:
        selection = json.loads(args.event_ids_from.read_text())
        event_ids = set(selection.get("support_order", selection.get("query_events", [])))
        rows = [r for r in rows if r["event_id"] in event_ids]
    if args.fixed_kv_support:
        selection = json.loads(args.fixed_kv_support.read_text())
        support_ids = set(selection["support_order"])
        support = [r for r in rows if r["event_id"] in support_ids]
        validation = [r for r in rows if r.get("split") == "valid"][:args.validation_limit]
        query = [r for r in rows if r.get("split") == args.query_split]
        selected_ids = support_ids | {r["event_id"] for r in validation + query}
        rows = [r for r in rows if r["event_id"] in selected_ids]
    if args.limit is not None: rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def output_path(row):
        root = args.output_dir / row.get("condition", "full_0-8k") if args.conditions else args.output_dir
        return root / f'{row["event_id"]}.npz'
    pending = [r for r in rows if not output_path(r).exists()]
    if not pending: print("all representations already extracted"); return
    runner = QwenThinkerRunner(args.model_id)
    logging.disable(logging.CRITICAL)
    for index, row in enumerate(pending, 1):
        if args.tokenwise_layer is None:
            representation = runner.extract_audio_representations(
                row["audio_path"], cfg["evaluation"]["prompt"]
            )
            token_representation = None
        else:
            representation, token_representation = \
                runner.extract_audio_representations_with_tokens(
                    row["audio_path"], cfg["evaluation"]["prompt"], args.tokenwise_layer
                )
        output = output_path(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.npz")
        payload = dict(representation=representation.astype(np.float16),
                       event_id=row["event_id"], label=row["label"],
                       recording_id=row.get("recording_id", row["event_id"]),
                       split=row.get("split", ""),
                       condition=row.get("condition", args.condition),
                       model_id=args.model_id)
        if token_representation is not None:
            payload.update(token_representation=token_representation.astype(np.float16),
                           tokenwise_layer=args.tokenwise_layer)
        np.savez_compressed(temporary, **payload)
        temporary.replace(output)
        if index == 1 or index % 25 == 0 or index == len(pending):
            token_shape = "" if token_representation is None else f" tokens={token_representation.shape}"
            print(f"[{index}/{len(pending)}] {row['event_id']} {representation.shape}{token_shape}",
                  flush=True)


if __name__ == "__main__": main()
