#!/usr/bin/env python3
"""Evaluate query-label-free token-preserving KV repair on a fixed split."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from animal_omni.conditional_kv import (
    ConditionalTokenGradientRouter,
    flatten_token_gradient,
    place_audio_token_delta,
    unflatten_token_gradient,
)
from animal_omni.kv_hooks import KVDeltaHooks
from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def representation(root: Path, event_id: str, layer: int) -> np.ndarray:
    with np.load(root / f"{event_id}.npz", allow_pickle=False) as record:
        return record["representation"][layer].astype(np.float32)


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", default="lp_0-1000")
    parser.add_argument("--query-split", default="valid")
    parser.add_argument("--limit-query", type=int)
    parser.add_argument("--gradient-dir", type=Path, required=True)
    parser.add_argument("--support-split", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--feature-layer", type=int, default=28)
    parser.add_argument("--support-k", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--etas", type=float, nargs="+", default=[100, 300, 1000, 3000])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    config = yaml.safe_load(args.config.read_text())
    labels = config["dataset"]["labels"]
    rows = [
        row for row in csv.DictReader(args.manifest.open(newline="", encoding="utf-8"))
        if row["condition"] == args.condition and row["split"] == args.query_split
    ]
    if args.limit_query is not None:
        rows = rows[:args.limit_query]
    split = json.loads(args.support_split.read_text())
    support = split["support_order"][:args.support_k]
    features, fields = [], []
    keys = None
    token_count = width = None
    for event_id in support:
        record = torch.load(args.gradient_dir / f"{event_id}.pt", map_location="cpu",
                            weights_only=True)
        vector, event_keys, event_tokens, event_width = flatten_token_gradient(
            record["tokenwise_audio_gradient"]
        )
        if keys is None:
            keys, token_count, width = event_keys, event_tokens, event_width
        elif (event_keys, event_tokens, event_width) != (keys, token_count, width):
            raise ValueError("support tokenwise gradient shapes are not aligned")
        fields.append(vector)
        features.append(representation(args.representation_dir, event_id, args.feature_layer))
    router = ConditionalTokenGradientRouter.fit(
        np.stack(features), np.stack(fields), rank=args.rank, alpha=args.alpha
    )
    output = list(csv.DictReader(args.output.open(newline="", encoding="utf-8"))) \
        if args.resume and args.output.exists() else []
    complete = {(row["event_id"], row["method"], float(row["eta"])) for row in output}
    runner = QwenThinkerRunner(args.model_id)
    audio_token_id = runner.model.thinker.config.audio_token_id
    total = len(rows) * 2 * len(args.etas)
    for row in rows:
        feature = representation(args.representation_dir, row["event_id"], args.feature_layer)
        inputs = runner.prepare_inputs(row["audio_path"], config["evaluation"]["prompt"])
        audio_mask = inputs["input_ids"].eq(audio_token_id)
        input_length = inputs["input_ids"].shape[1]
        vectors = {"fixed_token_field": router.fixed_mean(),
                   "conditional_token_field": router.predict(feature)}
        for method, vector in vectors.items():
            field = unflatten_token_gradient(vector, keys, token_count, width)
            for eta in args.etas:
                identity = (row["event_id"], method, eta)
                if identity in complete:
                    continue
                deltas = place_audio_token_delta(field, audio_mask, eta)
                with torch.inference_mode(), KVDeltaHooks(runner.model.thinker, deltas):
                    generated = runner.model.generate(
                        **inputs, return_audio=False, do_sample=False,
                        max_new_tokens=config["evaluation"]["max_new_tokens"],
                        use_audio_in_video=False,
                    )
                raw = runner.processor.batch_decode(
                    generated[:, input_length:], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                prediction = normalize_label(raw, labels) or ""
                output.append({
                    "event_id": row["event_id"], "target": row["label"],
                    "query_split": args.query_split, "condition": args.condition,
                    "support_k": args.support_k, "method": method, "eta": eta,
                    "requested_rank": args.rank, "actual_rank": router.rank,
                    "feature_layer": args.feature_layer, "token_count": token_count,
                    "prediction": prediction, "raw_prediction": raw,
                    "correct": str(prediction == row["label"]).lower(),
                })
                write(args.output, output)
                if len(output) == 1 or len(output) % 50 == 0 or len(output) == total:
                    print(f"[{len(output)}/{total}] {row['event_id']} {method} eta={eta} -> {prediction}",
                          flush=True)


if __name__ == "__main__":
    main()
