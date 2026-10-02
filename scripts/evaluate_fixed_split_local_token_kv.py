#!/usr/bin/env python3
"""Evaluate locally aligned tokenwise KV repair on an official fixed split."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from animal_omni.conditional_kv import ConditionalLocalTokenRouter, place_audio_token_delta
from animal_omni.kv_hooks import KVDeltaHooks
from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def load_representation(root: Path, event_id: str, layer: int):
    with np.load(root / f"{event_id}.npz", allow_pickle=False) as record:
        return (record["token_representation"].astype(np.float32),
                record["representation"][layer].astype(np.float32))


def atomic_write(path: Path, rows: list[dict]) -> None:
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
    parser.add_argument("--local-rank", type=int, default=64)
    parser.add_argument("--global-rank", type=int, default=16)
    parser.add_argument("--gradient-rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20250813)
    parser.add_argument("--etas", type=float, nargs="+", default=[1, 3, 10, 30, 100])
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
    token_features, global_features, gradients = [], [], []
    for event_id in support:
        tokens, global_feature = load_representation(
            args.representation_dir, event_id, args.feature_layer
        )
        record = torch.load(args.gradient_dir / f"{event_id}.pt", map_location="cpu",
                            weights_only=True)
        token_features.append(tokens); global_features.append(global_feature)
        gradients.append(record["tokenwise_audio_gradient"])
    router = ConditionalLocalTokenRouter.fit(
        np.stack(token_features), np.stack(global_features), gradients,
        local_rank=args.local_rank, global_rank=args.global_rank,
        gradient_rank=args.gradient_rank, alpha=args.alpha, seed=args.seed,
    )
    output = list(csv.DictReader(args.output.open(newline="", encoding="utf-8"))) \
        if args.resume and args.output.exists() else []
    complete = {(row["event_id"], row["method"], float(row["eta"])) for row in output}
    runner = QwenThinkerRunner(args.model_id)
    audio_token_id = runner.model.thinker.config.audio_token_id
    expected = len(rows) * 2 * len(args.etas)
    for row in rows:
        tokens, global_feature = load_representation(
            args.representation_dir, row["event_id"], args.feature_layer
        )
        predicted = router.predict(tokens, global_feature)
        fields = {
            "fixed_local_mean": router.fixed_mean(len(tokens)),
            "conditional_local_token": predicted,
        }
        inputs = runner.prepare_inputs(row["audio_path"], config["evaluation"]["prompt"])
        audio_mask = inputs["input_ids"].eq(audio_token_id)
        input_length = inputs["input_ids"].shape[1]
        for method, field in fields.items():
            tensor_field = {key: torch.from_numpy(value) for key, value in field.items()}
            for eta in args.etas:
                identity = (row["event_id"], method, eta)
                if identity in complete:
                    continue
                deltas = place_audio_token_delta(tensor_field, audio_mask, eta)
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
                    "local_rank": args.local_rank, "global_rank": args.global_rank,
                    "gradient_rank": router.gradient_rank, "feature_layer": args.feature_layer,
                    "prediction": prediction, "raw_prediction": raw,
                    "correct": str(prediction == row["label"]).lower(),
                })
                atomic_write(args.output, output)
                if len(output) == 1 or len(output) % 50 == 0 or len(output) == expected:
                    print(f"[{len(output)}/{expected}] {row['event_id']} {method} "
                          f"eta={eta} -> {prediction}", flush=True)


if __name__ == "__main__":
    main()
