#!/usr/bin/env python3
"""Validation-only conditional-KV feature-layer and subspace-rank ablation."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge

from animal_omni.conditional_kv import (
    ConditionalGradientRouter,
    broadcast_audio_delta,
    flatten_gradient,
    unflatten_gradient,
)
from animal_omni.kv_hooks import KVDeltaHooks
from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def representation(root: Path, event_id: str, layer: int) -> np.ndarray:
    with np.load(root / f"{event_id}.npz", allow_pickle=False) as record:
        return record["representation"][layer].astype(np.float32)


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", default="lp_0-1000")
    parser.add_argument("--query-split", default="valid")
    parser.add_argument("--limit-query", type=int, default=64)
    parser.add_argument("--support-k", type=int, default=20)
    parser.add_argument("--support-sizes", type=int, nargs="+",
                        help="ablate support size; overrides --support-k")
    parser.add_argument("--gradient-dir", type=Path, required=True)
    parser.add_argument("--support-split", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--feature-layers", type=int, nargs="+", default=[0, 8, 16, 28])
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--methods", nargs="+", choices=["fixed_mean", "conditional"],
                        default=["conditional"])
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)

    cfg = yaml.safe_load(args.config.read_text())
    labels = cfg["dataset"]["labels"]
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["condition"] == args.condition and row["split"] == args.query_split
        ][:args.limit_query]
    split = json.loads(args.support_split.read_text())
    support_sizes = args.support_sizes or [args.support_k]
    support_ids = split["support_order"][:max(support_sizes)]
    gradients = {
        event_id: torch.load(
            args.gradient_dir / f"{event_id}.pt", map_location="cpu", weights_only=True
        )["pooled_audio_gradient"]
        for event_id in support_ids
    }
    first, keys = flatten_gradient(gradients[support_ids[0]])
    width = first.size // len(keys)
    gradient_matrix = np.stack([flatten_gradient(gradients[event_id])[0] for event_id in support_ids])
    routers = {}
    for support_k in support_sizes:
        selected_gradients = gradient_matrix[:support_k]
        gradient_mean = selected_gradients.mean(0)
        centered = selected_gradients - gradient_mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        maximum_rank = max(0, min(max(args.ranks), support_k - 1, len(vt)))
        maximum_basis = vt[:maximum_rank].astype(np.float32)
        maximum_targets = centered @ maximum_basis.T
        for layer in args.feature_layers:
            support_x = np.stack([
                representation(args.representation_dir, event_id, layer)
                for event_id in support_ids[:support_k]
            ])
            feature_mean = support_x.mean(0)
            feature_scale = support_x.std(0)
            feature_scale[feature_scale < 1e-6] = 1.0
            z = (support_x - feature_mean) / feature_scale
            if maximum_rank:
                ridge = Ridge(alpha=args.alpha).fit(z, maximum_targets)
                maximum_coefficients = np.asarray(ridge.coef_, dtype=np.float32)
                if maximum_coefficients.ndim == 1:
                    maximum_coefficients = maximum_coefficients[None, :]
                maximum_intercept = np.atleast_1d(ridge.intercept_).astype(np.float32)
            else:
                maximum_coefficients = np.empty((0, support_x.shape[1]), dtype=np.float32)
                maximum_intercept = np.empty(0, dtype=np.float32)
            for rank in args.ranks:
                actual_rank = max(0, min(rank, maximum_rank))
                routers[(support_k, layer, rank)] = ConditionalGradientRouter(
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    gradient_mean=gradient_mean.astype(np.float32),
                    basis=maximum_basis[:actual_rank],
                    coefficients=maximum_coefficients[:actual_rank],
                    intercept=maximum_intercept[:actual_rank],
                    rank=actual_rank,
                    alpha=args.alpha,
                )

    output_rows = []
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            output_rows = list(csv.DictReader(handle))
    completed = {
        (row["event_id"], int(row["support_k"]), int(row["feature_layer"]),
         int(row["requested_rank"]), row["method"])
        for row in output_rows
    }
    runner = QwenThinkerRunner(args.model_id)
    audio_token_id = runner.model.thinker.config.audio_token_id
    total = len(rows) * len(routers) * len(args.methods)
    for row in rows:
        prepared = runner.prepare_inputs(row["audio_path"], cfg["evaluation"]["prompt"])
        mask = prepared["input_ids"].eq(audio_token_id)
        input_length = prepared["input_ids"].shape[1]
        features = {
            layer: representation(args.representation_dir, row["event_id"], layer)
            for layer in args.feature_layers
        }
        for (support_k, layer, requested_rank), router in routers.items():
            vectors = {
                "conditional": router.predict(features[layer]),
                "fixed_mean": router.fixed_mean(),
            }
            for method in args.methods:
                identity = (row["event_id"], support_k, layer, requested_rank, method)
                if identity in completed:
                    continue
                vector = vectors[method]
                pooled = unflatten_gradient(vector.astype(np.float32), keys, width)
                deltas = broadcast_audio_delta(pooled, mask, args.eta)
                with torch.inference_mode(), KVDeltaHooks(runner.model.thinker, deltas):
                    generated = runner.model.generate(
                        **prepared,
                        return_audio=False,
                        do_sample=False,
                        max_new_tokens=cfg["evaluation"]["max_new_tokens"],
                        use_audio_in_video=False,
                    )
                raw = runner.processor.batch_decode(
                    generated[:, input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                prediction = normalize_label(raw, labels) or ""
                output_rows.append({
                    "event_id": row["event_id"],
                    "target": row["label"],
                    "feature_layer": layer,
                    "requested_rank": requested_rank,
                    "actual_rank": router.rank,
                    "support_k": support_k,
                    "eta": args.eta,
                    "method": method,
                    "prediction": prediction,
                    "raw_prediction": raw,
                    "correct": str(prediction == row["label"]).lower(),
                })
                write_rows(args.output, output_rows)
                if len(output_rows) == 1 or len(output_rows) % 100 == 0 or len(output_rows) == total:
                    print(f"[{len(output_rows)}/{total}] {row['event_id']} K={support_k} layer={layer} rank={requested_rank} {method}", flush=True)


if __name__ == "__main__":
    main()
