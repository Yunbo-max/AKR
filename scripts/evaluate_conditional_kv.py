#!/usr/bin/env python3
"""Evaluate fixed and query-label-free conditional pooled-KV adaptation.

The recording split is made before fitting. Only labeled support gradients enter
the mean, PCA basis, or router. Query labels are read solely for post-hoc scoring.
"""
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
    ConditionalGradientRouter, broadcast_audio_delta, flatten_gradient,
    unflatten_gradient,
)
from animal_omni.kv_hooks import KVDeltaHooks
from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def load_representation(path: Path, layer: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as record:
        return record["representation"][layer].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gradient-dir", type=Path, required=True)
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--condition", default="lp_0-1000")
    parser.add_argument("--support-sizes", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--query-group-count", type=int, default=10)
    parser.add_argument("--query-scope", choices=["all", "eligible"], default="all",
                        help="all is the deployable query evaluation; eligible is a post-hoc recovery diagnostic")
    parser.add_argument("--feature-layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--etas", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    parser.add_argument("--seed", type=int, default=20250813)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--limit-query", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    cfg = yaml.safe_load(args.config.read_text())
    labels = cfg["dataset"]["labels"]

    manifest_rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    rows = {r["event_id"]: r for r in manifest_rows if r["condition"] == args.condition}
    prediction_rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    by_event: dict[str, dict[str, dict]] = {}
    for row in prediction_rows:
        by_event.setdefault(row["event_id"], {})[row["condition"]] = row

    gradients = {}
    for path in sorted(args.gradient_dir.glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        gradients[str(record["event_id"])] = record["pooled_audio_gradient"]
    eligible = sorted(set(gradients) & set(rows))
    # Query groups are sampled from *all* recordings before looking at failure
    # eligibility. This avoids conditioning the deployable query distribution on
    # labels or baseline correctness. Support gradients can then be drawn only
    # from eligible examples in the remaining groups.
    groups = sorted({row["recording_id"] for row in rows.values()})
    eligible_groups = {rows[event]["recording_id"] for event in eligible}
    rng = np.random.default_rng(args.seed); rng.shuffle(groups)
    if args.query_group_count >= len(groups):
        raise ValueError("query-group-count must leave support recordings")
    query_groups = set(groups[:args.query_group_count])
    support_groups = [group for group in groups[args.query_group_count:]
                      if group in eligible_groups]
    # One support example per recording prevents a large recording dominating K.
    support_order = []
    for group in support_groups:
        candidates = [event for event in eligible if rows[event]["recording_id"] == group]
        support_order.append(sorted(candidates)[0])
    max_k = max(args.support_sizes)
    if max_k > len(support_order):
        raise ValueError(f"K={max_k} exceeds {len(support_order)} support recordings")
    query_pool = sorted(rows) if args.query_scope == "all" else eligible
    query_events = [event for event in query_pool if rows[event]["recording_id"] in query_groups]
    if args.limit_query is not None: query_events = query_events[:args.limit_query]
    split = {
        "seed": args.seed, "condition": args.condition,
        "query_groups": sorted(query_groups), "query_scope": args.query_scope,
        "query_events": query_events,
        "query_eligible_events": [event for event in query_events if event in set(eligible)],
        "support_order": support_order, "support_sizes": args.support_sizes,
        "query_label_policy": "labels used only for post-hoc scoring; eligibility is diagnostic",
    }
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text(json.dumps(split, indent=2))

    first_vector, keys = flatten_gradient(gradients[eligible[0]])
    width = first_vector.size // len(keys)
    models = {}
    for k in args.support_sizes:
        events = support_order[:k]
        x = np.stack([load_representation(args.representation_dir / f"{e}.npz", args.feature_layer)
                      for e in events])
        g = np.stack([flatten_gradient(gradients[e])[0] for e in events])
        models[k] = ConditionalGradientRouter.fit(x, g, rank=args.rank, alpha=args.alpha)

    output_rows = []
    if args.resume and args.output.exists():
        output_rows = list(csv.DictReader(args.output.open(newline="", encoding="utf-8")))
    completed = {(r["event_id"], int(r["support_k"]), r["method"], float(r["eta"]))
                 for r in output_rows}
    runner = QwenThinkerRunner(args.model_id)
    audio_token_id = runner.model.thinker.config.audio_token_id
    total = len(query_events) * len(args.support_sizes) * 2 * len(args.etas)
    done = len(completed)
    for event in query_events:
        row = rows[event]
        feature = load_representation(args.representation_dir / f"{event}.npz", args.feature_layer)
        prepared = runner.prepare_inputs(row["audio_path"], cfg["evaluation"]["prompt"])
        audio_mask = prepared["input_ids"].eq(audio_token_id)
        input_length = prepared["input_ids"].shape[1]
        for k in args.support_sizes:
            router = models[k]
            vectors = {"fixed_mean": router.fixed_mean(), "conditional": router.predict(feature)}
            for method, vector in vectors.items():
                pooled = unflatten_gradient(vector.astype(np.float32), keys, width)
                for eta in args.etas:
                    identity = (event, k, method, eta)
                    if identity in completed: continue
                    deltas = broadcast_audio_delta(pooled, audio_mask, eta)
                    with torch.inference_mode(), KVDeltaHooks(runner.model.thinker, deltas):
                        generated = runner.model.generate(
                            **prepared, return_audio=False, do_sample=False,
                            max_new_tokens=cfg["evaluation"]["max_new_tokens"],
                            use_audio_in_video=False,
                        )
                    raw = runner.processor.batch_decode(
                        generated[:, input_length:], skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                    prediction = normalize_label(raw, labels) or ""
                    output_rows.append({
                        "event_id": event, "recording_id": row["recording_id"],
                        "target": row["label"], "condition": args.condition,
                        "support_k": k, "method": method, "eta": eta,
                        "rank": router.rank, "feature_layer": args.feature_layer,
                        "raw_prediction": raw, "prediction": prediction,
                        "correct": str(prediction == row["label"]).lower(),
                    })
                    done += 1; write_rows(args.output, output_rows)
                    print(f"[{done}/{total}] {event} K={k} {method} eta={eta} -> {prediction}", flush=True)


if __name__ == "__main__": main()
