#!/usr/bin/env python3
"""Matched-norm controls for locally routed tokenwise KV interventions."""
from __future__ import annotations

import argparse
import csv
import hashlib
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


def representation(root: Path, event_id: str, layer: int):
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


def token_average(field: dict) -> dict:
    return {
        key: np.repeat(value.mean(0, keepdims=True), len(value), axis=0)
        for key, value in field.items()
    }


def token_permutation(field: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(next(iter(field.values()))))
    return {key: value[order].copy() for key, value in field.items()}


def matched_random_field(field: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    randomized = {}
    for key, value in field.items():
        random = rng.standard_normal(value.shape).astype(np.float32)
        random *= np.linalg.norm(value) / max(np.linalg.norm(random), 1e-12)
        randomized[key] = random
    return randomized


def stable_seed(base: int, event_id: str, suffix: str) -> int:
    digest = hashlib.sha256(f"{base}:{event_id}:{suffix}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


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
    parser.add_argument("--router-alpha", type=float, default=10.0)
    parser.add_argument("--relative-alphas", type=float, nargs="+",
                        default=[.001, .003, .01, .03, .1])
    parser.add_argument("--methods", nargs="+", default=[
        "fixed_mean", "conditional_pooled", "conditional_tokenwise",
        "token_permuted", "random_field",
    ])
    parser.add_argument("--method-batch-size", type=int, default=1,
                        help="batch same-query methods; batch=1 is the reference protocol")
    parser.add_argument("--seed", type=int, default=20260814)
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
    support_labels = split.get("labels", [])[:args.support_k]
    token_features, global_features, gradients = [], [], []
    for event_id in support:
        tokens, global_feature = representation(
            args.representation_dir, event_id, args.feature_layer
        )
        record = torch.load(args.gradient_dir / f"{event_id}.pt", map_location="cpu",
                            weights_only=True)
        token_features.append(tokens); global_features.append(global_feature)
        gradients.append(record["tokenwise_audio_gradient"])
    router = ConditionalLocalTokenRouter.fit(
        np.stack(token_features), np.stack(global_features), gradients,
        local_rank=args.local_rank, global_rank=args.global_rank,
        gradient_rank=args.gradient_rank, alpha=args.router_alpha, seed=args.seed,
    )
    output = list(csv.DictReader(args.output.open(newline="", encoding="utf-8"))) \
        if args.resume and args.output.exists() else []
    complete = {
        (row["event_id"], row["method"], float(row["relative_alpha"])) for row in output
    }
    runner = QwenThinkerRunner(args.model_id)
    audio_token_id = runner.model.thinker.config.audio_token_id
    expected = len(rows) * len(args.methods) * len(args.relative_alphas)

    def add_record(row, method, relative_alpha, raw, prediction, ratios, scales):
        output.append({
            "event_id": row["event_id"], "target": row["label"],
            "query_split": args.query_split, "condition": args.condition,
            "support_k_total": args.support_k,
            "support_k_per_class_min": min(
                support_labels.count(label) for label in labels
            ) if support_labels else "",
            "method": method, "relative_alpha": relative_alpha,
            "local_rank": args.local_rank, "global_rank": args.global_rank,
            "gradient_rank": router.gradient_rank,
            "feature_layer": args.feature_layer,
            "applied_ratio_mean": float(np.mean(ratios)),
            "applied_ratio_min": float(np.min(ratios)),
            "applied_ratio_max": float(np.max(ratios)),
            "raw_scale_mean": float(np.mean(scales)),
            "prediction": prediction, "raw_prediction": raw,
            "correct": str(prediction == row["label"]).lower(),
        })
        complete.add((row["event_id"], method, relative_alpha))

    for row in rows:
        tokens, global_feature = representation(
            args.representation_dir, row["event_id"], args.feature_layer
        )
        predicted = router.predict(tokens, global_feature)
        fields = {
            "fixed_mean": router.fixed_mean(len(tokens)),
            "conditional_pooled": token_average(predicted),
            "conditional_tokenwise": predicted,
            "token_permuted": token_permutation(
                predicted, stable_seed(args.seed, row["event_id"], "permutation")
            ),
            "random_field": matched_random_field(
                predicted, stable_seed(args.seed, row["event_id"], "random")
            ),
        }
        unknown = set(args.methods) - set(fields)
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
        inputs = runner.prepare_inputs(row["audio_path"], config["evaluation"]["prompt"])
        audio_mask = inputs["input_ids"].eq(audio_token_id)
        input_length = inputs["input_ids"].shape[1]
        unit_by_method = {}
        for method in args.methods:
            tensor_field = {key: torch.from_numpy(value) for key, value in fields[method].items()}
            unit_by_method[method] = place_audio_token_delta(tensor_field, audio_mask, 1.0)
        for start in range(0, len(args.methods), args.method_batch_size):
            chunk = args.methods[start:start + args.method_batch_size]
            identities = [
                (row["event_id"], method, alpha)
                for method in chunk for alpha in args.relative_alphas
            ]
            if all(identity in complete for identity in identities):
                continue
            if any(identity in complete for identity in identities) or len(chunk) == 1:
                # Partial resume falls back to the batch=1 reference path.
                for method in chunk:
                    for relative_alpha in args.relative_alphas:
                        identity = (row["event_id"], method, relative_alpha)
                        if identity in complete:
                            continue
                        with torch.inference_mode(), KVDeltaHooks(
                            runner.model.thinker, unit_by_method[method],
                            relative_alpha=relative_alpha,
                        ) as hooks:
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
                        add_record(
                            row, method, relative_alpha, raw, prediction,
                            list(hooks.applied_relative_norms.values()),
                            list(hooks.applied_scales.values()),
                        )
                        atomic_write(args.output, output)
            else:
                batch_inputs = runner.prepare_batch(
                    [row["audio_path"]] * len(chunk), config["evaluation"]["prompt"]
                )
                if batch_inputs["input_ids"].shape[1] != input_length:
                    raise RuntimeError("same-audio batch changed input sequence length")
                keys = sorted(unit_by_method[chunk[0]])
                batch_deltas = {
                    key: torch.cat([unit_by_method[method][key] for method in chunk], dim=0)
                    for key in keys
                }
                for relative_alpha in args.relative_alphas:
                    with torch.inference_mode(), KVDeltaHooks(
                        runner.model.thinker, batch_deltas, relative_alpha=relative_alpha
                    ) as hooks:
                        generated = runner.model.generate(
                            **batch_inputs, return_audio=False, do_sample=False,
                            max_new_tokens=config["evaluation"]["max_new_tokens"],
                            use_audio_in_video=False,
                        )
                    raws = runner.processor.batch_decode(
                        generated[:, input_length:], skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for index, (method, raw) in enumerate(zip(chunk, raws)):
                        raw = raw.strip(); prediction = normalize_label(raw, labels) or ""
                        ratios = [values[index] for values in
                                  hooks.applied_relative_norms_by_example.values()]
                        scales = [values[index] for values in
                                  hooks.applied_scales_by_example.values()]
                        add_record(row, method, relative_alpha, raw, prediction, ratios, scales)
                    atomic_write(args.output, output)
            if len(output) == 1 or len(output) % 50 == 0 or len(output) == expected:
                print(f"[{len(output)}/{expected}] {row['event_id']} chunk={chunk}", flush=True)


if __name__ == "__main__":
    main()
