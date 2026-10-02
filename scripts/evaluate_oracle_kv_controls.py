#!/usr/bin/env python3
"""Matched-norm causal controls for label-gradient KV Oracle interventions.

This is an upper-bound/capacity experiment. Query labels are deliberately used
to construct the correct-label direction and therefore the result is never
reported as a deployable repair method.
"""
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

from animal_omni.kv_hooks import KVDeltaHooks, label_kv_gradients
from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def audio_only(direction: dict, audio_mask: torch.Tensor, prompt_length: int) -> dict:
    mask = audio_mask[:, :prompt_length].detach().cpu().bool()
    output = {}
    for key, value in direction.items():
        field = torch.zeros_like(value[:, :prompt_length]).float()
        field[mask] = value[:, :prompt_length][mask].float()
        output[key] = field
    return output


def token_permuted(direction: dict, audio_mask: torch.Tensor, seed: int) -> dict:
    mask = audio_mask.detach().cpu().bool()
    positions = torch.where(mask[0])[0]
    rng = np.random.default_rng(seed)
    order = torch.as_tensor(rng.permutation(len(positions)), dtype=torch.long)
    output = {}
    for key, value in direction.items():
        shuffled = value.clone()
        shuffled[:, positions] = value[:, positions[order]]
        output[key] = shuffled
    return output


def matched_random(direction: dict, seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = {}
    for key, value in direction.items():
        active = value.square().sum(-1) > 0
        random = torch.zeros_like(value).float()
        sample = torch.randn(value[active].shape, generator=generator)
        sample *= value[active].float().norm() / sample.norm().clamp_min(1e-12)
        random[active] = sample
        output[key] = random
    return output


def stable_seed(seed: int, event_id: str, suffix: str) -> int:
    digest = hashlib.sha256(f"{seed}:{event_id}:{suffix}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 31)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--intervention-manifest", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--relative-alphas", type=float, nargs="+",
                        default=[.001, .003, .01, .03, .1])
    parser.add_argument("--methods", nargs="+", default=[
        "correct_label", "wrong_label", "token_permuted", "random_field"
    ])
    parser.add_argument("--scope", choices=["audio", "full_prefill"], default="audio")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    config = yaml.safe_load(args.config.read_text())
    labels = config["dataset"]["labels"]
    prediction_rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    by_event = {}
    for row in prediction_rows:
        by_event.setdefault(row["event_id"], {})[row["condition"]] = row
    eligible = sorted(
        event for event, values in by_event.items()
        if values.get("full_0-8k", {}).get("correct") == "true"
        and values.get(args.condition, {}).get("correct") != "true"
    )[:args.limit]
    manifest_rows = list(csv.DictReader(
        args.intervention_manifest.open(newline="", encoding="utf-8")
    ))
    manifest = {(row["event_id"], row["condition"]): row for row in manifest_rows}
    output = list(csv.DictReader(args.output.open(newline="", encoding="utf-8"))) \
        if args.resume and args.output.exists() else []
    complete = {
        (row["event_id"], row["method"], float(row["relative_alpha"])) for row in output
    }
    runner = QwenThinkerRunner(args.model_id)
    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)
    audio_token_id = runner.model.thinker.config.audio_token_id
    total = len(eligible) * len(args.methods) * len(args.relative_alphas)
    for event_id in eligible:
        row = manifest[(event_id, args.condition)]
        wrong_label = labels[(labels.index(row["label"]) + 1) % len(labels)]
        correct_inputs = runner.teacher_forced_inputs(
            row["audio_path"], config["evaluation"]["prompt"], row["label"]
        )
        prompt_length = int((correct_inputs["labels"] == -100).sum())
        audio_mask = correct_inputs["input_ids"].eq(audio_token_id)[:, :prompt_length]
        _, correct_full = label_kv_gradients(runner.model.thinker, correct_inputs)
        correct = audio_only(correct_full, correct_inputs["input_ids"].eq(audio_token_id),
                             prompt_length) if args.scope == "audio" else {
                                 key: value[:, :prompt_length].float()
                                 for key, value in correct_full.items()
                             }
        wrong_inputs = runner.teacher_forced_inputs(
            row["audio_path"], config["evaluation"]["prompt"], wrong_label
        )
        _, wrong_full = label_kv_gradients(runner.model.thinker, wrong_inputs)
        wrong = audio_only(wrong_full, wrong_inputs["input_ids"].eq(audio_token_id),
                           prompt_length) if args.scope == "audio" else {
                               key: value[:, :prompt_length].float()
                               for key, value in wrong_full.items()
                           }
        fields = {
            "correct_label": correct,
            "wrong_label": wrong,
            "token_permuted": token_permuted(
                correct, audio_mask, stable_seed(args.seed, event_id, "permutation")
            ),
            "random_field": matched_random(
                correct, stable_seed(args.seed, event_id, "random")
            ),
        }
        # Reusing four large full-prefill fields from GPU avoids retransferring
        # every layer's delta for every alpha/generation call.
        fields = {
            method: {
                key: value.to(device=runner.model.device, dtype=runner.model.dtype)
                for key, value in direction.items()
            }
            for method, direction in fields.items()
        }
        unknown = set(args.methods) - set(fields)
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
        for method in args.methods:
            for alpha in args.relative_alphas:
                identity = (event_id, method, alpha)
                if identity in complete:
                    continue
                with KVDeltaHooks(
                    runner.model.thinker, fields[method], relative_alpha=alpha
                ) as hooks:
                    raw = runner.predict(
                        row["audio_path"], config["evaluation"]["prompt"],
                        max_new_tokens=config["evaluation"]["max_new_tokens"],
                    )
                prediction = normalize_label(raw, labels) or ""
                ratios = list(hooks.applied_relative_norms.values())
                output.append({
                    "event_id": event_id, "target": row["label"],
                    "wrong_label": wrong_label, "condition": args.condition,
                    "method": method, "relative_alpha": alpha,
                    "scope": ("prefill_audio_tokens_only" if args.scope == "audio"
                              else "full_prefill_tokens"),
                    "baseline_prediction": by_event[event_id][args.condition]["prediction"],
                    "prediction": prediction, "raw_prediction": raw,
                    "correct": str(prediction == row["label"]).lower(),
                    "wrong_target_hit": str(prediction == wrong_label).lower(),
                    "applied_ratio_mean": float(np.mean(ratios)),
                    "applied_ratio_min": float(np.min(ratios)),
                    "applied_ratio_max": float(np.max(ratios)),
                    "oracle_query_label_used": str(method in {"correct_label", "token_permuted", "random_field"}).lower(),
                })
                complete.add(identity); atomic_write(args.output, output)
                print(f"[{len(output)}/{total}] {event_id} {method} alpha={alpha} "
                      f"-> {prediction}", flush=True)


if __name__ == "__main__":
    main()
