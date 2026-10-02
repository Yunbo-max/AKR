#!/usr/bin/env python3
"""Evaluate free audio ICL under counterbalanced support-example orders."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from scipy.stats import binomtest

from animal_omni.metrics import classification_metrics, normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def ordered_support(split: dict, labels: list[str], k: int, rotation: int, mode: str):
    order = labels[rotation:] + labels[:rotation]
    by_label = split["support_by_label"]
    if mode == "blocked":
        events = [event for label in order for event in by_label[label][:k]]
    else:
        events = [by_label[label][index] for index in range(k) for label in order]
    return events, order[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--order-registry", type=Path, required=True)
    parser.add_argument("--support-k-per-class", type=int, required=True)
    parser.add_argument("--order-mode", choices=("blocked", "interleaved"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    labels = config.get("labels") or config["dataset"]["labels"]
    prompt = config.get("prompts", {}).get("bare") or config["evaluation"]["prompt"]
    max_new_tokens = config.get("evaluation", {}).get("max_new_tokens", 8)
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    by_event = {row["event_id"]: row for row in rows}
    split = json.loads(args.split.read_text())
    registry = json.loads(args.order_registry.read_text())
    if registry["labels"] != labels or set(registry["rotation_by_query"]) != set(split["query_events"]):
        raise RuntimeError("order registry does not match the frozen split")
    k = args.support_k_per_class
    if str(k) not in split["support_sets"]:
        raise ValueError(f"split has no K={k} support set")

    output = list(csv.DictReader(args.output.open(newline="", encoding="utf-8"))) \
        if args.resume and args.output.exists() else []
    complete = {row["event_id"] for row in output}
    runner = QwenThinkerRunner(args.model_id)
    for event in split["query_events"]:
        if event in complete:
            continue
        rotation = int(registry["rotation_by_query"][event])
        support_events, last_label = ordered_support(
            split, labels, k, rotation, args.order_mode
        )
        support = [(by_event[item]["audio_path"], by_event[item]["label"])
                   for item in support_events]
        raw = runner.predict_icl(
            support, by_event[event]["audio_path"], prompt,
            max_new_tokens=max_new_tokens,
        )
        prediction = normalize_label(raw, labels) or ""
        output.append({
            "event_id": event,
            "target": by_event[event]["label"],
            "support_k_per_class": k,
            "support_k_total": len(support),
            "order_mode": args.order_mode,
            "rotation": rotation,
            "last_support_label": last_label,
            "raw_prediction": raw,
            "prediction": prediction,
            "correct": str(prediction == by_event[event]["label"]).lower(),
            "copies_last_support_label": str(prediction == last_label).lower(),
        })
        complete.add(event)
        atomic_write(args.output, output)
        print(
            f"[{len(output)}/{len(split['query_events'])}] {event} "
            f"rotation={rotation} last={last_label} -> {prediction}", flush=True,
        )

    selected = [row for row in output if row["event_id"] in set(split["query_events"])]
    if len(selected) != len(split["query_events"]):
        raise RuntimeError("counterbalanced order run is incomplete")
    metrics = classification_metrics(
        [row["target"] for row in selected],
        [row["prediction"] or None for row in selected], labels,
    )
    copies = [row["prediction"] == row["last_support_label"] for row in selected]
    wrong_last = [row for row in selected if row["target"] != row["last_support_label"]]
    by_last = defaultdict(list)
    for row in selected:
        by_last[row["last_support_label"]].append(row)
    payload = {
        "model_id": args.model_id,
        "split": str(args.split),
        "order_registry": str(args.order_registry),
        "support_k_per_class": k,
        "support_k_total": len(labels) * k,
        "order_mode": args.order_mode,
        "n_query": len(selected),
        **metrics,
        "invalid_response_rate": sum(not row["prediction"] for row in selected) / len(selected),
        "prediction_counts": dict(Counter(row["prediction"] for row in selected)),
        "last_label_copy_rate": sum(copies) / len(copies),
        "last_label_copy_exact_binomial_p_vs_uniform_label_prior": float(
            binomtest(
                sum(copies), len(copies), 1 / len(labels), alternative="greater"
            ).pvalue
        ),
        "copy_rate_when_last_label_is_wrong": (
            sum(row["prediction"] == row["last_support_label"] for row in wrong_last)
            / len(wrong_last)
        ),
        "accuracy_when_target_equals_last_label": None,
        "accuracy_when_target_differs_from_last_label": (
            sum(row["prediction"] == row["target"] for row in wrong_last) / len(wrong_last)
        ),
        "by_last_support_label": {
            label: {
                "n": len(cell),
                "prediction_counts": dict(Counter(row["prediction"] for row in cell)),
                "copy_rate": sum(row["prediction"] == label for row in cell) / len(cell),
            }
            for label, cell in by_last.items()
        },
        "query_labels_used_for": (
            "counterbalanced experimental blocking and post-hoc scoring only"
        ),
    }
    target_last = [
        row for row in selected if row["target"] == row["last_support_label"]
    ]
    payload["accuracy_when_target_equals_last_label"] = (
        sum(row["prediction"] == row["target"] for row in target_last) / len(target_last)
        if target_last else None
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
