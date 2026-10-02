#!/usr/bin/env python3
"""Evaluate multi-audio ICL using exactly a registered K-per-class support set."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path

import yaml

from animal_omni.metrics import classification_metrics, normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def parse_prediction(raw: str, labels: list[str], *, strict: bool) -> str:
    if not strict:
        return normalize_label(raw, labels) or ""
    cleaned = re.sub(r"[\s.,:;!?]+", "", raw)
    return cleaned if cleaned in labels else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", help="optional manifest condition filter")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--support-k-per-class", type=int, nargs="+", default=[1])
    parser.add_argument("--readout", choices=["free", "candidate"], default="free")
    parser.add_argument("--label-map", type=Path,
                        help="JSON mapping from dataset labels to arbitrary output strings")
    parser.add_argument("--max-new-tokens", type=int,
                        help="override configured generation length")
    parser.add_argument("--limit-query", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    config = yaml.safe_load(args.config.read_text())
    labels = config.get("labels") or config["dataset"]["labels"]
    prompt = config.get("prompts", {}).get("bare") or config["evaluation"]["prompt"]
    label_map = {label: label for label in labels}
    if args.label_map:
        label_map = json.loads(args.label_map.read_text())
        if set(label_map) != set(labels) or len(set(label_map.values())) != len(labels):
            raise ValueError("label map must bijectively cover every dataset label")
        prompt = (
            "Classify this sound into one of the registered arbitrary acoustic categories. "
            f"Choose exactly one label from: {', '.join(label_map[label] for label in labels)}. "
            "Answer with only the label."
        )
    output_labels = [label_map[label] for label in labels]
    max_new_tokens = (args.max_new_tokens or
                      config.get("evaluation", {}).get("max_new_tokens", 8))
    manifest_rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if args.condition is not None:
        manifest_rows = [row for row in manifest_rows if row.get("condition") == args.condition]
    by_event = {row["event_id"]: row for row in manifest_rows}
    split = json.loads(args.split.read_text())
    query_events = split["query_events"]
    if args.limit_query is not None:
        query_events = query_events[:args.limit_query]
    output = list(csv.DictReader(args.output.open(newline="", encoding="utf-8"))) \
        if args.resume and args.output.exists() else []
    complete = {(row["event_id"], int(row["support_k_per_class"]), row["readout"])
                for row in output}
    runner = QwenThinkerRunner(args.model_id)
    total = len(query_events) * len(args.support_k_per_class)
    for k in args.support_k_per_class:
        key = str(k)
        if k == 0:
            support_events = []
        elif key not in split["support_sets"]:
            raise ValueError(f"split has no K={k} support set")
        else:
            support_events = split["support_sets"][key]
        support = [(by_event[event]["audio_path"], label_map[by_event[event]["label"]])
                   for event in support_events]
        for event in query_events:
            identity = (event, k, args.readout)
            if identity in complete:
                continue
            row = by_event[event]
            if args.readout == "free":
                raw = runner.predict_icl(
                    support, row["audio_path"], prompt, max_new_tokens=max_new_tokens
                )
                prediction = parse_prediction(
                    raw, output_labels, strict=args.label_map is not None
                )
                scores_json = ""
            else:
                scores = runner.score_candidates_icl(
                    support, row["audio_path"], prompt, output_labels
                )
                prediction = max(scores, key=lambda item: item["mean_token_logprob"])["candidate"]
                raw = ""; scores_json = json.dumps(scores, separators=(",", ":"))
            output.append({
                "event_id": event, "target": row["label"],
                "target_output": label_map[row["label"]],
                "support_k_per_class": k, "support_k_total": len(support),
                "method": f"audio_icl_{args.readout}", "readout": args.readout,
                "raw_prediction": raw, "candidate_scores_json": scores_json,
                "prediction": prediction,
                "correct": str(prediction == label_map[row["label"]]).lower(),
            })
            complete.add(identity); atomic_write(args.output, output)
            print(f"[{len(output)}/{total}] {event} K/class={k} -> {prediction}", flush=True)
    summaries = {}
    for k in args.support_k_per_class:
        selected = [row for row in output if int(row["support_k_per_class"]) == k and
                    row["readout"] == args.readout and
                    row["event_id"] in set(query_events)]
        summaries[str(k)] = {
            "support_k_per_class": k,
            "support_k_total": 0 if k == 0 else len(split["support_sets"][str(k)]),
            "n_query": len(selected),
            **classification_metrics(
                [row["target_output"] for row in selected],
                [row["prediction"] or None for row in selected], output_labels,
            ),
            "invalid_response_rate": (
                sum(not row["prediction"] for row in selected) / len(selected)
                if selected else None
            ),
            "prediction_counts": dict(Counter(row["prediction"] for row in selected)),
        }
    payload = {
        "model_id": args.model_id, "manifest": str(args.manifest),
        "condition": args.condition, "readout": args.readout,
        "label_map": label_map,
        "split": str(args.split), "query_labels_used_for": "post_hoc_scoring_only",
        "results": summaries,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
