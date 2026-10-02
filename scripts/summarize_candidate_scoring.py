#!/usr/bin/env python3
"""Detailed accuracy, rank, and margin summary for candidate-likelihood runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from animal_omni.metrics import classification_metrics


def summarize(rows: list[dict], labels: list[str], score_key: str, prediction_key: str) -> dict:
    ranks, margins = [], []
    for row in rows:
        scores = {item["candidate"]: float(item[score_key])
                  for item in json.loads(row["candidate_scores_json"])}
        ordered = sorted(labels, key=lambda label: scores[label], reverse=True)
        ranks.append(ordered.index(row["target"]) + 1)
        margins.append(scores[row["target"]] - max(
            scores[label] for label in labels if label != row["target"]
        ))
    predictions = [row[prediction_key] for row in rows]
    return {
        **classification_metrics([row["target"] for row in rows], predictions, labels),
        "mean_reciprocal_rank": float(np.mean(1 / np.asarray(ranks))),
        "median_target_rank": float(np.median(ranks)),
        "correct_margin_mean": float(np.mean(margins)),
        "correct_margin_median": float(np.median(margins)),
        "positive_margin_fraction": float(np.mean(np.asarray(margins) > 0)),
        "prediction_counts": dict(Counter(predictions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    payload = {"n_rows": len(rows), "prompts": {}}
    for prompt in sorted({row["prompt_name"] for row in rows}):
        selected = [row for row in rows if row["prompt_name"] == prompt]
        payload["prompts"][prompt] = {
            "n": len(selected),
            "mean_token_logprob": summarize(
                selected, args.labels, "mean_token_logprob", "prediction_mean"
            ),
            "sequence_logprob": summarize(
                selected, args.labels, "sequence_logprob", "prediction_sum"
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
