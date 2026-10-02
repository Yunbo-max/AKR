#!/usr/bin/env python3
"""Summarize native MarmAudio readouts on the frozen equal-support query set."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from animal_omni.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--candidate-scoring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    labels = split["labels"]
    query = split["query_events"]
    query_set = set(query)
    generation = {
        row["event_id"]: row
        for row in csv.DictReader(args.generation.open(newline="", encoding="utf-8"))
        if row["condition"] == "full_0-8k" and row["event_id"] in query_set
    }
    candidates = {
        (row["event_id"], row["prompt_name"]): row
        for row in csv.DictReader(args.candidate_scoring.open(newline="", encoding="utf-8"))
        if row["event_id"] in query_set
    }
    if set(generation) != query_set:
        raise RuntimeError("generation file does not cover the frozen query exactly")
    if set(candidates) != {
        (event, prompt) for event in query for prompt in ("bare", "definition")
    }:
        raise RuntimeError("candidate file does not cover both prompts on the frozen query")
    targets = [generation[event]["target"] for event in query]
    results = {
        "free_generation": classification_metrics(
            targets, [generation[event]["prediction"] for event in query], labels
        )
    }
    for prompt in ("bare", "definition"):
        for score, prediction_column in (
            ("mean_token_logprob", "prediction_mean"),
            ("sequence_sum", "prediction_sum"),
        ):
            results[f"{prompt}/{score}"] = classification_metrics(
                targets,
                [candidates[(event, prompt)][prediction_column] for event in query],
                labels,
            )
    payload = {
        "split": str(args.split), "n_query": len(query),
        "query_labels_used_for": "post_hoc_scoring_only", "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
