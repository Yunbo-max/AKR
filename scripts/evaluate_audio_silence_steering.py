#!/usr/bin/env python3
"""Support-calibrated audio--silence final-representation steering baseline.

This implements the inference-time intervention form from "Are Audio-Language
Models Listening?" using a fixed transferred layer set.  It is a matched
conceptual baseline on Qwen2.5-Omni, not a claim to reproduce the paper's
model-specific specialist-head discovery.
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

from animal_omni.metrics import classification_metrics
from animal_omni.qwen_runner import QwenThinkerRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--silence-manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--support-k-per-class", type=int, required=True)
    parser.add_argument("--condition")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[14, 18, 19, 20, 21, 22, 23, 24, 26])
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0, .1, .3, 1, 3])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--icl-options", action="store_true",
                        help="use registered support as audio ICL with single-token A.. options")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    config = yaml.safe_load(args.config.read_text())
    labels = config.get("labels") or config["dataset"]["labels"]
    prompt = config.get("prompts", {}).get("bare") or config["evaluation"]["prompt"]
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if args.condition is not None:
        rows = [row for row in rows if row.get("condition") == args.condition]
    by_event = {row["event_id"]: row for row in rows}
    silence = {row["event_id"]: row for row in csv.DictReader(
        args.silence_manifest.open(newline="", encoding="utf-8")
    )}
    split = json.loads(args.split.read_text())
    support = split["support_sets"][str(args.support_k_per_class)]
    query = split["query_events"]
    events = support + query
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    runner = QwenThinkerRunner(args.model_id)
    levels = runner.model.thinker.config.text_config.num_hidden_layers + 1
    if any(not -levels <= layer < levels for layer in args.layers):
        raise ValueError(f"layers must lie inside {levels} hidden-state levels")
    tokenizer = runner.processor.tokenizer
    output_labels = [chr(ord("A") + index) for index in range(len(labels))] \
        if args.icl_options else labels
    label_map = dict(zip(labels, output_labels))
    if args.icl_options:
        prompt = (
            "Use the labeled audio examples to infer the arbitrary acoustic category. "
            f"Choose exactly one label from: {', '.join(output_labels)}. "
            "Answer with only the label."
        )
    candidate_ids = []
    for label in output_labels:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"empty candidate: {label}")
        candidate_ids.append(ids[0])
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("labels must have distinct first tokens for this baseline")
    candidate_ids_tensor = torch.tensor(candidate_ids, device=runner.model.device)
    support_examples = [
        (by_event[event]["audio_path"], label_map[by_event[event]["label"]])
        for event in support
    ]

    for index, event in enumerate(events, 1):
        path = args.cache_dir / f"{event}.npz"
        if path.exists():
            continue
        def run_hidden(inputs: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
            """Return only the required final-token states, freeing the long ICL graph."""
            with torch.inference_mode():
                output = runner.model.thinker(
                    **inputs, use_cache=False, output_hidden_states=True, return_dict=True
                )
            final = output.hidden_states[-1][0, -1].float().cpu().numpy()
            selected = np.stack([
                output.hidden_states[layer][0, -1].float().cpu().numpy()
                for layer in args.layers
            ])
            del output, inputs
            torch.cuda.empty_cache()
            return final, selected

        if args.icl_options:
            audio_inputs = runner.prepare_icl_inputs(
                support_examples, by_event[event]["audio_path"], prompt
            )
        else:
            audio_inputs = runner.prepare_inputs(by_event[event]["audio_path"], prompt)
        audio_shape = tuple(audio_inputs["input_ids"].shape)
        audio_final, audio_layers = run_hidden(audio_inputs)
        if args.icl_options:
            silence_inputs = runner.prepare_icl_inputs(
                support_examples, silence[event]["audio_path"], prompt
            )
        else:
            silence_inputs = runner.prepare_inputs(silence[event]["audio_path"], prompt)
        silence_shape = tuple(silence_inputs["input_ids"].shape)
        if audio_shape != silence_shape:
            raise RuntimeError(f"audio/silence token mismatch for {event}")
        _, silence_layers = run_hidden(silence_inputs)
        directions = audio_layers - silence_layers
        np.savez_compressed(
            path, audio_final=audio_final.astype(np.float16),
            direction=directions.mean(0).astype(np.float16),
            event_id=event, target=by_event[event]["label"],
        )
        if index == 1 or index % 25 == 0 or index == len(events):
            print(f"states [{index}/{len(events)}] {event}", flush=True)

    def predict(event: str, beta: float) -> tuple[str, float]:
        with np.load(args.cache_dir / f"{event}.npz", allow_pickle=False) as record:
            hidden = record["audio_final"].astype(np.float32)
            direction = record["direction"].astype(np.float32)
        state = torch.from_numpy(hidden + beta * direction).to(
            device=runner.model.device, dtype=runner.model.dtype
        )
        with torch.inference_mode():
            logits = runner.model.thinker.lm_head(state).float()[candidate_ids_tensor]
        best = int(logits.argmax())
        target = output_labels.index(label_map[by_event[event]["label"]])
        other = torch.cat([logits[:target], logits[target + 1:]])
        margin = float((logits[target] - other.max()).cpu())
        return output_labels[best], margin

    calibration = []
    for beta in args.betas:
        predictions, margins = zip(*(predict(event, beta) for event in support))
        metrics = classification_metrics(
            [label_map[by_event[event]["label"]] for event in support],
            list(predictions), output_labels,
        )
        calibration.append({
            "beta": beta, **metrics, "mean_correct_margin": float(np.mean(margins))
        })
    selected = max(
        calibration,
        key=lambda row: (row["accuracy"], row["macro_f1"],
                         row["mean_correct_margin"], -abs(row["beta"])),
    )
    query_predictions = [predict(event, selected["beta"])[0] for event in query]
    baseline_predictions = [predict(event, 0)[0] for event in query]
    payload = {
        "model_id": args.model_id,
        "method": "transferred-layer audio-silence final-representation steering",
        "relationship_to_published_baseline": (
            "same audio-minus-matched-silence intervention form; fixed transferred "
            "specialist-layer set, without model-specific head rediscovery"
        ),
        "layers": args.layers, "support_k_per_class": args.support_k_per_class,
        "support_k_total": len(support), "n_query": len(query),
        "candidate_readout": "highest first-token label logit",
        "icl_options": args.icl_options, "label_map": label_map,
        "calibration": calibration, "selected_beta": selected["beta"],
        "query_baseline": classification_metrics(
            [label_map[by_event[event]["label"]] for event in query],
            baseline_predictions, output_labels,
        ),
        "query_steered": classification_metrics(
            [label_map[by_event[event]["label"]] for event in query],
            query_predictions, output_labels,
        ),
        "query_labels_used_for": "post_hoc_scoring_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
