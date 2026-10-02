#!/usr/bin/env python3
"""Stream tokenwise support gradients into layer/time energy summaries."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def stats(distributions: np.ndarray) -> dict:
    probability = distributions / np.maximum(distributions.sum(-1, keepdims=True), 1e-30)
    tokens = probability.shape[1]
    positions = (np.arange(tokens) + .5) / tokens
    entropy = -(probability * np.log(np.maximum(probability, 1e-30))).sum(-1)
    thirds = np.array_split(np.arange(tokens), 3)
    return {
        "n": len(probability), "token_count": tokens,
        "effective_token_count_mean": float(np.exp(entropy).mean()),
        "effective_token_fraction_mean": float((np.exp(entropy) / tokens).mean()),
        "energy_center_of_mass_mean": float((probability * positions).sum(-1).mean()),
        "early_third_energy_mean": float(probability[:, thirds[0]].sum(-1).mean()),
        "middle_third_energy_mean": float(probability[:, thirds[1]].sum(-1).mean()),
        "late_third_energy_mean": float(probability[:, thirds[2]].sum(-1).mean()),
        "peak_position_mean": float(positions[np.argmax(probability, axis=1)].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.gradient_dir.glob("*.pt"))
    if not files:
        raise SystemExit("no gradient files")
    events, labels, by_layer = [], [], defaultdict(list)
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=True)
        gradient = record.get("tokenwise_audio_gradient")
        if gradient is None:
            raise ValueError(f"{path} has no tokenwise_audio_gradient")
        layers = sorted({layer for layer, _ in gradient})
        for layer in layers:
            energy = sum(
                gradient[(layer, kind)].float().square().sum(-1).numpy()
                for kind in ("k", "v") if (layer, kind) in gradient
            )
            by_layer[layer].append(energy)
        events.append(str(record["event_id"])); labels.append(str(record["target"]))
    labels_array = np.asarray(labels)
    payload = {
        "n": len(events), "events": events,
        "label_counts": {label: int(np.sum(labels_array == label))
                         for label in sorted(set(labels))},
        "layers": {},
    }
    for layer, values in sorted(by_layer.items()):
        matrix = np.stack(values)
        layer_summary = {"all": stats(matrix), "by_class": {}}
        for label in sorted(set(labels)):
            layer_summary["by_class"][label] = stats(matrix[labels_array == label])
        payload["layers"][str(layer)] = layer_summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "n": payload["n"],
        "most_concentrated_layer": min(
            payload["layers"], key=lambda layer:
            payload["layers"][layer]["all"]["effective_token_fraction_mean"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
