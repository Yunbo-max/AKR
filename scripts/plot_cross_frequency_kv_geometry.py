#!/usr/bin/env python3
"""Plot matched-support cross-frequency corrective-gradient geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    conditions = payload["conditions"]
    cutoff = np.asarray([8, 1, 2, 4, 6, 8], dtype=float)
    # Draw the unfiltered anchor at the right edge but give it a distinct label.
    order = sorted(range(len(conditions)), key=lambda index: (cutoff[index], index == 0))
    ordered = [conditions[index] for index in order]
    x = cutoff[order]
    layer_ids = sorted(map(int, payload["layers"]))
    heat = np.asarray([
        [payload["layers"][str(layer)][condition]["same_minus_different_cosine"]
         for condition in ordered]
        for layer in layer_ids
    ])
    anchor_cosine = [
        payload["condition_summary"][condition]["mean_paired_to_anchor_cosine"]
        for condition in ordered
    ]
    rank = [
        payload["condition_summary"][condition]["median_layer_effective_rank"]
        for condition in ordered
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)
    axes[0].plot(range(len(ordered)), anchor_cosine, marker="o", color="#2166ac")
    axes[0].set_xticks(range(len(ordered)), [
        "1", "2", "4", "6", "8 LP", "Full" if ordered[-1] == "full_0-8k" else "8"
    ])
    axes[0].set_ylim(0.45, 1.03)
    axes[0].set_ylabel("paired cosine to full-gradient field")
    axes[0].set_xlabel("observable cutoff (kHz)")
    axes[0].set_title("Correction rotates with frequency")
    axes[0].grid(alpha=.25)

    image = axes[1].imshow(heat, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(ordered)), ["1", "2", "4", "6", "8 LP", "Full"])
    axes[1].set_yticks([0, 6, 13, 20, 27])
    axes[1].set_xlabel("observable cutoff (kHz)")
    axes[1].set_ylabel("Thinker layer")
    axes[1].set_title("Within-class − between-class cosine")
    figure.colorbar(image, ax=axes[1], shrink=.8)

    axes[2].plot(range(len(ordered)), rank, marker="s", color="#b2182b")
    axes[2].set_xticks(range(len(ordered)), ["1", "2", "4", "6", "8 LP", "Full"])
    axes[2].set_ylim(0, max(rank) * 1.18)
    axes[2].set_ylabel("median centered effective rank")
    axes[2].set_xlabel("observable cutoff (kHz)")
    axes[2].set_title("Corrective field stays compact")
    axes[2].grid(alpha=.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
