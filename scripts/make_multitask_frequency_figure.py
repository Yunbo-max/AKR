#!/usr/bin/env python3
"""Plot low-pass curves and band-removal importance for all three task levels."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CUTOFFS = [1000, 2000, 4000, 6000, 8000]
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]


def load(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["condition"]: float(row["accuracy"]) for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--call-type", type=Path, required=True)
    parser.add_argument("--individual", type=Path, required=True)
    parser.add_argument("--species", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = {
        "Call type (MarmAudio)": load(args.call_type),
        "Individual (Dogs)": load(args.individual),
        "Species (Watkins)": load(args.species),
    }
    fig, (curve_ax, heat_ax) = plt.subplots(1, 2, figsize=(11.2, 4.0),
                                            gridspec_kw={"width_ratios": [1.15, 1]})
    positions = np.arange(6)
    for label, metrics in tasks.items():
        values = [metrics[f"lp_0-{cutoff}"] for cutoff in CUTOFFS]
        full = metrics["full_0-8k"]
        curve_ax.plot(positions[:5], values, "o-", label=label)
        curve_ax.plot(positions[5], full, marker="*", markersize=10,
                      color=curve_ax.lines[-1].get_color())
    curve_ax.set_xticks(positions, ["≤1", "≤2", "≤4", "≤6", "≤8", "Full"])
    curve_ax.set_xlabel("Observable spectrum (kHz; Full = processor 0–8 kHz)")
    curve_ax.set_ylabel("Accuracy")
    curve_ax.set_ylim(bottom=0)
    curve_ax.legend(frameon=False, fontsize=8)
    curve_ax.set_title("a  Cumulative frequency evidence")

    matrix = np.array([
        [100 * (metrics["full_0-8k"] - metrics[f"remove_{lo}-{hi}"])
         for lo, hi in BANDS]
        for metrics in tasks.values()
    ])
    bound = max(1.0, float(np.nanmax(np.abs(matrix))))
    image = heat_ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    heat_ax.set_xticks(range(5), ["0–1", "1–2", "2–4", "4–6", "6–8"])
    heat_ax.set_yticks(range(3), ["Call type", "Individual", "Species"])
    heat_ax.set_xlabel("Removed band (kHz)")
    heat_ax.set_title("b  Full minus band-removed accuracy")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            heat_ax.text(column, row, f"{matrix[row, column]:+.1f}",
                         ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=heat_ax, label="Accuracy change (percentage points)", shrink=.85)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
