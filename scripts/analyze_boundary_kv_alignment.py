#!/usr/bin/env python3
"""Relate spectral decoder drift to corrective-KV field rotation."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


PROBE_FILES = {
    "lp_0-1000": "probe_lp1k_7b_oof_summary.json",
    "lp_0-2000": "probe_lp2000_7b_oof_summary.json",
    "lp_0-4000": "probe_lp4000_7b_oof_summary.json",
    "lp_0-6000": "probe_lp6000_7b_oof_summary.json",
    "lp_0-8000": "probe_lp8000_7b_oof_summary.json",
}


def exact_permutation_p(x: np.ndarray, y: np.ndarray, statistic) -> float:
    observed = abs(float(statistic(x, y)))
    values = [
        abs(float(statistic(x, np.asarray(permutation))))
        for permutation in itertools.permutations(y.tolist())
    ]
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    results = args.root.resolve() / "results"
    geometry = json.loads(
        (results / "kv_geometry_marmaudio_equal_support_k2_cross_frequency_7b.json")
        .read_text()
    )
    transfer = {
        row["condition"]: row["accuracy"]
        for row in json.loads(
            (results / "probe_fulltrained_frequency_7b_oof_summary.json").read_text()
        )["conditions"]
    }
    rows = []
    for condition, probe_file in PROBE_FILES.items():
        mean_alignment = float(np.mean([
            layer[condition]["paired_to_anchor_cosine_mean"]
            for layer in geometry["layers"].values()
        ]))
        condition_specific = float(json.loads((results / probe_file).read_text())["accuracy"])
        rows.append({
            "condition": condition,
            "cutoff_hz": int(condition.removeprefix("lp_0-")),
            "mean_corrective_field_cosine_to_full": mean_alignment,
            "corrective_field_rotation_one_minus_cosine": 1.0 - mean_alignment,
            "condition_specific_probe_accuracy": condition_specific,
            "fulltrained_transfer_accuracy": float(transfer[condition]),
            "decoder_boundary_drift_gap": condition_specific - float(transfer[condition]),
        })
    rotation = np.asarray([
        row["corrective_field_rotation_one_minus_cosine"] for row in rows
    ])
    boundary_gap = np.asarray([row["decoder_boundary_drift_gap"] for row in rows])
    spearman = float(spearmanr(rotation, boundary_gap).statistic)
    pearson = float(pearsonr(rotation, boundary_gap).statistic)
    payload = {
        "scope": "five preregistered degraded MarmAudio bandwidth conditions",
        "n_conditions": len(rows),
        "rows": rows,
        "spearman_rho": spearman,
        "spearman_exact_two_sided_permutation_p": exact_permutation_p(
            rotation, boundary_gap,
            lambda left, right: spearmanr(left, right).statistic,
        ),
        "pearson_r": pearson,
        "pearson_exact_two_sided_permutation_p": exact_permutation_p(
            rotation, boundary_gap,
            lambda left, right: pearsonr(left, right).statistic,
        ),
        "interpretation_limit": (
            "descriptive cross-condition association; n=5 and both quantities "
            "co-vary with bandwidth, so this is triangulation rather than a "
            "causal mediation estimate"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4.6, 3.7), constrained_layout=True)
    axis.scatter(rotation, 100 * boundary_gap, s=48, color="#3b6fb6")
    for row, x_value, y_value in zip(rows, rotation, 100 * boundary_gap):
        axis.annotate(
            f"{row['cutoff_hz'] // 1000}k", (x_value, y_value),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
    fit = np.polyfit(rotation, 100 * boundary_gap, 1)
    x_line = np.linspace(0, max(rotation) * 1.05, 100)
    axis.plot(x_line, np.polyval(fit, x_line), color="#d95f02", linewidth=1.5)
    axis.axhline(0, color="0.4", linewidth=.8)
    axis.set_xlabel("corrective-field rotation (1 - cosine to full)")
    axis.set_ylabel("matched probe - full-trained transfer (pp)")
    axis.set_title("Boundary drift tracks KV-field rotation")
    axis.grid(alpha=.2)
    axis.text(
        .03, .96,
        f"Spearman rho={spearman:.2f}, exact p={payload['spearman_exact_two_sided_permutation_p']:.3f}",
        transform=axis.transAxes, va="top", fontsize=8,
    )
    fig.savefig(args.figure, dpi=240)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
