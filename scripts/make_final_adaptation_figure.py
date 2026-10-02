#!/usr/bin/env python3
"""Render KV recovery plus LoRA/probe/specialist comparison as final Figure 3."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditional-diagnostic", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--kv-table", type=Path, required=True)
    parser.add_argument("--benchmark-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    diagnostic = [row for row in read(args.conditional_diagnostic)
                  if int(row["support_k"]) == 20 and float(row["eta"]) == 300]
    fixed = [row for row in diagnostic if row["method"] == "fixed_mean"]
    conditional = [row for row in diagnostic if row["method"] == "conditional"]
    oracle = read(args.oracle)
    oracle_columns = [column for column in oracle[0] if column.startswith("correct_eta_")]
    recovery = [
        0,
        sum(row["correct"] == "true" for row in fixed) / len(fixed),
        sum(row["correct"] == "true" for row in conditional) / len(conditional),
        max(sum(row[column] == "true" for row in oracle) / len(oracle)
            for column in oracle_columns),
    ]

    kv = read(args.kv_table)
    benchmarks = read(args.benchmark_table)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1), gridspec_kw={"width_ratios": [1, 1.15, 1.6]})

    names = ["Degraded", "Fixed KV", "Conditional KV", "Oracle KV"]
    colors = ["#8b8b8b", "#5b8db8", "#e0903c", "#4a9b67"]
    bars = axes[0].bar(names, recovery, color=colors)
    axes[0].set(ylabel="Eligible-failure recovery", ylim=(0, 1.08),
                title="a  MarmAudio recovery diagnostic")
    axes[0].tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, recovery):
        axes[0].text(bar.get_x() + bar.get_width()/2, value + .025, f"{value:.0%}", ha="center", fontsize=8)

    datasets = [row["dataset"].replace("BEANS ", "") for row in kv]
    positions = np.arange(len(kv)); width = .24
    for offset, key, label, color in [
        (-width, "baseline_accuracy", "Degraded", "#8b8b8b"),
        (0, "fixed_accuracy", "Fixed KV", "#5b8db8"),
        (width, "conditional_accuracy", "Conditional KV", "#e0903c"),
    ]:
        axes[1].bar(positions + offset, [float(row[key]) for row in kv], width, label=label, color=color)
    axes[1].set(xticks=positions, xticklabels=datasets, ylabel="Query accuracy", ylim=(0, .42),
                title="b  Label-free query adaptation")
    axes[1].legend(frameon=False, fontsize=8)

    task_names = ["MarmAudio", "Dogs", "Watkins"]
    direct = [] ; conditional_values = [] ; specialist = [] ; probe = []
    for task in task_names:
        rows = [row for row in benchmarks if task in row["dataset"]]
        direct.append(float(next(row for row in rows if "Qwen-7B zero-shot" in row["model_method"])["accuracy"]))
        conditional_row = next((row for row in rows if "conditional KV" in row["model_method"]), None)
        conditional_values.append(float(conditional_row["accuracy"]) if conditional_row else np.nan)
        reference = next((row for row in rows if "AVES" in row["model_method"]), None)
        if reference is None:
            reference = next((row for row in rows if "LoRA" in row["model_method"]), None)
        specialist.append(float(reference["accuracy"]) if reference else np.nan)
        probe.append(float(next(row for row in rows if "linear probe" in row["model_method"])["accuracy"]))
    positions = np.arange(3); width = .19
    for offset, values, label, color in [
        (-1.5*width, direct, "Qwen zero-shot", "#777777"),
        (-.5*width, conditional_values, "Conditional KV", "#e0903c"),
        (.5*width, specialist, "AVES / LoRA", "#6b6ecf"),
        (1.5*width, probe, "Qwen probe", "#4a9b67"),
    ]:
        axes[2].bar(positions + offset, values, width, label=label, color=color)
    axes[2].set(xticks=positions, xticklabels=task_names, ylabel="Accuracy", ylim=(0, 1),
                title="c  Generation, adaptation, and readout references")
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    axes[2].text(.01, .02, "AVES: MarmAudio; LoRA: Watkins", transform=axes[2].transAxes, fontsize=7)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
