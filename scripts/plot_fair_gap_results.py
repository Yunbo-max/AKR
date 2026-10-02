#!/usr/bin/env python3
"""Build the corrected equal-supervision and causal-control paper figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    return json.loads(path.read_text())


def readout_series(payload: dict, method: str):
    cells = [cell for cell in payload["results"].values() if cell["method"] == method]
    cells.sort(key=lambda cell: cell["support_k_per_class"])
    return ([cell["support_k_per_class"] for cell in cells],
            [100 * cell["accuracy"] for cell in cells])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    root = args.root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})

    datasets = [
        ("MarmAudio call type", "results/marmaudio_equal_support_readouts_7b_summary.json"),
        ("Dogs identity (lp 1 kHz)",
         "results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_readouts_7b_summary.json"),
        ("Watkins species", "results/beans_watkins_equal_support_readouts_7b_summary.json"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)
    for axis, (title, relative) in zip(axes, datasets):
        payload = load(root / relative)
        for method, label, marker in (
            ("nearest_centroid", "centroid", "o"),
            ("linear_probe", "ridge probe", "s"),
        ):
            x, y = readout_series(payload, method)
            axis.plot(x, y, marker=marker, linewidth=2, label=label)
        axis.set_title(title)
        axis.set_xlabel("support examples / class")
        axis.set_ylabel("accuracy (%)")
        axis.set_xticks(x)
        axis.grid(alpha=.2)
        axis.set_ylim(0, 90)
    native = load(root / "results/marmaudio_equal_support_native_query75_7b_summary.json")
    axes[0].axhline(100 * native["results"]["bare/sequence_sum"]["accuracy"],
                    color="0.35", linestyle="--", label="native candidate (0-shot)")
    icl_specs = (
        (0, 75, "results/marmaudio_equal_support_audio_icl_k{}_7b_summary.json", (1, 2, 4, 8)),
        (1, 139, "results/beans_dogs_lp1_equal_support_audio_icl_k{}_7b_summary.json", (1, 2)),
        (2, 339, "results/beans_watkins_equal_support_audio_icl_k{}_7b_summary.json", (1, 2, 4)),
    )
    for axis_index, expected_query, template, levels in icl_specs:
        points = []
        for k in levels:
            path = root / template.format(k)
            if not path.exists():
                continue
            payload = load(path)
            cell = payload["results"].get(str(k), {})
            if cell.get("n_query") != expected_query:
                continue
            points.append((k, 100 * cell["accuracy"]))
        if points:
            axes[axis_index].scatter(
                [point[0] for point in points], [point[1] for point in points],
                marker="x", s=55, linewidth=2, label="audio ICL",
            )
    candidate_points = []
    for k in (1, 2, 4, 8):
        path = root / f"results/marmaudio_equal_support_audio_icl_candidate_k{k}_7b_summary.json"
        if not path.exists():
            continue
        payload = load(path)
        cell = payload["results"].get(str(k), {})
        if cell.get("n_query") != 75:
            continue
        candidate_points.append((k, 100 * cell["accuracy"]))
    if candidate_points:
        axes[0].scatter(
            [point[0] for point in candidate_points],
            [point[1] for point in candidate_points],
            marker="+", s=70, linewidth=2, label="audio ICL candidate",
        )
    order_points = []
    for k, mode in ((2, "interleaved"), (8, "interleaved")):
        path = root / f"results/marmaudio_icl_order_control_k{k}_{mode}_7b_summary.json"
        if not path.exists():
            continue
        payload = load(path)
        if payload.get("n_query") == 75:
            order_points.append((k, 100 * payload["accuracy"]))
    if order_points:
        axes[0].scatter(
            [point[0] for point in order_points],
            [point[1] for point in order_points],
            marker="*", s=85, linewidth=.8, label="order-balanced ICL",
        )
    for axis_index, k, expected_query, artifact in (
        (1, 2, 139, "beans_dogs_lp1"),
        (2, 1, 339, "beans_watkins"),
    ):
        path = root / f"results/{artifact}_icl_order_control_k{k}_interleaved_7b_summary.json"
        if not path.exists():
            continue
        payload = load(path)
        if payload.get("n_query") == expected_query:
            axes[axis_index].scatter(
                [k], [100 * payload["accuracy"]], marker="*", s=85,
                linewidth=.8, label="order-balanced ICL",
            )
    for axis in axes:
        axis.legend(frameon=False, fontsize=7.5)
    fig.savefig(args.output_dir / "fair_fig1_equal_supervision.png", dpi=240)
    plt.close(fig)

    kv = load(root / "results/beans_dogs_relative_token_kv_lp1_fullvalid_summary.json")
    oracle = load(root / "results/marmaudio_oracle_kv_controls_fullprefill_lp1_7b_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    labels = {
        "fixed_mean": "fixed mean", "conditional_pooled": "conditional pooled",
        "conditional_tokenwise": "ordered tokenwise",
        "token_permuted": "token permuted", "random_field": "matched random",
    }
    for method, label in labels.items():
        cells = [cell for cell in kv["complete_cells"] if cell["method"] == method]
        cells.sort(key=lambda cell: cell["relative_alpha"])
        axes[0].plot([cell["relative_alpha"] for cell in cells],
                     [100 * cell["accuracy"] for cell in cells], marker="o", label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("relative KV norm alpha")
    axes[0].set_ylabel("Dogs validation accuracy (%)")
    axes[0].set_title("Query-label-free repair")
    axes[0].grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=7)
    alpha = [float(value) for value in oracle["registered_alphas"]]
    for method, metric, label in (
        ("correct_label", "accuracy", "correct target"),
        ("wrong_label", "wrong_target_hit", "chosen wrong target"),
        ("token_permuted", "accuracy", "audio-token permuted"),
        ("random_field", "accuracy", "matched random"),
    ):
        axes[1].plot(alpha, [100 * oracle["cells"][method][str(value)][metric]
                             for value in alpha], marker="o", label=label)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("relative KV norm alpha")
    axes[1].set_ylabel("target-hit rate (%)")
    axes[1].set_title("Full-prefill Oracle capacity (n=12)")
    axes[1].grid(alpha=.2)
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(args.output_dir / "fair_fig2_kv_controls.png", dpi=240)
    plt.close(fig)

    dogs = load(root / "results/kv_geometry_beans_dogs_train_lp1_7b_v2.json")["layers"]
    marm = load(root / "results/kv_geometry_lp1k_7b_v2.json")["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    for values, label in ((dogs, "Dogs balanced support"),
                          (marm, "Marm failure-selected")):
        layer = sorted(map(int, values))
        axes[0].plot(layer, [values[str(index)]["same_minus_different_cosine"]
                             for index in layer], label=label)
        axes[1].plot(layer, [values[str(index)]["global_effective_rank_centered"]
                             for index in layer], label=label)
    axes[0].set_title("Class separation of corrective gradients")
    axes[0].set_ylabel("same-label minus different-label cosine")
    axes[1].set_title("Corrective-field complexity")
    axes[1].set_ylabel("centered effective rank")
    for axis in axes:
        axis.set_xlabel("Thinker layer")
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=7.5)
    fig.savefig(args.output_dir / "fair_fig3_gradient_geometry.png", dpi=240)
    plt.close(fig)

    arbitrary_path = root / "results/marmaudio_arbitrary_labels_counterbalanced_summary.json"
    if arbitrary_path.exists():
        arbitrary = load(arbitrary_path)
        cells = arbitrary["mappings"]
        names = [cell["mapping"].replace("identity", "id") for cell in cells]
        position = np.arange(len(cells))
        width = .36
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
        axes[0].bar(position - width / 2,
                    [100 * cell["k0_accuracy"] for cell in cells], width,
                    label="K=0")
        axes[0].bar(position + width / 2,
                    [100 * cell["k1_accuracy"] for cell in cells], width,
                    label="K=1/class")
        axes[0].axhline(100 / 6, color="0.25", linestyle="--", linewidth=1.4,
                        label="six-way chance")
        axes[0].set_xticks(position, names)
        axes[0].set_ylabel("candidate accuracy (%)")
        axes[0].set_title("Counterbalanced arbitrary labels")
        axes[0].legend(frameon=False, fontsize=7.5)
        k0_dominance = [100 * max(cell["k0_prediction_counts"].values()) / 75
                        for cell in cells]
        k1_dominance = [100 * max(cell["k1_prediction_counts"].values()) / 75
                        for cell in cells]
        axes[1].bar(position - width / 2, k0_dominance, width, label="K=0")
        axes[1].bar(position + width / 2, k1_dominance, width, label="K=1/class")
        axes[1].set_xticks(position, names)
        axes[1].set_ylabel("largest output share (%)")
        axes[1].set_ylim(0, 105)
        axes[1].set_title("Output collapse survives support")
        axes[1].legend(frameon=False, fontsize=7.5)
        for axis in axes:
            axis.grid(axis="y", alpha=.2)
        fig.suptitle(
            "K1 mean 17.33%; not above frequency-preserving association null (p=.152)",
            fontsize=9,
        )
        fig.savefig(args.output_dir / "fair_fig4_arbitrary_binding.png", dpi=240)
        plt.close(fig)


if __name__ == "__main__":
    main()
