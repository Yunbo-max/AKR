#!/usr/bin/env python3
"""Plot condition-matched frozen-probe versus native-generation accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


CONDITIONS = ["lp_0-1000", "lp_0-2000", "lp_0-4000", "lp_0-6000", "lp_0-8000", "full_0-8k"]
TICKS = ["1", "2", "4", "6", "8 LP", "Full"]


def keyed(path: Path, key: str) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {row["condition"]: row for row in payload[key]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = args.root.resolve() / "results"

    marm_native = keyed(results / "expert_qwen7b_b1_summary.json", "conditions")
    marm_probe_files = {
        "full_0-8k": results / "probe_7b_oof_summary.json",
        "lp_0-1000": results / "probe_lp1k_7b_oof_summary.json",
        "lp_0-2000": results / "probe_lp2000_7b_oof_summary.json",
        "lp_0-4000": results / "probe_lp4000_7b_oof_summary.json",
        "lp_0-6000": results / "probe_lp6000_7b_oof_summary.json",
        "lp_0-8000": results / "probe_lp8000_7b_oof_summary.json",
    }
    marm_probe = {
        condition: json.loads(path.read_text())["accuracy"]
        for condition, path in marm_probe_files.items()
    }
    marm_transfer = keyed(
        results / "probe_fulltrained_frequency_7b_oof_summary.json", "conditions"
    )
    tasks = [("MarmAudio call type", marm_probe,
              {condition: marm_transfer[condition]["accuracy"] for condition in CONDITIONS},
              {condition: marm_native[condition]["accuracy"] for condition in CONDITIONS})]
    for dataset, title in (("dogs", "BEANS Dogs individual"),
                           ("watkins", "BEANS Watkins species")):
        summary = json.loads(
            (results / f"beans_{dataset}_frequency_probe_7b_summary.json").read_text()
        )
        cells = {row["condition"]: row for row in summary["conditions"]}
        transfer = keyed(
            results / f"beans_{dataset}_probe_fulltrained_cross_frequency_7b_summary.json",
            "conditions",
        )
        tasks.append((
            title,
            {condition: cells[condition]["test_accuracy"] for condition in CONDITIONS},
            {condition: transfer[condition]["accuracy"] for condition in CONDITIONS},
            {condition: cells[condition]["native_generation_accuracy"] for condition in CONDITIONS},
        ))

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.65), sharey=True, constrained_layout=True)
    for axis, (title, probe, transfer, native) in zip(axes, tasks):
        x = list(range(len(CONDITIONS)))
        probe_values = [100 * probe[condition] for condition in CONDITIONS]
        transfer_values = [100 * transfer[condition] for condition in CONDITIONS]
        native_values = [100 * native[condition] for condition in CONDITIONS]
        axis.fill_between(x, native_values, probe_values, color="#92c5de", alpha=.28)
        axis.plot(
            x, probe_values, marker="o", color="#2166ac",
            label="full-supervision frozen probe",
        )
        axis.plot(
            x, native_values, marker="s", color="#b2182b",
            label="zero-shot native generation",
        )
        axis.plot(
            x, transfer_values, marker="^", linestyle="--", color="#1b7837",
            label="full-trained probe transfer",
        )
        axis.set_xticks(x, TICKS)
        axis.set_xlabel("observable cutoff (kHz)")
        axis.set_title(title)
        axis.grid(alpha=.25)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(loc="best", frameon=False)
    figure.suptitle(
        "Condition-matched decodability vs native output "
        "(supervision-unmatched diagnostic)"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
