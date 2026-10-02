#!/usr/bin/env python3
"""Report every class's fixed-test recall across observable low-pass cutoffs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("full_0-8k", "lp_0-1000", "lp_0-2000", "lp_0-4000", "lp_0-6000", "lp_0-8000")
TICKS = ("1", "2", "4", "6", "8")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dataset_payload(results: Path, dataset: str) -> dict:
    paths = [results / f"beans_{dataset}_probe_7b_test.csv"] + [
        results / f"beans_{dataset}_probe_lp{cutoff}_7b_test.csv"
        for cutoff in (1, 2, 4, 6, 8)
    ]
    by_condition = {}
    for condition, path in zip(CONDITIONS, paths):
        rows = read(path)
        mapped = {row["event_id"]: row for row in rows}
        if len(mapped) != len(rows):
            raise RuntimeError(f"duplicate events in {path}")
        by_condition[condition] = mapped
    event_sets = [set(mapped) for mapped in by_condition.values()]
    if any(events != event_sets[0] for events in event_sets[1:]):
        raise RuntimeError(f"unpaired events for {dataset}")
    for event in event_sets[0]:
        targets = {mapped[event]["target"] for mapped in by_condition.values()}
        if len(targets) != 1:
            raise RuntimeError(f"target mismatch for {dataset}/{event}")
    labels = sorted({row["target"] for row in by_condition[CONDITIONS[0]].values()})
    classes = []
    for label in labels:
        events = sorted(
            event for event, row in by_condition[CONDITIONS[0]].items()
            if row["target"] == label
        )
        recalls = {
            condition: float(np.mean([
                by_condition[condition][event]["correct"].lower() == "true"
                for event in events
            ]))
            for condition in CONDITIONS
        }
        classes.append({
            "label": label,
            "n_test": len(events),
            "recall": recalls,
            "full_minus_lp1_recall": recalls["full_0-8k"] - recalls["lp_0-1000"],
            "recall_range": max(recalls.values()) - min(recalls.values()),
        })
    return {
        "dataset": dataset,
        "conditions": list(CONDITIONS),
        "n_test": len(event_sets[0]),
        "classes": classes,
        "protocol": "all classes shown; fixed official test predictions; no class selected for reporting",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    results = args.root.resolve() / "results"
    payload = {
        dataset: dataset_payload(results, dataset) for dataset in ("dogs", "watkins")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(
        1, 2, figsize=(11.4, 9.2), gridspec_kw={"width_ratios": (1, 1.9)},
        constrained_layout=True,
    )
    image = None
    for axis, dataset, title in zip(
        axes, ("dogs", "watkins"), ("Dogs: 10 individuals", "Watkins: 31 species")
    ):
        cells = payload[dataset]["classes"]
        cells = sorted(cells, key=lambda cell: (-cell["full_minus_lp1_recall"], cell["label"]))
        matrix = np.asarray([
            [cell["recall"][condition] - cell["recall"]["full_0-8k"]
             for condition in CONDITIONS[1:]]
            for cell in cells
        ])
        image = axis.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(5), TICKS)
        axis.set_yticks(range(len(cells)), [cell["label"].replace("_", " ") for cell in cells])
        axis.set_xlabel("low-pass cutoff (kHz)")
        axis.set_title(title)
        axis.tick_params(axis="y", labelsize=6.5 if dataset == "watkins" else 8)
    figure.colorbar(image, ax=axes, label="class recall minus full-input recall")
    figure.suptitle("Class-resolved frequency sensitivity (all registered test classes)")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=220)
    plt.close(figure)
    print(json.dumps({dataset: {"n_test": value["n_test"], "n_classes": len(value["classes"])}
                      for dataset, value in payload.items()}, indent=2))


if __name__ == "__main__":
    main()
