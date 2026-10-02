#!/usr/bin/env python3
"""Summarize MarmAudio nested-support scaling and paired saturation tests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired(a: dict[str, bool], b: dict[str, bool], rng: np.random.Generator,
           draws: int) -> dict:
    if set(a) != set(b):
        raise RuntimeError("paired event sets differ")
    events = sorted(a)
    av = np.asarray([a[event] for event in events])
    bv = np.asarray([b[event] for event in events])
    delta = av.astype(float) - bv.astype(float)
    indices = rng.integers(0, len(events), size=(draws, len(events)))
    bootstrap = delta[indices].mean(1)
    a_only = int(np.sum(av & ~bv)); b_only = int(np.sum(~av & bv))
    discordant = a_only + b_only
    return {
        "n": len(events), "accuracy_a": float(av.mean()), "accuracy_b": float(bv.mean()),
        "difference_a_minus_b": float(delta.mean()),
        "bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
        "a_only_correct": a_only, "b_only_correct": b_only,
        "mcnemar_exact_two_sided_p": (
            float(binomtest(a_only, discordant, .5).pvalue) if discordant else 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--icl-k1", type=Path, required=True)
    parser.add_argument("--icl-k2", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = read(args.readouts)
    cells = {}
    for row in rows:
        key = (int(row["support_k_per_class"]), row["method"])
        cells.setdefault(key, {})[row["event_id"]] = row["correct"] == "true"
    scaling = []
    for (k, method), mapped in sorted(cells.items()):
        scaling.append({"support_k_per_class": k, "method": method,
                        "n": len(mapped), "accuracy": float(np.mean(list(mapped.values())))})
    icl = {}
    for path in (args.icl_k1, args.icl_k2):
        for row in read(path):
            k = int(row["support_k_per_class"])
            icl.setdefault(k, {})[row["event_id"]] = row["correct"] == "true"
    support_to_decision = []
    for k in sorted(icl):
        result = paired(cells[(k, "linear_probe")], icl[k], rng, args.draws)
        support_to_decision.append({"support_k_per_class": k,
                                    "definition": "linear_probe minus audio_ICL", **result})
    saturation = []
    for method in ("linear_probe", "nearest_centroid"):
        saturation.append({"method": method, "comparison": "K16 minus K8",
                           **paired(cells[(16, method)], cells[(8, method)], rng, args.draws)})
    payload = {
        "protocol": "same 75 recording-disjoint queries and identical nested K<=8 prefixes; K16 extends the deterministic support order",
        "scaling": scaling, "support_to_decision_gap": support_to_decision,
        "k16_saturation_tests": saturation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
