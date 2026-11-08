#!/usr/bin/env python3
"""Select per-cell batches in the [60e9, 76e9] sustained-memory window.

Reads the 1050-measurement probe JSON and the final rebuttal run matrix
(docs/rebuttal/FINAL_CASES_ZH.md) and emits a machine-readable batch plan for
the production runner (``--batch-map``) plus Markdown rows for the plan doc.
No GPU work happens here.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

LO = 60e9
HI = 76e9
DATASETS = ("dogs", "watkins", "bats", "humbugdb")
CONDS = ("full", "lp:1000")

# Final matrix from docs/rebuttal/FINAL_CASES_ZH.md section 4.
MATRIX = {
    "c01": dict(models=("qwen7b", "qwen3b"), datasets=DATASETS, conds=CONDS),
    "c02": dict(models=("qwen7b",), datasets=DATASETS, conds=CONDS),
    "c03": dict(models=("qwen7b",), datasets=DATASETS, conds=CONDS),
    "c04": dict(models=("qwen7b",), datasets=DATASETS, conds=CONDS),
    "c05": dict(models=("qwen7b",), datasets=("dogs", "watkins"), conds=CONDS),
    "c06": dict(models=("qwen7b",), datasets=("dogs", "watkins"), conds=CONDS),
    "c07": dict(models=("qwen7b",), datasets=("dogs", "watkins"), conds=CONDS),
    "c08": dict(models=("qwen7b",), datasets=("dogs", "watkins"), conds=CONDS),
    "c09": dict(models=("qwen7b",), datasets=DATASETS, conds=CONDS),
    "c10": dict(models=("qwen7b",), datasets=("dogs", "watkins"), conds=CONDS),
    "c11": dict(models=("qwen7b",), datasets=DATASETS, conds=CONDS),
    "c12": dict(models=("glm4voice9b",), datasets=DATASETS, conds=CONDS),
    "c13": dict(models=("qwen7b",), datasets=DATASETS, conds=CONDS),
    "c14": dict(models=("qwen7b", "qwen3b"), datasets=("dogs", "watkins"), conds=CONDS),
    "c15": dict(models=("qwen7b", "naturelm"), datasets=DATASETS, conds=("full",)),
}


def workload_for(case, model):
    if model == "glm4voice9b":
        return "glm"
    if model == "naturelm":
        return "naturelm"
    if model == "beats":
        return "beats"
    if case == "c13":
        return "lora"
    if case == "c04":
        return "icl50"
    return "forward"


def build_curves(doc):
    curves = collections.defaultdict(dict)
    ooms = collections.defaultdict(set)
    for m in doc["measurements"]:
        if m.get("workload") == "maxlen_b1" or m.get("batch") is None:
            continue
        key = (m.get("case"), m.get("model"), m.get("condition"), m.get("dataset"))
        if m.get("status") != "ok" or not m.get("max_memory_allocated_bytes"):
            if m.get("status") == "oom":
                ooms[key].add(m["batch"])
            continue
        b = int(m["batch"])
        curves[key].setdefault(b, []).append(int(m["max_memory_allocated_bytes"]))
    # max over repeat measurements at the same batch
    return {k: {b: max(v) for b, v in by_b.items()} for k, by_b in curves.items()}, ooms


def select(curve):
    if not curve:
        return None, dict(status="no_measurement")
    inside = [b for b, p in curve.items() if LO <= p <= HI]
    below = sorted(b for b, p in curve.items() if p < LO)
    above = sorted(b for b, p in curve.items() if p > HI)
    if inside:
        b = max(inside)
        return b, dict(status="in_interval", peak=curve[b],
                        below=(below[-1], curve[below[-1]]) if below else None,
                        above=(above[0], curve[above[0]]) if above else None)
    if not below:
        b = min(curve)
        return None, dict(status="all_above", peak=curve[b], min_batch=b)
    if not above:
        b = max(below)
        return None, dict(status="all_below", peak=curve[b], max_batch=b)
    lo, hi = below[-1], above[0]
    frac = (LO - curve[lo]) / max(curve[hi] - curve[lo], 1)
    return None, dict(status="gap", below=(lo, curve[lo]), above=(hi, curve[hi]),
                      estimate_batch=round(lo + frac * (hi - lo), 1))


def cell_key(case, model, cond, dataset):
    return f"{case}|{model}|{cond.replace(':', '_')}|{dataset}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=Path,
                   default=Path("results/probe_mem_70gb_20261108.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/batch_floor60_plan_20261108.json"))
    p.add_argument("--print", action="store_true")
    args = p.parse_args(argv)
    doc = json.loads(args.probe.read_text())
    curves, ooms = build_curves(doc)
    cells, unavailable, extractors = {}, [], {}
    rows = []
    for case in sorted(MATRIX):
        spec = MATRIX[case]
        for model in spec["models"]:
            for cond in spec["conds"]:
                for dataset in spec["datasets"]:
                    wl = workload_for(case, model)
                    key = (case, model, cond, dataset)
                    curve = curves.get(key, {})
                    batch, info = select(curve)
                    entry = dict(case=case, model=model, condition=cond, dataset=dataset,
                                 workload=wl, batch=batch, **info,
                                 ooms=sorted(ooms.get(key, [])))
                    if batch is not None:
                        cells[cell_key(case, model, cond, dataset)] = dict(
                            batch=batch, peak_bytes=info["peak"], workload=wl)
                    else:
                        unavailable.append(entry)
                    rows.append(entry)
    # C03 BEATs feature extraction is a separate GPU engine per dataset.
    for cond in CONDS:
        for dataset in DATASETS:
            key = ("c03", "beats", cond, dataset)
            batch, info = select(curves.get(key, {}))
            entry = dict(case="c03", model="beats", condition=cond, dataset=dataset,
                         workload="beats", batch=batch, **info,
                         ooms=sorted(ooms.get(key, [])))
            if batch is not None:
                extractors[cell_key("c03", "beats", cond, dataset)] = dict(
                    batch=batch, peak_bytes=info["peak"], workload="beats")
            else:
                unavailable.append(entry)
            rows.append(entry)
    plan = dict(version=1, interval_bytes=[int(LO), int(HI)],
                source=str(args.probe), cells=cells, extractors=extractors,
                unavailable=unavailable,
                policy=("largest measured batch whose two-step peak is inside "
                        "[60e9,76e9]; unmeasured/out-of-window cells are listed "
                        "for packing or ladder follow-up"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"cells={len(cells)} extractors={len(extractors)} unavailable={len(unavailable)} -> {args.out}")
    if args.print:
        for r in rows:
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
