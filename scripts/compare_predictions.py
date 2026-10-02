#!/usr/bin/env python3
"""Paired event-level comparison between two prediction artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from animal_omni.bootstrap import paired_accuracy_delta_ci


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--a",type=Path,required=True);parser.add_argument("--b",type=Path,required=True)
    parser.add_argument("--condition",default="full_0-8k");parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=10000);parser.add_argument("--seed",type=int,default=20250813)
    args=parser.parse_args()
    def load(path):
        with path.open(newline="",encoding="utf-8") as f:
            return {r["event_id"]:r for r in csv.DictReader(f) if r.get("condition",args.condition)==args.condition}
    a,b=load(args.a),load(args.b);common=sorted(a.keys()&b.keys())
    targets=[a[k].get("target") for k in common]
    if any(a[k].get("target")!=b[k].get("target") for k in common):raise ValueError("target mismatch")
    result={"condition":args.condition,"n":len(common),"delta_definition":"accuracy_a_minus_b",
            **paired_accuracy_delta_ci(targets,[a[k].get("prediction") or None for k in common],
                                       [b[k].get("prediction") or None for k in common],samples=args.samples,seed=args.seed)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":main()
