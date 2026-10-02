#!/usr/bin/env python3
"""Nested recording-grouped OOF linear probe over layerwise representations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier


def score(y, prediction): return f1_score(y, prediction, average="macro", zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1, 10, 100])
    args = parser.parse_args()
    files = sorted(args.representation_dir.glob("*.npz"))
    records = [np.load(path, allow_pickle=False) for path in files]
    x = np.stack([r["representation"].astype(np.float32) for r in records])
    y = np.array([str(r["label"]) for r in records])
    groups = np.array([str(r["recording_id"]) for r in records])
    event_ids = np.array([str(r["event_id"]) for r in records])
    layers = list(range(x.shape[1]))
    outer = StratifiedGroupKFold(args.outer_folds, shuffle=True, random_state=20250813)
    oof = np.empty(len(y), dtype=object); selections = []
    for fold, (train, test) in enumerate(outer.split(x[:,0], y, groups), 1):
        inner = StratifiedGroupKFold(args.inner_folds, shuffle=True, random_state=20250813 + fold)
        candidates = []
        for layer in layers:
            for alpha in args.alphas:
                values = []
                for inner_train_rel, validation_rel in inner.split(x[train,layer], y[train], groups[train]):
                    tr, va = train[inner_train_rel], train[validation_rel]
                    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
                    model.fit(x[tr,layer], y[tr]); values.append(score(y[va], model.predict(x[va,layer])))
                candidates.append((float(np.mean(values)), layer, alpha))
        best_score, layer, alpha = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
        model.fit(x[train,layer], y[train]); oof[test] = model.predict(x[test,layer])
        selections.append({"fold": fold, "layer": layer, "alpha": alpha,
                           "inner_macro_f1": best_score, "n_train": len(train), "n_test": len(test)})
    rows = [{"event_id": e, "target": target, "prediction": prediction,
             "correct": str(target == prediction).lower(), "recording_id": group}
            for e,target,prediction,group in zip(event_ids,y,oof,groups)]
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.output_predictions.open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary={"n":len(y),"accuracy":accuracy_score(y,oof),"macro_f1":score(y,oof),
             "protocol":"nested_stratified_recording_group_oof","selections":selections}
    args.output_summary.write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
