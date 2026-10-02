#!/usr/bin/env python3
"""Paired uncertainty and exact discordance tests for frozen fair comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read(path: Path, predicate=lambda row: True) -> dict[str, dict[str, str]]:
    rows = [row for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
            if predicate(row)]
    result = {row["event_id"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate events after filtering: {path}")
    return result


def paired(
    name: str, a: dict[str, dict[str, str]], b: dict[str, dict[str, str]],
    rng: np.random.Generator, draws: int,
) -> dict:
    if set(a) != set(b):
        raise RuntimeError(f"event mismatch for {name}: {len(a)} vs {len(b)}")
    events = sorted(a)
    a_correct = np.asarray([a[event]["prediction"] == a[event]["target"] for event in events])
    b_correct = np.asarray([b[event]["prediction"] == b[event]["target"] for event in events])
    delta = a_correct.astype(float) - b_correct.astype(float)
    indices = rng.integers(0, len(events), size=(draws, len(events)))
    bootstrap = delta[indices].mean(1)
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    discordant = a_only + b_only
    p_value = float(binomtest(a_only, discordant, .5).pvalue) if discordant else 1.0
    return {
        "comparison": name, "n": len(events),
        "accuracy_a": float(a_correct.mean()), "accuracy_b": float(b_correct.mean()),
        "paired_difference_a_minus_b": float(delta.mean()),
        "bootstrap_95_ci": [float(value) for value in np.quantile(bootstrap, [.025, .975])],
        "a_only_correct": a_only, "b_only_correct": b_only,
        "mcnemar_exact_two_sided_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    marm_probe = read(
        root / "results/marmaudio_equal_support_readouts_7b.csv",
        lambda row: row["support_k_per_class"] == "1" and row["method"] == "linear_probe",
    )
    marm_probe_k2 = read(
        root / "results/marmaudio_equal_support_readouts_7b.csv",
        lambda row: row["support_k_per_class"] == "2" and row["method"] == "linear_probe",
    )
    marm_icl_free = read(root / "results/marmaudio_equal_support_audio_icl_k1_7b.csv")
    marm_icl_free_k2 = read(root / "results/marmaudio_equal_support_audio_icl_k2_7b.csv")
    marm_icl_candidate = read(
        root / "results/marmaudio_equal_support_audio_icl_candidate_k1_7b.csv"
    )
    marm_query = set(json.loads(
        (root / "results/marmaudio_equal_support_split_seed20260814.json").read_text()
    )["query_events"])
    marm_candidate_sum = read(
        root / "results/marmaudio_candidate_scoring_full546_7b.csv",
        lambda row: row["event_id"] in marm_query and row["prompt_name"] == "bare",
    )
    for row in marm_candidate_sum.values():
        row["prediction"] = row["prediction_sum"]
    marm_free = read(
        root / "results/expert_qwen7b_b1_clean.csv",
        lambda row: row["event_id"] in marm_query and row["condition"] == "full_0-8k",
    )
    marm_arbitrary_path = root / "results/marmaudio_arbitrary_labels_candidate_k0_k1_7b.csv"
    marm_arbitrary_k0 = read(
        marm_arbitrary_path, lambda row: row["support_k_per_class"] == "0"
    )
    marm_arbitrary_k1 = read(
        marm_arbitrary_path, lambda row: row["support_k_per_class"] == "1"
    )
    for row in list(marm_arbitrary_k0.values()) + list(marm_arbitrary_k1.values()):
        row["target"] = row["target_output"]
    dogs_probe_k1 = read(
        root / "results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_readouts_7b.csv",
        lambda row: row["support_k_per_class"] == "1" and row["method"] == "linear_probe",
    )
    dogs_probe_k2 = read(
        root / "results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_readouts_7b.csv",
        lambda row: row["support_k_per_class"] == "2" and row["method"] == "linear_probe",
    )
    dogs_icl = read(root / "results/beans_dogs_lp1_equal_support_audio_icl_k1_7b.csv")
    dogs_lora = read(root / "results/lora_beans_dogs_lp1_equal_support_k2_7b_valid.csv")
    relative = root / "results/beans_dogs_relative_token_kv_lp1_fullvalid.csv"
    kv_pooled = read(
        relative,
        lambda row: row["method"] == "conditional_pooled" and row["relative_alpha"] == "0.03",
    )
    kv_tokenwise = read(
        relative,
        lambda row: row["method"] == "conditional_tokenwise" and row["relative_alpha"] == "0.03",
    )
    kv_fixed = read(
        relative,
        lambda row: row["method"] == "fixed_mean" and row["relative_alpha"] == "0.03",
    )
    kv_permuted = read(
        relative,
        lambda row: row["method"] == "token_permuted" and row["relative_alpha"] == "0.03",
    )
    rng = np.random.default_rng(args.seed)
    comparisons = [
        paired("Marm K1 probe - zero-shot bare candidate sequence-sum", marm_probe,
               marm_candidate_sum, rng, args.draws),
        paired("Marm bare candidate sequence-sum - free generation", marm_candidate_sum,
               marm_free, rng, args.draws),
        paired("Marm K1 probe - K1 audio ICL free", marm_probe, marm_icl_free, rng, args.draws),
        paired("Marm K1 probe - K1 audio ICL candidate", marm_probe, marm_icl_candidate,
               rng, args.draws),
        paired("Marm K2 probe - K2 audio ICL free", marm_probe_k2, marm_icl_free_k2,
               rng, args.draws),
        paired("Marm arbitrary-label candidate K1 - K0", marm_arbitrary_k1,
               marm_arbitrary_k0, rng, args.draws),
        paired("Dogs K1 probe - K1 audio ICL free", dogs_probe_k1, dogs_icl, rng, args.draws),
        paired("Dogs K2 probe - K2 LoRA", dogs_probe_k2, dogs_lora, rng, args.draws),
        paired("Dogs conditional pooled - fixed mean at alpha=.03", kv_pooled, kv_fixed,
               rng, args.draws),
        paired("Dogs ordered tokenwise - permuted at alpha=.03", kv_tokenwise, kv_permuted,
               rng, args.draws),
        paired("Dogs ordered tokenwise - conditional pooled at alpha=.03", kv_tokenwise,
               kv_pooled, rng, args.draws),
    ]
    optional_specs = []
    marm_readouts = root / "results/marmaudio_equal_support_readouts_7b.csv"
    for k in (2, 4, 8):
        probe = read(
            marm_readouts,
            lambda row, selected=k: row["support_k_per_class"] == str(selected)
            and row["method"] == "linear_probe",
        )
        readouts = (("_candidate", "candidate"),) if k == 2 else (
            ("", "free"), ("_candidate", "candidate")
        )
        for suffix, label in readouts:
            path = root / f"results/marmaudio_equal_support_audio_icl{suffix}_k{k}_7b.csv"
            optional_specs.append((f"Marm K{k} probe - K{k} audio ICL {label}", probe, path))
    optional_specs.append((
        "Dogs K2 probe - K2 audio ICL free", dogs_probe_k2,
        root / "results/beans_dogs_lp1_equal_support_audio_icl_k2_7b.csv",
    ))
    watkins_readouts = root / "results/beans_watkins_equal_support_readouts_7b.csv"
    for k in (1, 2, 4):
        probe = read(
            watkins_readouts,
            lambda row, selected=k: row["support_k_per_class"] == str(selected)
            and row["method"] == "linear_probe",
        )
        optional_specs.append((
            f"Watkins K{k} probe - K{k} audio ICL free", probe,
            root / f"results/beans_watkins_equal_support_audio_icl_k{k}_7b.csv",
        ))
    for name, probe, path in optional_specs:
        if not path.exists():
            continue
        icl = read(path)
        if set(icl) != set(probe):
            # A resumable run may leave a partial CSV after an OOM. Such an
            # artifact is intentionally excluded until all registered queries
            # are present and the event sets match exactly.
            continue
        comparisons.append(paired(name, probe, icl, rng, args.draws))
    order_specs = (
        (
            "Marm K2 counterbalanced blocked - fixed grouped order",
            root / "results/marmaudio_icl_order_control_k2_blocked_7b.csv",
            root / "results/marmaudio_equal_support_audio_icl_k2_7b.csv",
        ),
        (
            "Marm K2 counterbalanced interleaved - fixed grouped order",
            root / "results/marmaudio_icl_order_control_k2_interleaved_7b.csv",
            root / "results/marmaudio_equal_support_audio_icl_k2_7b.csv",
        ),
        (
            "Marm K8 counterbalanced interleaved - fixed grouped order",
            root / "results/marmaudio_icl_order_control_k8_interleaved_7b.csv",
            root / "results/marmaudio_equal_support_audio_icl_k8_7b.csv",
        ),
    )
    for name, left_path, right_path in order_specs:
        if not left_path.exists() or not right_path.exists():
            continue
        left, right = read(left_path), read(right_path)
        if set(left) == set(right):
            comparisons.append(paired(name, left, right, rng, args.draws))
    for k, mode in ((2, "blocked"), (2, "interleaved"), (8, "interleaved")):
        path = root / f"results/marmaudio_icl_order_control_k{k}_{mode}_7b.csv"
        if not path.exists():
            continue
        probe = read(
            marm_readouts,
            lambda row, selected=k: row["support_k_per_class"] == str(selected)
            and row["method"] == "linear_probe",
        )
        order_control = read(path)
        if set(probe) == set(order_control):
            comparisons.append(paired(
                f"Marm K{k} probe - K{k} audio ICL {mode} order",
                probe, order_control, rng, args.draws,
            ))
    cross_dataset_order = (
        (
            "Dogs K2 counterbalanced interleaved - fixed grouped order",
            root / "results/beans_dogs_lp1_icl_order_control_k2_interleaved_7b.csv",
            root / "results/beans_dogs_lp1_equal_support_audio_icl_k2_7b.csv",
            dogs_probe_k2,
            "Dogs K2 probe - counterbalanced interleaved audio ICL",
        ),
        (
            "Watkins K1 counterbalanced interleaved - fixed grouped order",
            root / "results/beans_watkins_icl_order_control_k1_interleaved_7b.csv",
            root / "results/beans_watkins_equal_support_audio_icl_k1_7b.csv",
            read(
                watkins_readouts,
                lambda row: row["support_k_per_class"] == "1"
                and row["method"] == "linear_probe",
            ),
            "Watkins K1 probe - counterbalanced interleaved audio ICL",
        ),
    )
    for fixed_name, order_path, grouped_path, probe, probe_name in cross_dataset_order:
        if not order_path.exists():
            continue
        order_control = read(order_path)
        if grouped_path.exists():
            grouped = read(grouped_path)
            if set(order_control) == set(grouped):
                comparisons.append(paired(
                    fixed_name, order_control, grouped, rng, args.draws
                ))
        if set(order_control) == set(probe):
            comparisons.append(paired(
                probe_name, probe, order_control, rng, args.draws
            ))
    payload = {
        "bootstrap_draws": args.draws, "seed": args.seed,
        "inference": "paired example bootstrap; exact two-sided discordant-pair binomial test",
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
