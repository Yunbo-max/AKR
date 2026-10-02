#!/usr/bin/env python3
"""Materialize compact paper tables for the corrected fair protocol."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_readouts(rows: list[dict], dataset: str, condition: str, payload: dict) -> None:
    for cell in payload["results"].values():
        rows.append({
            "dataset": dataset, "condition": condition,
            "support_k_per_class": cell["support_k_per_class"],
            "support_k_total": cell["support_k_total"], "method": cell["method"],
            "n_query": cell["n_query"], "accuracy": cell["accuracy"],
            "macro_f1": cell["macro_f1"], "query_label_free": True,
        })


def add_icl(
    rows: list[dict], dataset: str, condition: str, payload: dict, *, expected_query: int
) -> None:
    for cell in payload["results"].values():
        if cell["n_query"] != expected_query:
            raise RuntimeError(
                f"refusing partial {dataset} ICL cell: "
                f"n_query={cell['n_query']} expected={expected_query}"
            )
        rows.append({
            "dataset": dataset, "condition": condition,
            "support_k_per_class": cell["support_k_per_class"],
            "support_k_total": cell["support_k_total"],
            "method": f"audio_icl_{payload['readout']}",
            "n_query": cell["n_query"], "accuracy": cell["accuracy"],
            "macro_f1": cell["macro_f1"], "query_label_free": True,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/fair_gap_tables"))
    args = parser.parse_args()
    root = args.root
    table = []
    add_readouts(table, "MarmAudio", "full_0-8k",
                 load(root / "results/marmaudio_equal_support_readouts_7b_summary.json"))
    add_readouts(table, "BEANS Dogs", "lp_0-1000",
                 load(root / "results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_readouts_7b_summary.json"))
    add_readouts(table, "BEANS Watkins", "full_0-8k",
                 load(root / "results/beans_watkins_equal_support_readouts_7b_summary.json"))
    for dataset, condition, expected_query, relative in (
        ("MarmAudio", "full_0-8k", 75, "results/marmaudio_equal_support_audio_icl_k1_7b_summary.json"),
        ("MarmAudio", "full_0-8k", 75, "results/marmaudio_equal_support_audio_icl_k2_7b_summary.json"),
        ("MarmAudio", "full_0-8k", 75, "results/marmaudio_equal_support_audio_icl_candidate_k1_7b_summary.json"),
        ("BEANS Dogs", "lp_0-1000", 139, "results/beans_dogs_lp1_equal_support_audio_icl_k1_7b_summary.json"),
    ):
        payload = load(root / relative)
        add_icl(table, dataset, condition, payload, expected_query=expected_query)
    for dataset, condition, expected_query, relative in (
        ("MarmAudio", "full_0-8k", 75, "results/marmaudio_equal_support_audio_icl_candidate_k2_7b_summary.json"),
        ("MarmAudio", "full_0-8k", 75, "results/marmaudio_equal_support_audio_icl_k4_7b_summary.json"),
        ("MarmAudio", "full_0-8k", 75, "results/marmaudio_equal_support_audio_icl_candidate_k4_7b_summary.json"),
        ("MarmAudio", "full_0-8k", 75, "results/marmaudio_equal_support_audio_icl_k8_7b_summary.json"),
        ("MarmAudio", "full_0-8k", 75, "results/marmaudio_equal_support_audio_icl_candidate_k8_7b_summary.json"),
        ("BEANS Dogs", "lp_0-1000", 139, "results/beans_dogs_lp1_equal_support_audio_icl_k2_7b_summary.json"),
        ("BEANS Watkins", "full_0-8k", 339, "results/beans_watkins_equal_support_audio_icl_k1_7b_summary.json"),
        ("BEANS Watkins", "full_0-8k", 339, "results/beans_watkins_equal_support_audio_icl_k2_7b_summary.json"),
        ("BEANS Watkins", "full_0-8k", 339, "results/beans_watkins_equal_support_audio_icl_k4_7b_summary.json"),
    ):
        path = root / relative
        if path.exists():
            add_icl(
                table, dataset, condition, load(path), expected_query=expected_query
            )
    write(args.output_dir / "table_equal_supervision.csv", table)

    kv = load(root / "results/beans_dogs_relative_token_kv_lp1_fullvalid_summary.json")
    write(args.output_dir / "table_relative_kv_validation.csv", [
        {**cell, "split": "official_valid", "support_k_per_class": 2,
         "support_k_total": 20, "query_label_free": True}
        for cell in kv["complete_cells"]
    ])
    oracle = load(root / "results/marmaudio_oracle_kv_controls_fullprefill_lp1_7b_summary.json")
    oracle_rows = []
    for method, by_alpha in oracle["cells"].items():
        for alpha, cell in by_alpha.items():
            oracle_rows.append({
                "scope": oracle["scope"], "method": method, "relative_alpha": alpha,
                **cell, "query_label_used": method != "random_field",
            })
    write(args.output_dir / "table_oracle_controls.csv", oracle_rows)

    steering_rows = []
    for relative, dataset in (
        ("results/marmaudio_audio_silence_steering_k1_7b_summary.json", "MarmAudio"),
        ("results/beans_dogs_lp1_audio_silence_steering_icl_options_k1_7b_summary.json",
         "BEANS Dogs A-J"),
    ):
        path = root / relative
        if not path.exists():
            continue
        payload = load(path)
        steering_rows.extend([
            {"dataset": dataset, "state": "baseline", "beta": 0,
             "support_k_total": payload["support_k_total"], "n_query": payload["n_query"],
             **payload["query_baseline"]},
            {"dataset": dataset, "state": "steered", "beta": payload["selected_beta"],
             "support_k_total": payload["support_k_total"], "n_query": payload["n_query"],
             **payload["query_steered"]},
        ])
    if steering_rows:
        write(args.output_dir / "table_audio_silence_steering.csv", steering_rows)

    class_path = root / "results/beans_dogs_probe_routed_class_kv_lp1_valid_summary.json"
    if class_path.exists():
        payload = load(class_path)
        write(args.output_dir / "table_probe_routed_class_kv_validation.csv",
              payload["complete_cells"])
    arbitrary_class_path = root / "results/beans_dogs_probe_routed_class_kv_AJ_lp1_valid_summary.json"
    if arbitrary_class_path.exists():
        payload = load(arbitrary_class_path)
        write(args.output_dir / "table_probe_routed_class_kv_AJ_validation.csv",
              payload["complete_cells"])

    arbitrary_panel_path = root / "results/marmaudio_arbitrary_labels_counterbalanced_summary.json"
    if arbitrary_panel_path.exists():
        payload = load(arbitrary_panel_path)
        write(args.output_dir / "table_marm_arbitrary_label_panel.csv", payload["mappings"])

    order_rows = []
    for k, mode in ((2, "blocked"), (2, "interleaved"), (8, "interleaved")):
        path = root / f"results/marmaudio_icl_order_control_k{k}_{mode}_7b_summary.json"
        if not path.exists():
            continue
        payload = load(path)
        if payload.get("n_query") != 75:
            continue
        order_rows.append({
            "dataset": "MarmAudio",
            "support_k_per_class": k,
            "support_k_total": payload["support_k_total"],
            "order_mode": mode,
            "n_query": payload["n_query"],
            "accuracy": payload["accuracy"],
            "macro_f1": payload["macro_f1"],
            "last_label_copy_rate": payload["last_label_copy_rate"],
            "copy_rate_when_last_label_is_wrong": payload[
                "copy_rate_when_last_label_is_wrong"
            ],
            "accuracy_when_target_differs_from_last_label": payload[
                "accuracy_when_target_differs_from_last_label"
            ],
        })
    for dataset, artifact, k, expected_query in (
        ("BEANS Dogs", "beans_dogs_lp1", 2, 139),
        ("BEANS Watkins", "beans_watkins", 1, 339),
    ):
        path = root / f"results/{artifact}_icl_order_control_k{k}_interleaved_7b_summary.json"
        if not path.exists():
            continue
        payload = load(path)
        if payload.get("n_query") != expected_query:
            continue
        order_rows.append({
            "dataset": dataset,
            "support_k_per_class": k,
            "support_k_total": payload["support_k_total"],
            "order_mode": "interleaved",
            "n_query": payload["n_query"],
            "accuracy": payload["accuracy"],
            "macro_f1": payload["macro_f1"],
            "last_label_copy_rate": payload["last_label_copy_rate"],
            "copy_rate_when_last_label_is_wrong": payload[
                "copy_rate_when_last_label_is_wrong"
            ],
            "accuracy_when_target_differs_from_last_label": payload[
                "accuracy_when_target_differs_from_last_label"
            ],
        })
    if order_rows:
        write(args.output_dir / "table_marm_icl_order_controls.csv", order_rows)
    positional_path = root / "results/icl_order_controls_combined_7b.json"
    if positional_path.exists():
        positional = load(positional_path)
        if positional.get("cells"):
            write(
                args.output_dir / "table_icl_position_copying.csv",
                positional["cells"],
            )

    cross_prompt_path = root / "results/beans_dogs_AJ_cross_prompt_7b_summary.json"
    if cross_prompt_path.exists():
        payload = load(cross_prompt_path)
        cross_prompt_rows = []
        for prompt in payload["prompts"]:
            for method, key in (("native", "native"), ("class_routed_pooled", "pooled_alpha_03")):
                cross_prompt_rows.append({
                    "prompt": prompt["name"], "method": method, "n_query": prompt["n"],
                    "accuracy": prompt[key]["accuracy"],
                    "macro_f1": prompt[key]["macro_f1"],
                    "invalid_response_rate": prompt[key]["invalid_response_rate"],
                    "paired_pooled_minus_native": (
                        prompt["paired_accuracy_gain"] if method == "class_routed_pooled" else ""
                    ),
                })
        write(args.output_dir / "table_cross_prompt_pooled_kv.csv", cross_prompt_rows)

    beans_zero_path = root / "results/beans_zero_targets_fullscan_cap10_qwen7b_summary.json"
    if beans_zero_path.exists():
        payload = load(beans_zero_path)
        write(args.output_dir / "table_beans_zero_7b.csv", [
            {"component": component, **cell}
            for component, cell in payload.items()
        ])


if __name__ == "__main__":
    main()
