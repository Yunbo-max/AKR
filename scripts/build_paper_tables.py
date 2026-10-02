#!/usr/bin/env python3
"""Build paper-ready benchmark and adaptation tables from authoritative artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from animal_omni.metrics import classification_metrics


MARM_LABELS = ["Infant Cry", "Phee", "Seep", "Trill", "Tsik", "Twitter"]


def load(path: Path):
    return json.loads(path.read_text())


def condition(summary, name):
    return next(row for row in summary["conditions"] if row["condition"] == name)


def csv_metrics(path: Path, labels: list[str]):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = classification_metrics(
        [row["target"] for row in rows],
        [row["prediction"] or None for row in rows],
        labels,
    )
    return {"n": len(rows), **result}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); results = root / "results"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    marm3 = condition(load(results / "expert_qwen3b_b1_all_summary.json"), "full_0-8k")
    marm7 = condition(load(results / "expert_qwen7b_b1_summary.json"), "full_0-8k")
    marm_probe = load(results / "probe_7b_oof_summary.json")
    aves_probe = load(results / "aves_bio_probe_oof_summary.json")
    specialist = csv_metrics(results / "expert_marmaudio_specialist.csv", MARM_LABELS)
    dogs_freq = condition(load(results / "beans_dogs_frequency_qwen7b_summary.json"), "full_0-8k")
    dogs_probe = load(results / "beans_dogs_probe_7b_summary.json")
    dogs_aves = load(results / "summary_beans_dogs_aves_bio_fixed.json")
    dogs_lora = load(results / "beans_dogs_lora_full_7b_summary.json")
    dogs_kv = load(results / "beans_dogs_conditional_kv_lp1_7b_layer28_rank8_followup_summary.json")
    dogs_cond = next(row for row in dogs_kv["adapted"] if row["method"] == "conditional")
    watkins_freq = condition(load(results / "beans_watkins_frequency_qwen7b_summary.json"), "full_0-8k")
    watkins_probe = load(results / "beans_watkins_probe_7b_summary.json")
    watkins_aves = load(results / "summary_beans_watkins_aves_bio_fixed.json")
    watkins_lora = load(results / "beans_watkins_lora_7b_summary.json")
    watkins_kv = load(results / "beans_watkins_conditional_kv_lp1_7b_summary.json")
    watkins_cond = next(row for row in watkins_kv["adapted"]
                        if row["support_k"] == 20 and row["method"] == "conditional")

    main_rows = [
        {"task": "Call type", "dataset": "MarmAudio", "model_method": "Official ResNet-50",
         "input": "original 0–48 kHz", "protocol": "fixed expert corpus", **specialist},
        {"task": "Call type", "dataset": "MarmAudio", "model_method": "AVES-bio + linear probe",
         "input": "Qwen-matched 0–8 kHz", "protocol": "nested recording-group OOF", "n": aves_probe["n"],
         "accuracy": aves_probe["accuracy"], "macro_f1": aves_probe["macro_f1"]},
        {"task": "Call type", "dataset": "MarmAudio", "model_method": "Qwen-3B zero-shot",
         "input": "observable full 0–8 kHz", "protocol": "fixed expert corpus", "n": marm3["n"],
         "accuracy": marm3["accuracy"], "macro_f1": marm3["macro_f1"]},
        {"task": "Call type", "dataset": "MarmAudio", "model_method": "Qwen-7B zero-shot",
         "input": "observable full 0–8 kHz", "protocol": "fixed expert corpus", "n": marm7["n"],
         "accuracy": marm7["accuracy"], "macro_f1": marm7["macro_f1"]},
        {"task": "Call type", "dataset": "MarmAudio", "model_method": "Qwen-7B + linear probe",
         "input": "observable full 0–8 kHz", "protocol": "nested recording-group OOF", "n": marm_probe["n"],
         "accuracy": marm_probe["accuracy"], "macro_f1": marm_probe["macro_f1"]},
        {"task": "Individual", "dataset": "BEANS Dogs", "model_method": "Qwen-7B zero-shot",
         "input": "observable full 0–8 kHz", "protocol": "official fixed test", "n": dogs_freq["n"],
         "accuracy": dogs_freq["accuracy"], "macro_f1": dogs_freq["macro_f1"]},
        {"task": "Individual", "dataset": "BEANS Dogs", "model_method": "Qwen-7B Thinker LoRA",
         "input": "observable full 0–8 kHz", "protocol": "one epoch; official fixed test", "n": dogs_lora["n"],
         "accuracy": dogs_lora["accuracy"], "macro_f1": dogs_lora["macro_f1"]},
        {"task": "Individual", "dataset": "BEANS Dogs", "model_method": "Qwen-7B + linear probe",
         "input": "observable full 0–8 kHz", "protocol": "official fixed test", "n": dogs_probe["n_test"],
         "accuracy": dogs_probe["test_accuracy"], "macro_f1": dogs_probe["test_macro_f1"]},
        {"task": "Individual", "dataset": "BEANS Dogs", "model_method": "AVES-bio + linear probe",
         "input": "Qwen-matched 0–8 kHz", "protocol": "official fixed test", "n": dogs_aves["n_test"],
         "accuracy": dogs_aves["test_accuracy"], "macro_f1": dogs_aves["test_macro_f1"]},
        {"task": "Species", "dataset": "BEANS Watkins", "model_method": "Qwen-7B zero-shot",
         "input": "observable full 0–8 kHz", "protocol": "paired official fixed test", "n": watkins_freq["n"],
         "accuracy": watkins_freq["accuracy"], "macro_f1": watkins_freq["macro_f1"]},
        {"task": "Species", "dataset": "BEANS Watkins", "model_method": "Qwen-7B Thinker LoRA",
         "input": "observable full 0–8 kHz", "protocol": "one epoch; official fixed test", "n": watkins_lora["n"],
         "accuracy": watkins_lora["accuracy"], "macro_f1": watkins_lora["macro_f1"]},
        {"task": "Species", "dataset": "BEANS Watkins", "model_method": "Qwen-7B + linear probe",
         "input": "observable full 0–8 kHz", "protocol": "official fixed test", "n": watkins_probe["n_test"],
         "accuracy": watkins_probe["test_accuracy"], "macro_f1": watkins_probe["test_macro_f1"]},
        {"task": "Species", "dataset": "BEANS Watkins", "model_method": "AVES-bio + linear probe",
         "input": "Qwen-matched 0–8 kHz", "protocol": "official fixed test", "n": watkins_aves["n_test"],
         "accuracy": watkins_aves["test_accuracy"], "macro_f1": watkins_aves["test_macro_f1"]},
    ]
    write_csv(args.output_dir / "table_main_benchmarks.csv", main_rows)

    marm_cond = load(results / "conditional_kv_lp1k_7b_randomquery_multiseed_summary.json")
    oracle_rows = list(csv.DictReader((results / "oracle_kv_lp1k_7b.csv").open()))
    oracle_columns = [column for column in oracle_rows[0] if column.startswith("correct_eta_")]
    oracle_best = max(sum(row[column] == "true" for row in oracle_rows) / len(oracle_rows)
                      for column in oracle_columns)
    adaptation = [
        {"dataset": "MarmAudio", "protocol": "5 random recording-group query splits; K=20",
         "baseline_accuracy": marm_cond["aggregate"]["baseline_accuracy"]["mean"],
         "fixed_accuracy": marm_cond["aggregate"]["fixed_mean_accuracy"]["mean"],
         "conditional_accuracy": marm_cond["aggregate"]["conditional_accuracy"]["mean"],
         "oracle_recovery": oracle_best,
         "evidence_status": "exploratory_pre_equal_support_protocol",
         "note": "means over splits; Oracle on eligible failures; not a primary paper result"},
        {"dataset": "BEANS Dogs", "protocol": "fixed test; K=20 layer28 rank8 follow-up",
         "baseline_accuracy": dogs_kv["baseline"]["accuracy"],
         "fixed_accuracy": next(row for row in dogs_kv["adapted"] if row["method"] == "fixed_mean")["accuracy"],
         "conditional_accuracy": dogs_cond["accuracy"], "oracle_recovery": "",
         "evidence_status": "exploratory_test_selected_do_not_claim",
         "note": "no full-correct/lp1-wrong Oracle-eligible examples; invalid for primary claim"},
        {"dataset": "BEANS Watkins", "protocol": "fixed test; K=20 layer0 rank4",
         "baseline_accuracy": watkins_kv["baseline"]["accuracy"],
         "fixed_accuracy": next(row for row in watkins_kv["adapted"]
                                if row["support_k"] == 20 and row["method"] == "fixed_mean")["accuracy"],
         "conditional_accuracy": watkins_cond["accuracy"], "oracle_recovery": 1.0,
         "evidence_status": "exploratory_undercovered_support_do_not_claim",
         "note": "Oracle is only 5 eligible examples; K=20 total cannot cover 31 classes"},
    ]
    write_csv(args.output_dir / "table_kv_adaptation.csv", adaptation)

    beans_zero = load(results / "beans_zero_targets_fullscan_cap10_qwen7b_summary.json")
    zero_rows = [{"component": key, **value} for key, value in beans_zero.items()]
    write_csv(args.output_dir / "table_beans_zero_full.csv", zero_rows)

    frequency_probe_rows = []
    marm_native = {
        row["condition"]: row
        for row in load(results / "expert_qwen7b_b1_summary.json")["conditions"]
    }
    marm_probe_paths = {
        "full_0-8k": "probe_7b_oof_summary.json",
        "lp_0-1000": "probe_lp1k_7b_oof_summary.json",
        "lp_0-2000": "probe_lp2000_7b_oof_summary.json",
        "lp_0-4000": "probe_lp4000_7b_oof_summary.json",
        "lp_0-6000": "probe_lp6000_7b_oof_summary.json",
        "lp_0-8000": "probe_lp8000_7b_oof_summary.json",
    }
    marm_transfer = {
        row["condition"]: row
        for row in load(results / "probe_fulltrained_frequency_7b_oof_summary.json")["conditions"]
    }
    for condition_name, filename in marm_probe_paths.items():
        probe = load(results / filename)
        native = marm_native[condition_name]
        frequency_probe_rows.append({
            "task": "Call type", "dataset": "MarmAudio",
            "condition": condition_name, "n": probe["n"],
            "native_accuracy": native["accuracy"],
            "probe_accuracy": probe["accuracy"],
            "probe_macro_f1": probe["macro_f1"],
            "fulltrained_transfer_probe_accuracy": marm_transfer[condition_name]["accuracy"],
            "condition_specific_minus_fulltrained_transfer": (
                probe["accuracy"] - marm_transfer[condition_name]["accuracy"]
            ),
            "condition_specific_minus_transfer_ci_low": "",
            "condition_specific_minus_transfer_ci_high": "",
            "condition_specific_minus_transfer_mcnemar_p": "",
            "probe_minus_native_accuracy": probe["accuracy"] - native["accuracy"],
            "probe_minus_native_ci_low": "",
            "probe_minus_native_ci_high": "",
            "probe_minus_native_mcnemar_p": "",
            "full_minus_condition_probe_accuracy": "",
            "full_minus_condition_ci_low": "",
            "full_minus_condition_ci_high": "",
            "full_minus_condition_mcnemar_p": "",
            "protocol": "condition-specific nested recording-group OOF",
            "comparison_role": "supervision-unmatched upper-ceiling diagnostic",
        })
    for dataset, task in (("dogs", "Individual"), ("watkins", "Species")):
        payload = load(results / f"beans_{dataset}_frequency_probe_7b_summary.json")
        transfer = {
            row["condition"]: row
            for row in load(
                results / f"beans_{dataset}_probe_fulltrained_cross_frequency_7b_summary.json"
            )["conditions"]
        }
        transfer_summary = load(
            results / f"beans_{dataset}_probe_fulltrained_cross_frequency_7b_summary.json"
        )
        boundary_drift = transfer_summary[
            "paired_condition_specific_minus_fulltrained_transfer"
        ]
        for cell in payload["conditions"]:
            gap = cell["paired_probe_minus_native"]
            frequency = cell["paired_full_probe_minus_condition_probe"]
            frequency_probe_rows.append({
                "task": task, "dataset": f"BEANS {dataset.title()}",
                "condition": cell["condition"], "n": cell["n_test"],
                "native_accuracy": cell["native_generation_accuracy"],
                "probe_accuracy": cell["test_accuracy"],
                "probe_macro_f1": cell["test_macro_f1"],
                "fulltrained_transfer_probe_accuracy": transfer[cell["condition"]]["accuracy"],
                "condition_specific_minus_fulltrained_transfer": boundary_drift[
                    cell["condition"]
                ]["delta"],
                "condition_specific_minus_transfer_ci_low": boundary_drift[
                    cell["condition"]
                ]["ci_low"],
                "condition_specific_minus_transfer_ci_high": boundary_drift[
                    cell["condition"]
                ]["ci_high"],
                "condition_specific_minus_transfer_mcnemar_p": boundary_drift[
                    cell["condition"]
                ]["mcnemar_exact_p"],
                "probe_minus_native_accuracy": cell["probe_minus_native_accuracy"],
                "probe_minus_native_ci_low": gap["bootstrap_ci_low"],
                "probe_minus_native_ci_high": gap["bootstrap_ci_high"],
                "probe_minus_native_mcnemar_p": gap["mcnemar_exact_p"],
                "full_minus_condition_probe_accuracy": frequency["delta_accuracy"],
                "full_minus_condition_ci_low": frequency["bootstrap_ci_low"],
                "full_minus_condition_ci_high": frequency["bootstrap_ci_high"],
                "full_minus_condition_mcnemar_p": frequency["mcnemar_exact_p"],
                "protocol": "condition-specific official fixed split",
                "comparison_role": "supervision-unmatched upper-ceiling diagnostic",
            })
    write_csv(args.output_dir / "table_frequency_probe_gap.csv", frequency_probe_rows)
    transfer_rows = []
    for dataset in ("dogs", "watkins"):
        with (results / f"beans_{dataset}_probe_transfer_matrix_7b.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            for row in csv.DictReader(stream):
                transfer_rows.append({"dataset": f"BEANS {dataset.title()}", **row})
    write_csv(
        args.output_dir / "table_frequency_decoder_transfer_matrix.csv", transfer_rows
    )
    boundary_kv = load(results / "marmaudio_boundary_kv_alignment_7b.json")
    boundary_kv_rows = [
        {
            **row,
            "spearman_rho_all_conditions": boundary_kv["spearman_rho"],
            "spearman_exact_p": boundary_kv[
                "spearman_exact_two_sided_permutation_p"
            ],
        }
        for row in boundary_kv["rows"]
    ]
    write_csv(args.output_dir / "table_boundary_kv_alignment.csv", boundary_kv_rows)
    (args.output_dir / "paper_metrics.json").write_text(json.dumps({
        "main_benchmarks": main_rows,
        "kv_adaptation": adaptation,
        "beans_zero_full": zero_rows,
        "frequency_probe_gap": frequency_probe_rows,
        "frequency_decoder_transfer_matrix": transfer_rows,
        "boundary_kv_alignment": boundary_kv,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
