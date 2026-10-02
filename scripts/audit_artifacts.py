#!/usr/bin/env python3
"""Audit pairing, completeness, split isolation, and deterministic decoding artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)


def audit_paired(audit: Audit, root: Path, relative: str, expected: set[str]) -> None:
    path = root / relative
    rows = read_csv(path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["event_id"]].append(row)
    complete = all({row["condition"] for row in items} == expected for items in grouped.values())
    consistent = all(
        len({row["label"] for row in items}) == 1
        and len({row.get("split", "") for row in items}) == 1
        for items in grouped.values()
    )
    files_exist = all(Path(row["audio_path"]).is_file() for row in rows)
    audit.check(f"paired:{relative}", complete and consistent and files_exist,
                f"events={len(grouped)} rows={len(rows)} conditions={len(expected)}")


def audit_generation(audit: Audit, root: Path, predictions: str, manifest: str) -> None:
    prediction_rows = read_csv(root / predictions)
    manifest_rows = read_csv(root / manifest)
    expected = {(row["event_id"], row.get("condition", "full_0-8k")) for row in manifest_rows}
    observed = [(row["event_id"], row["condition"]) for row in prediction_rows]
    unique = len(observed) == len(set(observed))
    batch_one = all(row.get("batch_size", "1") == "1" for row in prediction_rows)
    audit.check(f"generation:{predictions}", unique and batch_one and set(observed) == expected,
                f"expected={len(expected)} observed={len(observed)} unique={unique} batch1={batch_one}")


def audit_fixed_kv(
    audit: Audit,
    root: Path,
    manifest: str,
    support_split: str,
    predictions: str,
) -> None:
    manifest_rows = read_csv(root / manifest)
    split = json.loads((root / support_split).read_text())
    prediction_rows = read_csv(root / predictions)
    by_event = {row["event_id"]: row for row in manifest_rows if row["condition"] == "lp_0-1000"}
    support = set(split["support_order"])
    support_train = all(by_event[event_id]["split"] == "train" for event_id in support)
    query = {row["event_id"] for row in prediction_rows}
    query_test = all(by_event[event_id]["split"] == "test" for event_id in query)
    identities = Counter(
        (row["event_id"], int(row["support_k"]), row["method"])
        for row in prediction_rows
    )
    expected = {
        (event_id, support_k, method)
        for event_id in query
        for support_k in (1, 5, 10, 20)
        for method in ("fixed_mean", "conditional")
    }
    complete = set(identities) == expected and all(value == 1 for value in identities.values())
    audit.check(f"fixed_kv:{predictions}",
                support_train and query_test and support.isdisjoint(query) and complete,
                f"support={len(support)} query={len(query)} train={support_train} "
                f"test={query_test} disjoint={support.isdisjoint(query)} complete={complete}")


def audit_marm_splits(audit: Audit, root: Path) -> None:
    manifest = read_csv(root / "data/manifests/marmaudio_expert_validation_interventions.csv")
    event_group = {row["event_id"]: row["recording_id"] for row in manifest}
    paths = sorted((root / "results").glob("conditional_kv_lp1k_7b_randomquery_seed*_split.json"))
    disjoint = True
    for path in paths:
        split = json.loads(path.read_text())
        query_groups = {event_group[event_id] for event_id in split["query_events"]}
        support_groups = {event_group[event_id] for event_id in split["support_order"]}
        disjoint &= query_groups.isdisjoint(support_groups)
    audit.check("marm_conditional_recording_disjoint", bool(paths) and disjoint,
                f"splits={len(paths)} disjoint={disjoint}")


def audit_simple_counts(audit: Audit, root: Path) -> None:
    expected = {
        "results/beans_zero_core_cap10_qwen7b.csv": (300, ("event_id",)),
        "results/marmaudio_controls_qwen7b.csv": (1800, ("event_id", "condition")),
        "results/marmaudio_prompt_reversed_qwen7b.csv": (300, ("event_id",)),
        "results/marmaudio_prompt_permuted_qwen7b.csv": (300, ("event_id",)),
        "results/beans_dogs_conditional_kv_lp1_7b_layer28_rank8_followup.csv":
            (278, ("event_id", "support_k", "method")),
    }
    for relative, (count, columns) in expected.items():
        rows = read_csv(root / relative)
        identities = [tuple(row[column] for column in columns) for row in rows]
        passed = len(rows) == count and len(identities) == len(set(identities))
        audit.check(f"count:{relative}", passed,
                    f"expected={count} observed={len(rows)} unique={len(set(identities))}")
    summaries = {
        "results/probe_7b_oof_summary.json": 546,
        "results/probe_lp1k_7b_oof_summary.json": 546,
        "results/aves_bio_probe_oof_summary.json": 546,
        "results/beans_dogs_probe_7b_summary.json": 139,
        "results/beans_watkins_probe_7b_summary.json": 339,
    }
    for relative, expected_n in summaries.items():
        value = json.loads((root / relative).read_text())
        observed = value.get("n", value.get("n_test"))
        audit.check(f"summary:{relative}", observed == expected_n,
                    f"expected={expected_n} observed={observed}")


def audit_aves_fixed_split(
    audit: Audit,
    root: Path,
    dataset: str,
    expected_total: int,
    expected_test: int,
) -> None:
    manifest_path = root / f"data/manifests/beans_{dataset}_protocol.csv"
    representation_dir = root / f"results/reps_beans_{dataset}_aves_bio"
    predictions_path = root / f"results/predictions_beans_{dataset}_aves_bio_fixed.csv"
    summary_path = root / f"results/summary_beans_{dataset}_aves_bio_fixed.json"
    manifest = read_csv(manifest_path)
    by_event = {row["event_id"]: row for row in manifest}
    files = sorted(representation_dir.glob("*.npz"))
    file_events = {path.stem for path in files}
    metadata_errors = 0
    representation_errors = 0
    for path in files:
        if path.stem not in by_event:
            metadata_errors += 1
            continue
        row = by_event[path.stem]
        try:
            with np.load(path, allow_pickle=False) as payload:
                expected_keys = {
                    "representation", "event_id", "label", "recording_id", "split", "model_id"
                }
                if set(payload.files) != expected_keys:
                    metadata_errors += 1
                    continue
                representation = payload["representation"]
                if (representation.shape != (1, 768)
                        or representation.dtype != np.float16
                        or not np.isfinite(representation).all()):
                    representation_errors += 1
                expected_recording = row.get("recording_id", row["event_id"])
                expected_metadata = {
                    "event_id": row["event_id"], "label": row["label"],
                    "recording_id": expected_recording, "split": row["split"],
                    "model_id": "AVES-bio-ONNX",
                }
                metadata_errors += sum(
                    str(payload[key]) != str(value)
                    for key, value in expected_metadata.items()
                )
        except Exception:
            representation_errors += 1
    predictions = read_csv(predictions_path)
    prediction_by_event = {row["event_id"]: row for row in predictions}
    test = {event: row for event, row in by_event.items() if row["split"] == "test"}
    prediction_errors = 0
    for event, row in prediction_by_event.items():
        if event not in test or row["target"] != test[event]["label"]:
            prediction_errors += 1
        elif (row["target"] == row["prediction"]) != (row["correct"] == "true"):
            prediction_errors += 1
    recomputed_accuracy = (
        sum(row["correct"] == "true" for row in predictions) / len(predictions)
        if predictions else 0.0
    )
    summary = json.loads(summary_path.read_text())
    passed = len(manifest) == expected_total and len(by_event) == expected_total
    passed &= len(files) == expected_total and file_events == set(by_event)
    passed &= metadata_errors == 0 and representation_errors == 0
    passed &= len(predictions) == expected_test and len(prediction_by_event) == expected_test
    passed &= set(prediction_by_event) == set(test) and prediction_errors == 0
    passed &= summary["n_train"] == sum(row["split"] == "train" for row in manifest)
    passed &= summary["n_valid"] == sum(row["split"] == "valid" for row in manifest)
    passed &= summary["n_test"] == expected_test
    passed &= abs(summary["test_accuracy"] - recomputed_accuracy) < 1e-12
    passed &= summary["protocol"] == "select_on_valid_refit_train_plus_valid_evaluate_test"
    audit.check(
        f"aves_bio_fixed_split:{dataset}",
        passed,
        f"manifest={len(manifest)} reps={len(files)} metadata_errors={metadata_errors} "
        f"representation_errors={representation_errors} test={len(predictions)} "
        f"prediction_errors={prediction_errors} accuracy={recomputed_accuracy:.6f}",
    )


def audit_quarantined_artifacts(audit: Audit, root: Path) -> None:
    authoritative = [
        root / "scripts/build_paper_tables.py",
        root / "scripts/build_fair_gap_tables.py",
        root / "scripts/plot_fair_gap_results.py",
        root / "FINAL_EXTENSION_RESULTS.md",
        root / "FAIR_GAP_RESULTS.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in authoritative)
    forbidden_paths = [
        path.relative_to(root).as_posix()
        for path in (root / "results/partial_extensions").glob("*")
    ]
    forbidden_paths += [
        "results/beans_watkins_conditional_kv_lp1_7b_eta300_partial_invalid.csv",
        "results/beans_zero_core_qwen7b_uncapped_partial_invalid.csv",
        "results/expert_qwen7b_partial_summary.csv",
        "results/expert_qwen7b_partial_summary.json",
    ]
    referenced = sorted(path for path in forbidden_paths if path in text)
    audit.check(
        "quarantined_partial_smoke_invalid_excluded",
        not referenced,
        f"quarantined={len(forbidden_paths)} referenced_by_authoritative={referenced}",
    )


def audit_dogs_full_lora(audit: Audit, root: Path) -> None:
    manifest = read_csv(root / "data/manifests/beans_dogs_protocol.csv")
    test = {row["event_id"]: row for row in manifest if row["split"] == "test"}
    predictions = read_csv(root / "results/beans_dogs_lora_full_7b.csv")
    by_event = {row["event_id"]: row for row in predictions}
    summary = json.loads((root / "results/beans_dogs_lora_full_7b_summary.json").read_text())
    paired = json.loads((root / "results/beans_dogs_lora_full_7b_paired_comparisons.json").read_text())
    history = json.loads((root / "results/lora_beans_dogs_7b_full/history.json").read_text())
    metadata_ok = len(by_event) == len(predictions) == len(test) == 139
    metadata_ok &= set(by_event) == set(test)
    metadata_ok &= all(
        row["target"] == test[event]["label"]
        and (row["prediction"] == row["target"]) == (row["correct"] == "true")
        and row.get("split") == "test"
        for event, row in by_event.items()
    )
    accuracy = sum(row["correct"] == "true" for row in predictions) / len(predictions)
    training_ok = len(history) == 1
    training_ok &= history[0].get("n_train") == 415
    training_ok &= history[0].get("n_valid_monitor") == 64
    training_ok &= (root / "results/lora_beans_dogs_7b_full/epoch_1/adapter_model.safetensors").is_file()
    summary_ok = summary["n"] == 139 and summary["coverage_verified_against_manifest"]
    summary_ok &= abs(summary["accuracy"] - accuracy) < 1e-12
    summary_ok &= summary["split"] == "test"
    summary_ok &= len(paired["comparisons"]) == 3
    summary_ok &= {row["reference"] for row in paired["comparisons"]} == {
        "native_generation", "frozen_qwen_probe", "aves_bio_probe"
    }
    summary_ok &= all(row["n"] == 139 for row in paired["comparisons"])
    audit.check(
        "dogs_full_train_lora_fixed_test",
        metadata_ok and training_ok and summary_ok,
        f"train={history[0].get('n_train')} valid_monitor={history[0].get('n_valid_monitor')} "
        f"test={len(predictions)} accuracy={accuracy:.6f}",
    )


def audit_marm_cross_frequency_geometry(audit: Audit, root: Path) -> None:
    conditions = [
        "full_0-8k", "lp_0-1000", "lp_0-2000",
        "lp_0-4000", "lp_0-6000", "lp_0-8000",
    ]
    registered = json.loads(
        (root / "results/marmaudio_equal_support_split_seed20260814.json").read_text()
    )["support_sets"]["2"]
    expected = set(registered)
    passed = len(expected) == 12
    details = []
    for condition in conditions:
        tag = condition.replace("-", "_")
        directory = root / f"results/gradients_marmaudio_equal_support_k2_{tag}_7b"
        files = sorted(directory.glob("*.pt"))
        split = json.loads((root / f"results/gradients_marmaudio_equal_support_k2_{tag}_7b_split.json").read_text())
        passed &= {path.stem for path in files} == expected and len(files) == 12
        passed &= set(split["support_order"]) == expected
        passed &= split["condition"] == condition and split["support_k_per_class"] == 2
        details.append(f"{condition}={len(files)}")
    comparison = json.loads(
        (root / "results/kv_geometry_marmaudio_equal_support_k2_cross_frequency_7b.json").read_text()
    )
    passed &= comparison["n_support"] == 12
    passed &= set(comparison["event_ids"]) == expected
    passed &= comparison["conditions"] == conditions
    passed &= len(comparison["layers"]) == 28
    numeric = []
    for layer in comparison["layers"].values():
        for condition in conditions:
            numeric.extend(value for value in layer[condition].values()
                           if isinstance(value, (int, float)))
    passed &= bool(numeric) and bool(np.isfinite(np.asarray(numeric)).all())
    passed &= (root / "results/figures/fig_cross_frequency_kv_geometry.png").is_file()
    audit.check(
        "marm_equal_support_cross_frequency_kv_geometry",
        passed,
        f"support=12 layers={len(comparison['layers'])} " + " ".join(details),
    )


def audit_fixed_split_frequency_probes(
    audit: Audit, root: Path, dataset: str, expected_split: tuple[int, int, int]
) -> None:
    verification_path = root / f"results/beans_{dataset}_frequency_representations_7b_audit.json"
    summary_path = root / f"results/beans_{dataset}_frequency_probe_7b_summary.json"
    if not verification_path.exists() or not summary_path.exists():
        audit.check(
            f"fixed_split_frequency_probe:{dataset}", False,
            "complete representation audit or combined summary missing",
        )
        return
    verification = json.loads(verification_path.read_text())
    summary = json.loads(summary_path.read_text())
    conditions = [
        "full_0-8k", "lp_0-1000", "lp_0-2000",
        "lp_0-4000", "lp_0-6000", "lp_0-8000",
    ]
    cells = {row["condition"]: row for row in summary["conditions"]}
    manifest = read_csv(root / f"data/manifests/beans_{dataset}_protocol.csv")
    test = {row["event_id"]: row for row in manifest if row["split"] == "test"}
    passed = verification["passed"] and set(cells) == set(conditions)
    passed &= tuple(summary["split_sizes"][key] for key in ("train", "valid", "test")) == expected_split
    details = []
    for condition in conditions:
        if condition == "full_0-8k":
            predictions_path = root / f"results/beans_{dataset}_probe_7b_test.csv"
        else:
            cutoff = int(condition.split("-")[-1]) // 1000
            predictions_path = root / f"results/beans_{dataset}_probe_lp{cutoff}_7b_test.csv"
        rows = read_csv(predictions_path)
        by_event = {row["event_id"]: row for row in rows}
        correct = sum(row["correct"] == "true" for row in rows)
        metadata_ok = len(rows) == len(by_event) == len(test) and set(by_event) == set(test)
        metadata_ok &= all(
            row["target"] == test[event]["label"]
            and (row["target"] == row["prediction"]) == (row["correct"] == "true")
            for event, row in by_event.items()
        )
        passed &= metadata_ok
        passed &= cells[condition]["n_train"] == expected_split[0]
        passed &= cells[condition]["n_valid"] == expected_split[1]
        passed &= cells[condition]["n_test"] == expected_split[2]
        passed &= abs(cells[condition]["test_accuracy"] - correct / len(rows)) < 1e-12
        for paired_key in (
            "paired_probe_minus_native",
            "paired_full_probe_minus_condition_probe",
        ):
            paired = cells[condition].get(paired_key, {})
            paired_values = [
                paired.get("delta_accuracy"),
                paired.get("bootstrap_ci_low"),
                paired.get("bootstrap_ci_high"),
                paired.get("mcnemar_exact_p"),
            ]
            passed &= paired.get("n") == expected_split[2]
            passed &= all(isinstance(value, (int, float)) for value in paired_values)
            if all(isinstance(value, (int, float)) for value in paired_values):
                passed &= bool(np.isfinite(np.asarray(paired_values)).all())
                passed &= paired["bootstrap_ci_low"] <= paired["delta_accuracy"]
                passed &= paired["delta_accuracy"] <= paired["bootstrap_ci_high"]
                passed &= 0.0 <= paired["mcnemar_exact_p"] <= 1.0
        details.append(f"{condition}={correct}/{len(rows)}")
    audit.check(
        f"fixed_split_frequency_probe:{dataset}", passed,
        f"split={expected_split} " + " ".join(details),
    )


def audit_frequency_probe_per_class(audit: Audit, root: Path) -> None:
    path = root / "results/beans_frequency_probe_per_class_7b.json"
    figure = root / "results/figures/fig_frequency_probe_per_class.png"
    if not path.exists() or not figure.exists():
        audit.check("frequency_probe_per_class", False, "summary or figure missing")
        return
    payload = json.loads(path.read_text())
    expected = {"dogs": (139, 10), "watkins": (339, 31)}
    passed = set(payload) == set(expected)
    details = []
    for dataset, (n_test, n_classes) in expected.items():
        cell = payload.get(dataset, {})
        classes = cell.get("classes", [])
        passed &= cell.get("n_test") == n_test and len(classes) == n_classes
        passed &= sum(item.get("n_test", 0) for item in classes) == n_test
        numeric = []
        for item in classes:
            numeric.extend(item.get("recall", {}).values())
            numeric.extend((item.get("full_minus_lp1_recall"), item.get("recall_range")))
        passed &= all(isinstance(value, (int, float)) for value in numeric)
        if numeric and all(isinstance(value, (int, float)) for value in numeric):
            passed &= bool(np.isfinite(np.asarray(numeric)).all())
        details.append(f"{dataset}={len(classes)}/{n_test}")
    audit.check("frequency_probe_per_class", passed, " ".join(details))


def audit_fixed_probe_cross_condition(
    audit: Audit, root: Path, dataset: str, expected_test: int
) -> None:
    predictions_path = root / f"results/beans_{dataset}_probe_fulltrained_cross_frequency_7b.csv"
    summary_path = root / f"results/beans_{dataset}_probe_fulltrained_cross_frequency_7b_summary.json"
    if not predictions_path.exists() or not summary_path.exists():
        audit.check(f"fixed_probe_cross_condition:{dataset}", False, "artifacts missing")
        return
    rows = read_csv(predictions_path)
    summary = json.loads(summary_path.read_text())
    conditions = [
        "full_0-8k", "lp_0-1000", "lp_0-2000",
        "lp_0-4000", "lp_0-6000", "lp_0-8000",
    ]
    grouped = {condition: [] for condition in conditions}
    for row in rows:
        if row.get("condition") in grouped:
            grouped[row["condition"]].append(row)
    passed = set(row["condition"] for row in summary["conditions"]) == set(conditions)
    passed &= len(rows) == expected_test * len(conditions)
    event_sets = []
    for condition in conditions:
        cell = grouped[condition]
        events = {row["event_id"] for row in cell}
        event_sets.append(events)
        passed &= len(cell) == len(events) == expected_test
        passed &= all(
            (row["target"] == row["prediction"]) == (row["correct"] == "true")
            for row in cell
        )
    passed &= all(events == event_sets[0] for events in event_sets[1:])
    full_summary = json.loads(
        (root / f"results/beans_{dataset}_probe_7b_summary.json").read_text()
    )
    passed &= summary["selected_layer"] == full_summary["selected_layer"]
    passed &= summary["alpha"] == full_summary["alpha"]
    passed &= set(summary["paired_full_minus_condition"]) == set(conditions[1:])
    boundary = summary.get("paired_condition_specific_minus_fulltrained_transfer", {})
    passed &= set(boundary) == set(conditions)
    for condition in conditions:
        cell = boundary.get(condition, {})
        numeric = [cell.get(key) for key in ("delta", "ci_low", "ci_high", "mcnemar_exact_p")]
        passed &= cell.get("n") == expected_test
        passed &= all(isinstance(value, (int, float)) for value in numeric)
        if all(isinstance(value, (int, float)) for value in numeric):
            passed &= cell["ci_low"] <= cell["delta"] <= cell["ci_high"]
            passed &= 0.0 <= cell["mcnemar_exact_p"] <= 1.0
    audit.check(
        f"fixed_probe_cross_condition:{dataset}", passed,
        f"rows={len(rows)} conditions={len(grouped)} test/condition={expected_test}",
    )


def audit_probe_transfer_matrix(
    audit: Audit, root: Path, dataset: str, expected_test: int
) -> None:
    csv_path = root / f"results/beans_{dataset}_probe_transfer_matrix_7b.csv"
    summary_path = root / f"results/beans_{dataset}_probe_transfer_matrix_7b_summary.json"
    figure_path = root / f"results/figures/fig_{dataset}_probe_transfer_matrix.png"
    if not csv_path.exists() or not summary_path.exists() or not figure_path.exists():
        audit.check(f"probe_transfer_matrix:{dataset}", False, "artifacts missing")
        return
    rows = read_csv(csv_path)
    summary = json.loads(summary_path.read_text())
    conditions = [
        "full_0-8k", "lp_0-1000", "lp_0-2000",
        "lp_0-4000", "lp_0-6000", "lp_0-8000",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("selection_mode", "")].append(row)
    expected_pairs = {(source, target) for source in conditions for target in conditions}
    matrix = np.asarray(summary.get("accuracy_matrix", []), dtype=float)
    shared = np.asarray(summary.get("full_shared_accuracy_matrix", []), dtype=float)
    passed = len(rows) == 72 and set(grouped) == {"source_selected", "full_shared"}
    passed &= all(
        {(row.get("source_condition"), row.get("target_condition")) for row in cell}
        == expected_pairs
        for cell in grouped.values()
    )
    passed &= summary.get("conditions") == conditions
    passed &= summary.get("n_test") == expected_test
    passed &= matrix.shape == (6, 6) and bool(np.isfinite(matrix).all())
    passed &= bool(((0 <= matrix) & (matrix <= 1)).all())
    passed &= shared.shape == (6, 6) and bool(np.isfinite(shared).all())
    passed &= bool(((0 <= shared) & (shared <= 1)).all())
    checks = summary.get("diagonal_reproduction", {})
    passed &= set(checks) == set(conditions)
    for index, condition in enumerate(conditions):
        cell = checks.get(condition, {})
        passed &= cell.get("absolute_difference") == 0.0
        if matrix.shape == (6, 6):
            passed &= cell.get("matrix_accuracy") == float(matrix[index, index])
    audit.check(
        f"probe_transfer_matrix:{dataset}", passed,
        f"cells={len(rows)} modes={len(grouped)} diagonal={len(checks)} "
        f"test/target={expected_test}",
    )


def audit_boundary_kv_alignment(audit: Audit, root: Path) -> None:
    path = root / "results/marmaudio_boundary_kv_alignment_7b.json"
    figure = root / "results/figures/fig_boundary_kv_alignment.png"
    if not path.exists() or not figure.exists():
        audit.check("boundary_kv_alignment", False, "summary or figure missing")
        return
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    expected = {
        "lp_0-1000", "lp_0-2000", "lp_0-4000", "lp_0-6000", "lp_0-8000"
    }
    numeric = [
        row.get(key)
        for row in rows
        for key in (
            "mean_corrective_field_cosine_to_full",
            "corrective_field_rotation_one_minus_cosine",
            "condition_specific_probe_accuracy",
            "fulltrained_transfer_accuracy",
            "decoder_boundary_drift_gap",
        )
    ]
    passed = payload.get("n_conditions") == 5 and len(rows) == 5
    passed &= {row.get("condition") for row in rows} == expected
    passed &= all(isinstance(value, (int, float)) for value in numeric)
    passed &= bool(np.isfinite(np.asarray(numeric)).all())
    for key in (
        "spearman_rho", "pearson_r",
        "spearman_exact_two_sided_permutation_p",
        "pearson_exact_two_sided_permutation_p",
    ):
        passed &= isinstance(payload.get(key), (int, float))
    passed &= 0 <= payload.get("spearman_exact_two_sided_permutation_p", -1) <= 1
    passed &= 0 <= payload.get("pearson_exact_two_sided_permutation_p", -1) <= 1
    audit.check(
        "boundary_kv_alignment", passed,
        f"conditions={len(rows)} spearman={payload.get('spearman_rho')}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = Audit()
    all_conditions = {
        "full_0-8k", "lp_0-1000", "lp_0-2000", "lp_0-4000", "lp_0-6000", "lp_0-8000",
        "remove_0-1000", "remove_1000-2000", "remove_2000-4000",
        "remove_4000-6000", "remove_6000-8000",
    }
    for manifest in (
        "data/manifests/marmaudio_expert_validation_interventions.csv",
        "data/manifests/beans_dogs_test_all_interventions.csv",
        "data/manifests/beans_watkins_test_all_interventions.csv",
    ):
        audit_paired(audit, root, manifest, all_conditions)
    generation_pairs = [
        ("results/expert_qwen7b_b1_clean.csv", "data/manifests/marmaudio_expert_validation_interventions.csv"),
        ("results/beans_dogs_frequency_qwen7b.csv", "data/manifests/beans_dogs_test_all_interventions.csv"),
        ("results/beans_watkins_frequency_qwen7b.csv", "data/manifests/beans_watkins_test_all_interventions.csv"),
    ]
    if (root / "results/expert_qwen3b_b1_all.csv").exists():
        generation_pairs.append(("results/expert_qwen3b_b1_all.csv",
                                 "data/manifests/marmaudio_expert_validation_interventions.csv"))
    for predictions, manifest in generation_pairs:
        if (root / predictions).exists():
            audit_generation(audit, root, predictions, manifest)
        else:
            audit.check(f"generation:{predictions}", False, "missing")
    for dataset in ("dogs", "watkins"):
        audit_fixed_kv(
            audit, root,
            f"data/manifests/beans_{dataset}_all_full_lp1.csv",
            f"results/beans_{dataset}_support_k_split.json",
            f"results/beans_{dataset}_conditional_kv_lp1_7b.csv",
        )
    audit_marm_splits(audit, root)
    audit_simple_counts(audit, root)
    audit_aves_fixed_split(audit, root, "dogs", 693, 139)
    audit_aves_fixed_split(audit, root, "watkins", 1695, 339)
    audit_dogs_full_lora(audit, root)
    audit_marm_cross_frequency_geometry(audit, root)
    audit_fixed_split_frequency_probes(audit, root, "dogs", (415, 139, 139))
    audit_fixed_split_frequency_probes(audit, root, "watkins", (1017, 339, 339))
    audit_fixed_probe_cross_condition(audit, root, "dogs", 139)
    audit_fixed_probe_cross_condition(audit, root, "watkins", 339)
    audit_probe_transfer_matrix(audit, root, "dogs", 139)
    audit_probe_transfer_matrix(audit, root, "watkins", 339)
    audit_boundary_kv_alignment(audit, root)
    audit_frequency_probe_per_class(audit, root)
    audit_quarantined_artifacts(audit, root)
    result = {"passed": audit.passed, "checks": audit.checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
