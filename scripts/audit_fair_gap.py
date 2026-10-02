#!/usr/bin/env python3
"""Audit the equal-supervision 12-hour protocol and promoted artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


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


def audit_candidate_scoring(audit: Audit, root: Path) -> None:
    manifest = read_csv(root / "data/manifests/marmaudio_expert_validation.csv")
    rows = read_csv(root / "results/marmaudio_candidate_scoring_full546_7b.csv")
    identities = [(row["event_id"], row["prompt_name"]) for row in rows]
    expected = {
        (row["event_id"], prompt_name)
        for row in manifest
        for prompt_name in ("bare", "definition")
    }
    audit.check(
        "candidate_scoring_full546",
        set(identities) == expected and len(identities) == len(set(identities)),
        f"expected={len(expected)} observed={len(rows)} unique={len(set(identities))}",
    )


def audit_support_split(
    audit: Audit,
    root: Path,
    name: str,
    manifest_path: str,
    split_path: str,
    expected_k: tuple[int, ...],
    query_split: str | None,
    require_recording_disjoint: bool,
) -> None:
    rows = read_csv(root / manifest_path)
    by_event = {row["event_id"]: row for row in rows}
    split = json.loads((root / split_path).read_text())
    query = set(split["query_events"])
    support_sets = split["support_sets"]
    passed = set(map(int, support_sets)) == set(expected_k)
    details = []
    previous: set[str] = set()
    for k in expected_k:
        support = support_sets[str(k)]
        counts = Counter(by_event[event]["label"] for event in support)
        current = set(support)
        passed &= len(counts) > 1 and set(counts.values()) == {k}
        passed &= previous.issubset(current) and current.isdisjoint(query)
        if query_split is not None:
            passed &= all(by_event[event]["split"] == "train" for event in current)
            passed &= all(by_event[event]["split"] == query_split for event in query)
        if require_recording_disjoint:
            support_groups = {by_event[event]["recording_id"] for event in current}
            query_groups = {by_event[event]["recording_id"] for event in query}
            passed &= support_groups.isdisjoint(query_groups)
        previous = current
        details.append(f"K{k}={len(current)}")
    audit.check(name, passed, f"query={len(query)} " + " ".join(details))


def audit_readouts(audit: Audit, root: Path) -> None:
    expected = {
        "results/marmaudio_equal_support_readouts_7b_summary.json":
            ({1: 6, 2: 12, 4: 24, 8: 48}, 75),
        "results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_readouts_7b_summary.json":
            ({1: 10, 2: 20}, 139),
        "results/beans_watkins_equal_support_readouts_7b_summary.json":
            ({1: 31, 2: 62, 4: 124}, 339),
    }
    for relative, (support_sizes, query_n) in expected.items():
        payload = json.loads((root / relative).read_text())
        passed = True
        for k, support_n in support_sizes.items():
            for method in ("nearest_centroid", "linear_probe"):
                cell = payload["results"][f"K={k}/{method}"]
                passed &= cell["support_k_total"] == support_n
                passed &= cell["n_query"] == query_n
        audit.check(f"equal_support_readouts:{relative}", passed,
                    f"K={sorted(support_sizes)} query={query_n}")
    native = json.loads(
        (root / "results/marmaudio_equal_support_native_query75_7b_summary.json").read_text()
    )
    audit.check(
        "marm_native_readouts_same_query",
        native["n_query"] == 75 and set(native["results"]) == {
            "free_generation", "bare/mean_token_logprob", "bare/sequence_sum",
            "definition/mean_token_logprob", "definition/sequence_sum",
        },
        f"query={native['n_query']} readouts={len(native['results'])}",
    )


def audit_icl(audit: Audit, root: Path) -> None:
    expected = {
        "results/marmaudio_equal_support_audio_icl_k1_7b.csv": 75,
        "results/marmaudio_equal_support_audio_icl_k2_7b.csv": 75,
        "results/marmaudio_equal_support_audio_icl_candidate_k1_7b.csv": 75,
        "results/beans_dogs_lp1_equal_support_audio_icl_k1_7b.csv": 139,
    }
    for relative, count in expected.items():
        rows = read_csv(root / relative)
        events = [row["event_id"] for row in rows]
        audit.check(f"icl:{relative}", len(rows) == count and len(set(events)) == count,
                    f"expected={count} observed={len(rows)} unique={len(set(events))}")
    optional = (
        ("results/marmaudio_equal_support_audio_icl_candidate_k2_7b.csv",
         "results/marmaudio_equal_support_audio_icl_candidate_k2_7b_summary.json", 75, 2),
        ("results/marmaudio_equal_support_audio_icl_k4_7b.csv",
         "results/marmaudio_equal_support_audio_icl_k4_7b_summary.json", 75, 4),
        ("results/marmaudio_equal_support_audio_icl_candidate_k4_7b.csv",
         "results/marmaudio_equal_support_audio_icl_candidate_k4_7b_summary.json", 75, 4),
        ("results/marmaudio_equal_support_audio_icl_k8_7b.csv",
         "results/marmaudio_equal_support_audio_icl_k8_7b_summary.json", 75, 8),
        ("results/marmaudio_equal_support_audio_icl_candidate_k8_7b.csv",
         "results/marmaudio_equal_support_audio_icl_candidate_k8_7b_summary.json", 75, 8),
        ("results/beans_dogs_lp1_equal_support_audio_icl_k2_7b.csv",
         "results/beans_dogs_lp1_equal_support_audio_icl_k2_7b_summary.json", 139, 2),
        ("results/beans_watkins_equal_support_audio_icl_k1_7b.csv",
         "results/beans_watkins_equal_support_audio_icl_k1_7b_summary.json", 339, 1),
        ("results/beans_watkins_equal_support_audio_icl_k2_7b.csv",
         "results/beans_watkins_equal_support_audio_icl_k2_7b_summary.json", 339, 2),
        ("results/beans_watkins_equal_support_audio_icl_k4_7b.csv",
         "results/beans_watkins_equal_support_audio_icl_k4_7b_summary.json", 339, 4),
    )
    for csv_relative, summary_relative, count, support_k in optional:
        csv_path, summary_path = root / csv_relative, root / summary_relative
        if not csv_path.exists() and not summary_path.exists():
            continue
        passed = csv_path.exists() and summary_path.exists()
        rows = read_csv(csv_path) if csv_path.exists() else []
        events = [row["event_id"] for row in rows]
        passed &= len(rows) == count and len(set(events)) == count
        passed &= all(int(row["support_k_per_class"]) == support_k for row in rows)
        if summary_path.exists():
            payload = json.loads(summary_path.read_text())
            cell = payload.get("results", {}).get(str(support_k), {})
            passed &= cell.get("n_query") == count
            passed &= cell.get("support_k_per_class") == support_k
        audit.check(
            f"optional_full_icl:{csv_relative}", passed,
            f"K={support_k} expected={count} observed={len(rows)} unique={len(set(events))}",
        )
    arbitrary_outputs = [
        ("identity", root / "results/marmaudio_arbitrary_labels_candidate_k0_k1_7b.csv")
    ] + [
        (f"shift{shift}", root / f"results/marmaudio_arbitrary_labels_candidate_shift{shift}_k0_k1_7b.csv")
        for shift in range(1, 6)
    ]
    for mapping_name, arbitrary_path in arbitrary_outputs:
        if not arbitrary_path.exists():
            audit.check(
                f"marm_arbitrary_label_candidate:{mapping_name}", False, "output missing"
            )
            continue
        rows = read_csv(arbitrary_path)
        cells = defaultdict(list)
        for row in rows:
            cells[int(row["support_k_per_class"])].append(row)
        passed = set(cells) == {0, 1}
        passed &= all(len(cell) == 75 for cell in cells.values())
        passed &= all(len({row["event_id"] for row in cell}) == 75 for cell in cells.values())
        passed &= all(row["target_output"] in set("ABCDEF") for row in rows)
        passed &= all(row["prediction"] in set("ABCDEF") for row in rows)
        audit.check(
            f"marm_arbitrary_label_candidate:{mapping_name}",
            passed,
            f"K0={len(cells.get(0, []))} K1={len(cells.get(1, []))}",
        )


def audit_relative_kv(audit: Audit, root: Path) -> None:
    rows = read_csv(root / "results/beans_dogs_relative_token_kv_lp1_fullvalid.csv")
    middle = {0.003, 0.01, 0.03}
    methods = {
        "fixed_mean", "conditional_pooled", "conditional_tokenwise",
        "token_permuted", "random_field",
    }
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], float(row["relative_alpha"]))].append(row)
    passed = True
    query_sets = []
    maximum_norm_error = 0.0
    for method in methods:
        for alpha in middle:
            cell = grouped[(method, alpha)]
            passed &= len(cell) == 139
            passed &= all(row["query_split"] == "valid" for row in cell)
            query_sets.append({row["event_id"] for row in cell})
            maximum_norm_error = max(
                maximum_norm_error,
                max(abs(float(row["applied_ratio_mean"]) - alpha) for row in cell),
            )
    passed &= all(events == query_sets[0] for events in query_sets[1:])
    passed &= maximum_norm_error < 5e-4
    test_outputs = list((root / "results").glob("beans_dogs_relative_token_kv_lp1_test*"))
    passed &= not test_outputs
    audit.check(
        "relative_kv_validation_gate",
        passed,
        f"complete_cells={len(methods) * len(middle)} query=139 "
        f"max_norm_error={maximum_norm_error:.3g} fair_test_outputs={len(test_outputs)}",
    )


def audit_class_routed_kv(audit: Audit, root: Path) -> None:
    """Check both name-label and arbitrary-label validation-only KV studies."""
    methods = {
        "probe_class_pooled", "probe_class_tokenwise", "probe_class_permuted",
    }
    alphas = {0.003, 0.01, 0.03}
    specifications = (
        (
            "class_routed_kv_name_labels",
            "results/beans_dogs_probe_routed_class_kv_lp1_valid.csv",
            "results/beans_dogs_probe_routed_class_kv_lp1_valid_summary.json",
            False,
            None,
        ),
        (
            "class_routed_kv_arbitrary_labels",
            "results/beans_dogs_probe_routed_class_kv_AJ_lp1_valid.csv",
            "results/beans_dogs_probe_routed_class_kv_AJ_lp1_valid_summary.json",
            True,
            set("ABCDEFGHIJ"),
        ),
    )
    for name, csv_relative, summary_relative, expect_native, output_labels in specifications:
        csv_path = root / csv_relative
        summary_path = root / summary_relative
        if not csv_path.exists() or not summary_path.exists():
            audit.check(name, False, "predictions or summary missing")
            continue
        rows = read_csv(csv_path)
        repair = [row for row in rows if row["method"] in methods]
        grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
        for row in repair:
            grouped[(row["method"], float(row["relative_alpha"]))].append(row)
        passed = set(grouped) == {(method, alpha) for method in methods for alpha in alphas}
        query_sets = []
        maximum_norm_error = 0.0
        for cell in grouped.values():
            passed &= len(cell) == 139
            passed &= all(row["query_split"] == "valid" for row in cell)
            query_sets.append({row["event_id"] for row in cell})
            maximum_norm_error = max(
                maximum_norm_error,
                max(abs(float(row["applied_ratio_mean"]) - float(row["relative_alpha"]))
                    for row in cell),
            )
        passed &= bool(query_sets) and all(events == query_sets[0] for events in query_sets[1:])
        passed &= maximum_norm_error < 5e-4
        native = [row for row in rows if row["method"] == "native"]
        passed &= (len(native) == 139) if expect_native else (len(native) == 0)
        if output_labels is not None:
            passed &= all(row.get("target_output") in output_labels for row in rows)
            passed &= all(
                row["prediction"] == "" or row["prediction"] in output_labels for row in rows
            )
        summary = json.loads(summary_path.read_text())
        passed &= len(summary["complete_cells"]) == (10 if expect_native else 9)
        passed &= not summary["incomplete_cells"]
        passed &= summary["gate"]["passed"] is False
        passed &= summary["gate"]["test_action"] == "no_test"
        audit.check(
            name,
            passed,
            f"repair_rows={len(repair)} native_rows={len(native)} "
            f"max_norm_error={maximum_norm_error:.3g} gate={summary['gate']['passed']}",
        )


def audit_lora(audit: Audit, root: Path) -> None:
    combined = read_csv(root / "data/manifests/beans_dogs_lp1_equal_support_k2_fullvalid.csv")
    support = [row for row in combined if row["equal_support_role"] == "labeled_support"]
    registered_query = [row for row in combined if row["equal_support_role"] == "untouched_query"]
    evaluation = read_csv(root / "results/lora_beans_dogs_lp1_equal_support_k2_7b_valid.csv")
    counts = Counter(row["label"] for row in support)
    passed = len(support) == 20 and set(counts.values()) == {2}
    passed &= all(row["split"] == "train" for row in support)
    passed &= len(registered_query) == 139
    passed &= len(evaluation) == 139 and len({row["event_id"] for row in evaluation}) == 139
    passed &= {row["event_id"] for row in evaluation} == {
        row["event_id"] for row in registered_query
    }
    audit.check("equal_support_lora_dogs", passed,
                f"support={len(support)} classes={len(counts)} valid={len(evaluation)}")


def audit_oracle_controls(audit: Audit, root: Path) -> None:
    payload = json.loads((root / "results/marmaudio_oracle_kv_controls_fullprefill_lp1_7b_summary.json").read_text())
    cells = payload["cells"]
    passed = payload["n_events"] == 12 and payload["scope"] == "full_prefill_tokens"
    passed &= cells["correct_label"]["0.0003"]["accuracy"] == 1.0
    passed &= cells["wrong_label"]["0.001"]["wrong_target_hit"] >= 0.9
    passed &= all(cell["accuracy"] == 0 for cell in cells["random_field"].values())
    audit.check("oracle_label_specific_capacity_controls", passed,
                "n=12; correct and wrong-label target effects checked; random null checked")


def audit_paired_statistics(audit: Audit, root: Path) -> None:
    payload = json.loads((root / "results/fair_gap_paired_statistics.json").read_text())
    names = {row["comparison"] for row in payload["comparisons"]}
    required = {
        "Marm K1 probe - zero-shot bare candidate sequence-sum",
        "Marm bare candidate sequence-sum - free generation",
        "Marm K1 probe - K1 audio ICL free",
        "Marm K1 probe - K1 audio ICL candidate",
        "Marm K2 probe - K2 audio ICL free",
        "Marm arbitrary-label candidate K1 - K0",
        "Dogs K1 probe - K1 audio ICL free",
        "Dogs K2 probe - K2 LoRA",
        "Dogs conditional pooled - fixed mean at alpha=.03",
        "Dogs ordered tokenwise - permuted at alpha=.03",
        "Dogs ordered tokenwise - conditional pooled at alpha=.03",
    }
    passed = payload["bootstrap_draws"] == 20000 and required.issubset(names)
    passed &= all(row["n"] in {75, 139, 339} for row in payload["comparisons"])
    audit.check("paired_fair_gap_statistics", passed,
                f"comparisons={len(names)} draws={payload['bootstrap_draws']}")


def audit_audio_silence(audit: Audit, root: Path) -> None:
    marm = json.loads((root / "results/marmaudio_audio_silence_steering_k1_7b_summary.json").read_text())
    passed = marm["support_k_total"] == 6 and marm["n_query"] == 75
    dogs_path = root / "results/beans_dogs_lp1_audio_silence_steering_icl_options_k1_7b_summary.json"
    detail = "MarmAudio support=6 query=75"
    if dogs_path.exists():
        dogs = json.loads(dogs_path.read_text())
        passed &= dogs["support_k_total"] == 10 and dogs["n_query"] == 139
        passed &= dogs.get("icl_options") is True
        detail += "; Dogs A-J support=10 query=139"
    else:
        passed = False
        detail += "; Dogs A-J output missing"
    audit.check("matched_audio_silence_steering", passed, detail)


def audit_arbitrary_panel_summary(audit: Audit, root: Path) -> None:
    path = root / "results/marmaudio_arbitrary_labels_counterbalanced_summary.json"
    if not path.exists():
        audit.check("marm_arbitrary_counterbalanced_summary", False, "summary missing")
        return
    payload = json.loads(path.read_text())
    passed = payload["n_mappings"] == 6
    passed &= payload["n_unique_queries"] == 75
    passed &= payload["n_query_mapping_pairs"] == 450
    passed &= len(payload["mappings"]) == 6
    passed &= payload["draws"] == 20000
    audit.check(
        "marm_arbitrary_counterbalanced_summary",
        passed,
        f"mappings={payload['n_mappings']} queries={payload['n_unique_queries']} "
        f"pairs={payload['n_query_mapping_pairs']}",
    )


def audit_cross_prompt_pooled(audit: Audit, root: Path) -> None:
    summary_path = root / "results/beans_dogs_AJ_cross_prompt_7b_summary.json"
    prediction_paths = [
        root / "results/beans_dogs_AJ_cross_prompt_paraphrase_7b.csv",
        root / "results/beans_dogs_AJ_cross_prompt_reverse_order_7b.csv",
    ]
    if not summary_path.exists() and not any(path.exists() for path in prediction_paths):
        return
    passed = summary_path.exists() and all(path.exists() for path in prediction_paths)
    details = []
    for path in prediction_paths:
        rows = read_csv(path) if path.exists() else []
        grouped = defaultdict(list)
        for row in rows:
            grouped[row.get("method", "")].append(row)
        passed &= set(grouped) == {"native", "probe_class_pooled"}
        passed &= all(len(cell) == 139 for cell in grouped.values())
        passed &= all(
            len({row["event_id"] for row in cell}) == 139 for cell in grouped.values()
        )
        details.append(f"{path.stem}={len(rows)}")
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        passed &= len(payload.get("prompts", [])) == 2
        passed &= all(prompt.get("n") == 139 for prompt in payload.get("prompts", []))
    audit.check(
        "dogs_AJ_cross_prompt_pooled_validation",
        passed,
        " ".join(details),
    )


def audit_beans_zero_7b(audit: Audit, root: Path) -> None:
    manifest_path = root / "data/manifests/beans_zero_targets_fullscan_cap10.csv"
    predictions_path = root / "results/beans_zero_targets_fullscan_cap10_qwen7b.csv"
    summary_path = root / "results/beans_zero_targets_fullscan_cap10_qwen7b_summary.json"
    if not predictions_path.exists() or not summary_path.exists():
        audit.check("beans_zero_7b_fullscan", False, "predictions or summary missing")
        return
    source = read_csv(manifest_path)
    predictions = read_csv(predictions_path)
    source_events = {row["event_id"] for row in source}
    prediction_events = {row["event_id"] for row in predictions}
    summary = json.loads(summary_path.read_text())
    components = {key for key in summary if key != "overall"}
    expected_components = {row["dataset_name"] for row in source}
    passed = len(source) == len(predictions) == 2950
    passed &= len(prediction_events) == len(predictions)
    passed &= source_events == prediction_events
    passed &= components == expected_components and len(components) == 12
    passed &= summary["overall"]["n"] == 2950
    audit.check(
        "beans_zero_7b_fullscan",
        passed,
        f"predictions={len(predictions)} unique={len(prediction_events)} "
        f"components={len(components)}",
    )


def audit_icl_order_controls(audit: Audit, root: Path) -> None:
    registry_path = root / "results/marmaudio_icl_order_registry_seed20260814.json"
    if not registry_path.exists():
        audit.check("marm_icl_order_registry", False, "registry missing")
        return
    registry = json.loads(registry_path.read_text())
    rotations = list(registry.get("rotation_by_query", {}).values())
    counts = Counter(rotations)
    by_target = registry.get("rotation_counts_by_target", {})
    passed = registry.get("n_query") == 75 and len(rotations) == 75
    passed &= set(counts) == set(range(6))
    passed &= max(counts.values()) - min(counts.values()) <= 1
    for cell in by_target.values():
        values = [int(cell.get(str(index), cell.get(index, 0))) for index in range(6)]
        passed &= max(values) - min(values) <= 1
    audit.check(
        "marm_icl_order_registry", passed,
        f"query={len(rotations)} rotations={dict(sorted(counts.items()))}",
    )
    for k, mode in ((2, "blocked"), (2, "interleaved"), (8, "interleaved")):
        csv_path = root / f"results/marmaudio_icl_order_control_k{k}_{mode}_7b.csv"
        summary_path = root / f"results/marmaudio_icl_order_control_k{k}_{mode}_7b_summary.json"
        if not csv_path.exists() and not summary_path.exists():
            continue
        rows = read_csv(csv_path) if csv_path.exists() else []
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        events = {row.get("event_id") for row in rows}
        cell_passed = csv_path.exists() and summary_path.exists()
        cell_passed &= len(rows) == len(events) == 75
        cell_passed &= all(int(row["support_k_per_class"]) == k for row in rows)
        cell_passed &= all(row["order_mode"] == mode for row in rows)
        cell_passed &= summary.get("n_query") == 75
        cell_passed &= summary.get("support_k_per_class") == k
        cell_passed &= summary.get("order_mode") == mode
        audit.check(
            f"marm_icl_order_control:k{k}_{mode}", cell_passed,
            f"expected=75 observed={len(rows)} unique={len(events)}",
        )
    for name, registry_relative, expected_query, label_count in (
        ("dogs", "results/beans_dogs_lp1_icl_order_registry_seed20260814.json", 139, 10),
        ("watkins", "results/beans_watkins_icl_order_registry_seed20260814.json", 339, 31),
    ):
        other = json.loads((root / registry_relative).read_text())
        other_rotations = list(other.get("rotation_by_query", {}).values())
        other_counts = Counter(other_rotations)
        registry_passed = len(other_rotations) == expected_query
        registry_passed &= set(other_counts) == set(range(label_count))
        registry_passed &= max(other_counts.values()) - min(other_counts.values()) <= 1
        audit.check(
            f"{name}_icl_order_registry", registry_passed,
            f"query={len(other_rotations)} rotations={len(other_counts)}",
        )
    for dataset, k, expected_query in (
        ("beans_dogs_lp1", 2, 139), ("beans_watkins", 1, 339)
    ):
        csv_path = root / f"results/{dataset}_icl_order_control_k{k}_interleaved_7b.csv"
        summary_path = root / f"results/{dataset}_icl_order_control_k{k}_interleaved_7b_summary.json"
        if not csv_path.exists() and not summary_path.exists():
            continue
        rows = read_csv(csv_path) if csv_path.exists() else []
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        events = {row.get("event_id") for row in rows}
        passed = csv_path.exists() and summary_path.exists()
        passed &= len(rows) == len(events) == expected_query
        passed &= summary.get("n_query") == expected_query
        passed &= summary.get("support_k_per_class") == k
        passed &= summary.get("order_mode") == "interleaved"
        audit.check(
            f"{dataset}_icl_order_control:k{k}", passed,
            f"expected={expected_query} observed={len(rows)} unique={len(events)}",
        )
    combined_path = root / "results/icl_order_controls_combined_7b.json"
    combined_figure = root / "results/figures/fig_icl_order_position_copying.png"
    if combined_path.exists() or combined_figure.exists():
        combined = json.loads(combined_path.read_text()) if combined_path.exists() else {}
        cells = combined.get("cells", [])
        combined_passed = combined_path.exists() and combined_figure.exists()
        combined_passed &= len(cells) >= 3
        combined_passed &= all(cell.get("n_query", 0) > 0 for cell in cells)
        combined_passed &= all(
            0 <= cell.get("first_support_class_copy_rate", -1) <= 1
            and 0 <= cell.get("last_support_class_copy_rate", -1) <= 1
            for cell in cells
        )
        audit.check(
            "icl_order_positional_copying_summary", combined_passed,
            f"complete_cells={len(cells)} figure={combined_figure.exists()}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = Audit()
    audit_candidate_scoring(audit, root)
    audit_support_split(
        audit, root, "marm_recording_disjoint_nested_support",
        "data/manifests/marmaudio_expert_validation.csv",
        "results/marmaudio_equal_support_split_seed20260814.json",
        (1, 2, 4, 8), None, True,
    )
    audit_support_split(
        audit, root, "dogs_official_nested_support",
        "data/manifests/beans_dogs_all_full_lp1.csv",
        "results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_split.json",
        (1, 2), "valid", False,
    )
    audit_support_split(
        audit, root, "watkins_official_nested_support",
        "data/manifests/beans_watkins_all_full_lp1.csv",
        "results/beans_watkins_equal_support_split_seed20260814.json",
        (1, 2, 4), "valid", False,
    )
    audit_readouts(audit, root)
    audit_icl(audit, root)
    audit_relative_kv(audit, root)
    audit_class_routed_kv(audit, root)
    audit_lora(audit, root)
    audit_oracle_controls(audit, root)
    audit_paired_statistics(audit, root)
    audit_audio_silence(audit, root)
    audit_arbitrary_panel_summary(audit, root)
    audit_icl_order_controls(audit, root)
    audit_cross_prompt_pooled(audit, root)
    audit_beans_zero_7b(audit, root)
    result = {"passed": audit.passed, "checks": audit.checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
