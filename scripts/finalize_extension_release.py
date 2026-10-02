#!/usr/bin/env python3
"""Validate and summarize the post-12-hour extension artifacts for release."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from animal_omni.metrics import classification_metrics


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    aj_csv = ROOT / "results/beans_dogs_probe_routed_class_kv_AJ_lp1_valid.csv"
    zero_csv = ROOT / "results/beans_zero_targets_fullscan_cap10_qwen7b.csv"
    aj_rows = read_csv(aj_csv)
    zero_rows = read_csv(zero_csv)
    if len(aj_rows) != 1390:
        raise RuntimeError(f"A--J artifact incomplete: {len(aj_rows)}/1390")
    if len(zero_rows) != 2950:
        raise RuntimeError(f"BEANS-Zero artifact incomplete: {len(zero_rows)}/2950")

    aj_summary_path = ROOT / "results/beans_dogs_probe_routed_class_kv_AJ_lp1_valid_summary.json"
    zero_summary_path = ROOT / "results/beans_zero_targets_fullscan_cap10_qwen7b_summary.json"
    aj = json.loads(aj_summary_path.read_text(encoding="utf-8"))
    zero = json.loads(zero_summary_path.read_text(encoding="utf-8"))
    if aj["incomplete_cells"]:
        raise RuntimeError(f"A--J summary has incomplete cells: {aj['incomplete_cells']}")
    if zero.get("overall", {}).get("n") != 2950:
        raise RuntimeError("BEANS-Zero summary does not cover 2,950 examples")

    dogs_lora_rows = read_csv(
        ROOT / "results/lora_beans_dogs_lp1_equal_support_k2_7b_valid.csv"
    )
    dogs_labels = [
        "Farley", "Freid", "Keri", "Louie", "Luke", "Mac", "Roodie",
        "Rudy", "Siggy", "Zoe",
    ]
    dogs_lora = {
        **classification_metrics(
            [row["target"] for row in dogs_lora_rows],
            [row["prediction"] or None for row in dogs_lora_rows],
            dogs_labels,
        ),
        "n": len(dogs_lora_rows),
        "prediction_counts": dict(Counter(
            row["prediction"] or "<invalid>" for row in dogs_lora_rows
        )),
        "protocol": "K=2/class, 20 train supports, official validation only",
    }
    (ROOT / "results/lora_beans_dogs_lp1_equal_support_k2_7b_valid_summary.json").write_text(
        json.dumps(dogs_lora, indent=2), encoding="utf-8"
    )

    cells = sorted(
        aj["complete_cells"],
        key=lambda cell: (cell["method"], cell["relative_alpha"]),
    )
    lines = [
        "# Post-12-hour extension results",
        "",
        "This report is generated only after completeness checks pass. It extends",
        "`FAIR_GAP_RESULTS.md` without changing the frozen validation/test rules.",
        "",
        "## Dogs arbitrary A--J validation replication",
        "",
        "The registered K=2/class support and all 139 official validation queries",
        "were bijectively mapped to single-token outputs A--J. Query labels were used",
        "only for final scoring. No test evaluation was permitted unless the frozen",
        "validation gate passed.",
        "",
        "| Method | Relative alpha | Accuracy | Macro-F1 | Invalid | Router accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell['method']} | {cell['relative_alpha']:.3g} | "
            f"{percent(cell['accuracy'])} | {percent(cell['macro_f1'])} | "
            f"{percent(cell['invalid_response_rate'])} | "
            f"{percent(cell['router_accuracy'])} |"
        )
    gate = aj["gate"]
    lines.extend([
        "",
        f"Frozen gate: **{'passed' if gate['passed'] else 'failed'}**. "
        f"Test action: `{gate['test_action']}`. Tokenwise beat pooled at alphas: "
        f"`{gate['tokenwise_beats_pooled_alphas']}`; selected tokenwise alpha: "
        f"`{gate['selected_tokenwise_alpha']}`; selected field beat permutation: "
        f"`{gate['selected_tokenwise_beats_permuted']}`.",
        "",
        "## BEANS-Zero complete registered target scan",
        "",
        "All 2,950 examples across the 12 requested components were evaluated with",
        "their official instruction text after the declared 16 kHz / 10 s cap /",
        "100 ms minimum-duration protocol.",
        "",
        "| Component | N | Exact matches | Exact match |",
        "|---|---:|---:|---:|",
    ])
    for component, record in sorted(zero.items()):
        if component == "overall":
            continue
        lines.append(
            f"| {component} | {record['n']} | {record['correct']} | "
            f"{percent(record['exact_match'])} |"
        )
    overall = zero["overall"]
    lines.extend([
        f"| **Overall** | **{overall['n']}** | **{overall['correct']}** | "
        f"**{percent(overall['exact_match'])}** |",
        "",
        "The scan is an external zero-shot diagnostic. Mixed and unknown source",
        "licenses prevent mirroring the audio; `DATASETS.md` records the release",
        "policy and the tracked manifest preserves per-example license metadata.",
        "",
        "## Release checks",
        "",
        "- A--J logical CSV records: 1,390/1,390.",
        "- BEANS-Zero logical CSV records: 2,950/2,950.",
        "- Query-label-free validation gate retained; no disallowed test output.",
        "- Model usage and limitations: `MODEL_USAGE.md`.",
        "- Dataset provenance and redistribution: `DATASETS.md`.",
        "",
    ])
    output = ROOT / "FINAL_EXTENSION_RESULTS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
