#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    with (RESULTS / name).open() as handle:
        return json.load(handle)


checks = []


def check(name, passed, detail):
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


dogs = load("beans_dogs_frequency_probe_7b_summary.json")
watkins = load("beans_watkins_frequency_probe_7b_summary.json")
for name, obj, expected in [("dogs_frequency", dogs, (415, 139, 139)),
                            ("watkins_frequency", watkins, (1017, 339, 339))]:
    conds = obj["conditions"]
    sizes = obj["split_sizes"]
    ok = (len(conds) == 6 and
          (sizes["train"], sizes["valid"], sizes["test"]) == expected and
          all(c["n_test"] == expected[2] for c in conds))
    check(name, ok, f"conditions={len(conds)} split={sizes}")

geometry = load("correction_geometry_decomposition_7b.json")
expected_geometry = {"marm_failures", "marm_k2_full", "marm_k2_lp1", "dogs_k2_lp1"}
actual_geometry = set(geometry["datasets"])
check("correction_geometry", actual_geometry == expected_geometry,
      f"datasets={sorted(actual_geometry)}")

scaling = load("marmaudio_support_scaling_k16_statistics.json")
ks = sorted({row["support_k_per_class"] for row in scaling["scaling"]})
check("support_scaling", ks == [1, 2, 4, 8, 16] and len(scaling["support_to_decision_gap"]) == 2,
      f"K={ks} strict_gap_tests={len(scaling['support_to_decision_gap'])}")

split = load("beans_dogs_AJ_factorized_validation_split.json")
selection, confirmation = split["selection"], split["confirmation"]
check("factorized_validation_split",
      len(selection) == 30 and len(confirmation) == 109 and not set(selection) & set(confirmation),
      f"selection={len(selection)} confirmation={len(confirmation)} disjoint={not set(selection) & set(confirmation)}")

partial = pd.read_csv(RESULTS / "beans_dogs_AJ_factorized_kv_lp1_rank_selection.csv")
n_query = partial["event_id"].nunique()
check("factorized_is_explicitly_partial", n_query < 30 and len(partial) == n_query * 5,
      f"queries={n_query}/30 rows={len(partial)}")

candidate = pd.read_csv(RESULTS / "marmaudio_equal_support_audio_icl_candidate_k8_7b.csv")
check("candidate_k8_is_explicitly_partial", len(candidate) < 75,
      f"queries={candidate['event_id'].nunique()}/75 rows={len(candidate)}")

passed = all(item["passed"] for item in checks)
output = {"passed": passed, "checks": checks}
with (RESULTS / "two_hour_extension_audit.json").open("w") as handle:
    json.dump(output, handle, indent=2)
    handle.write("\n")
print(json.dumps(output, indent=2))
raise SystemExit(0 if passed else 1)
