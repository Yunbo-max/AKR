# Project checkpoint — 2026-11-07

This branch preserves completed preparation and development experiments. The requested forty-calibration/full-test expansion was stopped by the user before any of its GPU jobs started. Publishing this checkpoint does not resume experiments.

## Completed development experiments

All numbers below use Qwen2.5-Omni-3B, Dogs ten-class identity recognition, LP1 audio, forty labelled calibration clips and the same previously inspected forty development clips. They are not full-dataset test results.

| Method / iteration | Development accuracy |
| --- | ---: |
| Frozen Speech LM | 10% |
| Original CV-refit AKR reference | 12.5% |
| First matched comparison: answer-CE-selected AKR | 10% |
| First matched comparison: answer-CE-selected LoRA | 25% |
| Second iteration: accuracy-selected centered-class AKR, alpha 0.3 | 25% |
| Second iteration: LoRA selection policy | 10% — selected native, no adapter |

The new AKR variant matched the earlier adapted LoRA accuracy on this development subset; it did not establish superiority. The second iteration used information learned from the first. Current label budgets and candidate counts were matched; historical search effort and compute were not. Official-test audio was excluded from these development runs.

- [First fair comparison and CE sensitivity](FAIR_SPEECH_LM_RESULTS_20261107.md)
- [Second development iteration and recognition sensitivity](ACCURACY_SELECTED_AKR_RESULTS_20261107.md)
- [Published compact evidence](../results/development_20261107/EXPORT.json)

The compact evidence includes byte-preserved reports, locked policies, sensitivity summaries and plots. Large audio, model weights, adapters, feature tensors and per-worker caches remain local. Historical absolute paths and runtime source checksums are provenance, not portable cache locations. A fresh clone cannot resume those runs without their local artifacts; new executions must create new registrations.

## Prepared broader coverage

The primary backbones are Qwen2.5-Omni-3B and 7B. GLM-4-Voice and Qwen3-Omni backends are implemented but remain conditional; they are not claimed as completed production comparisons. Dataset collections are MarmAudio, Dogs, Watkins and BEANS-Zero, the last containing twelve separately registered derived supervised component tasks.

The comprehensive preparation and specialist diagnostics have their own protocol in [COMPREHENSIVE_RUN.md](COMPREHENSIVE_RUN.md). Published external LoRA adapters have different training budgets and must not be described as the forty-label matched LoRA baseline.

## Stopped full-evaluation draft

`scripts/cal40_full_eval.py`, `scripts/cal40_queue.py` and their plan/tests are an unfinished checkpoint, **not ready for production execution**. The proposed matrix is two primary models × Full/LP1 × fifteen tasks, with native, AKR and same-forty-label LoRA in each job. No full-matrix result is published.

- Dogs and Watkins have 139 and 339 official-test rows respectively. The draft preserves the whole official test and chooses calibration clips from training after recording/PCM overlap checks.
- MarmAudio requires a derived recording-group split. Partial preflight found forty calibration clips, 142 disjoint query clips from 53 recordings, and 364 additional clips excluded because they share calibration recordings. It would be incorrect to call every remaining clip independent.
- BEANS-Zero uses existing eligible canonical-group manifests and forty calibration clips per component; it is a derived supervised comparison, not the original 2,950-example zero-shot benchmark. Its components may overlap.
- Real-data preflight was interrupted before all datasets finished. Do not present provisional counts as a completed registration.

The interrupted independent review identified two unresolved completion-integrity issues:

1. The aggregate report must check query IDs against the registration, require all native/AKR/LoRA methods and validate job/policy identity and successful cache-only verification. Currently it trusts the report's own query list too broadly.
2. Runner construction/preflight/cache validation must reset and handle status inside the error boundary. Currently a failure before that boundary can leave an older `complete=true` status behind.

These issues must be fixed and regression-tested before this draft is used. GPU scheduling and the remaining review also need completion. The earlier seven draft unit tests did not cover these failure paths.

## Publication validation

Fresh CPU-only validation: **279 passed, 6 skipped, 1 warning**. The output is published as `results/development_20261107/PUBLICATION_TESTS.log`. It validates software behavior covered by the suite; it does not certify the unfinished full-evaluation workflow or substitute for scientific experiments. This checkpoint is submitted as a draft pull request, with the known issues retained above.
