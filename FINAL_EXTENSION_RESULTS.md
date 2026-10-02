# Post-12-hour extension results

This report is generated only after completeness checks pass. It extends
`FAIR_GAP_RESULTS.md` without changing the frozen validation/test rules.

## Dogs arbitrary A--J validation replication

The registered K=2/class support and all 139 official validation queries
were bijectively mapped to single-token outputs A--J. Query labels were used
only for final scoring. No test evaluation was permitted unless the frozen
validation gate passed.

| Method | Relative alpha | Accuracy | Macro-F1 | Invalid | Router accuracy |
|---|---:|---:|---:|---:|---:|
| native | 0 | 8.63% | 2.73% | 0.00% | 35.25% |
| probe_class_permuted | 0.003 | 7.91% | 5.64% | 25.90% | 35.25% |
| probe_class_permuted | 0.01 | 4.32% | 5.74% | 57.55% | 35.25% |
| probe_class_permuted | 0.03 | 0.72% | 1.82% | 95.68% | 35.25% |
| probe_class_pooled | 0.003 | 14.39% | 11.84% | 0.00% | 35.25% |
| probe_class_pooled | 0.01 | 18.71% | 15.26% | 0.72% | 35.25% |
| probe_class_pooled | 0.03 | 20.14% | 18.49% | 6.47% | 35.25% |
| probe_class_tokenwise | 0.003 | 15.11% | 11.22% | 24.46% | 35.25% |
| probe_class_tokenwise | 0.01 | 6.47% | 9.26% | 74.82% | 35.25% |
| probe_class_tokenwise | 0.03 | 0.00% | 0.00% | 99.28% | 35.25% |

Frozen gate: **failed**. Test action: `no_test`. Tokenwise beat pooled at alphas: `[0.003]`; selected tokenwise alpha: `0.003`; selected field beat permutation: `True`.

## BEANS-Zero complete registered target scan

All 2,950 examples across the 12 requested components were evaluated with
their official instruction text after the declared 16 kHz / 10 s cap /
100 ms minimum-duration protocol.

| Component | N | Exact matches | Exact match |
|---|---:|---:|---:|
| call-type | 283 | 174 | 61.48% |
| unseen-family-cmn | 117 | 0 | 0.00% |
| unseen-family-sci | 130 | 2 | 1.54% |
| unseen-family-tax | 123 | 0 | 0.00% |
| unseen-genus-cmn | 265 | 0 | 0.00% |
| unseen-genus-sci | 262 | 0 | 0.00% |
| unseen-genus-tax | 258 | 0 | 0.00% |
| unseen-species-cmn | 373 | 0 | 0.00% |
| unseen-species-sci | 364 | 0 | 0.00% |
| unseen-species-tax | 336 | 0 | 0.00% |
| watkins | 115 | 0 | 0.00% |
| zf-indiv | 324 | 217 | 66.98% |
| **Overall** | **2950** | **393** | **13.32%** |

The scan is an external zero-shot diagnostic. Mixed and unknown source
licenses prevent mirroring the audio; `DATASETS.md` records the release
policy and the tracked manifest preserves per-example license metadata.

## Release checks

- A--J logical CSV records: 1,390/1,390.
- BEANS-Zero logical CSV records: 2,950/2,950.
- Query-label-free validation gate retained; no disallowed test output.
- Model usage and limitations: `MODEL_USAGE.md`.
- Dataset provenance and redistribution: `DATASETS.md`.
