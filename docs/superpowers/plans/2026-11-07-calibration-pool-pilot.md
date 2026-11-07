# AKR Single Calibration Pool Pilot

**Goal:** Test and improve AKR using only one registered labelled calibration pool, comparing fixed hyperparameters, internal holdout selection and cross-validation with full-pool refitting.

**Authorization:** User explicitly requested testing and improving the two proposed calibration-only approaches. No full LoRA training or conditional GLM/Qwen3 production run is included.

**Design:** Use existing frozen-backbone feature, support-gradient and actual K/V intervention interfaces. Add an isolated pilot runner rather than changing historical experiment definitions. Defaults are the assistant's bounded pilot choices, not claimed user-specified hyperparameters.

- Start with Qwen2.5-Omni-3B, Dogs, all 10 classes, LP1, seed 20261107.
- Deterministically choose four distinct training recordings/class for a single 40-example calibration pool; choose four official-validation recordings/class for 40 development queries, without consulting predictions. Exclude all official test rows. Never present development scores as untouched-test evidence.
- Use source-recording and decoded-waveform identity for grouping and disjointness. Lock actual source/config/code/waveform hashes; resumable per-example caches and prediction/loss journals.
- Fixed approach: fit all 40 pool examples; rank 4, ridge 10, alpha .01, relative-cap arithmetic. No hyperparameter selection.
- Internal split: two/class fit, one/class selection, one/class guard; select alpha from native/0, .003, .01, .03 by answer CE, one guard veto; retain the selected map without refitting on guard examples.
- Improvement candidate: two group-disjoint stratified folds, fit each map on 20 examples and score only out-of-fold examples; select the same alpha grid, default native unless CE improves; refit on all 40 pool examples before query inference. This uses all available labelled data without an extra labelled dataset; OOF selection does not guarantee full-pool refit improvement.
- Use identical frozen model, class prompt, audio preprocessing, generation cap, calibration pool and development queries across all methods. Include native, fixed-relative normalization reference and Ridge-on-full-pool diagnostic. No development-query label enters map fitting, candidate selection or loss evaluation.
- Record labelled budget, per-method CE selection/guard evidence, native no-op check, query accuracy/macro-F1/invalid rate, paired repaired/harmed counts, model calls, elapsed phases, peak GPU memory. Shared-cache reuse and cross-fold compute costs are disclosed rather than claimed equal.
- Reject nonfinite/corrupt caches and changed identities; never silently turn a model/runtime error into native fallback. Explicit --execute and budget cap required for real weights. Use an available GPU without stopping other workloads.

## Implementation

- [ ] Add deterministic pilot registration and pure alpha-selection/fold utilities, with leakage/tie/failure tests.
- [ ] Add isolated real/synthetic runner reusing DatasetRunner cache and inference paths; freeze selection before development predictions; validate exact no-op and resumability.
- [ ] Add CLI, preflight, bounded real 3B pilot and complete-only evidence summary.
- [ ] Obtain independent review through requesting-code-review; fix material issues, run relevant regression checks, and document measured limitations.

No positive result is required. If the improvement candidate fails to improve development recognition, report that and retain the original method; do not tune against the official test or repeatedly search the same development scores in this turn.
