# AKR final rebuttal cases implementation plan

> **For agentic workers:** implement the assigned independent modules; root integrates and performs final verification.

**Goal:** Publish independently executable and independently saved final AKR rebuttal cases with an honest reviewer-coverage map.

**Architecture:** Reuse existing model backends and the original continuous regressor. Add a Train-50 protocol, pure correction maps and baseline features, and a case runner with immutable registrations and exact coverage reports.

**Tech stack:** Python, NumPy/SciPy, existing Torch/Transformers backends; optional feature dependencies isolated from core imports.

**Spec:** `docs/superpowers/specs/2026-11-07-akr-final-cases-design.md`.

## Global constraints

- Dogs/Watkins/Bats/HumBugDB, official train support, whole official test; fixed outcome-blind seed.
- Original and revised AKR named and reported separately; no new scientific results claimed.
- No private historical caches required, no changes to GitHub workflows, no GPU launches.
- GitHub update is already authorized; preserve existing results and manuscript numbers.

## Review focus

- Train/test source or decoded-audio overlap must exclude training candidates, never trim the test.
- A class count greater than a requested support budget must fail, never silently omit classes.
- A failed preflight or corrupt cache must not leave complete=true.
- A partial or foreign report must not be accepted by aggregation.
- Support-prior centering, soft routing and norm capping must preserve their specified mathematical behavior.

## Task 1: Correction and readout methods

Files: `akr_final/train50_methods.py`, `tests/test_train50_methods.py`.
Interface: `fit_correction(x, gradients, labels, n_classes, spec, seed=0)` returns
an object with `predict(x)` and JSON-safe `metadata`; class routing additionally
exposes `probabilities(x)`. Spec keys include `kind`, `ridge`, `rank`, `temperature`,
`centered`. Original kind calls the repository's original fit function.

- [x] Test prior-centered cancellation, soft/hard behavior, original parity, deterministic shuffling and invalid inputs.
- [x] Implement corrections and metadata; run meaningful tests.

## Task 2: Dataset preparation and feature baselines

Files: `akr_final/train50_features.py`, `scripts/prepare_train50_data.py`, related tests and requirements.
Interface: `extract_feature(path, kind, assets=None)` -> finite 1-D NumPy vector;
`feature_provenance(kind, assets=None)` -> JSON-safe identity. Prepare official
split manifests without modifying existing manifests or uploading audio.

- [x] Verify primary dataset sources and real extraction APIs.
- [x] Implement MFCC/eGeMAPS/BEATs and deterministic manifest preparation with tests.

## Task 3: Protocol, runner and independent cases

Files: `akr_final/train50_protocol.py`, `akr_final/train50_cases.py`,
`akr_final/train50_runner.py`, `configs/akr_train50.yaml`, independent entry points and tests.

- [x] Test exact Train-50 selection, full-test retention, overlap rejection and nested support budgets.
- [x] Implement immutable registration, per-item inference journals, all declared method paths, policy provenance and strict reports.
- [x] Test stale completion reset, changed identity rejection, partial/foreign results and case isolation.

## Task 4: Reviewer map, validation and publication

Files: `docs/rebuttal/FINAL_CASES_ZH.md`, `docs/rebuttal/REVIEWER_MAP.md`, validation receipt and README links.

- [x] Map every recovered reviewer theme to cases or clearly labeled non-experimental edits/evidence gaps.
- [x] Run CPU tests, entry-point plans and a synthetic end-to-end integrity check; record optional dependency skips.
- [x] Obtain independent review and address material findings with targeted regressions.

## Final decisions and review dispositions

- Main soft settings are fixed (tau=1, eta=300, cap=.3, ridge=10). An all-class
  two-fold selection scheme is invalid when Train-50 contains singleton classes;
  sensitivity cases do not change the main policy using test outcomes.
- The original continuous mapping remains unchanged; its new-protocol control
  uses rank4/relative_norm .01. Historical raw-eta parity is not asserted.
- C16 was added for real source-registered historical readouts. Missing original
  registrations remain a blocking input; no fresh split substitutes for them.
- Independent review caught the full-prefill scope enum, unmatched scaling in
  the shuffled-pair control, external feature identity, tokenizer.model identity
  and NatureLM cost instrumentation. Those defects were corrected and verified.
- Additional regressions cover stale completion during CLI config failure,
  report completeness, specialist event IDs, injected-backend scientific flags,
  and first-pass feature/backward/readout cost accounting.
- C02 now includes complete probability/support-prior text evidence using the
  same softmax as the correction router, closing the top-1-only information gap.
- NatureLM remains a native specialist case with joint runtime acceptance
  pending; BirdSet/natural drift and NatureLM+AKR remain explicit claim limits.

## Authorized publication

The user authorized updating GitHub before reviewing the plan. The completed
implementation and validation receipt are to be committed and pushed to the
existing `research/rebuttal-train50-cases-20261107` branch. The final handoff
records the actual remotely verified commit; this pre-commit plan does not
invent a future commit hash. No GPU execution or external rebuttal submission
is part of publication.
