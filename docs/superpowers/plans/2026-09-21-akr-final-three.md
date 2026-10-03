# AKR final-three implementation plan

**Goal:** Add two bounded ablations and a focused reuse-capable 3B replication, without changing the primary method or rerunning E1-E9.
**Architecture:** New `akr_final/` package consumes existing source-run locks, support IDs and confirmation IDs. Reuse the repository regressor and direct Thinker backend. Never alter `akr_closing/`, historical runs or prior gates.
**Scope:** MarmAudio, semantic labels, k=8 per class, source seeds (maximum three), full and lp:1000 only. This is locked re-evaluation of previously inspected data, not a new untouched test. No success-only seed selection.

## Experiments
- F1: grouped cross-validation of saved support gradients. Refit normalizers, SVD and ridge inside every training fold. Compare continuous prediction, fixed mean, shuffled-feature prediction and a explicitly labelled true-class centroid diagnostic. Report projection versus regression error separately. No gradients on evaluation queries.
- F2: 2x2 source-bandwidth -> query-bandwidth correction transfer. Match support/query IDs. Use one source 1-kHz lock per seed for both bandwidths, with no new tuning; source standardization and basis travel with the map. Native, fixed, predicted and deranged-query predicted controls; output all completed cells regardless of sign. Include direct-readout references.
- F3: 3B replication of corresponding 7B locks on the same support and confirmation episodes. Same family, different size, not cross-family generalization. Refit each model's own basis/regressor (never transfer dimension-incompatible KV tensors). Reuse existing complete E9 predictions only when identity checks pass.

## Implementation tasks
- [x] Write failing tests for group disjointness, fold-only fitting, error decomposition, source/target separation, unlabelled-query API, strict reuse and resumable journals.
- [x] Implement predictability and source-run ingestion, including exact episode selection and support/query overlap checks.
- [x] Implement source-aware cache import, bounded transfer and 3B comparisons; preserve all outcome directions and no-op checks.
- [x] Add preflight/execute/report CLI, a single shell entry, documentation and source-bundle export checksums.
- [x] Run CPU tests, CLI/compile checks and fresh remote CI. Local: 84 passed, one production-import test deferred. Full-repository CI: 26 passed, no skips (run 35564015188). GPU science remains for the execution agent.

## Review focus
Existing source runs can have local paths or altered configs: read actual RUN.json and locks, do not assume README defaults. Never silently truncate audio, discard failed seeds or overwrite source outputs. Incomplete or incompatible evidence is BLOCKED, not zero or a complete result. Report saved-input reuse separately from new model work; no new empirical gain is claimed by this code delivery.
