# AKR E1–E9 Closing Suite Implementation Plan

**Goal:** Implement the supplied animal-only closing protocol without overwriting historical results or claiming new GPU results.
**Architecture:** Isolated additive `akr_closing` package. E1 wraps the existing evaluator and locks its exact historical 75/48 experiment. E2–E5 share continuous repository gradient regression, immutable journals and explicitly labelled re-evaluation splits. E7 holds the support set fixed across every query and counterbalances the first class. E8 fits LoRA on the same support, chooses settings on development only, and saves optimizer/RNG state. E6/E9 are optional.
**Spec:** `docs/AKR_CLOSING_E1_E9.md` (user-supplied plan, copied unchanged).
**Tech Stack:** Existing repository Python/PyTorch/Transformers/scikit-learn environment. CPU tests use numeric tensors and synthetic backends, not synthetic scientific results.

## Global Constraints
- No GPU training or inference in this delivery.
- No new test split is called untouched: these datasets have already been inspected.
- Original `ConditionalGradientRouter` is the production continuous regressor.
- Query inference receives a target-free Query object.
- Results exist only when all expected identities are present; missing is never zero.
- No automatic replacement of an old failed tokenwise/factorized gate.
- Reuse existing data and exact historical E1 protocol; fail on missing or changed artifacts.
- GitHub access was attempted and returned 403; code will be delivered as a checked additive patch unless permissions change.

## Tasks
1. Test E1 exact IDs, K/class coverage, duplicate rejection, both candidate score reductions, and row-preserving checkpoint continuation; implement in `legacy.py`.
2. Test fixed-set counterbalanced support order and no-op/control invariants; implement in `orders.py` and `protocol.py`.
3. Test target-controlled sound duration/RMS handling, gradient residual statistics, and continuous-router identity; implement support and deployment controls.
4. Test complete-only reports, honest selection/confirmation status, lock invalidation, paired-cluster statistics, cost accounting, and strict zero-no-op.
5. Add E8 optimizer-budget schedule, development-only selection, label masks, adapter verification and RNG checkpoint state; validate helper contracts on CPU.
6. Add CLI, configurable stages, fail-closed execution/gate behavior, README section, source provenance and single-command runbook.
7. Run CPU baseline and new tests, compileall, shell syntax, dry-run/preflight checks, and install/patch verification; record exact limitations.
