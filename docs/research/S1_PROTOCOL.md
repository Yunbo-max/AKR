# S1: simplest-alternative comparison

Status: IMPLEMENTED, NOT RUN ON A REAL MODEL. Registered here on 2026-10-03; this is not a retroactive claim of preregistration for older inspected datasets.

## Question

On the same bioacoustic recognition recordings and annotation budget, what does native-state AKR correction offer relative to directly using a frozen-feature classifier or providing its prediction as text evidence?

## Fixed configuration

Source of truth: `configs/akr_standalone.yaml`, mirrored in `configs/akr_registry.json`. Task MA-CT, labels Infant Cry/Phee/Seep/Trill/Tsik/Twitter, Qwen2.5-Omni-7B pinned revision, Full and LP1, seed 20260914, eight support clips per class, recording_id grouping, rank 4, ridge penalty 10, active-block relative alpha .01. Use all deterministically selected confirmation queries; no success-dependent truncation. The count is computed from the input manifest; do not claim it equals the old 97 without ID comparison. No new labels, models, conditions or hyperparameter search in the default queue.

## Arms

1. Native: unchanged audio/prompt -> original text decoder.
2. Fixed mean: average support correction, broadcast at the same selected audio K/V blocks.
3. Continuous AKR: support-fitted negative-gradient coordinate prediction -> per-recording internal correction.
4. Ridge direct: same support, frozen feature classifier, argmax label.
5. Probe-to-text: identical ridge prediction rendered as its label string; prediction and accuracy must equal arm 4.
6. Probe-to-LM: same query audio plus a prompt containing the ridge prediction, explicitly described as possibly wrong; original model selects its own answer.

A zero-alpha auxiliary control must reproduce native raw and parsed outputs. All features/normalizers/ridge targets use support only. Truth labels enter query scoring after predictions, never the inference interface.

## Reporting and stop rule

Report accuracy, macro-F1, invalid fraction, repaired/harmed counts, model-forward counts and time fields. For arm 6 additionally report agreement with the classifier, correct classifier outputs spoiled, and wrong classifier outputs rescued. Every arm must cover the exact expected IDs before a final score is accepted. Missing is not zero. Report all six arms, not the one that looks best.

AKR matching/losing to a simple baseline limits the claim of necessity or practical superiority; it does not trigger another parameter search. A positive score difference in one condition is a local result, not cross-model generalization. No fixed numerical PASS threshold is retrofitted after seeing results; preserve per-query outcomes for an appropriately declared comparison.

Total wall-clock budget is at most 24 hours for preflight and execution, plus at most 90 seconds for checkpoint-preserving process termination. Reuse verified complete caches only; stop with an incomplete status if the budget expires. The release command defaults to planning and requires `run --execute` for GPU/model use.
