# AKR final rebuttal cases

## Authorized scope

The user requested completed experiment code, one independent Python entry point
and output directory per case, reviewer-concern coverage, and publication to
GitHub before reviewing the final plan. This document implements that request;
GPU experiments and external review replies are not launched by this change.

The latest recovered user decision supersedes the earlier Cal-40 design: the
primary suite is **Dogs, Watkins, Bats, HumBugDB**, with **50 total stratified
official-training examples** and the **complete official test**. Model outcomes
must not influence support selection. All methods in a comparison share the
same support/test IDs, audio condition, vocabulary and output protocol. Historical
MarmAudio, the selected 20-support protocol and derived BEANS-Zero remain separate.

## Methods and comparisons

Keep the paper's continuous centered-SVD/ridge AKR unchanged and explicitly named.
The revision is `akr_soft_cap`: frozen acoustic features -> ridge class scores ->
softmax -> support-prior-centered class-gradient prototypes -> capped additive
K/V injection -> frozen Speech LM answer. Keep hard routing as an ablation. No
tokenwise/factorized replacement or new trainable attention module is introduced.

Prepare native/original/revised comparisons, direct ridge and probe-to-LM,
MFCC/eGeMAPS/BEATs readouts, matched audio ICL, hard/soft x norm/cap,
centering, shuffled/random/fixed controls, K/V/layer controls, support-size and
hyperparameter sensitivity, Full/LP1 transfer, GLM, same-support LoRA, no-op and
cost diagnostics. Specialist and additional-dataset questions must have explicit
evidence requirements; absent assets/results cannot be marked complete.

## Implementation decisions

Use new focused `akr_final/train50_*.py` modules, existing frozen model backends,
and thin `experiments/rebuttal_cases/cXX_*.py` entry points. A fresh registration
must require no private development caches. Keep one run per case/dataset/model/
condition/seed and lock input, code, policy and checkpoint identities. Preserve
partial item journals; reject changed identities and corrupted results. Reset
completion status before all fallible run work. Aggregate only exact expected
query populations and all required methods. Results remain pending until actual
audio/model execution completes.

## Evidence limits

Reviewer IDs tbdR, h8E8, 7PoR and concern summaries were recovered from the saved
2026-11-07 13:19 AKR design, not original review text. Do not invent original
review order, scores, W/Q numbering or quotations. Record this provenance in the
reviewer map and mark exact wording verification pending.
