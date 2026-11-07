# Matched Speech LM comparison implementation plan

**Goal:** Actually compare native Speech LM, Speech LM + AKR and Speech LM + LoRA under one labelled calibration-pool budget. Ridge remains an auxiliary diagnostic, absent from the main comparison.

**Authorization:** User explicitly requests testing and making the comparison fair. The bounded implementation choices below are assistant-selected within that scope; no new approval gate or full matrix launch is needed.

**Architecture:** Reuse the source-frozen 3B calibration pilot's 40 pool IDs, two folds, frozen checkpoint, preprocessing, prompt, and locked AKR selection. Add an isolated script with its own source identity, train fresh LoRA adapters on exactly those IDs, and evaluate all three speech-model generators on a newly registered development subset. Historical evidence stays unchanged.

**Environment:** WSL root, /root/.venvs/akr; existing Transformers 4.52.3 / PEFT 0.17.1 / BF16 / SDPA; one available GPU. No dependency upgrades.

## Frozen design

- Source pilot: pool_dogs_3b_20261107_v2. Validate its completed report, original code/config/audio/array hashes, and selected policies before importing support arrays. Never import arrays from development queries.
- Qwen2.5-Omni-3B at the identical pinned revision. Dogs all 10 labels, LP1. Reuse the exact 40 calibration examples and 20/20 folds.
- Select four further official-validation examples/class deterministically (seed 20261108), excluding every calibration and previously evaluated source recording and decoded waveform. Exclude official test. These are new pilot queries, but the dataset/protocol has been inspected, so report development evidence, not untouched-test confirmation.
- AKR primary is the previously selected CV-refit map (rank 4, ridge 10, alpha .03, relative_cap, eta 300); import its calibration-only selection evidence and refit all 40 without new tuning.
- LoRA: language decoder self-attention q_proj/v_proj only, across all decoder layers; audio/vision towers and base weights frozen. Rank 8, alpha 16, dropout .05, AdamW lr 2e-4, weight decay .01, clip norm 1, batch 1 / accumulation 4, no LR schedule. This is a specified standard LoRA variant, not a claim to optimize every possible LoRA configuration.
- LoRA candidate epochs 0/native, 1, 3, 5. Fresh initialization per fold; train only complementary 20 IDs, evaluate mean answer-token CE on held-out 20 at each candidate epoch. Pick pooled OOF mean CE, native on ties/no improvement. Then fresh initialization and train all 40 for the chosen epoch count. All tokenization and answer-only masking identical to AKR's loss objective; no EOS target added to either.
- Both methods have three nonzero candidates plus native and identical fold/sample boundaries. Their compute and adaptable parameter counts differ and must be reported, not claimed matched. No evaluation-label hyperparameter selection, no post-result tuning.
- Native, AKR, LoRA use the same fresh 40 queries, canonical prompt, greedy 12-token generation cap and normalizer. LoRA's final answer comes from the same language model, not an external classifier.
- Check adapter-zero/disabled parity, only intended adapter parameters trainable, frozen-base identity, finite gradients/updates, and exact training/validation IDs. Strict checksummed training stage receipts and adapters. Resume completed stages; rerun an interrupted stage from deterministic initialization (explicitly not optimizer-step resume).
- Lock both adaptation policies before any fresh-query prediction. Record accuracy, macro-F1, invalid rate, repaired/harmed counts, paired uncertainty, training exposure/steps/forward calls/wall time/peak allocation, checkpoint hashes, and initialization seeds.
- Bound real execution to 90 minutes; do not stop other workloads. Reject errors, incomplete evidence and changed provenance rather than converting them into native fallback.

## User steering before any new development predictions

The user additionally requested using both GPUs toward 85 GB each, debugging, and improving AKR performance. The following bounded search supersedes the single-learning-rate / unchanged-AKR-only design above; the original AKR remains a reported reference.

- Run the two LoRA folds on separate GPUs. LoRA LR candidates are 5e-5, 2e-4, 5e-4; epoch candidates are 1, 3, 5: nine nonzero configurations, plus native.
- AKR candidates use three mappings: original gradient regression; residual-shrunk regression (mean + .25 * predicted deviation); class-pooled gradient routing from a ridge acoustic readout (hard predicted-class gradient centroid). The latter still supplies only K/V corrections; the Speech LM generates the final answer. It must be labelled as an AKR variant, not silently replace the original algorithm.
- Each mapping tests relative-norm alpha .01, .03, .1: nine nonzero configurations, plus native. Select pooled out-of-fold answer CE, then fit on all 40. No development labels choose mappings or strengths.
- Thus this new search has identical per-method calibration pool, folds, criterion and count of candidate configurations. Wall-clock/backward compute is not matched; historical research/search is not erased by this matching.
- GPU target means approximately 85 GiB total allocation including already-running workloads, with headroom; use actual training and inference, no padding allocations. Record NVIDIA telemetry and each worker's actual peak. Each worker has a conservative dynamic process-memory cap based on existing occupancy, and fails cleanly if the available budget cannot accommodate the unchanged batch/precision settings. Exact 85 GiB occupancy is not guaranteed by these small-model tasks.
- After CV, run full-pool LoRA refit and AKR calibration on separate GPUs where available; lock both policies before new-query inference. Keep all scientific conclusions bounded to this pilot and report whichever method wins.

## Tasks

**Preflight correction, before any new outcome:** Dogs official validation has only four Rudy examples, all already in the prior pilot. A new class-balanced 40-query official-validation set is impossible. Reuse exactly the prior 40 development IDs for every method and explicitly report previously inspected development evidence. Keep official test unused during this debugging/improvement round. This supersedes the fresh-query clause; adaptation and hyperparameter selection still use only the separate 40-example calibration pool.

- [ ] Add registration, masked-label/target-module helpers and fold/selection tests, then implement isolated runner.
- [ ] Train/test a tiny CPU decoder to verify real PEFT adapter updates, base freezing, save/reload and disabled parity.
- [ ] Independent code review and appropriate regression checks before source freeze.
- [ ] Run the real matched 3B comparison, verify final artifacts and write a report with explicit scientific limits.
