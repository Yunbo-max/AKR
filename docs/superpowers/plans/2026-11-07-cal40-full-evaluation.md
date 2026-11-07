# Forty-label full-evaluation implementation plan

> **For agentic workers:** Use superpowers:executing-plans. User has explicitly authorized 40 calibration clips plus complete independent evaluation across all prepared datasets; do not request this authorization again.

**Goal:** Execute frozen Speech LM, adapted Speech LM+LoRA and centered-class Speech LM+AKR on full eligible evaluation populations with exactly 40 shared labelled calibration clips per task.

**Architecture:** Add an independent registered runner and durable WSL queue. Reuse validated frozen-backbone extraction, PEFT training and strict journals without changing historical code. First freeze every split, numerical policy and source hash; then execute independent dataset/model/bandwidth jobs on both GPUs. Per-job reports and aggregate status distinguish running, failed and complete.

**Tech Stack:** Existing WSL root Python environment, PyTorch/Transformers/PEFT, NumPy, existing Qwen2.5 Omni backends.

**Spec:** User-approved conversation: exactly 40 calibration labels, complete independent test, same treatment across datasets. Historical GLM/Qwen3 models remain conditional, run only if needed.

## Global constraints

- Primary model matrix: Qwen2.5-Omni-3B and 7B; full bandwidth and LP1; one registered seed 20261107. This implements the existing two primary models, not optional GLM/Qwen3 expansion.
- Dogs and Watkins: preserve every official-test row; choose 40 from train after excluding any recording/PCM connected to test. Validation rows are unused.
- MarmAudio: no official test; allocate 40 unique source groups, score every remaining row from disjoint groups. Exclude other rows in calibration groups with an explicit receipt. Label this derived held-out re-evaluation, not a new untouched official test.
- BEANS-Zero: twelve separate registered eligible components, each gets 40 shared calibration clips and every remaining eligible canonical group. Preserve existing class exclusions and original 2,950-row zero-shot artifacts. This is the authorized derived supervised comparison, not the original benchmark; components overlap and are not independent datasets.
- Every retained class must occur in calibration; fail if impossible within 40. Balanced round-robin deterministic selection, unique recording groups. All registered components currently have <=36 classes and >=95 eligible groups.
- Group equivalence is the transitive union of declared source group and canonical PCM equality. No support/query overlap. Record official-test duplicate groups rather than silently shortening official test.
- Fixed hyperparameters, no new label budget for validation: AKR centered-class direction, ridge10, relative alpha0.3; LoRA rank8/alpha16/dropout.05, q/v decoder only, AdamW lr5e-4, five epochs, effective batch4, weight decay.01. Both configurations were developed on prior Dogs validation; not globally optimal nor matched historical search.
- Same prompts, waveform preprocessing, vocabulary and decoding within each comparison. 12 generated tokens for original datasets; existing 64-token limit for long BEANS-Zero labels, shared by all methods.
- No fallback to native for this fixed LoRA comparison; train the adapter and evaluate it. No tuning on query labels. Freeze all methods before evaluating any query.
- Leave previous source and results unchanged; no unrequested git commit/push. Existing dedicated feature branch is reused to preserve uncommitted project setup and checkpoints.
- Actual GPU tasks only; about90GiB total occupancy is a ceiling, not artificial allocation. One3B and one7B lane per GPU only if measured headroom permits, otherwise queue. Track sampled NVIDIA totals and task reservations.

## Review focus

Transitive duplicate bridges; 31/36-class coverage with only40 labels; official-test completeness; correct7B backbone despite legacy training helper name; resumability rejects changed adapters and source, and never declares partial predictions complete.

### Task 1: Register full populations

- [ ] Write failing tests for exact40, deterministic class coverage, transitive leakage, official-test preservation and derived group exclusion.
- [ ] Implement `scripts/cal40_full_eval.py` split registration and report audits.
- [ ] Run tests and real data preflight; enumerate every calibration/query count before GPU launch.

### Task 2: Execute fixed methods

- [ ] Test model routing, gradient access restriction and all-record completeness.
- [ ] Implement per-job extraction, locked policy, same-pool LoRA fitting, resumable native/AKR/LoRA prediction and complete-only report.
- [ ] Run fresh-context code review and fix material findings before registration/launch.

### Task 3: Durable execution and evidence

- [ ] Implement WSL queue with bounded memory admission, per-job subprocess logs, explicit failure/retry state and complete-only aggregation.
- [ ] Validate scheduler state and source binding; launch GPU jobs only after final code freeze.
- [ ] Verify actual GPU progress and first completed full-test jobs. Keep long comprehensive jobs running with durable logs/status; report completed and pending counts honestly until all jobs finish.

## Execution ledger

- Design ruling: use fixed transferred settings rather than 2-fold selection; Watkins31 classes and BEANS36 classes cannot all occur in both20-example folds. This implements the user's allowed fixed-hyperparameter approach with exactly40 labels.
- Design ruling: treat each of twelve BEANS components as a task, preserving its label space; one shared40-clip pool over unrelated component vocabularies would not define comparable classification tasks.
- Design ruling: execute existing primary3B/7B and full/LP1 matrix, leaving optional larger models conditional per user's prior instruction.
