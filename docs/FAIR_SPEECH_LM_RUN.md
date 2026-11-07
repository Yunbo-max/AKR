# Matched Speech LM pilot

`scripts/fair_speech_lm.py` implements the bounded Qwen2.5-Omni-3B / Dogs pilot.
It compares generated answers from native Speech LM, original AKR, selected AKR,
and a freshly trained language-decoder LoRA. It is not yet an all-model/all-dataset
LoRA training matrix. Published adapters with different training sets are excluded.

The 40 labelled calibration IDs and two folds come from
`results/calibration_pilot/pool_dogs_3b_20261107_v2`. Every method uses the same
40 development queries. These development queries were previously inspected;
official test remains unused. A fresh balanced validation subset was impossible
because the Rudy class has only four official-validation examples.

Both adaptation searches have nine nonzero candidates plus native and use pooled
held-out answer CE. LoRA searches three learning rates and epochs 1/3/5. AKR
searches three gradient mappings and three correction strengths. The candidate
count and label budget match; compute, parameter count and historical research
effort do not. Auxiliary parameter diagnostics never select the main winner.

## WSL execution

Use the existing root environment `/root/.venvs/akr` and a **new run ID after any
code/config/environment change**. Preflight checks source hashes. The frozen plan
is `docs/superpowers/plans/2026-11-07-fair-speech-lm.md`.

```bash
cd /mnt/c/ai/akr
PY=/root/.venvs/akr/bin/python
RUN=fair_dogs_new_run
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
$PY scripts/fair_speech_lm.py --run-id "$RUN" --stage prepare
$PY scripts/fair_speech_lm.py --run-id "$RUN" --stage sensitivity

CUDA_VISIBLE_DEVICES=0 $PY scripts/fair_speech_lm.py --run-id "$RUN" --stage fold0 --execute --gpu-target-gib 90 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/fair_speech_lm.py --run-id "$RUN" --stage fold1 --execute --gpu-target-gib 90 &
wait
```

Check both stage receipts for completion before proceeding. Run AKR selection
and full-pool LoRA refitting on available GPUs; these stages are independent once
both LoRA folds finish. AKR selection can also run earlier if memory permits.

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/fair_speech_lm.py --run-id "$RUN" --stage akr_select --execute --gpu-target-gib 90 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/fair_speech_lm.py --run-id "$RUN" --stage lora_refit --execute --gpu-target-gib 90 &
wait
$PY scripts/fair_speech_lm.py --run-id "$RUN" --stage lock

CUDA_VISIBLE_DEVICES=0 $PY scripts/fair_speech_lm.py --run-id "$RUN" --stage eval_base --execute --gpu-target-gib 90 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/fair_speech_lm.py --run-id "$RUN" --stage eval_lora --execute --gpu-target-gib 90 &
wait
$PY scripts/fair_speech_lm.py --run-id "$RUN" --stage report
```

The memory argument is a scheduling target, not a request to allocate padding.
Monitor actual total occupancy with `nvidia-smi`; WSL CUDA allocator accounting
can differ from total-device occupancy. All model workers require explicit
`--execute` and have a 90-minute maximum per stage. Do not displace other jobs.

Completed checksummed stages are reusable. An interrupted training stage restarts
from deterministic initialization; optimizer-step resume is not implemented.
Damaged or misplaced checkpoints are rejected. Selection evidence and checkpoint
hashes are locked before development prediction, with serialized policy auditing
for the two inference workers.

Inspect `STATUS.json`, stage receipts, `POLICY_LOCK.json`, and `REPORT.json` in the
run directory. A historical report alone does not establish current completeness.
For the November 7 run, `COST_AUDIT.json` reconstructs training forward counts
missed by the original PEFT module hook; scientific results and original receipts
are preserved, and this distinction must remain in any compute comparison.
