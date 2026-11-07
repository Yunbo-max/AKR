# Reproduce the accuracy-selected development round

Run from `/mnt/c/ai/akr` as the WSL root user with `/root/.venvs/akr/bin/python`. The registered run is `acc_dogs_3b_20261107_v1`. It depends on the completed and archived `fair_dogs_3b_20261107_v2` experiment and its LoRA adapters.

The complete stage order is:

1. `prepare`
2. `cv_akr0`, `cv_akr1`, `cv_lora0`, `cv_lora1` (independent; two workers per GPU were used)
3. `refit`
4. `lock`
5. `eval_akr`, `eval_lora` (independent)
6. `report`

Example GPU stage:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/.venvs/akr/bin/python scripts/accuracy_selected_akr.py \
  --run-id acc_dogs_3b_20261107_v1 --stage cv_akr0 --execute \
  --gpu-target-gib 90
```

The target is a scheduling ceiling in GiB, evaluated against both NVIDIA total occupancy and CUDA free memory. It does not reserve 90 GiB or force each job to consume that amount. Other workloads contribute to total occupancy. Stage telemetry records admission-time memory; `GPU_SAMPLES.jsonl` records sampled NVIDIA totals.

`prepare`, `lock` and `report` omit `--execute` and need no GPU. The report verifies selected inference caches with model loading prohibited. Re-running completed inference stages validates source, data, array, adapter and journal hashes; it does not silently accept stale outputs. Source edits require a new run identifier and registration. Archived first-round code preserves the version used for its original numerical results; the current first-round utility additionally fixes PEFT forward accounting and WSL memory admission.

Generate the complete-only summary and figures after `report` succeeds:

```bash
/root/.venvs/akr/bin/python results/accuracy_selected_akr/summarize_accuracy.py \
  acc_dogs_3b_20261107_v1
```

This is a bounded second development iteration, selected on the same 40 calibration clips and evaluated on the same previously inspected 40 development clips. The official test set is unused. Current label budget and candidate counts are matched across AKR and LoRA; historical search effort and compute are not. Do not present its development result as an independent test result or global superiority claim.
