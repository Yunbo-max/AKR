# Single calibration pool pilot

This isolated pilot evaluates AKR using **one pool of 40 labelled recordings**.
It does not require an additional adaptation training set. AKR still learns a
small feature-to-gradient mapping; all backbone weights remain frozen.

The first registered pilot uses Qwen2.5-Omni-3B, Dogs (10 individual-dog labels),
LP1 audio, seed 20261107, four official training recordings per label for the
calibration pool, and four official validation recordings per label for
development evaluation. Official test recordings are excluded. This is a
small, single-seed development experiment, not untouched-test evidence.

| Method | Use of the same 40-example labelled pool |
| --- | --- |
| Native | No adaptation |
| Fixed-relative reference | Fit all 40; original relative-norm correction at alpha 0.01 |
| Fixed-cap AKR | Fit all 40; capped correction at alpha 0.01 |
| Internal split | Fit 20; choose alpha on 10; one veto on remaining 10; retain 20-example map |
| CV-refit candidate | Two 20/20 folds; choose alpha from pooled held-out losses; refit on all 40 |
| Ridge diagnostic | Fit a direct feature-to-label readout on all 40 |

A seventh method, `zero`, verifies that alpha zero reproduces native generated
tokens and outputs exactly. Rank 4, ridge coefficient 10, raw eta 300 and a
12-token generation cap are fixed in advance. Alpha selection uses native/0,
0.003, 0.01 and 0.03; it minimizes mean correct-answer token cross-entropy.
Ties or no improvement select native. The internal-split guard can only veto
the selected alpha; it cannot select another. Each CV fold refits its own
feature normalization, gradient basis and ridge mapping. The full-pool refit
can behave differently from the held-out folds, so it has no guaranteed gain.

Source recordings and decoded waveforms are distinct across selected examples.
Selection is locked before any development query features or predictions are
computed. Development labels are used for class-balanced sampling and final
scoring, never for gradient extraction, mapping fits, or alpha selection.

Features and gradients are shared across methods to avoid redundant work.
Methods therefore have the same available label budget but **not equal
computation**. The report records shared array extraction, answer-loss calls,
generation calls/timings, paired repaired/harmed counts, and GPU peak allocation
for loss evaluation. Shared-array extraction timing includes the initial model
load; these wall times are not clean per-method training-time benchmarks.

## Run in WSL

Use root because the existing isolated environment and checkpoint cache belong
to root. Select a GPU with at least 24 GiB free on this shared workstation.
The `24gb` loader profile and conservative free-memory requirement are separate.

```bash
cd /mnt/c/ai/akr
/root/.venvs/akr/bin/python scripts/pilot_calibration_pool.py --run-id pool_example
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/.venvs/akr/bin/python scripts/pilot_calibration_pool.py \
  --run-id pool_example --execute --max-seconds 3600
```

Verify a completed run without loading weights or requiring CUDA:

```bash
/root/.venvs/akr/bin/python scripts/pilot_calibration_pool.py \
  --run-id pool_example --cached-only
```

Source/config/environment changes require a new run ID. Interrupted calls can
resume from valid checksummed arrays and journals. Damaged artifacts are
rejected rather than silently accepted. Processed audio is written atomically
and validated before reuse. An incomplete cache-only check fails if it would
need to load the model.

Outputs live under `results/calibration_pilot/<run-id>/`. `REGISTRATION.json`
records the selected source examples. `dogs/POLICY_LOCK.json` records choices
made before development inference. `dogs/REPORT.json` contains scores and costs;
`dogs/STATUS.json` is the current completion state. `VALIDATION.json` records the
initial successful run and no-reload verification. Always check current status
and report checksum; a retained historical report alone is not completion.

Published LoRA adapters have different training data budgets. They are not
included here as a purported fair competitor. A later matched-data LoRA
comparison must use these same registered calibration-pool IDs and separate
development/test evaluation rules.
