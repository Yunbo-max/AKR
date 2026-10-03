# AKR — Acoustic-to-KV Regression

**Bioacoustic Recognition in Speech Language Models.** Canonical consolidation: `Yunbo-max/AKR`.

**2026-10-03 verification:** 148 CPU tests passed with no failures or skips; all 116 offline release checks passed. [Remote verification](docs/research/REMOTE_VERIFICATION_20261003.json) · [Tested materialization run](https://github.com/Yunbo-max/AKR/actions/runs/37131602379) · [Integration PR #1](https://github.com/Yunbo-max/AKR/pull/1). No new GPU experiment was run; S1 remains pending real-model evaluation.

AKR predicts continuous **additive K/V corrections**, not animal labels or replacement model weights. Labelled support recordings supply negative answer-loss gradients; centred SVD and ridge regression map frozen acoustic features to correction coefficients. An unlabelled query uses a feature pass followed by an intervened generation pass. The Speech LM stays frozen.

## Start here

- [Final experiments: exact data, models, metrics and commands](docs/FINAL_EXPERIMENTS_ZH.md)
- [Research Autopilot audit (actual 2026-10-02 export)](docs/research/RESEARCH_AUTOPILOT_AUDIT_20261003.md)
- [Machine-readable experiment/model/dataset registry](configs/akr_registry.json)
- [Migration coverage and limits](MIGRATION_STATUS.md)
- [Manuscript and figure sources](paper/README_RELEASE.md)
- [Raw-data sources and redistribution boundaries](DATASETS.md)

**No old `RUN.json` or `LOCK.json` is needed by the standalone entrypoint.** Raw audio and accessible model weights are still needed. Missing feature/gradient caches are recomputed; missing historical sample identities cannot be invented. New runs receive new metadata and are not silently called historical replications.

Fresh CPU verification environment (omit the CPU-wheel installation in an existing working GPU environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
PYTHONPATH=src:. python -m pytest -q
python scripts/check_akr_release.py
python scripts/run_akr_release.py plan
```

For real GPU execution, first install PyTorch/torchvision compatible with the actual GPU driver, then `pip install -r requirements-qwen.txt`. Prepare the official waveform files described in the runbook. Plan and check commands do not load model weights.

```bash
python scripts/run_akr_release.py check --tasks S1 --run-id akr_s1_v1
python scripts/run_akr_release.py run --tasks S1 --run-id akr_s1_v1 \
  --profile 24gb --budget-hours 24 --execute
```

`S1` is the only default **new** task: compare native, fixed, continuous AKR, direct ridge, probe-to-text, and probe-to-LM on the same MA-CT support/query identities. It is implemented but **has not been run on a real model in this release**. No 24-hour completion speed is promised; 24 hours is an enforced wall-clock cap with checkpointing.

## What is already measured?

| Evidence | Protocol / result | Scope |
|---|---|---|
| Matched support | MA-CT, 12 clips: ridge 58.67%, audio ICL 17.33% | Same supervision, different fitting routes |
| Historical RQ1 | MA-CT/LP1: native 12.35%, fixed 13.01%, continuous AKR 32.47% | 20-total error-enriched support; five overlapping recording-group draws |
| Historical RQ2 | BD-ID: conditional pooled improves over fixed/random at the largest recorded norm | All strengths retained; not class-routed prompt-transfer results |
| F1 | Rank-4 NMSE .6813 Full / .7641 LP1 | Completed scalar export, not new inference; original large tensors absent |
| F2/F3 | Mixed or limited fixed-setting recognition benefits | Retained, not replaced by historical positive results |

F1/F2/F3 are already completed studies in the supplied report, not mandatory new experiments. Request `--tasks F1,F2,F3` only for recovery/re-evaluation. `--tasks S1,F1,F2,F3` runs all standalone tasks under one time cap. New software/environment/configurations require new run IDs; results are not expected to be byte-identical to an old unavailable environment.

## Models and tasks

Real implemented backbone: frozen **Qwen2.5-Omni Thinker 7B and 3B**, Talker off, BF16, batch one, greedy decoding. Exact pinned revisions and layers are in the registry. **GLM-4-Voice and Qwen3-Omni are not implemented/validated here** and are excluded from automatic queues.

MA-CT is MarmAudio call-type recognition; BD-ID is BEANS Dogs individual identification; BW-SP is BEANS Watkins species recognition. BZ-12 is a separately capped BEANS-Zero diagnostic, not an official full-duration score. No restricted waveform dataset or pretrained model weights are mirrored.

## Repository map

- `src/animal_omni/`, `scripts/`, `results/`: preserved original research implementation and public evidence.
- `akr_closing/`: restored E1–E9 experiments and controls (legacy source-run compatibility retained).
- `akr_final/`: standalone F1/F2/F3 and new S1; frozen support/query identities, strict no-op and hash checks.
- `paper/`: manuscript, bibliography, original source packages/figure assets where available, deterministic plotting data and figure prompts.
- `docs/research/`: provenance-aware audit, scope limits, implementation plan.
- `migration/`: source manifests, overwritten source backups, prior READMEs/workflows.

The historical instructions remain in `migration/legacy_public_README.md` and `REPRODUCE.md`; their runtime versions, untouched-test terminology and earlier claims must be read in their historical context. The modern runbook and evidence audit take precedence for the release's current claims.

## Validation is not experimental success

CPU tests verify mathematics, grouping, data plumbing, checkpoint identity, no query-gradient path and no-op controls. They do not establish GPU throughput, new accuracy, broad model-family generalization, or superiority to a direct classifier. The public source is pinned; the inaccessible later repository was restored only to the extent covered by supplied packages. See the migration ledger for exact coverage.
