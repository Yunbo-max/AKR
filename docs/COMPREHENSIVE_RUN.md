# Comprehensive experiments in WSL

The comprehensive runner extends the original registered rebuttal protocol. It prepares both Qwen2.5 sizes on the three core supervised tasks and 12 derived BEANS-Zero component tasks, with explicit baseline coverage. The original `configs/akr_rebuttal.yaml` remains available for the original 12-job study.

## Scope

| Comparison | Datasets | Supervision |
|---|---|---|
| Qwen2.5 7B and 3B AKR, Ridge, probe-to-text, probe-to-LM, no-op and matched controls | MarmAudio, Dogs, Watkins; Full and LP1; three fixed support draws | Shared fit/calibration/guard/query identities across model sizes |
| AKR and all matched controls on derived BEANS-Zero | 12 separate component tasks; Full and LP1; three fixed draws | One recording per class for each of fit/calibration/guard; remaining eligible recordings for query; rank 1 |
| AVES + fixed Ridge | MarmAudio, Dogs, Watkins and derived BEANS-Zero components; same registered episodes | Fit only on each episode's combined support |
| MarmAudio ResNet-50 | MarmAudio; same query IDs; original 96 kHz and lowpass at original rate | External pretrained specialist; not matched training supervision |
| Published Dogs/Watkins LoRA adapters | Corresponding Qwen7B query IDs and conditions | External published adapter training; not matched training supervision |
| Original BEANS-Zero instruction evaluation | Exactly 2,950 registered examples, 12 components, 10-second cap | Separate from the new supervised comparison; not an official full-duration benchmark |
| Conditional GLM-4-Voice and Qwen3-Omni | Same supervised task matrix and BEANS-Zero diagnostic when explicitly selected | Full AKR backend interfaces implemented; production execution pending |

The primary AKR configuration has 180 model/task/seed/condition jobs and 3,180 method units, including no-op checks, plus 110 baseline/diagnostic jobs. The original three supervised tasks account for 36 AKR jobs and 636 units. These are not independent statistical trials. Specialist jobs are shared across model sizes, and all published adapter comparisons disclose their different training data.

The derived BEANS-Zero registration retains 2,002 canonical recording groups across all 12 components, excluding 945 rows from classes with fewer than four distinct groups and three duplicate recordings. Every draw uses 212 fit, 212 calibration, 212 guard and 1,366 query recordings in total. A component has its own eligible label vocabulary and closed-label prompt, with a 64-token output limit for long taxonomic names. This is a new supervised re-evaluation of previously available evaluation data, not the original zero-shot benchmark. Components and repeated draws can share sources; report them separately and do not treat them as independent datasets.

## Environments

The primary environment is `/root/.venvs/akr`; source is `/mnt/c/ai/akr`. The compatibility link `/root/animal-omni-kv` resolves old manifest paths. Use `wsl.exe -u root` from PowerShell because another project's WSL default user may not have access to this environment/cache.

Optional environments are isolated from the primary Transformers version:

```bash
# Inside WSL, from /mnt/c/ai/akr
bash scripts/setup_qwen3_env.sh
bash scripts/setup_glm_env.sh
```

They reuse the primary environment's CUDA/scientific packages through a read-only search path. Qwen3 pins Transformers 4.57.3; GLM pins 4.44.1; the primary environment remains 4.52.3. Setup scripts and requirement files record the compatible versions. The GLM source and tokenizer revisions are recorded in backend provenance.

## Plan and check

```powershell
wsl.exe -u root bash -lc 'cd /mnt/c/ai/akr && /root/.venvs/akr/bin/python scripts/run_akr_comprehensive.py plan'
wsl.exe -u root bash -lc 'cd /mnt/c/ai/akr && /root/.venvs/akr/bin/python scripts/run_akr_comprehensive.py check --run-id comprehensive_primary_v1'
```

`check` validates local checkpoint shards, architecture imports, audio, baseline assets and frozen role identities in each required interpreter. It does not load large model weights. Missing baseline data makes preflight fail before any experiment starts.

## Full primary execution

```powershell
wsl.exe -u root bash -lc 'cd /mnt/c/ai/akr && CUDA_VISIBLE_DEVICES=0 /root/.venvs/akr/bin/python scripts/run_akr_comprehensive.py run --execute --run-id comprehensive_primary_v1 --budget-hours 24'
```

Use the same command to resume after the time cap. The cap applies to the whole invocation. Changed code, environment, config or source audio requires a fresh run ID. Full completion is established from per-query journals, never from a successful process exit or a partial average.

Use `--sections akr`, `--sections baselines`, or `--sections diagnostic` for a separately identified subset. Use `--models qwen7b` or `--models qwen3b` to run one size. A different selection requires a different run ID. `--assets path.yaml` overrides checkpoint locations using keys `aves`, `marmaudio_code`, `marmaudio_checkpoint`, `marmaudio_labels`, `lora_dogs`, and `lora_watkins`. AVES and the specialist use CPU in this launcher; neural language-model workers use the selected CUDA device.

## Conditional families

After deciding that the primary results justify another family, explicitly select it and enable it:

```powershell
wsl.exe -u root bash -lc 'cd /mnt/c/ai/akr && /root/.venvs/akr/bin/python scripts/run_akr_comprehensive.py check --models glm4voice9b,qwen3omni --enable-conditional --run-id comprehensive_conditional_v1'
```

The equivalent `run --execute` command launches those experiments. No default expansion enables them. The launcher chooses `/root/.venvs/akr-glm/bin/python` and `/root/.venvs/akr-qwen3/bin/python` automatically; `--model-python TAG=/absolute/path/to/python` overrides a worker interpreter.

CPU tests exercise actual tiny model architectures and masks, gradients, generation and K/V intervention. They do not prove that the full 9B/30B checkpoints fit the available memory or complete long audio backward passes. Those real model acceptance checks remain conditional, as requested.

## Reports and data preparation

```powershell
wsl.exe -u root bash -lc 'cd /mnt/c/ai/akr && /root/.venvs/akr/bin/python scripts/run_akr_comprehensive.py report --run-id comprehensive_primary_v1'
```

The aggregate plan/status/coverage live under `results/akr_comprehensive/<run-id>`. AKR journals live under `results/akr_rebuttal/<run-id>-<model>`. Baseline journals, fixed input hashes and metrics live in the aggregate run's `baselines` directory. Historical results are not copied into a new run.

BEANS-Zero was materialized and checked against the registered manifest: all 2,950 IDs, metadata and waveform paths are present. To prepare it again from the pinned local snapshot:

```bash
/root/.venvs/akr/bin/python scripts/materialize_beans_zero_subset.py \
  --source-dir /root/.cache/huggingface/hub/datasets--EarthSpeciesProject--BEANS-Zero/snapshots/e9e45ffff7500867c885ef74c245c07e66677084 \
  --target-manifest data/manifests/beans_zero_targets_fullscan_cap10.csv \
  --output-dir data/beans/beans_zero_targets_fullscan_cap10 \
  --output-manifest data/downloads/beans_zero_verification.csv \
  --per-task 0 --target-rate 16000 --duration-cap 10 --minimum-duration 0.1 --resume
```

The derived supervised manifests and complete exclusion/grouping record are in `data/manifests/beans_zero_akr_v1/registration.json`. Preparation groups by source recording and decoded mono PCM identity, including transitive matches. It chooses a deterministic canonical example per group and excludes conflicting labels. It never reads model outcomes. Repeating preparation verifies the same registration; changed inputs require a new output directory:

```bash
/root/.venvs/akr/bin/python scripts/prepare_beans_zero_akr.py
```

Planning verifies the component manifest, vocabulary and original source against this registration. Preflight also checks that the actual waveform bytes still match those used to establish the groups. Full and LP1 use identical role identities, shared across all model families.

## Local validation on 2026-11-06

- Full primary preflight passed for all 180 AKR jobs and 110 applicable baseline/diagnostic jobs, without starting full GPU experiments. Run ID: `comprehensive_ready_20261106_v3`. Consolidated evidence: `results/readiness/FINAL_READINESS.json`.
- Core suite: **254 passed, 6 skipped**. The skips require the optional architecture environments; GLM's isolated suite passed 11 tests and Qwen3's passed 9 tests.
- Actual Qwen2.5-Omni-3B: **27 methods, 54 query records** on a small eight-recording Dogs smoke. Alpha-zero and zero-field matched native output exactly; interventions and reports were verified. A second pass resumed entirely from the journals without loading the model. Evidence: `results/readiness/akr_methods/smoke_methods_20261106_final/VALIDATION.json`.
- Real baseline checks passed for AVES, the MarmAudio specialist, Dogs and Watkins LoRA, and the original BEANS-Zero Qwen7B diagnostic. Fresh checks after final changes also passed for supervised BEANS-Zero AVES and the original diagnostic. Each baseline smoke predicts one query; it cannot mark a full experiment complete. Evidence is under `results/readiness/baseline_integration_20261106_220824` and `results/readiness/bz_final_smoke_20261106_221731`.
- All four main checkpoints and their required backend imports passed local dependency checks. Optional GLM/Qwen3 production-weight GPU inference remains conditional and untested.

These are software/readiness checks, not accuracy results for the registered full experiments. The small samples do not establish whether AKR improves recognition. Earlier smoke provenance explicitly records later source changes; the final 27-method run and fresh BEANS checks match the final source.
