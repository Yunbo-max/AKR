# Reproduction runbook

All reported Qwen generation uses Thinker-only BF16 inference, deterministic
decoding, and batch size one. Audio interventions are materialized before model
inference so the paired WAVs can be inspected independently.

## Environment

The verified host is Ubuntu/glibc 2.35, Python 3.10, RTX 3090 24 GB, CUDA 12.6,
Torch 2.13.0+cu126, torchvision 0.28.0+cu126, Transformers 4.52.3,
qwen-omni-utils 0.0.9, scikit-learn 1.7.2, ONNX Runtime 1.23.2, PEFT 0.17.1,
and datasets 3.6.0.

Cached immutable revisions used here are Qwen2.5-Omni-3B
`f75b40e3da2003cdd6e1829b1f420ca70797c34e`, Qwen2.5-Omni-7B
`ae9e1690543ffd5c0221dc27f79834d0294cba00`, and BEANS-Zero
`e9e45ffff7500867c885ef74c245c07e66677084`.

```bash
cd /root/animal-omni-kv
python3 -m venv .venv
.venv/bin/pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -e '.[test,qwen,analysis,data,adaptation]'
.venv/bin/pytest -q
export ANIMAL_HF_HOME="$PWD/.hf-cache"
```

Do not install the AVES repository's historical torchaudio dependency into this
environment. The verified AVES result uses its official ONNX port because the
repository dependency resolver selected an incompatible CUDA 13 torchaudio wheel.

## Frequency inputs and generation

MarmAudio's authoritative intervention manifest contains 546 events × 11 paired
conditions. BEANS commands below use the official fixed test splits and published
duration rules already recorded in their protocol manifests.

```bash
HF_HOME="$ANIMAL_HF_HOME" .venv/bin/python scripts/evaluate_qwen.py \
  --config configs/marmaudio_minimal.yaml \
  --manifest data/manifests/marmaudio_expert_validation_interventions.csv \
  --model-id Qwen/Qwen2.5-Omni-7B --batch-size 1 --resume \
  --output results/expert_qwen7b_b1_clean.csv

HF_HOME="$ANIMAL_HF_HOME" .venv/bin/python scripts/evaluate_qwen.py \
  --config configs/beans_dogs.yaml \
  --manifest data/manifests/beans_dogs_test_all_interventions.csv \
  --model-id Qwen/Qwen2.5-Omni-7B --batch-size 1 --resume \
  --output results/beans_dogs_frequency_qwen7b.csv

HF_HOME="$ANIMAL_HF_HOME" .venv/bin/python scripts/evaluate_qwen.py \
  --config configs/beans_watkins.yaml \
  --manifest data/manifests/beans_watkins_test_all_interventions.csv \
  --model-id Qwen/Qwen2.5-Omni-7B --batch-size 1 --resume \
  --output results/beans_watkins_frequency_qwen7b.csv
```

Each result is summarized with `scripts/summarize_results.py`, which reports
accuracy, macro-F1, and paired event-bootstrap full-minus-condition intervals.

## Frozen-state probes

Representations are mean-pooled only over Qwen audio placeholder tokens at every
Thinker hidden-state level. MarmAudio uses nested recording-grouped OOF selection;
BEANS uses validation-selected layer/Ridge alpha and one untouched fixed test.

```bash
HF_HOME="$ANIMAL_HF_HOME" .venv/bin/python scripts/extract_representations.py \
  --config configs/marmaudio_minimal.yaml \
  --manifest data/manifests/marmaudio_expert_validation_interventions.csv \
  --condition full_0-8k --model-id Qwen/Qwen2.5-Omni-7B \
  --output-dir results/reps_full_7b

.venv/bin/python scripts/nested_group_probe.py \
  --representation-dir results/reps_full_7b \
  --output-predictions results/probe_7b_oof.csv \
  --output-summary results/probe_7b_oof_summary.json
```

`scripts/nested_group_probe_cross_condition.py` selects the probe using only full
inner folds and applies that same readout to paired degraded OOF queries.

## Oracle and query-label-free KV

`KVDeltaHooks` injects into pre-RoPE `k_proj`/`v_proj` outputs. Oracle gradients
use query labels and are upper bounds. Deployable runs select gradients only from
labeled support examples; query labels are read after generation solely for
metrics.

```bash
HF_HOME="$ANIMAL_HF_HOME" .venv/bin/python scripts/evaluate_oracle_kv.py \
  --config configs/marmaudio_minimal.yaml \
  --manifest data/manifests/marmaudio_expert_validation_interventions.csv \
  --predictions results/expert_qwen7b_b1_clean.csv \
  --model-id Qwen/Qwen2.5-Omni-7B --condition lp_0-1000 \
  --gradient-dir results/gradients_lp1k_7b \
  --output results/oracle_kv_lp1k_7b.csv --resume

.venv/bin/python scripts/analyze_kv_geometry.py \
  --gradient-dir results/gradients_lp1k_7b \
  --output results/kv_geometry_lp1k_7b.json
```

For BEANS fixed splits, run `extract_support_kv_gradients.py`, extract degraded
support/validation/test representations, select eta on validation with
`evaluate_fixed_split_conditional_kv.py`, then invoke it once on test with
K=1/5/10/20. The exact support ordering and selected validation artifacts are
stored beside each result.

## Specialist and adaptation baselines

- `extract_aves_onnx_representations.py` + `nested_group_probe.py` or
  `fixed_split_probe.py`: AVES-bio. The extractor accepts an explicit CPU/CUDA
  ONNX provider; CUDA output was checked against CPU before resuming the full
  Dogs/Watkins runs.
- `evaluate_marmaudio_specialist.py`: official 96 kHz MarmAudio ResNet-50.
- `train_thinker_lora.py` + `evaluate_thinker_lora.py`: Qwen Thinker LoRA.
- `fixed_split_probe.py`: fixed BEANS frozen-state probe.
- `evaluate_beans_zero.py`: official instruction text on the declared streamed
  BEANS-Zero core subset.

## Equal-supervision correction

The current paper protocol is frozen in `FAIR_GAP_12H_RUN.md`. The central
commands below replace zero-shot-versus-full-probe comparisons with identical
K-per-class support and replace raw eta with relative KV norm alpha.

```bash
HF_HOME="$ANIMAL_HF_HOME" .venv/bin/python scripts/evaluate_candidate_scoring.py \
  --config configs/marmaudio_fair_gap.yaml \
  --manifest data/manifests/marmaudio_expert_validation.csv \
  --model-id Qwen/Qwen2.5-Omni-7B --candidate-batch-size 6 \
  --output results/marmaudio_candidate_scoring_full546_7b.csv

.venv/bin/python scripts/evaluate_equal_support_readouts.py \
  --manifest data/manifests/marmaudio_expert_validation.csv \
  --split results/marmaudio_equal_support_split_seed20260814.json \
  --representation-dir results/reps_full_7b --feature-layer 28 \
  --output results/marmaudio_equal_support_readouts_7b.csv \
  --summary results/marmaudio_equal_support_readouts_7b_summary.json

HF_HOME="$ANIMAL_HF_HOME" .venv/bin/python \
  scripts/evaluate_fixed_split_relative_token_kv.py \
  --config configs/beans_dogs.yaml \
  --manifest data/manifests/beans_dogs_all_full_lp1.csv \
  --gradient-dir results/gradients_beans_dogs_train_lp1_tokenwise_7b \
  --support-split results/beans_dogs_support_k_tokenwise_split.json \
  --representation-dir results/token_reps_beans_dogs_lp1_7b_layer28 \
  --model-id Qwen/Qwen2.5-Omni-7B --query-split valid --support-k 20 \
  --relative-alphas .003 .01 .03 --method-batch-size 1 \
  --output results/beans_dogs_relative_token_kv_lp1_fullvalid.csv --resume

HF_HOME="$ANIMAL_HF_HOME" .venv/bin/python scripts/evaluate_oracle_kv_controls.py \
  --config configs/marmaudio_minimal.yaml \
  --predictions results/expert_qwen7b_b1_clean.csv \
  --intervention-manifest data/manifests/marmaudio_expert_validation_interventions.csv \
  --condition lp_0-1000 --limit 12 \
  --model-id Qwen/Qwen2.5-Omni-7B --scope full_prefill \
  --relative-alphas .0001 .0003 .001 .003 \
  --output results/marmaudio_oracle_kv_controls_fullprefill_lp1_7b.csv
```

The Dogs class-dictionary fallback is validation-only and has its gate written
before evaluation in `FAIR_GAP_12H_RUN.md`. Do not invoke it on test unless its
machine-readable summary says `gate.passed=true`.

## Final integrity and figures

```bash
.venv/bin/python scripts/audit_artifacts.py \
  --root . --output results/artifact_audit.json
.venv/bin/python scripts/audit_fair_gap.py \
  --root . --output results/fair_gap_artifact_audit.json
.venv/bin/python scripts/analyze_fair_gap_statistics.py \
  --root . --output results/fair_gap_paired_statistics.json
.venv/bin/python scripts/build_fair_gap_tables.py
MPLBACKEND=Agg .venv/bin/python scripts/plot_fair_gap_results.py
.venv/bin/pytest -q
```

`scripts/make_core_figures.py` renders KV geometry/recovery, while
`scripts/make_multitask_frequency_figure.py` renders the three task-level
low-pass curves and band-removal heatmap. `RESULTS.md` is the authoritative ledger;
exploratory or invalidated files are explicitly excluded there.
