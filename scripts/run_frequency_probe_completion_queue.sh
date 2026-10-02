#!/usr/bin/env bash
set -euo pipefail

cd /root/animal-omni-kv
exec >> results/frequency_probe_completion_queue.log 2>&1
export HF_HOME=/root/animal-omni-kv/.hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {
  echo "[$(date -u +%FT%TZ)] $*"
}

probe_condition() {
  local dataset=$1
  local cutoff=$2
  local directory=$3
  stage "probe ${dataset} lp${cutoff}"
  .venv/bin/python scripts/fixed_split_probe.py \
    --representation-dir "$directory" \
    --output-predictions "results/beans_${dataset}_probe_lp${cutoff}_7b_test.csv" \
    --output-summary "results/beans_${dataset}_probe_lp${cutoff}_7b_summary.json"
}

stage "resume Dogs 2/4/6/8 kHz representations"
.venv/bin/python scripts/extract_representations.py \
  --config configs/beans_dogs.yaml \
  --manifest data/manifests/beans_dogs_all_lowpass_extra.csv \
  --conditions lp_0-2000 lp_0-4000 lp_0-6000 lp_0-8000 \
  --model-id Qwen/Qwen2.5-Omni-7B \
  --output-dir results/reps_beans_dogs_frequency_extra_7b

stage "fill Dogs 1 kHz complete official split"
.venv/bin/python scripts/extract_representations.py \
  --config configs/beans_dogs.yaml \
  --manifest data/manifests/beans_dogs_all_full_lp1.csv \
  --condition lp_0-1000 \
  --model-id Qwen/Qwen2.5-Omni-7B \
  --output-dir results/reps_beans_dogs_lp1_7b

stage "verify Dogs representations before any probe"
.venv/bin/python scripts/verify_fixed_split_frequency_representations.py \
  --root . --dataset dogs \
  --output results/beans_dogs_frequency_representations_7b_audit.json

probe_condition dogs 1 results/reps_beans_dogs_lp1_7b
probe_condition dogs 2 results/reps_beans_dogs_frequency_extra_7b/lp_0-2000
probe_condition dogs 4 results/reps_beans_dogs_frequency_extra_7b/lp_0-4000
probe_condition dogs 6 results/reps_beans_dogs_frequency_extra_7b/lp_0-6000
probe_condition dogs 8 results/reps_beans_dogs_frequency_extra_7b/lp_0-8000
.venv/bin/python scripts/combine_fixed_split_frequency_probes.py \
  --root . --dataset dogs --output results/beans_dogs_frequency_probe_7b_summary.json

stage "extract Watkins 2/4/6/8 kHz complete official splits"
.venv/bin/python scripts/extract_representations.py \
  --config configs/beans_watkins.yaml \
  --manifest data/manifests/beans_watkins_all_lowpass_extra.csv \
  --conditions lp_0-2000 lp_0-4000 lp_0-6000 lp_0-8000 \
  --model-id Qwen/Qwen2.5-Omni-7B \
  --output-dir results/reps_beans_watkins_frequency_extra_7b

stage "fill Watkins 1 kHz complete official split"
.venv/bin/python scripts/extract_representations.py \
  --config configs/beans_watkins.yaml \
  --manifest data/manifests/beans_watkins_all_full_lp1.csv \
  --condition lp_0-1000 \
  --model-id Qwen/Qwen2.5-Omni-7B \
  --output-dir results/reps_beans_watkins_lp1_7b

stage "verify Watkins representations before any probe"
.venv/bin/python scripts/verify_fixed_split_frequency_representations.py \
  --root . --dataset watkins \
  --output results/beans_watkins_frequency_representations_7b_audit.json

probe_condition watkins 1 results/reps_beans_watkins_lp1_7b
probe_condition watkins 2 results/reps_beans_watkins_frequency_extra_7b/lp_0-2000
probe_condition watkins 4 results/reps_beans_watkins_frequency_extra_7b/lp_0-4000
probe_condition watkins 6 results/reps_beans_watkins_frequency_extra_7b/lp_0-6000
probe_condition watkins 8 results/reps_beans_watkins_frequency_extra_7b/lp_0-8000
.venv/bin/python scripts/combine_fixed_split_frequency_probes.py \
  --root . --dataset watkins --output results/beans_watkins_frequency_probe_7b_summary.json

stage "build frequency-gap figure and authoritative tables"
.venv/bin/python scripts/plot_probe_native_frequency_gap.py \
  --root . --output results/figures/fig_probe_native_frequency_gap.png
.venv/bin/python scripts/build_paper_tables.py \
  --root . --output-dir results/paper_tables
.venv/bin/python scripts/audit_artifacts.py \
  --root . --output results/artifact_audit.json
.venv/bin/python scripts/audit_fair_gap.py \
  --root . --output results/fair_gap_artifact_audit.json
.venv/bin/python -m pytest -q
stage "frequency-probe completion queue finished"
