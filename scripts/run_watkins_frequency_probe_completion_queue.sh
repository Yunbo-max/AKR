#!/usr/bin/env bash
set -euo pipefail

cd /root/animal-omni-kv
exec >> results/watkins_frequency_probe_completion_queue.log 2>&1
export HF_HOME=/root/animal-omni-kv/.hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {
  echo "[$(date -u +%FT%TZ)] $*"
}

probe_condition() {
  local cutoff=$1
  local directory=$2
  stage "probe Watkins lp${cutoff}"
  .venv/bin/python scripts/fixed_split_probe.py \
    --representation-dir "$directory" \
    --output-predictions "results/beans_watkins_probe_lp${cutoff}_7b_test.csv" \
    --output-summary "results/beans_watkins_probe_lp${cutoff}_7b_summary.json"
}

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

probe_condition 1 results/reps_beans_watkins_lp1_7b
probe_condition 2 results/reps_beans_watkins_frequency_extra_7b/lp_0-2000
probe_condition 4 results/reps_beans_watkins_frequency_extra_7b/lp_0-4000
probe_condition 6 results/reps_beans_watkins_frequency_extra_7b/lp_0-6000
probe_condition 8 results/reps_beans_watkins_frequency_extra_7b/lp_0-8000
.venv/bin/python scripts/combine_fixed_split_frequency_probes.py \
  --root . --dataset watkins --output results/beans_watkins_frequency_probe_7b_summary.json
.venv/bin/python scripts/fixed_split_probe_cross_condition.py --root . --dataset dogs \
  --output-predictions results/beans_dogs_probe_fulltrained_cross_frequency_7b.csv \
  --output-summary results/beans_dogs_probe_fulltrained_cross_frequency_7b_summary.json
.venv/bin/python scripts/fixed_split_probe_cross_condition.py --root . --dataset watkins \
  --output-predictions results/beans_watkins_probe_fulltrained_cross_frequency_7b.csv \
  --output-summary results/beans_watkins_probe_fulltrained_cross_frequency_7b_summary.json
.venv/bin/python scripts/analyze_frequency_probe_per_class.py --root . \
  --output results/beans_frequency_probe_per_class_7b.json \
  --figure results/figures/fig_frequency_probe_per_class.png

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
stage "Watkins frequency-probe completion queue finished"
