#!/usr/bin/env bash
set -uo pipefail

cd /root/animal-omni-kv
exec >> results/watkins_k4_icl_gate.log 2>&1
export HF_HOME=/root/animal-omni-kv/.hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {
  echo "[$(date -u +%FT%TZ)] $*"
}

output=results/beans_watkins_equal_support_audio_icl_k4_7b.csv
summary=results/beans_watkins_equal_support_audio_icl_k4_7b_summary.json
stage "Watkins K=4/class (124 support) one-query feasibility gate"
if .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
    --config configs/beans_watkins.yaml \
    --manifest data/manifests/beans_watkins_protocol.csv \
    --split results/beans_watkins_equal_support_split_seed20260814.json \
    --model-id Qwen/Qwen2.5-Omni-7B \
    --support-k-per-class 4 --readout free --limit-query 1 \
    --output results/partial_extensions/smoke_beans_watkins_audio_icl_k4_7b.csv \
    --summary results/partial_extensions/smoke_beans_watkins_audio_icl_k4_7b_summary.json \
    --resume
then
  stage "Watkins K=4/class gate passed; run complete official validation"
  if ! .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
      --config configs/beans_watkins.yaml \
      --manifest data/manifests/beans_watkins_protocol.csv \
      --split results/beans_watkins_equal_support_split_seed20260814.json \
      --model-id Qwen/Qwen2.5-Omni-7B \
      --support-k-per-class 4 --readout free \
      --output "$output" --summary "$summary" --resume
  then
    stage "Watkins K=4/class incomplete; quarantine partial output"
    if [[ -f "$output" ]]; then
      mv --backup=numbered "$output" \
        "results/partial_extensions/incomplete_$(basename "$output")"
    fi
    if [[ -f "$summary" ]]; then
      mv --backup=numbered "$summary" \
        "results/partial_extensions/incomplete_$(basename "$summary")"
    fi
  fi
else
  stage "Watkins K=4/class blocked by memory/context feasibility gate"
fi
stage "Watkins K=4/class gate finished"
