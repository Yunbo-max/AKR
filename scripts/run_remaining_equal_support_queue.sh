#!/usr/bin/env bash
set -uo pipefail

cd /root/animal-omni-kv
exec >> results/remaining_equal_support_queue.log 2>&1
export HF_HOME=/root/animal-omni-kv/.hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() { echo "[$(date -u +%FT%TZ)] $*"; }

quarantine_partial() {
  local output=$1
  local summary=$2
  if [[ -f "$output" ]]; then
    mv --backup=numbered "$output" "results/partial_extensions/incomplete_$(basename "$output")"
  fi
  if [[ -f "$summary" ]]; then
    mv --backup=numbered "$summary" "results/partial_extensions/incomplete_$(basename "$summary")"
  fi
}

run_fixed_level() {
  local dataset=$1
  local level=$2
  local config=$3
  local manifest=$4
  local condition=$5
  local split=$6
  local condition_args=()
  if [[ "$condition" != "none" ]]; then condition_args=(--condition "$condition"); fi
  stage "${dataset} K=${level}/class one-query gate"
  if .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
      --config "$config" --manifest "$manifest" "${condition_args[@]}" \
      --split "$split" --model-id Qwen/Qwen2.5-Omni-7B \
      --support-k-per-class "$level" --readout free --limit-query 1 \
      --output "results/partial_extensions/smoke_${dataset}_audio_icl_k${level}_7b.csv" \
      --summary "results/partial_extensions/smoke_${dataset}_audio_icl_k${level}_7b_summary.json" \
      --resume
  then
    local output="results/${dataset}_equal_support_audio_icl_k${level}_7b.csv"
    local summary="results/${dataset}_equal_support_audio_icl_k${level}_7b_summary.json"
    stage "${dataset} K=${level}/class gate passed; run complete validation"
    if ! .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
        --config "$config" --manifest "$manifest" "${condition_args[@]}" \
        --split "$split" --model-id Qwen/Qwen2.5-Omni-7B \
        --support-k-per-class "$level" --readout free \
        --output "$output" --summary "$summary" --resume
    then
      stage "${dataset} K=${level}/class incomplete; quarantine"
      quarantine_partial "$output" "$summary"
    fi
  else
    stage "${dataset} K=${level}/class gate failed"
  fi
}

stage "resume stages skipped when the original queue was externally terminated"
run_fixed_level beans_dogs_lp1 2 configs/beans_dogs.yaml \
  data/manifests/beans_dogs_all_full_lp1.csv lp_0-1000 \
  results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_split.json
run_fixed_level beans_watkins 1 configs/beans_watkins.yaml \
  data/manifests/beans_watkins_protocol.csv none \
  results/beans_watkins_equal_support_split_seed20260814.json
run_fixed_level beans_watkins 2 configs/beans_watkins.yaml \
  data/manifests/beans_watkins_protocol.csv none \
  results/beans_watkins_equal_support_split_seed20260814.json

stage "validation-only A-J cross-prompt pooled-cache transfer"
bash scripts/run_cross_prompt_pooled_kv.sh || stage "cross-prompt diagnostic incomplete"
stage "remaining equal-support queue finished"
