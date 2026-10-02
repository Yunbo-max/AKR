#!/usr/bin/env bash
set -uo pipefail

cd /root/animal-omni-kv
exec >> results/cross_dataset_icl_order_controls.log 2>&1
export HF_HOME=/root/animal-omni-kv/.hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_control() {
  local dataset=$1
  local config=$2
  local manifest=$3
  local split=$4
  local registry=$5
  local k=$6
  local output="results/${dataset}_icl_order_control_k${k}_interleaved_7b.csv"
  local summary="results/${dataset}_icl_order_control_k${k}_interleaved_7b_summary.json"
  echo "[$(date -u +%FT%TZ)] ${dataset} K=${k}/class interleaved order control"
  if ! .venv/bin/python scripts/evaluate_counterbalanced_audio_icl.py \
      --config "$config" --manifest "$manifest" --split "$split" \
      --order-registry "$registry" --support-k-per-class "$k" \
      --order-mode interleaved --model-id Qwen/Qwen2.5-Omni-7B \
      --output "$output" --summary "$summary" --resume
  then
    echo "[$(date -u +%FT%TZ)] incomplete ${output}; quarantine"
    if [[ -f "$output" ]]; then
      mv --backup=numbered "$output" \
        "results/partial_extensions/incomplete_$(basename "$output")"
    fi
    if [[ -f "$summary" ]]; then
      mv --backup=numbered "$summary" \
        "results/partial_extensions/incomplete_$(basename "$summary")"
    fi
  fi
}

run_control beans_dogs_lp1 configs/beans_dogs.yaml \
  data/manifests/beans_dogs_lp1_equal_support_k2_fullvalid.csv \
  results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_split.json \
  results/beans_dogs_lp1_icl_order_registry_seed20260814.json 2

run_control beans_watkins configs/beans_watkins.yaml \
  data/manifests/beans_watkins_protocol.csv \
  results/beans_watkins_equal_support_split_seed20260814.json \
  results/beans_watkins_icl_order_registry_seed20260814.json 1

echo "[$(date -u +%FT%TZ)] cross-dataset order controls finished"
