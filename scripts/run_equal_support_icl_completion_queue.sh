#!/usr/bin/env bash
set -uo pipefail

cd /root/animal-omni-kv
exec >> results/equal_support_icl_completion_queue.log 2>&1
export HF_HOME=/root/animal-omni-kv/.hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {
  echo "[$(date -u +%FT%TZ)] $*"
}

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

run_marm_level() {
  local level=$1
  stage "MarmAudio K=${level}/class free one-query memory gate"
  if .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
      --config configs/marmaudio_fair_gap.yaml \
      --manifest data/manifests/marmaudio_expert_validation.csv \
      --split results/marmaudio_equal_support_split_seed20260814.json \
      --model-id Qwen/Qwen2.5-Omni-7B \
      --support-k-per-class "$level" --readout free --limit-query 1 \
      --output "results/partial_extensions/smoke_marmaudio_audio_icl_k${level}_7b.csv" \
      --summary "results/partial_extensions/smoke_marmaudio_audio_icl_k${level}_7b_summary.json" \
      --resume
  then
    stage "MarmAudio K=${level}/class free gate passed; run frozen 75-query set"
    local free_output="results/marmaudio_equal_support_audio_icl_k${level}_7b.csv"
    local free_summary="results/marmaudio_equal_support_audio_icl_k${level}_7b_summary.json"
    if .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
      --config configs/marmaudio_fair_gap.yaml \
      --manifest data/manifests/marmaudio_expert_validation.csv \
      --split results/marmaudio_equal_support_split_seed20260814.json \
      --model-id Qwen/Qwen2.5-Omni-7B \
      --support-k-per-class "$level" --readout free \
      --output "$free_output" \
      --summary "$free_summary" \
      --resume
    then
      stage "MarmAudio K=${level}/class candidate one-query memory gate"
    else
      stage "MarmAudio K=${level}/class full free run incomplete; quarantine partial"
      quarantine_partial "$free_output" "$free_summary"
      return
    fi
    if .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
        --config configs/marmaudio_fair_gap.yaml \
        --manifest data/manifests/marmaudio_expert_validation.csv \
        --split results/marmaudio_equal_support_split_seed20260814.json \
        --model-id Qwen/Qwen2.5-Omni-7B \
        --support-k-per-class "$level" --readout candidate --limit-query 1 \
        --output "results/partial_extensions/smoke_marmaudio_audio_icl_candidate_k${level}_7b.csv" \
        --summary "results/partial_extensions/smoke_marmaudio_audio_icl_candidate_k${level}_7b_summary.json" \
        --resume
    then
      local candidate_output="results/marmaudio_equal_support_audio_icl_candidate_k${level}_7b.csv"
      local candidate_summary="results/marmaudio_equal_support_audio_icl_candidate_k${level}_7b_summary.json"
      if ! .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
        --config configs/marmaudio_fair_gap.yaml \
        --manifest data/manifests/marmaudio_expert_validation.csv \
        --split results/marmaudio_equal_support_split_seed20260814.json \
        --model-id Qwen/Qwen2.5-Omni-7B \
        --support-k-per-class "$level" --readout candidate \
        --output "$candidate_output" \
        --summary "$candidate_summary" \
        --resume
      then
        stage "MarmAudio K=${level}/class full candidate run incomplete; quarantine partial"
        quarantine_partial "$candidate_output" "$candidate_summary"
      fi
    else
      stage "MarmAudio K=${level}/class candidate blocked by one-query memory/context gate"
    fi
  else
    stage "MarmAudio K=${level}/class free blocked by one-query memory/context gate"
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
  if [[ "$condition" != "none" ]]; then
    condition_args=(--condition "$condition")
  fi
  stage "${dataset} K=${level}/class free one-query memory gate"
  if .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
      --config "$config" --manifest "$manifest" "${condition_args[@]}" \
      --split "$split" --model-id Qwen/Qwen2.5-Omni-7B \
      --support-k-per-class "$level" --readout free --limit-query 1 \
      --output "results/partial_extensions/smoke_${dataset}_audio_icl_k${level}_7b.csv" \
      --summary "results/partial_extensions/smoke_${dataset}_audio_icl_k${level}_7b_summary.json" \
      --resume
  then
    stage "${dataset} K=${level}/class gate passed; run complete official validation"
    local full_output="results/${dataset}_equal_support_audio_icl_k${level}_7b.csv"
    local full_summary="results/${dataset}_equal_support_audio_icl_k${level}_7b_summary.json"
    if ! .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
      --config "$config" --manifest "$manifest" "${condition_args[@]}" \
      --split "$split" --model-id Qwen/Qwen2.5-Omni-7B \
      --support-k-per-class "$level" --readout free \
      --output "$full_output" \
      --summary "$full_summary" \
      --resume
    then
      stage "${dataset} K=${level}/class full run incomplete; quarantine partial"
      quarantine_partial "$full_output" "$full_summary"
    fi
  else
    stage "${dataset} K=${level}/class blocked by one-query memory/context gate"
  fi
}

stage "complete missing MarmAudio equal-support cells"
stage "MarmAudio K=2/class candidate readout (K=2 free already complete)"
if ! .venv/bin/python scripts/evaluate_equal_support_audio_icl.py \
  --config configs/marmaudio_fair_gap.yaml \
  --manifest data/manifests/marmaudio_expert_validation.csv \
  --split results/marmaudio_equal_support_split_seed20260814.json \
  --model-id Qwen/Qwen2.5-Omni-7B \
  --support-k-per-class 2 --readout candidate \
  --output results/marmaudio_equal_support_audio_icl_candidate_k2_7b.csv \
  --summary results/marmaudio_equal_support_audio_icl_candidate_k2_7b_summary.json \
  --resume
then
  stage "MarmAudio K=2 candidate failed; quarantine partial"
  quarantine_partial results/marmaudio_equal_support_audio_icl_candidate_k2_7b.csv \
    results/marmaudio_equal_support_audio_icl_candidate_k2_7b_summary.json
fi
run_marm_level 4
run_marm_level 8

stage "complete missing Dogs and Watkins equal-support audio ICL cells"
run_fixed_level beans_dogs_lp1 2 configs/beans_dogs.yaml \
  data/manifests/beans_dogs_all_full_lp1.csv lp_0-1000 \
  results/beans_dogs_lp1_equal_support_k1_k2_fullvalid_split.json
run_fixed_level beans_watkins 1 configs/beans_watkins.yaml \
  data/manifests/beans_watkins_protocol.csv none \
  results/beans_watkins_equal_support_split_seed20260814.json
run_fixed_level beans_watkins 2 configs/beans_watkins.yaml \
  data/manifests/beans_watkins_protocol.csv none \
  results/beans_watkins_equal_support_split_seed20260814.json

stage "validation-only cross-prompt transfer for the frozen A-J pooled class field"
bash scripts/run_cross_prompt_pooled_kv.sh || \
  stage "cross-prompt pooled diagnostic failed or was interrupted; no summary promoted"

stage "rebuild fair tables/audits after feasible ICL cells"
.venv/bin/python scripts/analyze_fair_gap_statistics.py --root . \
  --output results/fair_gap_paired_statistics.json || \
  stage "paired statistics rebuild failed; inspect incomplete ICL artifacts"
.venv/bin/python scripts/build_fair_gap_tables.py --root . \
  --output-dir results/fair_gap_tables || true
.venv/bin/python scripts/plot_fair_gap_results.py --root . \
  --output-dir figures || true
.venv/bin/python scripts/audit_fair_gap.py --root . \
  --output results/fair_gap_artifact_audit.json || true
.venv/bin/python -m pytest -q
stage "equal-support ICL completion queue finished"
