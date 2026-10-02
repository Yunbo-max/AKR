#!/usr/bin/env bash
set -uo pipefail

cd /root/animal-omni-kv
exec >> results/marm_icl_order_controls.log 2>&1
export HF_HOME=/root/animal-omni-kv/.hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for specification in "2 blocked" "2 interleaved" "8 interleaved"; do
  read -r k mode <<< "$specification"
  echo "[$(date -u +%FT%TZ)] MarmAudio K=${k}/class ${mode} counterbalanced order"
  output="results/marmaudio_icl_order_control_k${k}_${mode}_7b.csv"
  summary="results/marmaudio_icl_order_control_k${k}_${mode}_7b_summary.json"
  if ! .venv/bin/python scripts/evaluate_counterbalanced_audio_icl.py \
      --config configs/marmaudio_fair_gap.yaml \
      --manifest data/manifests/marmaudio_expert_validation.csv \
      --split results/marmaudio_equal_support_split_seed20260814.json \
      --order-registry results/marmaudio_icl_order_registry_seed20260814.json \
      --support-k-per-class "$k" --order-mode "$mode" \
      --model-id Qwen/Qwen2.5-Omni-7B \
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
done
echo "[$(date -u +%FT%TZ)] start Dogs/Watkins interleaved order controls"
bash scripts/run_cross_dataset_icl_order_controls.sh
echo "[$(date -u +%FT%TZ)] MarmAudio order controls finished"
