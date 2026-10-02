#!/usr/bin/env bash
set -euo pipefail

cd /root/animal-omni-kv
export HF_HOME=/root/animal-omni-kv/.hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_prompt() {
  local name=$1
  local prompt=$2
  local output="results/beans_dogs_AJ_cross_prompt_${name}_7b.csv"
  .venv/bin/python scripts/evaluate_probe_routed_class_kv.py \
    --config configs/beans_dogs.yaml \
    --manifest data/manifests/beans_dogs_all_full_lp1.csv \
    --condition lp_0-1000 --query-split valid \
    --equal-support-split results/beans_dogs_lp1_tokenwise_equal_support_fullvalid_split.json \
    --support-k-per-class 2 \
    --gradient-dir results/gradients_beans_dogs_train_lp1_tokenwise_7b_AJ \
    --representation-dir results/token_reps_beans_dogs_lp1_7b_layer28 \
    --feature-layer 28 --ridge-alpha 1 \
    --label-map configs/beans_dogs_arbitrary_AJ_labels.json \
    --model-id Qwen/Qwen2.5-Omni-7B \
    --prompt "$prompt" --prompt-name "$name" --max-new-tokens 1 \
    --methods native --relative-alphas 0 \
    --output "$output"
  .venv/bin/python scripts/evaluate_probe_routed_class_kv.py \
    --config configs/beans_dogs.yaml \
    --manifest data/manifests/beans_dogs_all_full_lp1.csv \
    --condition lp_0-1000 --query-split valid \
    --equal-support-split results/beans_dogs_lp1_tokenwise_equal_support_fullvalid_split.json \
    --support-k-per-class 2 \
    --gradient-dir results/gradients_beans_dogs_train_lp1_tokenwise_7b_AJ \
    --representation-dir results/token_reps_beans_dogs_lp1_7b_layer28 \
    --feature-layer 28 --ridge-alpha 1 \
    --label-map configs/beans_dogs_arbitrary_AJ_labels.json \
    --model-id Qwen/Qwen2.5-Omni-7B \
    --prompt "$prompt" --prompt-name "$name" --max-new-tokens 1 \
    --methods probe_class_pooled --relative-alphas .03 \
    --output "$output" --resume
}

run_prompt paraphrase \
  "Listen to the recording and select its registered sound code. Valid codes are A, B, C, D, E, F, G, H, I, and J. Reply with exactly one code and nothing else."

run_prompt reverse_order \
  "Which arbitrary acoustic category best matches this recording? Choose exactly one from J, I, H, G, F, E, D, C, B, A. Output only that letter."

.venv/bin/python scripts/summarize_cross_prompt_kv.py \
  --predictions results/beans_dogs_AJ_cross_prompt_paraphrase_7b.csv \
  --predictions results/beans_dogs_AJ_cross_prompt_reverse_order_7b.csv \
  --expected-n 139 --output results/beans_dogs_AJ_cross_prompt_7b_summary.json
