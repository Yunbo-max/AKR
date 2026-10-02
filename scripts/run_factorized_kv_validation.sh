#!/usr/bin/env bash
set -euo pipefail

cd /root/animal-omni-kv
export HF_HOME=/root/animal-omni-kv/.hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

.venv/bin/python scripts/make_factorized_validation_split.py \
  --manifest data/manifests/beans_dogs_all_full_lp1.csv \
  --condition lp_0-1000 --split valid --selection-per-class 3 \
  --output results/beans_dogs_AJ_factorized_validation_split.json

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
  --model-id Qwen/Qwen2.5-Omni-7B --max-new-tokens 1 \
  --query-event-list results/beans_dogs_AJ_factorized_validation_split.json \
  --query-event-key selection \
  --factorized-ranks 2 4 8 \
  --methods probe_class_pooled probe_class_factorized_r2 \
            probe_class_factorized_r4 probe_class_factorized_r8 \
            probe_class_tokenwise \
  --method-batch-size 5 \
  --relative-alphas .01 \
  --output results/beans_dogs_AJ_factorized_kv_lp1_rank_selection.csv --resume

.venv/bin/python scripts/select_factorized_rank.py \
  --predictions results/beans_dogs_AJ_factorized_kv_lp1_rank_selection.csv \
  --expected-n 30 --output results/beans_dogs_AJ_factorized_rank_selection.json

selected_method=$(jq -r .selected_method results/beans_dogs_AJ_factorized_rank_selection.json)
selected_rank=$(jq -r .selected_rank results/beans_dogs_AJ_factorized_rank_selection.json)

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
  --model-id Qwen/Qwen2.5-Omni-7B --max-new-tokens 1 \
  --query-event-list results/beans_dogs_AJ_factorized_validation_split.json \
  --query-event-key confirmation \
  --factorized-ranks "$selected_rank" \
  --methods probe_class_pooled "$selected_method" probe_class_tokenwise \
  --method-batch-size 3 --relative-alphas .003 .01 .03 \
  --output results/beans_dogs_AJ_factorized_kv_lp1_confirmation.csv --resume

.venv/bin/python scripts/summarize_factorized_kv.py \
  --predictions results/beans_dogs_AJ_factorized_kv_lp1_confirmation.csv \
  --expected-n 109 --output results/beans_dogs_AJ_factorized_kv_lp1_confirmation_summary.json
