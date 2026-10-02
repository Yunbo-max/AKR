#!/usr/bin/env bash
set -euo pipefail

cd /root/animal-omni-kv
export HF_HOME=/root/animal-omni-kv/.hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for n in 40 80 120 200 400 415; do
  if [[ "$n" == 415 ]]; then
    adapter=results/lora_beans_dogs_7b_full/epoch_1
  else
    adapter=$(printf 'results/lora_beans_dogs_7b_full/checkpoint_1_%06d' "$n")
  fi
  output=$(printf 'results/beans_dogs_lora_full_n%03d_7b_valid.csv' "$n")
  .venv/bin/python scripts/evaluate_thinker_lora.py \
    --config configs/beans_dogs.yaml \
    --manifest data/manifests/beans_dogs_protocol.csv \
    --model-id Qwen/Qwen2.5-Omni-7B \
    --adapter "$adapter" --split valid --output "$output" --resume
done

.venv/bin/python scripts/summarize_lora_data_scaling.py \
  --manifest data/manifests/beans_dogs_protocol.csv \
  --representation-dir results/reps_beans_dogs_7b \
  --prediction 40=results/beans_dogs_lora_full_n040_7b_valid.csv \
  --prediction 80=results/beans_dogs_lora_full_n080_7b_valid.csv \
  --prediction 120=results/beans_dogs_lora_full_n120_7b_valid.csv \
  --prediction 200=results/beans_dogs_lora_full_n200_7b_valid.csv \
  --prediction 400=results/beans_dogs_lora_full_n400_7b_valid.csv \
  --prediction 415=results/beans_dogs_lora_full_n415_7b_valid.csv \
  --output results/beans_dogs_lora_probe_data_scaling_7b_summary.json
