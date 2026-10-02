---
base_model: Qwen/Qwen2.5-Omni-7B
library_name: peft
pipeline_tag: audio-classification
license: other
license_name: research-use-only
tags:
- qwen2.5-omni
- audio
- bioacoustics
- marine-mammals
- peft
- lora
---

# Qwen2.5-Omni-7B Watkins Thinker LoRA

This is a 15.4 MB PEFT LoRA adapter for the Thinker of
`Qwen/Qwen2.5-Omni-7B`. It predicts one of 31 marine-mammal species labels from
audio under the fixed BEANS Watkins protocol. It is a research artifact, not a
general-purpose species classifier.

## Evaluation

On the untouched 339-example official test split, deterministic generation
reached **31.56% accuracy** and **23.44% macro-F1**, with 8 invalid outputs. The
frozen linear probe reached 88.20%, so this adapter does not close the measured
support-to-decision gap.

## Training

- Base: `Qwen/Qwen2.5-Omni-7B`, Talker disabled.
- Data: 1,017 BEANS Watkins training examples; 64 fixed validation examples
  monitored without test-label selection.
- One epoch, BF16, learning rate `2e-4`, gradient accumulation 8, seed
  `20250813`.
- PEFT LoRA: rank 8, alpha 16, dropout 0.05, `q_proj` and `v_proj`, no bias.
- Final train loss 0.8018; validation loss 0.4663.
- Hardware: one NVIDIA RTX 3090 (24 GiB).

Watkins source audio is free for personal and academic use. Users must follow
the original dataset terms; this repository contains adapter weights only.

## Usage

Install the experiment repository and its pinned Qwen dependencies:

```bash
git clone https://github.com/Yunbo-max/animal-omni-kv.git
cd animal-omni-kv
python3 -m venv .venv
.venv/bin/pip install -e '.[qwen,adaptation]'
```

Run one WAV file:

```bash
.venv/bin/python scripts/predict_thinker_lora.py \
  --config configs/beans_watkins.yaml \
  --adapter humanlong/qwen2.5-omni-7b-watkins-lora \
  --audio /path/to/marine_mammal.wav
```

The implementation first loads the full Omni model and then attaches this
adapter specifically to `model.thinker` using
`PeftModel.from_pretrained(model.thinker, adapter_id)`. Loading it as a generic
text-only causal LM is incorrect.

## Limitations

The label set, prompt, preprocessing, and fixed split are task-specific. The
model can inherit errors and biases from Qwen and the source recordings. Do not
use it for ecological population estimates, safety-critical monitoring, or
commercial identification.

Full protocols and results are in the associated GitHub repository,
particularly `RESULTS.md`, `REPRODUCE.md`, and `MODEL_USAGE.md`.
