---
base_model: Qwen/Qwen2.5-Omni-7B
library_name: peft
pipeline_tag: audio-classification
license: other
license_name: research-artifact
tags:
- qwen2.5-omni
- audio
- bioacoustics
- dogs
- peft
- lora
- negative-result
---

# Qwen2.5-Omni-7B Dogs K=2 Thinker LoRA — negative result

This 15.4 MB adapter is released for reproducibility of a matched-supervision
negative baseline. It is **not a useful dog-identity classifier**: after
training, it predicted `Rudy` for every one of the 139 registered validation
examples, yielding **2.88% accuracy** and **0.56% macro-F1**.

## Training

- Base: `Qwen/Qwen2.5-Omni-7B`, Talker disabled.
- Data: exactly 20 registered training examples, K=2 per each of ten dog
  identities, at the 0–1 kHz low-pass condition.
- Validation: all 139 official validation examples; test was not touched.
- One epoch, BF16, learning rate `2e-4`, accumulation 8, seed `20250813`.
- PEFT LoRA: rank 8, alpha 16, dropout 0.05, `q_proj` and `v_proj`, no bias.
- Final train loss 1.8567; validation loss 2.0280.
- Hardware: one NVIDIA RTX 3090 (24 GiB).

## Usage

```bash
git clone https://github.com/Yunbo-max/animal-omni-kv.git
cd animal-omni-kv
python3 -m venv .venv
.venv/bin/pip install -e '.[qwen,adaptation]'

.venv/bin/python scripts/predict_thinker_lora.py \
  --config configs/beans_dogs.yaml \
  --adapter humanlong/qwen2.5-omni-7b-dogs-k2-lora \
  --audio /path/to/dog.wav
```

The adapter must be attached to `model.thinker`, not loaded as a standalone
text model. See `MODEL_USAGE.md` in the GitHub repository for the full command
and evaluation pathway.

## Intended use and limitations

Use this artifact only to reproduce the paper's equal-support LoRA comparison,
study optimization collapse, or test alternative training protocols. Do not use
it for animal identification or monitoring. The source dataset does not provide
a clear redistribution license in its local card, so no audio is included and
users must obtain it under the original terms.
