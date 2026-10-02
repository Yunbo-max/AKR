# Using the trained adapters

The release contains two PEFT LoRA adapters for the Thinker submodule of
`Qwen/Qwen2.5-Omni-7B`. The base Qwen weights are not duplicated. Both adapters
target every Thinker `q_proj` and `v_proj` with rank 8, alpha 16, and dropout
0.05.

## Installation

```bash
git clone https://github.com/Yunbo-max/animal-omni-kv.git
cd animal-omni-kv
python3 -m venv .venv
.venv/bin/pip install -e '.[qwen,adaptation]'
```

Qwen2.5-Omni-7B Thinker-only inference needs roughly 22 GiB of GPU memory in
BF16 with the pinned environment used here.

## One audio file

Watkins marine-mammal species adapter:

```bash
.venv/bin/python scripts/predict_thinker_lora.py \
  --config configs/beans_watkins.yaml \
  --adapter humanlong/qwen2.5-omni-7b-watkins-lora \
  --audio /path/to/example.wav
```

Dogs equal-support negative-control adapter:

```bash
.venv/bin/python scripts/predict_thinker_lora.py \
  --config configs/beans_dogs.yaml \
  --adapter humanlong/qwen2.5-omni-7b-dogs-k2-lora \
  --audio /path/to/example.wav
```

The command prints both the raw continuation and its parsed registered label.
It disables Qwen's Talker and uses greedy text generation, matching evaluation.

## Full registered evaluation

```bash
.venv/bin/python scripts/evaluate_thinker_lora.py \
  --config configs/beans_watkins.yaml \
  --manifest data/manifests/beans_watkins_protocol.csv \
  --model-id Qwen/Qwen2.5-Omni-7B \
  --adapter /path/to/downloaded/adapter \
  --split test \
  --output results/reproduced_watkins_lora_test.csv
```

The manifest records the fixed split and provenance but does not redistribute
third-party audio. Materialize or obtain audio under the original dataset terms
before running the command.

## Important limitations

- Watkins was trained for one epoch on its 1,017-example official training
  split and reached 31.56% accuracy on 339 test examples. It is an experimental
  task adapter, not a general marine-mammal classifier.
- Dogs was trained on only 20 examples (K=2 per identity) at the registered
  1 kHz condition. It collapsed to `Rudy` and reached 2.88% validation accuracy.
  It is released solely for reproduction of the matched-supervision negative
  baseline and should not be presented as a useful classifier.
- Dataset source terms apply independently of the Qwen Apache-2.0 base-model
  license. Watkins audio is restricted to personal and academic use.

