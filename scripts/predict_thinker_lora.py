#!/usr/bin/env python3
"""Run one deterministic audio prediction with a Thinker LoRA adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from peft import PeftModel

from animal_omni.metrics import normalize_label
from animal_omni.qwen_runner import QwenThinkerRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--adapter", required=True,
                        help="local adapter directory or Hugging Face repo ID")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    labels = config["dataset"]["labels"]
    runner = QwenThinkerRunner(args.model_id)
    runner.model.thinker = PeftModel.from_pretrained(
        runner.model.thinker, args.adapter
    )
    runner.model.thinker.eval()
    raw = runner.predict(
        args.audio,
        config["evaluation"]["prompt"],
        max_new_tokens=(args.max_new_tokens or
                        config["evaluation"]["max_new_tokens"]),
    )
    prediction = normalize_label(raw, labels)
    print(json.dumps({
        "audio": str(args.audio),
        "adapter": args.adapter,
        "raw_prediction": raw,
        "prediction": prediction,
        "valid_label": prediction is not None,
    }, indent=2))


if __name__ == "__main__":
    main()
