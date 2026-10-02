#!/usr/bin/env python3
"""Train a resumable q/v-projection Thinker LoRA on a fixed train split."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model

from animal_omni.qwen_runner import QwenThinkerRunner


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every % args.accumulation:
        raise ValueError("--checkpoint-every must be divisible by --accumulation")

    random.seed(20250813)
    np.random.seed(20250813)
    torch.manual_seed(20250813)
    config = yaml.safe_load(args.config.read_text())
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    train = [row for row in rows if row["split"] == "train"]
    valid = [row for row in rows if row["split"] == "valid"]
    random.shuffle(train)
    if args.limit_train:
        train = train[:args.limit_train]
    if args.limit_valid:
        valid = valid[:args.limit_valid]

    state_path = args.output_dir / "checkpoint_state.json"
    state = json.loads(state_path.read_text()) if args.resume and state_path.exists() else None
    history_path = args.output_dir / "history.json"
    existing_history = json.loads(history_path.read_text()) if history_path.exists() else []
    if (args.resume and state is None and len(existing_history) >= args.epochs
            and (args.output_dir / f"epoch_{args.epochs}").is_dir()):
        print(f"training already complete through epoch {args.epochs}", flush=True)
        return
    if state and state["train_event_ids"] != [row["event_id"] for row in train]:
        raise RuntimeError("resume train order differs from saved checkpoint")
    if state and state["epochs"] != args.epochs:
        raise RuntimeError("resume epoch count differs from saved checkpoint")

    runner = QwenThinkerRunner(args.model_id)
    logging.disable(logging.CRITICAL)
    runner.model.thinker.enable_input_require_grads()
    runner.model.thinker.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if state:
        runner.model.thinker = PeftModel.from_pretrained(
            runner.model.thinker, state["adapter_dir"], is_trainable=True
        )
    else:
        lora = LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        runner.model.thinker = get_peft_model(runner.model.thinker, lora)
    runner.model.thinker.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in runner.model.thinker.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    if state:
        optimizer.load_state_dict(torch.load(state["optimizer_path"], map_location="cpu",
                                             weights_only=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = existing_history
    start_epoch = int(state["epoch"]) if state else 1
    completed = int(state["completed"]) if state else 0
    losses = [float(value) for value in state.get("losses", [])] if state else []

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch != start_epoch:
            completed = 0
            losses = []
        runner.model.thinker.train()
        optimizer.zero_grad(set_to_none=True)
        for index in range(completed + 1, len(train) + 1):
            row = train[index - 1]
            inputs = runner.teacher_forced_inputs(
                row["audio_path"], config["evaluation"]["prompt"], row["label"]
            )
            loss = runner.model.thinker(
                **inputs, use_cache=False, return_dict=True
            ).loss / args.accumulation
            loss.backward()
            losses.append(float(loss.detach()) * args.accumulation)
            if index % args.accumulation == 0 or index == len(train):
                torch.nn.utils.clip_grad_norm_(runner.model.thinker.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if index == 1 or index % 25 == 0 or index == len(train):
                print(
                    f"epoch={epoch} [{index}/{len(train)}] "
                    f"loss={np.mean(losses[-25:]):.4f}",
                    flush=True,
                )
            if index % args.checkpoint_every == 0 and index < len(train):
                checkpoint_dir = args.output_dir / f"checkpoint_{epoch}_{index:06d}"
                optimizer_path = args.output_dir / f"optimizer_{epoch}_{index:06d}.pt"
                runner.model.thinker.save_pretrained(checkpoint_dir)
                torch.save(optimizer.state_dict(), optimizer_path)
                atomic_json(state_path, {
                    "epoch": epoch,
                    "completed": index,
                    "epochs": args.epochs,
                    "adapter_dir": str(checkpoint_dir),
                    "optimizer_path": str(optimizer_path),
                    "losses": losses,
                    "train_event_ids": [row["event_id"] for row in train],
                    "protocol": "checkpoint only after optimizer step",
                })

        runner.model.thinker.eval()
        validation_losses = []
        with torch.inference_mode():
            for row in valid:
                inputs = runner.teacher_forced_inputs(
                    row["audio_path"], config["evaluation"]["prompt"], row["label"]
                )
                validation_losses.append(float(runner.model.thinker(
                    **inputs, use_cache=False, return_dict=True
                ).loss))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "valid_loss": float(np.mean(validation_losses)),
            "n_train": len(train),
            "n_valid_monitor": len(valid),
        }
        history.append(record)
        print(record, flush=True)
        runner.model.thinker.save_pretrained(args.output_dir / f"epoch_{epoch}")
        atomic_json(history_path, history)
        state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
