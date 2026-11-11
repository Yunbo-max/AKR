"""Source-pinned, separately addressable experimental cases (stdlib only)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "mtp_opsd"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
HARNESS_REVISION = "d6de81643928d653435c431bae19945d41d32520"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def available_cases() -> list[str]:
    return sorted(p.stem for p in CONFIG_ROOT.glob("c[0-9][0-9]_*.json"))


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def load_case(case_id: str, *, assets_root: str | Path | None = None,
              seed: int | None = None, overrides: dict | None = None) -> dict[str, Any]:
    if case_id not in available_cases():
        raise ValueError(f"Unknown case {case_id!r}; choose {available_cases()}")
    common = json.loads((CONFIG_ROOT / "common.json").read_text())
    case = json.loads((CONFIG_ROOT / f"{case_id}.json").read_text())
    config = merge(common, case)
    if overrides:
        config = merge(config, overrides)
    if config.get("case_id") != case_id or config.get("kind") != case["kind"]:
        raise ValueError("Overrides cannot rename a case or change its native/trained identity")
    if seed is not None:
        config["training"]["seed"] = seed
        config["sampling"]["seed"] = seed
        config["evaluation"]["seed"] = seed + 1
    if assets_root is not None:
        root = Path(assets_root).expanduser().resolve()
        config["model"]["path"] = str(root / "models" / "Qwen3.5-4B")
        config["data"]["local_root"] = str(root / "data")
    validate(config)
    return config


def validate(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema_version")
    if config.get("kind") not in {"native", "train"}:
        raise ValueError("kind must be native or train")
    model = config["model"]
    if model["id"] != MODEL_ID or model["revision"] != MODEL_REVISION:
        raise ValueError("This backend is qualified in source only for the pinned Qwen3.5-4B")
    if model["lora"]["dropout"] != 0:
        raise ValueError("Teacher/student dropout must be zero for this baseline")
    if bool(model["lora"]["enabled"]) != (config["kind"] == "train"):
        raise ValueError("Training cases require LoRA; native cases must disable it")
    sampling = config["sampling"]
    mode = sampling["mode"]
    if mode not in {"ntp", "mtp"}:
        raise ValueError("sampling.mode must be ntp or mtp")
    if not isinstance(sampling["group_size"], int) or sampling["group_size"] < 1:
        raise ValueError("sampling.group_size must be a positive integer")
    if config["kind"] == "train" and sampling["group_size"] < 2:
        raise ValueError("Majority-reference disagreeing-trajectory OPSD needs group_size >= 2")
    if not 0 < sampling["top_p"] <= 1 or not math.isfinite(sampling["temperature"]) or sampling["temperature"] <= 0:
        raise ValueError("Use positive sampling temperature and 0 < top_p <= 1")
    if sampling.get("top_k", 0) != 0:
        raise ValueError("This sampling implementation requires top_k=0")
    train = config["training"]
    if train["supervision"] != "all_prefixes":
        raise ValueError("This transfer baseline implements all-prefix per-horizon FKL")
    expected = [1.0, 1.0] if mode == "mtp" else [1.0, 0.0]
    if train["head_weights"] != expected:
        raise ValueError(f"Expected head_weights={expected} for {mode}")
    for key in ("max_prompts", "max_updates", "logit_chunk_size", "save_every_updates"):
        if train[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if not math.isfinite(train["max_hours"]) or train["max_hours"] <= 0:
        raise ValueError("An explicit finite training time budget is required")
    select = config["selection"]
    if not 0.5 <= select["threshold"] <= 1.0:
        raise ValueError("selection.threshold must be between 0.5 and 1")
    if select["reference"] != "longest" or select["distill"] != "disagreeing":
        raise ValueError("This baseline uses longest agreeing reference and disagreeing targets")
    if config["data"]["train_revision"] != GSM8K_REVISION:
        raise ValueError("GSM8K training revision mismatch")
    if set(config["evaluation"]["generation_modes"]) - {"ntp", "mtp"}:
        raise ValueError("Invalid evaluation generation mode")


def run_directory(outputs_root: str | Path, run_id: str, case_id: str, seed: int) -> Path:
    if not SAFE_NAME.fullmatch(run_id):
        raise ValueError("run-id must contain only letters, numbers, dot, dash, underscore")
    if case_id not in available_cases():
        raise ValueError("Invalid case ID")
    return Path(outputs_root).expanduser().resolve() / run_id / case_id / f"seed_{seed}"
