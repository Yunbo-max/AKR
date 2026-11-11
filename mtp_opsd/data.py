"""Pinned native datasets; training exposes questions and identities only.

Gold answers are never returned by ``load_training_examples``.  The evaluator
loads the released test splits separately through the upstream task builders.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import GSM8K_REVISION, MATH500_REVISION


DATASETS = {
    "gsm8k": {
        "repo_id": "openai/gsm8k",
        "revision": GSM8K_REVISION,
        "configuration": "main",
        "files": {
            "train": "main/train-00000-of-00001.parquet",
            "test": "main/test-00000-of-00001.parquet",
        },
        "counts": {"train": 7473, "test": 1319},
        "builder": "parquet",
    },
    "MATH-500": {
        "repo_id": "HuggingFaceH4/MATH-500",
        "revision": MATH500_REVISION,
        "configuration": "default",
        "files": {"test": "test.jsonl"},
        "counts": {"test": 500},
        "builder": "json",
    },
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_load_spec(name: str, *, local_root: str | Path | None = None,
                      cache_dir: str | None = None) -> dict[str, Any]:
    """Return source-pinned ``datasets.load_dataset`` keyword arguments.

    A local root must have the acquisition manifest.  Missing/corrupt local
    files never fall back to a moving remote revision.
    """
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")
    info = DATASETS[name]
    if local_root is None:
        return {
            "path": info["repo_id"], "name": info["configuration"],
            "revision": info["revision"], "cache_dir": cache_dir,
        }

    root = Path(local_root).expanduser().resolve()
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run the pinned acquisition stage first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("datasets", {}).get(name)
    if not isinstance(record, dict):
        raise ValueError(f"Dataset {name} is missing from {manifest_path}")
    if (record.get("repo_id"), record.get("revision")) != (
        info["repo_id"], info["revision"]
    ):
        raise ValueError(f"Dataset provenance mismatch for {name}")
    data_files = {}
    for split, relative in info["files"].items():
        path = root / name / relative
        expected = record.get("files", {}).get(relative, {}).get("sha256")
        if not path.is_file() or not expected or file_digest(path) != expected:
            raise ValueError(f"Missing or modified pinned dataset file: {path}")
        data_files[split] = str(path)
    return {
        "path": info["builder"], "name": None,
        "data_files": data_files, "cache_dir": cache_dir,
    }


def load_training_examples(config: dict) -> list[dict[str, Any]]:
    """Load all 7,473 released GSM8K train questions, in native source order.

    The trainer owns seeded shuffling and its declared prompt/update budget.
    No test examples, reference solutions, answers, or reward labels leave this
    function, including when a local dataset snapshot is used.
    """
    from datasets import load_dataset

    data = config["data"]
    expected = ("openai/gsm8k", GSM8K_REVISION, "main", "train")
    actual = (
        data.get("train_dataset"), data.get("train_revision"),
        data.get("train_config"), data.get("train_split"),
    )
    if actual != expected:
        raise ValueError(f"Training dataset must be pinned GSM8K main/train: {actual}")
    spec = dataset_load_spec(
        "gsm8k", local_root=data.get("local_root"), cache_dir=data.get("cache_dir")
    )
    dataset = load_dataset(**spec, split="train")
    if len(dataset) != DATASETS["gsm8k"]["counts"]["train"]:
        raise ValueError(f"Incomplete GSM8K training split: {len(dataset)} != 7473")
    # Remove answer-bearing columns before iterating or constructing examples.
    dataset = dataset.select_columns(["question"])
    examples = []
    for index, row in enumerate(dataset):
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid native GSM8K training question at index {index}")
        examples.append({
            "id": f"gsm8k/main/train/{index:05d}",
            "source_index": index,
            "question": question,
        })
    return examples
