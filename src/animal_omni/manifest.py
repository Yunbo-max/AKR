from __future__ import annotations

import csv
import hashlib
from pathlib import Path


REQUIRED = {"event_id", "audio_path", "label", "recording_id"}


def read_manifest(path: str | Path, labels: list[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = REQUIRED - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    invalid = sorted({r["label"] for r in rows} - set(labels))
    if invalid:
        raise ValueError(f"unknown labels: {invalid}")
    return rows


def deterministic_split(group: str, seed: int, train_fraction: float, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < train_fraction:
        return "train"
    if u < train_fraction + validation_fraction:
        return "validation"
    return "test"

