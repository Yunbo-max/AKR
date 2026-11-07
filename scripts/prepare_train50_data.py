#!/usr/bin/env python3
"""Prepare complete official BEANS mirror splits, without selecting support.

Defaults are immutable HF revisions verified on 2026-11-07. No dataset is accessed
until this CLI is explicitly executed. Audio bytes are preserved, never selected
by predictions, resampled, truncated, or taken from BEANS-Zero.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from collections import Counter
from collections.abc import Mapping

SOURCES = {
    "dogs": dict(repo="DBD-research-group/beans_dogs", revision="b5928b67fcf4949cf782f3c4c64fecfc3c6be72f",
                 counts=dict(train=415, valid=139, test=139), shards=dict(train=2, valid=1, test=1)),
    "watkins": dict(repo="DBD-research-group/beans_watkins", revision="e57b7eb6e0868ceec03f75b31e6d7bac28dbd98a",
                    counts=dict(train=1017, valid=339, test=339), shards=dict(train=2, valid=1, test=1)),
    "bats": dict(repo="DBD-research-group/beans_bats", revision="fb9d6b65fa3f0f41721e1f0d089f6b5cc23099e3",
                 counts=dict(train=6000, valid=2000, test=2000), shards=dict(train=22, valid=8, test=8)),
    "humbugdb": dict(repo="DBD-research-group/beans_humbugdb", revision="3b1dc1e30c6f6e382749f1ca2ae5f993ec67a537",
                     counts=dict(train=5577, valid=1859, test=1859), shards=dict(train=7, valid=6, test=6)),
}
FIELDS = ["event_id", "audio_path", "label", "split", "source_index", "source_filename",
          "source_path", "source_group", "group_provenance", "sha256", "dataset", "source_repo",
          "source_revision", "source_shard", "provenance"]
SHARD = re.compile(r"(train_low|train|valid|test)-(\d{5})-of-(\d{5})\.parquet")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_shards(snapshot, dataset):
    """Reject incomplete shard populations and unknown split names."""
    if dataset not in SOURCES:
        raise ValueError(f"unknown official dataset: {dataset}")
    paths = sorted((Path(snapshot) / "data").glob("*.parquet"))
    populations = {}
    result = []
    for p in paths:
        match = SHARD.fullmatch(p.name)
        if match is None:
            raise ValueError(f"unknown split/shard filename: {p.name}")
        split, number, total = match.group(1), int(match.group(2)), int(match.group(3))
        populations.setdefault(split, []).append((number, total))
        if split != "train_low":
            result.append((split, p))
    for split, expected in SOURCES[dataset]["shards"].items():
        pairs = populations.get(split, [])
        if pairs != [(i, expected) for i in range(expected)]:
            raise ValueError(f"incomplete {split} shards: expected {expected}, found {pairs}")
    # train_low is a redundant low-resource subset, excluded as a split by design.
    if "train_low" in populations:
        pairs = populations["train_low"]
        expected = pairs[0][1]
        if expected <= 0 or pairs != [(i, expected) for i in range(expected)]:
            raise ValueError("incomplete train_low shards")
    return result


def validate_record(record, dataset, split, revision, shard, audio_dir, seen):
    """Normalize one real mirror row; no invented recording IDs or labels."""
    if dataset not in SOURCES or split not in {"train", "valid", "test"}:
        raise ValueError("unknown dataset or official split")
    if not isinstance(record, Mapping) or {"Unnamed: 0", "path", "label"} - set(record):
        raise ValueError("mirror record requires Unnamed: 0, path, label")
    index = record["Unnamed: 0"]
    if isinstance(index, bool) or not isinstance(index, (int, str)) or not re.fullmatch(r"\d+", str(index)):
        raise ValueError("source index must be a nonnegative integer")
    index = str(int(index))
    event_id = f"{dataset}_{split}_{index}"
    if event_id in seen:
        raise ValueError(f"duplicate event ID: {event_id}")
    seen.add(event_id)
    label = record["label"]
    if isinstance(label, bool) or not isinstance(label, (str, int)) or not str(label).strip():
        raise ValueError("label must be a nonempty official string or integer")
    audio = record["path"]
    if not isinstance(audio, Mapping) or not isinstance(audio.get("bytes"), (bytes, bytearray)) or not audio["bytes"]:
        raise ValueError("mirror path must contain nonempty embedded audio bytes")
    source_path = audio.get("path")
    if not isinstance(source_path, str) or not source_path.strip() or "\x00" in source_path:
        raise ValueError("real source audio filename is required; no event-ID fallback")
    source_path = source_path.replace("\\", "/")
    name = PurePosixPath(source_path).name
    suffix = PurePosixPath(name).suffix.lower()
    if name in {".", "..", ""} or suffix not in {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".au"}:
        raise ValueError(f"unsupported or missing source audio suffix: {source_path!r}")
    blob = bytes(audio["bytes"])
    target = Path(audio_dir).resolve() / split / (event_id + suffix)
    row = dict(event_id=event_id, audio_path=str(target), label=str(label), split=split,
               source_index=index, source_filename=name, source_path=source_path,
               source_group=name, group_provenance="source_filename_only_parent_recording_unknown",
               sha256=hashlib.sha256(blob).hexdigest(), dataset=dataset,
               source_repo=SOURCES[dataset]["repo"], source_revision=revision, source_shard=shard,
               provenance=f"hf://datasets/{SOURCES[dataset]['repo']}@{revision}/{shard}")
    return row, blob


def _write_collision_safe(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"existing output collision: {path}")
        return
    # Exclusive creation also prevents accidental concurrent replacement.
    with path.open("xb") as f:
        f.write(content)


def _verify_snapshot(snapshot, dataset, revision):
    snapshot = Path(snapshot).absolute()
    # A normal hf snapshot directory encodes its resolved immutable revision.
    if snapshot.name == revision and snapshot.parent.name == "snapshots":
        expected = "datasets--" + SOURCES[dataset]["repo"].replace("/", "--")
        if snapshot.parent.parent.name != expected:
            raise ValueError("HF snapshot cache belongs to a different repository")
        return "huggingface_cache_snapshot_path"
    try:
        head = subprocess.check_output(["git", "-C", str(snapshot), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
        origin = subprocess.check_output(["git", "-C", str(snapshot), "remote", "get-url", "origin"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("--snapshot-dir must be a pinned HF cache snapshot or official dataset Git checkout") from exc
    expected_url = "https://huggingface.co/datasets/" + SOURCES[dataset]["repo"]
    if head != revision or origin.removesuffix(".git").rstrip("/") != expected_url:
        raise ValueError("local dataset Git revision/repository differs from requested immutable source")
    dirty = subprocess.check_output(["git", "-C", str(snapshot), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ValueError("local dataset Git checkout is modified")
    return "official_dataset_clean_git_revision"


def prepare(dataset, snapshot, revision, output_dir=Path("data/manifests/train50"),
            audio_dir=None):
    if dataset not in SOURCES or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("known dataset and full immutable 40-character revision are required")
    verification = _verify_snapshot(snapshot, dataset, revision)
    shards = discover_shards(snapshot, dataset)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("prepare_train50_data requires pyarrow") from exc
    audio_dir = Path(audio_dir or Path("data/beans/train50") / dataset)
    output = Path(output_dir) / f"{dataset}.csv"
    receipt_path = output.with_suffix(".provenance.json")
    # Validate the entire population before making the manifest reviewable.
    rows, seen, counts, source_hashes = [], set(), Counter(), {}
    for split, path in shards:
        relative = str(path.relative_to(snapshot))
        source_hashes[relative] = sha256_file(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=16, columns=["Unnamed: 0", "path", "label"]):
            for record in batch.to_pylist():
                row, blob = validate_record(record, dataset, split, revision, relative, audio_dir, seen)
                _write_collision_safe(row["audio_path"], blob)
                rows.append(row)
                counts[split] += 1
    if dict(counts) != SOURCES[dataset]["counts"]:
        raise ValueError(f"official split population mismatch: {dict(counts)} != {SOURCES[dataset]['counts']}")
    rows.sort(key=lambda r: (r["split"], int(r["source_index"])))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    receipt = dict(schema_version=1, dataset=dataset, source_repo=SOURCES[dataset]["repo"],
                   source_revision=revision, source_verification=verification,
                   verified_default_revision=SOURCES[dataset]["revision"],
                   split_counts=dict(counts), expected_split_counts=SOURCES[dataset]["counts"],
                   source_parquet_sha256=source_hashes, manifest_sha256=hashlib.sha256(content).hexdigest(),
                   label_counts={s: dict(sorted(Counter(r["label"] for r in rows if r["split"] == s).items()))
                                 for s in ("train", "valid", "test")},
                   excluded_split="train_low (redundant subset of train)", support_selection="none",
                   audio_policy="unaltered embedded source bytes; downstream protocol decodes and checks overlap",
                   recording_group_limit="source filename only; parent recording identity not independently known",
                   preparation_code_sha256=sha256_file(__file__), complete=True)
    readme = Path(snapshot) / "README.md"
    if readme.is_file():
        receipt["source_readme_sha256"] = sha256_file(readme)
    encoded_receipt = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    # Check both collisions before publishing either file.
    for p, body in ((output, content), (receipt_path, encoded_receipt)):
        if p.exists() and p.read_bytes() != body:
            raise ValueError(f"existing output collision: {p}")
    _write_collision_safe(output, content)
    _write_collision_safe(receipt_path, encoded_receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--revision", help="full immutable HF commit (defaults to verified pinned revision)")
    parser.add_argument("--snapshot-dir", type=Path, help="pinned HF cache snapshot or clean official dataset Git checkout")
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/train50"))
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    revision = args.revision or SOURCES[args.dataset]["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("--revision must be a full immutable HF commit SHA; branches/tags are forbidden")
    snapshot = args.snapshot_dir
    if snapshot is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("install huggingface_hub or provide --snapshot-dir") from exc
        snapshot = Path(snapshot_download(repo_id=SOURCES[args.dataset]["repo"], repo_type="dataset",
                                          revision=revision, cache_dir=args.cache_dir,
                                          local_files_only=args.local_files_only,
                                          allow_patterns=["README.md", "data/train-*.parquet", "data/valid-*.parquet", "data/test-*.parquet"]))
    receipt = prepare(args.dataset, Path(snapshot), revision, args.output_dir, args.audio_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
