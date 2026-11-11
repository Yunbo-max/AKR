"""Explicit acquisition and integrity checks for the three pinned input assets.

Acquisition is a user/Local execution step. Importing this module is offline and
does not import the model stack. Manifests distinguish upstream hashes from local
identity hashes; successful download is not a model/evaluator qualification.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path

from .config import MODEL_ID, MODEL_REVISION, GSM8K_REVISION, MATH500_REVISION

DATASETS = {
    "gsm8k": {
        "repo_id": "openai/gsm8k", "revision": GSM8K_REVISION,
        "files": ["main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet"],
    },
    "MATH-500": {
        "repo_id": "HuggingFaceH4/MATH-500", "revision": MATH500_REVISION,
        "files": ["test.jsonl"],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def exclusive_lock(path: Path):
    """Kernel lock, automatically released even after a killed process (Linux)."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}; do not run the same case/cache concurrently") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _file_record(path: Path, sibling) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty downloaded asset: {path}")
    with path.open("rb") as handle:
        if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
            raise RuntimeError(f"Git LFS pointer is not a downloaded asset: {path}")
    digest = sha256_file(path)
    record = {"sha256": digest, "size_bytes": path.stat().st_size}
    lfs = getattr(sibling, "lfs", None)
    upstream = getattr(lfs, "sha256", None) if lfs else None
    if isinstance(lfs, dict):
        upstream = lfs.get("sha256")
    if upstream:
        if upstream != digest:
            raise RuntimeError(f"Upstream LFS SHA-256 mismatch: {path}")
        record["upstream_lfs_sha256"] = upstream
    else:
        upstream_blob = getattr(sibling, "blob_id", None)
        if upstream_blob:
            blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    blob.update(chunk)
            if blob.hexdigest() != upstream_blob:
                raise RuntimeError(f"Upstream Git blob hash mismatch: {path}")
            record["upstream_git_blob_sha1"] = upstream_blob
    return record


def download_assets(root: str | Path, *, max_workers: int = 4) -> dict:
    """Download complete official checkpoint plus released dataset files, resumably."""
    from huggingface_hub import HfApi, snapshot_download
    root = Path(root).expanduser().resolve()
    api = HfApi()
    with exclusive_lock(root / ".acquisition.lock"):
        model_dir = root / "models" / "Qwen3.5-4B"
        info = api.model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
        if info.sha != MODEL_REVISION:
            raise RuntimeError("Model server returned a different immutable revision")
        # A full snapshot retains every referenced shard, tokenizer/template and license.
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(model_dir),
                          max_workers=max_workers)
        model_manifest = {
            "repo_id": MODEL_ID, "revision": MODEL_REVISION,
            "files": {s.rfilename: _file_record(model_dir / s.rfilename, s) for s in info.siblings},
        }
        atomic_json(model_dir / "model_manifest.json", model_manifest)
        dataset_manifest = {"datasets": {}}
        for name, spec in DATASETS.items():
            destination = root / "data" / name
            info = api.dataset_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
            if info.sha != spec["revision"]:
                raise RuntimeError(f"Dataset server returned a different revision for {name}")
            siblings = {item.rfilename: item for item in info.siblings}
            if set(spec["files"]) - set(siblings):
                raise RuntimeError(f"Pinned dataset is missing declared files: {name}")
            snapshot_download(spec["repo_id"], repo_type="dataset", revision=spec["revision"],
                              local_dir=str(destination), allow_patterns=spec["files"],
                              max_workers=max_workers)
            dataset_manifest["datasets"][name] = {
                "repo_id": spec["repo_id"], "revision": spec["revision"],
                "files": {file: _file_record(destination / file, siblings[file]) for file in spec["files"]},
            }
        atomic_json(root / "data" / "dataset_manifest.json", dataset_manifest)
        receipt = {
            "status": "assets_downloaded_and_hash_checked",
            "model_manifest": str(model_dir / "model_manifest.json"),
            "model_manifest_sha256": sha256_file(model_dir / "model_manifest.json"),
            "dataset_manifest": str(root / "data" / "dataset_manifest.json"),
            "dataset_manifest_sha256": sha256_file(root / "data" / "dataset_manifest.json"),
            "native_model_qualification": "pending",
            "native_evaluator_qualification": "pending",
        }
        atomic_json(root / "asset_manifest.json", receipt)
        return receipt


def _verify_manifest(directory: Path, manifest: dict, *, full_hashes: bool) -> None:
    for name, entry in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError(f"Invalid path in asset manifest: {name}")
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Missing/truncated/changed asset: {path}")
        # Small metadata is cheap to validate on every invocation; full weights are optional.
        if full_hashes or path.stat().st_size <= 16 * 1024 * 1024:
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Asset hash mismatch: {path}")


def check_assets(root: str | Path, *, full_hashes: bool = False) -> dict:
    root = Path(root).expanduser().resolve()
    model_dir = root / "models" / "Qwen3.5-4B"
    model = json.loads((model_dir / "model_manifest.json").read_text())
    if model["repo_id"] != MODEL_ID or model["revision"] != MODEL_REVISION:
        raise ValueError("Model manifest has the wrong repository or revision")
    _verify_manifest(model_dir, model, full_hashes=full_hashes)
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        if required not in model["files"]:
            raise ValueError(f"Model manifest omits required file {required}")
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    mtp = {key: value for key, value in weight_map.items() if key.startswith("mtp.")}
    if len(mtp) != 15:
        raise ValueError(f"Expected 15 released native MTP tensors, found {len(mtp)}")
    if set(weight_map.values()) - set(model["files"]):
        raise ValueError("Not every model shard is covered by the complete download manifest")
    data = json.loads((root / "data" / "dataset_manifest.json").read_text())
    for name, spec in DATASETS.items():
        actual = data["datasets"][name]
        if actual["repo_id"] != spec["repo_id"] or actual["revision"] != spec["revision"]:
            raise ValueError(f"Dataset identity mismatch: {name}")
        if set(spec["files"]) - set(actual["files"]):
            raise ValueError(f"Dataset manifest omits native split files: {name}")
        _verify_manifest(root / "data" / name, actual, full_hashes=True)
    return {
        "status": "local_asset_identity_checked", "full_model_hashes": full_hashes,
        "model_revision": MODEL_REVISION, "native_mtp_tensor_count": len(mtp),
        "datasets": {name: spec["revision"] for name, spec in DATASETS.items()},
        "native_gpu_qualification": "not_implied_by_asset_check",
    }
