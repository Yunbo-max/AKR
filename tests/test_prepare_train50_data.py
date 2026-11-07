import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.prepare_train50_data import (SOURCES, discover_shards, validate_record,
    _write_collision_safe, _verify_snapshot, prepare, main)


def record(index=1, label="dog", name="original.wav", blob=b"audio"):
    return {"Unnamed: 0": index, "label": label, "path": {"path": name, "bytes": blob}}


def test_record_preserves_real_source_and_numeric_labels(tmp_path):
    row, blob = validate_record(record(label=3, name="parent/original.wav"), "bats", "test",
                                "a"*40, "data/test.parquet", tmp_path, set())
    assert row["source_filename"] == row["source_group"] == "original.wav"
    assert row["source_path"] == "parent/original.wav"
    assert row["label"] == "3" and row["split"] == "test"
    assert row["source_group"] != row["event_id"]
    assert row["sha256"] == hashlib.sha256(blob).hexdigest()
    assert "parent_recording_unknown" in row["group_provenance"]


@pytest.mark.parametrize("bad", [{}, record(index=None), record(index=True), record(label=[]),
                               record(name=""), record(blob=b""), record(name="source.dat")])
def test_malformed_records_fail(bad, tmp_path):
    with pytest.raises(ValueError):
        validate_record(bad, "dogs", "train", "a"*40, "x", tmp_path, set())


def test_duplicate_ids_and_unknown_split_fail(tmp_path):
    seen = set()
    validate_record(record(), "dogs", "train", "a"*40, "x", tmp_path, seen)
    with pytest.raises(ValueError, match="duplicate"):
        validate_record(record(index="01"), "dogs", "train", "a"*40, "y", tmp_path, seen)
    with pytest.raises(ValueError, match="official split"):
        validate_record(record(), "dogs", "validation", "a"*40, "x", tmp_path, set())


def shards(tmp_path, dataset="dogs"):
    directory = tmp_path / "data"
    directory.mkdir()
    for split, count in SOURCES[dataset]["shards"].items():
        for i in range(count):
            (directory / f"{split}-{i:05d}-of-{count:05d}.parquet").touch()
    return directory


def test_shards_must_be_complete_and_known(tmp_path):
    directory = shards(tmp_path)
    assert len(discover_shards(tmp_path, "dogs")) == 4
    (directory / "training-00000-of-00001.parquet").touch()
    with pytest.raises(ValueError, match="unknown split"):
        discover_shards(tmp_path, "dogs")
    (directory / "training-00000-of-00001.parquet").unlink()
    (directory / "train-00001-of-00002.parquet").unlink()
    with pytest.raises(ValueError, match="incomplete train"):
        discover_shards(tmp_path, "dogs")


def test_output_collision_is_not_overwritten(tmp_path):
    p = tmp_path / "example.wav"
    _write_collision_safe(p, b"original")
    _write_collision_safe(p, b"original")
    with pytest.raises(ValueError, match="collision"):
        _write_collision_safe(p, b"different")
    assert p.read_bytes() == b"original"


def test_local_snapshot_must_encode_exact_source_revision(tmp_path):
    sha = SOURCES["dogs"]["revision"]
    good = tmp_path / "datasets--DBD-research-group--beans_dogs" / "snapshots" / sha
    assert _verify_snapshot(good, "dogs", sha) == "huggingface_cache_snapshot_path"
    with pytest.raises(ValueError, match="different repository"):
        _verify_snapshot(good, "watkins", sha)
    with pytest.raises(ValueError, match="pinned HF"):
        _verify_snapshot(tmp_path, "dogs", sha)


def test_branch_revision_is_forbidden_without_network():
    with pytest.raises(SystemExit):
        main(["--dataset", "dogs", "--revision", "main"])


def test_real_parquet_materialization_retains_whole_test(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    dataset = "dogs"
    revision = SOURCES[dataset]["revision"]
    snap = tmp_path / "datasets--DBD-research-group--beans_dogs" / "snapshots" / revision
    directory = snap / "data"
    directory.mkdir(parents=True)
    wav = io.BytesIO()
    sf.write(wav, np.linspace(-0.1, 0.1, 400), 16000, format="WAV")
    blob = wav.getvalue()
    for split, total in SOURCES[dataset]["counts"].items():
        rows = [record(index=i, label=str(i % 10), name=f"recording_{i}.wav", blob=blob) for i in range(total)]
        n_shards = SOURCES[dataset]["shards"][split]
        for i, chunk in enumerate(np.array_split(np.array(rows, dtype=object), n_shards)):
            pq.write_table(pa.Table.from_pylist(list(chunk)), directory / f"{split}-{i:05d}-of-{n_shards:05d}.parquet")
    output, audio = tmp_path / "manifests", tmp_path / "audio"
    receipt = prepare(dataset, snap, revision, output, audio)
    manifest_rows = list(csv.DictReader((output / "dogs.csv").open()))
    assert len([r for r in manifest_rows if r["split"] == "test"]) == 139
    assert len(manifest_rows) == 693
    assert receipt["support_selection"] == "none" and receipt["complete"] is True
    assert receipt["manifest_sha256"] == hashlib.sha256((output / "dogs.csv").read_bytes()).hexdigest()
    assert all(Path(r["audio_path"]).read_bytes() == blob for r in manifest_rows)
    assert prepare(dataset, snap, revision, output, audio) == receipt
