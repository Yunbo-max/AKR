"""The registered BEANS-Zero scan can read a pinned local parquet snapshot."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def test_materialize_beans_zero_from_local_parquet(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "sample-1",
                    "dataset_name": "call-type",
                    "audio": [0.1] * 1600,
                    "metadata": json.dumps({"sample_rate": 16000}),
                    "output": "Phee",
                    "instruction_text": "Which call?",
                    "source_dataset": "test-source",
                    "license": "CC0",
                    "file_name": "sample.wav",
                }
            ]
        ),
        source / "shard-000.parquet",
    )
    output = tmp_path / "audio"
    manifest = tmp_path / "manifest.csv"
    script = Path(__file__).resolve().parents[1] / "scripts/materialize_beans_zero_subset.py"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-dir", str(source),
            "--output-dir", str(output),
            "--output-manifest", str(manifest),
            "--per-task", "0",
            "--target-rate", "16000",
            "--duration-cap", "10",
            "--minimum-duration", "0.1",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stderr
    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 1
    assert rows[0]["event_id"] == "sample-1"
    assert rows[0]["label"] == "Phee"
    assert (output / "call-type" / "sample-1.wav").is_file()


def test_materialize_only_registered_target_ids(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    samples = []
    for event_id in ("wanted", "extra"):
        samples.append({
            "id": event_id, "dataset_name": "call-type", "audio": [0.1] * 1600,
            "metadata": json.dumps({"sample_rate": 16000}), "output": "Phee",
            "instruction_text": "Which call?", "source_dataset": "test-source",
            "license": "CC0", "file_name": f"{event_id}.wav",
        })
    pq.write_table(pa.Table.from_pylist(samples), source / "shard-000.parquet")
    target = tmp_path / "target.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["event_id"])
        writer.writeheader()
        writer.writerow({"event_id": "wanted"})
    output = tmp_path / "audio"
    manifest = tmp_path / "manifest.csv"
    script = Path(__file__).resolve().parents[1] / "scripts/materialize_beans_zero_subset.py"
    run = subprocess.run(
        [sys.executable, str(script), "--source-dir", str(source),
         "--target-manifest", str(target), "--output-dir", str(output),
         "--output-manifest", str(manifest), "--per-task", "0"],
        capture_output=True, text=True, timeout=30,
    )
    assert run.returncode == 0, run.stderr
    assert [row["event_id"] for row in csv.DictReader(manifest.open())] == ["wanted"]
    assert (output / "call-type" / "wanted.wav").is_file()
    assert not (output / "call-type" / "extra.wav").exists()
