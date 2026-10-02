#!/usr/bin/env python3
"""Materialize embedded-audio BEANS parquet mirrors with fixed split provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-audio-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for parquet in sorted((args.dataset_dir / "data").glob("*.parquet")):
        split = parquet.name.split("-", 1)[0]
        # train_low duplicates a subset of train and is not a canonical split.
        if split == "train_low": continue
        table = pq.read_table(parquet, columns=["Unnamed: 0", "path", "label"])
        for record in table.to_pylist():
            audio = record["path"]["bytes"]
            source_name = Path(record["path"]["path"] or "audio.wav").name
            source_id = str(record["Unnamed: 0"])
            event_id = f"{split}_{source_id}"
            output = args.output_audio_dir / split / f"{event_id}.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists(): output.write_bytes(audio)
            rows.append({
                "event_id": event_id, "audio_path": str(output.resolve()),
                "label": record["label"], "split": split,
                "source_index": source_id, "source_filename": source_name,
                "sha256": hashlib.sha256(audio).hexdigest(),
                "provenance": f"hf:{args.dataset_dir.name}/{parquet.name}",
            })
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(f"materialized {len(rows)} rows; labels={len(set(r['label'] for r in rows))}")
    for split in sorted(set(r["split"] for r in rows)):
        print(split, sum(r["split"] == split for r in rows))


if __name__ == "__main__": main()
