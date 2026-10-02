#!/usr/bin/env python3
"""Build a smoke-test manifest from the official MarmAudio Audio_Examples.zip.

This 60-example balanced subset is for pipeline validation only, never a paper result.
"""
from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path


LABEL_MAP = {"Infant": "Infant Cry"}
SIX_LABELS = {"Infant Cry", "Phee", "Seep", "Trill", "Tsik", "Twitter"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--examples-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    by_id = {}
    with args.annotations.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            event_id = Path(row["file_name"]).stem.rsplit("_", 1)[-1]
            by_id[event_id] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(args.examples_zip) as archive:
        for member in archive.infolist():
            if not member.filename.startswith("Audio_Examples/audios/") or not member.filename.endswith(".wav"):
                continue
            stem = Path(member.filename).stem
            event_id = stem.rsplit("_", 1)[-1]
            annotation = by_id[event_id]
            # Audio_Examples is a separately curated artifact and sometimes
            # disagrees with Annotations.tsv. Preserve both; use the example's
            # own filename label for this plumbing-only smoke set.
            nominal = stem.rsplit("_", 1)[0].replace("_", " ").title()
            label = "Infant Cry" if nominal == "Infant Cry" else nominal
            if label not in SIX_LABELS:
                continue
            output = args.output_dir / Path(member.filename).name
            with archive.open(member) as source, output.open("wb") as target:
                target.write(source.read())
            rows.append({
                "event_id": event_id,
                "audio_path": str(output.resolve()),
                "label": label,
                "recording_id": annotation["parent_name"],
                "provenance": "official_audio_example",
                "annotation_tsv_label": LABEL_MAP.get(annotation["label"], annotation["label"]),
                "example_annotation_agree": str(
                    label == LABEL_MAP.get(annotation["label"], annotation["label"])
                ).lower(),
            })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x["label"], x["event_id"])))
    print(f"wrote {len(rows)} examples to {args.manifest}")


if __name__ == "__main__":
    main()
