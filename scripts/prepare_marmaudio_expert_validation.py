#!/usr/bin/env python3
"""Extract the unanimous expert-affirmed six-class MarmAudio validation set."""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path


LABEL_MAP = {"Infant_cry": "Infant Cry"}
SIX = {"Infant Cry", "Phee", "Seep", "Trill", "Tsik", "Twitter"}
AFFIRMATIVE = {"Yes", "Yes-Yes", "No-Yes"}  # matches authors' validation script


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    annotations = {}
    with args.annotations.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            annotations[Path(row["file_name"]).stem.rsplit("_", 1)[-1]] = row

    with zipfile.ZipFile(args.archive) as archive:
        rating_files = [n for n in archive.namelist() if "/validation_labels/" in n and n.endswith(".tsv")]
        ratings: dict[str, list[str]] = {}
        for name in rating_files:
            table = csv.DictReader(io.StringIO(archive.read(name).decode()), delimiter="\t")
            for row in table:
                ratings.setdefault(row["id"], []).append(row["type"])
        if len(rating_files) != 4:
            raise RuntimeError(f"expected four expert files, found {len(rating_files)}")

        members = {Path(n).stem: n for n in archive.namelist() if "/audios/" in n and n.endswith(".wav")}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for item_id, votes in ratings.items():
            raw_label, event_id = item_id.rsplit("_", 1)
            label = LABEL_MAP.get(raw_label, raw_label)
            if label not in SIX or not all(vote in AFFIRMATIVE for vote in votes):
                continue
            member = members[item_id]
            output = args.output_dir / f"{item_id}.wav"
            with archive.open(member) as source, output.open("wb") as target:
                target.write(source.read())
            annotation = annotations[event_id]
            rows.append({
                "event_id": event_id, "audio_path": str(output.resolve()), "label": label,
                "recording_id": annotation["parent_name"], "provenance": "four_experts_unanimous_affirmative",
                "expert_votes": "|".join(votes),
            })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda r: (r["label"], r["event_id"])))
    print(f"wrote {len(rows)} unanimous expert-affirmed events to {args.manifest}")


if __name__ == "__main__":
    main()

