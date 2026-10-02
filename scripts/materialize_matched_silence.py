#!/usr/bin/env python3
"""Create duration/rate/channel-matched zero waveforms for registered events."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--support-k-per-class", type=int, required=True)
    parser.add_argument("--condition")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if args.condition is not None:
        rows = [row for row in rows if row.get("condition") == args.condition]
    by_event = {row["event_id"]: row for row in rows}
    split = json.loads(args.split.read_text())
    events = split["support_sets"][str(args.support_k_per_class)] + split["query_events"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for index, event in enumerate(events, 1):
        row = by_event[event]
        info = sf.info(row["audio_path"])
        path = args.output_dir / f"{event}.wav"
        if not path.exists():
            silence = np.zeros((info.frames, info.channels), dtype=np.float32)
            if info.channels == 1:
                silence = silence[:, 0]
            sf.write(path, silence, info.samplerate, subtype="PCM_16")
        output.append({
            "event_id": event, "audio_path": str(path.resolve()),
            "label": row["label"], "recording_id": row.get("recording_id", ""),
            "condition": "matched_silence", "source_audio_path": row["audio_path"],
            "source_condition": row.get("condition", "full_0-8k"),
            "sample_rate": info.samplerate, "frames": info.frames,
            "channels": info.channels,
        })
        if index == 1 or index % 50 == 0 or index == len(events):
            print(f"[{index}/{len(events)}] {event}", flush=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_manifest.with_suffix(args.output_manifest.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0])
        writer.writeheader(); writer.writerows(output)
    temporary.replace(args.output_manifest)


if __name__ == "__main__":
    main()
