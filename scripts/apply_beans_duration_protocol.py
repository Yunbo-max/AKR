#!/usr/bin/env python3
"""Apply BEANS' deterministic mono, prefix-truncate, zero-pad duration protocol."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-duration", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    output_rows = []
    for row in rows:
        waveform, rate = sf.read(row["audio_path"], always_2d=True, dtype="float32")
        mono = waveform.mean(axis=1)
        count = int(round(args.max_duration * rate))
        mono = mono[:count]
        if len(mono) < count: mono = np.pad(mono, (0, count - len(mono)))
        output = args.output_dir / row["split"] / f'{row["event_id"]}.wav'
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, mono, rate, subtype="FLOAT")
        output_rows.append({**row, "audio_path": str(output.resolve()),
                            "beans_max_duration_s": args.max_duration,
                            "duration_protocol": "mono_mean_prefix_truncate_zero_pad_before_qwen_resample"})
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0]); writer.writeheader(); writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} examples at {args.max_duration:g}s")


if __name__ == "__main__": main()
