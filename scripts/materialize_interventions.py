#!/usr/bin/env python3
"""Materialize paired filtered 16 kHz WAVs from a canonical event manifest."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import soundfile as sf
import yaml

from animal_omni.audio import apply_intervention, intervention_grid, resample_for_qwen


def condition_name(item) -> str:
    if item.name == "full": return "full_0-8k"
    if item.name == "lowpass": return f"lp_0-{int(item.high_hz)}"
    return f"remove_{int(item.low_hz)}-{int(item.high_hz)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--conditions", nargs="+")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    with args.manifest.open(newline="", encoding="utf-8") as f:
        events = list(csv.DictReader(f))
    if args.split:
        events = [event for event in events if event.get("split") == args.split]
    grid = intervention_grid(cfg["audio"]["lowpass_cutoffs_hz"], cfg["audio"]["removed_bands_hz"])
    target_sr = cfg["model"]["target_sample_rate"]
    output_rows = []
    for event in events:
        waveform, sr = sf.read(event["audio_path"], always_2d=False)
        for item in grid:
            name = condition_name(item)
            if args.conditions and name not in args.conditions:
                continue
            filtered = apply_intervention(waveform, sr, item, cfg["audio"]["filter_order"])
            model_audio = resample_for_qwen(filtered, sr, target_sr)
            output = args.output_dir / name / f'{event["event_id"]}.wav'
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, model_audio, target_sr, subtype="FLOAT")
            output_rows.append({**event, "audio_path": str(output.resolve()), "condition": name})
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader(); writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} paired inputs to {args.output_manifest}")


if __name__ == "__main__":
    main()
