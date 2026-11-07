#!/usr/bin/env python3
"""Real intervention/no-op integrity and end-to-end component costs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from akr_final.train50_runner import main

if __name__ == "__main__":
    raise SystemExit(main("c14"))
