#!/usr/bin/env python3
"""AKR release: plan/check without weights; explicit bounded GPU execution."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from akr_final.release import main
if __name__ == '__main__':
    raise SystemExit(main())
