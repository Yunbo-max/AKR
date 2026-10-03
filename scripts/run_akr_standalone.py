#!/usr/bin/env python3
"""Standalone F1/F2/F3 entry point (no legacy RUN/LOCK inputs)."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from akr_final.standalone import main
if __name__=='__main__':raise SystemExit(main())
