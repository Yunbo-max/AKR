#!/usr/bin/env python3
"""Entry point for the isolated AKR rebuttal protocol."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from akr_final.rebuttal_cli import main
if __name__=='__main__':raise SystemExit(main())
