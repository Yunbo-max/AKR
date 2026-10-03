#!/usr/bin/env python3
"""No model work without --execute. See docs/AKR_FINAL_THREE_ZH.md."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from akr_final.runner import main
if __name__=='__main__':
    raise SystemExit(main())
