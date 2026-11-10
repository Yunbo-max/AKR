# Project-sanctioned torchaudio ABI workaround (see scripts/probe_mem_70gb.py).
# torchaudio 2.7.1 is the last wheel whose C ABI matches the NGC torch 2.8.0a0
# build; it was compiled for CUDA 12.6 while the image reports 12.9.  Report
# 12.6 only for the duration of the import, then restore the real value.
import torch

_real = torch.version.cuda
try:
    if _real and _real.startswith("12.9"):
        torch.version.cuda = "12.6"
    import torchaudio  # noqa: F401
finally:
    torch.version.cuda = _real

import os as _os
if _os.environ.get("AKR_DETERMINISTIC") == "1":
    torch.use_deterministic_algorithms(True)
