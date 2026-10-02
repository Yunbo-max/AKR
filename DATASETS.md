# Dataset provenance and redistribution policy

This repository tracks compact manifests, split registrations, preprocessing
settings, hashes, and evaluation outputs. It intentionally does not commit the
18 GB local `data/` tree because the experiments combine several third-party
datasets with different redistribution terms.

## What is tracked

- Every CSV under `data/manifests/`, including event IDs, fixed split roles,
  source filenames, SHA-256 values where available, conditions, and absolute
  local paths used during the run.
- Dataset and model configuration under `configs/`.
- Materialization and preprocessing code under `scripts/`.
- The BEANS-Zero source card at `data/beans/BEANS-Zero_README.md`.
- Final predictions, summaries, statistical outputs, and artifact audits.

Absolute paths in manifests are provenance records from the frozen run. Change
their common workspace prefix after materializing the same files elsewhere.

## MarmAudio

The local source bundle is the expert-validation release associated with the
MarmAudio technical validation material. Its bundled code carries a BSD
3-clause license, but that does not automatically establish a redistribution
license for every recording. The Git repository therefore includes annotations,
derived manifests, preprocessing code, and hashes—not the source ZIP or audio.

## BEANS Dogs

The local dataset card cites <https://doi.org/10.1016/j.anbehav.2003.07.016> and
contains 415/139/139 train/validation/test examples plus the BEANS low-resource
split. The card does not declare a clear audio redistribution license, so raw or
materialized dog recordings are not re-uploaded.

## BEANS Watkins

The local dataset card cites <https://doi.org/10.1121/2.0000358> and contains
1,017/339/339 train/validation/test examples. The source terms describe the
recordings as free for personal and academic use. To avoid silently broadening
those terms, Git and the model repositories contain manifests and adapter
weights but not Watkins audio.

## BEANS-Zero full target scan

The frozen scan materialized all 2,950 examples matching the 12 registered
components from `EarthSpeciesProject/BEANS-Zero`, with 16 kHz resampling, a
10-second prefix cap, and 100 ms minimum padding. It is reproducible with:

```bash
.venv/bin/python scripts/materialize_beans_zero_subset.py \
  --output-dir data/beans/beans_zero_targets_fullscan_cap10 \
  --output-manifest data/manifests/beans_zero_targets_fullscan_cap10.csv \
  --per-task 0 --target-rate 16000 --duration-cap 10 \
  --minimum-duration 0.1 --resume
```

The 2,950 records contain mixed licensing: 1,661 `CC BY-NC`, 330 explicit
CC-BY-NC 4.0 URLs, 324 `CC-BY`, 115 personal/academic-use records, 75 records
from the Animal Sound Archive, and 443 records whose manifest license is
`unknown`, plus small CC0/CC-BY-SA/CC-BY-NC-SA groups. Audio is therefore not
mirrored. Each tracked manifest row preserves its own source and license field.

## Local artifact inventory

After the final run, `scripts/build_local_artifact_inventory.py` writes
`LOCAL_ARTIFACTS.md`, recording byte sizes and file counts for ignored local
assets. `REPRODUCE.md` records model IDs and dataset revisions; the manifests
record per-event provenance.

