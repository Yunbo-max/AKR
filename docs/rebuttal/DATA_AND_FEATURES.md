# Train-50 sources and frozen acoustic baselines

The primary suite is Dogs, Watkins, Bats, and HumBugDB. Preparation preserves every
row of each official BEANS train/valid/test split as mirrored by the DBD research
group. The downstream protocol selects **50 total stratified training examples**
with an outcome-blind seed, excludes training candidates that overlap test by
source or decoded audio, and evaluates the **complete official test**. Preparation
performs no support selection. `train_low` is a redundant subset, never a fourth
independent partition. BEANS-Zero is a separate derived task and is not a source
for this suite.

## Verified sources and immutable revisions

The DBD group's [BEANS collection](https://huggingface.co/collections/DBD-research-group/beans-datasets)
identifies the four mirrors. On 2026-11-07, the public HF dataset metadata API and
small README files were read to verify repository IDs, immutable HEAD revisions,
parquet shard names, schema, and full split populations. This did **not** download
any dataset parquet/audio or model checkpoints. The original
[BEANS repository](https://github.com/earthspecies/beans) supplies benchmark
preparation code; mirror cards describe their splits and preprocessing as BEANS.
These are DBD-hosted mirrors, not repositories owned by the original BEANS authors.

| Dataset | HF repository | Frozen revision | Train | Valid | Complete test |
|---|---|---|---:|---:|---:|
| Dogs | [DBD-research-group/beans_dogs](https://huggingface.co/datasets/DBD-research-group/beans_dogs) | `b5928b67fcf4949cf782f3c4c64fecfc3c6be72f` | 415 | 139 | 139 |
| Watkins | [DBD-research-group/beans_watkins](https://huggingface.co/datasets/DBD-research-group/beans_watkins) | `e57b7eb6e0868ceec03f75b31e6d7bac28dbd98a` | 1,017 | 339 | 339 |
| Bats | [DBD-research-group/beans_bats](https://huggingface.co/datasets/DBD-research-group/beans_bats) | `fb9d6b65fa3f0f41721e1f0d089f6b5cc23099e3` | 6,000 | 2,000 | 2,000 |
| HumBugDB | [DBD-research-group/beans_humbugdb](https://huggingface.co/datasets/DBD-research-group/beans_humbugdb) | `3b1dc1e30c6f6e382749f1ca2ae5f993ec67a537` | 5,577 | 1,859 | 1,859 |

The public metadata API endpoint is
`https://huggingface.co/api/datasets/<repository>`; a pinned README URL has the
form `https://huggingface.co/datasets/<repository>/raw/<revision>/README.md`.
Expected train/valid/test shard counts are respectively 2/1/1 for Dogs and
Watkins, 22/8/8 for Bats, and 7/6/6 for HumBugDB. The mirror README YAML contains
malformed numeric `num_bytes` fields for Bats and HumBugDB; direct parquet reading
avoids relying on that metadata parser. Actual parquet validation remains a
necessary first materialization step; metadata inspection does not establish
successful extraction of all upstream bytes.

All mirrors expose `Unnamed: 0`, an embedded-audio `path` structure containing
`bytes` and `path`, and `label`. Dogs, Watkins and HumBugDB label fields are
strings; Bats labels are integer IDs. Integer labels are converted losslessly to
strings. No semantic class names are inferred for Bats. The source studies cited
by the mirror cards are [Dogs](https://doi.org/10.1016/j.anbehav.2003.07.016),
[Watkins](https://doi.org/10.1121/2.0000358),
[Bats](https://doi.org/10.1038/sdata.2017.143), and
[HumBugDB](https://doi.org/10.48550/arXiv.2110.07607).

## Portable preparation

Install the independent CPU preparation/features requirements:

```bash
python -m pip install -r requirements-train50-features.txt
```

The following commands download only the pinned dataset's canonical parquet
splits and README, preserve the embedded source bytes locally, and publish a CSV
plus a provenance JSON beside it. They are commands for the experiment operator;
no dataset download or scientific evaluation was run during code preparation.

```bash
python scripts/prepare_train50_data.py --dataset dogs
python scripts/prepare_train50_data.py --dataset watkins
python scripts/prepare_train50_data.py --dataset bats
python scripts/prepare_train50_data.py --dataset humbugdb
```

Defaults are `data/manifests/train50/<dataset>.csv`, with the sidecar
`<dataset>.provenance.json`, and audio beneath `data/beans/train50/<dataset>`.
Use `--output-dir` and `--audio-dir` for a different writable location. The four
pinned snapshots together contain substantial source audio; provision disk space
for both HF cache and materialized audio. `--revision` accepts a full immutable
40-character SHA only, never a moving branch or tag; it is optional because each
dataset already has an explicit pinned default.

An existing HF cache can be used without networking:

```bash
python scripts/prepare_train50_data.py --dataset dogs --local-files-only
```

`--snapshot-dir` accepts a standard
`datasets--DBD-research-group--beans_dogs/snapshots/<revision>` directory, or a
clean dataset Git checkout with matching immutable HEAD and official HF origin.
The receipt records which verification mechanism was used. A cache path is a
local cache identity convention, not independent cryptographic proof that bytes
match a remote repository; per-shard hashes and manifest hashes make the exact
local input population auditable. Existing outputs with different bytes fail as
collisions. Existing identical outputs can be reused. Missing shards, unknown
split filenames, malformed records, missing true source filenames, duplicate
event IDs, and unexpected split populations fail. There is no option to truncate
test or select support here. Validation is preserved for provenance; the main
Train-50 experiment does not fit using its labels.

The receipt includes repository/revision identity, every input parquet SHA-256,
README hash when present, manifest SHA-256, preparation code SHA-256, observed
and expected split counts, per-split class counts, and audio policy. Every CSV row
preserves source index, source filename, original embedded path, label, canonical
split, raw audio SHA-256, source shard and revision. Event IDs namespace dataset,
split and original source index; materialized file names use those safe IDs.

`source_filename` and `source_group` contain the real upstream audio basename,
not the generated event ID. `group_provenance` explicitly states
`source_filename_only_parent_recording_unknown`. The mirror schema does not
establish an independent parent-recording/session identifier. Matching filenames
can conservatively join examples, but different clip names can still share a
parent recording. No stronger recording-disjointness claim follows from this
metadata alone. The downstream protocol additionally checks decoded PCM overlap
and preserves all test examples while excluding overlapping training candidates.

## Frozen feature interfaces and definitions

```python
from akr_final.train50_features import FeatureExtractor, extract_feature, feature_provenance

extractor = FeatureExtractor("mfcc")
x = extractor.extract("example.wav")  # nonempty finite NumPy vector, shape (78,)
identity = extractor.provenance()       # JSON-safe, no path-only identities
# One-shot wrappers expose the same extraction/provenance API.
```

The extractor applies channel-mean mono conversion and deterministic SciPy
polyphase resampling to 16 kHz. It performs no peak normalization or duration
truncation; any experiment duration/condition rule must be applied identically to
all compared methods before extraction. A returned vector must be finite and
one-dimensional. Unexpected schemas or nonfinite outputs fail; missing optional
backends fail explicitly and never substitute another feature. Extraction does
not fit a standardizer or classifier. Downstream readouts fit standardization
and Ridge using support examples only.

| Kind | Actual extraction | Output |
|---|---|---|
| `mfcc` | Fixed SciPy/NumPy conventional MFCC DSP described below | 78 dimensions |
| `egemaps` / `egemapsv02` | `opensmile.Smile(FeatureSet.eGeMAPSv02, FeatureLevel.Functionals).process_signal` | 88 functionals |
| `beats` | Official Microsoft BEATs pretrained checkpoint, last hidden representations, token mean | Checkpoint encoder width; standard base checkpoint 768 |

MFCC uses pre-emphasis 0.97, a symmetric Hamming window of 400 samples (25 ms),
hop 160 (10 ms), a 512-point real FFT, power spectrum divided by 512, 40 HTK-mel
triangular filters spanning 0–8 kHz without area normalization, natural log with
floor `1e-10`, and orthonormal DCT-II retaining 13 coefficients including C0.
The last incomplete frame is zero-padded. First/second deltas use regression
width two with replicated edge frames. The vector concatenates population mean
and standard deviation of the 13 static, 13 delta and 13 delta-delta coefficients.
All settings and the extraction module hash are included in provenance, along
with NumPy, SciPy and SoundFile versions.

The eGeMAPSv02 implementation follows
[the official openSMILE Python usage](https://audeering.github.io/opensmile-python/usage.html).
It requires the actual optional `opensmile` package, pinned to 2.6.0 in the supplied
requirements. Provenance includes installed package version, ordered feature
names, and hashes of packaged configuration/include files and shared libraries.
The extractor validates a single row of exactly 88 finite functionals. This is a
real eGeMAPSv02 extraction; no MFCC approximation is used. A fixed 250 ms
right-zero-pad is applied to inputs shorter than 4000 samples before openSMILE
(the same policy as the BEATs frontend; recorded in provenance as
`short_audio=right_zero_pad`, `minimum_samples=4000`). Without it, openSMILE
fills a too-short segment with NaN; the pad makes the functionals finite and is
applied identically to support and query audio.

## Official BEATs assets

The implementation follows the
[official inference example](https://github.com/microsoft/unilm/blob/master/beats/README.md)
and [official BEATs code](https://github.com/microsoft/unilm/blob/master/beats/BEATs.py).
The inspected UNILM HEAD on 2026-11-07 was
`31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd`. An operator must provision a clean
checkout and a trusted **pretrained**, non-finetuned checkpoint from the links
in that README; no checkpoint was downloaded or invented for this change.
Torch and torchaudio must be mutually compatible and appropriate to the operator's
CPU/CUDA environment. The official BEATs `backbone.py` and `modules.py` are loaded
from the same pinned source directory. No third-party BEATs reimplementation is
used. The supplied CPU requirements do not replace an existing Speech-LM Torch
installation.

```python
assets = {
    "source_dir": "/path/to/unilm/beats",
    "source_revision": "31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd",
    "checkpoint": "/path/to/BEATs_iter3_plus_AS2M.pt",
    "checkpoint_sha256": "<the trusted checkpoint's actual 64-character SHA-256>",
    "device": "cpu",  # choose an available device explicitly for the real run
}
extractor = FeatureExtractor("beats", assets)
identity = extractor.provenance()
x = extractor.extract("example.wav")
```

`checkpoint_sha256` must be supplied and match the actual bytes; the placeholder
above is intentionally not executable. The source Git HEAD and official origin
are checked, modifications to the source directory are rejected, and the actual
Python source files are hashed. The checkpoint hash, source hashes/revision,
Torch/torchaudio versions, device, dtype and all preprocessing/pooling settings
are included in provenance. Assets are checked again before lazy loading. The
checkpoint is loaded with the official `cfg`/`model` structure and a strict state
dict. `finetuned_model=True` is rejected because the official API returns AudioSet
class probabilities for that configuration rather than the intended hidden
representations. The model runs in evaluation/inference mode, float32, with all
parameters frozen. The official fbank normalization (`mean=15.41663`,
`std=6.55582`, division by twice std) is used. A fixed 250 ms minimum pads very
short inputs for the official patch frontend and is recorded in identity. Every
example runs separately; no batch padding enters the temporal mean.

## Verification and remaining execution

Tests cover MFCC repeatability, finite schema, response to changed audio,
stereo/resampling/short clips, rejection of nonfinite PCM, provenance independence,
missing-backend failure, source/checkpoint identity checks, actual eGeMAPSv02
functionals when installed, canonical shard completeness, malformed source rows,
numeric labels, duplicate IDs, output collisions, immutable revisions, and real
parquet materialization with the full 139-example Dogs test population. The
parquet fixture is synthetic audio with real split sizes; it is a preparation
integrity test, not a Dogs scientific experiment.

```bash
python -m pytest -q tests/test_train50_features.py tests/test_prepare_train50_data.py
```

The BEATs execution test requires explicitly provisioned local assets via
`AKR_TEST_BEATS_ASSETS=/path/to/assets.json`; otherwise it skips with an explicit
reason. It never downloads model weights and forces CPU. Actual full-dataset
materialization, BEATs extraction, and all classifier/AKR scientific outcomes
remain pending until the operator provisions the data/model assets and executes
the registered experiment. Metadata verification and synthetic tests must not be
reported as completed scientific baselines.
