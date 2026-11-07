# Historical source-locked readout supplement

`experiments/rebuttal_cases/c16_historical_readouts.py` makes the existing S1
readout comparison independently runnable against an explicit, locally available
original registration. It uses `FinalRunner` source mode and never invokes its
standalone split builder. This supplement is separate from the Dogs/Watkins/Bats/
HumBugDB Train-50 cases. It does not recover missing historical artifacts or
update manuscript numbers.

## Required source and supported protocol

Provide an original source directory containing `RUN.json`, registered split
files, and each requested episode's `EPISODE.json` and `LOCK.json`, together with
the manifest and local audio referenced by that registration. The options YAML
uses the existing `configs/akr_final_three.yaml` schema. Its dataset, number of
seeds and `k_per_class` select existing registrations; they do not sample a fresh
support set. The wrapper retains every selected episode's exact ordered support
and confirmation query IDs. Missing locks or registrations cause failure rather
than episode omission. Existing `FinalRunner` source mode supports Qwen2.5-Omni
7B, semantic labels, and both `full` and `lp:1000` conditions.

Only registered `relative_active_state_norm` injection locks are supported.
**The original raw eta=300 main-table protocol remains unresolved.** If that
protocol has no compatible recovered registration, this entry point explicitly
reports that the historical raw protocol was not recovered. It does not invent a
20-support protocol, reinterpret raw eta as relative alpha, or claim exact
reproduction of old main-table numbers. Even a successful compatible source run
is a newly labeled readout extension on previously inspected confirmation data.

The existing runner compares loaded backend model ID, revision, layers, feature
layer, dtype and batch size against the registered source episode. Reports retain
that source backbone metadata, source lock hashes, any newly loaded backend
provenance, and source-cache compatibility. Reused source artifacts remain subject
to the existing runner's code and provenance checks. A report with no newly loaded
backend does not invent backend metadata; it retains the registered source
metadata and cache compatibility status.

## Commands

Run each command independently from the repository root, replacing the paths and
run name with the actual local registration. The first two commands do not load
a model or launch GPU work.

```bash
python experiments/rebuttal_cases/c16_historical_readouts.py plan \
  --source-run results/akr_closing/ORIGINAL_RUN \
  --config configs/akr_final_three.yaml --run-id historical_readouts_v1

python experiments/rebuttal_cases/c16_historical_readouts.py check \
  --source-run results/akr_closing/ORIGINAL_RUN \
  --config configs/akr_final_three.yaml --run-id historical_readouts_v1

python experiments/rebuttal_cases/c16_historical_readouts.py run \
  --source-run results/akr_closing/ORIGINAL_RUN \
  --config configs/akr_final_three.yaml --run-id historical_readouts_v1

python experiments/rebuttal_cases/c16_historical_readouts.py report \
  --source-run results/akr_closing/ORIGINAL_RUN \
  --config configs/akr_final_three.yaml --run-id historical_readouts_v1
```

`--root` changes the repository/data root. Relative source and config paths are
resolved against it. `--profile` accepts `16gb`, `24gb` or `48gb`; `--cache-only`
blocks model loading when saved arrays or predictions are unavailable. Each run
writes independently to `results/akr_historical_readouts/<run-id>/`, without a
global GPU queue or writes inside the original source directory.

## Methods and verification

S1 evaluates `native`, `fixed_mean`, `continuous_pooled`, `ridge_direct`,
`probe_to_text`, and `probe_to_lm` on the same registered query IDs. The direct
ridge label and its text rendering intentionally have identical predictions;
probe-to-LM supplies the support-trained classifier prediction as fallible text
evidence to the frozen LM. No query target enters inference. A zero-alpha no-op
check is also required by S1.

`check`, `run` and `report` reset completion before config loading, runner
construction or preflight, so missing source artifacts, changed registration,
missing waveforms, incompatible backend metadata and corrupt results cannot leave
a stale successful status. A complete report requires every expected episode,
all six methods, exact ordered support/query IDs, per-episode completion records,
matching metrics and scored predictions, and valid journal identities/checksums.
`ARTIFACTS.json` records the full S1 artifact population and hashes; reporting
rejects missing, added or changed artifacts. Model outputs may contain invalid
labels and are scored as such; completion is not an accuracy threshold.

Outputs include `REPORT.json`, `STATUS.json`, the immutable runner identity,
waveform/source provenance, S1 per-method predictions and metrics, and the artifact
receipt. The wrapper corrects S1's generic standalone settings-origin wording to
identify the actual registered source lock. Completion means this compatible
source-locked supplement passed integrity checks; it does not establish raw eta
historical parity or resolve the missing original main-table registration.
