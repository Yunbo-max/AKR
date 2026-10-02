# Animal Omni KV

Reproducible experiments asking what Qwen2.5-Omni hears in animal vocalizations within its **observable 0–8 kHz baseband**, and whether Thinker KV-state adaptation can recover frequency-induced failures.

> **Protocol correction (2026-08-14):** zero-shot generation versus a fully
> supervised probe is not a fair grounding gap, especially for arbitrary dog
> names. The current claim, equal-support results, failed tokenwise gate, and
> invalidated artifacts are recorded in `FAIR_GAP_RESULTS.md` and
> `FAIR_GAP_12H_RUN.md`. `RESULTS.md` and `FINAL_REPORT.md` contain the earlier
> experiment ledger and must not be cited without those corrections.

Release documentation:

- `FAIR_GAP_12H_RUN.md`: preregistered equal-supervision protocol and gates.
- `FAIR_GAP_RESULTS.md`: authoritative main-run results and claim decision.
- `FINAL_EXTENSION_RESULTS.md`: generated A--J and complete BEANS-Zero extension.
- `REPRODUCE.md`: exact commands and pinned environment.
- `MODEL_USAGE.md`: loading and inference for released Thinker LoRA adapters.
- `DATASETS.md`: provenance, licenses, manifests, and redistribution boundaries.
- `CT_KV_BASELINE_ASSESSMENT.md`: requirements and current blocker for a faithful
  multimodal Context-Tuning baseline.

## First-stage protocol

The minimal gate is Qwen2.5-Omni-3B plus the six reliable MarmAudio call types. Every event is filtered at its original sample rate (96 kHz for MarmAudio) and only then anti-alias resampled to 16 kHz. “Full” always means the full 0–8 kHz signal visible to Qwen, not the animal recording's full spectrum.

Interventions are paired by event: full; low-pass at 1/2/4/6/8 kHz; and removal of 0–1, 1–2, 2–4, 4–6, or 6–8 kHz. Recording-group splits prevent nearby events from the same source recording leaking across train, validation, and test. The untouched test split is reserved for final reporting.

The expansion gate requires both (1) systematic spectral degradation and (2) meaningful Oracle-KV recovery on samples that are correct under full input and wrong when degraded. Oracle gradients are an upper bound only; query-label-free conditional KV may use labels on support examples but never on queries.

Qwen2.5-Omni-7B is a pre-registered fallback and confirmation model, not an after-the-fact replacement. On the fixed validation subset, 7B is triggered if 3B full-band accuracy is at most 25%, the paired bootstrap does not support a positive frequency trend, or Oracle-KV recovers at most 20% of eligible failures. Regardless of the trigger, a balanced paired core subset (50 test events per class) is eventually evaluated with both sizes. On a 24GB GPU, the official AWQ checkpoint may screen 7B generation; all reported KV-gradient geometry uses the unquantized checkpoint in BF16 with CPU offload.

The 3B trigger fired because its full-baseband accuracy was 20.33%. The unquantized
7B therefore became the main diagnostic model and improved full accuracy to 29.49%
(paired +9.16 points, 95% CI +4.76 to +13.37). All reported 7B generation and KV
experiments use BF16; the AWQ screening option was not needed.

## Environment

```bash
cd /root/animal-omni-kv
python3 -m venv .venv
.venv/bin/pip install -e '.[test,qwen]'
.venv/bin/pytest
```

On this host, PyPI's default Torch wheel targeted CUDA 13 and was incompatible with the installed driver. The verified runtime is Torch 2.13.0+cu126, Transformers 4.52.3, torchvision 0.28.0+cu126, and qwen-omni-utils 0.0.9. A real 3B Thinker-only forward pass used 9.41 GB after load and peaked at 11.98 GB for a short MarmAudio example.

The canonical configuration is `configs/marmaudio_minimal.yaml`. The expected manifest is CSV with columns `event_id,audio_path,label,recording_id`; `audio_path` may point at a pre-segmented WAV or a future materialization step derived from the official onset/offset annotations.

The official Zenodo release stores already segmented vocalizations; annotations use `Infant` while the paper names the class `Infant Cry`, so the preparation script records that explicit mapping. The separately curated `Audio_Examples.zip` filename labels disagree with `Annotations.tsv` for a substantial subset; the smoke manifest preserves both labels and an agreement flag. Its balanced 60 files can validate plumbing but are prohibited as a reported benchmark result. Formal sampling uses TSV labels only.

For the first trustworthy diagnostic, `Technical_Validation_Data.zip` contains 100 candidates per class independently reviewed by four experts. Following the authors' own vote normalization, requiring unanimous affirmative votes leaves 546 examples across the six named classes (Infant Cry 100, Phee 100, Twitter 99, Tsik 85, Trill 83, Seep 79). Generative baselines evaluate this set directly. Any supervised analysis on it must produce strictly out-of-fold query predictions with recording-group separation; it cannot be presented as a fixed external test or used for final router/LoRA claims.

Because Zenodo ignores HTTP Range requests for the 59 GB vocalization archive, the first linear-probe diagnostic uses nested, stratified, recording-grouped out-of-fold predictions on the expert set. Each query is classified by a probe trained without its recording; layer and ridge strength are selected only in an inner grouped split. These OOF results are distinct from fitting and evaluating on the same expert examples and must be described as cross-validation, not a fixed external test.

The official MarmAudio release also includes its six-class 96 kHz ResNet-50 checkpoint and exact frontend (1–48 kHz filterbank). It is evaluated as an animal-specialist, original-full-spectrum reference. It is not a frequency-matched 0–8 kHz comparator to Qwen, so tables must label its accessible spectrum explicitly.

## Reproducibility invariants

- Filter original audio before 16 kHz resampling.
- Never normalize each intervention independently; that would erase attenuation evidence. Peak normalization is off by default.
- Use the same event IDs and prompt across all conditions.
- Use batch size 1 for authoritative Qwen results. Variable-length batching was empirically non-invariant for the same audio, likely through padding/position interactions; batched runs are exploratory throughput tests only.
- Decode deterministically and count invalid/multiple-label text as incorrect.
- Choose filter order, Oracle step size, layers, rank, and router settings on validation only.
- Report accuracy and macro-F1 with paired bootstrap confidence intervals in the final pipeline.
- Do not describe MarmAudio classifier-derived labels as purely human ground truth; use the authors' confidence-qualified six-label subset and report provenance.

## Status

The frequency diagnostics remain complete, but the paper protocol has been
repaired around equal K-per-class supervision. MarmAudio candidate likelihood,
definitions, recording-disjoint audio ICL, centroid, and ridge readouts are
complete. Dogs uses the exact same registered support for ICL, centroid, ridge,
one-epoch LoRA, relative-norm KV, token permutation, and random-field controls.
Watkins has been rebuilt with 31/62/124 support examples for K=1/2/4 per class;
its old K=20-total KV result is withdrawn from method interpretation.

On Dogs validation, conditional pooled KV improves over fixed mean, and ordered
tokenwise improves over token permutation, but ordered tokenwise does not beat
pooled or remain stable across alpha. The preregistered test gate therefore
fails, and no new fair-protocol test result is promoted. Full-prefill Oracle
recovery is now described only as label-specific decision capacity; audio-only
matched controls are the narrower causal acoustic evidence.

The official AVES-bio ONNX port has been evaluated under the MarmAudio grouped
nested OOF protocol and on the full official Dogs/Watkins fixed splits. BEANS Dogs
and Watkins fixed splits have been materialized with
the benchmark's mono/prefix-truncate/zero-pad duration rules and Qwen-7B zero-shot
test results, frozen probes, frequency interventions, and query-label-free KV runs
are complete. The full 2,950-example, 12-component BEANS-Zero capped diagnostic is
complete with official per-example instructions. One-epoch full-train Thinker
LoRA baselines are complete on both Watkins (31.56% test) and Dogs (25.18% test),
and the AVES/official MarmAudio specialist baselines are complete. Matched-support
MarmAudio KV geometry is complete at full/1/2/4/6/8 kHz. Full official-split
Dogs degraded probe curves are complete with 693/693 representations at every
condition and 87.77--94.96% test accuracy; none differs significantly from the
92.81% full-input probe. Watkins is also complete with 1,695/1,695
representations at every condition; its condition-specific probe rises from
74.04% at 1 kHz to 88.20% at full input. Unchanged full-trained probes expose
far larger 1-kHz boundary drift (Dogs 40.29%, Watkins 9.73%) despite high
condition-specific decodability. Complete 6x6 source-by-target decoder matrices
also show reciprocal 1-kHz-to-full failures (Dogs 38.85%, Watkins 20.35%) while
reproducing every condition-specific diagonal exactly. Holding the layer and
Ridge strength fixed at the full-selected values makes the reciprocal cells
30.22% and 23.01%, so layer selection is not the cause. No earlier support-only
lp1 directory is promoted as a full probe. See `RESULTS.md` for authoritative metrics and
`REPRODUCE.md` for commands.

The corrected result summary is `FAIR_GAP_RESULTS.md`. Machine-readable audit and
paired statistics are `results/fair_gap_artifact_audit.json` and
`results/fair_gap_paired_statistics.json`. Corrected paper tables are under
`results/fair_gap_tables/`; the three corrected figures are
`figures/fair_fig1_equal_supervision.png`, `fair_fig2_kv_controls.png`, and
`fair_fig3_gradient_geometry.png`.
