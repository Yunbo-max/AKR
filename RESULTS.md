# Results ledger

> **Historical ledger warning (2026-08-14):** authoritative equal-supervision
> conclusions now live in `FAIR_GAP_RESULTS.md`. In particular, do not headline
> zero-shot generation versus fully supervised probes, the raw-eta 111/111
> Oracle, Dogs test-selected follow-ups, or Watkins K=20-total as fair evidence.

This file distinguishes authoritative results from exploratory or invalidated runs. “Full” for Qwen means its complete observable 0–8 kHz baseband after the official 16 kHz processing path; it never means MarmAudio's original 0–48 kHz Nyquist range.

## Authoritative completed results

All Qwen generation below uses deterministic decoding, batch size 1, the same fixed label prompt, strict single-label parsing, and 546 four-expert-unanimous MarmAudio examples.

| Model | Accessible spectrum | Accuracy | Macro-F1 | Notes |
|---|---:|---:|---:|---|
| Qwen2.5-Omni-3B Thinker | 0–8 kHz | 20.33% | 12.73% | Outputs dominated by Phee |
| Qwen2.5-Omni-7B Thinker | 0–8 kHz | 29.49% | 21.94% | 7B fallback trigger fired and improved +9.16 pp |
| Official MarmAudio ResNet-50 | 0–48 kHz | 88.64% | 88.68% | Animal specialist, original full spectrum |
| Official MarmAudio ResNet-50 | low-pass 0–8 kHz at 96 kHz rate | 18.32% | 5.16% | 546/546 outputs collapse to Phee; frontend is out of its trained spectral regime |
| AVES-bio + nested grouped linear probe | 0–8 kHz | 93.41% | 93.09% | Same 546 expert examples and grouped OOF protocol as Qwen probe |

Per-class accuracy for Qwen-7B full observable baseband: Infant Cry 0%, Phee 63%, Seep 18.99%, Trill 0%, Tsik 82.35%, Twitter 13.13%.

The paired event bootstrap for 7B minus 3B accuracy is +9.16 percentage points, 95% CI [+4.76, +13.37], n=546.

The full paired scaling sweep shows that larger is not uniformly more robust.
At 1 kHz low-pass, 3B reaches 17.22% while 7B reaches 12.64%, so 7B-minus-3B is
**-4.58 points** (95% CI [-7.51, -1.65]). At 2 kHz the difference is statistically
indistinguishable from zero (+0.55 points, CI [-2.20, +3.48]); at 4, 6, and 8 kHz
7B is ahead by +3.48, +7.88, and +8.42 points respectively, with all three CIs
excluding zero. The 7B advantage therefore grows with accessible bandwidth rather
than reflecting universal resilience to severe spectral degradation.

## Engineering validation

- Qwen-3B real generation: 9.41 GB after load, 11.98 GB observed peak.
- Qwen-7B BF16 generation and KV gradients fit on the 24 GB RTX 3090; Oracle backward observed peak was 22.37 GB.
- 7B has 28 Thinker layers and yielded 56 pre-RoPE projection directions per labeled example; 3B has 36 layers and yielded 72.
- On one 7B eligible-style example, an Oracle direction reduced teacher-forced label loss from 0.567 to 0.014 at eta=1. Loss reduction alone is not counted as recognition recovery; the authoritative Oracle pipeline regenerates text.

## Frozen representation probe

The Qwen-7B full-baseband probe uses mean-pooled audio-token hidden states from all
29 representation levels. Layer and Ridge regularization are selected inside each
outer training fold; evaluation is five-fold stratified, recording-grouped OOF over
all 546 events (93 recordings). It reaches **95.42% accuracy** (recording bootstrap
95% CI [93.58, 97.13]) and **95.32% macro-F1** (95% CI [93.30, 97.07]). Per-class
accuracy is 99.00% Infant Cry, 97.00% Phee, 96.20% Seep, 89.16% Trill, 92.94%
Tsik, and 96.97% Twitter.

This is supervised grouped OOF evidence, not an external held-out benchmark score.
Its 65.93-point accuracy advantage over direct generation on the same expert corpus
is an **upper-ceiling diagnostic**, not an equal-supervision grounding gap: the
probe learns a label map that zero-shot generation never receives. It establishes
that call-type information can be decoded from frozen Qwen states and motivates the
registered equal-support experiments; those experiments, rather than this raw
difference, determine whether the native decision interface under-uses that
information.

Under the identical grouped nested OOF protocol, the official 16 kHz AVES-bio
encoder reaches 93.41% accuracy / 93.09% macro-F1. Qwen-7B states are therefore
competitive with this animal-specialist representation baseline (95.42% / 95.32%);
the dramatic Qwen generation gap cannot be attributed simply to absent acoustic
class information. AVES uses its official ONNX port, batch size one, last-layer
time-mean pooling, and the same inner selection of Ridge regularization.

After 1 kHz low-pass, the condition-specific grouped Qwen probe still reaches
**77.29% accuracy / 76.91% macro-F1**, while direct generation is only 12.64% /
5.88%. Relative to the full probe (95.42% / 95.32%), this shows genuine spectral
information loss while leaving substantial condition-specific decodability. The
unmatched native/probe difference is descriptive only; the unchanged full-trained
decoder transfer below is the stricter evidence for decision-boundary drift.

The full condition-specific probe curve at 1/2/4/6/8 kHz is **77.29, 79.85,
82.60, 93.96, and 94.87%** accuracy. A stricter transfer probe that selects and
fits only on full-band training folds, then applies the unchanged classifier to
paired degraded OOF queries, reaches only **34.98, 54.58, 59.52, 88.64, and
95.42%**. Narrow-band states therefore retain substantial supervised class
information but undergo a large distribution/readout shift. Condition-specific
probe recovery is diagnostic supervision, not zero-shot robustness.

## BEANS fixed-split zero-shot expansion

Official BEANS-mirror train/valid/test splits were materialized from embedded WAV
bytes. The published preprocessing contract is applied before Qwen: mono channel
mean, prefix truncation, and zero padding to 10 s for Dogs or 3 s for Watkins;
Qwen's official processor then resamples to 16 kHz. On test, Qwen-7B obtains **2.88%
accuracy / 0.56% macro-F1** on Dogs individual recognition (139 examples), predicting
Rudy for all examples. On Watkins species recognition (339 examples), it obtains
**5.60% / 1.87%**, with outputs concentrated in False Killer Whale (140), Beluga
White Whale (107), and Bottlenose Dolphin (84). These are zero-shot classification
results, not adapted results.

Using the fixed train/valid/test splits for supervised frozen probes, Qwen-7B
reaches **92.81% accuracy / 90.95% macro-F1** on Dogs (layer 10) and **88.20% /
88.25%** on Watkins (layer 8). Layers and Ridge alpha are selected on validation,
then the probe is refit on train+validation and evaluated once on test. Thus the
representation–generation gap generalizes across individual identity and species,
not only MarmAudio call type.

Under the same official fixed splits and untouched tests, the official AVES-bio
ONNX encoder with a validation-selected Ridge probe reaches **83.45% accuracy /
80.72% macro-F1** on Dogs and **85.84% / 85.48%** on Watkins. Every 693 Dogs and
1,695 Watkins train/validation/test representation is present. Qwen's frozen states
therefore outperform AVES-bio by 9.35 points on Dogs and 2.36 points on Watkins,
even though Qwen's direct generation remains near floor. The initial CPU-only
partial extraction was superseded after CUDA ONNX inference was numerically
validated against three stored CPU representations (cosine approximately 1.0;
mean absolute error about 5–6e-5).

A one-epoch Qwen-7B Thinker LoRA baseline was trained on the 1,017-example Watkins
train split (rank 8, q/v projections, 3.83M trainable parameters = 0.043%, learning
rate 2e-4, accumulation 8). The fixed 64-example validation monitor ends at loss
0.466; no test labels are used for selection. Deterministic test generation reaches
**31.56% accuracy / 23.44% macro-F1** with 8/339 invalid outputs. This is a large
gain over zero-shot generation (5.60% / 1.87%) but remains far below the frozen
linear probe (88.20% / 88.25%).

The same one-epoch rank-8 q/v protocol was run on all 415 official Dogs training
examples, with validation loss monitored on 64 official validation examples.
The locked adapter was evaluated once on all 139 official test examples and
reaches **25.18% accuracy / 12.26% macro-F1**. It predicts only Mac (101), Luke
(32), or Zoe (6). This improves the 2.88% native result but remains far below
the same-split frozen probe (92.81%) and AVES-bio probe (83.45%), even though all
three supervised methods have access to the complete official training split.

## BEANS-Zero full capped diagnostic

The complete materialized 10-second-cap manifest contains 2,950 examples from the
same 12 requested components: call-type, zf-indiv, Watkins, and the common,
scientific, and taxonomic versions of unseen species, genus, and family. Each
official `instruction_text` is used unchanged. Audio is prefix-capped at 10 s;
examples shorter than Qwen's frontend minimum are tail-zero-padded to 100 ms. This
is a declared compute protocol, not an official BEANS-Zero duration rule.

Canonicalized exact match is **13.32% (393/2,950)** overall. Qwen-7B reaches
**61.48% (174/283)** on bird song-vs-call and **66.98% (217/324)** on zebra-finch
individual counting. Watkins and every open species/genus/common/taxonomic
component score 0%; unseen-family scientific-name reaches 1.54% exact (2.31%
under a target-aware containment diagnostic). Overall target containment is
13.36%, so formatting does not explain the open-label failures. This external
zero-shot diagnostic is not an equal-supervision method comparison.

## BEANS fixed-split query-label-free KV

Support examples are selected by a seeded label-round-robin from the official
training split. Their degraded-audio labels may be used to form gradients; test
labels are never used by the mean direction, PCA subspace, router, or injected
delta. Eta is selected on the first fixed 64 validation examples, then the test
split is evaluated once. This protocol does **not** select support examples by
whether full generation was correct.

For Dogs at 1 kHz low-pass, validation selects eta=3000. The unadapted test
baseline is 2.88% accuracy / 0.56% macro-F1. Conditional KV reaches **17.27% /
9.45%** at K=5 and **11.51% / 8.49%** at K=10, versus 2.88% / 0.56% and 3.60% /
3.06% for the corresponding fixed mean directions. K=5 conditional minus
baseline is +14.39 points (paired bootstrap 95% CI [+8.63, +20.14]). K=20 falls
to 7.19% / 6.92%, so benefit is not monotonic in support size.

A later validation-only layer/rank ablation (run after the initial rank-4 test was
already observed) selects feature layer 28 and rank 8 at K=20: 10.94% validation
accuracy / 10.32% macro-F1, versus 6.25% / 4.19% for layer 0/rank 4. A single
explicitly labeled follow-up test reaches **20.86% / 18.07%**, or +17.99 points
over the paired baseline (95% CI [+11.51, +25.18]); its fixed mean is 3.60% /
2.57%. This supports layer/rank sensitivity, but it is not presented as the
original pre-registered test result because the analysis sequence is known.

For Watkins, validation selects eta=1000. Its paired 1 kHz test baseline is 6.19%
accuracy / 1.20% macro-F1. No K improves accuracy reliably: conditional results
range from 5.60% to 5.90%, and fixed results from 5.60% to 6.49%. This is an
important generalization failure rather than a tunable test-set result.

The Oracle gate is also task-dependent. Dogs has zero test examples satisfying
full-correct/lp1-wrong because all full and lp1 predictions collapse to Rudy, so
Oracle recovery is undefined. Watkins has only five eligible test examples;
eta=1 recovers 5/5, but this tiny label-using diagnostic cannot establish
deployable recovery. Its query-label-free test result above remains negative.

## Frequency diagnostics (Qwen-7B, authoritative batch size 1)

| Condition | Accuracy | Full minus condition, pp | Paired bootstrap 95% CI, pp |
|---|---:|---:|---:|
| Full observable 0–8 kHz | 29.49% | — | — |
| Low-pass 1 kHz | 12.64% | +16.85 | [+13.00, +20.70] |
| Low-pass 2 kHz | 15.20% | +14.29 | [+10.81, +17.95] |
| Low-pass 4 kHz | 19.23% | +10.26 | [+7.14, +13.55] |
| Low-pass 6 kHz | 23.08% | +6.41 | [+3.85, +9.16] |
| Low-pass 8 kHz | 28.39% | +1.10 | [-0.37, +2.56] |

No single band-removal condition (0–1, 1–2, 2–4, 4–6, or 6–8 kHz) had a paired accuracy change whose 95% CI excluded zero. The low-pass result supports cumulative spectral dependence; it does not support claiming one uniquely necessary narrow band.

## Prompt, audio, and energy controls

On a fixed balanced 300-event core, canonical-prompt accuracy is 30.33%. Reversing
the label order lowers it to 19.00% (paired -11.33 points, 95% CI [-16.33,
-6.67]) and only 43.0% of predictions agree with the canonical order. A seeded
nontrivial permutation reaches 31.33% (difference +1.00 point, CI [-3.33, +5.33])
but only 63.67% prediction agreement. Prompt order is therefore a material
measurement factor even under deterministic decoding; every main comparison fixes
one prompt and label order.

Cyclically replacing every event with an audio example from the next class lowers
accuracy against the original target to exactly balanced chance (16.67%). Scoring
the same outputs against the inserted audio's true class gives 30.33%, exactly the
canonical score because predictions follow the substituted waveform. Silence
reaches 15.00% accuracy / 5.72% macro-F1 and produces only Phee (242/300) or Seep
(58/300). Thus generation uses acoustic evidence but combines it with a strong
prompt/output prior.

Matching low-pass RMS to each event's full RMS partially improves accuracy: 1 kHz
12.33%→18.33% (paired +6.00 points, CI [+2.67, +9.33]), 2 kHz 14.00%→16.00%
(+2.00, CI [0.00, +4.33]), and 4 kHz 16.67%→19.33% (+2.67, CI [-0.33, +5.67]).
The large residual gaps to 30.33% full accuracy show that attenuation contributes
to the intervention effect but does not explain away loss of spectral structure.

On Dogs, every one of the 1,529 test-condition predictions is Rudy, so all 11
conditions remain at 2.88% accuracy / 0.56% macro-F1. On Watkins, low-pass
accuracies stay between 5.60% and 6.49%; only removing 0–1 kHz significantly lowers
accuracy relative to full (6.19% to 3.54%; full-minus-removal +2.65 points, paired
95% CI [+0.29, +5.01]). These flat curves are floor effects of a collapsed
generation head, not evidence that individual or species information is
frequency-independent: the corresponding frozen probes reach 92.81% and 88.20%.

The complete condition-matched Dogs frozen-probe curve uses all 415/139/139
official train/validation/test events at every condition. Test accuracy is
92.81% at full input and 87.77%, 91.37%, 91.37%, 89.21%, and 94.96% after 1, 2,
4, 6, and 8 kHz low-pass filtering. Layer and Ridge strength are selected only
on the matching validation condition. None of the five paired differences from
full is significant: every 95% bootstrap interval includes zero and every exact
McNemar p-value is at least .189. The 84.89--92.09 point probe-minus-native gaps
are significant under paired tests, but on Dogs they must be interpreted as an
acoustic representation plus arbitrary-name-binding disparity, not a fair
zero-shot grounding estimate.

Freezing the full-input probe instead of retraining it per frequency reveals
substantial decoder drift: Dogs transfer accuracy is only 40.29%, 59.71%, and
81.29% at 1, 2, and 4 kHz, before recovering to 92.09%/91.37% at 6/8 kHz. Thus
the 87.77% condition-specific 1-kHz score does not mean the full-band decision
boundary is invariant; it means a new linear boundary can recover the retained
identity information.
The 1-kHz condition-specific-minus-transfer difference is +47.48 points (95%
paired-bootstrap CI +38.13--+56.12; exact McNemar p=1.29e-16).

Watkins provides the cleanest three-way separation. Its condition-specific
probe is 88.20% at full input and 74.04%, 77.88%, 85.84%, 86.73%, and 84.07% at
1/2/4/6/8 kHz. The full-minus-condition losses at 1 and 2 kHz are 14.16 points
(95% CI 9.73--18.58) and 10.32 points (CI 6.49--14.45); 4 and 6 kHz are not
significantly different from full, while the 8-kHz filtered control is 4.13
points lower (CI 1.18--7.37). An unchanged full-trained probe scores only 9.73%,
24.19%, 71.09%, 82.60%, and 86.73% across those cutoffs. Condition-specific
readout therefore recovers 64.31 points over fixed transfer at 1 kHz: spectral
degradation removes some information but induces much larger representation/
decision-boundary drift (95% CI +59.29--+69.32; p=4.75e-66). Every Dogs and
Watkins class is included in the class-resolved supplemental heatmap.

The complete source-condition by target-condition decoder matrices strengthen
this interpretation. A 1-kHz-trained decoder reaches its registered matched
score (Dogs 87.77%, Watkins 74.04%) but transfers to full input at only 38.85%
and 20.35%. The converse full-to-1-kHz cells are 40.29% and 9.73%. All 12
diagonal cells exactly reproduce the original condition-specific probes, while
each off-diagonal decoder is selected on its source validation set and applied
unchanged to the paired target test set. The drift is therefore reciprocal, not
an artifact of choosing full input as the sole source condition.
A shared-hyperparameter control fixes all source decoders to the full-selected
layer and Ridge alpha. The 1-kHz source then scores 87.77%/75.52% on matched
Dogs/Watkins but only 30.22%/23.01% on full input, so layer selection does not
explain the reciprocal shift.

On MarmAudio, the five degraded-condition decoder-drift gaps and corrective-KV
rotations are perfectly rank-aligned: Spearman rho=1.00 with an exact two-sided
permutation p=.0167; Pearson r=.969 with exact p=.0167. This is a five-bandwidth
descriptive association with a shared cutoff driver, not a causal mediation
estimate.

## Oracle KV and geometry gate

For 111 samples that were correct under full input and wrong after 1 kHz low-pass, pre-RoPE K/V Oracle recovery was 12.61%, 23.42%, 53.15%, 100%, and 93.69% for eta 0.01, 0.03, 0.1, 0.3, and 1.0 respectively. Every eligible sample was recovered by at least one registered eta. This is a label-using upper bound, not a deployable method.

Across 111 audio-token-pooled gradients, rank-4 explained roughly 82–92% at representative layers, with meaningful layer/projection variation. This supports testing conditional or layer-dependent subspaces rather than asserting a universal fixed rank-4 direction.

Using the same pooled-gradient analysis on 20 label-round-robin training supports,
the median rank-4 energy is 76.99% for Dogs (layer/projection range 69.10–96.63%)
and 75.44% for Watkins (50.52–91.58%), versus 85.40% on the 111 MarmAudio
eligible failures (73.79–93.24%). Median off-diagonal cosine is only 0.026 for
Dogs, 0.208 for Watkins, and 0.238 for MarmAudio. Thus the cross-task geometry
does not justify a universal shared mean direction or a universally sufficient
rank-4 subspace.

Cosine geometry is nevertheless strongly label-structured. Across MarmAudio
layers/projections, median same-label gradient cosine is 0.734 versus -0.145 for
different labels (median gap 0.867), and nearest-cosine-neighbor label accuracy is
96.40–100% (median 100%). With two round-robin supports per dog, the corresponding
median same/different cosines are 0.776 and -0.012 (gap 0.784); nearest-neighbor
accuracy ranges from 45% to 100% across projections. Watkins K=20 has fewer
supports than its 31 labels, so same-label clustering is undefined there rather
than estimated from nonexistent pairs. This structure motivates a conditional
router but also explains why small, class-incomplete support sets are fragile.

On an exactly balanced MarmAudio K=2/class support repeated across full and five
low-pass conditions, the corrective gradient field rotates smoothly toward the
full-band field as bandwidth returns: the all-layer mean paired cosine is 0.515,
0.625, 0.723, 0.825, and 0.979 at 1, 2, 4, 6, and 8 kHz. Across those conditions,
mean layerwise same-minus-different-label cosine remains 0.774--0.826 and median
centered effective rank remains 4.03--4.67. Frequency changes the required
correction direction without eliminating its compact class structure.

## Query-label-free conditional KV

We fixed 10 recording groups as held-out queries and drew nested support sets of
K=1/5/10/20 from distinct, disjoint recordings. The router uses layer-0 frozen
degraded-audio representations to predict coefficients in a support-gradient PCA
subspace (maximum rank 4). Query labels never enter the support mean, subspace,
router, or injected delta. Because pooling changes the numerical scale relative to
the full per-token Oracle, pooled-KV results use an explicitly reported eta sweep.

On the 21 held-out full-correct/lp1-wrong diagnostic examples, at K=20 and eta=300,
fixed mean KV recovers 5/21 (23.81%), while conditional KV recovers 16/21 (76.19%).
At eta=1000 the rates are 52.38% and 76.19%. K=1 necessarily makes conditional
and fixed identical; its apparent 47.62% recovery at eta 300/1000 reflects that the
single support and 10/21 held-out targets are Tsik, so it is not evidence of routing.

The original 75-event evaluation uses query recordings drawn from recordings known
to contain an eligible failure. Its lp1 baseline is 14.67% accuracy / 6.84%
macro-F1. At K=20 and eta=300, fixed KV reaches 12.00% / 10.41%, whereas
conditional KV reaches 44.00% / 29.98%. This is retained as a recovery-enriched
diagnostic, not the primary deployable estimate.

For the primary protocol, query groups are sampled from **all 93 recordings before
checking eligibility**, and support gradients come only from disjoint remaining
recordings. Across five fixed random splits (39–82 query events each), the mean ±
sample SD accuracies are baseline **12.35 ± 4.16%**, fixed KV **13.01 ± 10.20%**,
and conditional KV **32.47 ± 7.50%**. Conditional minus baseline is **+20.12 ±
8.02 points** and conditional minus fixed is **+19.46 ± 5.15 points**; both are
positive in all five splits. Mean macro-F1 rises from 5.92% to 24.54%.

These runs establish query-label-free adaptation across several recording splits,
while also showing that fixed mean KV is unstable and that support-label imbalance
remains a limitation. Means across partially overlapping random splits are reported
without pretending their events are independent or pooling them into one CI.

## Invalidated/exploratory artifacts

`results/expert_qwen7b.csv` contains a complete throughput-oriented frequency run that mixed batch sizes. Qwen predictions were empirically not invariant to variable-length batch padding: the same event/condition changed between batch size 1 and 5. Only its original full-band batch-size-1 rows were copied into the clean run. Frequency metrics from the mixed-batch artifact must not be cited.

The 60-file `Audio_Examples.zip` smoke set is also prohibited for scientific metrics because its filename labels agree with `Annotations.tsv` for only 17/60 examples.

Historical logs from the superseded CPU-only AVES extraction remain under
`results/partial_extensions/` for provenance. They are not metric inputs; the
authoritative full fixed-split directories and summaries are listed above.
