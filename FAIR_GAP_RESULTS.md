# Equal-supervision and causal-control results

Run window: 2026-08-13 23:50 UTC to 2026-08-14 11:50 UTC.  Model:
Qwen2.5-Omni-7B, Thinker only, BF16, greedy decoding, one RTX 3090.  This report
supersedes every earlier statement that equated zero-shot generation versus a
fully supervised probe with an "Acoustic Grounding Gap."

## Protocol decision

All few-shot comparisons use exactly the same nested K-per-class support IDs.
MarmAudio support/query recordings are disjoint. Dogs and Watkins use official
train support and official validation query. Query labels are used only for
post-hoc scoring. KV interventions are normalized by the relative Frobenius norm
of each modified layer/projection. Validation selection never uses a manifest-
order prefix, and no fair-protocol test output is produced unless the frozen
tokenwise gate passes.

## 1. What candidate scoring changes

On all 546 unanimous-expert MarmAudio events:

| Readout | Accuracy | Macro-F1 |
|---|---:|---:|
| Free generation | 29.49% | 21.94% |
| Bare candidate, mean token log-probability | 30.95% | 20.26% |
| Bare candidate, sequence sum | 33.88% | 25.36% |
| Definition candidate, mean token log-probability | 22.34% | 10.46% |
| Definition candidate, sequence sum | 18.86% | 6.58% |

Bare mean-normalized scoring never predicts Infant Cry, Trill, or Twitter.
Definitions do not repair the label interface. On the frozen balanced 75-query
subset, however, free/mean/sum scores are 41.33/42.67/49.33%. Candidate
normalization therefore explains part of the disparity and must be reported; it
does not support a universal "encoded but cannot decide" claim by itself. In
fact, the K=1 ridge score (46.67%) and zero-shot sequence-sum candidate score
(49.33%) are statistically indistinguishable on these 75 examples (paired
difference -2.67 points, CI -16.00 to +12.00). The robust failure appears when
the native model must consume labeled audio support, or when the frozen readout
gets enough support to rise well above the fixed native candidate boundary.

## 2. Equal-support frozen readouts

### MarmAudio, recording-disjoint query n=75

| K/class | Total support | Audio ICL free | Audio ICL candidate | Centroid | Ridge probe |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 8.00% | 4.00% | 29.33% | 46.67% |
| 2 | 12 | 17.33% | not run | 46.67% | 58.67% |
| 4 | 24 | context not run | context not run | 50.67% | 73.33% |
| 8 | 48 | context not run | context not run | 78.67% | 84.00% |

At K=1 the paired probe minus free-ICL difference is +38.67 points (95% paired
bootstrap CI +25.33 to +52.00; exact discordant-pair p=1.08e-6). Against
candidate-scored ICL it is +42.67 points (CI +30.67 to +54.67; p=4.07e-9).
At K=2, free ICL predicts Twitter for all 75 queries. Its 17.33% accuracy and
4.92% macro-F1 are therefore a single-class collapse rather than evidence that
the additional support was used; the same-support ridge result is 58.67%. The
paired ridge-minus-ICL difference is +41.33 points (95% paired bootstrap CI
+28.00--+54.67; exact discordant-pair p=1.23e-7).

Under a frozen bijection from the six calls to arbitrary single-token labels
A--F, zero-support candidate scoring reaches 9.33% and predicts only A/D/B
(48/19/8). One audio demonstration per class changes the support-conditioned
distribution to A/E only (48/27) but reaches just 12.00%. The paired K1-minus-K0
difference is +2.67 points (95% bootstrap CI -2.67--+8.00; exact discordant-pair
p=.625). Thus candidate restriction does not rescue arbitrary label binding;
the demonstrations alter letter priors without a measurable accuracy gain.

A six-mapping cyclic counterbalance then pairs every acoustic class once with
every output letter and moves each letter through every candidate-list position
(450 query--mapping pairs). Averaged over mappings, K0 is 9.56% and K1 is
17.33%; the query-clustered K1--K0 difference is +7.78 points (95% CI
+2.67--+12.44; sign-flip p=.004). This is not evidence of acoustic binding:
K1 is essentially six-way chance and four mappings collapse almost completely
to one letter. A shared-query permutation that preserves every mapping's output
frequency places K1's null interval at 15.33--17.78%; observed 17.33% is not
better than shuffled audio--label association (one-sided p=.152). The support
therefore repairs a bad arbitrary-letter prior toward chance without learning a
reliable query-dependent correspondence.

### BEANS Dogs lp0--1 kHz, official validation n=139

| K/class | Total support | Audio ICL free | Centroid | Ridge probe |
|---:|---:|---:|---:|---:|
| 1 | 10 | 7.19% | 24.46% | 35.25% |
| 2 | 20 | not run | 22.30% | 35.25% |

At K=1 the paired probe minus ICL difference is +28.06 points (CI +20.14 to
+36.69; p=8.65e-10). The result is an arbitrary-category support-utilization
problem, not evidence that Qwen should know dog identity names zero-shot.

### BEANS Watkins full 0--8 kHz, official validation n=339

| K/class | Total support | Centroid | Ridge probe |
|---:|---:|---:|---:|
| 1 | 31 | 26.84% | 32.15% |
| 2 | 62 | 34.22% | 42.48% |
| 4 | 124 | 48.38% | 57.52% |

The old Watkins K=20-total conditional-KV result covered fewer support examples
than labels and is withdrawn from method interpretation.

## 3. Matched-support adaptation on Dogs validation

The exact K=2/class support contains 20 training examples. A one-epoch rank-8
q/v Thinker LoRA predicts Rudy for all 139 validation examples: 2.88% accuracy,
0.56% macro-F1. The same-support ridge probe reaches 35.25%; the paired
difference is +32.37 points (CI +23.74 to +41.01; p=1.97e-11).

As a separate full-supervision baseline, the identical rank-8 q/v protocol was
trained for one epoch on all 415 official Dogs training examples. The locked
adapter was then evaluated once on all 139 official test examples: **25.18%
accuracy / 12.26% macro-F1**. Its outputs still occupy only three of ten classes
(Mac 101, Luke 32, Zoe 6). Full-train LoRA therefore improves the 2.88% native
test result, but remains far below the same official-split frozen probe (92.81%)
and AVES-bio probe (83.45%). This is no longer attributable to an unfair
zero-shot-versus-supervised comparison: both LoRA and probe use the full labeled
training split, though their optimization capacity and objectives differ.
Paired against native generation, LoRA gains +22.30 points (95% event-bootstrap
CI [+14.39, +30.22]; exact McNemar p=3.35e-7). It remains 67.63 points below the
Qwen probe (CI [-75.54, -59.71]; p=1.01e-28) and 58.27 points below AVES-bio
(CI [-67.63, -48.92]; p=8.27e-21).

Relative-norm KV results on all 139 validation examples:

| Method | alpha=.003 | alpha=.01 | alpha=.03 |
|---|---:|---:|---:|
| Fixed mean | 2.88% | 2.88% | 5.04% |
| Conditional pooled | 2.88% | 2.88% | **12.95%** |
| Conditional ordered tokenwise | 2.88% | 7.19% | 10.79% |
| Token-permuted | 2.88% | 3.60% | 3.60% |
| Matched random | 2.88% | 2.88% | 2.88% |

At alpha=.03 pooled conditional beats fixed by +7.91 points (CI +2.88 to
+13.67; p=.0074). Ordered tokenwise beats its permutation by +7.19 points (CI
+1.44 to +12.95; p=.0309), so token location has a causal effect. But ordered
tokenwise does not beat pooled (-2.16 points, CI -9.35 to +5.04; p=.70), and the
ordering advantage is not stable across alpha. The primary tokenwise validation
gate fails; no fair-protocol test run is permitted.

### Geometry-driven class-dictionary fallback

The preregistered fallback routes each unlabeled validation query with the same
K=2/class ridge readout and selects that pseudo-class's mean support-gradient
field. The router itself is 35.25% accurate. On all 139 validation examples:

| Method | alpha=.003 | alpha=.01 | alpha=.03 |
|---|---:|---:|---:|
| Class-routed pooled | 2.88% | **15.83%** | **20.86%** |
| Class-routed ordered tokenwise | **10.79%** | 13.67% | 0.00% |
| Class-routed token-permuted | 3.60% | 2.16% | 0.72% |

At alpha=.003, ordered tokenwise beats pooled by +7.91 points (95% paired
bootstrap CI +2.88--+13.67; exact discordant-pair p=.0074) and token-permuted
by +7.19 points (CI +2.88--+12.23; p=.0063). This ordering effect is real but
not stable: at alpha=.01 pooled is higher by 2.16 points, and at alpha=.03 the
ordered field produces 99.28% invalid responses. The registered gate required
ordered tokenwise to beat pooled at at least two alpha values. It therefore
fails, and no class-dictionary test evaluation is run. The pooled 20.86%
validation result is a useful partial repair, not permission to promote a
tokenwise method claim.

The registered A--J arbitrary-output replication recomputes all 20 support
gradients against the permuted single-token symbols. Native one-token generation
is 8.63% (macro-F1 2.73%). On the same 139 validation examples:

| Method | alpha=.003 | alpha=.01 | alpha=.03 |
|---|---:|---:|---:|
| Class-routed pooled | 14.39% | 18.71% | **20.14%** |
| Class-routed ordered tokenwise | **15.11%** | 6.47% | 0.00% |
| Class-routed token-permuted | 7.91% | 4.32% | 0.72% |

At alpha=.003 ordered exceeds its token permutation by +7.19 points (paired
bootstrap CI +0.72--+14.39), but the exact discordant-pair test is not
significant at .05 (p=.064). Ordered exceeds pooled by only +0.72 points at this
one alpha (CI -5.76--+7.19; p=1.0), then falls below pooled by 12.23 points at
alpha=.01. At alpha=.03 it produces 99.28% invalid output. Pooled alpha=.03
improves over native by +11.51 points descriptively, but remains below the
35.25% same-support ridge router ceiling. The arbitrary-symbol result therefore
confirms a partial class-conditioned cache effect while independently failing
the tokenwise stability gate; no test evaluation is run.

### Audio--silence steering baseline

A matched conceptual implementation of the 2026 listening intervention averages
the transferred specialist-layer audio-minus-matched-silence residual and adds
it to the final representation. It does not re-discover Omni-specific specialist
heads and is not labeled a faithful reproduction. Support-only beta calibration
selects beta=.1 on MarmAudio and improves the frozen 75-query first-token
candidate readout from 41.33% to 42.67% (+1.33 points). On Dogs, the support
defines arbitrary single-token labels A--J; beta=.3 leaves the complete 139-query
result unchanged at 5.04%. A generic listening direction therefore does not by
itself learn an arbitrary acoustic-category-to-symbol map.

## 4. Oracle capacity and its causal scope

On 12 frozen MarmAudio full-correct/lp1-wrong events:

| Intervention scope/control | Result |
|---|---|
| Full-prefill correct label, alpha=.0003 | 12/12 correct |
| Full-prefill selected wrong label, alpha=.001 | 11/12 hit wrong target; 0/12 correct |
| Full-prefill matched random | 0/12 correct at every registered alpha |
| Full-prefill, audio positions permuted only | 12/12 correct at alpha=.0003 |
| Audio-only correct label, any alpha | 5/12 ever correct |
| Audio-only token permutation, any alpha | 1/12 ever correct |
| Audio-only matched random, any alpha | 0/12 ever correct |

The full-prefill result proves label-specific decision capacity, but its
audio-token permutation control shows that prompt/decision-token gradients may
be decisive. The audio-only result is the narrower evidence for a causal
acoustic-token pathway and is not universal.

## 5. Gradient geometry

Dogs uses exactly two support examples per class. Its best combined K/V layer is
layer 22: same-label cosine 0.862, different-label cosine -0.042, gap 0.903;
global effective rank is 8.04 and between/within trace ratio is 7.68. The support
corrective field is therefore strongly class-conditioned. At layer 22 only about
15.7 of 250 audio tokens carry the effective gradient energy, and 63.95% lies in
the first temporal third.

MarmAudio failure-selected gradients also show strong separation at layer 20
(same 0.838, different -0.190, gap 1.028), but the set contains only four of six
labels and is highly imbalanced. It is mechanism evidence, not a balanced method
training set. Watkins' old 20-gradient set has one example per observed label and
cannot estimate within-class geometry.

The frequency comparison removes that imbalance by using the exact same
registered K=2/class MarmAudio support (12 examples, all six labels) at full,
1, 2, 4, 6, and 8 kHz. Averaged over all 28 Thinker layers, the paired cosine of
each condition's corrective field to the full-band field is 0.515, 0.625, 0.723,
0.825, and 0.979 as the cutoff rises from 1 to 8 kHz. Thus spectral degradation
systematically rotates the required correction. Yet mean layerwise class
separation (same-label minus different-label cosine) stays between 0.774 and
0.826, and median centered effective rank stays between 4.03 and 4.67. The
corrective field remains compact and class-conditioned while its direction is
frequency-conditioned. This supports a condition/query-specific repair target;
it does not override the failed unlabeled-query tokenwise validation gate.

### Fixed-split frequency readout

The complete BEANS Dogs official split is represented independently at all six
observable-bandwidth conditions. Each cell contains exactly 415 train, 139
validation, and 139 test events; layer and Ridge alpha are selected only on that
condition's validation split before refitting train+validation and reading test
once. Accuracy is **92.81%** at full input and **87.77%, 91.37%, 91.37%, 89.21%,
and 94.96%** at 1, 2, 4, 6, and 8 kHz low-pass. Every paired full-minus-low-pass
95% bootstrap interval includes zero (all exact McNemar p>=.189). In contrast,
native generation is 2.88% at every condition because it always emits Rudy.
Thus Dogs supplies strong evidence that individual identity remains decodable
throughout the observable band, but it does **not** support a claim that higher
baseband frequencies are required for this particular task. The native flat
curve is a label-collapse floor effect, not acoustic invariance.

A stricter transfer probe exposes a different phenomenon. Selecting layer 10
and alpha 100 on full-input validation, fitting once on full train+validation,
and applying that unchanged decoder gives **40.29%, 59.71%, 81.29%, 92.09%, and
91.37%** at 1/2/4/6/8 kHz. At 1 kHz the same-condition probe is 47.48 points
higher than full-trained transfer (87.77% versus 40.29%). Individual identity is
still present, but its linear coordinate system has shifted under low-pass input.
The paired difference is +47.48 points (95% CI +38.13--+56.12; p=1.29e-16).

Watkins shows both information loss and stronger boundary drift. Its complete
condition-specific curve is **88.20%** at full input and **74.04%, 77.88%,
85.84%, 86.73%, and 84.07%** at 1/2/4/6/8 kHz. Full-minus-low-pass losses are
significant at 1 kHz (+14.16 points, 95% CI +9.73--+18.58, p=1.18e-9), 2 kHz
(+10.32, CI +6.49--+14.45, p=6.87e-7), and 8 kHz (+4.13, CI +1.18--+7.37,
p=.0125), but not at 4 or 6 kHz. The unchanged full-trained decoder is far more
fragile: **9.73%, 24.19%, 71.09%, 82.60%, and 86.73%**. At 1 kHz it trails the
condition-specific probe by 64.31 points (95% CI +59.29--+69.32; p=4.75e-66).

A source-by-target transfer matrix makes this result symmetric and reproduces
all 12 condition-specific diagonal cells exactly. On Dogs, a 1-kHz-trained
decoder scores 87.77% on 1-kHz test audio but only 38.85% on full input; the
full-trained decoder shows the converse 92.81%/40.29% pattern. On Watkins, the
corresponding diagonal/off-condition pairs are 74.04%/20.35% for a 1-kHz-trained
decoder and 88.20%/9.73% for a full-trained decoder. The 1-kHz representation is
therefore not merely a lower-information point on a shared coordinate system:
its best source-selected linear boundary is sharply condition-specific in both
directions. Every source model is selected on source validation only and then
transferred unchanged to paired target test examples. A matched-layer control
then fixes every source to the full-selected layer/alpha: the 1-kHz-trained
decoder still scores 87.77%/75.52% on matched Dogs/Watkins but only 30.22%/23.01%
when transferred to full input. The reciprocal drift is therefore not explained
by condition-dependent layer selection.

Together with the matched-gradient result above, this separates two effects:
low-pass filtering removes some label information, while a much larger portion
remains decodable after the decision boundary and corrective KV field rotate.
Across all five registered degraded MarmAudio conditions, the
condition-specific-minus-full-trained probe gap is rank-aligned with corrective
field rotation `1-cos(g_f,g_full)` (Spearman rho=1.00, exact two-sided
permutation p=.0167; Pearson r=.969, exact p=.0167). This is descriptive
triangulation across only five bandwidths--both quantities co-vary with
cutoff--not a causal mediation estimate.
This is the paper's strongest within-observable-spectrum mechanism result. The
all-class supplemental heatmap reports every 10 Dogs and 31 Watkins test class;
no class is selected post hoc for display. The two 6x6 decoder-transfer matrices
are reported as audited supplemental figures.

## 6. External BEANS-Zero diagnostic

Qwen2.5-Omni-7B Thinker-only is evaluated on all 2,950 examples from the frozen
12-component BEANS-Zero cap-10 manifest using each official instruction,
greedy decoding, a 32-token limit, and canonical exact match:

| Component | n | Exact match |
|---|---:|---:|
| call-type | 283 | **61.48%** |
| zf-indiv | 324 | **66.98%** |
| unseen-family-sci | 130 | 1.54% |
| watkins | 115 | 0.00% |
| unseen-family-cmn / tax | 117 / 123 | 0.00% / 0.00% |
| unseen-genus-cmn / sci / tax | 265 / 262 / 258 | 0.00% / 0.00% / 0.00% |
| unseen-species-cmn / sci / tax | 373 / 364 / 336 | 0.00% / 0.00% / 0.00% |
| **Overall** | **2,950** | **13.32%** |

The 7B model is not uniformly deaf to animal audio: it performs substantially
above chance on the closed song/call and zebra-finch-count tasks. Its native
interface fails almost completely on open taxonomic/species-name binding and on
Watkins species output. This is an external zero-shot diagnostic, not an
equal-support method comparison and not a claim of competition with a
bioacoustically trained model.

As a post-hoc formatting control, a target-aware whole-phrase containment score
is 13.36% overall versus 13.32% exact match. It remains 0% for Watkins and all
open genus/species/common/taxonomic components; unseen-family-sci moves only
from 1.54% exact to 2.31% contained. Thus the near-zero open-label results are
not explained by the model merely wrapping correct names in template sentences.
Containment is diagnostic only and does not replace the registered exact score.

## 7. Invalidated or quarantined artifacts

- The original zero-shot Dogs names versus fully supervised probe comparison is
  an upper-ceiling disparity, not a grounding gap.
- The manifest-order Dogs validation prefix of 64 covered only 6/10 labels and
  is moved to `results/partial_extensions/`.
- Batch=5 changes 2/25 BF16 predictions despite matched norms; all promoted KV
  results use batch=1.
- Raw-eta tokenwise sweeps are engineering pilots. Relative alpha is the only
  promoted intervention scale.
- Oracle 111/111 at some raw eta is replaced by the frozen matched-norm 12-event
  control panel above.
- The old Watkins K=20-total result is not used to infer method failure.

## Current ICLR decision

The equal-support support-to-decision failure is real and statistically strong.
The cache has label-specific causal capacity; support gradients have class and
token structure; and pooled conditional KV yields a real validation gain over a
fixed direction. Neither the continuous tokenwise router nor the preregistered
class-dictionary fallback passes its stability gate. The honest paper is
therefore a diagnostic/causal paper with a partial pooled adaptation result, not
a finished tokenwise-repair methods paper.

The observation that multi-audio demonstrations can hurt Qwen is not itself a
novelty claim: SpeechJudge already reports this behavior for Qwen2.5-Omni-7B,
while Audio Flamingo and SICL-AT show that dedicated interleaved/post-training
can enable useful audio ICL. Context Tuning is also a direct methodological
neighbor because CT-KV optimizes a demonstration-derived KV prefix. The paper's
remaining defensible contribution is the conjunction of equal-supervision
arbitrary bioacoustic binding, causal label/token-resolved gradient geometry,
and a query-conditioned correction placed on the query's own audio state. A
faithful audio CT-KV implementation remains required before a submission can
claim a methods advantage.

Machine-readable evidence: `results/fair_gap_paired_statistics.json`,
`results/fair_gap_artifact_audit.json`, and the per-experiment summaries under
`results/`.
