# Working ICLR idea: Spectral Decision Drift and Equal-Supervision Cache Repair

## Working title and current decision

**It Encoded the Sound, But the Boundary Moved: Spectral Decision Drift in
Generalist Audio-Language Models**

Alternative method-forward title: **Acoustic Cache Repair: Support-Gradient KV
Fields for Bioacoustic Distribution Shift**.

The first title is the currently supported diagnostic claim. The 12-hour fair
protocol validates a support-to-decision/binding failure, strong spectral
decision-boundary drift, and a partial pooled-cache intervention, but both
preregistered tokenwise gates fail.
An ICLR submission should use the diagnostic framing unless a new method beats
pooled KV under the frozen gate and then transfers across prompts. A methods
paper is not yet justified by the current validation results.

## The gap

The original probe-versus-generation observation is a discovery signal, not by
itself a fair gap: a supervised probe learns a label map while zero-shot Qwen
does not.  This is especially decisive for arbitrary identities such as dog
names.  The paper therefore asks a narrower question: under exactly the same
labeled K-shot support, can frozen Qwen states support a good acoustic category
readout while Qwen's native candidate decision remains unstable? Existing work
attacks adjacent, but different, problems:

1. [NatureLM-audio](https://openreview.net/forum?id=hJVdwBpWjt) trains a
   bioacoustic audio-language model and introduces BEANS-Zero. It establishes
   that domain training works, but does not diagnose whether a frozen generalist
   model fails at acoustic encoding or at converting an encoded animal signal
   into a textual decision.
2. Speech-language "modality gap" studies compare semantically equivalent speech
   and transcripts. The 2026
   [Anatomy of the Modality Gap](https://arxiv.org/abs/2603.01502) specifically
   points to late, token-level condensation as a bottleneck. Its simple KV token
   merging improves only 0.1–0.5 points, and its limitations call for practical,
   task-selective remedies beyond token count. Animal calls have no equivalent
   transcript, so text alignment is unavailable; our setting tests whether
   label-gradient fields can supply the missing selective sharpening signal.
3. [Are Audio-Language Models Listening?](https://arxiv.org/abs/2603.06854)
   identifies audio-specialist heads and uses an audio-versus-silence residual
   direction to improve two Qwen-based Audio-LMs by up to 8 points on MMAU. It
   establishes that decisive audio evidence can be under-used and makes
   audio--silence steering a mandatory baseline. It does not learn a supervised
   corrective gradient field for arbitrary bioacoustic categories.
4. [Beyond the Baseband](https://arxiv.org/abs/2604.27936) addresses information
   above a 16 kHz model's Nyquist limit through multi-band encoding. It does not
   ask why a model fails when discriminative evidence is already present inside
   its observable 0–8 kHz baseband.
5. [KV Cache Steering](https://arxiv.org/abs/2507.08799) inserts fixed
   contrastive text-derived directions to induce a global reasoning style.
   Conditional activation methods such as
   [CAST](https://arxiv.org/abs/2409.05907) gate a fixed behavior intervention.
   [SADI](https://arxiv.org/abs/2410.12299) and CAST already make activation
   steering conditional or semantics-adaptive. Thus neither "KV steering" nor
   "conditional steering" is our novelty. The missing object is a supervised
   **corrective gradient field** learned from audio-token states and applied to
   an unlabeled query at the KV causal interface.
6. [Audio Flamingo](https://proceedings.mlr.press/v235/kong24a.html) is trained
   on interleaved audio--text episodes and demonstrates audio in-context
   learning and retrieval. The recent
   [SICL-AT](https://openreview.net/forum?id=QBSfyizMCE) likewise post-trains
   auditory LLMs in a demonstration-conditioned speech format and reports that
   vanilla ICL can improve conventional speech/audio tasks for selected models.
   These works establish that audio ICL is possible and that post-training can
   strengthen it; we must not claim otherwise. Our target is the failure mode
   of an otherwise frozen generalist model on domain-external, support-defined
   acoustic categories and a support-only state intervention rather than ICL
   post-training.
7. Most directly, ICLR 2026
   [SpeechJudge](https://openreview.net/forum?id=I9ED9VWZq6) reports that
   Qwen2.5-Omni-7B falls from 60.6% zero-shot accuracy to 48--53% after 2--16
   multi-audio demonstrations on a speech-naturalness judgment task, and
   attributes the drop to limited multi-audio association and long-context
   handling. Therefore the observation that demonstrations can hurt Qwen is
   already known. Our defensible gap is narrower: no prior result above
   separates acoustic decodability, arbitrary audio-to-symbol binding, and
   label-conditioned causal KV geometry under identical support, then learns a
   query-specific corrective field without query labels or model updates.
8. The closest adaptation method is ICML 2026
   [Context Tuning](https://arxiv.org/abs/2507.04221). Its CT-KV variant turns
   the KV prefix produced by text demonstrations into trainable memory and
   optimizes it with leave-one-out masking. Consequently, neither "support-
   supervised KV optimization" nor "weight-free few-shot adaptation" is new.
   CT-KV learns one query-agnostic demonstration prefix and retains the example
   context. Our remaining methodological hypothesis is different and strictly
   testable: compress domain-external audio supports into label-discriminative
   correction fields, route an unseen query from its audio states, and place a
   sparse ordered field at the query's own audio tokens. A publication-quality
   comparison must include a faithful multimodal CT-KV adaptation, not merely a
   fixed-vector cache-steering baseline.
9. Few-shot audio classification itself is crowded. Recent methods such as
   [MUKA](https://arxiv.org/abs/2602.14127),
   [Subspace Tuning](https://arxiv.org/abs/2606.18560), and
   [Acoustic Prompting](https://arxiv.org/abs/2606.15751) adapt contrastive or
   encoder-style audio--language representations efficiently. They are strong
   category-recognition baselines, but they do not repair the native text
   decision of a generative Omni model. The paper therefore needs both axes:
   specialist/readout accuracy and preservation of the model-native generative
   interface, eventually tested by cross-prompt transfer or downstream
   reasoning.

The concrete missing question is therefore:

> Under matched K-shot supervision, when do frozen Audio-LM states make an
> arbitrary acoustic category decodable while a frozen model fails to bind the
> same support to its native label decision; when spectral shift preserves
> decodability but rotates both the readout boundary and corrective KV field;
> and can a support-derived, condition-aware field repair an unlabeled query
> without changing model weights?

## Empirical wedge: corrected status

The following full-supervision numbers motivated the investigation but are not
an equal-supervision gap and must not be used as the headline causal result:

| Task | Direct generation | Frozen Qwen probe | AVES-bio probe |
|---|---:|---:|---:|
| Dogs, 10 individuals | 2.88% | 92.81% | 83.45% |
| Watkins, 31 species | 6.19% | 88.20% | 85.84% |

On the 546-example expert-reviewed MarmAudio set, free generation is 29.49% and
the fully supervised recording-group OOF probe is 95.42%. We now call this an
**upper-ceiling disparity**, not an Acoustic Grounding Gap.

The repaired protocol has already yielded the first fair evidence:

| Setting | Same support | Native/readout result |
|---|---:|---:|
| MarmAudio bare native candidate likelihood, all 546 | zero-shot | 30.95% |
| MarmAudio candidate likelihood + acoustic definitions | zero-shot | 22.34% |
| MarmAudio free generation, frozen balanced query | zero-shot | 41.33% |
| MarmAudio bare candidate mean / sequence-sum, same query | zero-shot | 42.67% / 49.33% |
| MarmAudio audio ICL, recording-disjoint, K=1/class | 6 examples | 8.00% |
| MarmAudio candidate-scored audio ICL, same K=1/class | 6 examples | 4.00% |
| MarmAudio free audio ICL, K=2/class | 12 examples | 17.33% (all Twitter) |
| MarmAudio arbitrary A--F candidate, K=0 / K=1 per class | 0 / 6 examples | 9.33% / 12.00% |
| MarmAudio cosine centroid, same K=1/class | 6 examples | 29.33% |
| MarmAudio final-layer ridge probe, same K=1/class | 6 examples | 46.67% |
| Dogs centroid, official valid, K=2/class | 20 examples | 22.30% |
| Dogs final-layer ridge probe, same K=2/class | 20 examples | 35.25% |
| Watkins ridge probe, K=1/2/4 per class | 31/62/124 examples | 32.15/42.48/57.52% |
| Dogs full-train Thinker LoRA, official test | 415 examples | 25.18% (only 3/10 output classes) |
| BEANS-Zero call-type / zf-indiv, zero-shot | none | 61.48% / 66.98% exact match |
| BEANS-Zero Watkins / open taxonomy, zero-shot | none | 0% / approximately 0% exact match |

On all 546 examples, length-normalized candidate scoring remains near free
generation and never selects three of six labels; definitions make it worse. On
the frozen balanced query, however, sequence-sum scoring reaches 49.33%, so
decoding and candidate normalization explain a nonzero part of the disparity.
It is statistically indistinguishable from the K=1 ridge score of 46.67%
(paired difference -2.67 points, CI -16.00--12.00), which rules out a blanket
K=1 native-decision gap. Candidate scoring does not explain the
support-utilization failure: K=1 audio ICL reaches 8.00%
with free decoding and 4.00% with candidate scoring, while the exact same six
supports yield a 46.67% ridge readout. The paired probe--ICL difference is 38.67
points (95% paired-bootstrap CI 25.33--52.00). Dogs shows the same pattern:
35.25% K=1 ridge versus 7.19% K=1 audio ICL, a 28.06-point difference (CI
20.14--36.69). Thus the current evidence supports a **support-to-decision
interface failure**, not a blanket claim that every native readout is below a
few-shot probe.

The arbitrary-label control strengthens the interpretation without overclaiming
it: MarmAudio A--F candidate accuracy is 9.33% without support and 12.00% with
one audio example per class (paired +2.67 points, CI -2.67--8.00; p=.625).
Support changes the model's output prior but does not measurably establish the
registered audio-to-symbol mapping. At K=2, free ICL reaches 17.33% only because
all 75 queries collapse to Twitter, whereas the same-support ridge reaches
58.67% (paired gap +41.33 points, CI 28.00--54.67; p=1.23e-7).

The failure is not confined to tiny support. A one-epoch q/v Thinker LoRA trained
on all 415 official Dogs training examples improves test generation from 2.88%
to 25.18%, but emits only Mac/Luke/Zoe and remains 67.63 points below the
same-split full-train frozen probe. LoRA and probe are not capacity-matched, so
this is a baseline rather than a causal theorem; it nevertheless shows that the
native label interface is not trivially repaired by the paper's standard
parameter-efficient fine-tuning baseline.

The full 2,950-example BEANS-Zero 7B screen sharpens the scope. Native exact
match is 61.48% on call-type and 66.98% on zebra-finch individual-count, but 0%
on Watkins and every open genus/species component; only unseen-family-sci is
nonzero at 1.54%. Overall exact match is 13.32%. The model can use animal audio
for closed decisions, while open bioacoustic name/taxonomy binding collapses.
This supports a category-binding/interface question rather than a blanket
claim that the generalist audio encoder contains no animal information.
The target-aware whole-phrase containment diagnostic is only 13.36% overall and
remains 0% on Watkins and every open genus/species component, ruling out template
wrapping as the main explanation for their zero exact-match scores.

This conclusion survives exact output-symbol counterbalancing. Across all six
cyclic A--F mappings (450 query--mapping pairs), K1 candidate accuracy rises
from a biased K0 9.56% to 17.33% (+7.78 points, query-cluster CI
2.67--12.44). But K1 is only at six-way chance and is not above a
frequency-preserving audio--label permutation null (null 15.33--17.78%,
one-sided p=.152). Demonstrations recalibrate the letter prior; they do not
show reliable query-dependent acoustic binding.

A high probe score alone is not causal evidence. Under relative-norm controls on
12 frozen MarmAudio spectral failures, full-prefill correct-label gradients
recover 12/12 at alpha 0.0003, wrong-label gradients drive 11/12 toward the chosen
wrong target at alpha 0.001, and matched random fields recover 0/12. But permuting
only the audio-token part also recovers 12/12: the decisive full-prefill capacity
may reside in prompt/decision-token gradients. Restricting the intervention to
audio tokens recovers only 5/12 at any alpha, versus 1/12 for token permutation
and 0/12 random. Oracle therefore proves label-specific **causal decision
capacity**, not universal or purely acoustic repair.

Matched cross-frequency geometry now uses the exact same balanced 12 supports at
full/1/2/4/6/8 kHz. The all-layer mean paired cosine to the full corrective
field rises 0.515→0.625→0.723→0.825→0.979 with cutoff, while mean class
separation remains 0.774--0.826 and median centered effective rank remains
4.03--4.67. Frequency removal rotates the required correction without erasing
its compact label structure. This is a stronger mechanistic reason for a
condition-specific router than the old imbalanced 1-kHz-only spectrum.

The full official Dogs probe curve provides a useful negative frequency result.
Condition-specific probes score 92.81% at full input and 87.77--94.96% across
1/2/4/6/8 kHz low-pass inputs; no paired difference from full is significant.
Native generation remains 2.88% at every cutoff because it emits Rudy for every
event. Thus this task supports robust within-baseband acoustic decodability and
arbitrary-name binding failure, but not the hypothesized dependence of individual
identity on upper-baseband detail. Frequency is a diagnostic moderator, not a
claim that every animal task degrades monotonically.

The fixed-decoder transfer control changes the mechanistic interpretation in an
important way. A full-trained probe transferred unchanged to 1 kHz falls from
92.81% to 40.29% on Dogs, even though a 1-kHz-trained probe reaches 87.77%. On
Watkins it falls from 88.20% to 9.73%, while a 1-kHz-trained probe reaches
74.04%. The corresponding 1-kHz corrective gradient field has only 0.515 mean
cosine to its full-input counterpart but retains strong label separation. This
triangulates a **spectral decision-boundary drift**: much of the class evidence
survives, but both the linear readout and the causal cache correction required
to use it rotate with the observed spectrum. A frequency-aware router is thus a
response to a measured mechanism, not merely another conditional-steering
variant.

Both matched-versus-transfer gaps are decisive under paired tests: +47.48 points
on Dogs (95% CI +38.13--+56.12; p=1.29e-16) and +64.31 on Watkins (CI
+59.29--+69.32; p=4.75e-66).

The full 6x6 source-by-target transfer matrices rule out a one-directional
artifact. A decoder trained at 1 kHz recovers its matched condition (Dogs
87.77%, Watkins 74.04%) yet transfers back to full input at only 38.85% and
20.35%; conversely, the full-trained boundaries score only 40.29% and 9.73% on
1-kHz targets. All diagonal cells exactly reproduce the registered
condition-specific probes, and every off-diagonal transfer uses no target label
for fit or selection. This reciprocal failure is the cleanest evidence that the
spectral intervention moves the representation boundary rather than merely
reducing class signal along a fixed axis.
Fixing every source to the full-selected representation layer and Ridge alpha
strengthens the control: the 1-kHz source still reaches 87.77% on matched Dogs
and 75.52% on matched Watkins, but only 30.22% and 23.01% on full targets. Thus
condition-dependent layer selection cannot account for the reciprocal drift.

The decoder and cache geometries also move together across the five degraded
MarmAudio bandwidths. The matched-probe minus full-trained-transfer gap is
perfectly rank-aligned with `1-cos(g_f,g_full)` (Spearman rho=1.00, exact
permutation p=.0167; Pearson r=.969, exact p=.0167). Because `n=5` and bandwidth
is a common driver, this is mechanism triangulation rather than a mediation or
causality claim; the paper must state that limitation next to the plot.

## Proposed method: Conditional Tokenwise Acoustic Cache Repair

For labeled support examples `(x_i, y_i)`, freeze every model parameter and
differentiate the answer loss with respect to pre-RoPE key/value states at each
audio token:

`G_i = -d L(y_i | x_i) / d (K_audio, V_audio)`.

Instead of averaging over time, retain the ordered gradient field. Factor the
centered support fields into a low-rank basis `U`, then learn a small ridge router
from a frozen audio representation to field coefficients:

`a_i = U^T (G_i - mean(G))`,

`a_hat(x) = R phi(h_x)`,

`Delta KV(x) = mean(G) + U a_hat(x)`.

At query time, no query label, gradient, optimizer step, or weight update is used.
The predicted field is placed once over the prefilling audio-token KV states and
ordinary greedy decoding follows. This differs from fixed cache steering in three
testable ways: it is input-conditional, label-discriminative, and preserves the
temporal audio-token field.

Mechanistically, the hypothesis is not simply that there are too many audio
tokens. The predicted field should selectively sharpen label-relevant token-to-
decision pathways. A decisive analysis will therefore compare decision-token
attention concentration and label-logit margins before and after repair, rather
than treating accuracy gain alone as an explanation.

## Falsifiable claims

1. **An equal-supervision decision gap exists.** With identical K/class support,
   native candidate/ICL decisions remain below frozen-state readouts on at least
   MarmAudio and Dogs. Dogs is framed as arbitrary few-shot acoustic grounding,
   never as zero-shot knowledge of individual names.
2. **The cache has causal decision capacity.** Correct-label Oracle gradients
   can recover failures, but correct directions must beat wrong-label,
   token-permuted, and matched-random controls at the same intervention norm.
   This upper bound is not itself evidence that the query-label-free gap is
   repairable.
3. **Repairs have structured geometry.** Support gradients are low-rank within
   layers and more aligned within labels than between labels; structure changes
   with removed frequency evidence.
4. **Conditioning is essential.** Conditional repair must beat the same support
   field averaged into a fixed direction, published-style fixed cache steering,
   and pooled conditional KV.
5. **Token resolution is essential.** A token-preserving field must beat the
   current mean-pool-and-broadcast variant on held-out validation and untouched
   test data.
6. **Few-shot state adaptation is competitive.** With `K_class in {1,2,4,8}`, repair
   should recover a meaningful fraction of the probe–generation gap without the
   parameter updates and larger training set used by Thinker LoRA.

Current validation status is mixed. On all 139 Dogs validation examples at the
same alpha 0.03, conditional pooled KV reaches 12.95% versus 5.04% fixed mean
(paired +7.91 points, CI +2.88--13.67), while ordered tokenwise reaches 10.79%
versus 3.60% after token permutation (+7.19, CI +1.44--12.95). Ordered tokenwise
does **not** beat pooled (-2.16 points, CI -9.35--5.04) and the advantage is not
stable across alpha. The preregistered test gate therefore fails and no new test
run is allowed. The separately preregistered class-dictionary fallback improves
pooled KV to 20.86% at alpha=.03. Ordered tokenwise is significantly better
than pooled at alpha=.003 (10.79% versus 2.88%) but not at alpha=.01 and
collapses to 0% with 99.28% invalid output at alpha=.03. It therefore also fails
the two-alpha stability gate. The current paper must be framed as
diagnostic/causal with partial pooled repair, rather than as a successful
tokenwise method paper; no class-dictionary test run is permitted.

The independently recomputed A--J Dogs replication reaches 8.63% natively and
20.14% with class-routed pooled KV at alpha=.03. Ordered tokenwise is 15.11% at
alpha=.003 but only 0.72 points above pooled (p=1.0), then degrades to 6.47% and
0% as the norm increases; its invalid rate reaches 99.28%. Arbitrary outputs
therefore do not remove the stability failure. They support only a partial
class-conditioned pooled-cache effect, not the proposed tokenwise claim.

The matched audio--silence baseline is deliberately labeled conceptual rather
than a faithful reproduction because it transfers the published specialist-layer
set without re-discovering Omni-specific heads. Support-calibrated steering moves
the MarmAudio 75-query first-token readout from 41.33% to 42.67%. Under Dogs'
arbitrary A--J mapping it leaves all 139 queries unchanged at 5.04%. This is
consistent with the distinction between a generic listening direction and a
label-supervised corrective field, but it is not evidence that the published
method fails under its original model and MMAU protocol.

## ICLR experiment package

### Main tasks

- Species: BEANS Watkins, official train/valid/test.
- Individual: BEANS Dogs, official train/valid/test.
- Call type: MarmAudio six expert-reviewed call types, recording-group OOF.
- General zero-shot screen: BEANS-Zero is an external diagnostic only; its
  complete capped scan contains 2,950 examples over all 12 registered components
  and remains lower priority than causal/fair protocol evidence.

### Required comparisons

- Direct Qwen-3B and Qwen-7B generation.
- Frozen Qwen linear probe and animal-specialist AVES-bio probe.
- Thinker LoRA.
- Native constrained candidate scoring and multi-audio ICL.
- Audio--silence activation steering from the 2026 listening work.
- Fixed pooled KV, conditional pooled KV, fixed tokenwise KV, conditional
  tokenwise KV.
- A faithful fixed contrastive/cache-steering baseline where task semantics make
  it definable.
- A faithful CT-KV/Context-Tuning baseline initialized from the same audio
  demonstrations and optimized with the same K/class support budget.
- Full per-example Oracle-KV upper bound.

### Critical ablations

- `K_class = 1, 2, 4, 8`; rank `1, 2, 4, 8, 16`; layer and relative intervention
  norm selected only on validation.
- Keys only, values only, and K+V.
- Pooled versus tokenwise repair under identical support examples.
- Full-band, low-pass sweep, and band removal; cross-condition transfer.
- Correctly answered versus failed examples, and probe-high versus probe-low
  strata.
- Query-label audit proving labels are used only for post-hoc scoring.
- Arbitrary label permutation (`Phee -> Lantern`, dog identity -> permuted class
  number) under support supervision.
- Ordered token field versus token permutation, token averaging, and matched
  random field at identical relative KV norm.
- Cross-prompt transfer and a small recognition-to-reasoning demonstration if,
  and only if, query-label-free repair passes validation.
- Runtime, extra parameters, support storage, and one-shot intervention cost.

## Three-figure story

1. **Encode vs decide:** equal-support native candidate/ICL, centroid, and probe
   curves against K/class; frequency is a diagnostic overlay rather than the
   main method.
2. **Why:** layer-by-frequency gradient spectra, same-label versus different-label
   cosine geometry, and Oracle causal recovery.
3. **Repair:** degraded baseline < fixed cache steering < pooled conditional KV <
   tokenwise conditional KV < Oracle, with LoRA and AVES as external references.

## Reviewer-risk checklist

- Never claim that linear decodability alone means the model uses the feature;
  pair it with causal cache intervention.
- Never compare zero-shot generation to a fully supervised probe as the main
  gap, and never interpret zero-shot dog-name failure as acoustic failure.
- Never claim to invent KV steering; cite fixed cache steering, conditional
  activation steering, and speech KV work explicitly.
- Keep `0–8 kHz full` distinct from an animal recording's original full spectrum.
- Do not sell high-frequency remapping as the contribution.
- Keep validation selection and untouched test reporting explicit.
- Treat Watkins failure as a boundary condition rather than hiding it.
