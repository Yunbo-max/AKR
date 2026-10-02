# Final experimental report

> **Superseded for paper claims on 2026-08-14.** This file preserves the earlier
> experiment report, but its probe-versus-generation and conditional-KV framing
> predates the equal-supervision correction. Use `FAIR_GAP_RESULTS.md` for the
> current conclusions, matched controls, failed tokenwise validation gate, and
> invalidated artifacts.

## Answer to the research question

Yes, a human/general audio-language model recognizes some animal vocalizations,
but direct text generation substantially understates what its frozen states encode.
Within Qwen2.5-Omni's observable 0–8 kHz window, call-type generation improves
monotonically as the low-pass cutoff grows, while supervised probes remain strong
even under severe degradation. Query-label-free KV adaptation can recover part of
the readout failure on MarmAudio and Dogs, but it does not generalize uniformly to
Watkins species recognition.

## Main findings

1. **7B helps when bandwidth is available.** On 546 unanimous-expert MarmAudio
   events, Qwen-3B reaches 20.33% accuracy and Qwen-7B reaches 29.49% (paired
   +9.16 points, 95% CI +4.76 to +13.37). At 1 kHz, however, 7B is worse than 3B
   by 4.58 points. Scaling gains grow from 4–8 kHz rather than making the model
   universally robust to narrow-band input.

2. **Call-type generation depends cumulatively on frequency.** Qwen-7B accuracy
   at low-pass 1/2/4/6/8 kHz is 12.64/15.20/19.23/23.08/28.39%, versus 29.49%
   full. No individual narrow-band deletion is uniquely necessary on MarmAudio.
   Dogs and Watkins generation curves are mostly floor-limited; only removing
   0–1 kHz significantly hurts Watkins.

3. **The information/readout gap is large.** Qwen-7B's grouped frozen probe reaches
   95.42% on full MarmAudio, 92.81% on Dogs, and 88.20% on Watkins, versus direct
   generation at 29.49%, 2.88%, and about 6%. At low-pass 1/2/4/6/8 kHz, the
   condition-specific MarmAudio probes still reach 77.29/79.85/82.60/93.96/94.87%.
   A probe trained only on full states transfers less well, demonstrating both
   surviving class information and substantial hidden-state distribution shift.

4. **KV gradients are structured but task-dependent.** MarmAudio rank-4 explained
   gradient energy has median 85.40%; Dogs and Watkins support gradients have
   medians 76.99% and 75.44%. MarmAudio same-label gradients have median cosine
   0.734 versus -0.145 for different labels. This favors conditional routing, not
   a universal mean direction or universal rank.

5. **Oracle recovery is an upper bound, not the method.** On 111 MarmAudio samples
   that are full-correct/lp1-wrong, Oracle pre-RoPE K/V updates recover 100% at the
   best registered eta. Watkins also recovers 5/5, but that eligible set is too
   small; Dogs has no eligible test examples because its predictions collapse.

6. **Label-free conditional KV works selectively.** Across five MarmAudio random
   recording-group splits, mean degraded/fixed/conditional accuracies are
   12.35/13.01/32.47%. On Dogs fixed test, a validation-selected layer-28/rank-8
   follow-up reaches 20.86% versus 2.88% baseline and 3.60% fixed KV (paired
   gain +17.99 points, CI +11.51 to +25.18). Watkins conditional KV remains at
   5.60%, so recovery does not automatically transfer to 31-way species naming.

7. **Supervised baselines define the remaining gap.** AVES-bio reaches 93.41% on
   the same MarmAudio grouped OOF protocol; Qwen's own probe reaches 95.42%.
   On official fixed tests, AVES-bio reaches 83.45% on Dogs and 85.84% on
   Watkins, versus Qwen frozen-probe results of 92.81% and 88.20%.
   Watkins one-epoch Thinker LoRA reaches 31.56%, above zero-shot but far below the
   88.20% frozen probe. The official MarmAudio full-spectrum classifier reaches
   88.64% on original 0–48 kHz audio, a reference with a different accessible
   spectrum.

8. **Prompt and energy controls matter.** Reversing label order lowers balanced-core
   accuracy from 30.33% to 19.00%; shuffled audio predictions follow the inserted
   waveform, while silence collapses mostly to Phee. RMS matching partially
   restores low-pass accuracy but leaves a large gap to full, so attenuation and
   spectral structure both contribute.

9. **BEANS-Zero remains hard for taxonomy.** On the declared 300-example streamed
   core, exact match is 11.0% overall: 60% song-vs-call, 72% zebra-finch count,
   and 0% on sampled Watkins and unseen species/genus/family name variants. This
   is a compute-bounded screening subset, not the full 91,965-example benchmark.

## Deliverables

- Main benchmark table: `results/paper_tables/table_main_benchmarks.csv`
- KV table: `results/paper_tables/table_kv_adaptation.csv`
- BEANS-Zero table: `results/paper_tables/table_beans_zero_core.csv`
- Fig. 1, frequency and band removal: `figures/fig1_frequency_recognition.png`
- Fig. 2, gradient SVD and cosine geometry: `figures/fig2_frequency_error_kv_geometry.png`
- Fig. 3, KV recovery and baselines: `figures/fig3_kv_recovery.png`
- Reproduction commands: `REPRODUCE.md`
- Detailed authoritative/invalidated ledger: `RESULTS.md`
- Machine-readable integrity audit: `results/artifact_audit.json`

## Scope limits

“Full” for Qwen always means its complete processor-visible 0–8 kHz baseband, not
the source recording's full spectrum. Oracle uses query labels and is never a
deployable result. MarmAudio supervised scores are recording-grouped OOF rather
than an external fixed test. The Dogs layer/rank result is transparently labeled
as a validation-selected follow-up after the initial rank-4 test had been seen.
BEANS-Zero in this frozen report is a balanced core subset; a larger requested
component scan is tracked separately in `EXTENSION_AUDIT.md`. The earlier partial
AVES extension has since been completed on the full Dogs and Watkins fixed splits;
their promoted summaries supersede the historical logs in
`results/partial_extensions/`.
