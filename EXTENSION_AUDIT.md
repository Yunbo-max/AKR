# Full-extension audit

> The equal-supervision correction in `FAIR_GAP_12H_RUN.md` remains authoritative.
> Expansion resumed only after its candidate-scoring, causal-control, and frozen
> validation gates were completed. Negative gates are preserved rather than tuned
> on test.

This ledger tracks every result previously described as subset, partial, smoke,
single-task, or exploratory. A row is complete only when its authoritative
artifact, protocol, and report integration are all verified.

| Area | Earlier limitation | Required extension | Current evidence | Status |
|---|---|---|---|---|
| AVES-bio / Dogs | CPU extraction stopped after 30/693 | All official train/valid/test representations and fixed-split probe | 693/693 representations; fixed test 83.45%; metadata/finite/coverage audit passes | Complete and integrated |
| AVES-bio / Watkins | CPU extraction stopped after 128/1695 | All official train/valid/test representations and fixed-split probe | 1,695/1,695 representations; fixed test 85.84%; metadata/finite/coverage audit passes | Complete and integrated |
| BEANS-Zero | 25 examples × 12 components | Complete streamed scan of the same 12 requested components under the declared 16 kHz / 10 s cap protocol, then Qwen-7B evaluation | 2,950/2,950 predictions; 12/12 components; overall exact 13.32% | Complete and integrated |
| Thinker LoRA | Watkins only | Add Dogs under the same one-epoch official fixed-split protocol | 415 full train, 64 valid monitor, one locked 139-example test: 25.18% / 12.26% Macro-F1 | Complete and integrated |
| Thinker LoRA / MarmAudio | No fixed train/test split; expert corpus only | A same-protocol fixed-test extension is not identifiable without inventing a split; grouped OOF LoRA would answer a different question | Dataset protocol documented in `README.md` | Not applicable; explain in report |
| Conditional KV | MarmAudio, Dogs, and Watkins already evaluated, but Watkins negative | Preserve the negative result; test token-aligned improvement only after validation selection | Both continuous and registered class-routed tokenwise gates failed the frozen two-alpha rule; no test generated | Complete negative validation result |
| Frequency generation | Complete on MarmAudio, Dogs, Watkins | None | Full 11-condition Qwen-7B curves exist for all three | Complete |
| Frequency probe / Dogs | Full-state probe only | Add degraded fixed-split probe curve | All six conditions pass 693-event and 415/139/139 split audit; test accuracy 87.77--94.96%, with no significant paired difference from 92.81% full | Complete and integrated |
| Frequency probe / Watkins | Full-state probe only | Add degraded fixed-split probe curve | All six conditions pass 1,695-event and 1,017/339/339 split audit; condition-specific test accuracy is 74.04--88.20% | Complete and integrated |
| Cross-frequency decoder transfer | Condition-specific probes alone cannot distinguish information retention from boundary drift | Fit once on full train+valid and transfer unchanged to every low-pass test | Dogs 1-kHz transfer 40.29% vs 87.77% matched probe; Watkins 9.73% vs 74.04%; all six paired cells audited | Complete and integrated |
| Reciprocal decoder-transfer matrix | Full-to-low-pass transfer alone could be direction-specific | Select on each source validation, refit source train+valid, and transfer unchanged to all six paired target tests | Both 6x6 matrices reproduce every diagonal exactly; 1-kHz-to-full is 38.85% Dogs and 20.35% Watkins | Complete and integrated |
| Decoder/KV geometry alignment | Decoder drift and corrective-field rotation were reported separately | Correlate the two on all five preregistered degraded MarmAudio bandwidths with exact permutation inference | Spearman rho=1.00, exact p=.0167; explicitly limited to descriptive n=5 triangulation | Complete and integrated |
| Class-resolved frequency analysis | Aggregate accuracy could hide heterogeneous classes | Report all registered test classes without selection | Supplemental heatmap covers all 10 Dogs and 31 Watkins classes | Complete and integrated |
| KV geometry | Main geometry uses 1 kHz degradation | Add cross-frequency gradients/geometry on MarmAudio to test whether the subspace changes with evidence removal | Exact same 12 K=2/class supports at full/1/2/4/6/8 kHz; 72/72 gradients and 28-layer comparison complete | Complete and integrated |
| Token-preserving KV | Newly motivated by 2026 modality-gap literature | Validation-only comparison against pooled/broadcast KV before untouched test | Ordered field has a one-alpha token-order effect but fails stability and pooled-comparison gates in both dog-name and A--J replications | Complete negative validation result |
| Smoke/invalid files | Several engineering runs | Keep segregated and excluded from metrics | 11 explicit partial/smoke/invalid artifacts; none referenced by authoritative builders/reports | Complete; automated audit passes |

## Completion gates

1. Complete and evaluate BEANS-Zero target scan. **Passed: 2,950/2,950.**
2. Run Dogs Thinker LoRA under the Watkins protocol. **Passed: full train and one
   locked test complete.**
3. Finish tokenwise validation; only touch test if validation improves. **Passed
   procedurally: gate failed, therefore no test was produced.**
4. Add at least two MarmAudio frequency conditions to KV geometry. **Passed: six
   total matched conditions and 72 gradients.**
5. Complete Dogs/Watkins degraded fixed-split probe curves. **Passed: every
   condition contains the full official split; matched and fixed-transfer probes
   are complete.**
6. Rebuild tables/figures/reports and run both artifact audits and tests. **The
   frequency-stage artifact audit passes 30/30 and tests are 19/19; the final
   equal-support ICL stage will rerun both.**
