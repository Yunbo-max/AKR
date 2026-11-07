# Accuracy-selected AKR development follow-up

**Goal:** Continue the user's requested algorithm improvement after the first fair pilot showed LoRA25%, original AKR12.5%, and CE-selected AKR10%. Preserve that negative primary result. Test whether removing the common gradient component and selecting the actual recognition metric improves AKR.

**Scope and status:** This is an explicitly subsequent development iteration on the same previously inspected 40 development examples. Official test remains unused. No claim of independent confirmation or matched historical search effort is permitted.

- Keep the exact pinned 3B Speech LM, 40 calibration IDs, two20/20folds, LP1 preprocessing, prompt and greedy12-token decoding.
- New AKR mappings: existing residual-shrunk regression; centered regression (predicted gradient minus training-fold mean); centered class-pooled correction (predicted-class centroid minus training-fold global mean). Every final answer still comes from the Speech LM. Ridge classification is only an internal route to a gradient centroid, never the reported final answer.
- For each mapping test relative-norm strengths .03,.1,.3: nine nonzero configurations plus native.
- Re-score the same nine existing LoRA configurations (3LRs x epochs1/3/5) using their strictly verified original fold-specific adapters. Reuse completed training, not mismatched published adapters.
- For BOTH methods, choose maximum pooled OOF recognition accuracy; then macro-F1; then lower invalid rate; then native/smaller predeclared configuration order. Use only calibration-fold labels for selection. Record all 40 OOF predictions for every candidate, including native; require exact native agreement across the two method pipelines.
- Refit selected AKR on all40. Reuse the prior full-pool LoRA only if its selected LR/epoch/seed/train IDs exactly match; otherwise refit it from the same base on all40. Lock selection and adapter hashes before development generation.
- Prior native and original-AKR development predictions may be imported only after exact ID/source/hash validation; new selected AKR and LoRA predictions must be generated and journalled.
- Preserve all prior sources before instrumentation maintenance. Fix PEFT top-level forward counting and GPU budget checks using both NVIDIA total occupancy and CUDA's WSL view. These fixes must not modify model training equations or source scientific results.
- Use available GPUs for independent fold jobs, within actual global memory headroom; per-worker runtime cap90minutes. Do not allocate filler or displace other workloads.
- This follow-up is bounded to this nine-candidate grid. Report any failure to outperform LoRA candidly; do not continue searching this development set indefinitely.

## Implementation and verification

- [ ] Finish preserving and verifying the first round; fix/test instrumentation only.
- [ ] Add centered mapping / accuracy selector / source-adapter validation tests and a separate staged runner.
- [ ] Obtain independent review, then freeze source and run calibration-only candidate predictions.
- [ ] Lock/refit/evaluate both selected methods, verify cached inference and publish the second-round evidence alongside the first round.
