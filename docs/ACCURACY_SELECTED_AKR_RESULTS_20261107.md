# Accuracy-selected AKR development experiment

This is a second development iteration on the same previously inspected 40 Dogs validation clips. It is not an independent confirmation. The official test set remains unused.

Qwen2.5-Omni-3B generates every final answer. Both methods use the same 40 labelled calibration clips, two 20/20 folds, prompt, class normalizer and greedy 12-token decoding. Each searches nine adaptation configurations plus native. Selection uses pooled calibration-fold accuracy, then macro-F1, invalid rate and a predeclared simpler-candidate order. Selected models refit on all 40 calibration clips before development evaluation.

| Method | Development accuracy | Macro-F1 | Repaired / harmed vs native |
| --- | ---: | ---: | ---: |
| native | 10.0% | 3.58% | 0 / 0 |
| akr_reference | 12.5% | 5.00% | 1 / 0 |
| akr | 25.0% | 15.83% | 6 / 0 |
| lora selection policy (native fallback; no adapter) | 10.0% | 3.58% | 0 / 0 |

## Locked calibration choices

- akr: `centered_class_a0.3`, pooled calibration-fold accuracy 22.5%, macro-F1 11.07%.
- lora: `native`, pooled calibration-fold accuracy 17.5%, macro-F1 6.90%.

AKR compares residual-shrunk, centered-regression and centered-class gradient directions, each at relative K/V correction strength 0.03, 0.1 and 0.3. Centering subtracts the training-fold mean gradient. The class variant uses an internal ridge readout to choose a calibration gradient centroid; final answers still come from the Speech LM.

LoRA reuses strictly verified calibration-fold adapters trained at learning rates 5e-5, 2e-4 and 5e-4 for 1, 3 and 5 epochs. It adapts only language-decoder q/v projections (rank 8, alpha 16, dropout 0.05); the audio encoder, vision encoder and base weights remain frozen.
Full-pool LoRA refit: `native`.

## Parameter sensitivity

| AKR mapping | Accuracy at alpha 0.03 | At 0.1 | At 0.3 | Range |
| --- | ---: | ---: | ---: | ---: |
| shrunk | 12.5% | 7.5% | 7.5% | 5.0% |
| centered_regression | 12.5% | 5.0% | 10.0% | 7.5% |
| centered_class | 17.5% | 20.0% | 22.5% | 5.0% |

These are calibration-fold recognition accuracies, not development scores. They describe this small pool and this grid only.
The earlier answer-CE sweep found an alpha effect range of 0.178137 for 0.01/0.03/0.1. At fixed alpha 0.03, rank 4→16 changed CE by -0.018211; ridge 10→3000 changed CE by +0.006316; changing both gave -0.007506. Lower CE is better. Gradient-prediction SSE improved under stronger ridge, but that did not imply better Speech LM CE or recognition.

Prioritize correction strength and direction construction in a future registered sweep; check rank and ridge jointly using held-out recognition. Layer positions, feature layer and calibration size were held fixed and their sensitivity remains unknown. This experiment does not establish a globally most sensitive parameter.

## Prior result and limits

| First iteration, CE-selected | Development accuracy |
| --- | ---: |
| native | 10.0% |
| akr_reference | 12.5% |
| akr_selected | 10.0% |
| lora | 25.0% |

**The second-round LoRA policy selected native, so its development row is not an adapted LoRA model. The first-round CE-selected LoRA result remains the adapted comparator; do not claim AKR beats LoRA by comparing only against this native fallback.**

The first negative AKR result is preserved. This second iteration was designed after inspecting it. Label budget and current candidate counts are matched; historical search effort, compute and adaptable parameter counts are not matched. One model, one dataset, one seed and 40 inspected development examples cannot establish superiority across models or datasets. LoRA at the grid boundary is not proven converged.

## Verification and GPU use

- The current calibration/fair-comparison/centered-direction targeted test run passed 18 tests; TARGETED_TESTS.log preserves the output. The preceding instrumentation maintenance suite passed 39 tests (overlapping coverage; do not add the counts).
- All selected inference records were checked through the cache-only report path, with model loading prohibited. Native calibration predictions agree exactly between the AKR and LoRA pipelines. Input, source, fold and adapter identities are checksummed.
- Sampled NVIDIA total-device peaks, including other workloads: GPU 0 78.42 GiB (84.20 GB); GPU 1 52.96 GiB (56.87 GB). The scheduling target was around 90 GiB; this is not a constant-occupancy or per-job-memory claim.
- Report SHA256: `b2c23a6ef7bf199bf9c1dac3e9c3870e59cd45666140a2268b7b6d32666e4711`.
