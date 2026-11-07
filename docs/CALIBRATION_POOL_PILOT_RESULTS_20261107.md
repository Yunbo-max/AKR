# Single calibration pool pilot: measured results

Qwen2.5-Omni-3B; Dogs, all 10 individual-dog classes; LP1; seed 20261107.
One pool of 40 labelled training recordings; 40 independent official-validation recordings for development evaluation.
**Official test excluded. One small development pilot does not establish general superiority or a fair LoRA comparison.**

| Method | Accuracy | Macro-F1 | Repaired | Harmed |
| --- | ---: | ---: | ---: | ---: |
| native | 10.0% | 3.58% | 0 | 0 |
| zero | 10.0% | 3.58% | 0 | 0 |
| fixed_relative | 10.0% | 3.44% | 0 | 0 |
| fixed_cap | 10.0% | 3.33% | 0 | 0 |
| split_guarded | 10.0% | 3.58% | 0 | 0 |
| cv_refit | 12.5% | 5.00% | 1 | 0 |
| ridge | 40.0% | 37.53% | 12 | 0 |

Repaired/harmed are paired counts relative to native on the same 40 recordings.

## Calibration-only choices

- cv_refit: alpha=0.03; heldout_ce_improved; mean CE={'0.0': 1.9941868104040623, '0.003': 1.9837416246533395, '0.01': 1.9833285637199878, '0.03': 1.9820448003709317}.
- split_guarded: alpha=0.01; heldout_ce_improved; mean CE={'0.0': 2.0411347150802612, '0.003': 2.0321798592805864, '0.01': 2.022143769264221, '0.03': 2.031286269426346}.
  Guard CE: {'native': 2.0233733654022217, 'proposed': 2.005486136674881}.

Both automatic choices were locked before development predictions. All backbone weights remained frozen.
Internal split fits on 20 examples; CV-refit uses 20/20 folds then fits all 40. No extra labelled set is introduced.

## Exploratory calibration diagnostic

The held-out gradient prediction SSE relative improvement over the training-fold mean-gradient predictor was -6.83% (negative means worse).
This uses only calibration-pool arrays. It suggests mapping generalization merits investigation; it does not establish a causal explanation of development accuracy.

## Execution and checks

- Initial full invocation: 870.7 seconds, including registration, real run and no-reload verification.
- Unique labelled examples: 40; development examples: 40.
- Shared arrays: 80 feature records and 40 gradient records.
- Answer-loss records/calls: 220/220; loss time 144.8 seconds.
- Peak allocated GPU memory during loss calls: 9.30 GiB (not total device/reserved memory).
- 29 relevant regression tests passed; independent code review completed with material findings fixed.
- Zero-alpha predictions matched native text, class, and generated-token count exactly on all 40 queries.
- Completed cache resume succeeded without loading weights. Corrupt journals/audio and stale completion state are covered by tests.
- Shared caching means the methods are label-budget matched, not compute matched.

Run directory: `results/calibration_pilot/pool_dogs_3b_20261107_v2`.
Report SHA256: `52c96c9c4adf65ceff987db6382c1a21d201b89d2e1176766909ad9f2363dd4f`.
No parameters were changed after observing development scores. Published LoRA adapters were not used as matched-data competitors.
