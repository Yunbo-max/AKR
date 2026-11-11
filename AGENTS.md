# AKR: active MTP OPSD baseline branch

The active request on `research/mtp-opsd-baseline` is the user's selected transferred
baseline: sample several responses from the same pretrained MTP model, select a
self-generated reference, and distil its distributions on student trajectories.
The user explicitly requested a separate GitHub branch. Do not merge it into
`main` or rewrite historical acoustic results as part of this handoff.

## Start here

Read [LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md) and
[the Web handoff](rounds/mtp_opsd_baseline_20261111/WEB_HANDOFF.md).
The exact method is in [METHOD.md](rounds/mtp_opsd_baseline_20261111/METHOD.md).

## Execution boundary and evidence

- This delivery is `generated_unexecuted`: source review and static parsing only.
  GPU qualification, training and benchmark results remain pending.
- The Local execution agent should inspect the actual host, install the separate
  pinned environment, acquire the official pinned assets, run `check`, and pass
  `qualify` before any experiment. A passed download check is not qualification.
- No invented SSH host, GPU model, cached input, throughput or test result.
  Record the actual environment and commands. Do not use toy data or smaller
  test subsets as substitutes for the released native tasks.
- Preserve the six independent case identities and case/run/seed directories.
  Never reuse a run ID after changing source, configuration or adapters.
- Train inputs expose questions only. Gold answers are for native evaluation,
  not majority selection, teacher context, or optimization.
- Keep the native MTP head active in both teacher and student for MTP cases.
  Missing released MTP tensors, cache misalignment, or nonfinite gradients must
  fail explicitly; there is no random-head or NTP-only fallback.
- Default development budget per trained case: 512 prompts, 128 optimizer steps,
  cumulative 24-hour training budget, checked at safe operation boundaries.
  Record stops, failed/replayed cost, and zero-update runs.
- Use the full native test denominators. Keep failed and truncated predictions
  visible. Do not report a partial pass as a complete benchmark score.
- Raw weights, data, rollouts and checkpoints stay out of Git. Publish compact
  factual run records only after actual execution, and preserve raw evidence.
- No new consensus objective, latent predictor, internal-update distillation,
  external stronger teacher or additional research method is in this round.
