# Comprehensive AKR Readiness Implementation Plan

> **For agentic workers:** Use the parallel-agent workflow for the independent model backends; execute common planning and orchestration inline. Follow verification-before-completion.

**Goal:** Make the downloaded model families, registered datasets, AKR methods and applicable specialist/adaptation baselines executable in WSL with honest coverage and completion reports.

**Architecture:** Preserve the original rebuttal protocol and add a comprehensive configuration and launcher. Reuse the existing intervention and journal interfaces through a model backend factory. Build GLM and Qwen3 adapters now, but require explicit conditional activation for their experiments. Preserve the original BEANS-Zero diagnostic and add an explicitly derived supervised comparison, as subsequently authorized by the user.

**Tech Stack:** Python, PyTorch, Hugging Face, ONNX Runtime, YAML, pytest, WSL. Existing Qwen environment remains /root/.venvs/akr; optional families can use separate compatible environments.

**Spec:** User requested full runnable preparation across downloaded models/datasets and confirmed "Build now; run only if needed" for GLM/Qwen3. Preparation and bounded smoke checks are authorized; full research experiment execution is a later command.

## Constraints and decisions

- Work in C:/ai/akr on feat/comprehensive-ready, preserving existing local edits and asset locations.
- Main comparisons use both Qwen2.5 sizes on MarmAudio, Dogs and Watkins with the existing fit/calibration/guard/query identities and all common methods. Controls explicitly configured.
- Optional GLM and Qwen3 backends implement features, support gradients, inference intervention and calibration loss, with immutable checkpoint revisions and family-specific audio masks.
- Original BEANS-Zero uses its exact 2,950 registered IDs and original instruction strings. User subsequently approved a separately labelled supervised re-evaluation: per-component canonical recording groups, one per class for fit/calibration/guard, remaining eligible groups query, rank 1. This must never be presented as the original zero-shot benchmark.
- AVES probes share registered support/query rows; specialist and published adapters are clearly labelled external pretrained baselines with different training supervision.
- Plans/checks do not load large models. Full execution requires --execute; conditional model selection requires --enable-conditional.
- Preserve byte/config/environment identity on resume and report failed or incomplete jobs as incomplete.
- Routine implementation choices proceed under the user's request; no new design approval loop.

## Tasks

- [x] Generalize model dispatch, suite order and controls while retaining original config behavior; add full primary and conditional configurations.
- [x] Implement Qwen3-Omni backend in akr_closing/qwen3_backend.py with contract tests and isolated dependency setup. Own files only; integrate through factory.
- [x] Implement GLM-4-Voice backend in akr_closing/glm_backend.py, including audio tokenizer and fused QKV intervention handling, with contract tests and isolated dependency setup.
- [x] Add a comprehensive CLI for planning, checking, bounded execution and reporting AKR, AVES, specialist, LoRA and BEANS-Zero jobs. Record applicable scope rather than claiming every baseline fits every task.
- [x] Add reproducible environment instructions, evidence inventory and a concise runbook with exact commands.
- [x] Run targeted and full tests, real bounded smoke checks where authorized, all primary preflights; independently review integrations and fix material defects.

## Review focus

- Conditional models cannot run through implicit task expansion.
- Model preparation and masks remain correct through autoregressive generation and support answer concatenation.
- Baseline training never reads query labels; pretrained baselines disclose unmatched supervision.
- Resume rejects changed inputs and cannot declare success from a truncated output.
- Missing dependencies or unvalidated families produce an explicit readiness limitation rather than a false success.

## Evidence ledger

- Before changes: primary Qwen7B/3B real feature/gradient/injection smoke passed, 27 targeted tests passed, 2,950 BEANS-Zero IDs and metadata verified. GLM/Qwen3 weights cached but backend absent.
- Final core regression suite: 254 passed, 6 optional-environment architecture tests skipped. In their prepared environments, GLM 11 passed (two upstream warnings), Qwen3 9 passed. All four checkpoint dependency checks passed without large model loads.
- Added derived supervised BEANS registration: 12 components, 2,002 canonical groups, 948 exclusions; three seeded allocations each use 636 supports and 1,366 queries. Strict registration/source/manifest hashes and waveform checks guard role identity.
- Review fixes: report actual baseline scores, preserve adapter token limits, use the Qwen wrapper's thinker-specific output limit, reject ambiguous nested class names, bind aggregate reports to study identity, route compatible interpreters, fail missing-baseline checks, and bind derived BEANS files to their original registration.
- Final real Qwen3B smoke: all 27 methods and 54 query records passed; both no-op checks exact; interventions/report verified; cached resume passed without a model reload. Final source fingerprints match. Real applicable baseline smokes passed; supervised BEANS AVES and original Qwen7B diagnostic were checked again after the registration changes.
- Full primary preflight passed with exit 0: 180 AKR jobs, 110 baseline/diagnostic jobs, seven worker groups. `results/readiness/FINAL_READINESS.json` consolidates evidence. Full scientific experiments remain unstarted; optional production-weight GPU acceptance remains conditional.
