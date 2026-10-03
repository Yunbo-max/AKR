# AKR completion implementation plan

Goal: finish the existing consolidation PR, not create another method or run GPUs.
Source baseline: remote 87f99b1fffde709f46987e9d3028e2c043cf17b0.
User-approved destination: Yunbo-max/AKR; old repositories preserved.

- [x] Obtain actual 2026-10-02 research-autopilot export and inspect references.
- [x] Reconcile existing migration PR #1 and export its exact materialized tree.
- [x] Run existing CPU tests: 137 passed before edits.
- [x] Test-first release integrity checks: ordinary manuscript/bibliography, all figure references, immutable default model config, consistent task statuses, no GPU import.
- [x] Restore missing source files and manuscript from provided packages; retain newer runtime fixes rather than overwrite them with historical overlays.
- [x] Add actual-skill retrospective audit, claim/evidence and source/version registry; distinguish hypothesis, implementation, CPU validation and scientific execution.
- [x] Freeze bounded S1 specification (already implemented, not run), explain strongest simple alternative and negative outcome handling. Do not invent retrospective Gate A approval.
- [x] Compile manuscript and verify unchanged numerical tables; preserve transfer/scale limitations.
- [x] Run full CPU suite, artifact checks and clean working-copy checks.
- [ ] Publish ordinary recovered files, verify remote branch, then merge PR #1 without rewriting history.

No raw audio, model weights, secrets, unrelated files, font files or private custom skill implementation will be published. Exact inaccessible yuhanlydia/animal HEAD equivalence is not claimed. Large source PNGs are kept in the downloadable author backup; published paper uses already documented portable image derivatives. S1 default is a new simplest-alternative comparison; F1/F2/F3 are optional recovery/re-evaluation, not new missing scientific runs.
