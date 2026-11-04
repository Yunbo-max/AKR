# AKR rebuttal implementation record

Goal: implement the user-requested P0/P1/P2 controls and an executable rebuttal matrix while retaining original numerical defaults and evidence.
Architecture: reuse the frozen Omni backend, original SVD/ridge regressor, FinalRunner caches and journals; add opt-in actual-path gain calibration and complete-only reports.
Tech stack: existing Python/NumPy/PyTorch/sklearn packages; no new architecture or model family.

- [x] Verify available upstream main and snapshot source blobs.
- [x] Preserve the original 148 tests and add failing tests for the identified behaviors.
- [x] Measure post-add BF16 updates and explicit hook coverage.
- [x] Add named relative-cap/raw paths without replacing historical defaults.
- [x] Add bounded support-only answer-CE coordinate calibration and scalar control.
- [x] Separate fit/calibration/guard/query; allocate exact label coverage before model work.
- [x] Preserve all official BEANS queries; exclude training-side source overlap in new protocol only.
- [x] Add matched-total-label readout and original-AKR baselines, no-op, random/shuffle and K/V controls.
- [x] Add repeatable core/scale/Dogs queue, optional Watkins, preflight, wall cap and resume.
- [x] Run all 182 collected CPU tests in disjoint batches; verify original 116 release checks.
- [ ] Run real Omni/audio experiments on the user's GPU; CPU verification does not substitute.
- [x] Publish through the connected GitHub interface after fresh remote CPU verification; see docs/rebuttal/REMOTE_PUBLICATION_20261104.json.

Guard makes one final comparison per predeclared calibration method. Invalid execution raises
an error; lack of benefit is a recorded native fallback. No reviewer wording, new test
population, novelty certification or positive accuracy is fabricated.
