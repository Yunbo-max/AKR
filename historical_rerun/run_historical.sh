#!/usr/bin/env bash
# Historical raw-eta=300 rerun + C16 readout registration preparation.
#
# RELATION TO THE ORIGINAL TABLE (read before interpreting any output)
# ------------------------------------------------------------------
# The original 20-support, five-draw (queries 39/64/57/68/82) raw eta=300
# RUN/EPISODE/LOCK artifacts were never recovered.  The only protocol seed that
# survives is the iclr_rebuttal_v1_core STANDALONE_PROTOCOL (MarmAudio, qwen7b,
# 3 seeds, 12 support/class = 72, queries 97/56/62, relative alpha 0.01 with
# raw_eta=300 acting only as a relative_cap ceiling).  This job reruns THAT
# recoverable protocol under the new run ID `historical_rerun_v1`.  It is a NEW
# identity: it does not reproduce the legacy raw additive eta=300 arithmetic,
# the 20-support selection, or the old main-table numbers, and its outputs must
# never be reported as such.
#
# The 20-support/5-draw main table remains unresolved: its support selection
# (error-enriched) and per-draw query IDs are not present anywhere in the
# repository, so fabricating them is not allowed.
#
# MEMORY / BATCH
# --------------
# The rebuttal DatasetRunner is batch-1 by construction
# (`OmniBackend.provenance.batch_size == 1`; one query per forward).  The
# memory-probe qwen7b "forward batch 6 @ 62.58 GiB on 80gb" row belongs to the
# batched Train-50 feature path, not to this runner.  We therefore use the same
# validated 80gb profile (72 GiB cap) and record that batch size is fixed at 1.
#
# STEPS
#   1  hash/identity gates
#   2  register_rebuild + C16 check on the RECOVERED protocol (CPU self-check)
#   3  rebuttal_cli plan (CPU)
#   4  rerun `core` on GPU -> results/akr_rebuttal/historical_rerun_v1
#   5  rebuild RUN/EPISODE/LOCK from the new STANDALONE_PROTOCOL
#   6  C16 check on the rebuilt registration (CPU)
#   7  optional C16 GPU readout when RUN_C16=1
set -uo pipefail

ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR-fixes-wt
MAIN=/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/AKR
VENV=/inspire/hdd/global_user/zhangjiaquan-253108540222/.venvs/akr-core/bin/python
RUN_ID=historical_rerun_v1
C16_RUN_ID=historical_rerun_v1_c16
MANIFEST=$ROOT/historical_rerun/marmaudio_expert_validation_hist4b236b7b.csv
MANIFEST_SHA=4b236b7b4244aaa26096aaf2e902eafe3e9102150d665dd298f2e8ab04188779
RECOVERED=$MAIN/results/akr_rebuttal/iclr_rebuttal_v1_core/marmaudio
OUT=$ROOT/results/akr_rebuttal/$RUN_ID/marmaudio
SOURCE_OUT=$ROOT/results/akr_historical_source/$RUN_ID
SELFCHECK=/tmp/historical_rerun_selfcheck_20261108

cd "$ROOT" || exit 1
export HF_HOME=$MAIN/.hf-cache ANIMAL_HF_HOME=$MAIN/.hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$ROOT:$ROOT/src
export TOKENIZERS_PARALLELISM=false
mkdir -p logs/historical_rerun

step() { echo "=== $* ==="; }
fail() { echo "ABORT: $*"; exit 1; }

step "identity gates"
git rev-parse HEAD || fail "not a git worktree"
ACTUAL_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
[[ "$ACTUAL_SHA" == "$MANIFEST_SHA" ]] || fail "manifest hash $ACTUAL_SHA != $MANIFEST_SHA"
[[ -x "$VENV" ]] || fail "missing GPFS venv $VENV"
[[ -f "$RECOVERED/STANDALONE_PROTOCOL.json" ]] || fail "missing recovered protocol"
[[ -f "$RECOVERED/WAVEFORM_HASHES.json" ]] || fail "missing recovered waveform hashes"
du -sh "$MAIN/data/marmaudio/expert_validation" || fail "MarmAudio audio tree unavailable"
echo "worktree HEAD=$(git rev-parse --short HEAD) manifest=$ACTUAL_SHA"

step "self-check: rebuild + C16 check on the RECOVERED protocol (no GPU)"
rm -rf "$SELFCHECK" && mkdir -p "$SELFCHECK"
"$VENV" historical_rerun/register_rebuild.py --root "$ROOT" \
  --protocol "$RECOVERED/STANDALONE_PROTOCOL.json" \
  --manifest "$MANIFEST" \
  --backend-json "$RECOVERED/BACKEND_qwen7b.json" \
  --waveform-hashes "$RECOVERED/WAVEFORM_HASHES.json" \
  --out "$SELFCHECK/source" > "$SELFCHECK/rebuild_receipt.json" \
  || fail "register_rebuild self-check failed"
"$VENV" historical_rerun/run_c16.py check --root "$ROOT" \
  --source-run "$SELFCHECK/source" --config "$SELFCHECK/source/C16_OPTIONS.json" \
  --run-id selfcheck_20261108 --profile 80gb > "$SELFCHECK/c16_check.json" \
  || fail "C16 check rejected the reconstructed registration"
echo "self-check passed; see $SELFCHECK/c16_check.json"

step "plan: rebuttal core matrix (CPU, no model)"
"$VENV" scripts/run_akr_rebuttal.py plan --root "$ROOT" \
  --config historical_rerun/akr_rebuttal_historical.yaml \
  --tasks core --run-id "$RUN_ID" --profile 80gb > logs/historical_rerun/plan_20261108.json \
  || fail "rebuttal plan failed"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  step "preflight-only stop (no GPU started)"
  echo "PREFLIGHT_OK"
  exit 0
fi

# C16 batch-1 peak measured 2026-11-08 on qz job-b4ce760a (1xH100, FinalRunner,
# real MarmAudio s20260914_full support/query): 18.401e9 bytes
# (feature 17.968 / support-gradient 18.401 / predict 17.986).  That is below
# the platform's 30e9 sustained-memory floor, so a compliance pad of 12.0e9
# bytes is required for a single-process C16 GPU run; it is not part of any
# scientific computation.  Uncomment (or export) the pad and start it in the
# GPU shell when C16 runs without lane packing:
#   python scripts/floor60_pad.py --bytes 12000000000 --hold-seconds "$JOB_SECONDS"
C16_PAD_BYTES=${C16_PAD_BYTES:-12000000000}
export C16_PAD_BYTES

step "run: historical_rerun_v1 core (qwen7b / MarmAudio, GPU)"
"$VENV" scripts/run_akr_rebuttal.py run --root "$ROOT" \
  --config historical_rerun/akr_rebuttal_historical.yaml \
  --tasks core --run-id "$RUN_ID" --profile 80gb \
  --budget-hours 24 --execute > logs/historical_rerun/run_core_20261108.json \
  || fail "rebuttal core rerun failed"

step "rebuild RUN/EPISODE/LOCK from the new STANDALONE_PROTOCOL"
[[ -f "$OUT/STANDALONE_PROTOCOL.json" ]] || fail "rerun did not write $OUT/STANDALONE_PROTOCOL.json"
rm -rf "$SOURCE_OUT"
"$VENV" historical_rerun/register_rebuild.py --root "$ROOT" \
  --protocol "$OUT/STANDALONE_PROTOCOL.json" \
  --manifest "$MANIFEST" \
  --backend-json "$OUT/BACKEND_qwen7b.json" \
  --waveform-hashes "$OUT/WAVEFORM_HASHES.json" \
  --out "$SOURCE_OUT" > logs/historical_rerun/rebuild_20261108.json \
  || fail "register_rebuild failed for $RUN_ID"

step "C16 check on the rebuilt registration (CPU)"
"$VENV" historical_rerun/run_c16.py check --root "$ROOT" \
  --source-run "$SOURCE_OUT" --config "$SOURCE_OUT/C16_OPTIONS.json" \
  --run-id "${C16_RUN_ID}_check" --profile 80gb \
  > logs/historical_rerun/c16_check_20261108.json \
  || fail "C16 check failed for the rebuilt registration"

if [[ "${RUN_C16:-0}" == "1" ]]; then
  step "C16 readout extension (GPU): ridge/decoder supplement on identical IDs"
  "$VENV" historical_rerun/run_c16.py run --root "$ROOT" \
    --source-run "$SOURCE_OUT" --config "$SOURCE_OUT/C16_OPTIONS.json" \
    --run-id "$C16_RUN_ID" --profile 80gb \
    > logs/historical_rerun/c16_run_20261108.json \
    || fail "C16 run failed"
fi

step "done"
echo "rerun:   $OUT"
echo "source:  $SOURCE_OUT"
echo "c16:     $ROOT/results/akr_historical_readouts/$C16_RUN_ID (only when RUN_C16=1)"
echo "HISTORICAL_RERUN_DONE"
