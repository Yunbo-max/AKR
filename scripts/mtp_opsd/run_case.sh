#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/mtp_opsd/run_case.sh CASE {train|evaluate|run} --run-id ID [CLI options]" >&2
  exit 2
fi
case_id="$1"
action="$2"
shift 2
case "$case_id" in
  c00_native_ntp|c01_native_mtp|c02_ntp_opsd_g4|c03_mtp_opsd_g4|c04_ntp_opsd_g8|c05_mtp_opsd_g8) ;;
  *) echo "Unknown independent case: $case_id" >&2; exit 2 ;;
esac
case "$action" in
  train|evaluate|run) ;;
  *) echo "Action must be train, evaluate, or run" >&2; exit 2 ;;
esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
exec "${MTP_PYTHON:-python}" -m mtp_opsd.cli "$action" "$@" --case "$case_id"
