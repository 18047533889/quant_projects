#!/usr/bin/env bash
# Fail-closed CI gate for factor_engine's production-critical test tier.
#
# Usage:
#   factor_engine/scripts/check_production_critical.sh [--workers N] [--junitxml-out DIR]
#
# Runs ONLY the production_critical tier. Any test failure OR any file listed in
# production_critical.txt that does not exist on disk is a gate failure
# (exit non-zero) — production regressions can never hide behind the pre-existing
# operator/evidence drift that lives in the research/legacy tiers.
set -u

cd "$(dirname "$0")/../.." || exit 3   # repo root (quant_projects)

WORKERS="${WORKERS:-1}"
JUNIT=""
if [[ $# -gt 0 ]]; then
  case "$1" in
    --workers) WORKERS="${2:-1}"; shift 2 ;;
    --junitxml-out) JUNIT="--junitxml-out ${2}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$( [ "$WORKERS" -gt 31 ] && echo 31 || echo "$WORKERS" )}"

echo "[gate] production_critical via run_test_tiers.py --gate (workers=${WORKERS})"
# shellcheck disable=SC2086
exec .venv/bin/python factor_engine/scripts/run_test_tiers.py --gate production_critical \
  --workers "$WORKERS" ${JUNIT}
