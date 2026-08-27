#!/usr/bin/env bash
# R46 CI helper — install the production lock's THIRD-PARTY deps only
# (--require-hashes) into the CI python.
#
# The six in-repo packages (factor_engine, quant_evaluator, factor_optimizer,
# factor_assets, factor_preprocess, data_access) are NOT on PyPI.  For the pytest
# gates they are imported from source via the repo root (pytest rootdir conftest
# puts the repo root on sys.path), so no `pip install -e` is needed here.  The
# FRESH_WHEEL_MATRIX job builds real wheels from their own pyprojects instead.
#
# The lock's internal-dist lines (data-access==, factor-engine==, ...) and the
# `python==` marker are filtered out so pip sees ONLY the hashed third-party
# requirements.  Nothing is dropped silently: every real third-party pin with a
# --hash is installed as-is.
set -euo pipefail

REPO="${REPO:-$PWD}"
PY="${PY:-python}"
LOCK="$REPO/requirements-production.lock"

test -f "$LOCK" || { echo "FATAL: lockfile missing at $LOCK"; exit 2; }

TMP_REQ="$REPO/.ci-thirdparty.txt"
trap 'rm -f "$TMP_REQ"' EXIT

awk '
  /^#/ { next }
  /^python==/ { next }   # WHY: the interpreter version is an environment property, not a
                         # pip-installable requirement — actions/setup-python picks the CI
                         # CPython, so the python== marker in the lock must be dropped.
  /^data-access==/ || /^factor-engine==/ || /^quant-evaluator==/ || \
  /^factor-optimizer==/ || /^factor-assets==/ || /^factor-preprocess==/ { next }
  { print }
' "$LOCK" > "$TMP_REQ"

echo "==> installing $(grep -c '==' "$TMP_REQ") third-party lock deps (--require-hashes)"
"$PY" -m pip install -q --require-hashes -r "$TMP_REQ"
echo "==> lock deps installed"
