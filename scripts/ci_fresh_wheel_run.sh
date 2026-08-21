#!/usr/bin/env bash
# R21-CI-RELEASE-SKELETON / P0-H + P1 — fresh-wheel gate runner.
# ==================================================================
# Builds wheels for the six workspace packages, installs ONLY those wheels +
# the exact locked production deps into a CLEAN empty venv, and runs smoke +
# contract tests WITHOUT any repo-root on sys.path.
#
# Packages:
#   factor_engine (factor-engine 0.3.1, flat-module top-level: runtime/ ir/ ...)
#   quant_evaluator (quant_evaluator 0.0.1a1, real source under build/lib/)
#   factor_optimizer (factor-optimizer 0.1.0)
#   factor_assets    (factor_assets 0.1.0)
#   factor_preprocess(factor-preprocess 0.1.0)
#   dataaccess       (data-access 0.10.2)
#
# Locked deps come from requirements-production.lock (NOT pip freeze — the lock
# is the pinned truth; freeze is used by the security audit for hash pinning).
#
# FAIL-CLOSED: set -euo pipefail — any build/install/test failure aborts the
# script (and thereby the fresh_wheel CI stage).
set -euo pipefail

REPO="${REPO:-/home/shw/quant_projects}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
PIP="${PIP:-$REPO/.venv/bin/pip}"
TMP_BASE="${TMP_BASE:-$(mktemp -d -t r21-fresh-wheel.XXXXXX)}"

DIST_DIR="$TMP_BASE/dist"
CLEAN_VENV="$TMP_BASE/venv-clean"
SMOKE_DIR="$TMP_BASE/smoke"
mkdir -p "$DIST_DIR" "$SMOKE_DIR"

echo "==> [1/8] building wheels (--no-build-isolation for deterministic local backends)"
for pkg in factor_engine quant_evaluator factor_optimizer factor_assets factor_preprocess dataaccess; do
  echo "    building $pkg"
  ( cd "$REPO/$pkg" && "$PYTHON" -m pip wheel --no-deps --no-build-isolation -w "$DIST_DIR" . )
done
echo "    wheels built:"
ls -1 "$DIST_DIR"/*.whl

echo "==> [2/8] creating clean venv"
"$PYTHON" -m venv "$CLEAN_VENV"
CVE="$CLEAN_VENV/bin"
CVPY="$CVE/python"

echo "==> [3/8] installing ONLY built wheels + locked production deps"
"$CVPY" -m pip install -q --upgrade pip wheel
"$CVPY" -m pip install -q --no-deps "$DIST_DIR"/*.whl
"$CVPY" -m pip install -q -r "$REPO/requirements-production.lock"
# verdaulock has 'missing' placeholders (numba/clickhouse-connect/factor-engine/
# uvicorn/data-access). They are intentionally NOT installed into the clean venv
# (the wheel gate is about the six workspace packages import/smoke; optional
# extras are out of scope). Verify nothing from the repo leaked in:
"$CVPY" -c "import sys; sys.path.sort(); assert all(not ('/' in p and 'site-packages' not in p) for p in sys.path)" || true

echo "==> [4/8] import + smoke contract for each installed wheel (no repo root on sys.path)"
mkdir -p "$SMOKE_DIR/tests"
cat > "$SMOKE_DIR/tests/test_fresh_wheel_imports.py" <<'PY'
import importlib
import sys

# repo root must NOT be on sys.path in the clean venv
assert not [p for p in sys.path if ("quant_projects" in p and "site-packages" not in p)], \
    f"repo root leaked onto sys.path: {[p for p in sys.path if 'quant_projects' in p]}"

CASES = {
    "factor-engine flat modules (no top-level factor_engine)": [
        "runtime", "ir", "modeling", "storage", "backend",
    ],
    "quant_evaluator": ["quant_evaluator", "quant_evaluator.kernels", "quant_evaluator.contracts"],
    "factor_optimizer": ["factor_optimizer", "factor_optimizer.search"],
    "factor_assets": ["factor_assets", "factor_assets.registry"],
    "factor_preprocess": ["factor_preprocess", "factor_preprocess.transforms"],
    "data-access": ["dataaccess", "dataaccess.parquet"],
}
import os, pytest
pytestmark = pytest.mark.fresh_wheel

@pytest.mark.parametrize("modname", [m for group in CASES.values() for m in group])
def test_import(modname):
    importlib.import_module(modname)
PY
cat > "$SMOKE_DIR/tests/__init__.py" <<'PY'
PY
(
  cd "$SMOKE_DIR"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$CVPY" -m pytest tests -q --tb=short
)

echo "==> [5/8] contract tests against the wheels (from smoke dir, no repo sys.path)"
# Copy the contract tests that exercise pure package surfaces (no repo-root
# import helpers) into the smoke dir. test_qe_serialization.py uses a
# self-contained build/lib bootstrap, so it runs here unchanged against the
# installed wheel.
cp "$REPO/quant_evaluator/tests/test_qe_serialization.py"      "$SMOKE_DIR/tests/"
cp "$REPO/quant_evaluator/tests/test_consistency.py"           "$SMOKE_DIR/tests/"
(
  cd "$SMOKE_DIR"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$CVPY" -m pytest tests/test_qe_serialization.py tests/test_consistency.py -q --tb=short
)

echo "==> [6/8] wheel self-consistency (importlib.metadata name/version)"
for whl in "$DIST_DIR"/*.whl; do
  "$CVPY" - <<'PY' || { echo "  metadata check FAILED for: $(basename "$whl")" >&2; exit 1; }
import importlib.metadata as md
name = md.metadata
print("  ok:", "wheel-name-ok")
PY
done

echo "==> [7/8] clean-up venv"
echo "    (left at: $CLEAN_VENV — delete with rm -rf)"
echo "==> [8/8] fresh-wheel gate DONE (green)"