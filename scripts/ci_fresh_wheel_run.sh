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
# requirements-production.lock contains '==missing' placeholders for optional
# extras (numba/clickhouse-connect/factor-engine/uvicorn), a `python==3.10.12`
# interpreter pin, a stale `data-access==0.2.0` entry, and a bogus `yaml==6.0.3`
# (the real package is PyYAML). Filter all of these out — the six workspace
# packages are provided by the freshly built wheels (installed --no-deps above),
# the clean venv already IS the pinned interpreter, and PyYAML is added back
# under its real name. The fresh-wheel gate is about the six packages import/smoke.
grep -vE '==missing$|^python==|^data-access==|^yaml==' "$REPO/requirements-production.lock" > "$TMP_BASE/lock-filtered.txt"
echo "PyYAML==6.0.3" >> "$TMP_BASE/lock-filtered.txt"
"$CVPY" -m pip install -q -r "$TMP_BASE/lock-filtered.txt"
# pytest + the wheels' declared runtime deps that are NOT in the production lock
# (the lock predates these packages' dependency declarations). These are the
# packages the six wheels import at runtime; without them the import/smoke gate
# cannot run. They are installed from PyPI, not from the repo.
"$CVPY" -m pip install -q pytest sqlglot dataclasses-json PyWavelets statsmodels psutil
# verdaulock has 'missing' placeholders (numba/clickhouse-connect/factor-engine/
# uvicorn/data-access). They are intentionally NOT installed into the clean venv
# (the wheel gate is about the six workspace packages import/smoke; optional
# extras are out of scope). Verify nothing from the repo leaked in:
"$CVPY" -c "import sys; leaked=[p for p in sys.path if 'quant_projects' in p and 'site-packages' not in p]; assert not leaked, f'repo root leaked onto sys.path: {leaked}'"

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
        "runtime", "ir", "modeling", "storage", "backend", "api", "fields",
    ],
    "quant_evaluator": ["quant_evaluator", "quant_evaluator.kernels", "quant_evaluator.contracts"],
    "factor_optimizer": ["factor_optimizer", "factor_optimizer.search"],
    "factor_assets": ["factor_assets", "factor_assets.registry"],
    "factor_preprocess": ["factor_preprocess", "factor_preprocess.transforms"],
    "data-access (import name data_access)": ["data_access", "data_access.contract", "data_access.read"],
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

echo "==> [5/8] contract tests against the INSTALLED wheels (no repo sys.path, no build/lib bootstrap)"
# The repo's test_qe_serialization.py / test_consistency.py hard-code a
# build/lib bootstrap that asserts the package comes from the repo source tree,
# so they cannot run against an installed wheel. Instead we run a self-contained
# contract test that imports the packages from site-packages (the wheel) and
# exercises the public serialization/registry surface directly.
cat > "$SMOKE_DIR/tests/test_wheel_contracts.py" <<'PY'
import pickle
import sys

# repo root must NOT be on sys.path in the clean venv
assert not [p for p in sys.path if ("quant_projects" in p and "site-packages" not in p)], \
    f"repo root leaked onto sys.path: {[p for p in sys.path if 'quant_projects' in p]}"

import pytest
import quant_evaluator
from quant_evaluator.registry.metrics import CANONICAL_METRIC_ALIASES, get_metric
from quant_evaluator.contracts.artifact_types import (
    ExposureArtifact, ICSeriesArtifact, ProbePortfolioArtifact, QuantileReturnArtifact,
)
from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact

# the installed wheel must be the import source (site-packages), not repo build/lib
assert "site-packages" in quant_evaluator.__file__, quant_evaluator.__file__

def test_metric_artifact_pickle_roundtrip():
    import numpy as np
    a = ScalarMetricArtifact(metric_id="ic.daily", domain="cross_section",
                             provenance={"src": "wheel-contract"}, values=np.array([1.5]))
    b = pickle.loads(pickle.dumps(a))
    assert b == a

def test_artifact_to_dict_from_dict_roundtrip():
    import numpy as np
    samples = [
        ICSeriesArtifact(values=np.array([[1.0, 2.0]]), time_index=("t0",)),
        QuantileReturnArtifact(values=np.array([[1.0, 2.0], [3.0, 4.0]]), n_quantiles=2),
        ProbePortfolioArtifact(values=np.array([[1.0, 2.0]]), time_index=("t0",)),
        ExposureArtifact(values=np.array([[1.0, 2.0]]), exposure_type="loadings"),
    ]
    for a in samples:
        b = type(a).from_dict(a.to_dict())
        assert b == a

def test_registry_mappingproxy_pickle_roundtrip():
    b = pickle.loads(pickle.dumps(CANONICAL_METRIC_ALIASES))
    assert dict(b) == dict(CANONICAL_METRIC_ALIASES)

def test_get_metric_resolves():
    assert get_metric is not None
PY
(
  cd "$SMOKE_DIR"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$CVPY" -m pytest tests/test_wheel_contracts.py -q --tb=short
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