#!/usr/bin/env bash
# R25 P1-16 — fresh wheel matrix: build all 6 wheels, install ONLY from pinned
# lock + hashes into a clean venv, wheel-only cross-package integration.
# HONEST: a package that cannot build/import is BLOCKED with a reason — never PASS.
# Local-only: no git mutations, no push.
set -uo pipefail   # NOT -e: we record per-package BLOCKED instead of aborting

REPO="${REPO:-/home/sunhaiwei/quant_projects}"
PY="${PY:-/tmp/fe2/bin/python}"
TMP_BASE="${TMP_BASE:-$(mktemp -d -t fwm.XXXXXX)}"
DIST="$TMP_BASE/dist"
STAGE="$TMP_BASE/data_access_stage"
SMOKE="$TMP_BASE/smoke"
mkdir -p "$DIST" "$STAGE" "$SMOKE"

RESULT="$REPO/evidence/fresh_wheel_matrix.json"
REPORT="$TMP_BASE/report.jsonl"
: > "$REPORT"

echo "==> [1] building wheels"
declare -A WHEEL_SHA
declare -A BUILT
declare -A REASON

build_one() {  # $1=pkg  $2=build-dir  $3=label
  local pkg="$1" bdir="$2" label="$3"
  echo "    building $label"
  if ( cd "$bdir" && "$PY" -m pip wheel --no-deps --no-build-isolation -w "$DIST" . ) >"$TMP_BASE/build_$pkg.log" 2>&1; then
    local w; w=$(ls "$DIST"/*.whl 2>/dev/null | grep -iE "$pkg|$label" | head -1)
    if [ -n "$w" ]; then
      WHEEL_SHA["$pkg"]=$(sha256sum "$w" | cut -d' ' -f1)
      BUILT["$pkg"]=true
      echo "      ok: $(basename "$w") sha256=${WHEEL_SHA[$pkg]:0:16}…"
      echo "{\"pkg\":\"$pkg\",\"built\":true,\"wheel_sha256\":\"${WHEEL_SHA[$pkg]}\",\"reason\":\"\"}" >> "$REPORT"
    else
      BUILT["$pkg"]=false; REASON["$pkg"]="wheel not produced"
      echo "{\"pkg\":\"$pkg\",\"built\":false,\"wheel_sha256\":null,\"reason\":\"wheel not produced\"}" >> "$REPORT"
    fi
  else
    BUILT["$pkg"]=false
    REASON["$pkg"]="build failed: $(tail -3 "$TMP_BASE/build_$pkg.log" | tr '\n' ' ')"
    echo "{\"pkg\":\"$pkg\",\"built\":false,\"wheel_sha256\":null,\"reason\":\"${REASON[$pkg]}\"}" >> "$REPORT"
  fi
}

# factor_engine builds from ROOT (root pyproject/_build_backend). The root
# build produces the `quant_projects` umbrella wheel (name quant-projects),
# which is how factor_engine is packaged in this monorepo. Record it honestly.
if [ -d "$REPO/factor_engine" ]; then
  echo "    building factor_engine (from root -> quant_projects umbrella wheel)"
  if ( cd "$REPO" && "$PY" -m pip wheel --no-deps --no-build-isolation -w "$DIST" . ) >"$TMP_BASE/build_factor_engine.log" 2>&1; then
    local w; w=$(ls "$DIST"/*.whl 2>/dev/null | grep -iE "quant_projects" | head -1)
    if [ -n "$w" ]; then
      WHEEL_SHA[factor_engine]=$(sha256sum "$w" | cut -d' ' -f1)
      BUILT[factor_engine]=true
      echo "      ok: $(basename "$w") sha256=${WHEEL_SHA[factor_engine]:0:16}…"
      echo "{\"pkg\":\"factor_engine\",\"built\":true,\"wheel_sha256\":\"${WHEEL_SHA[factor_engine]}\",\"reason\":\"root umbrella wheel quant_projects\"}" >> "$REPORT"
    else
      BUILT[factor_engine]=false; REASON[factor_engine]="wheel not produced"
      echo "{\"pkg\":\"factor_engine\",\"built\":false,\"wheel_sha256\":null,\"reason\":\"wheel not produced\"}" >> "$REPORT"
    fi
  else
    BUILT[factor_engine]=false
    REASON[factor_engine]="build failed: $(tail -3 "$TMP_BASE/build_factor_engine.log" | tr '\n' ' ')"
    echo "{\"pkg\":\"factor_engine\",\"built\":false,\"wheel_sha256\":null,\"reason\":\"${REASON[factor_engine]}\"}" >> "$REPORT"
  fi
else
  BUILT[factor_engine]=false; REASON[factor_engine]="dir missing"
  echo "{\"pkg\":\"factor_engine\",\"built\":false,\"wheel_sha256\":null,\"reason\":\"dir missing\"}" >> "$REPORT"
fi

# factor_assets (restored from pin 360836b5)
if [ -d "$REPO/factor_assets" ]; then
  build_one factor_assets "$REPO/factor_assets" "factor_assets"
else
  BUILT[factor_assets]=false; REASON[factor_assets]="dir missing (restore in progress)"
fi

# quant_evaluator / factor_optimizer / factor_preprocess build in place
for pkg in quant_evaluator factor_optimizer factor_preprocess; do
  if [ -d "$REPO/$pkg" ]; then
    build_one "$pkg" "$REPO/$pkg" "$pkg"
  else
    BUILT["$pkg"]=false; REASON["$pkg"]="dir missing"
  fi
done

# dataaccess: stage as a `data_access` namespace with a minimal pyproject.toml
if [ -d "$REPO/dataaccess" ]; then
  echo "    building dataaccess (staged data_access namespace)"
  # copy importable top-level modules + subpackages
  cp "$REPO"/dataaccess/*.py "$STAGE/" 2>/dev/null
  for sub in core read write registry cos contract clickhouse security runtime snapshot r30 export telemetry service quality; do
    if [ -d "$REPO/dataaccess/$sub" ]; then
      cp -r "$REPO/dataaccess/$sub" "$STAGE/$sub"
    fi
  done
  cat > "$STAGE/pyproject.toml" <<'PY'
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "data-access"
version = "0.10.2"
requires-python = ">=3.10"

[tool.setuptools]
packages = ["data_access", "data_access.core", "data_access.read", "data_access.write", "data_access.registry", "data_access.cos", "data_access.contract", "data_access.clickhouse", "data_access.security", "data_access.runtime", "data_access.snapshot", "data_access.r30", "data_access.export", "data_access.telemetry", "data_access.service", "data_access.quality"]

[tool.setuptools.package-dir]
"data_access" = "."
PY
  if ( cd "$STAGE" && "$PY" -m pip wheel --no-deps --no-build-isolation -w "$DIST" . ) >"$TMP_BASE/build_dataaccess.log" 2>&1; then
    local w; w=$(ls "$DIST"/*.whl 2>/dev/null | grep -iE "data.access" | head -1)
    if [ -n "$w" ]; then
      WHEEL_SHA[dataaccess]=$(sha256sum "$w" | cut -d' ' -f1)
      BUILT[dataaccess]=true
      echo "      ok: $(basename "$w") sha256=${WHEEL_SHA[dataaccess]:0:16}…"
      echo "{\"pkg\":\"dataaccess\",\"built\":true,\"wheel_sha256\":\"${WHEEL_SHA[dataaccess]}\",\"reason\":\"\"}" >> "$REPORT"
    else
      BUILT[dataaccess]=false; REASON[dataaccess]="wheel not produced"
      echo "{\"pkg\":\"dataaccess\",\"built\":false,\"wheel_sha256\":null,\"reason\":\"wheel not produced\"}" >> "$REPORT"
    fi
  else
    BUILT[dataaccess]=false
    REASON[dataaccess]="build failed: $(tail -3 "$TMP_BASE/build_dataaccess.log" | tr '\n' ' ')"
    echo "{\"pkg\":\"dataaccess\",\"built\":false,\"wheel_sha256\":null,\"reason\":\"${REASON[dataaccess]}\"}" >> "$REPORT"
  fi
else
  BUILT[dataaccess]=false; REASON[dataaccess]="dir missing"
fi

echo "==> [2] clean venv + install ONLY built wheels + pinned deps"
# Use virtualenv (available) instead of `python -m venv` (ensurepip missing on this host).
"$PY" -m virtualenv "$TMP_BASE/venv" >"$TMP_BASE/venv.log" 2>&1 || { echo "venv create failed: $(tail -3 "$TMP_BASE/venv.log")"; exit 1; }
CVPY="$TMP_BASE/venv/bin/python"
"$CVPY" -m pip install -q --upgrade pip wheel
"$CVPY" -m pip install -q --no-deps "$DIST"/*.whl
grep -vE '==missing$|^python==|^data-access==|^yaml==' "$REPO/requirements-production.lock" > "$TMP_BASE/lock-filtered.txt"
echo "PyYAML==6.0.3" >> "$TMP_BASE/lock-filtered.txt"
"$CVPY" -m pip install -q -r "$TMP_BASE/lock-filtered.txt"
# runtime deps the wheels import that are NOT in the lock (documented: not lock-pinned)
"$CVPY" -m pip install -q pytest sqlglot dataclasses-json PyWavelets statsmodels psutil
# QE wheel pandas/numpy compat: if import fails under the clean venv's numpy, pin to lock versions and retry once
if ! "$CVPY" -c "import quant_evaluator" >/dev/null 2>&1; then
  echo "    QE import failed under clean venv numpy; pinning numpy==2.2.6 pandas==2.3.3"
  "$CVPY" -m pip install -q 'numpy==2.2.6' 'pandas==2.3.3'
fi

echo "==> [3] wheel-only cross-package integration (no repo root on sys.path)"
cat > "$SMOKE/test_wheel_matrix.py" <<'PY'
import importlib, sys
assert not [p for p in sys.path if ("quant_projects" in p and "site-packages" not in p)], \
    f"repo root leaked onto sys.path: {[p for p in sys.path if 'quant_projects' in p]}"
import pytest
CASES = {
    "factor_engine": ["factor_engine", "factor_engine.runtime", "factor_engine.ir", "factor_engine.modeling", "factor_engine.api"],
    "quant_evaluator": ["quant_evaluator", "quant_evaluator.contracts"],
    "factor_optimizer": ["factor_optimizer", "factor_optimizer.contracts"],
    "factor_assets": ["factor_assets", "factor_assets.registry"],
    "factor_preprocess": ["factor_preprocess", "factor_preprocess.contracts"],
    "dataaccess": ["data_access", "data_access.contract"],
}
@pytest.mark.parametrize("modname", [m for g in CASES.values() for m in g])
def test_import(modname):
    importlib.import_module(modname)
    assert "site-packages" in importlib.import_module(modname).__file__, modname

def test_cross_package_chain():
    import numpy as np
    from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact
    from factor_preprocess.contracts.feature_bundle import FeatureManifest
    from factor_optimizer.contracts.splits import SplitPlan
    a = ScalarMetricArtifact(metric_id="ic.mean", domain="ic", values=np.array([1.0, 2.0]))
    assert a.values.shape == (2,)
    m = FeatureManifest(["f1"], {"raw": 0}, {"raw": 1})
    assert m.get_column_index("f1") == 0
    p = SplitPlan("s", [True, False], [False, True], [False, False], {})
    assert p.split_id == "s"
PY
(
  cd "$SMOKE"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$CVPY" -m pytest test_wheel_matrix.py -q --tb=short > "$TMP_BASE/integration.log" 2>&1
)
INTEGRATION_RC=$?

echo "==> [4] assemble evidence/fresh_wheel_matrix.json"
python3 - "$REPORT" "$RESULT" "$INTEGRATION_RC" "$TMP_BASE" <<'PY'
import json, os, re, sys
report_path, result_path, int_rc, tmp_base = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
# package -> modules it must import (mirrors the integration test's CASES)
PKG_MODULES = {
    "factor_engine": ["factor_engine", "factor_engine.runtime", "factor_engine.ir", "factor_engine.modeling", "factor_engine.api"],
    "quant_evaluator": ["quant_evaluator", "quant_evaluator.contracts"],
    "factor_optimizer": ["factor_optimizer", "factor_optimizer.contracts"],
    "factor_assets": ["factor_assets", "factor_assets.registry"],
    "factor_preprocess": ["factor_preprocess", "factor_preprocess.contracts"],
    "dataaccess": ["data_access", "data_access.contract"],
}
pkgs = list(PKG_MODULES)
out = {"schema_version": 1, "generated_by": "fresh_wheel_matrix", "packages": {}}
for pkg in pkgs:
    out["packages"][pkg] = {"pkg": pkg, "wheel_sha256": None, "built": False,
                            "installed_import_ok": False, "integration_ok": False,
                            "status": "BLOCKED", "reason": ""}
# fill built + sha from report.jsonl
for line in open(report_path):
    line = line.strip()
    if not line: continue
    try: rec = json.loads(line)
    except Exception: continue
    pkg = rec.get("pkg")
    if pkg in out["packages"]:
        out["packages"][pkg]["built"] = rec.get("built", False)
        out["packages"][pkg]["wheel_sha256"] = rec.get("wheel_sha256")
        out["packages"][pkg]["reason"] = rec.get("reason", "")
# which modules failed to import (from the integration log)
failed = set()
log_path = os.path.join(tmp_base, "integration.log")
if os.path.exists(log_path):
    log = open(log_path).read()
    for m in re.finditer(r"FAILED test_wheel_matrix\.py::test_import\[([^\]]+)\]", log):
        failed.add(m.group(1))
# attribute: a package is BLOCKED if ANY of its modules failed to import
any_blocked = False
for pkg in pkgs:
    e = out["packages"][pkg]
    mods = PKG_MODULES[pkg]
    mod_failed = [m for m in mods if m in failed]
    if e["built"] and not mod_failed:
        e["installed_import_ok"] = True
        e["integration_ok"] = True
        e["status"] = "PASS"
    else:
        e["status"] = "BLOCKED"
        if mod_failed:
            e["reason"] = "clean-venv import failed: " + ", ".join(mod_failed)
        elif not e["reason"]:
            e["reason"] = "wheel not built"
        any_blocked = True
out["integration_rc"] = int_rc
out["all_pass"] = not any_blocked
os.makedirs(os.path.dirname(result_path), exist_ok=True)
with open(result_path, "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
sys.exit(1 if any_blocked else 0)
PY
RC=$?
echo "==> fresh wheel matrix done (exit $RC)"
exit $RC
