#!/usr/bin/env bash
# R46 — fresh wheel matrix: build all SIX wheels from THEIR OWN canonical
# pyproject.toml, install ONLY the lockfile deps via `--require-hashes` into a
# clean venv, then run wheel-only recursive-import + __file__ residency +
# cross-package contract checks.
#
# HONEST GATE: `all_pass` requires all(pkg.status == PASS) AND integration_rc == 0.
# If the cross-package integration fails (or any package's wheel cannot build or
# import), NO package may be reported PASS — every package is BLOCKED with the
# honest reason. A BLOCKED package is never silently upgraded to PASS.
#
# Local-only: no git mutations, no push, no evidence/ edits by this agent beyond
# writing fresh_wheel_matrix.json.
set -uo pipefail   # NOT -e: we record per-package BLOCKED instead of aborting

REPO="${REPO:-/home/sunhaiwei/quant_projects}"
PY="${PY:-/tmp/fe2/bin/python}"
LOCK="$REPO/requirements-production.lock"
TMP_BASE="${TMP_BASE:-$(mktemp -d -t fwm.XXXXXX)}"
DIST="$TMP_BASE/dist"
SMOKE="$TMP_BASE/smoke"
mkdir -p "$DIST" "$SMOKE"

RESULT="$REPO/evidence/fresh_wheel_matrix.json"
REPORT="$TMP_BASE/report.jsonl"
: > "$REPORT"

echo "==> [0] env"
echo "    PY=$PY  ($("$PY" --version 2>&1))"
echo "    lock: $LOCK"
test -f "$LOCK" || { echo "FATAL: lockfile missing"; exit 2; }

echo "==> [1] building the SIX canonical wheels from their own pyprojects"
declare -A WHEEL_SHA
declare -A BUILT
declare -A REASON

# pkg  build_dir  distribution-name  dist-normalized-for-grep
PKGS=(
  "factor_engine    $REPO/factor_engine        factor-engine    factor_engine"
  "quant_evaluator  $REPO/quant_evaluator      quant-evaluator  quant_evaluator"
  "factor_optimizer $REPO/factor_optimizer     factor-optimizer factor_optimizer"
  "factor_assets    $REPO/factor_assets        factor-assets    factor_assets"
  "factor_preprocess $REPO/factor_preprocess   factor-preprocess factor_preprocess"
  "data_access      $REPO/data_access           data-access      data_access"
)

build_one() {  # $1=pkg  $2=build_dir  $3=dist-name  $4=grep-token
  local pkg="$1" bdir="$2" dist="$3" token="$4"
  echo "    building $pkg  (from $bdir/pyproject.toml)"
  if [ ! -f "$bdir/pyproject.toml" ]; then
    BUILT["$pkg"]=false; REASON["$pkg"]="canonical pyproject missing at $bdir/pyproject.toml"
    echo "{\"pkg\":\"$pkg\",\"built\":false,\"wheel_sha256\":null,\"reason\":\"${REASON[$pkg]}\"}" >> "$REPORT"
    return
  fi
  if ( cd "$bdir" && "$PY" -m pip wheel --no-deps --no-build-isolation -w "$DIST" . ) >"$TMP_BASE/build_$pkg.log" 2>&1; then
    local w; w=$(ls "$DIST"/*.whl 2>/dev/null | grep -iE "$token|$dist" | head -1)
    if [ -n "$w" ]; then
      WHEEL_SHA["$pkg"]=$(sha256sum "$w" | cut -d' ' -f1)
      BUILT["$pkg"]=true
      echo "      ok: $(basename "$w") sha256=${WHEEL_SHA[$pkg]:0:16}…"
      echo "{\"pkg\":\"$pkg\",\"built\":true,\"wheel_sha256\":\"${WHEEL_SHA[$pkg]}\",\"reason\":\"built from $bdir/pyproject.toml\"}" >> "$REPORT"
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

while IFS= read -r line; do
  [ -z "${line// }" ] && continue
  set -- $line
  build_one "$1" "$2" "$3" "$4"
done <<< "$(printf '%s\n' "${PKGS[@]}")"

echo "==> [2] clean venv; install SIX built wheels (--no-deps) + lockfile deps (--require-hashes)"
"$PY" -m virtualenv "$TMP_BASE/venv" >"$TMP_BASE/venv.log" 2>&1 || { echo "venv create failed: $(tail -3 "$TMP_BASE/venv.log")"; exit 1; }
CVPY="$TMP_BASE/venv/bin/python"
echo "    installing six built wheels --no-deps"
"$CVPY" -m pip install -q --no-deps "$DIST"/*.whl
echo "    installing lockfile deps ONLY (--require-hashes); no pip upgrade, no unpinned extras"
# The lock contains six internal dist lines (artifacts built fresh above — their
# hashes are verified at build time, not baked in the lock) plus a python==
# marker. Filter those informational lines so pip sees ONLY the hashed
# third-party requirements. Nothing is dropped silently: every real third-party
# pin in the lock (with --hash) is installed as-is.
awk '
  /^#/ { next }
  /^python==/ { next }
  /^data-access==/ || /^factor-engine==/ || /^quant-evaluator==/ || \
  /^factor-optimizer==/ || /^factor-assets==/ || /^factor-preprocess==/ { next }
  { print }
' "$LOCK" > "$TMP_BASE/thirdparty-only.txt"
"$CVPY" -m pip install -q --require-hashes -r "$TMP_BASE/thirdparty-only.txt" >"$TMP_BASE/install.log" 2>&1
INSTALL_RC=$?
if [ "$INSTALL_RC" -ne 0 ]; then
  echo "    third-party lock install FAILED (rc=$INSTALL_RC): $(tail -5 "$TMP_BASE/install.log" | tr '\n' ' ')"
fi

echo "==> [3] wheel-only recursive-import + residency + cross-package integration"
# Standalone driver (no pytest): keeps the venv install STRICTLY lock-only,
# with no unpinned test-runner extras. Emits machine-parseable lines:
#   IMPORT_FAIL <modname>   -- import missing / not from site-packages
#   CHAIN_FAIL              -- cross-package contract chain failed
#   RESULT <rc>
cat > "$SMOKE/run_integration.py" <<'PY'
import importlib, sys, os
# No repo root on sys.path: every import must resolve from site-packages only.
leak = [p for p in sys.path if ("quant_projects" in p and "site-packages" not in p)]
if leak:
    print(f"PATH_LEAK {leak}")
    sys.exit(1)
CASES = {
    "factor_engine": ["factor_engine", "factor_engine.runtime", "factor_engine.ir", "factor_engine.modeling", "factor_engine.api"],
    "quant_evaluator": ["quant_evaluator", "quant_evaluator.contracts"],
    "factor_optimizer": ["factor_optimizer", "factor_optimizer.contracts"],
    "factor_assets": ["factor_assets", "factor_assets.registry"],
    "factor_preprocess": ["factor_preprocess", "factor_preprocess.contracts"],
    "data_access": ["data_access", "data_access.contract"],
}
rc = 0
for pkg, mods in CASES.items():
    for m in mods:
        try:
            mod = importlib.import_module(m)
        except Exception as e:
            print(f"IMPORT_FAIL {m} :: {type(e).__name__}: {e}")
            rc = 1
            continue
        f = getattr(mod, "__file__", None)
        if not f or "site-packages" not in f:
            print(f"IMPORT_FAIL {m} :: not from site-packages: {f}")
            rc = 1
# Cross-package contract chain
try:
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
    print("CHAIN_OK")
except Exception as e:
    print(f"CHAIN_FAIL :: {type(e).__name__}: {e}")
    rc = 1
print(f"RESULT {rc}")
sys.exit(rc)
PY
(
  cd "$SMOKE"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$CVPY" run_integration.py > "$TMP_BASE/integration.log" 2>&1
)
INTEGRATION_RC=$?
echo "    integration_rc=$INTEGRATION_RC"
echo "    --- integration.log ---"
cat "$TMP_BASE/integration.log"
echo "    ------------------------"

echo "==> [4] assemble evidence/fresh_wheel_matrix.json"
python3 - "$REPORT" "$RESULT" "$INTEGRATION_RC" "$INSTALL_RC" "$TMP_BASE" <<'PY'
import json, os, re, sys
report_path, result_path, int_rc, inst_rc, tmp_base = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
PKG_MODULES = {
    "factor_engine": ["factor_engine", "factor_engine.runtime", "factor_engine.ir", "factor_engine.modeling", "factor_engine.api"],
    "quant_evaluator": ["quant_evaluator", "quant_evaluator.contracts"],
    "factor_optimizer": ["factor_optimizer", "factor_optimizer.contracts"],
    "factor_assets": ["factor_assets", "factor_assets.registry"],
    "factor_preprocess": ["factor_preprocess", "factor_preprocess.contracts"],
    "data_access": ["data_access", "data_access.contract"],
}
pkgs = list(PKG_MODULES)
out = {"schema_version": 1, "generated_by": "fresh_wheel_matrix",
       "r46_note": "Six canonical wheels from their own pyprojects; lock deps installed ONLY via --require-hashes",
       "packages": {}}
for pkg in pkgs:
    out["packages"][pkg] = {"pkg": pkg, "wheel_sha256": None, "built": False,
                            "installed_import_ok": False, "integration_ok": False,
                            "status": "BLOCKED", "reason": ""}
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

failed = set()
log_path = os.path.join(tmp_base, "integration.log")
if os.path.exists(log_path):
    log = open(log_path).read()
    for line in log.splitlines():
        if line.startswith("IMPORT_FAIL "):
            mod = line.split()[1]
            failed.add(mod)
        elif line.startswith("CHAIN_FAIL"):
            failed.add("__cross_package_chain__")

# HONEST GATE: a package is PASS only if it built AND all its modules import
# from site-packages AND there are no cross-package failures at all AND the
# lock install succeeded. If the whole gate fails, every package is BLOCKED.
gate_failed_reasons = []
if inst_rc != 0:
    gate_failed_reasons.append(f"third-party lock install failed (rc={inst_rc})")
if int_rc != 0:
    gate_failed_reasons.append(f"cross-package integration failed (rc={int_rc})")

any_blocked = False
for pkg in pkgs:
    e = out["packages"][pkg]
    mods = PKG_MODULES[pkg]
    mod_failed = [m for m in mods if m in failed]
    chain_failed = "__cross_package_chain__" in failed
    if e["built"] and not mod_failed and not chain_failed and not gate_failed_reasons:
        e["installed_import_ok"] = True
        e["integration_ok"] = True
        e["status"] = "PASS"
    else:
        e["status"] = "BLOCKED"
        if gate_failed_reasons:
            e["reason"] = "; ".join(gate_failed_reasons)
        elif chain_failed:
            e["reason"] = "cross-package contract chain FAILED"
        elif mod_failed:
            e["reason"] = "clean-venv import failed: " + ", ".join(mod_failed)
        elif not e["reason"]:
            e["reason"] = "wheel not built"
        any_blocked = True

out["integration_rc"] = int_rc
out["all_pass"] = (not any_blocked) and (int_rc == 0) and (inst_rc == 0)
os.makedirs(os.path.dirname(result_path), exist_ok=True)
with open(result_path, "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
sys.exit(1 if (not out["all_pass"]) else 0)
PY
RC=$?
echo "==> fresh wheel matrix done (exit $RC)"
exit $RC
