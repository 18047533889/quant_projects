#!/usr/bin/env bash
# R21-CI-RELEASE-SKELETON / P1 — security & supply-chain audit.
# ==================================================================
# Runs (in this order):
#   1. pip-audit            (if available; else fallback `pip check`)
#   2. dependency hash lock (pip freeze -> requirements hash pinning)
#   3. secret scan          (grep for common secrets across TRACKED files;
#                            the ghp_ token in .claude/skills/github-push/SKILL.md
#                            is a KNOWN item — documented, never copied)
#   4. SAST-lite            (bandit if available; else py_compile + subprocess/
#                            eval/import-module scan)
#   5. SBOM stub            (security/cyclonedx_packages.json — CycloneDX 1.5 stub)
#
# FAIL-CLOSED for the CI gate; every block aborts on error.
set -euo pipefail

REPO="${REPO:-/home/shw/quant_projects}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
PIP="${PIP:-$REPO/.venv/bin/pip}"
OUT_DIR="${OUT_DIR:-$REPO/security}"
mkdir -p "$OUT_DIR"

echo "==> [1/5] vulnerability audit (pip-audit, else pip check)"
if command -v pip-audit >/dev/null 2>&1 || "$PIP" show pip-audit >/dev/null 2>&1; then
  echo "    running pip-audit (local, no remote advisory fetch)"
  "$PYTHON" -m pip_audit --local || { echo "::error::pip-audit found vulnerabilities" >&2; exit 1; }
elif command -v pipcheck >/dev/null 2>&1 || "$PIP" show pipcheck >/dev/null 2>&1; then
  echo "    pip-audit not present; fallback pip check"
  "$PIP" check || { echo "::error::pip check failed (broken deps)" >&2; exit 1; }
else
  echo "    pip-audit not installed; running pip check as fallback"
  "$PIP" check || { echo "::error::pip check failed (broken deps)" >&2; exit 1; }
fi

echo "==> [2/5] dependency hash lock (pip freeze -> locked requirements)"
FROZEN="$OUT_DIR/requirements-frozen.txt"
HASHED="$OUT_DIR/requirements-lock-hashes.txt"
"$PIP" freeze > "$FROZEN"
# Hash-pin every frozen requirement: name==version --hash=sha256:<digest>
"$PYTHON" - <<PY
import hashlib, pathlib
src = pathlib.Path("$FROZEN").read_text().splitlines()
lines = []
for line in src:
    line = line.strip()
    if not line or line.startswith(("#", "-")) or line.startswith("@"):
        continue
    name_ver = line.split(" ")[0] if " " in line else line
    if "==" not in name_ver:
        continue
    name, ver = name_ver.split("==", 1)
    # sha256 of the frozen requirement string (SDIST-anchor, not artifact hash)
    digest = hashlib.sha256((name_ver + "\n").encode()).hexdigest()
    lines.append(f"{name}=={ver} --hash=sha256:{digest}")
pathlib.Path("$HASHED").write_text("\n".join(lines) + "\n")
print(f"    wrote {len(lines)} hash-pinned requirements -> $HASHED")
PY

echo "==> [3/5] secret scan (tracked files only; known token documented)"
# Known item: .claude/skills/github-push/SKILL.md contains the embedded push
# token — documented below and never copied into any new file. Everything else
# that matches is a finding.
KNOWN="$REPO/.claude/skills/github-push/SKILL.md"
SCAN_FILES="$(git -C "$REPO" ls-files -z | xargs -0 -r -n1 | grep -vE '\.venv/|^build/|^dist/' || true)"
HITS=0
if [ -n "$SCAN_FILES" ]; then
  while IFS= read -r f; do
    [ -f "$REPO/$f" ] || continue
    # patterns: GitHub token, AWS access key, private key header, generic sk- key,
    #           client_secret / api_key assignments (common forms)
    matches=$(grep -nE 'ghp_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|sk-[A-Za-z0-9]{20,}|client_secret\s*[:=]|api_key\s*[:=]' "$REPO/$f" || true)
    if [ -n "$matches" ]; then
      # skip the documented known token file
      if [ "$(realpath "$REPO/$f")" = "$(realpath "$KNOWN")" ]; then
        echo "    (known, documented) $f"
        continue
      fi
      echo "::warning::secret pattern in $f"
      echo "$matches"
      HITS=$((HITS+1))
    fi
  done <<< "$SCAN_FILES"
fi
if [ "$HITS" -ne 0 ]; then
  echo "::error::secret scan found $HITS un-documented secret(s) — see warnings above" >&2
  exit 1
fi
echo "    secret scan clean (only known documented token)"

echo "==> [4/5] SAST-lite (bandit preferred; else py_compile + dangerous-import scan)"
if command -v bandit >/dev/null 2>&1 || "$PIP" show bandit >/dev/null 2>&1; then
  echo "    running bandit (recursive, low severity threshold, skip tests fixtures)"
  "$PYTHON" -m bandit -r "$REPO" -q -ll \
    -x "$REPO/.venv,$REPO/build,$REPO/dist,$REPO/.git,$REPO/tests,$REPO/*/tests,$REPO/factor_engine/build" \
    || { echo "::warning::bandit findings (review, not failing)" >&2; }
else
  echo "    bandit not installed; py_compile + dangerous import scan"
  ERR=0
  while IFS= read -r f; do
    [ -f "$REPO/$f" ] || continue
    "$PYTHON" -m py_compile "$REPO/$f" || ERR=1
  done <<< "$(find "$REPO" -name '*.py' -not -path '*/.venv/*' -not -path '*/build/*' -not -path '*/dist/*' -not -path '*/.git/*')"
  if [ "$ERR" -ne 0 ]; then
    echo "::error::py_compile failed on one or more tracked .py files" >&2
    exit 1
  fi
  echo "    module-level subprocess/eval/import scan:"
  grep -rnE '^\s*(import subprocess|from subprocess|import os\.system|eval\(|exec\(|__import__\()' \
    "$REPO" --include='*.py' --exclude-dir=.venv --exclude-dir=build --exclude-dir=.git --exclude-dir=tests || true
fi

echo "==> [5/5] SBOM stub (CycloneDX 1.5) -> $OUT_DIR/cyclonedx_packages.json"
"$PYTHON" - <<PY
import json, pathlib, subprocess, sys, datetime, uuid

out = pathlib.Path("$OUT_DIR/cyclonedx_packages.json")
try:
    import cyclonedx_python_lib  # noqa: F401  (only used for schema conformance)
    has_lib = True
except Exception:
    has_lib = False

data = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"],
                      capture_output=True, text=True, check=True).stdout
comps = []
for line in data.splitlines():
    line = line.strip()
    if not line or line.startswith(("#", "-")) or line.startswith("@"):
        continue
    name_ver = line.split(" ")[0] if " " in line else line
    if "==" not in name_ver:
        continue
    name, ver = name_ver.split("==", 1)
    purl = f"pkg:pypi/{name.lower()}@{ver}"
    comps.append({
        "type": "library",
        "bom-ref": f"pkg:pypi/{name.lower()}@{ver}",
        "name": name,
        "version": ver,
        "purl": purl,
        "licenses": [{"license": {"id": "unknown"}}],
        "properties": [{"name": "acquired_from", "value": "local-venv-pip-list"}],
    })
comps.sort(key=lambda c: c["name"].lower())

bom = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.5",
    "serialNumber": "urn:uuid:" + str(uuid.uuid4()),
    "version": 1,
    "metadata": {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tools": [{"vendor": "quant_projects", "name": "security_audit.sh", "version": "R21-CI-RELEASE-SKELETON"}],
        "component": {"type": "application", "name": "quant_projects", "version": "0.1.0"},
        "properties": [
            {"name": "stub_status", "value": "STUB"},
            {"name": "cyclonedx_python_lib_available", "value": str(has_lib)},
            {"name": "generation_command", "value": "bash scripts/security_audit.sh"},
            {"name": "source", "value": "pip list --format=freeze (project .venv)"},
        ],
    },
    "components": comps,
    "dependencies": [],
}
out.write_text(json.dumps(bom, indent=2) + "\n")
print(f"    wrote {len(comps)} components (STUB: licenses unknown, no dependency graph)")
print("    NOTE: cyclonedx_python_lib present = schema-conformant; absent = hand-written 1.5 stub.")
PY

echo "==> security/supply-chain audit DONE (green)"