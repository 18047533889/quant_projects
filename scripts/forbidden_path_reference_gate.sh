#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# P0-02 — forbidden-path-reference gate for the live dependency tree.
#
# DATAACCESS NAMING AUTHORITY:
#   * The ONLY canonical importable package name is  data_access
#     (source root: <repo>/data_access, dist: data-access).
#   * The historical bare name dataaccess (no underscore) must NOT be used
#     as a functional reference in the categories scanned below.
#
# The gate scans:
#     *.py  *.sh  *.yaml  *.yml  *.toml  *.json  *.lock  .pre-commit-config.yaml
#   under  scripts/  factor_engine/  factor_preprocess/  factor_optimizer/
#   quant_evaluator/  factor_assets/  .github/
#   plus root config CLAUDE.md requirements-production.lock .pre-commit-config.yaml
#
# EXCLUDED BY SCOPE:
#   * data_access/ itself (its internal mutation_owner / loggers / cache-dir
#     strings keep the legacy spelling by contract — the canonical python
#     package name is data_access; DataAccess-internal legacy strings are
#     above this rename board).
#   * tests/ trees (both factor_engine/tests and package tests): legacy
#     strings/test names are out of scope by design decision.
#   * evidence/ (generated attestation artifacts) and .claude/ (loop config).
#   * scripts/archive/ (historical job scripts frozen at a past state).
#   * build/ dist/ .git/ caches (generated build products).
#   * this gate script itself (it declares the forbidden token).
#
# A hit is any line that contains the bare token EXCEPT:
#   * lines that only reference our own whitelist filenames
#     (check_dataaccess_io_boundary.py — the historical checker, wired from
#     .pre-commit + ARCH-P0 tests by that exact name) and
#   * this gate script's own token declaration.
#
# Exit: 0 if no forbidden reference; 1 otherwise (fail-closed, CI).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

TOKEN="dataaccess"
# File names / invocation strings that are permitted (the historical checker
# itself + pre-commit hook id pointing at it).
ALLOW_PATTERNS=(
  "check_dataaccess_io_boundary.py"
  "check_dataaccess_io_boundary"
  "audit_fe_dataaccess_contract_drift.py"
  "audit_fe_dataaccess_contract_drift"
)
# The gate's own file — declares the token; excluded.
SELF="forbidden_path_reference_gate.sh"
# Generated build products — never scanned.
SKIP_DIRS=("/build/" "/dist/" "/.git/" "/__pycache__/" "/.pytest_cache/" "/archive/")

FILES="$( {
  find scripts factor_engine factor_preprocess factor_optimizer \
       quant_evaluator factor_assets .github -type f \( \
         -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' \
         -o -name '*.toml' -o -name '*.json' -o -name '*.lock' \) 2>/dev/null
  for f in CLAUDE.md requirements-production.lock .pre-commit-config.yaml; do
    [ -f "$f" ] && echo "$f"
  done
} | grep -v "$SELF" | grep -v '/tests/' )"

bad=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  skip=0
  for d in "${SKIP_DIRS[@]}"; do
    case "$f" in
      *"$d"*) skip=1 ;;
    esac
  done
  [ "$skip" -eq 1 ] && continue
  file -b "$f" 2>/dev/null | grep -q 'text' || continue
  while IFS=: read -r _ln line; do
    matched=1
    for a in "${ALLOW_PATTERNS[@]}"; do
      if [[ "$line" == *"$a"* ]]; then
        matched=0
        break
      fi
    done
    if [ "$matched" -eq 1 ]; then
      echo "P0-02 FORBIDDEN data_access ref: $f:$line"
      bad=1
    fi
  done < <(grep -n "$TOKEN" "$f" || true)
done <<< "$FILES"

if [ "$bad" -ne 0 ]; then
  echo "PATH_GATE: FAIL (bare 'dataaccess' references remain in the live tree)"
  exit 1
fi
echo "PATH_GATE: PASS (no bare 'dataaccess' references in the live tree)"
exit 0