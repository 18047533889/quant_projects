#!/usr/bin/env python3
"""Run real three-backend recipe tests and issue a fail-closed artifact."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FE_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    from backend.recipe_evidence import (
        CASE_PATH, VERIFIED_PATH, current_hashes, recipe_evidence_valid,
    )
    if args.check:
        if not recipe_evidence_valid(require_commit_ancestor=True):
            print("recipe_verified.json invalid or stale", file=sys.stderr)
            return 1
        print("recipe evidence valid")
        return 0
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/operators/test_recipe_backend_certification.py"],
        cwd=FE_ROOT,
    )
    if proc.returncode:
        return proc.returncode
    cases = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=FE_ROOT).decode().strip()
    payload = {
        "schema_version": 1,
        "artifact_kind": "test_passed",
        "commit_sha": commit,
        "passed_at": datetime.now(timezone.utc).isoformat(),
        "recipes": sorted(cases.get("recipes") or []),
        "backends": ["pandas_numpy", "polars_long_native", "duckdb_real_sql"],
        "edge_cases": ["null", "nan", "zero_denominator", "short_window"],
        "no_fallback_verified": True,
        "hashes": current_hashes(),
    }
    VERIFIED_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {VERIFIED_PATH} ({len(payload['recipes'])} recipes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
