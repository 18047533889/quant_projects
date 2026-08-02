#!/usr/bin/env python3
"""Issue/validate Pandas semantic-reference evidence for factor operators.

The certifier first runs the runtime-only all-factor audit. Only when every
retained factor canonical passes deterministic same-shape and prefix-causality
checks does it write ``evidence/factor_operator_verified.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FE_ROOT=Path(__file__).resolve().parents[1]
for p in (str(FE_ROOT.parent),str(FE_ROOT)):
    if p not in sys.path:sys.path.insert(0,p)


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    from backend.factor_operator_evidence import VERIFIED_PATH,current_hashes,validation_errors
    if args.check:
        errors=validation_errors()
        if errors:
            print("factor operator evidence invalid:",file=sys.stderr)
            for error in errors:print(f"- {error}",file=sys.stderr)
            return 1
        print("factor operator evidence valid")
        return 0

    proc=subprocess.run(
        [sys.executable,str(FE_ROOT/"scripts"/"audit_all_factor_production.py"),"--runtime-only"],
        cwd=str(FE_ROOT),
    )
    if proc.returncode:return proc.returncode

    from cleaned_operators import load_all
    from cleaned_operators.production_hardening import factor_production_targets
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from backend.evidence_provenance import collect_runtime_versions,current_commit_sha
    load_all()
    targets=sorted(set(factor_production_targets()).difference(DAILY_CANONICALS))
    payload={
        "schema_version":1,
        "artifact_kind":"test_passed",
        "passed_at":datetime.now(timezone.utc).isoformat(),
        "commit_sha":current_commit_sha(),
        "runtime_versions":collect_runtime_versions(),
        "operators":targets,
        "checks":["production_metadata","pit_policy","shape_preserving","determinism","prefix_causality","pandas_reference_execution"],
        "hashes":current_hashes(),
    }
    VERIFIED_PATH.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"wrote {VERIFIED_PATH} ({len(targets)} Pandas-reference production operators)")
    return 0

if __name__=="__main__":raise SystemExit(main())
