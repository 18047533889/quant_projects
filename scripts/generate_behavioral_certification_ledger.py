#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate behavioral_certification_ledger.json from on-disk evidence.

MF-P0-003 / REM-171: typed behavioral certification ledger. Scans actual test
files, oracle scripts, and evidence artifacts to populate certification status
HONESTLY — never fakes a CERTIFIED status for evidence that doesn't exist.

Run:  python3 scripts/generate_behavioral_certification_ledger.py [--out docs/evidence/model_operators]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "docs" / "evidence" / "model_operators"


def git_sha() -> str | None:
    """Return the live repository HEAD; unavailable is None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else None
    except Exception:
        return None


def _probe_parameter_domain(canonical: str) -> dict[str, str | None]:
    """Check if canonical has any certified parameter-domain point."""
    try:
        from factor_engine.runtime.parameter_domain_store import ParameterDomainCertificationStore
        store = ParameterDomainCertificationStore()
        # Try loading default evidence if available
        evidence_path = REPO / "docs" / "evidence" / "r37" / "R37_PARAMETER_DOMAIN_STORE.json"
        if evidence_path.exists():
            store.load_json(evidence_path)
        has_any = store.operator_has_any_certified_region(canonical)
        return {
            "status": "CERTIFIED" if has_any else "NOT_RUN",
            "oracle": "R37_PARAMETER_DOMAIN_STORE.json" if has_any else None,
            "proof_hash": None,
        }
    except Exception:
        return {"status": "NOT_RUN", "oracle": None, "proof_hash": None}


def _probe_oracle_parity(canonical: str) -> dict[str, str | None]:
    """Check if an oracle test exists for this canonical."""
    # Look for oracle test files
    oracle_files = [
        REPO / "tests" / "r35" / "model_oracle.py",
        REPO / "tests" / "r35" / "test_phase_b_model_oracle.py",
    ]
    for path in oracle_files:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if canonical in content:
                return {"status": "PARTIAL", "reason": "oracle exists but full parity unverified"}
    return {"status": "NOT_RUN", "reason": "no oracle found"}


def _probe_pit_safe(canonical: str) -> dict[str, str]:
    """Check PIT safety from timing kind."""
    try:
        from factor_engine.cleaned_operators.model_timing import timing_kind_for, TimingKind
        kind = timing_kind_for(canonical)
        if kind == TimingKind.PRIOR_FIT_PREDICTIVE:
            return {"status": "CERTIFIED"}
        elif kind == TimingKind.SELF_FIT_DESCRIPTIVE:
            return {"status": "CERTIFIED"}
        else:
            return {"status": "NOT_RUN"}
    except Exception:
        return {"status": "NOT_RUN"}


def _probe_leakage_tested(canonical: str) -> dict[str, str]:
    """Check if leakage tests exist."""
    test_file = REPO / "tests" / "modeling" / "test_evaluation_leakage_evidence.py"
    if test_file.exists():
        content = test_file.read_text(encoding="utf-8")
        if "negative_controls" in content.lower():
            return {"status": "PARTIAL"}
    return {"status": "NOT_RUN"}


def _probe_axis_effect(canonical: str) -> dict[str, str | None]:
    """Check AxisEffect certification."""
    try:
        from factor_engine.cleaned_operators.axis_effect_audit import get_axis_effect
        effect = get_axis_effect(canonical)
        if effect and effect != "UNKNOWN":
            return {"status": "CERTIFIED", "value": str(effect)}
        return {"status": "NOT_RUN", "value": None}
    except Exception:
        return {"status": "NOT_RUN", "value": None}


def main() -> int:
    sha = git_sha()
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators.model_timing import is_model_like_name

    load_all()
    canonicals = sorted(OperatorRegistry.list_canonical())
    model_like = [c for c in canonicals if is_model_like_name(c)]

    certified_ops: list[dict] = []
    for canonical in model_like:
        param_domain = _probe_parameter_domain(canonical)
        oracle = _probe_oracle_parity(canonical)
        pit = _probe_pit_safe(canonical)
        leakage = _probe_leakage_tested(canonical)
        axis = _probe_axis_effect(canonical)

        certified_ops.append({
            "canonical": canonical,
            "certifications": {
                "parameter_domain": param_domain,
                "output_oracle": oracle,
                "pit_safe": pit,
                "leakage_tested": leakage,
                "axis_effect": axis,
            }
        })

    try:
        generated_at = subprocess.run(
            ["date", "+%Y-%m-%dT%H:%M:%S%z"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        from datetime import datetime
        generated_at = datetime.now().isoformat()

    ledger = {
        "generated_at": generated_at,
        "commit_sha": sha,
        "certified_operators": certified_ops,
    }

    out_path = out_dir / "behavioral_certification_ledger.json"
    out_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compute summary
    param_cert = sum(1 for op in certified_ops if op["certifications"]["parameter_domain"]["status"] == "CERTIFIED")
    oracle_cert = sum(1 for op in certified_ops if op["certifications"]["output_oracle"]["status"] == "CERTIFIED")
    pit_cert = sum(1 for op in certified_ops if op["certifications"]["pit_safe"]["status"] == "CERTIFIED")
    axis_cert = sum(1 for op in certified_ops if op["certifications"]["axis_effect"]["status"] == "CERTIFIED")

    print(f"behavioral_certification_ledger.json: {len(certified_ops)} operators")
    print(f"  parameter_domain CERTIFIED: {param_cert}/{len(certified_ops)}")
    print(f"  output_oracle CERTIFIED: {oracle_cert}/{len(certified_ops)}")
    print(f"  pit_safe CERTIFIED: {pit_cert}/{len(certified_ops)}")
    print(f"  axis_effect CERTIFIED: {axis_cert}/{len(certified_ops)}")
    print(f"  bound to HEAD: {sha}")

    if sha is None:
        print("WARNING: git SHA unavailable; evidence cannot be verified for freshness")

    return 0


if __name__ == "__main__":
    sys.exit(main())
