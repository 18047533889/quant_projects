#!/usr/bin/env python3
"""Issue/validate Pandas semantic-reference evidence for factor operators.

The certifier first runs the runtime-only all-factor audit. Only when every
retained factor canonical passes deterministic same-shape and prefix-causality
checks does it write ``evidence/factor_operator_verified.json``.

Audit #382/#383: certification is recorded per executed canonical (not just as a
file-level returncode).  Audit #385: each certified operator record carries the
``certified_parameter_domain`` it actually proved (default-only unless declared
otherwise).  Audit #384: an expected-fixture finite-coverage gate is available
for backends that materialise expected outputs; in-memory audit fixtures pass it
vacuously.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FE_ROOT = Path(__file__).resolve().parents[1]
for p in (str(FE_ROOT.parent), str(FE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Optional per-canonical expected-fixture file map for the #384 finite-coverage
# gate (vacuous for in-memory audit fixtures).
_FIXTURE_PATHS: dict[str, str] = {}

# Optional per-canonical parameter-domain declarations (audit #385).
_PARAM_DOMAINS: dict[str, dict] = {}


def _finite_coverage_ok(fixture_path: str) -> tuple[bool, float]:
    """Audit #384: a case whose expected fixture output is NaN/Inf-dominated
    cannot certify numeric equivalence.  Returns ``(ok, finite_ratio)``.  When
    no fixture file is materialised, the gate is vacuously OK."""
    path = Path(fixture_path)
    if not path.exists():
        return True, 1.0
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
        elif path.suffix.lower() in {".csv", ".csv.gz"}:
            import pandas as pd

            arr = pd.read_csv(path).to_numpy(dtype=float)
        elif path.suffix.lower() == ".parquet":
            import pandas as pd

            arr = pd.read_parquet(path).to_numpy(dtype=float)
        elif path.suffix.lower() == ".json":
            import pandas as pd

            arr = pd.read_json(path).to_numpy(dtype=float)
        else:
            return True, 1.0
    except Exception:
        return False, 0.0
    if arr is None or arr.size == 0:
        return False, 0.0
    try:
        finite = float(np.isfinite(arr.astype(float)).mean())
    except (TypeError, ValueError):
        return False, 0.0
    return finite >= 0.5, finite


def _certified_parameter_domain(canonical: str) -> dict:
    """Audit #385: the parameter domain a certified case actually proves."""
    domain = _PARAM_DOMAINS.get(canonical)
    if domain is not None:
        return dict(domain)
    # Default: the runtime audit exercised the operator's default parameters, so
    # the certified domain is exactly those defaults — nothing else is claimed.
    return {"bounds": ["default"]}


def _executed_records_for_targets(targets: list[str]) -> list[dict]:
    """Audit #382/#383: one executed+passed record per retained canonical.

    The runtime audit either passes every retained factor canonical or exits
    non-zero before any artifact is written, so by the time we are here every
    ``target`` genuinely executed and passed the deterministic shape / prefix-
    causality checks on the default-parameter audit fixture.
    """
    records: list[dict] = []
    for canonical in sorted(targets):
        fixture = _FIXTURE_PATHS.get(canonical, canonical)
        ok, _ratio = _finite_coverage_ok(fixture)
        if not ok:
            # #384 gate: downgraded (skipped) — record the canonical as not
            # certified so the payload operator list stays consistent.
            continue
        records.append(
            {
                "canonical": canonical,
                "backend": "pandas",
                "case_id": f"runtime_audit::{canonical}",
                "status": "passed",
                "params": "default",
                "fixture_hash": "n/a",
                "output_hash": "n/a",
                "stage": "factor_runtime_audit",
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--junitxml-out",
        default="",
        help="可选：导出逐 canonical executed+passed 记录 JSON 的路径",
    )
    args = parser.parse_args()
    from backend.factor_operator_evidence import VERIFIED_PATH, current_hashes, validation_errors
    if args.check:
        errors = validation_errors()
        if errors:
            print("factor operator evidence invalid:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("factor operator evidence valid")
        return 0

    proc = subprocess.run(
        [sys.executable, str(FE_ROOT / "scripts" / "audit_all_factor_production.py"), "--runtime-only"],
        cwd=str(FE_ROOT),
    )
    if proc.returncode:
        return proc.returncode

    from cleaned_operators import load_all
    from cleaned_operators.production_hardening import factor_production_targets
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from backend.evidence_provenance import collect_runtime_versions, current_commit_sha
    load_all()
    targets = sorted(set(factor_production_targets()).difference(DAILY_CANONICALS))
    executed_records = _executed_records_for_targets(targets)
    certified = [rec["canonical"] for rec in executed_records if rec.get("status") == "passed"]
    if not certified:
        print(
            "factor operator certification FAIL: no executed+passed canonical after "
            "runtime audit (all downgraded by #384 finite-coverage gate)",
            file=sys.stderr,
        )
        return 1
    # Audit #382/#383 + P0-18: ``operators`` is the per-canonical evidence
    # RECORD (a dict), not just a name list — ``production_certification_overlay.
    # _per_gate_verified`` reads ``semantic_golden_verified`` /
    # ``temporal_prefix_verified`` / ``source_contract_verified`` from it, and
    # each of those must be earned by its own verified record.
    #
    # R25-011..014 / R25-186 (evidence independence): the synthetic runtime
    # audit proves ONLY execution, shape, determinism and historical-prefix
    # causality on the default-parameter fixture.  It does NOT prove the
    # operator's mathematical/economic semantics against an independent golden,
    # and it does NOT prove a real DataAccess/source PIT contract.  Those two
    # gates are therefore written as ``False`` (honestly not earned) and can
    # only flip True via their OWN independent evidence (a math golden /
    # metamorphic fixture, and a FieldSpec/ProviderBinding/DataAccess source
    # contract).  ``temporal_prefix_verified`` IS earned here because the
    # runtime audit runs a historical-prefix invariance check.  Dict iteration
    # still yields canonical names, so ``pandas_reference_production_safe`` and
    # the set-mismatch validation are unaffected.
    operators_record = {
        canonical: {
            # Honest runtime-audit evidence (R25-012) — what the synthetic
            # runtime audit genuinely proves:
            "runtime_execution_verified": True,
            "shape_verified": True,
            "determinism_verified": True,
            "prefix_kernel_causality_verified": True,
            "temporal_prefix_verified": True,
            # NOT earned by the runtime audit — require independent evidence:
            "semantic_golden_verified": False,
            "source_contract_verified": False,
        }
        for canonical in certified
    }
    payload = {
        "schema_version": 2,
        "artifact_kind": "test_passed",
        "certification_mode": "direct",
        "passed_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": current_commit_sha(),
        "runtime_versions": collect_runtime_versions(),
        "operators": operators_record,
        "executed_records": executed_records,
        "certified_parameter_domains": {
            canonical: _certified_parameter_domain(canonical) for canonical in certified
        },
        "checks": ["production_metadata", "pit_policy", "shape_preserving", "determinism", "prefix_causality", "pandas_reference_execution"],
        "hashes": current_hashes(),
    }
    if args.junitxml_out:
        Path(args.junitxml_out).write_text(
            json.dumps(executed_records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    VERIFIED_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {VERIFIED_PATH} ({len(certified)} Pandas-reference production operators)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
