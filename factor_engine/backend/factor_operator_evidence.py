# -*- coding: utf-8 -*-
"""Evidence for Pandas/Numpy semantic-reference production operators.

Primitive triple-backend evidence covers the Daily core.  This artifact covers
all retained factor-shaped operators whose first certified production path is the
Pandas/Numpy semantic reference.  It is issued only after execution,
determinism, shape and prefix-causality audits pass.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

FE_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_PATH = FE_ROOT / "evidence" / "factor_operator_verified.json"


def _hashes() -> dict[str, str]:
    from backend.evidence_provenance import _tree_hash, compute_implementation_hash

    audit = FE_ROOT / "scripts" / "audit_all_factor_production.py"
    hardening = FE_ROOT / "cleaned_operators" / "production_hardening.py"
    return {
        "cleaned_operator_tree_hash": _tree_hash(FE_ROOT / "cleaned_operators"),
        "audit_source_hash": compute_implementation_hash(audit.read_text(encoding="utf-8")),
        "hardening_source_hash": compute_implementation_hash(hardening.read_text(encoding="utf-8")),
    }


def load_factor_operator_evidence() -> dict[str, Any]:
    if not VERIFIED_PATH.is_file():
        return {}
    try:
        payload = json.loads(VERIFIED_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def validation_errors() -> list[str]:
    payload = load_factor_operator_evidence()
    if not payload:
        return ["missing factor_operator_verified.json"]
    errors: list[str] = []
    if payload.get("artifact_kind") != "test_passed":
        errors.append("artifact_kind != test_passed")
    if not payload.get("passed_at"):
        errors.append("passed_at missing")
    if dict(payload.get("hashes") or {}) != _hashes():
        errors.append("semantic/audit source hashes are stale")

    try:
        from cleaned_operators.production_hardening import factor_production_targets
        from cleaned_operators.operator_surface import DAILY_CANONICALS
        expected = set(factor_production_targets()).difference(DAILY_CANONICALS)
    except Exception as exc:
        return errors + [f"target resolution failed: {type(exc).__name__}: {exc}"]
    observed = set(str(x) for x in (payload.get("operators") or []))
    if observed != expected:
        errors.append(
            f"operator set mismatch: missing={sorted(expected-observed)!r} "
            f"extra={sorted(observed-expected)!r}"
        )
    return errors


@lru_cache(maxsize=1)
def factor_operator_evidence_valid() -> bool:
    return not validation_errors()


@lru_cache(maxsize=256)
def pandas_reference_production_safe(canonical: str) -> bool:
    if not factor_operator_evidence_valid():
        return False
    payload = load_factor_operator_evidence()
    return str(canonical) in set(str(x) for x in (payload.get("operators") or []))


def current_hashes() -> dict[str, str]:
    return _hashes()
