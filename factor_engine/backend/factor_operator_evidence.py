# -*- coding: utf-8 -*-
"""Test-backed evidence for Pandas/Numpy semantic-reference factor operators.

The artifact is bound not only to numerical kernels but also to authoring,
lowering, PIT source resolution, warmup, incremental replay and backend-admission
semantics. Generated docs/manifests are excluded to avoid self-referential
certification loops.
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
    files = {
        "operator_capability_hash": FE_ROOT / "backend" / "operator_capability.py",
        "pandas_signature_hash": FE_ROOT / "backend" / "pandas_first_signature.py",
        "execution_context_hash": FE_ROOT / "backend" / "context.py",
        "production_policy_hash": FE_ROOT / "runtime" / "production_policy.py",
        "incremental_hash": FE_ROOT / "runtime" / "incremental.py",
        "warmup_hash": FE_ROOT / "runtime" / "warmup_service.py",
        "run_window_hash": FE_ROOT / "runtime" / "run_window.py",
        "source_window_contract_hash": FE_ROOT / "runtime" / "source_window_contract_v2.py",
        "time_window_hash": FE_ROOT / "storage" / "time_window.py",
    }
    hashes = {
        "cleaned_operator_python_tree_hash": _tree_hash(
            FE_ROOT / "cleaned_operators", patterns=("*.py",)
        ),
        "api_python_tree_hash": _tree_hash(FE_ROOT / "api", patterns=("*.py",)),
        "ir_python_tree_hash": _tree_hash(FE_ROOT / "ir", patterns=("*.py",)),
        "planner_python_tree_hash": _tree_hash(
            FE_ROOT / "planner", patterns=("*.py",)
        ),
        "logical_source_python_tree_hash": _tree_hash(
            FE_ROOT / "storage" / "sources", patterns=("*.py",)
        ),
        "audit_source_hash": compute_implementation_hash(
            audit.read_text(encoding="utf-8")
        ),
    }
    for name, path in files.items():
        hashes[name] = compute_implementation_hash(path.read_text(encoding="utf-8"))
    return hashes


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
        errors.append("execution-semantic source hashes are stale")

    try:
        from cleaned_operators.production_hardening import factor_production_targets
        from cleaned_operators.operator_surface import DAILY_CANONICALS

        expected = set(factor_production_targets()).difference(DAILY_CANONICALS)
    except Exception as error:
        return errors + [
            f"target resolution failed: {type(error).__name__}: {error}"
        ]
    observed = set(str(value) for value in (payload.get("operators") or []))
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
    return str(canonical) in set(
        str(value) for value in (payload.get("operators") or [])
    )


def current_hashes() -> dict[str, str]:
    return _hashes()
