# -*- coding: utf-8
"""Primitive production evidence（JSON artifact，禁止从 production 白名单自证 parity）。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

FE_ROOT = Path(__file__).resolve().parents[1]
_CASE_REGISTRY_JSON = FE_ROOT / "evidence" / "primitive_case_registry.json"
_VERIFIED_JSON = FE_ROOT / "evidence" / "primitive_verified.json"


@lru_cache(maxsize=1)
def _load_case_registry() -> dict[str, Any]:
    if not _CASE_REGISTRY_JSON.is_file():
        # Development metadata only.  A case registry proves that a test case is
        # declared; it never proves that the test passed.
        if _VERIFIED_JSON.is_file():
            data = json.loads(_VERIFIED_JSON.read_text(encoding="utf-8"))
            if data.get("artifact_kind") == "case_registry":
                return data
        return {"artifact_kind": "case_registry"}
    return json.loads(_CASE_REGISTRY_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_verified() -> dict[str, Any]:
    if not _VERIFIED_JSON.is_file():
        raise FileNotFoundError(f"missing primitive evidence: {_VERIFIED_JSON}")
    return json.loads(_VERIFIED_JSON.read_text(encoding="utf-8"))


def _set_from(data: dict[str, Any], key: str) -> frozenset[str]:
    return frozenset(str(x) for x in (data.get(key) or []))


def _case_registry_set(key: str) -> frozenset[str]:
    return _set_from(_load_case_registry(), key)


def _verified_set(key: str) -> frozenset[str]:
    return _set_from(_load_verified(), key)


def evidence_artifact_valid() -> bool:
    from backend.evidence_provenance import evidence_artifact_valid as _valid

    return _valid()


# Case registry（测试 case 名交集，不表示 pytest 已通过）
CASE_REGISTRY_POLARS_REFERENCE: frozenset[str] = _case_registry_set("polars_reference_parity")
CASE_REGISTRY_POLARS_EDGE: frozenset[str] = _case_registry_set("polars_edge_verified")
CASE_REGISTRY_DUCKDB_REFERENCE: frozenset[str] = _case_registry_set("duckdb_reference_parity")
CASE_REGISTRY_DUCKDB_EDGE: frozenset[str] = _case_registry_set("duckdb_edge_verified")
CASE_REGISTRY_DUCKDB_NAN_EDGE: frozenset[str] = _case_registry_set("duckdb_nan_edge_verified")
CASE_REGISTRY_NO_FALLBACK: frozenset[str] = _case_registry_set("no_fallback_verified")

CASE_REGISTRY_SIX_WAY: frozenset[str] = (
    CASE_REGISTRY_POLARS_REFERENCE
    & CASE_REGISTRY_POLARS_EDGE
    & CASE_REGISTRY_DUCKDB_REFERENCE
    & (CASE_REGISTRY_DUCKDB_EDGE | CASE_REGISTRY_DUCKDB_NAN_EDGE)
    & CASE_REGISTRY_NO_FALLBACK
)


def _load_verified_fail_closed(key: str) -> frozenset[str]:
    """Return passed evidence only; invalid/missing artifacts certify nothing."""
    if not evidence_artifact_valid():
        return frozenset()
    try:
        return _verified_set(key)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return frozenset()


# Verified lists（须 pytest 通过后由 certify_primitive_evidence.py 写入）
# Never fall back to the case registry here.  Production routing consumes these
# names, so a stale or malformed artifact must fail closed.
POLARS_REFERENCE_PARITY_VERIFIED: frozenset[str] = _load_verified_fail_closed("polars_reference_parity")
POLARS_EDGE_VERIFIED: frozenset[str] = _load_verified_fail_closed("polars_edge_verified")
DUCKDB_REFERENCE_PARITY_VERIFIED: frozenset[str] = _load_verified_fail_closed("duckdb_reference_parity")
DUCKDB_REAL_SQL_VERIFIED: frozenset[str] = _load_verified_fail_closed("duckdb_real_sql_verified")
DUCKDB_EDGE_VERIFIED: frozenset[str] = _load_verified_fail_closed("duckdb_edge_verified")
DUCKDB_NULL_EDGE_VERIFIED: frozenset[str] = _load_verified_fail_closed("duckdb_null_edge_verified")
DUCKDB_NAN_EDGE_VERIFIED: frozenset[str] = _load_verified_fail_closed("duckdb_nan_edge_verified")
DUCKDB_INF_EDGE_VERIFIED: frozenset[str] = _load_verified_fail_closed("duckdb_inf_edge_verified")
POLARS_NO_FALLBACK_VERIFIED: frozenset[str] = _load_verified_fail_closed("no_fallback_verified")
NO_FALLBACK_VERIFIED: frozenset[str] = POLARS_NO_FALLBACK_VERIFIED

# 六证交集：backend execution certified（只接受 test-passed artifact）
PRIMITIVE_BACKEND_EXECUTION_CERTIFIED: frozenset[str] = (
    POLARS_REFERENCE_PARITY_VERIFIED
    & POLARS_EDGE_VERIFIED
    & DUCKDB_REFERENCE_PARITY_VERIFIED
    & DUCKDB_REAL_SQL_VERIFIED
    & (DUCKDB_EDGE_VERIFIED | DUCKDB_NAN_EDGE_VERIFIED)
    & NO_FALLBACK_VERIFIED
)

# 向后兼容别名
PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE: frozenset[str] = PRIMITIVE_BACKEND_EXECUTION_CERTIFIED


def primitive_operational_production_certified(canon: str) -> bool:
    """Operational production：六证 + 参数域 + production signature + artifact 有效。"""
    if canon not in PRIMITIVE_BACKEND_EXECUTION_CERTIFIED:
        return False
    if not evidence_artifact_valid():
        return False
    from cleaned_operators.operator_spec import build_operator_spec

    spec = build_operator_spec(canon)
    if spec is None or not spec.allow_in_production:
        return False
    from backend.production_signature import operational_production_allowed

    return operational_production_allowed(canon)


def operational_production_certified_set() -> frozenset[str]:
    return frozenset(
        c for c in PRIMITIVE_BACKEND_EXECUTION_CERTIFIED if primitive_operational_production_certified(c)
    )


def primitive_polars_reference_verified(canon: str) -> bool:
    return canon in POLARS_REFERENCE_PARITY_VERIFIED


def primitive_duckdb_real_sql_verified(canon: str) -> bool:
    return canon in DUCKDB_REAL_SQL_VERIFIED


def primitive_dual_backend_production_safe(canon: str) -> bool:
    return canon in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE


def primitive_dual_backend_parity_verified(canon: str) -> bool:
    return (
        canon in POLARS_REFERENCE_PARITY_VERIFIED
        and canon in DUCKDB_REAL_SQL_VERIFIED
    )
