# -*- coding: utf-8
"""Primitive production evidence（JSON artifact，禁止从 production 白名单自证 parity）。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_EVIDENCE_JSON = Path(__file__).resolve().parents[1] / "evidence" / "primitive_verified.json"


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    if not _EVIDENCE_JSON.is_file():
        raise FileNotFoundError(f"missing primitive evidence: {_EVIDENCE_JSON}")
    return json.loads(_EVIDENCE_JSON.read_text(encoding="utf-8"))


def _set(key: str) -> frozenset[str]:
    return frozenset(str(x) for x in (_load().get(key) or []))


POLARS_REFERENCE_PARITY_VERIFIED: frozenset[str] = _set("polars_reference_parity")
POLARS_EDGE_VERIFIED: frozenset[str] = _set("polars_edge_verified")
DUCKDB_REFERENCE_PARITY_VERIFIED: frozenset[str] = _set("duckdb_reference_parity")
DUCKDB_REAL_SQL_VERIFIED: frozenset[str] = _set("duckdb_real_sql_verified")
DUCKDB_EDGE_VERIFIED: frozenset[str] = _set("duckdb_edge_verified")
DUCKDB_NULL_EDGE_VERIFIED: frozenset[str] = _set("duckdb_null_edge_verified")
DUCKDB_NAN_EDGE_VERIFIED: frozenset[str] = _set("duckdb_nan_edge_verified")
DUCKDB_INF_EDGE_VERIFIED: frozenset[str] = _set("duckdb_inf_edge_verified")
# PolarsLong no-Pandas fallback（DuckDB 无 fallback 隐含于 real-SQL parity helper）
POLARS_NO_FALLBACK_VERIFIED: frozenset[str] = _set("no_fallback_verified")
NO_FALLBACK_VERIFIED: frozenset[str] = POLARS_NO_FALLBACK_VERIFIED

PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE: frozenset[str] = (
    POLARS_REFERENCE_PARITY_VERIFIED
    & POLARS_EDGE_VERIFIED
    & DUCKDB_REFERENCE_PARITY_VERIFIED
    & DUCKDB_REAL_SQL_VERIFIED
    & DUCKDB_EDGE_VERIFIED
    & NO_FALLBACK_VERIFIED
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
