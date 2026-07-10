# -*- coding: utf-8
"""Composite 算子 production 证据链（structural ≠ production-safe）。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_EVIDENCE_JSON = Path(__file__).resolve().parents[1] / "evidence" / "composite_verified.json"


@lru_cache(maxsize=1)
def _load_evidence_manifest() -> dict[str, Any]:
    if not _EVIDENCE_JSON.is_file():
        raise FileNotFoundError(f"missing composite evidence manifest: {_EVIDENCE_JSON}")
    return json.loads(_EVIDENCE_JSON.read_text(encoding="utf-8"))


def _verified_set(key: str) -> frozenset[str]:
    data = _load_evidence_manifest()
    raw = data.get(key) or []
    return frozenset(str(x) for x in raw)


# 原始 Pandas operator vs lowered-plan Pandas（非四后端 parity）
COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED: frozenset[str] = _verified_set(
    "reference_to_lowered_pandas"
)

# Lowered DAG 经 PolarsLong native 真实执行（无 registry/map_groups/python_rolling）
COMPOSITE_LOWERED_POLARS_VERIFIED: frozenset[str] = _verified_set("lowered_polars_native")

# Lowered DAG 经 DuckDB 真实 SQL 执行（sql_query_count>0, fallback=0）
COMPOSITE_LOWERED_DUCKDB_VERIFIED: frozenset[str] = _verified_set("lowered_duckdb_real_sql")

# 边界/除零/首行等 edge case 套件
COMPOSITE_EDGE_VERIFIED: frozenset[str] = _verified_set("edge_verified")

# 三者齐备才可称 full parity（当前多数 composite 尚未达到）
COMPOSITE_FULL_PARITY_VERIFIED: frozenset[str] = (
    COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED
    & COMPOSITE_LOWERED_POLARS_VERIFIED
    & COMPOSITE_LOWERED_DUCKDB_VERIFIED
)

# 向后兼容别名（deprecated）
COMPOSITE_REFERENCE_PARITY_VERIFIED = COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED
COMPOSITE_POLARS_NATIVE_VERIFIED = COMPOSITE_LOWERED_POLARS_VERIFIED
COMPOSITE_DUCKDB_REAL_SQL_VERIFIED = COMPOSITE_LOWERED_DUCKDB_VERIFIED


def composite_structurally_capable(canon: str) -> bool:
    """展开后 primitive 名称均在 dual-backend static safe 集合内。"""
    from planner.composite_lowering import composite_dual_backend_capable

    return composite_dual_backend_capable(canon)


def composite_reference_parity_verified(canon: str) -> bool:
    return canon in COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED


def composite_polars_native_verified(canon: str) -> bool:
    return canon in COMPOSITE_LOWERED_POLARS_VERIFIED


def composite_duckdb_real_sql_verified(canon: str) -> bool:
    return canon in COMPOSITE_LOWERED_DUCKDB_VERIFIED


def composite_edge_verified(canon: str) -> bool:
    return canon in COMPOSITE_EDGE_VERIFIED


def composite_full_parity_verified(canon: str) -> bool:
    return canon in COMPOSITE_FULL_PARITY_VERIFIED


def composite_production_safe(canon: str) -> bool:
    """Composite 可认证为 production-safe 的完整证据链。"""
    from planner.composite_lowering import has_composite_lowering
    from cleaned_operators.operator_spec import build_operator_spec, infer_production_policy

    if not has_composite_lowering(canon):
        return False
    spec = build_operator_spec(canon)
    if spec is None or not spec.allow_in_production:
        return False
    if infer_production_policy(canon) != "allowed":
        return False
    return (
        composite_structurally_capable(canon)
        and composite_reference_parity_verified(canon)
        and composite_polars_native_verified(canon)
        and composite_duckdb_real_sql_verified(canon)
        and composite_edge_verified(canon)
    )
