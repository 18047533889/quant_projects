# -*- coding: utf-8
"""Composite 算子 production 证据链（structural ≠ production-safe）。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# Prefer the wheel-packaged manifest (factor_engine/backend/_evidence/), fall
# back to the repo-root evidence/ tree when running from the source checkout.
_PACKAGED_EVIDENCE_JSON = Path(__file__).resolve().parent / "_evidence" / "composite_verified.json"
_REPO_EVIDENCE_JSON = Path(__file__).resolve().parents[2] / "evidence" / "composite_verified.json"


@lru_cache(maxsize=1)
def _load_evidence_manifest() -> dict[str, Any]:
    path = _PACKAGED_EVIDENCE_JSON if _PACKAGED_EVIDENCE_JSON.is_file() else _REPO_EVIDENCE_JSON
    if not path.is_file():
        raise FileNotFoundError(f"missing composite evidence manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
    """展开后 primitive 均已 dual-backend production 认证（可组合 nesting）。"""
    from factor_engine.planner.composite_lowering import composite_dual_backend_capable

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


def composite_evidence_complete(canon: str) -> bool:
    """Composite 五证齐全（不含 production policy / allow 判定）。"""
    from factor_engine.planner.composite_lowering import has_composite_lowering

    if not has_composite_lowering(canon):
        return False
    return (
        canon in COMPOSITE_REFERENCE_TO_LOWERED_PANDAS_VERIFIED
        and canon in COMPOSITE_LOWERED_POLARS_VERIFIED
        and canon in COMPOSITE_LOWERED_DUCKDB_VERIFIED
        and canon in COMPOSITE_EDGE_VERIFIED
    )


def composite_production_safe(canon: str) -> bool:
    """Composite 可认证为 production-safe 的完整证据链（不依赖 allow_in_production 循环）。"""
    from factor_engine.cleaned_operators.operator_spec import _infer_status, is_production_denied
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
    from factor_engine.planner.composite_lowering import has_composite_lowering

    if not has_composite_lowering(canon):
        return False
    if is_production_denied(canon):
        return False
    if not composite_structurally_capable(canon):
        return False
    backends_map = OperatorRegistry._operators.get(canon)
    if not backends_map:
        return False
    chosen = "pandas_numpy" if "pandas_numpy" in backends_map else next(iter(backends_map))
    op = backends_map.get(chosen)
    if op is None:
        return False
    catalog = OperatorRegistry._catalog.get(canon, {})
    status = _infer_status(catalog)
    if status in ("experimental", "deprecated", "stub", "doc_only"):
        return False
    policy = infer_operator_policy(op, canonical=canon)
    if not policy.pit_safe or not policy.shape_preserving:
        return False
    return composite_evidence_complete(canon)
