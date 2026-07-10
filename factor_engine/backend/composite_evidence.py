# -*- coding: utf-8
"""Composite 算子 production 证据链（structural ≠ production-safe）。"""
from __future__ import annotations

# 原始 Pandas operator vs lowered DAG 四路一致（reference parity 测试维护）
COMPOSITE_REFERENCE_PARITY_VERIFIED: frozenset[str] = frozenset(
    {
        "MOM",
        "ROC",
        "BollingerUpper",
        "BollingerLower",
        "BollingerBands",
        "DPO",
        "WilliamsR",
        "StochasticK",
        "StochasticD",
        "OBV",
        "operating_margin",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "real_turnover_rate",
        "micro_spread",
    }
)

# PolarsLong native、无 registry/map_groups/python_rolling fallback
COMPOSITE_POLARS_NATIVE_VERIFIED: frozenset[str] = frozenset(
    {
        "MOM",
        "ROC",
        "WilliamsR",
        "StochasticK",
        "OBV",
    }
)

# DuckDB 真实 SQL 执行（sql_query_count>0, fallback=0）
COMPOSITE_DUCKDB_REAL_SQL_VERIFIED: frozenset[str] = frozenset(
    {
        "MOM",
        "ROC",
        "WilliamsR",
        "StochasticK",
        "OBV",
    }
)

# 边界/除零/首行等 edge case 套件
COMPOSITE_EDGE_VERIFIED: frozenset[str] = frozenset(
    {
        "OBV",
        "operating_margin",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "real_turnover_rate",
        "micro_spread",
        "WilliamsR",
        "StochasticK",
        "StochasticD",
    }
)


def composite_structurally_capable(canon: str) -> bool:
    """展开后 primitive 名称均在 dual-backend static safe 集合内。"""
    from planner.composite_lowering import composite_dual_backend_capable

    return composite_dual_backend_capable(canon)


def composite_reference_parity_verified(canon: str) -> bool:
    return canon in COMPOSITE_REFERENCE_PARITY_VERIFIED


def composite_polars_native_verified(canon: str) -> bool:
    return canon in COMPOSITE_POLARS_NATIVE_VERIFIED


def composite_duckdb_real_sql_verified(canon: str) -> bool:
    return canon in COMPOSITE_DUCKDB_REAL_SQL_VERIFIED


def composite_edge_verified(canon: str) -> bool:
    return canon in COMPOSITE_EDGE_VERIFIED


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
