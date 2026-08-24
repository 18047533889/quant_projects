# -*- coding: utf-8
"""PolarsLong native 三层准入：Implemented / Parity Verified / Production Safe。"""
from __future__ import annotations

from factor_engine.backend.polars_long_policy import (
    POLARS_LONG_MAP_GROUPS,
    POLARS_LONG_NATIVE,
    infer_polars_long_tier,
)
from factor_engine.backend.primitive_evidence import (
    POLARS_REFERENCE_PARITY_VERIFIED,
    PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
)
from factor_engine.backend.production_fastpath_tiers import (
    P0_PRODUCTION_FASTPATH_CANONICALS,
    P1_POLARS_PRODUCTION_SAFE,
    FASTPATH_DEFERRED_CANONICALS,
    resolve_polars_native_canonical,
)

# 纯 Polars Expr native（不含 python_rolling）
POLARS_LONG_NATIVE_IMPLEMENTED: frozenset[str] = POLARS_LONG_NATIVE

POLARS_LONG_MAP_GROUPS_IMPLEMENTED: frozenset[str] = POLARS_LONG_MAP_GROUPS
POLARS_LONG_MAP_GROUPS_RESEARCH: frozenset[str] = POLARS_LONG_MAP_GROUPS
POLARS_LONG_MAP_GROUPS_PRODUCTION_ALLOWED: frozenset[str] = frozenset()

POLARS_LONG_NATIVE_PARITY_VERIFIED: frozenset[str] = POLARS_REFERENCE_PARITY_VERIFIED

from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS as _DAILY_CANONICALS

_STATIC_POLARS_CANDIDATES: frozenset[str] = frozenset(_DAILY_CANONICALS) | frozenset({"protected_div"})

POLARS_LONG_NATIVE_PRODUCTION_SAFE: frozenset[str] = frozenset(
    c
    for c in _STATIC_POLARS_CANDIDATES
    if c in POLARS_LONG_NATIVE
    and c in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    and c not in FASTPATH_DEFERRED_CANONICALS
)

APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION: frozenset[str] = POLARS_LONG_MAP_GROUPS_PRODUCTION_ALLOWED

POLARS_LONG_FASTPATH_DEFERRED: frozenset[str] = frozenset(FASTPATH_DEFERRED_CANONICALS)


def _resolve(canon: str) -> str:
    """将别名映射为 PolarsLong native canonical。"""
    return resolve_polars_native_canonical(canon)


def is_polars_long_native_implemented(canon: str) -> bool:
    """判断 canonical 是否在 PolarsLong native 已实现白名单内。"""
    return _resolve(canon) in POLARS_LONG_NATIVE_IMPLEMENTED


def is_polars_long_native_parity_verified(canon: str) -> bool:
    """判断 canonical 是否已通过 PolarsLong native parity 验证（evidence JSON）。"""
    return _resolve(canon) in POLARS_LONG_NATIVE_PARITY_VERIFIED


def is_polars_long_native_production_safe(canon: str) -> bool:
    """判断 canonical 是否可安全用于 production PolarsLong native（须 dual-backend evidence）。"""
    name = _resolve(canon)
    if name in POLARS_LONG_FASTPATH_DEFERRED:
        return False
    if infer_polars_long_tier(name) != "native":
        return False
    return name in POLARS_LONG_NATIVE_PRODUCTION_SAFE


def polars_long_production_tier(canon: str) -> str:
    """native 三层：implemented / parity_verified / production_safe / unsupported。"""
    name = _resolve(canon)
    if name not in POLARS_LONG_NATIVE:
        return infer_polars_long_tier(name)
    if name in POLARS_LONG_NATIVE_PRODUCTION_SAFE:
        return "production_safe"
    if name in POLARS_LONG_NATIVE_PARITY_VERIFIED:
        return "parity_verified"
    return "implemented"


def duckdb_triple_parity_verified(canon: str) -> bool:
    """DuckDB 真实 SQL parity 是否已通过（primitive evidence）。"""
    from factor_engine.backend.primitive_evidence import primitive_duckdb_real_sql_verified

    return primitive_duckdb_real_sql_verified(_resolve(canon))
