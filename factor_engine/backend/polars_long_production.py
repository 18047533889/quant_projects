# -*- coding: utf-8
"""PolarsLong native 三层准入：Implemented / Parity Verified / Production Safe。"""
from __future__ import annotations

from backend.polars_long_policy import (
    POLARS_LONG_MAP_GROUPS,
    POLARS_LONG_NATIVE,
    infer_polars_long_tier,
)
from backend.production_fast_path import (
    PRODUCTION_TRIPLE_PARITY_CANONICALS,
    PRODUCTION_TRIPLE_PARITY_DUCKDB,
)
from backend.production_fastpath_tiers import (
    P0_PRODUCTION_FASTPATH_CANONICALS,
    P1_POLARS_PRODUCTION_SAFE,
    FASTPATH_DEFERRED_CANONICALS,
    resolve_polars_native_canonical,
)
from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

# 纯 Polars Expr native（不含 python_rolling）
POLARS_LONG_NATIVE_IMPLEMENTED: frozenset[str] = POLARS_LONG_NATIVE

POLARS_LONG_MAP_GROUPS_IMPLEMENTED: frozenset[str] = POLARS_LONG_MAP_GROUPS
POLARS_LONG_MAP_GROUPS_RESEARCH: frozenset[str] = POLARS_LONG_MAP_GROUPS
POLARS_LONG_MAP_GROUPS_PRODUCTION_ALLOWED: frozenset[str] = frozenset()

_POLARS_NATIVE_PARITY_EXTRA: frozenset[str] = frozenset(
    {
        "c_mean",
        "c_std",
        "c_sum",
        "c_count",
        "cs_pct_rank",
        "ts_median",
        "group_percentile",
        "cs_mad",
        "cs_mad_zscore",
        "rolling_beta",
        "winsorize",
        "group_winsorize",
    }
)

POLARS_LONG_NATIVE_PARITY_VERIFIED: frozenset[str] = frozenset(
    c
    for c in (
        P0_PRODUCTION_FASTPATH_CANONICALS
        | P1_POLARS_PRODUCTION_SAFE
        | PRODUCTION_TRIPLE_PARITY_CANONICALS
        | _POLARS_NATIVE_PARITY_EXTRA
        | (SQL_PRODUCTION_SAFE_CANONICALS & POLARS_LONG_NATIVE)
    )
    if c in POLARS_LONG_NATIVE and c not in FASTPATH_DEFERRED_CANONICALS
)

_POLARS_PRODUCTION_SAFE_EXPLICIT: frozenset[str] = frozenset(
    c
    for c in (P0_PRODUCTION_FASTPATH_CANONICALS | P1_POLARS_PRODUCTION_SAFE)
    if c in POLARS_LONG_NATIVE
)

POLARS_LONG_NATIVE_PRODUCTION_SAFE: frozenset[str] = frozenset(
    c for c in _POLARS_PRODUCTION_SAFE_EXPLICIT if c not in FASTPATH_DEFERRED_CANONICALS
)

APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION: frozenset[str] = POLARS_LONG_MAP_GROUPS_PRODUCTION_ALLOWED

POLARS_LONG_FASTPATH_DEFERRED: frozenset[str] = frozenset(FASTPATH_DEFERRED_CANONICALS)


def _resolve(canon: str) -> str:
    """将别名映射为 PolarsLong native canonical。"""
    return resolve_polars_native_canonical(canon)


def is_polars_long_native_implemented(canon: str) -> bool:
    """判断 canonical 是否在 PolarsLong native 已实现白名单内。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否属于 ``POLARS_LONG_NATIVE_IMPLEMENTED``。
    """
    return _resolve(canon) in POLARS_LONG_NATIVE_IMPLEMENTED


def is_polars_long_native_parity_verified(canon: str) -> bool:
    """判断 canonical 是否已通过 PolarsLong native parity 验证。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否属于 ``POLARS_LONG_NATIVE_PARITY_VERIFIED``。
    """
    return _resolve(canon) in POLARS_LONG_NATIVE_PARITY_VERIFIED


def is_polars_long_native_production_safe(canon: str) -> bool:
    """判断 canonical 是否可安全用于 production PolarsLong native 路径。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        tier 为 native 且在 production-safe 白名单内且非 deferred。
    """
    name = _resolve(canon)
    if name in POLARS_LONG_FASTPATH_DEFERRED:
        return False
    if infer_polars_long_tier(name) != "native":
        return False
    return name in POLARS_LONG_NATIVE_PRODUCTION_SAFE


def polars_long_production_tier(canon: str) -> str:
    """native 三层：implemented / parity_verified / production_safe / unsupported。"""
    name = _resolve(canon)
    tier = infer_polars_long_tier(name)
    if tier == "python_rolling":
        return "python_rolling"
    if tier != "native":
        return tier if tier != "unsupported" else "unsupported"
    if name in POLARS_LONG_NATIVE_PRODUCTION_SAFE:
        return "production_safe"
    if name in POLARS_LONG_NATIVE_PARITY_VERIFIED:
        return "parity_verified"
    if name in POLARS_LONG_NATIVE_IMPLEMENTED:
        return "implemented"
    return "unsupported"


def duckdb_triple_parity_verified(canon: str) -> bool:
    """判断 canonical 是否在三后端 DuckDB parity 子集内。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否属于 ``PRODUCTION_TRIPLE_PARITY_DUCKDB``。
    """
    return _resolve(canon) in PRODUCTION_TRIPLE_PARITY_DUCKDB
