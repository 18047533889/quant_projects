# -*- coding: utf-8
"""PolarsLong native 三层准入：Implemented / Parity Verified / Production Safe。"""
from __future__ import annotations

from backend.polars_long_policy import POLARS_LONG_MAP_GROUPS, POLARS_LONG_NATIVE, infer_polars_long_tier
from backend.production_fast_path import (
    PRODUCTION_TRIPLE_PARITY_CANONICALS,
    PRODUCTION_TRIPLE_PARITY_DUCKDB,
)
from backend.production_fastpath_tiers import (
    P0_PRODUCTION_FASTPATH_CANONICALS,
    P1_POLARS_PRODUCTION_SAFE,
    P2_RESEARCH_ONLY,
    resolve_polars_native_canonical,
)
from backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

# 已实现 native expr（与 POLARS_LONG_NATIVE 同步）
POLARS_LONG_NATIVE_IMPLEMENTED: frozenset[str] = POLARS_LONG_NATIVE

# map_groups 分层
POLARS_LONG_MAP_GROUPS_IMPLEMENTED: frozenset[str] = POLARS_LONG_MAP_GROUPS
POLARS_LONG_MAP_GROUPS_RESEARCH: frozenset[str] = POLARS_LONG_MAP_GROUPS
POLARS_LONG_MAP_GROUPS_PRODUCTION_ALLOWED: frozenset[str] = frozenset()

# 与 pandas golden / 三后端 parity 对齐的 native 子集
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
        "ts_argmax",
        "ts_argmin",
        "ts_sharpe",
        "ts_autocorr",
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
    if c in POLARS_LONG_NATIVE
)

# production fast path 允许的 native（P0 + P1 Polars parity verified）
_POLARS_PRODUCTION_SAFE_EXPLICIT: frozenset[str] = frozenset(
    c
    for c in (P0_PRODUCTION_FASTPATH_CANONICALS | P1_POLARS_PRODUCTION_SAFE)
    if c in POLARS_LONG_NATIVE
)

POLARS_LONG_NATIVE_PRODUCTION_SAFE: frozenset[str] = frozenset(
    c for c in _POLARS_PRODUCTION_SAFE_EXPLICIT if c not in P2_RESEARCH_ONLY
)

# 向后兼容：map_groups 默认不允许 production fast path
APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION: frozenset[str] = POLARS_LONG_MAP_GROUPS_PRODUCTION_ALLOWED

POLARS_LONG_FASTPATH_DEFERRED: frozenset[str] = frozenset(P2_RESEARCH_ONLY)


def _resolve(canon: str) -> str:
    return resolve_polars_native_canonical(canon)


def is_polars_long_native_implemented(canon: str) -> bool:
    return _resolve(canon) in POLARS_LONG_NATIVE_IMPLEMENTED


def is_polars_long_native_parity_verified(canon: str) -> bool:
    return _resolve(canon) in POLARS_LONG_NATIVE_PARITY_VERIFIED


def is_polars_long_native_production_safe(canon: str) -> bool:
    name = _resolve(canon)
    if name in POLARS_LONG_FASTPATH_DEFERRED:
        return False
    return name in POLARS_LONG_NATIVE_PRODUCTION_SAFE


def polars_long_production_tier(canon: str) -> str:
    """native 三层：implemented / parity_verified / production_safe / unsupported。"""
    name = _resolve(canon)
    tier = infer_polars_long_tier(name)
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
    return _resolve(canon) in PRODUCTION_TRIPLE_PARITY_DUCKDB
