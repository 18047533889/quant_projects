# -*- coding: utf-8
"""Production fast path：三后端 parity 已验证 + PolarsLong native + SQL 下推。"""
from __future__ import annotations

from typing import Any

# 与 tests/backend_parity/test_production_core_triple_parity.py 保持同步（历史参考）
PRODUCTION_TRIPLE_PARITY_CANONICALS: frozenset[str] = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "protected_div",
        "protected_log",
        "ts_mean",
        "ts_std",
        "ts_zscore",
        "ts_median",
        "rank",
        "rank_pct",
        "zscore",
        "cs_pct_rank",
        "cs_mean",
        "log_returns",
        "volatility",
        "vwap",
        "power",
        "where",
        "gt",
        "group_mean",
        "group_zscore",
        "group_rank",
        "group_percentile",
        "group_winsorize",
        "ts_corr",
        "ts_cov",
        "ts_beta",
        "rolling_beta",
        "cs_mad",
        "cs_mad_zscore",
    }
)

PRODUCTION_TRIPLE_PARITY_DUCKDB: frozenset[str] = frozenset(
    {
        "ts_mean",
        "ts_std",
        "rank",
        "zscore",
        "protected_div",
        "protected_log",
        "group_mean",
        "group_zscore",
        "group_winsorize",
        "ts_corr",
        "ts_cov",
        "ts_beta",
        "cs_mad",
        "cs_mad_zscore",
    }
)


def is_any_backend_fastpath(canon: str) -> bool:
    """任一 backend production-safe 且未被 block。"""
    from factor_engine.backend.fastpath_coverage import build_fastpath_coverage_row
    from factor_engine.backend.operator_capability import resolve_canonical

    row = build_fastpath_coverage_row(resolve_canonical(canon))
    return row.production_fast_path or row.composite_dual_backend_fastpath


def is_dual_backend_fastpath(canon: str) -> bool:
    """直接 dual-backend fastpath（不含 composite lowering）。"""
    from factor_engine.backend.fastpath_coverage import build_fastpath_coverage_row
    from factor_engine.backend.operator_capability import resolve_canonical

    row = build_fastpath_coverage_row(resolve_canonical(canon))
    return row.dual_backend_fastpath


def is_effective_dual_backend_fastpath(canon: str) -> bool:
    """统一 production 口径：direct 或 composite dual-backend fastpath。"""
    from factor_engine.backend.fastpath_coverage import build_fastpath_coverage_row
    from factor_engine.backend.operator_capability import resolve_canonical

    row = build_fastpath_coverage_row(resolve_canonical(canon))
    return row.effective_dual_backend_fastpath


def is_production_fast_path(canon: str) -> bool:
    """Production 投递应使用 effective dual-backend fastpath。"""
    return is_effective_dual_backend_fastpath(canon)


def summarize_production_fast_path() -> dict[str, Any]:
    """生成 production fast path 覆盖报表摘要。"""
    from factor_engine.cleaned_operators.operator_spec import PRODUCTION_DUAL_BACKEND_CORE_CANONICALS

    from factor_engine.backend.operator_capability import resolve_canonical
    from factor_engine.backend.polars_long_policy import POLARS_LONG_NATIVE, infer_polars_long_tier
    from factor_engine.backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    from factor_engine.backend.sql_pushdown.sql_registry import SQL_CAPABLE_CANONICALS

    triple = sorted(PRODUCTION_TRIPLE_PARITY_CANONICALS)
    duckdb_triple = sorted(PRODUCTION_TRIPLE_PARITY_DUCKDB)
    native_triple = sorted(c for c in triple if infer_polars_long_tier(c) == "native")
    sql_triple = sorted(c for c in duckdb_triple if c in SQL_CAPABLE_CANONICALS)
    fast_path = sorted(c for c in triple if is_effective_dual_backend_fastpath(c))
    core_gap = sorted(
        PRODUCTION_DUAL_BACKEND_CORE_CANONICALS - set(fast_path) - {"column", "literal"}
    )
    native_not_triple = sorted(
        POLARS_LONG_NATIVE - PRODUCTION_TRIPLE_PARITY_CANONICALS - {"column", "literal", "materialized_series", "plan_ref"}
    )
    return {
        "triple_parity_count": len(triple),
        "triple_parity_native_count": len(native_triple),
        "triple_parity_duckdb_count": len(duckdb_triple),
        "primitive_dual_backend_evidence_count": len(PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE),
        "production_fast_path_count": len(fast_path),
        "production_fast_path": fast_path,
        "triple_parity_duckdb": duckdb_triple,
        "production_core_fast_path_gap": core_gap[:40],
        "production_core_fast_path_gap_count": len(core_gap),
        "polars_long_native_not_triple_parity_count": len(native_not_triple),
        "polars_long_native_not_triple_parity_sample": native_not_triple[:25],
    }
