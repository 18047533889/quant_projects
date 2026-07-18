# -*- coding: utf-8
"""SQL 下推三层准入：Implemented / Parity Verified / Production Safe。

``SQL_CAPABLE_CANONICALS`` 仅表示 emitter 有实现，不等于 production-safe。
"""
from __future__ import annotations

from backend.production_fastpath_tiers import (
    P0_PRODUCTION_FASTPATH_CANONICALS,
    P1_DUCKDB_PARITY_PENDING,
    P1_DUCKDB_PRODUCTION_SAFE,
    P2_RESEARCH_ONLY,
    FASTPATH_DEFERRED_CANONICALS,
)

# 向后兼容：``SQL_CAPABLE_CANONICALS`` == ``SQL_IMPLEMENTED_CANONICALS``

# emitter 有实现、可尝试编译（原 SQL_CAPABLE 大白名单）
SQL_IMPLEMENTED_CANONICALS: frozenset[str] = frozenset(
    {
        "column",
        "literal",
        "add",
        "subtract",
        "multiply",
        "divide",
        "neg",
        "abs",
        "sign",
        "log",
        "exp",
        "sqrt",
        "clip",
        "ts_mean",
        "ts_delay",
        "ts_delta",
        "ts_std",
        "ts_sum",
        "ts_max",
        "ts_min",
        "ts_pct",
        "ts_zscore",
        "ts_corr",
        "rank",
        "zscore",
        "scale",
        "cs_demean",
        "cs_resid",
        "cs_regression",
        "group_neutralize",
        "where",
        "if_else",
        "ts_median",
        "ts_var",
        "group_rank",
        "group_mean",
        "group_sum",
        "group_min",
        "group_max",
        "group_count",
        "group_zscore",
        "winsorize",
        "group_winsorize",
        "ts_beta",
        "ts_mad",
        "ts_ema",
        "ewm_mean",
        "standardize",
        "ts_rank",
        "ts_ema",
        "ffill",
        "fillna_const",
        "ts_decay_linear",
        "coalesce",
        "protected_div",
        "safe_div_null",
        "protected_log",
        "protected_sqrt",
        "nan_to_num",
        "div_or_default",
        "log_fill_invalid",
        "fillna",
        "is_nan",
        "is_null",
        "is_not_null",
        "is_finite",
        "is_infinite",
        "normalize",
        "group_normalize",
        "group_percentile",
        "group_decay_linear",
        "power",
        "gt",
        "lt",
        "eq",
        "ge",
        "le",
        "ne",
        "and_",
        "or_",
        "not_",
        "group_std",
        "ts_cov",
        "ts_quantile",
        "ts_product",
        "ts_regression_slope",
        "ts_skew",
        "cum_sum",
        "cum_max",
        "cum_min",
        "WMA",
        "ts_ewm_std",
        "ts_ewm_var",
        "cum_std",
        "expanding_std",
        "floor",
        "ceil",
        "inverse",
        "count",
        "ts_time_slope",
        "ts_argmax",
        "ts_argmin",
        "ts_ewm_corr",
        "ts_ewm_cov",
        "ts_sharpe",
        "ts_autocorr",
        "rolling_beta",
        "rank_pct",
        "cs_pct_rank",
        "cs_quantile",
        "c_percentile",
        "ts_log_return",
        "volatility",
        "vwap",
        "maximum",
        "minimum",
        "cum_prod",
        "cum_delta",
        "expanding_mean",
        "expanding_sum",
        "log_abs",
        "signed_log",
        "signed_sqrt",
        "tanh",
        "cbrt",
        "truncate",
        "cs_mean",
        "cs_std",
        "cs_sum",
        "cs_count",
        "cs_mad",
        "cs_mad_zscore",
        "RSI_WILDER",
        "ATR_WILDER",
        "ts_count_if",
        "ts_sum_if",
        "ts_mean_if",
        "ts_std_if",
        "ts_last_if",
        "ts_days_since",
        "ts_true_streak",
        "cs_bucket",
        "cs_multi_resid",
        "cs_wls_resid",
        "period_lag",
        "period_change",
        "period_average",
        "period_cagr",
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
        "ts_regression_tstat",
        "ts_trend_tstat",
        "ts_max_drawdown",
        "ts_partial_corr",
        "ts_nth_value",
    }
)

# DuckDB 分层
DUCKDB_SQL_PARITY_VERIFIED: frozenset[str] = frozenset()
DUCKDB_SQL_PRODUCTION_SAFE: frozenset[str] = frozenset()

# ClickHouse 独立分层（无真实 integration test 前默认空）
CLICKHOUSE_SQL_PARITY_VERIFIED: frozenset[str] = frozenset()
CLICKHOUSE_SQL_PRODUCTION_SAFE: frozenset[str] = frozenset()

from backend.primitive_evidence import (
    DUCKDB_REAL_SQL_VERIFIED,
    POLARS_REFERENCE_PARITY_VERIFIED,
    PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
)

SQL_PARITY_VERIFIED_CANONICALS: frozenset[str] = frozenset(
    {"column", "literal"}
) | DUCKDB_REAL_SQL_VERIFIED

from cleaned_operators.operator_surface import DAILY_CANONICALS as _DAILY_CANONICALS

_STATIC_SQL_CANDIDATES: frozenset[str] = (
    frozenset({"column", "literal", "protected_div"}) | frozenset(_DAILY_CANONICALS)
)

SQL_PRODUCTION_SAFE_CANONICALS: frozenset[str] = frozenset(
    c for c in _STATIC_SQL_CANDIDATES if c in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
) | frozenset({"column", "literal"})

DUCKDB_SQL_PARITY_VERIFIED = SQL_PARITY_VERIFIED_CANONICALS
DUCKDB_SQL_PRODUCTION_SAFE = SQL_PRODUCTION_SAFE_CANONICALS

# ClickHouse 暂不与 DuckDB 共享认证
CLICKHOUSE_SQL_PARITY_VERIFIED = frozenset()
CLICKHOUSE_SQL_PRODUCTION_SAFE = frozenset()

# 有 emitter 实现但暂不默认 production SQL 下推
SQL_PRODUCTION_DEFERRED_CANONICALS: frozenset[str] = frozenset(
    FASTPATH_DEFERRED_CANONICALS
)

# 向后兼容
SQL_CAPABLE_CANONICALS = SQL_IMPLEMENTED_CANONICALS


def is_sql_implemented(canon: str) -> bool:
    """判断 canonical 是否有 SQL emitter 实现。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否在 ``SQL_IMPLEMENTED_CANONICALS`` 内。
    """
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_IMPLEMENTED_CANONICALS


def is_sql_parity_verified(canon: str) -> bool:
    """判断 canonical 是否已通过 SQL parity 验证。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否在 ``SQL_PARITY_VERIFIED_CANONICALS`` 内。
    """
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_PARITY_VERIFIED_CANONICALS


def is_sql_production_safe(canon: str) -> bool:
    """判断 canonical 是否在静态 SQL production-safe 白名单内。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否在 ``SQL_PRODUCTION_SAFE_CANONICALS`` 内（不含运行时降级）。
    """
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_PRODUCTION_SAFE_CANONICALS


_DUCKDB_DOWNGRADE_CACHE: frozenset[str] | None = None


def duckdb_downgraded_canonicals(*, refresh: bool = False) -> frozenset[str]:
    """按当前 DuckDB 部署能力应从 production SQL 降级的 canonical。"""
    global _DUCKDB_DOWNGRADE_CACHE
    if _DUCKDB_DOWNGRADE_CACHE is None or refresh:
        try:
            from backend.sql_pushdown.duckdb_capabilities import (
                downgrade_sql_canonicals,
                get_duckdb_capability_report,
            )

            report = get_duckdb_capability_report(refresh=refresh)
            _DUCKDB_DOWNGRADE_CACHE = downgrade_sql_canonicals(report)
        except Exception:
            _DUCKDB_DOWNGRADE_CACHE = frozenset()
    return _DUCKDB_DOWNGRADE_CACHE


def effective_sql_production_safe(canon: str, *, refresh_duckdb: bool = False) -> bool:
    """静态 SQL_PRODUCTION_SAFE 减去 DuckDB 运行时能力降级。"""
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    if name not in SQL_PRODUCTION_SAFE_CANONICALS:
        return False
    return name not in duckdb_downgraded_canonicals(refresh=refresh_duckdb)
