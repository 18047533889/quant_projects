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
        "group_zscore",
        "winsorize",
        "group_winsorize",
        "ts_beta",
        "ts_mad",
        "ts_ema",
        "ts_rank",
        "ewm_mean",
        "ffill",
        "bfill",
        "fillna_const",
        "ts_decay_linear",
        "coalesce",
        "protected_div",
        "protected_log",
        "protected_sqrt",
        "nan_to_num",
        "fillna",
        "is_nan",
        "is_finite",
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
        "ts_regression",
        "ts_skew",
        "cum_sum",
        "cum_max",
        "cum_min",
        "WMA",
        "ewm_std",
        "ewm_var",
        "cum_std",
        "expanding_std",
        "floor",
        "ceil",
        "inverse",
        "count",
        "Slope",
        "ts_argmax",
        "ts_argmin",
        "ewm_corr",
        "ewm_cov",
        "ts_sharpe",
        "ts_autocorr",
        "rolling_beta",
        "rank_pct",
        "cs_pct_rank",
        "cs_quantile",
        "c_percentile",
        "log_returns",
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
        "c_mean",
        "c_std",
        "c_sum",
        "c_count",
        "cs_mad",
        "cs_mad_zscore",
        "RSI_WILDER",
        "ATR_WILDER",
    }
)

# DuckDB / ClickHouse 分层别名（ClickHouse 暂与 DuckDB parity 子集对齐）
DUCKDB_SQL_PARITY_VERIFIED: frozenset[str] = frozenset()  # set below
CLICKHOUSE_SQL_PARITY_VERIFIED: frozenset[str] = frozenset()
DUCKDB_SQL_PRODUCTION_SAFE: frozenset[str] = frozenset()
CLICKHOUSE_SQL_PRODUCTION_SAFE: frozenset[str] = frozenset()

# 与 pandas golden / 三后端 parity 对齐（可 research 默认下推验证）
SQL_PARITY_VERIFIED_CANONICALS: frozenset[str] = frozenset(
    {
        "column",
        "literal",
    }
    | P0_PRODUCTION_FASTPATH_CANONICALS
    | P1_DUCKDB_PRODUCTION_SAFE
    | P1_DUCKDB_PARITY_PENDING
    | frozenset({"cs_quantile", "c_percentile", "group_percentile", "ts_median"})
)

# production 默认可 SQL 下推（HybridLong fast path）
SQL_PRODUCTION_SAFE_CANONICALS: frozenset[str] = frozenset(
    {"column", "literal"}
    | P0_PRODUCTION_FASTPATH_CANONICALS
    | P1_DUCKDB_PRODUCTION_SAFE
)

DUCKDB_SQL_PARITY_VERIFIED = SQL_PARITY_VERIFIED_CANONICALS
DUCKDB_SQL_PRODUCTION_SAFE = SQL_PRODUCTION_SAFE_CANONICALS
CLICKHOUSE_SQL_PARITY_VERIFIED = frozenset(
    c for c in SQL_PARITY_VERIFIED_CANONICALS if c not in P1_DUCKDB_PARITY_PENDING
)
CLICKHOUSE_SQL_PRODUCTION_SAFE = CLICKHOUSE_SQL_PARITY_VERIFIED & SQL_PRODUCTION_SAFE_CANONICALS

# 有 emitter 实现但暂不默认 production SQL 下推
SQL_PRODUCTION_DEFERRED_CANONICALS: frozenset[str] = frozenset(
    P2_RESEARCH_ONLY
    | P1_DUCKDB_PARITY_PENDING
)

# 向后兼容
SQL_CAPABLE_CANONICALS = SQL_IMPLEMENTED_CANONICALS


def is_sql_implemented(canon: str) -> bool:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_IMPLEMENTED_CANONICALS


def is_sql_parity_verified(canon: str) -> bool:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_PARITY_VERIFIED_CANONICALS


def is_sql_production_safe(canon: str) -> bool:
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
