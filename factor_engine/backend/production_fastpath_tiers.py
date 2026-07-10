# -*- coding: utf-8
"""Production fast path 算子分层（P0 / P1 / P2 / forbidden）。

``POLARS_LONG_NATIVE`` / ``SQL_IMPLEMENTED`` 仅表示 *implemented*；
本模块定义 *parity verified* 与 *production safe* 准入批次。
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# P0：元素 / protected / 逻辑 / 基础 ts / 截面 / 价量 — 三后端 production safe
# ---------------------------------------------------------------------------
P0_ELEMENT_CANONICALS: frozenset[str] = frozenset(
    {
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
        "floor",
        "ceil",
        "inverse",
        "power",
        "maximum",
        "minimum",
    }
)

P0_PROTECTED_CANONICALS: frozenset[str] = frozenset(
    {"protected_div", "protected_log", "protected_sqrt"}
)

P0_LOGIC_CANONICALS: frozenset[str] = frozenset(
    {
        "gt",
        "lt",
        "eq",
        "ge",
        "le",
        "ne",
        "and_",
        "or_",
        "not_",
        "where",
        "if_else",
        "coalesce",
        "fillna",
        "fillna_const",
        "nan_to_num",
        "is_nan",
        "is_finite",
        "ffill",
    }
)

P0_TS_CANONICALS: frozenset[str] = frozenset(
    {
        "ts_delay",
        "ts_delta",
        "ts_pct",
        "ts_mean",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_std",
        "ts_var",
        "ts_median",
        "ts_zscore",
        "ts_ema",
        "ts_rank",
        "ts_decay_linear",
        "ts_sharpe",
        "ts_autocorr",
    }
)

P0_CROSS_SECTION_CANONICALS: frozenset[str] = frozenset(
    {
        "rank",
        "rank_pct",
        "cs_pct_rank",
        "zscore",
        "cs_demean",
        "scale",
        "normalize",
        "c_mean",
        "c_std",
        "c_sum",
        "c_count",
    }
)

P0_PRICE_VOLUME_CANONICALS: frozenset[str] = frozenset(
    {"log_returns", "volatility", "vwap"}
)

P0_PRODUCTION_FASTPATH_CANONICALS: frozenset[str] = (
    P0_ELEMENT_CANONICALS
    | P0_PROTECTED_CANONICALS
    | P0_LOGIC_CANONICALS
    | P0_TS_CANONICALS
    | P0_CROSS_SECTION_CANONICALS
    | P0_PRICE_VOLUME_CANONICALS
)

# ---------------------------------------------------------------------------
# P1：group / robust / binary ts — parity 后分 backend production safe
# ---------------------------------------------------------------------------
P1_GROUP_CANONICALS: frozenset[str] = frozenset(
    {
        "group_mean",
        "group_std",
        "group_zscore",
        "group_rank",
        "group_neutralize",
        "group_normalize",
        "group_percentile",
    }
)

P1_ROBUST_CANONICALS: frozenset[str] = frozenset(
    {"cs_mad", "cs_mad_zscore", "winsorize", "group_winsorize"}
)

P1_BINARY_TS_CANONICALS: frozenset[str] = frozenset(
    {"ts_corr", "ts_cov", "ts_beta", "rolling_beta"}
)

P1_CS_REGRESSION_CANONICALS: frozenset[str] = frozenset({"cs_resid", "cs_regression"})

P1_TECHNICAL_CANONICALS: frozenset[str] = frozenset({"RSI_WILDER", "ATR_WILDER"})

P1_DUCKDB_PARITY_PENDING: frozenset[str] = frozenset()

# ---------------------------------------------------------------------------
# P1 Extended：SQL/Polars 已实现、research fast path 全覆盖（非 production_core）
# ---------------------------------------------------------------------------
P1_EXTENDED_TS_CANONICALS: frozenset[str] = frozenset(
    {
        "WMA",
        "ts_mad",
        "ts_quantile",
        "ts_product",
        "ts_skew",
        "ts_regression",
        "Slope",
        "ts_argmax",
        "ts_argmin",
        "ts_ratio",
    }
)

P1_EXTENDED_EWM_CANONICALS: frozenset[str] = frozenset({"ewm_mean", "ewm_std", "ewm_var"})

P1_EXTENDED_CUM_CANONICALS: frozenset[str] = frozenset(
    {
        "cum_sum",
        "cum_max",
        "cum_min",
        "cum_prod",
        "cum_delta",
        "cum_std",
        "expanding_mean",
        "expanding_std",
        "expanding_sum",
        "count",
    }
)

P1_EXTENDED_MATH_CANONICALS: frozenset[str] = frozenset(
    {"log_abs", "signed_log", "signed_sqrt"}
)

P1_EXTENDED_CS_CANONICALS: frozenset[str] = frozenset(
    {"cs_quantile", "c_percentile", "group_decay_linear"}
)

P1_EXTENDED_CANONICALS: frozenset[str] = (
    P1_EXTENDED_TS_CANONICALS
    | P1_EXTENDED_EWM_CANONICALS
    | P1_EXTENDED_CUM_CANONICALS
    | P1_EXTENDED_MATH_CANONICALS
    | P1_EXTENDED_CS_CANONICALS
)

P1_POLARS_CORE_PRODUCTION_SAFE: frozenset[str] = (
    P1_GROUP_CANONICALS
    | P1_BINARY_TS_CANONICALS
    | P1_CS_REGRESSION_CANONICALS
    | P1_TECHNICAL_CANONICALS
    | frozenset({"cs_mad", "cs_mad_zscore", "winsorize", "group_winsorize"})
)

P1_DUCKDB_CORE_PRODUCTION_SAFE: frozenset[str] = (
    P1_GROUP_CANONICALS
    | P1_BINARY_TS_CANONICALS
    | P1_CS_REGRESSION_CANONICALS
    | P1_TECHNICAL_CANONICALS
    | frozenset({"winsorize", "group_winsorize", "cs_mad", "cs_mad_zscore"})
)

P1_EXTENDED_DUCKDB_CANONICALS: frozenset[str] = P1_EXTENDED_CANONICALS - frozenset({"ts_ratio"})

P1_POLARS_PRODUCTION_SAFE: frozenset[str] = P1_POLARS_CORE_PRODUCTION_SAFE | P1_EXTENDED_CANONICALS

P1_DUCKDB_PRODUCTION_SAFE: frozenset[str] = P1_DUCKDB_CORE_PRODUCTION_SAFE | P1_EXTENDED_DUCKDB_CANONICALS

# ---------------------------------------------------------------------------
# P2：仍走 map_groups / Python kernel — 不进 fast path
# ---------------------------------------------------------------------------
P2_MAP_GROUPS_CANONICALS: frozenset[str] = frozenset(
    {
        "ewm_corr",
        "ewm_cov",
        "ts_kurt",
        "ts_moment",
        "ts_max_buildup",
        "expanding_rank",
        "fillna_interpolate",
        "quantile",
    }
)

P2_FILL_INTERPOLATE_CANONICALS: frozenset[str] = frozenset({"bfill", "causal_bfill"})

P2_NATIVE_PARITY_PENDING: frozenset[str] = frozenset()

P2_RESEARCH_ONLY: frozenset[str] = P2_MAP_GROUPS_CANONICALS | P2_FILL_INTERPOLATE_CANONICALS

# ---------------------------------------------------------------------------
# 明确禁止 production fast path
# ---------------------------------------------------------------------------
FORBIDDEN_PRODUCTION_FASTPATH: frozenset[str] = frozenset(
    {
        # fundamental
        "quarter",
        "ttm",
        "yoy",
        "avg2",
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
        # CAPM / risk regression extensions
        "downside_beta",
        "tail_beta",
        "idio_vol",
        "idio_skew",
        "residual_momentum_capm",
        "coskewness_to_market",
        "rolling_beta_to_market",
        # matrix / transform
        "fft",
        "ifft",
        "wavelet",
        "svd",
        "eig",
        "pca",
    }
)

# PolarsLong emitter 将 rolling_beta 映射为 ts_beta native
POLARS_NATIVE_ALIASES: dict[str, str] = {
    "rolling_beta": "ts_beta",
    "cum_std": "expanding_std",
    "WMA": "ts_decay_linear",
}


def resolve_polars_native_canonical(canon: str) -> str:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return POLARS_NATIVE_ALIASES.get(name, name)


def is_p0_production_fastpath(canon: str) -> bool:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in P0_PRODUCTION_FASTPATH_CANONICALS


def is_forbidden_production_fastpath(canon: str) -> bool:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    if name.startswith("micro_"):
        return True
    return name in FORBIDDEN_PRODUCTION_FASTPATH
