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
    {"protected_div", "protected_log", "protected_sqrt", "safe_div_null"}
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
        "fillna_const",
        "nan_to_num",
        "is_nan",
        "is_finite",
        "ffill",
    }
)

# 简单 rolling — 可直接 production safe
P0_TS_SIMPLE_CANONICALS: frozenset[str] = frozenset(
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
    }
)

# 复杂 TS — 须逐算子 parity + 边界测试后才可升级 production safe
P1_TS_COMPLEX_PARITY_PENDING: frozenset[str] = frozenset(
    {
        "ts_ema",
        "ts_decay_linear",
    }
)

# golden 边界测试已通过 — 升级 production safe
P1_GOLDEN_VERIFIED_TS: frozenset[str] = frozenset(
    {
        "ts_rank",
        "ts_sharpe",
        "ts_autocorr",
    }
)

P0_TS_CANONICALS: frozenset[str] = P0_TS_SIMPLE_CANONICALS

P0_CROSS_SECTION_CANONICALS: frozenset[str] = frozenset(
    {
        "rank",
        "rank_pct",
        "cs_pct_rank",
        "zscore",
        "cs_demean",
        "scale",
        "normalize",
        "cs_mean",
        "cs_std",
        "cs_sum",
        "cs_count",
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
    }
)

P1_ROBUST_CANONICALS: frozenset[str] = frozenset(
    {"cs_mad", "cs_mad_zscore", "winsorize", "group_winsorize"}
)

P1_BINARY_TS_CANONICALS: frozenset[str] = frozenset(
    {"ts_corr", "ts_cov", "ts_beta", "rolling_beta"}
)

# 截面/滚动 OLS — 须 pairwise-null parity 后才可 production safe
P1_REGRESSION_PARITY_PENDING: frozenset[str] = frozenset({"ts_regression_slope", "ts_time_slope", "cs_resid", "cs_regression"})

P1_GOLDEN_VERIFIED_REGRESSION: frozenset[str] = frozenset()

# Wilder 递归指标 — research only，不进 production fastpath
P2_TECHNICAL_RESEARCH_ONLY: frozenset[str] = frozenset({"RSI_WILDER", "ATR_WILDER"})

# EWM SQL 为有限窗口近似 — 暂不进 DuckDB production safe
P1_EWM_PARITY_PENDING: frozenset[str] = frozenset({"ts_ema", "ewm_std", "ewm_var"})

P1_DUCKDB_PARITY_PENDING: frozenset[str] = (
    P1_TS_COMPLEX_PARITY_PENDING
    | P1_REGRESSION_PARITY_PENDING
    | P1_EWM_PARITY_PENDING
)

# ---------------------------------------------------------------------------
# P1 Extended：SQL/Polars 已实现；仅 cum/expanding 子集可 production safe
# ---------------------------------------------------------------------------
P1_EXTENDED_TS_CANONICALS: frozenset[str] = frozenset(
    {
        "WMA",
        "ts_mad",
        "ts_quantile",
        "ts_product",
        "ts_skew",
        "ts_regression_slope",
        "ts_time_slope",
        "ts_argmax",
        "ts_argmin",
        "ts_ratio",
    }
)

P1_EXTENDED_EWM_CANONICALS: frozenset[str] = frozenset({"ts_ema", "ewm_std", "ewm_var"})

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

# 经 parity 后可进 production safe 的 extended 子集（逐算子升级，非整批）
P1_EXTENDED_CUM_PRODUCTION_SAFE: frozenset[str] = frozenset(
    {
        "cum_sum",
        "cum_max",
        "cum_min",
        "cum_prod",
        "cum_delta",
        "expanding_mean",
        "expanding_sum",
        "count",
    }
)

P1_BATCH2_ROLLING_PARITY_PENDING: frozenset[str] = frozenset({"ts_argmax", "ts_argmin"})

P1_EXTENDED_PARITY_PENDING: frozenset[str] = (
    P1_EXTENDED_CANONICALS
    - P1_EXTENDED_CUM_PRODUCTION_SAFE
    - frozenset({"ts_ratio"})
) | P1_BATCH2_ROLLING_PARITY_PENDING

P1_POLARS_CORE_PRODUCTION_SAFE: frozenset[str] = (
    P1_GROUP_CANONICALS
    | P1_BINARY_TS_CANONICALS
    | P1_ROBUST_CANONICALS
    | P1_GOLDEN_VERIFIED_TS
    | P1_GOLDEN_VERIFIED_REGRESSION
)

P1_DUCKDB_CORE_PRODUCTION_SAFE: frozenset[str] = (
    P1_GROUP_CANONICALS
    | P1_BINARY_TS_CANONICALS
    | P1_ROBUST_CANONICALS
    | P1_GOLDEN_VERIFIED_TS
    | P1_GOLDEN_VERIFIED_REGRESSION
)

P1_EXTENDED_DUCKDB_CANONICALS: frozenset[str] = P1_EXTENDED_CUM_PRODUCTION_SAFE

P1_POLARS_PRODUCTION_SAFE: frozenset[str] = (
    P1_POLARS_CORE_PRODUCTION_SAFE | P1_EXTENDED_CUM_PRODUCTION_SAFE
)

P1_DUCKDB_PRODUCTION_SAFE: frozenset[str] = (
    P1_DUCKDB_CORE_PRODUCTION_SAFE | P1_EXTENDED_DUCKDB_CANONICALS
)


def dual_backend_production_safe() -> frozenset[str]:
    """PolarsLong + DuckDB 同时 production-safe（须 primitive evidence）。"""
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

    return frozenset(PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE)


def dual_backend_structural_candidates() -> frozenset[str]:
    """双后端 static 候选（implemented，非 evidence 认证）。"""
    from backend.polars_long_policy import POLARS_LONG_NATIVE
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    return frozenset(POLARS_LONG_NATIVE & SQL_IMPLEMENTED_CANONICALS)

# ---------------------------------------------------------------------------
# P2：map_groups / Python rolling / fill — 不进 production fast path
# ---------------------------------------------------------------------------
P2_MAP_GROUPS_CANONICALS: frozenset[str] = frozenset(
    {
        "ewm_corr",
        "ewm_cov",
        "ts_kurt",
        "ts_moment",
        "ts_max_buildup",
        "expanding_rank",
        "causal_linear_extrapolate",
        "quantile",
    }
)

P2_FILL_INTERPOLATE_CANONICALS: frozenset[str] = frozenset({"bfill", "causal_bfill"})

P2_NATIVE_PARITY_PENDING: frozenset[str] = frozenset()

P2_RESEARCH_ONLY: frozenset[str] = (
    P2_MAP_GROUPS_CANONICALS
    | P2_FILL_INTERPOLATE_CANONICALS
    | P2_TECHNICAL_RESEARCH_ONLY
)

# compile/runtime gate 统一 deferred 集合
FASTPATH_DEFERRED_CANONICALS: frozenset[str] = (
    P2_RESEARCH_ONLY
    | P1_TS_COMPLEX_PARITY_PENDING
    | P1_REGRESSION_PARITY_PENDING
    | P1_EWM_PARITY_PENDING
    | P1_EXTENDED_PARITY_PENDING
)

# The reviewed static daily surface supersedes historical tier batches.  A
# canonical cannot be both production daily and deferred.
from cleaned_operators.operator_surface import (
    DAILY_CANONICALS as _DAILY_CANONICALS,
    EXTENDED_ONLY_CANONICALS as _EXTENDED_CANONICALS,
    RESEARCH_ONLY_CANONICALS as _RESEARCH_CANONICALS,
)

# The old P0/P1 batches are retained above as historical documentation only.
# Runtime admission is authored from the reviewed surface, never from those
# mutable rollout lists.  ``protected_div`` is the sole internal lowering
# primitive that participates in production certification.
P0_PRODUCTION_FASTPATH_CANONICALS = frozenset(_DAILY_CANONICALS)
P1_POLARS_CORE_PRODUCTION_SAFE = frozenset()
P1_DUCKDB_CORE_PRODUCTION_SAFE = frozenset()
P1_POLARS_PRODUCTION_SAFE = frozenset()
P1_DUCKDB_PRODUCTION_SAFE = frozenset()

FASTPATH_DEFERRED_CANONICALS = frozenset(
    (_EXTENDED_CANONICALS | _RESEARCH_CANONICALS) - _DAILY_CANONICALS
)

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
FORBIDDEN_PRODUCTION_FASTPATH = frozenset(
    set(FORBIDDEN_PRODUCTION_FASTPATH) - set(_DAILY_CANONICALS)
)

# PolarsLong emitter 将 rolling_beta 映射为 ts_beta native
POLARS_NATIVE_ALIASES: dict[str, str] = {
    "rolling_beta": "ts_beta",
    "cum_std": "expanding_std",
    "WMA": "ts_decay_linear",
}


def resolve_polars_native_canonical(canon: str) -> str:
    """将 DSL 别名映射为 PolarsLong native emitter 使用的 canonical。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        解析别名后的 native canonical 名称。
    """
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return POLARS_NATIVE_ALIASES.get(name, name)


def is_p0_production_fastpath(canon: str) -> bool:
    """判断 canonical 是否属于 P0 production fast path 核心集。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否在 ``P0_PRODUCTION_FASTPATH_CANONICALS`` 白名单内。
    """
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in P0_PRODUCTION_FASTPATH_CANONICALS


def is_forbidden_production_fastpath(canon: str) -> bool:
    """判断 canonical 是否被禁止走 production fast path。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否为 ``micro_*`` 前缀或在显式禁止列表内。
    """
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    if name.startswith("micro_"):
        return True
    return name in FORBIDDEN_PRODUCTION_FASTPATH
