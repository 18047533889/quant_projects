# -*- coding: utf-8 -*-
"""Reviewed, fail-closed operator surfaces.

This module is the only source of truth for authoring surfaces.  Import-time
governance may validate these sets against the final registry, but must never
rewrite them.  ``daily`` is deliberately restricted to primitives with a
Pandas reference, native Polars execution, real DuckDB execution, edge parity,
and a no-fallback test.  Implemented operators that have not completed that
contract remain available on ``extended`` or ``research``.
"""
from __future__ import annotations

from typing import Iterable, Literal

OperatorSurface = Literal[
    "daily", "extended", "research", "unsafe", "legacy", "internal",
    "unclassified", "all",
]

DAILY_CANONICALS: frozenset[str] = frozenset({
    "abs", "add", "and_", "ceil", "clip", "coalesce", "cs_count",
    "cs_demean", "cs_mad", "cs_mad_zscore", "cs_mean", "cs_pct_rank",
    "cs_std", "cs_sum", "divide", "eq", "exp", "fillna_const", "floor",
    "ge", "group_count", "group_max", "group_mean", "group_min",
    "group_neutralize", "group_normalize", "group_rank", "group_std",
    "group_sum", "group_winsorize", "group_zscore", "gt", "inverse",
    "is_finite", "is_infinite", "is_not_null", "is_null", "le", "log",
    "log_abs", "lt", "maximum", "minimum", "multiply", "ne", "neg",
    "normalize", "not_", "or_", "period_average", "period_cagr",
    "period_change", "power", "quarter_from_cumulative", "rank",
    "safe_div_null", "sign", "signed_log", "signed_sqrt", "sqrt",
    "subtract", "tanh", "ts_autocorr", "ts_beta", "ts_corr", "ts_cov",
    "ts_delay", "ts_delta", "ts_log_return", "ts_max", "ts_mean",
    "ts_median", "ts_min", "ts_pct", "ts_rank", "ts_sharpe", "ts_std",
    "ts_sum", "ts_var", "ts_zscore", "ttm_from_cumulative",
    "ttm_from_quarterly", "where", "winsorize", "yoy_by_period", "zscore",
})

EXTENDED_ONLY_CANONICALS: frozenset[str] = frozenset({
    "ADX", "ATR_WILDER", "MACD_hist", "MACD_line", "MACD_signal",
    "RSI_WILDER", "acos", "arg", "asin", "atan", "atan2", "cbrt", "cos",
    "cosh", "cot", "cs_bucket", "cs_fill_mean", "cs_fill_median",
    "cs_multi_resid", "cs_neutralize", "cs_quantile", "cs_rank_gaussian",
    "cs_regression", "cs_resid", "cs_weighted_demean", "cs_weighted_mean",
    "cs_weighted_zscore", "cs_wls_resid", "csc", "exp_neg", "ffill_limit",
    "fix", "flex_max", "flex_min", "fundamental_staleness",
    "group_percentile", "group_weighted_mean", "group_weighted_zscore",
    "is_nan", "lerp", "log10", "log2", "period_lag", "period_stability",
    "price_spread_deviation", "real_turnover_rate", "revision_delta", "round", "scale",
    "saturate", "sec", "sigmoid", "signed_power", "sin", "sinh",
    "sqrt_abs", "square", "tan", "true_range", "truncate", "ts_argmax",
    "ts_argmin", "ts_bottomk_mean", "ts_bottomk_std", "ts_bottomk_sum",
    "ts_count_if", "ts_days_since", "ts_decay_exp_window", "ts_decay_linear",
    "ts_ema", "ts_ewm_corr", "ts_ewm_cov", "ts_ewm_std", "ts_ewm_var",
    "ts_kurt", "ts_last_if", "ts_mad", "ts_max_drawdown", "ts_mean_if",
    "ts_nth_value", "ts_partial_corr", "ts_product", "ts_quantile",
    "ts_ratio", "ts_regression_intercept", "ts_regression_r2",
    "ts_regression_resid", "ts_regression_slope", "ts_regression_tstat",
    "ts_skew", "ts_std_if", "ts_sum_if", "ts_tail_mean", "ts_time_slope",
    "ts_topk_mean", "ts_topk_std", "ts_topk_sum", "ts_trend_tstat",
    "ts_true_streak", "unitize", "winsorize_mean",
})

RESEARCH_ONLY_CANONICALS: frozenset[str] = frozenset({
    "coskewness_to_market", "digital_count", "group_decay_linear",
    "idio_skew", "intraday_vwap_deviation", "residual_momentum_capm",
    "rolling_beta_to_market", "tail_beta", "trade_when", "ts_max_buildup",
    "ts_moment", "ts_poly2_coeff", "ts_poly2_resid", "ts_sum_decay",
})

UNSAFE_CANONICALS: frozenset[str] = frozenset()
LEGACY_ONLY_CANONICALS: frozenset[str] = frozenset({"cube"})
INTERNAL_ONLY_CANONICALS: frozenset[str] = frozenset({
    "constant", "identity", "protected_div",
})

HIDDEN_DAILY_NAMES: frozenset[str] = frozenset({
    "cube", "cumulative_max", "cumulative_mean", "cumulative_min", "fmax",
    "fmin", "inv", "reciprocal", "sqr",
})


def classify_canonical(canonical: str) -> str:
    """Return the reviewed surface; unknown registrations fail closed."""
    if canonical in INTERNAL_ONLY_CANONICALS:
        return "internal"
    if canonical in EXTENDED_ONLY_CANONICALS:
        return "extended"
    if canonical in DAILY_CANONICALS:
        return "daily"
    if canonical in RESEARCH_ONLY_CANONICALS:
        return "research"
    if canonical in UNSAFE_CANONICALS:
        return "unsafe"
    if canonical in LEGACY_ONLY_CANONICALS:
        return "legacy"
    return "unclassified"


def is_dsl_name_allowed(name: str, canonical: str, *, surface: OperatorSurface = "daily") -> bool:
    """Return whether a registry name is visible on the requested surface."""
    if surface == "all":
        return True
    category = classify_canonical(canonical)
    if surface == "daily":
        return category == "daily" and name not in HIDDEN_DAILY_NAMES
    if surface == "extended":
        return category == "extended"
    if surface == "research":
        return category == "research"
    if surface == "unsafe":
        return category == "unsafe"
    if surface == "legacy":
        return category == "legacy" or name in HIDDEN_DAILY_NAMES
    if surface == "internal":
        return category == "internal"
    if surface == "unclassified":
        return category == "unclassified"
    raise ValueError(f"unknown operator surface: {surface!r}")


def unclassified_canonicals(canonicals: Iterable[str]) -> tuple[str, ...]:
    """Return runtime canonicals that have not passed surface review."""
    return tuple(sorted(c for c in canonicals if classify_canonical(c) == "unclassified"))


def surface_summary(canonicals: Iterable[str]) -> dict[str, int]:
    """Count canonical operators by reviewed surface."""
    out = {name: 0 for name in (
        "daily", "extended", "research", "unsafe", "legacy", "internal", "unclassified"
    )}
    for canonical in canonicals:
        out[classify_canonical(canonical)] += 1
    return out
