# -*- coding: utf-8 -*-
"""Backend-independent FactorEngine production and compatibility tiers.

Production safety and backend portability are separate properties.  An
operator may be production-grade with a reviewed Pandas/NumPy runtime while a
SQL implementation is unsupported or slower.  Conversely, merely being
parseable for LQTP compatibility never grants production admission.
"""
from __future__ import annotations


# Reviewed operators for which Pandas/NumPy is an acceptable production runtime
# even when Polars/DuckDB are not both certified.  Stateful EWM variants are
# intentionally excluded until their segmented/checkpoint contracts prove
# full-vs-incremental parity.
PANDAS_FIRST_PRODUCTION_CANONICALS: frozenset[str] = frozenset({
    "cs_resid",
    "round",
    "scale",
    "sigmoid",
    "true_range",
    "ts_argmax",
    "ts_argmin",
    "ts_decay_linear",
    "ts_ema",  # checkpoint contract already exists in the engine
    "ts_kurt",
    "ts_mad",
    "ts_max_buildup",
    "ts_moment",
    "ts_product",
    "ts_quantile",
    "ts_regression_slope",
    "ts_skew",
    "ts_time_slope",
    "ts_topk_sum",
    "safe_log_null",
})


# Factor-shaped research operators stay composable under surface='research'.
# They do not become production merely because they are executable.
FACTOR_LIKE_RESEARCH_CANONICALS: frozenset[str] = frozenset({
    "expanding_rank",
    "hump_decay",
    "idio_vol",
    "rank_corr",
})


# Historical LQTP formula corpus.  These names are parse-visible via a
# compatibility shell but production mode still checks lifecycle/PIT/runtime.
LQTP_COMPAT_PARSE_CANONICALS: frozenset[str] = frozenset({
    "coskewness_to_market",
    "cs_resid",
    "expanding_rank",
    "hump_decay",
    "idio_skew",
    "idio_vol",
    "rank_corr",
    "real_turnover_rate",
    "residual_momentum_capm",
    "rolling_beta_to_market",
    "scale",
    "sigmoid",
    "true_range",
    "ts_argmax",
    "ts_argmin",
    "ts_decay_linear",
    "ts_ema",
    "ts_ewm_corr",
    "ts_ewm_cov",
    "ts_ewm_std",
    "ts_ewm_var",
    "ts_kurt",
    "ts_mad",
    "ts_max_buildup",
    "ts_moment",
    "ts_poly2_resid",
    "ts_product",
    "ts_quantile",
    "ts_regression_slope",
    "ts_skew",
    "ts_time_slope",
    "ts_topk_sum",
})


# Known derived/source-backed names that must not be mistaken for physical
# StockDailyBar columns.
LQTP_SOURCE_DEPENDENT_NAMES: frozenset[str] = frozenset({
    "ebitda_approx",
    "enterprise_value",
})
