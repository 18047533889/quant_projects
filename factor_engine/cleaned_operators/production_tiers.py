# -*- coding: utf-8 -*-
"""Backend-independent FactorEngine production and compatibility tiers.

The old production gate historically treated "portable across Pandas + Polars +
DuckDB" as equivalent to "safe for production".  These are different
properties.  This module keeps them separate:

* semantic production: the operator is deterministic, causal/PIT-safe,
  shape-preserving, parameter-bounded, and has at least one reviewed runtime;
* backend certification: each backend is certified independently;
* portability: an optional stronger capability when several backends are
  certified for the same canonical operator.

``PANDAS_FIRST_PRODUCTION_CANONICALS`` is deliberately small and reviewable.
It contains operators whose canonical semantics are stable and whose Pandas /
NumPy implementation is an acceptable production runtime even when SQL or
Polars support is absent or slower.

``LQTP_COMPAT_PARSE_CANONICALS`` widens *parsing* only.  It does not grant
production admission.  Production mode still passes through the normal PIT,
shape, lifecycle and runtime gates.
"""
from __future__ import annotations


# Operators that may be production-grade with Pandas/NumPy as the certified
# runtime.  SQL/Polars implementations remain optional optimisations.
PANDAS_FIRST_PRODUCTION_CANONICALS: frozenset[str] = frozenset({
    "cs_resid",
    "round",
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
    "ts_product",
    "ts_quantile",
    "ts_regression_slope",
    "ts_skew",
    "ts_time_slope",
    "ts_topk_sum",
    # LQTP compatibility primitives added by cleaned_operators.lqtp_compat.
    "safe_log_null",
    "ts_sma_cn",
})


# Factor-like research operators that should stay executable through the
# FactorEngine research surface instead of being moved to the diagnostics-only
# ResearchToolRegistry.  They remain non-production unless separately promoted.
FACTOR_LIKE_RESEARCH_CANONICALS: frozenset[str] = frozenset({
    "expanding_rank",
    "hump_decay",
    "idio_vol",
    "rank_corr",
})


# Canonicals used by the historical LQTP formula corpus.  Visibility through a
# compatibility dispatcher is not a production certification.
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


# Names that are known to require additional logical sources rather than a
# physical column on the daily-bar table.  The LQTP normalizer uses this to fail
# early with a data-dependency error instead of leaking the name into DuckDB as
# a non-existent column.
LQTP_SOURCE_DEPENDENT_NAMES: frozenset[str] = frozenset({
    "ebitda_approx",
    "enterprise_value",
})
