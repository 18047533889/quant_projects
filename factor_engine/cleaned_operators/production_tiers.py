# -*- coding: utf-8 -*-
"""Backend-independent production, research-execution and LQTP compatibility tiers."""
from __future__ import annotations

PANDAS_FIRST_PRODUCTION_CANONICALS: frozenset[str] = frozenset({
    "cs_resid","round","scale","sigmoid","true_range","ts_argmax","ts_argmin",
    "ts_decay_linear","ts_ema","ts_kurt","ts_mad","ts_max_buildup","ts_moment",
    "ts_product","ts_quantile","ts_regression_slope","ts_skew","ts_time_slope","ts_topk_sum",
})

# These are panel -> panel research operators, not diagnostics. They remain
# callable from the research/LQTP DSL even when they are not production-certified.
FACTOR_LIKE_RESEARCH_CANONICALS: frozenset[str] = frozenset({
    "coskewness_to_market","digital_count","expanding_rank","group_decay_linear",
    "hump_decay","idio_skew","idio_vol","intraday_vwap_deviation",
    "lqtp_historical_cvar","rank_corr","residual_momentum_capm",
    "rolling_beta_to_market","tail_beta","trade_when","ts_max_buildup","ts_moment",
    "ts_poly2_coeff","ts_poly2_resid","ts_sma_cn","ts_sum_decay",
})

LQTP_COMPAT_PARSE_CANONICALS: frozenset[str] = frozenset({
    "coskewness_to_market","cs_resid","expanding_rank","hump_decay","idio_skew","idio_vol",
    "lqtp_historical_cvar","rank_corr","real_turnover_rate","residual_momentum_capm",
    "rolling_beta_to_market","scale","sigmoid","tail_beta","true_range","ts_argmax","ts_argmin",
    "ts_decay_linear","ts_ema","ts_ewm_corr","ts_ewm_cov","ts_ewm_std","ts_ewm_var","ts_kurt",
    "ts_mad","ts_max_buildup","ts_moment","ts_poly2_resid","ts_product","ts_quantile",
    "ts_regression_slope","ts_skew","ts_sma_cn","ts_time_slope","ts_topk_sum",
})

LQTP_SOURCE_DEPENDENT_NAMES: frozenset[str] = frozenset({
    "ebitda_approx","enterprise_value","turnover_base",
})
