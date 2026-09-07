# -*- coding: utf-8
"""SQL 下推三层准入：Implemented / Parity Verified / Production Safe。

``SQL_CAPABLE_CANONICALS`` 仅表示 emitter 有实现，不等于 production-safe。
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from factor_engine.backend.production_fastpath_tiers import (
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
        "fin_ratio",
        "sqrt_abs",
        "book_to_price",
        "earnings_yield",
        "float_share_ratio",
        "free_float_share_ratio",
        "benchmark_relative_price",
        "benchmark_excess_return",
        "holder_concentration",
        "ashare_limit_distance",
        "row_sum_skipna",
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
        "size_neutralize",
        "industry_size_neutralize",
        "index_weight",
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
        "cs_physical_panel_coverage",
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
        "NATR",
        "MACD_line",
        "MACD_signal",
        "MACD_hist",
        "ts_count_if",
        "ts_sum_if",
        "ts_mean_if",
        "ts_std_if",
        "ts_last_if",
        "ts_days_since",
        "ts_true_streak",
        # R55 platform-audit P0: LQTP helpers promoted to daily — avg2(a,b) /
        # ts_positive_streak(x) now have emitter support (avg2 → (a+b)/2;
        # ts_positive_streak → consecutive-positive run), so mark them SQL
        # implemented alongside ts_true_streak.
        "avg2",
        "ts_positive_streak",
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
        # elementwise math
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "log2",
        "log10",
        "csc",
        "sec",
        "cot",
        "square",
        "cube",
        "sigmoid",
        "exp_neg",
        "saturate",
        "round",
        "fix",
        "signed_power",
        "lerp",
        # technical indicators (EMA / ATR / rolling)
        "DEMA",
        "TEMA",
        "PPO",
        "PPO_signal",
        "PPO_hist",
        "PVO",
        "PVO_signal",
        "PVO_hist",
        "TSI",
        "TSI_signal",
        "CMO",
        "VortexPlus",
        "VortexMinus",
        "KeltnerMid",
        "KeltnerUpper",
        "KeltnerLower",
        "KeltnerPosition",
        "DMI_plus",
        "DMI_minus",
        "DX",
        "ADX",
        "UltimateOscillator",
        "ADL",
        "rolling_adl_flow",
        "ChaikinOscillator",
        "ForceIndex",
        "EaseOfMovement",
        "CMF",
        "MFI",
        # candle geometry / return decomposition
        "candle_body",
        "candle_abs_body",
        "candle_range",
        "candle_body_ratio",
        "candle_upper_shadow_ratio",
        "candle_lower_shadow_ratio",
        "candle_upper_shadow",
        "candle_lower_shadow",
        "candle_close_location",
        "candle_body_position",
        "candle_close_strength",
        "candle_rejection_upper",
        "candle_rejection_lower",
        "candle_direction",
        "candle_gap",
        "candle_gap_pct",
        "candle_range_atr",
        "candle_gap_atr",
        "candle_overlap_ratio",
        "candle_inside_ratio",
        "open_close_return",
        "overnight_return",
        "open_to_vwap_return",
        "vwap_to_close_return",
        "limit_up_close",
        "limit_down_close",
        "true_range",
        # OHLC volatility estimators
        "parkinson_vol",
        "garman_klass_vol",
        "rogers_satchell_vol",
        "yang_zhang_vol",
        "overnight_volatility",
        "intraday_volatility",
        "range_volatility",
        "ulcer_index",
        "high_low_spread_proxy",
        # volume / turnover rolling
        "average_volume",
        "ts_average_volume",
        "average_turnover",
        "volume_to_range",
        "abnormal_volume",
        "abnormal_turnover",
        "relative_volume",
        "volume_zscore",
        "turnover_zscore",
        "volume_shock",
        "turnover_shock",
        "volume_momentum",
        "turnover_momentum",
        "volume_volatility",
        "turnover_volatility",
        "volume_autocorr",
        "turnover_autocorr",
        "volume_acceleration",
        "turnover_acceleration",
        "up_volume_ratio",
        "down_volume_ratio",
        "signed_volume_imbalance",
        "up_down_volume_ratio",
        "volume_weighted_return",
        "volume_weighted_momentum",
        "rolling_vwap",
        "vwap_deviation",
        "rolling_obv",
        "rolling_pvt",
        "signed_volume",
        "signed_dollar_volume",
        "dollar_volume",
        "dollar_volume_zscore",
        "adv",
        "amihud_illiquidity",
        "price_impact",
        "return_per_turnover",
        "return_volume_corr",
        "return_volume_beta",
        "return_turnover_beta",
        "price_volume_divergence",
        "price_turnover_divergence",
        "turnover_adjusted_volatility",
        "zero_return_ratio",
        "roll_spread_proxy",
        "corwin_schultz_spread",
        "bounded_nvi",
        "bounded_pvi",
        # band / channel ops
        "bollinger_pct_b",
        "bollinger_width",
        "donchian_upper",
        "donchian_lower",
        "donchian_mid",
        "donchian_position",
        # misc technical: efficiency ratio / choppiness / market coskewness
        "efficiency_ratio",
        "choppiness_index",
        "coskewness_to_market",
        # ts_* rolling window stats (validity / concentration / deviation / impulse / prior-extreme)
        "ts_valid_count",
        "ts_coverage_ratio",
        "ts_abs_concentration",
        "ts_abs_entropy",
        "ts_downside_deviation",
        "ts_upside_deviation",
        "ts_impulse_return",
        "ts_impulse_strength",
        "ts_impulse_volume",
        # Alpha-language SQL subset (2026-08): window-function-natural ops.
        "event_frequency",
        "ts_semivariance_balance",
        "ts_realized_quarticity",
        "ts_vol_of_vol",
        "ts_vol_acceleration",
        "ts_vol_term_structure",
        "ts_prev_high",
        "ts_prev_low",
        "ts_distance_to_high",
        "ts_distance_to_low",
        "ts_breakout_high",
        "ts_breakdown_low",
        "ts_channel_position",
        "ts_new_high",
        "ts_new_low",
        "ts_argmax_age",
        "ts_argmin_age",
        "ts_argmax_index_from_oldest",
        "ts_argmin_index_from_oldest",
        "ts_staleness",
        "ts_days_since_high",
        "ts_days_since_low",
        # cross-sectional fill / count / weighted / residual-percentile
        "cs_valid_count",
        "cs_coverage_ratio",
        "cs_fill_mean",
        "cs_fill_median",
        "cs_impute_mean",
        "cs_impute_median",
        "cs_residual_percentile",
        "cs_weighted_mean",
        "cs_weighted_demean",
        "cs_weighted_zscore",
        # ichimoku family
        "ichimoku_tenkan",
        "ichimoku_kijun",
        "ichimoku_senkou_a",
        "ichimoku_senkou_b",
        "ichimoku_cloud_width",
        "ichimoku_cloud_position",
        # candlestick patterns (cdl_*)
        "cdl_doji",
        "cdl_hammer",
        "cdl_inverted_hammer",
        "cdl_shooting_star",
        "cdl_marubozu",
        "cdl_spinning_top",
        "cdl_engulfing",
        "cdl_inside_bar",
        "cdl_outside_bar",
        "cdl_dragonfly_doji",
        "cdl_gravestone_doji",
        "cdl_hanging_man",
        "cdl_harami",
        "cdl_harami_cross",
        "cdl_piercing",
        "cdl_dark_cloud_cover",
        "cdl_morning_star",
        "cdl_evening_star",
        "cdl_three_white_soldiers",
        "cdl_three_black_crows",
        "cdl_tweezer_top",
        "cdl_tweezer_bottom",
        # Round-7 gap closure: simple windowed sign-ratio / central moment and
        # group reducers are genuinely SQL-expressible (see emitter branches).
        "ts_positive_ratio",
        "ts_negative_ratio",
        "ts_zero_ratio",
        "ts_moment",
        "group_valid_count",
        "group_weighted_mean",
        # Simple window function operators (NEW - 45 operators)
        "ts_first_value",
        "ts_last_value",
        "ts_row_number",
        "ts_dense_rank",
        "ts_percent_rank",
        "ts_cummax",
        "ts_cummin",
        "ts_cumcount",
        "ts_count",
        "ts_avg",
        "ts_variance",
        "ts_stddev",
        "ts_kurtosis",
        "ts_skewness",
        "ts_range",
        "ts_midpoint",
        "ts_sum_abs",
        "ts_mean_abs",
        "ts_abs_max",
        "ts_positive_count",
        "ts_negative_count",
        "ts_zero_count",
        "ts_positive_sum",
        "ts_negative_sum",
        "ts_positive_mean",
        "ts_negative_mean",
        # Cross-sectional window functions
        "cs_row_number",
        "cs_dense_rank",
        "cs_percent_rank",
        "cs_first_value",
        "cs_last_value",
        "cs_min",
        "cs_max",
        "cs_range",
        "cs_sum_abs",
        "cs_mean_abs",
        "cs_variance",
        "cs_stddev",
        "cs_skewness",
        "cs_kurtosis",
        # Expanding window functions
        "expanding_min",
        "expanding_max",
        "expanding_var",
        "expanding_count",
        "expanding_product",
    }
)

# Drop emitter names that are neither active registry canonicals nor IR meta /
# historical alias targets still referenced by the compiler.  Keeps
# ``is_sql_implemented`` honest for coverage reports.
_SQL_IMPLEMENTED_DEAD: frozenset[str] = frozenset(
    {
        "div_or_default",
        "log_fill_invalid",
        "nan_to_num",
        "protected_log",
        "protected_sqrt",
        "ewm_mean",
        "standardize",
        "volatility",
        "vwap",
        "WMA",
        "c_percentile",
        "rank_pct",
        "count",
        # keep rolling_beta — alias target still asserted by production SQL tests
    }
)
SQL_IMPLEMENTED_CANONICALS = frozenset(
    c for c in SQL_IMPLEMENTED_CANONICALS if c not in _SQL_IMPLEMENTED_DEAD
) | frozenset({"column", "literal"})
# 2026-08 geometry/math expansion: SQL pushdown subset with emitter branches in
# backend/sql_pushdown/emitter.py (exact-parity DuckDB window aggregates).
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "intraday_volatility_concentration",
    "intraday_volatility_entropy",
    "intraday_realized_semivariance_balance",
    "ts_crossing_speed",
    "ts_crossing_acceleration",
})

# 2026-08-08 Gemini-recommended primitives: SQL pushdown subset with emitter
# branches in backend/sql_pushdown/emitter.py (exact-parity DuckDB window
# aggregates / correlated top-k selection).
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "group_topk_mean",
    "cs_weighted_percentile_rank",
    "ts_cov_if",
    "ts_value_at_argextreme",
    "ts_weighted_standardized_moment",
    # 2026-08-08 Gemini V2 round: PIT-safe Abdi-Ranaldo spread (lead()ed
    # eta_{s+1} with an exclusive-end rolling frame, see emitter branch).
    "ts_abdi_ranaldo_spread",
})

# 2026-08-09 三后端一致性轮：新增可下推的简单/窗口/分组算子（emitter 分支见
# backend/sql_pushdown/emitter.py，含 candle 滚动 zscore/percentile、组内矩、
# leave-one-out 均值、截面 max-abs 归一化、展开 rank、abs-return 滚动 corr）。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "identity",
    "unitize",
    "candle_body_zscore",
    "candle_range_zscore",
    "candle_upper_shadow_zscore",
    "candle_lower_shadow_zscore",
    "candle_body_percentile",
    "candle_range_percentile",
    "group_skewness",
    "group_kurtosis",
    "group_quantile_spread",
    "group_ex_self_mean",
    # 2026-08-13 Batch 1: missing operator implementation
    "ts_lead",
    "ts_lag",
    "cs_rank",
    "cs_normalize",
    "cs_zscore",
    "cs_median",
    "cs_iqr",
    "cs_clip",
    "cs_trim_mean",
    "winsorize_mean",
    "fin_lag",
    "fin_diff",
    "fin_pct_change",
    "fin_growth",
    "fin_yoy",
    "fin_qoq",
    "fin_std",
    "fin_cv",
    "fin_range",
    "ts_cumprod",
    "ts_cumsum",
    "HMA",
    "WMA",
    "ts_sma",
    "rank_corr",
    "cs_neutralize",
    "group_mean",
    "group_std",
    "group_rank",
    "fin_log_change",
    "ts_returns",
    "flex_max",
    "flex_min",
    "digital_count",
    "ts_ratio",
    # 2026-08-13 Batch 2: more time series, cross-sectional, and financial
    "ts_skew",
    "ts_kurt",
    "ts_trimmed_mean",
    "ts_topk_mean",
    "ts_topk_sum",
    "ts_topk_std",
    "ts_bottomk_mean",
    "ts_bottomk_sum",
    "ts_bottomk_std",
    "fin_ttm",
    "fin_mean_abs_deviation",
    "fin_median_abs_deviation",
    "fin_positive_streak",
    "fin_negative_streak",
    "fin_sign_change_count",
    "fin_monotonicity",
    "fin_turnover",
    "group_max",
    "group_min",
    "group_count",
    "group_sum",
    "group_median",
    "group_quantile",
    "cs_robust_scale",
    "acos_bounded",
    "asin_bounded",
    "cos_phase",
    "sin_phase",

    "abs_return_volume_corr",
    "expanding_rank",
})

# 2026-09 tech/candle family: pure rolling/ewm SQL branches added to
# backend/sql_pushdown/emitter.py (exact-parity DuckDB window aggregates).
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "ALMA",
    "CoppockCurve",
    "ElderRay",
    "atr_pct",
    "atr_acceleration",
    "atr_zscore",
    "atr_percentile",
    "atr_short_long_ratio",
    "candle_body_strength",
    "candle_wick_balance",
    "candle_range_pct",
    "candle_pattern_count",
})

# EWMA/Wilder 平滑族历史注记（2026-08-27 起不再从 SQL 白名单移除）：
# RSI/ATR/DMI/DX/ADX/MACD/DEMA/TEMA/PPO/PVO/TSI/Keltner/ADL/CMF/
# ChaikinOscillator/ForceIndex 的 pandas ewm(adjust=False) NaN 缺口语义
# （绝对位置衰减 + 有效观测重新归一化）已由 emitter 精确递归 CTE 复刻
# （行数级 exact parity，见 emitter._ewm_adjust_false_sql）。故这些 canonical
# 保留在 SQL_IMPLEMENTED 中，register_sql_backends() 自动登记 backend='sql'
# 占位；research/validation 模式可真实下推 DuckDB。production 下推仍受
# SQL_PRODUCTION_SAFE 静态白名单 + 参数域认证约束。
# 其余仍保持 SQL 回退的 canonical：
#   cdl_hammer/cdl_hanging_man：pandas 参考嵌入 prior-trend 上下文（嵌套窗口，
#   DuckDB 禁止嵌套窗口函数）。
#   ts_time_slope/ts_upside_deviation/ts_weighted_standardized_moment/
#   ts_abdi_ranaldo_spread/ts_value_at_argextreme：pandas 内核暖启动/窗口位置
#   重索引与 SQL 分支不一致（精确复刻成本高），polars 已与 pandas 完全一致。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS - frozenset({
    "cdl_hammer", "cdl_hanging_man",
    "ts_time_slope", "ts_upside_deviation", "ts_weighted_standardized_moment",
    "ts_abdi_ranaldo_spread", "ts_value_at_argextreme",
})

# 2026-09-07 fin_* elementwise algebraic family: pure-SQL emitter branches in
# backend/sql_pushdown/emitter.py (ratio / abs-ratio / sum-diff-ratio over up to
# seven operands).  The trailing ``period_id`` input is structural PIT-alignment
# only and is not compiled.  These are elementwise (no period walk), so they are
# SQL-implemented but NOT production-safe until parity-verified.
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "fin_common_size",
    "fin_cash_conversion",
    "fin_acquisition_cash_intensity",
    "fin_borrowing_intensity",
    "fin_capex_intensity",
    "fin_goodwill_intensity",
    "fin_debt_repayment_intensity",
    "fin_contract_asset_intensity",
    "fin_contract_liability_intensity",
    "fin_oci_to_equity",
    "fin_interest_coverage_proxy",
    "fin_discontinued_operation_ratio",
    "fin_minority_profit_share",
    "fin_fair_value_income_dependence",
    "fin_investment_income_dependence",
    "fin_other_earnings_dependence",
    "fin_rd_capitalization_ratio",
    "fin_expectation_dispersion",
    "fin_cash_burn_runway",
    "fin_accrual_ratio",
    "fin_cash_earnings_gap",
    "fin_impairment_intensity",
    "fin_lease_intensity",
    "fin_rd_total_intensity",
    "fin_contract_asset_liability_gap",
    "fin_lease_asset_liability_gap",
    "fin_deferred_tax_gap",
    "fin_comprehensive_income_gap",
    "fin_roe_cash_gap",
    "fin_debt_service_coverage_proxy",
    "fin_actual_expectation_divergence",
    "fin_surprise",
    "fin_net_borrowing_cashflow",
    "fin_financing_gap",
    "fin_core_earnings_ratio",
    "fin_noncore_income_ratio",
    # wave2 fin/valuation (2026-09-07): 纯元素级比值表条目（_FIN_ELEMENTWISE_OPS /
    # _FIN_SQL_ELEMENTWISE 双表复用）。
    "free_float_turnover",
    "real_turnover_rate",
    "true_turnover_rate",
    "market_cap_free_cap_gap",
})

# 2026-09-07 wave2 csg: cs_shrink_to_group_mean / group_weighted_zscore —
# polars native + DuckDB SQL 双后端（组轴聚合，三方 parity 测试见
# tests/backend_parity/test_csg_wave2_parity.py）。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "cs_shrink_to_group_mean",
    "group_weighted_zscore",
})

# 2026-09-07 wave2 ts: overnight/intraday decomposition family — polars native
# + DuckDB SQL（trailing 窗口协方差/均值/同号占比/滚动 beta mispricing；
# tests/backend_parity/test_ts_wave2_parity.py）。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "ts_overnight_intraday_cov",
    "ts_overnight_intraday_spread",
    "ts_overnight_intraday_sign_agreement",
    "ts_opening_mispricing_score",
})

# 2026-09-07 wave2 csg: cs_universe_coverage — finite-x fraction over the
# declared universe (polars native + DuckDB SQL, tests/backend_parity/).
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "cs_universe_coverage",
})

# 2026-09-07 wave2 fin/valuation: 纯元素级比值表条目（_FIN_ELEMENTWISE_OPS /
# _FIN_SQL_ELEMENTWISE 双表复用，tests/backend_parity/）。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "free_float_turnover",
    "real_turnover_rate",
    "true_turnover_rate",
    "market_cap_free_cap_gap",
})

# 2026-09-07 wave2 misc: price_spread_deviation — x/trailing-finite-mean - 1
# （polars native + DuckDB SQL）。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "price_spread_deviation",
})

# 2026-09-08 wave3 valuation/shareholder: 股本比值 / 集中度差 / log gap 族
# （polars native 表条目 + holder_pledge_change / circulating_cap_ratio_change
# 独立分支，DuckDB SQL；tests/backend_parity/test_valuation_wave3_parity.py）。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "a_share_cap_ratio",
    "free_float_ratio",
    "holder_pledge_ratio",
    "holder_freeze_ratio",
    "holder_locked_share_ratio",
    "holder_float_concentration_gap",
    "holder_pledge_change",
    "val1_relative_valuation_gap",
    "valuation_pe_ttm_lyr_gap",
    "valuation_pcf_definition_gap",
    "valuation_pe_gap_signed_log",
    "valuation_pcf_gap_signed_log",
    "valuation_pe_gap_positive",
    "valuation_pcf_gap_positive",
    "circulating_cap_ratio_change",
})

# 2026-09-08 wave3 flow/momentum/quality window family — trailing-window
# statistics over the wave1_orderflow / wave1_cs_momentum / wave1_earnings
# pandas references (polars native + DuckDB SQL, 三方 parity 见
# tests/backend_parity/test_flowmom_wave3_parity.py)。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "ofi_volume_imbalance",
    "ofi_abs_imbalance_trend",
    "ofi_dominant_direction",
    "ofi_imbalance_agreement",
    "ofi_imbalance_cv",
    "ofi_imbalance_persistence",
    "ofi_reversal_rate",
    "ofi_volume_flow_regime",
    "ofi_zero_flow_balance",
    "m1_momentum_strength",
    "m1_momentum_stability",
    "m1_momentum_speed_change",
    "m1_volume_adjusted_momentum",
    "sv_net_flow_direction",
    "sv_own_flow_fraction",
    "sv_signed_volume_volatility",
    "sv_self_relative_change",
    "aq1_cash_flow_volatility",
    "aq1_accrual_stability",
    "aq1_cash_conversion_strength",
    "aq1_accrual_ratio_dispersion",
    "aq1_working_capital_accrual",
})

# 2026-09-08 wave3e cs/group: cs_bucket_fixed（固定边界分桶）/ EB 收缩 /
# group_ex_self_weighted_mean（组内 ex-self 加权均值）/ 诊断计数 / 多帧偏离
# 指数 — polars native + DuckDB SQL 双后端（cs_quantile_resid 分位回归残差
# 需 LP 迭代求解，双后端均不可忠实表达，保持 DEFER）；三方 parity 见
# tests/backend_parity/test_csgrp_wave3_parity.py。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "cs_bucket_fixed",
    "cs_empirical_bayes_shrinkage",
    "group_ex_self_weighted_mean",
    "group_feature_valid_member_count",
    "group_peer_deviation_index",
})

# 2026-09-08 wave3 ts: ts_robust_zscore_inclusive（scale="std" 组合：center/scale
# 只被当前行消费，两遍窗口精确；scale="mad" 组合需要把 center_t 广播进窗口每一行，
# DuckDB 禁嵌套窗口 → SQL 返回 None 诚实回退 polars 精确内核）。  polars 侧走
# registry polars 内核；三方 parity 见 tests/backend_parity/test_ts_wave3_parity.py。
# 其余 wave3 ts 算子（ts_mean/median_abs_deviation 的「单中心」偏差、
# ts_monotonicity 两两 O(n²)、ts_turning_point_ratio / ts_endpoint_deviation /
# ts_vol_shift_score / ts_recovery_fraction 需「窗口内尾部连续段 + 再排序」、
# ts_time_under_water / ts_current_drawdown_duration 的 running peak 基线起点
# 依赖输出行（跨窗口起点的段不可一次物化），DuckDB 禁嵌套窗口均不可精确表达）
# 诚实保持 SQL fallback，不登记 tiers。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "ts_robust_zscore_inclusive",
})

# 2026-09-08 wave3d ashare limit family — 涨跌停触碰/炸板/开板布尔 + 触碰/
# 炸板滚动计数 + 不对称度/事件密度 + 涨跌停量比 + 连板长度（polars native
# 分支 + DuckDB SQL 分支；tests/backend_parity/test_ashare_wave3_parity.py）。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "ashare_limit_up_touch",
    "ashare_limit_down_touch",
    "ashare_open_at_upper_limit",
    "ashare_limit_failed",
    "ashare_limit_open_failed",
    "ashare_limit_touch_count",
    "ashare_failed_limit_count",
    "ashare_limit_asymmetry",
    "ashare_limit_event_density",
    "ashare_limit_up_volume_ratio",
    "ashare_limit_down_volume_ratio",
    "ashare_limit_up_streak",
})

# 2026-09-08 wave3f ts2: range / consolidation / liquidity-beta family —
# polars native + DuckDB SQL 双后端（trailing 窗口极值族与 days_since 的 SQL
# 分支此前已登记；本块补 ts_range_expansion / ts_consolidation_width /
# ts_market_liquidity_beta / ts_industry_liquidity_beta；三方 parity 见
# tests/backend_parity/test_ts2_wave3_parity.py）。
SQL_IMPLEMENTED_CANONICALS = SQL_IMPLEMENTED_CANONICALS | frozenset({
    "ts_range_expansion",
    "ts_consolidation_width",
    "ts_market_liquidity_beta",
    "ts_industry_liquidity_beta",
})

# DuckDB 分层
DUCKDB_SQL_PARITY_VERIFIED: frozenset[str] = frozenset()
DUCKDB_SQL_PRODUCTION_SAFE: frozenset[str] = frozenset()

# ClickHouse 独立分层（无真实 integration test 前默认空）
CLICKHOUSE_SQL_PARITY_VERIFIED: frozenset[str] = frozenset()
CLICKHOUSE_SQL_PRODUCTION_SAFE: frozenset[str] = frozenset()

from factor_engine.backend.primitive_evidence import (
    DUCKDB_REAL_SQL_VERIFIED,
    POLARS_REFERENCE_PARITY_VERIFIED,
    PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
)

SQL_PARITY_VERIFIED_CANONICALS: frozenset[str] = frozenset(
    {"column", "literal"}
) | DUCKDB_REAL_SQL_VERIFIED

from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS as _DAILY_CANONICALS

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

# Cheap elementwise emitters that exist in SQL_IMPLEMENTED but stay on the
# *extended* surface (not daily).  They push down in research/validation mode
# via ``is_sql_capable``; promoting them to production SQL would require daily
# surface + six-way dual evidence.  Not forced — pandas (and Polars for
# ``scale``) is fine when the formula stays off the daily allowlist.
SQL_RESEARCH_SPEED_CANDIDATES: frozenset[str] = frozenset(
    {
        "cbrt",
        "truncate",
        "is_nan",
        "scale",
    }
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
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_IMPLEMENTED_CANONICALS


def is_sql_parity_verified(canon: str) -> bool:
    """判断 canonical 是否已通过 SQL parity 验证。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否在 ``SQL_PARITY_VERIFIED_CANONICALS`` 内。
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_PARITY_VERIFIED_CANONICALS


def is_sql_production_safe(canon: str) -> bool:
    """判断 canonical 是否在静态 SQL production-safe 白名单内。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        是否在 ``SQL_PRODUCTION_SAFE_CANONICALS`` 内（不含运行时降级）。
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in SQL_PRODUCTION_SAFE_CANONICALS


class _BoundedLRU:
    """R40 #64：entry/byte 双界 LRU 缓存（``max_entries`` + ``max_bytes``）。

    ``get``/``put`` 线程安全；``put`` 超过任一上限时按 LRU 逐出最旧项。
    """

    def __init__(self, *, max_entries: int = 16, max_bytes: int = 256 * 1024) -> None:
        self._max_entries = max(1, int(max_entries))
        self._max_bytes = max(1, int(max_bytes))
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _size(key: Any, value: Any) -> int:
        try:
            return max(1, len(str(key))) + max(1, len(repr(value)))
        except Exception:
            return 64

    def get(self, key: Any) -> Any | None:
        with self._lock:
            if key in self._data:
                value = self._data.pop(key)
                self._data[key] = value  # MRU
                return value
            return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            if key in self._data:
                old = self._data.pop(key)
                self._bytes -= self._size(key, old)
            self._data[key] = value
            self._bytes += self._size(key, value)
            while len(self._data) > self._max_entries or self._bytes > self._max_bytes:
                k, v = self._data.popitem(last=False)
                self._bytes -= self._size(k, v)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._data),
                "bytes": self._bytes,
                "max_entries": self._max_entries,
                "max_bytes": self._max_bytes,
            }


#: R40 #64：DuckDB capability 报告的 entry/byte 双界 LRU（旧实现是单个 frozenset，
#: 每次 refresh 覆盖、无界增长语义不明确；现在按 capability fingerprint 缓存
#: 多个报告，超限按 LRU 逐出）。
_DUCKDB_DOWNGRADE_CACHE: _BoundedLRU = _BoundedLRU()


def _capability_fingerprint(report: Any) -> str:
    """capability 报告 → 稳定 fingerprint（cache key 维度）。

    绑定 duckdb version + features 字典——同版本同能力复用同一降级集。
    """
    try:
        d = report.to_dict()
        import hashlib
        import json

        payload = json.dumps(
            {"version": d.get("version"), "features": d.get("features")},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return str(getattr(report, "version", "") or "unknown")


def duckdb_downgraded_canonicals(*, refresh: bool = False) -> frozenset[str]:
    """按当前 DuckDB 部署能力应从 production SQL 降级的 canonical（R40 #64 有界缓存）。

    ``refresh=True`` 强制重新探测；否则按 capability fingerprint 命中缓存。
    """
    try:
        from factor_engine.backend.sql_pushdown.duckdb_capabilities import (
            downgrade_sql_canonicals,
            get_duckdb_capability_report,
        )

        report = get_duckdb_capability_report(refresh=refresh)
    except Exception:
        return frozenset()
    fp = _capability_fingerprint(report)
    cached = _DUCKDB_DOWNGRADE_CACHE.get(fp)
    if cached is not None and not refresh:
        return cached
    try:
        downgraded = downgrade_sql_canonicals(report)
    except Exception:
        downgraded = frozenset()
    _DUCKDB_DOWNGRADE_CACHE.put(fp, downgraded)
    return downgraded


def duckdb_downgrade_cache_info() -> dict[str, Any]:
    """capability 缓存状态（测试/诊断用）。"""
    return _DUCKDB_DOWNGRADE_CACHE.info()


# ---------------------------------------------------------------------------
# R40 #61：production SQL safety = 静态白名单 ∩ parameter-domain backend 认证
# ---------------------------------------------------------------------------


def _parameter_domain_backend_certified(
    canon: str, *, backend: str = "duckdb_sql",
) -> bool:
    """该 canonical 是否已有参数域认证证据覆盖指定 backend（fail-closed）。

    惰性 import ``runtime.parameter_domain_store``（避免 sql_tiers 被反向 import
    时成环）；store 未装载 / 无该 backend 的 passed 认证点 → False。这是
    canonical 级别的强认证要求——production SQL 下推绝不能在**没有任何**该
    backend 参数域证据时放行（具体调用点的 exact membership 由执行期
    ``assert_parameter_point_certified`` 强制）。
    """
    try:
        from factor_engine.runtime.parameter_domain_store import get_parameter_domain_store

        store = get_parameter_domain_store()
    except Exception:
        return False
    if store is None or not getattr(store, "_loaded", False):
        return False
    try:
        return bool(store.operator_has_any_certified_region_by_backend(canon, backend))
    except Exception:
        return False


def is_sql_production_safe(canon: str) -> bool:
    """R40 #61：canonical 必须**同时**满足静态白名单 + 该 backend 参数域已认证。

    旧实现只看静态 ``SQL_PRODUCTION_SAFE_CANONICALS``；现在额外要求
    ParameterDomainCertificationStore 对 ``duckdb_sql`` 有 passed 认证点——
    白名单是"emitter 能编译"，参数域认证才是"该 backend 行为已被独立 oracle
    验证"。
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    if name not in SQL_PRODUCTION_SAFE_CANONICALS:
        return False
    return _parameter_domain_backend_certified(name, backend="duckdb_sql")


def effective_sql_production_safe(canon: str, *, refresh_duckdb: bool = False) -> bool:
    """静态 SQL_PRODUCTION_SAFE ∩ 参数域认证 减去 DuckDB 运行时能力降级。"""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    if not is_sql_production_safe(name):
        return False
    return name not in duckdb_downgraded_canonicals(refresh=refresh_duckdb)
