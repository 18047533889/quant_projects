# -*- coding: utf-8 -*-
"""Reviewed factor authoring surfaces.

``research`` is intentionally empty for factor-shaped operators. Anything that
returns a factor panel and remains in the DSL must be production hardened.
Composite domain indicators may keep public names while lowering to recipes.
"""
from __future__ import annotations
from typing import Iterable, Literal

OperatorSurface = Literal["daily","extended","research","unsafe","legacy","internal","unclassified","all"]

DAILY_CANONICALS: frozenset[str] = frozenset({
    "abs","add","and_","ceil","clip","coalesce","cs_count","cs_demean","cs_mad","cs_mad_zscore","cs_mean","cs_pct_rank","cs_std","cs_sum","divide","eq","exp","fillna_const","floor","ge","group_count","group_max","group_mean","group_min","group_neutralize","group_normalize","group_rank","group_std","group_sum","group_winsorize","group_zscore","gt","inverse","is_finite","is_infinite","is_not_null","is_null","le","log","log_abs","lt","maximum","minimum","multiply","ne","neg","normalize","not_","or_","period_average","period_cagr","period_change","power","quarter_from_cumulative","rank","safe_div_null","sign","signed_log","signed_sqrt","sqrt","subtract","tanh","ts_autocorr","ts_beta","ts_corr","ts_cov","ts_delay","ts_delta","ts_log_return","ts_max","ts_mean","ts_median","ts_min","ts_pct","ts_rank","ts_sharpe","ts_std","ts_sum","ts_var","ts_zscore","ttm_from_cumulative","ttm_from_quarterly","where","winsorize","yoy_by_period","zscore",
})

_PROMOTED_RESEARCH_FACTORS=frozenset({"coskewness_to_market","digital_count","expanding_rank","group_decay_linear","hump_decay","idio_skew","idio_vol","intraday_vwap_deviation","lqtp_historical_cvar","rank_corr","residual_momentum_capm","rolling_beta_to_market","tail_beta","trade_when","ts_max_buildup","ts_moment","ts_poly2_coeff","ts_poly2_resid","ts_sma_cn","ts_sum_decay"})

_TECHNICAL_EXTENSION_CANONICALS=frozenset({
"ts_prev_high","ts_prev_low","ts_distance_to_high","ts_distance_to_low","ts_breakout_high","ts_breakdown_low","ts_new_high","ts_new_low","ts_channel_position","ts_days_since_high","ts_days_since_low","ts_range_expansion","ts_confirmed_pivot_high","ts_confirmed_pivot_low","ts_last_pivot_high","ts_last_pivot_low","ts_pivot_high_age","ts_pivot_low_age","ts_resistance_level","ts_support_level","ts_resistance_slope","ts_support_slope","ts_distance_to_resistance","ts_distance_to_support","ts_resistance_break","ts_support_break",
"rolling_vwap","vwap_deviation","relative_volume","volume_zscore","dollar_volume","dollar_volume_zscore","volume_momentum","turnover_momentum","turnover_zscore","return_volume_corr","abs_return_volume_corr","signed_volume","signed_dollar_volume","rolling_obv","rolling_pvt","CMF","MFI","donchian_upper","donchian_lower","donchian_mid","donchian_position","bollinger_pct_b","bollinger_width","efficiency_ratio","choppiness_index","parkinson_vol","garman_klass_vol","rogers_satchell_vol","yang_zhang_vol","overnight_volatility","intraday_volatility","range_volatility","ulcer_index",
"candle_body","candle_abs_body","candle_range","candle_body_ratio","candle_upper_shadow","candle_lower_shadow","candle_upper_shadow_ratio","candle_lower_shadow_ratio","candle_close_location","candle_gap","candle_gap_pct","candle_direction","candle_range_atr","cdl_doji","cdl_hammer","cdl_inverted_hammer","cdl_shooting_star","cdl_marubozu","cdl_spinning_top","cdl_engulfing","cdl_inside_bar","cdl_outside_bar","cdl_dragonfly_doji","cdl_gravestone_doji","cdl_hanging_man","cdl_harami","cdl_harami_cross","cdl_piercing","cdl_dark_cloud_cover","cdl_morning_star","cdl_evening_star","cdl_three_white_soldiers","cdl_three_black_crows","cdl_tweezer_top","cdl_tweezer_bottom",
})

_STRUCTURE_V2_CANONICALS=frozenset({
"ts_nth_pivot_high","ts_nth_pivot_low","ts_nth_pivot_high_age","ts_nth_pivot_low_age","ts_pivot_high_count","ts_pivot_low_count","ts_pivot_high_spacing","ts_pivot_low_spacing","ts_swing_amplitude","ts_swing_amplitude_pct","ts_swing_duration","ts_swing_velocity","ts_swing_amplitude_atr","ts_channel_width","ts_channel_width_pct","ts_channel_width_atr","ts_channel_width_slope","ts_line_convergence","ts_line_parallelism","ts_resistance_fit_r2","ts_support_fit_r2","ts_pattern_symmetry","ts_impulse_return","ts_impulse_strength","ts_impulse_volume","ts_consolidation_width","ts_consolidation_slope","ts_consolidation_volume_decay",
"pattern_double_top","pattern_double_bottom","pattern_head_shoulders","pattern_inverse_head_shoulders","pattern_sym_triangle","pattern_ascending_triangle","pattern_descending_triangle","pattern_rising_wedge","pattern_falling_wedge","pattern_rectangle","pattern_rising_channel","pattern_falling_channel","pattern_broadening","pattern_bull_flag","pattern_bear_flag",
})

_LIQUIDITY_V2_CANONICALS=frozenset({
"average_volume","average_turnover","adv","abnormal_volume","abnormal_turnover","volume_volatility","turnover_volatility","volume_autocorr","turnover_autocorr","amihud_illiquidity","price_impact","return_per_turnover","volume_shock","turnover_shock","volume_acceleration","turnover_acceleration","up_volume_ratio","down_volume_ratio","signed_volume_imbalance","up_down_volume_ratio","volume_weighted_return","volume_weighted_momentum","price_volume_divergence","price_turnover_divergence","return_volume_beta","return_turnover_beta","ADL","ChaikinOscillator","ForceIndex","EaseOfMovement","bounded_nvi","bounded_pvi","zero_return_ratio","roll_spread_proxy","corwin_schultz_spread","high_low_spread_proxy","turnover_adjusted_volatility","volume_to_range",
})

_TECHNICAL_V2_CANONICALS=frozenset({
"DMI_plus","DMI_minus","DX","NATR","PPO","PPO_signal","PPO_hist","PVO","PVO_signal","PVO_hist","CMO","VortexPlus","VortexMinus","KeltnerMid","KeltnerUpper","KeltnerLower","KeltnerPosition","TSI","TSI_signal","UltimateOscillator","DEMA","TEMA","ichimoku_tenkan","ichimoku_kijun","ichimoku_senkou_a","ichimoku_senkou_b","ichimoku_cloud_width","ichimoku_cloud_position","KAMA","Supertrend","SupertrendDirection","PSAR",
})

_FUNDAMENTAL_V2_CANONICALS=frozenset({
"fin_lag","fin_diff","fin_pct_change","fin_log_change","fin_qoq","fin_yoy","fin_ttm","fin_average_balance","fin_growth","fin_cagr","fin_growth_acceleration","fin_growth_change","fin_growth_volatility","fin_growth_stability","fin_growth_persistence","fin_std","fin_mad","fin_cv","fin_stability","fin_range","fin_zscore_history","fin_percentile_history","fin_trend_slope","fin_trend_r2","fin_trend_tstat","fin_trend_acceleration","fin_monotonicity","fin_positive_streak","fin_negative_streak","fin_sign_change_count","fin_ratio","fin_common_size","fin_turnover","fin_divergence","fin_cash_earnings_gap","fin_accrual_ratio","fin_cash_conversion","fin_working_capital_change",
})

EXTENDED_ONLY_CANONICALS: frozenset[str] = frozenset({
"ADX","ATR_WILDER","MACD_hist","MACD_line","MACD_signal","RSI_WILDER","acos","arg","asin","atan","atan2","cbrt","cos","cosh","cot","cs_bucket","cs_fill_mean","cs_fill_median","cs_multi_resid","cs_neutralize","cs_quantile","cs_rank_gaussian","cs_regression","cs_resid","cs_weighted_demean","cs_weighted_mean","cs_weighted_zscore","cs_wls_resid","csc","exp_neg","ffill_limit","fix","flex_max","flex_min","fundamental_staleness","group_percentile","group_weighted_mean","group_weighted_zscore","is_nan","lerp","log10","log2","period_lag","period_stability","price_spread_deviation","real_turnover_rate","revision_delta","round","scale","saturate","sec","sigmoid","signed_power","sin","sinh","sqrt_abs","square","tan","true_range","truncate","ts_argmax","ts_argmin","ts_bottomk_mean","ts_bottomk_std","ts_bottomk_sum","ts_count_if","ts_days_since","ts_decay_exp_window","ts_decay_linear","ts_ema","ts_ewm_corr","ts_ewm_cov","ts_ewm_std","ts_ewm_var","ts_kurt","ts_last_if","ts_mad","ts_max_drawdown","ts_mean_if","ts_nth_value","ts_partial_corr","ts_product","ts_quantile","ts_ratio","ts_regression_intercept","ts_regression_r2","ts_regression_resid","ts_regression_slope","ts_regression_tstat","ts_skew","ts_std_if","ts_sum_if","ts_tail_mean","ts_time_slope","ts_topk_mean","ts_topk_std","ts_topk_sum","ts_trend_tstat","ts_true_streak","unitize","winsorize_mean",
}) | _PROMOTED_RESEARCH_FACTORS | _TECHNICAL_EXTENSION_CANONICALS | _STRUCTURE_V2_CANONICALS | _LIQUIDITY_V2_CANONICALS | _TECHNICAL_V2_CANONICALS | _FUNDAMENTAL_V2_CANONICALS

RESEARCH_ONLY_CANONICALS=frozenset(); UNSAFE_CANONICALS=frozenset(); LEGACY_ONLY_CANONICALS=frozenset({"cube"}); INTERNAL_ONLY_CANONICALS=frozenset({"constant","identity","protected_div"})
HIDDEN_DAILY_NAMES=frozenset({"cube","cumulative_max","cumulative_mean","cumulative_min","fmax","fmin","inv","reciprocal","sqr"})

def classify_canonical(canonical:str)->str:
    if canonical in INTERNAL_ONLY_CANONICALS:return "internal"
    if canonical in EXTENDED_ONLY_CANONICALS:return "extended"
    if canonical in DAILY_CANONICALS:return "daily"
    if canonical in RESEARCH_ONLY_CANONICALS:return "research"
    if canonical in UNSAFE_CANONICALS:return "unsafe"
    if canonical in LEGACY_ONLY_CANONICALS:return "legacy"
    return "unclassified"

def is_dsl_name_allowed(name:str,canonical:str,*,surface:OperatorSurface="daily")->bool:
    if surface=="all":return True
    category=classify_canonical(canonical)
    if surface=="daily":return category=="daily" and name not in HIDDEN_DAILY_NAMES
    if surface=="extended":return category=="extended"
    if surface=="research":return category=="research"
    if surface=="unsafe":return category=="unsafe"
    if surface=="legacy":return category=="legacy" or name in HIDDEN_DAILY_NAMES
    if surface=="internal":return category=="internal"
    if surface=="unclassified":return category=="unclassified"
    raise ValueError(f"unknown operator surface: {surface!r}")

def unclassified_canonicals(canonicals:Iterable[str])->tuple[str,...]:return tuple(sorted(c for c in canonicals if classify_canonical(c)=="unclassified"))
def surface_summary(canonicals:Iterable[str])->dict[str,int]:
    out={name:0 for name in ("daily","extended","research","unsafe","legacy","internal","unclassified")}
    for canonical in canonicals:out[classify_canonical(canonical)]+=1
    return out
