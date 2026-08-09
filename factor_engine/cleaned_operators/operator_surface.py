# -*- coding: utf-8 -*-
"""Reviewed factor authoring surfaces.

``research`` is intentionally empty for factor-shaped operators. Anything that
returns a factor panel and remains in the DSL must be production hardened.
"""
from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import Iterable, Literal
OperatorSurface=Literal["daily","extended","research","unsafe","legacy","internal","unclassified","all"]

DAILY_CANONICALS=frozenset({
"abs","add","and_","ceil","clip","coalesce","cs_count","cs_demean","cs_mad","cs_mad_zscore","cs_mean","cs_pct_rank","cs_std","cs_sum","divide","eq","exp","fillna_const","floor","ge","group_count","group_max","group_mean","group_min","group_neutralize","group_normalize","group_rank","group_std","group_sum","group_winsorize","group_zscore","gt","inverse","is_finite","is_infinite","is_not_null","is_null","le","log","log_abs","lt","maximum","minimum","multiply","ne","neg","normalize","not_","or_","period_average","period_cagr","period_change","power","quarter_from_cumulative","rank","safe_div_null","sign","signed_log","signed_sqrt","sqrt","subtract","tanh","ts_autocorr","ts_beta","ts_corr","ts_cov","ts_delay","ts_delta","ts_log_return","ts_max","ts_mean","ts_median","ts_min","ts_pct","ts_rank","ts_sharpe","ts_std","ts_sum","ts_var","ts_zscore","ttm_from_cumulative","ttm_from_quarterly","where","winsorize","yoy_by_period","zscore",
})
# 2026-08 第三轮:most promoted-research factors are now certified and live on the
# daily surface (see _DAILY_RECERTIFIED_2026_08).  They must REMAIN in
# EXTENDED_ONLY_CANONICALS (the union below) because layer_governance's static
# partition check counts every registered operator against DAILY_CANONICALS ∪
# EXTENDED ∪ RESEARCH ∪ UNSAFE ∪ LEGACY ∪ INTERNAL and DAILY_FACTOR_MIGRATED is
# not one of those partitions.  classify_canonical checks the daily set first, so
# daily wins regardless.
_PROMOTED_RESEARCH_FACTORS=frozenset({"coskewness_to_market","digital_count","expanding_rank","group_decay_linear","hump_decay","idio_skew","idio_vol","intraday_vwap_deviation","lqtp_historical_cvar","rank_corr","residual_momentum_capm","rolling_beta_to_market","tail_beta","trade_when","ts_max_buildup","ts_moment","ts_poly2_coeff","ts_poly2_resid","ts_sma_cn","ts_sum_decay"})
_TECHNICAL_EXTENSION_CANONICALS=frozenset({
"ts_prev_high","ts_prev_low","ts_distance_to_high","ts_distance_to_low","ts_breakout_high","ts_breakdown_low","ts_new_high","ts_new_low","ts_channel_position","ts_days_since_high","ts_days_since_low","ts_range_expansion","ts_confirmed_pivot_high","ts_confirmed_pivot_low","ts_last_pivot_high","ts_last_pivot_low","ts_pivot_high_age","ts_pivot_low_age","ts_resistance_level","ts_support_level","ts_resistance_slope","ts_support_slope","ts_distance_to_resistance","ts_distance_to_support","ts_resistance_break","ts_support_break","rolling_vwap","vwap_deviation","relative_volume","volume_zscore","dollar_volume","dollar_volume_zscore","volume_momentum","turnover_momentum","turnover_zscore","return_volume_corr","abs_return_volume_corr","signed_volume","signed_dollar_volume","rolling_obv","rolling_pvt","CMF","MFI","donchian_upper","donchian_lower","donchian_mid","donchian_position","bollinger_pct_b","bollinger_width","efficiency_ratio","choppiness_index","parkinson_vol","garman_klass_vol","rogers_satchell_vol","yang_zhang_vol","overnight_volatility","intraday_volatility","range_volatility","ulcer_index","candle_body","candle_abs_body","candle_range","candle_body_ratio","candle_upper_shadow","candle_lower_shadow","candle_upper_shadow_ratio","candle_lower_shadow_ratio","candle_close_location","candle_gap","candle_gap_pct","candle_direction","candle_range_atr","cdl_doji","cdl_hammer","cdl_inverted_hammer","cdl_shooting_star","cdl_marubozu","cdl_spinning_top","cdl_engulfing","cdl_inside_bar","cdl_outside_bar","cdl_dragonfly_doji","cdl_gravestone_doji","cdl_hanging_man","cdl_harami","cdl_harami_cross","cdl_piercing","cdl_dark_cloud_cover","cdl_morning_star","cdl_evening_star","cdl_three_white_soldiers","cdl_three_black_crows","cdl_tweezer_top","cdl_tweezer_bottom",
})
_STRUCTURE_V2_CANONICALS=frozenset({
"ts_nth_pivot_high","ts_nth_pivot_low","ts_nth_pivot_high_age","ts_nth_pivot_low_age","ts_pivot_high_count","ts_pivot_low_count","ts_pivot_high_spacing","ts_pivot_low_spacing","ts_swing_amplitude","ts_swing_amplitude_pct","ts_swing_duration","ts_swing_velocity","ts_swing_amplitude_atr","ts_channel_width","ts_channel_width_pct","ts_channel_width_atr","ts_channel_width_slope","ts_line_convergence","ts_line_parallelism","ts_resistance_fit_r2","ts_support_fit_r2","ts_pattern_symmetry","ts_impulse_return","ts_impulse_strength","ts_impulse_volume","ts_consolidation_width","ts_consolidation_slope","ts_consolidation_volume_decay","pattern_double_top","pattern_double_bottom","pattern_head_shoulders","pattern_inverse_head_shoulders","pattern_sym_triangle","pattern_ascending_triangle","pattern_descending_triangle","pattern_rising_wedge","pattern_falling_wedge","pattern_rectangle","pattern_rising_channel","pattern_falling_channel","pattern_broadening","pattern_bull_flag","pattern_bear_flag",
})
_LIQUIDITY_V2_CANONICALS=frozenset({
"ts_average_volume","average_turnover","adv","abnormal_volume","abnormal_turnover","volume_volatility","turnover_volatility","volume_autocorr","turnover_autocorr","amihud_illiquidity","price_impact","return_per_turnover","volume_shock","turnover_shock","volume_acceleration","turnover_acceleration","up_volume_ratio","down_volume_ratio","signed_volume_imbalance","up_down_volume_ratio","volume_weighted_return","volume_weighted_momentum","price_volume_divergence","price_turnover_divergence","return_volume_beta","return_turnover_beta","ADL","ChaikinOscillator","ForceIndex","EaseOfMovement","bounded_nvi","bounded_pvi","zero_return_ratio","roll_spread_proxy","corwin_schultz_spread","high_low_spread_proxy","turnover_adjusted_volatility","volume_to_range",
})
_TECHNICAL_V2_CANONICALS=frozenset({
"DMI_plus","DMI_minus","DX","NATR","PPO","PPO_signal","PPO_hist","PVO","PVO_signal","PVO_hist","CMO","VortexPlus","VortexMinus","KeltnerMid","KeltnerUpper","KeltnerLower","KeltnerPosition","TSI","TSI_signal","UltimateOscillator","DEMA","TEMA","ichimoku_tenkan","ichimoku_kijun","ichimoku_senkou_a","ichimoku_senkou_b","ichimoku_cloud_width","ichimoku_cloud_position","KAMA","Supertrend","SupertrendDirection","PSAR",
})
_FUNDAMENTAL_V2_CANONICALS=frozenset({
"fin_lag","fin_diff","fin_pct_change","fin_log_change","fin_qoq","fin_yoy","fin_ttm","fin_average_balance","fin_growth","fin_cagr","fin_growth_acceleration","fin_growth_change","fin_growth_volatility","fin_growth_stability","fin_growth_persistence","fin_std","fin_mad","fin_cv","fin_stability","fin_range","fin_zscore_history","fin_percentile_history","fin_trend_slope","fin_trend_r2","fin_trend_tstat","fin_trend_acceleration","fin_monotonicity","fin_positive_streak","fin_negative_streak","fin_sign_change_count","fin_ratio","fin_common_size","fin_turnover","fin_divergence","fin_cash_earnings_gap","fin_accrual_ratio","fin_cash_conversion","fin_working_capital_change","fin_revision_delta","fin_revision_pct","fin_revision_direction","fin_revision_count","fin_revision_magnitude","fin_restated_flag","fin_days_since_update","fin_staleness",
})
_CANDLE_GEOMETRY_V2_CANONICALS=frozenset({
"candle_body_zscore","candle_range_zscore","candle_upper_shadow_zscore","candle_lower_shadow_zscore","candle_body_percentile","candle_range_percentile","candle_gap_atr","candle_body_position","candle_overlap_ratio","candle_inside_ratio","candle_close_strength","candle_rejection_upper","candle_rejection_lower","candlestick_pattern",
})
UNSAFE_CANONICALS=frozenset({"arg","tan","cot","sec","csc","cosh","sinh"})
_OPERATOR_EXPANSION_CANONICALS=frozenset({
"ts_quantile_range","ts_trimmed_mean","ts_robust_zscore","ts_positive_ratio","ts_negative_ratio","ts_zero_ratio","ts_abs_concentration","ts_abs_entropy","ts_min_if","ts_max_if","ts_quantile_if","ts_corr_if","ts_beta_if","ts_regression_resid_if","ts_transition_count","ts_time_since_change","ts_event_spacing_mean","ts_event_spacing_cv","ts_downside_deviation","ts_upside_deviation","ts_current_drawdown_duration","ts_time_under_water","ts_best_lag_corr","ts_price_delay","group_ex_self_mean","group_ex_self_weighted_mean","hierarchical_group_neutralize","cs_robust_resid","cs_huber_resid","cs_lad_resid","overnight_return","open_close_return","open_to_vwap_return","vwap_to_close_return","ashare_limit_distance","ashare_limit_up_touch","ashare_limit_down_touch","ashare_limit_one_price","ashare_limit_failed","ashare_open_at_upper_limit","ashare_limit_open_failed",
})
_REGRESSION_MODEL_CANONICALS=frozenset({
"ts_huber_regression_resid","ts_ridge_regression_resid","ts_quantile_regression_slope","ts_ar_coefficient","ts_variance_ratio","ts_cusum_break_score","ts_level_shift_score","ts_vol_shift_score",
})
_RELATION_EXPANSION_CANONICALS=frozenset({
"relation_hhi","relation_entropy","relation_topk_sum","relation_rank_weighted_sum","relation_category_share","relation_peer_weighted_mean_ex_self","relation_entry_count","relation_exit_count","relation_weighted_change","index_member","index_weight_change","index_entry_exit_event","index_membership_age","event_cumulative_return_past","event_abnormal_return_past","fin_applicability_mask","calendar_day_diff","fin_announcement_lag",
})
EXTENDED_ONLY_CANONICALS=frozenset({
"earnings_yield","book_to_price","float_share_ratio","free_float_share_ratio","true_turnover_rate","limit_up_close","limit_down_close","tradable_state","benchmark_excess_return","benchmark_relative_price","holder_concentration","ADX","ATR_WILDER","MACD_hist","MACD_line","MACD_signal","RSI_WILDER","acos","asin","atan","atan2","cbrt","cos","cs_bucket","cs_bucket_fixed","cs_bucket_historical","cs_fill_mean","cs_fill_median","cs_multi_resid","cs_neutralize","cs_quantile","cs_rank_gaussian","cs_regression","cs_resid","cs_weighted_demean","cs_weighted_mean","cs_weighted_zscore","cs_wls_resid","industry_size_neutralize","size_neutralize","exp_neg","ffill_limit","fix","flex_max","flex_min","fundamental_staleness","group_percentile","group_ts_decay_linear","group_weighted_mean","group_weighted_zscore","is_nan","lerp","log10","log2","period_lag","period_stability","price_spread_deviation","real_turnover_rate","revision_delta","round","scale","saturate","sigmoid","signed_power","sin","sqrt_abs","square","true_range","truncate","ts_argmax","ts_argmin","ts_bottomk_mean","ts_bottomk_std","ts_bottomk_sum","ts_count_if","ts_days_since","ts_decay_exp_window","ts_decay_linear","ts_ema","ts_ewm_corr","ts_ewm_cov","ts_ewm_std","ts_ewm_var","ts_kurt","ts_last_if","ts_mad","ts_max_drawdown","ts_mean_if","ts_nth_value","ts_partial_corr","ts_product","ts_quantile","ts_ratio","ts_regression_intercept","ts_regression_r2","ts_regression_resid","ts_regression_in_sample_resid","ts_regression_forecast_error","ts_regression_forecast_error_z","ts_regression_resid_mean","ts_regression_slope","ts_regression_tstat","ts_skew","ts_std_if","ts_sum_if","ts_tail_mean","ts_time_slope","ts_topk_mean","ts_topk_std","ts_topk_sum","ts_trend_tstat","ts_true_streak","unitize","winsorize_mean",
"coskewness_to_market","digital_count","expanding_rank","group_decay_linear","hump_decay","idio_skew","idio_vol","lqtp_historical_cvar","rank_corr","residual_momentum_capm","tail_beta","trade_when","ts_max_buildup","ts_moment","ts_poly2_coeff","ts_poly2_resid","ts_sma_cn","ts_sum_decay",
})
# 2026-08 stateful rule / episode / rotation pack (stateful.rule_language,
# stateful.events, stateful.sequential, stateful.episode, stateful.survival,
# stateful.rotation, stateful.drawdown_path).  Registered by a parallel session;
# classified extended (in-progress review, not yet promoted to daily).
_STATEFUL_RULE_CANONICALS=frozenset({
"cross_event","cs_rank_churn","cs_tail_retention","directional_change_extent","directional_change_state",
"event_refractory","state_deadband","state_ewm_if","state_hold","state_latch","state_since_reduce",
"state_since_trend_tstat","state_slew_limit","ts_current_drawdown_area","ts_cusum_pressure",
"ts_lag_of_peak_corr","ts_rank_if","ts_recovery_fraction","ts_state_age_percentile",
"ts_state_exit_hazard","ts_state_residual_life",
})
EXTENDED_ONLY_CANONICALS=EXTENDED_ONLY_CANONICALS|_PROMOTED_RESEARCH_FACTORS|_TECHNICAL_EXTENSION_CANONICALS|_STRUCTURE_V2_CANONICALS|_LIQUIDITY_V2_CANONICALS|_TECHNICAL_V2_CANONICALS|_FUNDAMENTAL_V2_CANONICALS|_CANDLE_GEOMETRY_V2_CANONICALS|_OPERATOR_EXPANSION_CANONICALS|_RELATION_EXPANSION_CANONICALS|_REGRESSION_MODEL_CANONICALS|_STATEFUL_RULE_CANONICALS|frozenset({"fin_mean_abs_deviation"})

# R5-50: the extended surface must NOT be mutated by ``EXTENDED_ONLY_CANONICALS =
# frozenset(set(old)|new)`` reassignments.  A consumer that did
# ``from ... import EXTENDED_ONLY_CANONICALS`` at import time keeps the OLD
# frozenset object, so a later module's reassignment silently never reaches it
# (module A sees 800 canonicals while module B sees 850).  Modules register via
# ``extend_extended_only`` (a live mutator) and consumers read via
# ``extended_only_canonicals()`` (always the current module global).


def extend_extended_only(names: frozenset[str] | set[str] | list[str]) -> None:
    """Register a batch of canonical names onto the extended-only surface.

    Mutates the module global in place (re-binding the frozenset) instead of
    letting each registering module reassign a private copy.  Every caller sees
    the union regardless of import order.
    """
    global EXTENDED_ONLY_CANONICALS
    EXTENDED_ONLY_CANONICALS = EXTENDED_ONLY_CANONICALS | frozenset(names)


def extended_only_canonicals() -> frozenset[str]:
    """Live read of the extended-only surface — always current, never a stale
    import-time snapshot."""
    return EXTENDED_ONLY_CANONICALS


def extend_research_only(names: frozenset[str] | set[str] | list[str]) -> None:
    """Register a batch of canonical names onto the research-only surface.

    Same live-mutator contract as :func:`extend_extended_only` (R5-50): mutates
    the module global in place instead of letting a registering module reassign
    a private copy, so every consumer sees the union regardless of import order.
    """
    global RESEARCH_ONLY_CANONICALS
    RESEARCH_ONLY_CANONICALS = RESEARCH_ONLY_CANONICALS | frozenset(names)


def retract_research_only(names: frozenset[str] | set[str] | list[str]) -> None:
    """Remove canonical names from the research-only surface (R9-P1-045).

    The add-only :func:`extend_research_only` cannot express a removal, which
    ``sequence_complexity`` needs to demote two canonicals off the research
    surface.  Same live-mutator contract: mutates the module global in place so
    every consumer sees the change regardless of import order.
    """
    global RESEARCH_ONLY_CANONICALS
    RESEARCH_ONLY_CANONICALS = RESEARCH_ONLY_CANONICALS - frozenset(names)


def retract_extended_only(names: frozenset[str] | set[str] | list[str]) -> None:
    """Remove canonical names from the extended-only surface (R9-P1-045).

    Symmetric live-mutator for the extended surface.
    """
    global EXTENDED_ONLY_CANONICALS
    EXTENDED_ONLY_CANONICALS = EXTENDED_ONLY_CANONICALS - frozenset(names)
# Factor-shaped extended operators that have passed semantic/PIT review and are
# NOT fail-closed migrate to the daily surface.  Recursive/stateful, source-blocked,
# non-factor, promoted-research (still under certification) and experimental model
# families stay on the extended surface.
DAILY_FACTOR_MIGRATED = frozenset({"ADL", "CMF", "CMO", "ChaikinOscillator", "DEMA", "DMI_minus", "DMI_plus", "DX", "EaseOfMovement", "ForceIndex", "KeltnerLower", "KeltnerMid", "KeltnerPosition", "KeltnerUpper", "MFI", "NATR", "PPO", "PPO_hist", "PPO_signal", "PVO", "PVO_hist", "PVO_signal", "TEMA", "TSI", "TSI_signal", "UltimateOscillator", "VortexMinus", "VortexPlus", "abnormal_turnover", "abnormal_volume", "abs_return_volume_corr", "acos", "adv", "amihud_illiquidity", "asin", "atan", "atan2", "average_turnover", "bollinger_pct_b", "bollinger_width", "bounded_nvi", "bounded_pvi", "candle_abs_body", "candle_body", "candle_body_percentile", "candle_body_position", "candle_body_ratio", "candle_body_zscore", "candle_close_location", "candle_close_strength", "candle_direction", "candle_gap", "candle_gap_atr", "candle_gap_pct", "candle_inside_ratio", "candle_lower_shadow", "candle_lower_shadow_ratio", "candle_lower_shadow_zscore", "candle_overlap_ratio", "candle_range", "candle_range_atr", "candle_range_percentile", "candle_range_zscore", "candle_rejection_lower", "candle_rejection_upper", "candle_upper_shadow", "candle_upper_shadow_ratio", "candle_upper_shadow_zscore", "candlestick_pattern", "cash_flow_lifecycle_stage", "cbrt", "cdl_dark_cloud_cover", "cdl_doji", "cdl_dragonfly_doji", "cdl_engulfing", "cdl_evening_star", "cdl_gravestone_doji", "cdl_hammer", "cdl_hanging_man", "cdl_harami", "cdl_harami_cross", "cdl_inside_bar", "cdl_inverted_hammer", "cdl_marubozu", "cdl_morning_star", "cdl_outside_bar", "cdl_piercing", "cdl_shooting_star", "cdl_spinning_top", "cdl_three_black_crows", "cdl_three_white_soldiers", "cdl_tweezer_bottom", "cdl_tweezer_top", "choppiness_index", "corwin_schultz_spread", "cos", "cs_bucket", "cs_bucket_fixed", "cs_bucket_historical", "cs_coverage_ratio", "cs_fill_mean", "cs_fill_median", "cs_impute_mean", "cs_impute_median", "cs_multi_resid", "cs_neutralize", "cs_quantile", "cs_rank_gaussian", "cs_regression", "cs_resid", "cs_valid_count", "cs_weighted_demean", "cs_weighted_mean", "cs_weighted_zscore", "cs_wls_resid", "date_diff_days", "dollar_volume", "dollar_volume_zscore", "donchian_lower", "donchian_mid", "donchian_position", "donchian_upper", "down_volume_ratio", "efficiency_ratio", "exp_neg", "ffill_limit", "fin_accrual_ratio", "fin_actual_expectation_divergence", "fin_average_balance", "fin_beat_streak", "fin_cagr", "fin_cash_conversion", "fin_cash_earnings_gap", "fin_common_size", "fin_component_score", "fin_cv", "fin_days_since_expectation_revision", "fin_days_since_update", "fin_diff", "fin_divergence", "fin_expectation_dispersion", "fin_expectation_revision", "fin_expectation_revision_count", "fin_expectation_revision_magnitude", "fin_expectation_revision_pct", "fin_expectation_revision_speed", "fin_growth", "fin_growth_acceleration", "fin_growth_change", "fin_growth_persistence", "fin_growth_stability", "fin_growth_volatility", "fin_lag", "fin_log_change", "fin_mad", "fin_miss_streak", "fin_monotonicity", "fin_negative_streak", "fin_pct_change", "fin_percentile_history", "fin_positive_streak", "fin_qoq", "fin_quarter_from_cumulative", "fin_range", "fin_ratio", "fin_restated_flag", "fin_revision_count", "fin_revision_delta", "fin_revision_direction", "fin_revision_magnitude", "fin_revision_pct", "fin_seasonal_percentile", "fin_seasonal_zscore", "fin_sign_change_count", "fin_stability", "fin_staleness", "fin_std", "fin_surprise", "fin_surprise_event_percentile", "fin_surprise_event_zscore", "fin_surprise_zscore", "fin_trend_acceleration", "fin_trend_r2", "fin_trend_slope", "fin_trend_tstat", "fin_ttm_cumulative", "fin_ttm_quarterly", "fin_turnover", "fin_working_capital_change", "fin_yoy", "fin_zscore_history", "fiscal_accrual_quality", "fiscal_ar_resid_std", "fiscal_asymmetric_elasticity", "fiscal_autocorr", "fiscal_change_direction_agreement", "fiscal_direction_consistency", "fiscal_pair_direction_agreement", "fiscal_perpetual_inventory", "fiscal_regression_resid_std", "fiscal_reversal_ratio", "fiscal_sign_agreement", "fiscal_sign_consistency", "fiscal_standardized_surprise", "fiscal_true_streak", "fix", "flex_max", "flex_min", "fundamental_staleness", "garman_klass_vol", "group_impute_median", "group_percentile", "group_ts_decay_linear", "group_valid_count", "group_weighted_mean", "group_weighted_zscore", "high_low_spread_proxy", "ichimoku_cloud_position", "ichimoku_cloud_width", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b", "ichimoku_tenkan", "industry_size_neutralize", "is_nan", "lerp", "limit_down_close", "limit_up_close", "log10", "log2", "log_positive_or_nan", "open_close_return", "open_to_vwap_return", "overnight_return", "overnight_volatility", "parkinson_vol", "pattern_123_bear", "pattern_123_bull", "pattern_ascending_triangle", "pattern_bear_flag", "pattern_bear_pennant", "pattern_breakdown_retest", "pattern_breakout_retest", "pattern_broadening", "pattern_bull_flag", "pattern_bull_pennant", "pattern_cup", "pattern_cup_handle", "pattern_descending_triangle", "pattern_double_bottom", "pattern_double_top", "pattern_falling_channel", "pattern_falling_wedge", "pattern_head_shoulders", "pattern_inverse_head_shoulders", "pattern_rectangle", "pattern_rising_channel", "pattern_rising_wedge", "pattern_rounding_bottom", "pattern_rounding_top", "pattern_sym_triangle", "pattern_triple_bottom", "pattern_triple_top", "period_lag", "period_stability", "price_impact", "price_spread_deviation", "price_turnover_divergence", "price_volume_divergence", "range_volatility", "relative_volume", "return_per_turnover", "return_turnover_beta", "return_volume_beta", "return_volume_corr", "revision_delta", "rogers_satchell_vol", "roll_spread_proxy", "rolling_obv", "rolling_pvt", "rolling_vwap", "round", "row_sum_skipna", "saturate", "scale", "sigmoid", "signed_dollar_volume", "signed_power", "signed_volume", "signed_volume_imbalance", "sin", "size_neutralize", "sqrt_abs", "square", "true_range", "true_turnover_rate", "truncate", "ts_argmax", "ts_argmax_age", "ts_argmax_index_from_oldest", "ts_argmin", "ts_argmin_age", "ts_argmin_index_from_oldest", "ts_average_volume", "ts_bottomk_mean", "ts_bottomk_std", "ts_bottomk_sum", "ts_breakdown_low", "ts_breakout_high", "ts_channel_position", "ts_channel_width", "ts_channel_width_atr", "ts_channel_width_pct", "ts_channel_width_slope", "ts_confirmed_pivot_high", "ts_confirmed_pivot_low", "ts_consolidation_slope", "ts_consolidation_volume_decay", "ts_consolidation_width", "ts_count_if", "ts_coverage_ratio", "ts_days_since", "ts_days_since_high", "ts_days_since_low", "ts_decay_exp_window", "ts_decay_linear", "ts_distance_to_high", "ts_distance_to_low", "ts_distance_to_resistance", "ts_distance_to_support", "ts_ffill_limited", "ts_impulse_return", "ts_impulse_strength", "ts_impulse_volume", "ts_kurt", "ts_last_if", "ts_last_pivot_high", "ts_last_pivot_low", "ts_line_convergence", "ts_line_parallelism", "ts_mad", "ts_max_drawdown", "ts_mean_if", "ts_new_high", "ts_new_low", "ts_nth_pivot_high", "ts_nth_pivot_high_age", "ts_nth_pivot_low", "ts_nth_pivot_low_age", "ts_nth_value", "ts_partial_corr", "ts_pattern_symmetry", "ts_pivot_high_age", "ts_pivot_high_count", "ts_pivot_high_spacing", "ts_pivot_low_age", "ts_pivot_low_count", "ts_pivot_low_spacing", "ts_prev_high", "ts_prev_low", "ts_product", "ts_quantile", "ts_range_expansion", "ts_ratio", "ts_regression_forecast_error", "ts_regression_forecast_error_z", "ts_regression_in_sample_resid", "ts_regression_intercept", "ts_regression_r2", "ts_regression_resid", "ts_regression_resid_mean", "ts_regression_slope", "ts_regression_tstat", "ts_resistance_break", "ts_resistance_fit_r2", "ts_resistance_level", "ts_resistance_slope", "ts_skew", "ts_staleness", "ts_std_if", "ts_sum_if", "ts_support_break", "ts_support_fit_r2", "ts_support_level", "ts_support_slope", "ts_swing_amplitude", "ts_swing_amplitude_atr", "ts_swing_amplitude_pct", "ts_swing_duration", "ts_swing_velocity", "ts_tail_mean", "ts_time_slope", "ts_topk_mean", "ts_topk_std", "ts_topk_sum", "ts_trend_tstat", "ts_true_streak", "ts_valid_count", "turnover_acceleration", "turnover_adjusted_volatility", "turnover_autocorr", "turnover_momentum", "turnover_shock", "turnover_volatility", "turnover_zscore", "ulcer_index", "unitize", "up_down_volume_ratio", "up_volume_ratio", "volume_acceleration", "volume_autocorr", "volume_momentum", "volume_shock", "volume_to_range", "volume_volatility", "volume_weighted_momentum", "volume_weighted_return", "volume_zscore", "vwap_deviation", "vwap_to_close_return", "winsorize_mean", "yang_zhang_vol", "zero_return_ratio"})

# 2026-08 第二轮提升:experimental 因子算子(非隔离、非递归、非 source-blocked、非
# promoted-research)升到 daily 表面,使其可用于 daily DSL。生命周期仍 experimental,
# 待证据域重新认证后转 production。
_DAILY_PROMOTED_EXPERIMENTAL = frozenset({"a_share_cap_ratio", "altman_z_score", "ashare_limit_distance", "ashare_limit_down_touch", "ashare_limit_failed", "ashare_limit_one_price", "ashare_limit_open_failed", "ashare_limit_up_touch", "ashare_open_at_upper_limit", "benchmark_excess_return", "benchmark_relative_price", "book_to_price", "calendar_day_diff", "capital_change_age", "capital_change_magnitude", "circulating_cap_unlock_proxy", "cs_actual_lof_score", "cs_knn_distance", "cs_local_density_score", "cs_mahalanobis_distance", "cs_quantile_resid", "cs_relative_density_ratio", "cs_residual_percentile", "cs_ridge_resid", "cs_robust_mahalanobis_mad", "cs_robust_resid", "cs_shrinkage_mahalanobis", "cs_spline_resid", "cs_huber_resid", "cs_lad_resid", "earnings_yield", "event_abnormal_return_past", "event_active_count", "event_arithmetic_return_sum", "event_cumulative_return_past", "event_decay_asof", "event_log_return_sum", "event_return_since_last", "fin_acquisition_cash_intensity", "fin_announcement_lag", "fin_applicability_mask", "fin_borrowing_intensity", "fin_capex_growth", "fin_capex_intensity", "fin_cash_burn_runway", "fin_cash_sales_divergence", "fin_cashflow_persistence", "fin_comprehensive_income_gap", "fin_contract_asset_growth", "fin_contract_asset_intensity", "fin_contract_asset_liability_gap", "fin_contract_liability_growth", "fin_contract_liability_intensity", "fin_core_earnings_ratio", "fin_debt_repayment_intensity", "fin_debt_service_coverage_proxy", "fin_deferred_tax_gap", "fin_delta_noa", "fin_discontinued_operation_ratio", "fin_earnings_cash_gap_volatility", "fin_earnings_persistence", "fin_earnings_smoothness", "fin_equity_capital_growth", "fin_expense_sales_divergence", "fin_fair_value_income_dependence", "fin_financing_gap", "fin_fundamental_strength_score", "fin_goodwill_intensity", "fin_goodwill_risk_score", "fin_impairment_intensity", "fin_interest_coverage_proxy", "fin_inventory_sales_divergence", "fin_investment_income_dependence", "fin_lease_asset_liability_gap", "fin_lease_intensity", "fin_margin_persistence", "fin_minority_profit_share", "fin_net_borrowing_cashflow", "fin_net_debt_issuance", "fin_noncore_income_ratio", "fin_oci_to_equity", "fin_other_earnings_dependence", "fin_rd_capitalization_ratio", "fin_rd_total_intensity", "fin_receivable_sales_divergence", "fin_roe_cash_gap", "fin_total_operating_accruals", "fin_working_capital_accruals", "float_share_ratio", "free_float_ratio", "free_float_share_ratio", "free_float_turnover", "free_to_circulating_ratio", "group_ex_self_mean", "group_ex_self_weighted_mean", "group_leader_laggard_exposure", "group_multi_level_rank_consistency", "group_peer_beta_deviation", "group_peer_deviation_index", "group_peer_information_diffusion", "group_return_dispersion_exposure", "hierarchical_group_neutralize", "holder_class_entropy", "holder_common_holding_peer_return", "holder_concentration", "holder_concentration_acceleration", "holder_concentration_slope", "holder_float_concentration_gap", "holder_freeze_concentration", "holder_freeze_ratio", "holder_id_matched_churn", "holder_id_matched_entry_share", "holder_id_matched_exit_share", "holder_id_overlap_ratio", "holder_locked_share_ratio", "holder_nature_entropy", "holder_peer_return_breadth", "holder_pledge_change", "holder_pledge_churn", "holder_pledge_concentration", "holder_pledge_ratio", "holder_pledged_holder_count", "holder_share_weighted_rank_migration", "holder_shareholder_network_centrality", "holder_shareholder_overlap_ratio", "index_entry_exit_event", "index_event_decay", "index_member", "index_membership_age", "index_weight", "index_weight_change", "index_weight_gap_to_free_float", "intra_abs_return_profile_cosine", "intra_amihud", "intra_amount_profile_cosine", "intra_amount_profile_jsd", "intra_beta_asymmetry", "intra_bipower_variation", "intra_concentration", "intra_continuous_variance", "intra_down_down_semibeta", "intra_down_up_semibeta", "intra_drawdown_depth", "intra_drawdown_duration", "intra_drawdown_recovery_half_life", "intra_entropy", "intra_extreme_bar_return", "intra_high_time", "intra_idiosyncratic_kurtosis", "intra_idiosyncratic_kurtosis_ex_self", "intra_idiosyncratic_skewness", "intra_idiosyncratic_skewness_ex_self", "intra_idiosyncratic_variance", "intra_idiosyncratic_variance_ex_self", "intra_interval_amount_share", "intra_interval_illiquidity", "intra_interval_realized_variance", "intra_interval_return", "intra_interval_volume_share", "intra_interval_vwap_deviation", "intra_jump_clustering", "intra_jump_concentration", "intra_jump_count", "intra_jump_first_time", "intra_jump_last_time", "intra_jump_ratio", "intra_jump_variation", "intra_kyle_lambda_proxy", "intra_limit_duration", "intra_limit_first_hit_time", "intra_limit_reopen_count", "intra_longest_above_vwap_streak", "intra_longest_below_vwap_streak", "intra_low_time", "intra_lunch_gap_return", "intra_market_model_r2", "intra_market_model_r2_ex_self", "intra_max_drawdown", "intra_max_drawup", "intra_negative_jump_variation", "intra_negative_tail_variation", "intra_path_efficiency", "intra_positive_jump_variation", "intra_positive_tail_variation", "intra_price_vwap_max_negative_excursion", "intra_price_vwap_max_positive_excursion", "intra_profile_earth_mover_distance", "intra_realized_beta", "intra_realized_beta_ex_self", "intra_realized_correlation", "intra_realized_correlation_ex_self", "intra_realized_kurtosis", "intra_realized_quarticity", "intra_realized_semivariance", "intra_realized_skewness", "intra_realized_variance", "intra_return_activity_corr", "intra_return_profile_cosine", "intra_same_slot_momentum", "intra_same_slot_reversal", "intra_segment_amount_share", "intra_segment_realized_vol", "intra_segment_return", "intra_segment_volume_share", "intra_segment_vwap_deviation", "intra_signed_imbalance_proxy", "intra_signed_jump_ratio", "intra_signed_return_profile_cosine", "intra_signed_tail_variation_ratio", "intra_tail_event_count", "intra_time_above_vwap", "intra_tripower_quarticity", "intra_up_down_semibeta", "intra_up_up_semibeta", "intra_volume_profile_cosine", "intra_volume_profile_jsd", "intra_vwap_above_ratio", "intra_vwap_cross_count", "intra_vwap_path_curvature", "intra_vwap_path_curvature_pct", "intra_vwap_path_slope", "intra_vwap_path_slope_pct", "intra_vwap_reversion_speed", "market_cap_free_cap_gap", "piotroski_f_score", "real_turnover_rate", "relation_category_share", "relation_entropy", "relation_hhi", "relation_peer_weighted_mean_ex_self", "relation_rank_weighted_sum", "relation_topk_sum", "suspension_status_coverage", "tradable_state", "ts_abs_concentration", "ts_abs_entropy", "ts_ar_coeff_stability", "ts_ar_coefficient", "ts_ar_forecast", "ts_ar_innovation", "ts_ar_innovation_z", "ts_ar_prior_coeff", "ts_ar_prior_forecast", "ts_ar_prior_innovation", "ts_ar_prior_innovation_z", "ts_best_lag_corr", "ts_beta_if", "ts_corr_if", "ts_current_drawdown_duration", "ts_cusum_break_score", "ts_downside_deviation", "ts_event_spacing_cv", "ts_event_spacing_mean", "ts_expectile_beta_spread", "ts_expectile_regression_coeff", "ts_expectile_regression_coeff_prior", "ts_expectile_regression_forecast_error", "ts_expectile_regression_resid", "ts_gap_fill_ratio", "ts_gap_reversion_ratio", "ts_gap_survival_duration", "ts_huber_regression_coeff", "ts_huber_regression_coeff_prior", "ts_huber_regression_forecast_error", "ts_huber_regression_forecast_error_z", "ts_huber_regression_resid", "ts_huber_regression_resid_z", "ts_industry_liquidity_beta", "ts_level_shift_score", "ts_market_liquidity_beta", "ts_max_if", "ts_mean_reversion_half_life", "ts_min_if", "ts_multi_regression_adjusted_r2_prior", "ts_multi_regression_coeff", "ts_multi_regression_coeff_prior", "ts_multi_regression_coeff_stability", "ts_multi_regression_forecast_error", "ts_multi_regression_forecast_error_z", "ts_multi_regression_r2", "ts_multi_regression_r2_prior", "ts_multi_regression_resid", "ts_multi_regression_resid_z", "ts_negative_ratio", "ts_opening_mispricing_score", "ts_overnight_intraday_cov", "ts_overnight_intraday_sign_agreement", "ts_overnight_intraday_spread", "ts_positive_ratio", "ts_price_delay", "ts_quantile_beta_spread", "ts_quantile_if", "ts_quantile_range", "ts_quantile_regression_coeff", "ts_quantile_regression_resid", "ts_quantile_regression_slope", "ts_regression_resid_if", "ts_ridge_regression_coeff", "ts_ridge_regression_coeff_prior", "ts_ridge_regression_forecast_error", "ts_ridge_regression_forecast_error_z", "ts_ridge_regression_resid", "ts_ridge_regression_resid_z", "ts_robust_zscore", "ts_time_since_change", "ts_time_under_water", "ts_transition_count", "ts_trimmed_mean", "ts_upside_deviation", "ts_variance_ratio", "ts_variance_ratio_slope", "ts_vol_shift_score", "ts_zero_ratio", "valuation_cashflow_disagreement", "valuation_growth_mismatch", "valuation_pcf_definition_gap", "valuation_pe_ttm_lyr_gap", "valuation_quality_mismatch", "zmijewski_score"})
# 2026-08:完成 unknown-state 重做(S9)后从隔离清单解除的算子,升到 daily。
_DAILY_UNISOLATED = frozenset({"index_reconstitution_churn", "listing_age", "suspension_frequency"})
# 2026-08 第三轮:逐个复核剩余 48 个 extended 算子后,将 43 个已达成
# production 级准入的算子升到 daily 表面 ——
#   * 递归/状态族(19):segmented checkpoint 运行时已全覆盖
#     (含本轮为 KAMA/Supertrend/SupertrendDirection/PSAR/hump_decay/
#     expanding_rank/ts_sma_cn 新增的分段实现;trade_when 经核实为逐元素
#     条件选择,非递归,从 full-replay 清单移除)。
#   * 隔离族重写(9):holder_* 改按 ShareholderId 匹配(_id_matched);
#     multi_index_entry_intensity 改 unknown-state 语义;
#     relation_entry/exit/weighted_change 已按逐对有效(PIT)验收。
#   * market-model 族(5):tail_beta/residual_momentum_capm/coskewness_to_market/
#     idio_vol/idio_skew 完成因果性(PIT)认证,移出 experimental。
#   * intraday_volatility:经核实为日频 open/close 波动率,解除 source-block。
# 保留 extended 的 5 个:fin_ttm / rolling_beta_to_market(迁移桩,新名已 daily)、
# relation_distinct_count / relation_overlap_ratio(逐日广播的非因子工具)、
# intraday_vwap_deviation(真 session-aware,需分钟源)。
_DAILY_RECERTIFIED_2026_08 = frozenset({
    "ADX", "ATR_WILDER", "KAMA", "MACD_hist", "MACD_line", "MACD_signal",
    "PSAR", "RSI_WILDER", "Supertrend", "SupertrendDirection",
    "coskewness_to_market", "digital_count", "expanding_rank",
    "group_decay_linear", "holder_entry_share", "holder_exit_share",
    "holder_net_entry_share", "holder_rank_stability", "holder_weighted_churn",
    "hump_decay", "idio_skew", "idio_vol", "intraday_volatility",
    "lqtp_historical_cvar", "multi_index_entry_intensity", "rank_corr",
    "relation_entry_count", "relation_exit_count", "relation_weighted_change",
    "residual_momentum_capm", "tail_beta", "trade_when", "ts_ema",
    "ts_ewm_corr", "ts_ewm_cov", "ts_ewm_std", "ts_ewm_var",
    "ts_max_buildup", "ts_moment", "ts_poly2_coeff", "ts_poly2_resid",
    "ts_sma_cn", "ts_sum_decay",
})
DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | _DAILY_PROMOTED_EXPERIMENTAL | _DAILY_UNISOLATED | _DAILY_RECERTIFIED_2026_08)
# 2026-08 final pack: 61 new atomics (robust tail / nonlinear dependence /
# complexity / A-share state machine / relation-group distribution / intraday
# time-structure v2) are production-hardened factor operators.  They must also
# REMAIN in EXTENDED_ONLY_CANONICALS (each module unions its own names into it at
# import) because layer_governance's static partition check counts every
# registered operator against the six partitions; DAILY_FACTOR_MIGRATED is a
# classification refinement, not a partition.  classify_canonical checks the
# daily set first, so daily wins regardless.
_DAILY_FINAL_PACK_2026_08 = frozenset({
    # group 1 — robust tail
    "ts_lower_partial_moment", "ts_upper_partial_moment", "ts_expected_shortfall",
    "ts_quantile_skew", "ts_quantile_kurtosis", "ts_tail_ratio", "ts_extreme_cluster_ratio",
    # group 2 — nonlinear dependence
    "ts_distance_corr", "ts_distance_cov", "ts_mutual_information",
    "ts_lagged_mutual_information", "ts_upper_tail_coexceedance_probability", "ts_lower_tail_dependence",
    # group 3 — complexity / long memory
    "ts_permutation_entropy", "ts_weighted_permutation_entropy",
    "ts_permutation_transition_entropy", "ts_sample_entropy", "ts_hurst_dfa",
    "ts_higuchi_fractal_dimension", "ts_variogram_slope", "ts_autocorr_decay_half_life",
    # group 4 — A-share limit/suspension state machine
    "ashare_limit_up_streak", "ashare_limit_down_streak", "ashare_days_since_limit_up",
    "ashare_days_since_limit_down", "ashare_limit_touch_count", "ashare_failed_limit_count",
    "ashare_one_price_limit_streak", "ashare_limit_event_density", "ashare_limit_asymmetry",
    "ashare_suspension_episode_length", "ashare_limit_open_up_streak",
    "ashare_limit_open_down_streak", "ashare_limit_up_volume_ratio",
    "ashare_limit_down_volume_ratio",
    # group 5 — relation / group distribution
    "relation_topk_concentration", "relation_distribution_skew", "relation_distribution_kurtosis",
    "relation_hhi_change", "relation_entropy_change", "relation_concentration_acceleration",
    "relation_rank_mobility", "relation_share_mobility", "group_skewness", "group_kurtosis",
    "group_quantile_spread", "group_tail_ratio",
    # group 6 — intraday time-structure v2
    "intra_bar_range_persistence", "intra_bar_range_deviation", "intra_tail_volume_share",
    "intra_volume_price_alignment", "intra_ute_high", "intra_ute_low",
    "intra_slot_volume_surprise", "intra_slot_amount_surprise", "intra_slot_volatility_surprise",
    "intra_market_lead_lag_ex_self", "intra_industry_lead_lag_ex_self",
    "intra_session_return_asymmetry", "intra_close_participation", "intra_high_low_affinity",
})
DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | _DAILY_FINAL_PACK_2026_08)
# 2026-08 alpha-language expansion: run/hysteresis state, path geometry,
# distribution shift, volatility structure, cs locality, events + report
# wrappers.  Same contract as the final pack: each module keeps its names in
# EXTENDED_ONLY_CANONICALS at import (partition check) and this frozenset
# migrates them to the daily surface.  Aliases (event_age etc.) resolve onto
# already-daily canonicals and need no surface entry here.
_DAILY_ALPHA_LANGUAGE_2026_08 = frozenset({
    "ts_run_strength", "ts_run_efficiency", "ts_run_concentration",
    "ts_hysteresis_state", "ts_hysteresis_age", "ts_state_integral",
    "ts_state_entry_strength", "ts_transition_intensity", "ts_sign_persistence",
    "ts_sign_cluster_index",
    "ts_monotonicity", "ts_turning_rate", "ts_turning_intensity",
    "ts_path_efficiency", "ts_roughness", "ts_trend_break_score",
    "ts_weighted_time_centroid", "ts_endpoint_deviation", "ts_mass_concentration",
    "ts_tail_imbalance", "ts_expected_shortfall_asymmetry", "ts_wasserstein_shift",
    "ts_ks_shift", "ts_location_shift", "ts_scale_shift",
    "ts_vol_of_vol", "ts_vol_acceleration", "ts_vol_term_structure",
    "ts_semivariance_balance", "ts_realized_quarticity", "ts_vol_clustering",
    "ts_leverage_effect", "ts_jump_bipower_proxy",
    "cs_neighbor_gap", "cs_local_density", "cs_isolation", "cs_local_curvature",
    "group_ex_self_std", "group_ex_self_mad", "group_ex_self_quantile",
    "relation_weighted_std_ex_self",
    "event_frequency", "event_cluster_count", "event_cluster_mean_size",
    "report_rolling_mean", "report_yoy_lag",
})
DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | _DAILY_ALPHA_LANGUAGE_2026_08)
# 2026-08 stateful rule / episode / rotation pack: latch/hold/slew/deadband rule
# language, refractory + crossing events, recursive CUSUM, episode reduce +
# directional-change intrinsic time, state survival, cross-sectional rotation,
# drawdown-path recovery.  Same contract as the final pack: each module keeps
# its names in EXTENDED_ONLY_CANONICALS at import (partition check) and this
# frozenset migrates them to the daily surface.
_DAILY_STATEFUL_PACK_2026_08 = frozenset({
    "state_latch", "state_hold", "state_slew_limit", "state_deadband",
    "event_refractory", "cross_event",
    "ts_cusum_pressure", "ts_rank_if", "state_ewm_if", "ts_lag_of_peak_corr",
    "state_since_reduce", "directional_change_state", "directional_change_extent",
    "state_since_trend_tstat",
    "ts_state_age_percentile", "ts_state_exit_hazard", "ts_state_residual_life",
    "cs_rank_churn", "cs_tail_retention",
    "ts_recovery_fraction", "ts_current_drawdown_area",
})
DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | _DAILY_STATEFUL_PACK_2026_08)
# 2026-08 turnover-survival / weighted-risk / behavioural / order-flow families.
# Each module keeps its names in EXTENDED_ONLY_CANONICALS at import (partition
# contract) and this frozenset migrates them to the daily surface.
_DAILY_CHIP_FLOW_PACK_2026_08 = frozenset({
    "ts_turnover_reference_price", "ts_turnover_cost_dispersion",
    "ts_turnover_profit_share", "ts_turnover_holding_age",
    "ts_turnover_near_cost_mass", "ts_turnover_cost_quantile_distance",
    "ts_stratified_mean_spread", "ts_weighted_semivariance",
    "ts_weighted_expected_shortfall", "ts_weighted_drawdown_area",
    "ts_cpt_value",
    "intraday_bvc_imbalance", "intraday_impact_beta",
    "intraday_impact_asymmetry", "intraday_return_wasserstein_shift",
    # NOTE: micro_bvc_vpin stays research-only (P2) — see flow_impact.py.
})
DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | _DAILY_CHIP_FLOW_PACK_2026_08)
# 2026-08 V2/V3 state-dynamics / event-response / spectral-crowding / minute
# volume-clock families.  Same contract as the final pack: each module keeps
# its names in EXTENDED_ONLY_CANONICALS at import (partition check) and this
# frozenset migrates the reviewed P1 operators to the daily surface.
_DAILY_DYNAMICS_PACK_2026_08 = frozenset({
    # state geometry / ordinal time asymmetry
    "ts_ordinal_irreversibility", "ts_state_density",
    # local Markov dynamics
    "ts_markov_persistence", "ts_markov_state_entropy",
    "ts_markov_transition_surprisal", "ts_kramers_moyal_local_stability",
    # first-passage
    "ts_first_passage_bias",
    # historical event response
    "event_historical_response_mean", "event_historical_response_sign_balance",
    # multivariate distribution break
    "ts_joint_energy_shift", "ts_energy_break_score",
    # cross-sectional spectral crowding (P1-G rename: feature-matrix SVD)
    "group_feature_mode_share", "group_feature_effective_rank",
    "group_feature_mode_localization",
    # session recovery + volume-clock path geometry (minute → daily)
    "session_event_recovery_score",
    "intraday_volume_clock_path_efficiency", "intraday_volume_clock_roughness",
    # dynamic KNN peers
    "cs_knn_peer_mean_ex_self", "cs_knn_neighbor_retention",
    # report timing / extreme tail
    "report_filing_delay_surprise", "ts_hill_tail_index",
})
DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | _DAILY_DYNAMICS_PACK_2026_08)
# 2026-08 vertical deepening pack: Markov committor/MFPT/spectral-gap/stationary,
# KM equilibrium/diffusion-gradient/quasipotential, first-passage probability &
# conditional time, event-response curve shape, extreme-value (extremal index /
# mean-excess slope / GPD-PWM), transfer-entropy peak, quantile-transport,
# MMD, chord geometry, chip-cost shape, group spectral gap/second mode, KNN
# Dirichlet energy, intraday RV signature, report change breadth/coherence and
# RQA.  All are strictly trailing / report-causal; see operator_policy for the
# pit-safe contract.
_DAILY_DEEPENING_PACK_2026_08 = frozenset({
    # Markov state deep-dive
    "ts_markov_committor", "ts_markov_mean_first_passage_time",
    "ts_markov_spectral_gap", "ts_markov_stationary_surprisal",
    # Kramers-Moyal deep-dive
    "ts_km_equilibrium_distance", "ts_km_diffusion_gradient",
    "ts_km_quasipotential_depth",
    # first-passage decomposition
    "ts_first_passage_hit_probability", "ts_first_passage_conditional_time",
    # event-response curve shape
    "event_response_peak_lag", "event_response_decay_rate",
    "event_response_dispersion", "event_response_reversal_strength",
    # extreme-value tail shape
    "ts_extremal_index", "ts_mean_excess_slope", "ts_gpd_shape_pwm",
    # transfer-entropy peak (fused)
    "ts_transfer_entropy_peak_strength", "ts_transfer_entropy_peak_lag",
    # distribution transport / MMD
    "ts_quantile_transport_slope", "ts_quantile_transport_curvature",
    "ts_mmd_rbf_shift",
    # chord / bow geometry
    "ts_chord_excursion_area", "ts_max_chord_excursion",
    # chip-cost shape
    "ts_turnover_cost_entropy", "ts_turnover_cost_mode_distance",
    "ts_turnover_cost_skew", "ts_turnover_age_dispersion",
    # group spectral deep-dive (P1-G rename)
    "group_feature_spectral_gap", "group_feature_second_mode_localization",
    # style-graph smoothness
    "cs_knn_graph_dirichlet_energy",
    # intraday volatility signature
    "intraday_rv_signature_slope",
    # report change breadth / coherence
    "report_change_breadth", "report_change_coherence",
    # recurrence quantification analysis
    "ts_recurrence_rate", "ts_recurrence_diagonal_entropy",
    "ts_recurrence_trapping_time", "ts_recurrence_divergence",
})
DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | _DAILY_DEEPENING_PACK_2026_08)
# 2026-08 geometry/math expansion: K-line interval geometry, directional-change
# structural levels, multiscale trend term structure, envelope / crossing
# quality, confirmed-extrema divergence, threshold cycles, dynamic state-episode
# excursions, jump-robust intraday variation, intraday volatility shape,
# volatility roughness, binned response curves, nonlinear dependence, 2D joint
# trajectory geometry, point-process interval stats, string/ordinal complexity,
# spectral shape, Hankel/SSA structure, multifractal spectrum, serial-dependence
# memory, L-moments / Hartigan dip, intrinsic dimension, persistence entropy,
# cs/group locality, session shape.  Same contract as the deepening pack: each
# module keeps its names in EXTENDED_ONLY_CANONICALS at import (partition check)
# and this frozenset migrates them to the daily surface.
_DAILY_GEOMETRY_MATH_2026_08 = frozenset({
    # interval geometry
    "ts_interval_union_coverage", "ts_interval_occupancy_entropy",
    "ts_interval_occupancy_mode_distance", "ts_interval_nesting_depth",
    "ts_interval_exploration_efficiency", "ts_interval_overlap_connected_component_ratio",
    # structural levels
    "ts_structural_level_density", "ts_nearest_structural_level_distance",
    "ts_structural_level_strength",
    # candle state space
    "ts_vector_state_mahalanobis", "ts_vector_state_local_density",
    "ts_matrix_profile_novelty", "ts_matrix_profile_motif_age",
    # multiscale trend
    "ts_multiscale_trend_consensus", "ts_multiscale_trend_dispersion",
    "ts_multiscale_trend_curvature",
    # envelope / crossing
    "ts_envelope_compression", "ts_envelope_pressure", "ts_envelope_boundary_dwell",
    "ts_crossing_speed", "ts_crossing_acceleration",
    # extrema divergence / threshold cycles
    "ts_extrema_divergence_strength", "ts_extrema_confirmation_rate",
    "ts_threshold_cycle_period", "ts_threshold_cycle_asymmetry",
    # state-episode excursions
    "state_episode_mfe", "state_episode_mae", "state_episode_efficiency",
    "state_episode_retrace_ratio", "state_episode_excursion_balance",
    # jump-robust intraday variation
    "intraday_medrv", "intraday_minrv", "intraday_jump_test_stat",
    # intraday volatility shape
    "intraday_volatility_time_centroid", "intraday_volatility_concentration",
    "intraday_volatility_entropy", "intraday_realized_semivariance_balance",
    "intraday_rv_signature_curvature",
    # volatility roughness
    "ts_vol_pvariation_roughness", "ts_vol_scaling_break",
    # binned response curves
    "ts_binned_response_monotonicity", "ts_binned_response_curvature",
    "ts_response_slope_asymmetry",
    # nonlinear dependence
    "ts_chatterjee_xi", "ts_hsic", "ts_conditional_mutual_information",
    "ts_partial_distance_correlation",
    # 2D joint trajectory geometry
    "ts_vector_path_efficiency", "ts_vector_turning_coherence",
    "ts_vector_path_curvature", "ts_vector_self_intersection_rate",
    # point-process interval stats
    "event_interval_memory", "event_local_variation", "event_fano_factor",
    # string / ordinal complexity
    "ts_lempel_ziv_complexity", "ts_forbidden_ordinal_pattern_ratio",
    # spectral shape
    "ts_spectral_centroid", "ts_spectral_flatness",
    "ts_spectral_peak_concentration", "ts_spectral_quality_factor",
    # Hankel / SSA
    "ts_hankel_effective_rank", "ts_hankel_singular_gap",
    "ts_ssa_reconstruction_residual",
    # multifractal
    "ts_generalized_hurst_exponent", "ts_multifractal_width",
    "ts_multifractal_curvature",
    # serial-dependence memory
    "ts_autocorrelation_time", "ts_fractional_difference",
    # L-moments / Hartigan dip / intrinsic dimension / persistence entropy
    "ts_l_skewness", "ts_l_kurtosis", "ts_hartigan_dip",
    "ts_delay_intrinsic_dimension",
    "ts_persistence_entropy_h0", "ts_persistence_entropy_h1",
    # cs / group locality
    "cs_knn_local_moran", "cs_isotonic_residual",
    "group_current_members_tail_coexceedance", "group_corr_mst_length",
    # intraday session shape
    "intraday_session_shape_novelty", "intraday_profile_pca_residual",
})
DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | _DAILY_GEOMETRY_MATH_2026_08)
# 2026-08-08 Gemini-recommended primitives: group Top-K routing, arg-extreme
# gathering, weighted percentile rank, group-vs-market JS divergence, event
# level-survival cohort, weighted standardized moment, conditional covariance,
# robust multi-regressor residual, activity-clock lag/age + max-drawdown
# activity cost, spectral entropy / dominant cycle period, relation PageRank
# centrality, intraday activity-duration curvature.  Each module keeps its names
# in EXTENDED_ONLY_CANONICALS at import (partition contract) and this frozenset
# migrates them to the daily surface.  Research-surface transforms
# (ts_wavelet_lowpass_reconstruct / ts_signature_mahalanobis_anomaly /
# ts_betti_crocker_bifurcation_score) stay RESEARCH_ONLY_CANONICALS.
_DAILY_GEMINI_PACK_2026_08 = frozenset({
    # gathering / distribution
    "group_topk_mean", "ts_value_at_argextreme",
    "cs_weighted_percentile_rank", "group_distribution_js_divergence",
    "event_level_survival_share",
    # weighted moment / conditional / robust resid
    "ts_weighted_standardized_moment", "ts_cov_if", "cs_multi_robust_resid",
    # activity clock
    "ts_activity_clock_lagged_value", "ts_activity_clock_age",
    "ts_max_drawdown_activity_cost",
    # spectral shape
    "ts_spectral_entropy", "ts_dominant_cycle_period",
    # NOTE: relation_pagerank_centrality was demoted to research-only (P0-008):
    # its group-panel adjacency is a fake complete graph (edge flow proportional
    # to node signal), so it is not a true network PageRank.  It stays out of the
    # daily/extended mining surface until a real relation graph layer exists.
    # intraday activity-duration curvature
    "intraday_activity_duration_curvature",
})
DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | _DAILY_GEMINI_PACK_2026_08)
# 2026-08-08 Gemini V2 round: HVG degree entropy / forward-backward asymmetry,
# RQA determinism / laminarity, GLR mean & variance shift, EDGE / Abdi-Ranaldo
# spreads, Qn scale, composition (CoDa) family, global-state cross-sectional
# Hartigan dip, cross-spectral coherence / phase, event Allan factor.  Each
# module unions its names into EXTENDED_ONLY_CANONICALS at import (partition
# contract); this frozenset migrates the daily-grade subset to the daily
# surface.  Batch-2 extended ops and batch-3 research ops stay in their own
# surfaces (extended / research).
_DAILY_GEMINI_V2_PACK_2026_08 = frozenset({
    # HVG (P0)
    "ts_hvg_degree_entropy", "ts_hvg_forward_backward_asymmetry",
    # RQA line structure (P0, STATE)
    "ts_recurrence_determinism", "ts_recurrence_laminarity",
    # change-point scores (P0, CONDITION)
    "ts_glr_mean_shift_score", "ts_glr_variance_shift_score",
    # microstructure spreads / robust scale (P0, ALPHA)
    "ts_edge_effective_spread", "ts_abdi_ranaldo_spread", "ts_qn_scale",
    # compositional data (P0, typed)
    "composition_clr_component", "composition_aitchison_distance",
    "composition_ilr_balance",
    # global-state cross-sectional shape (P0, CONDITION)
    "cs_hartigan_dip",
    # cross-spectral (P0, ALPHA)
    "ts_cross_spectral_coherence", "ts_cross_spectral_phase",
    # batch-2 daily-grade (state / fundamental)
    "event_allan_factor", "composition_entropy", "composition_js_divergence",
})
DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | _DAILY_GEMINI_V2_PACK_2026_08)
# =====================================================================
# #311: daily-factor migration reviewed manifest
# ---------------------------------------------------------------------
# ``DAILY_FACTOR_MIGRATED`` is a large manual frozenset accumulated across many
# 2026-08 packs (each ``_DAILY_*_PACK`` unioned itself in at import).  Going
# forward the reviewed migration surface is owned by ``REVIEWED_MIGRATION_MANIFEST``:
# a version-bound registry mapping each canonical to its review id, semantic hash
# and approved authoring tier.  The manual frozenset is retained only as the
# backward-compatible seed (tests still read it) and a derived view of the
# manifest.  New migrations MUST be added via :func:`register_daily_migration`
# (with a review_id) rather than by mutating the frozenset.
# =====================================================================
REVIEWED_MIGRATION_MANIFEST: dict[str, dict[str, str]] = {
    canonical: {
        "review_id": "seed-2026-08-manual",
        "semantic_hash": "",
        "approved_authoring_tier": "daily",
    }
    for canonical in sorted(DAILY_FACTOR_MIGRATED)
}


def register_daily_migration(
    canonical: str,
    *,
    review_id: str,
    semantic_hash: str = "",
    approved_authoring_tier: str = "daily",
) -> None:
    """Record a reviewed daily-surface migration in the manifest.

    This is the forward path for adding an operator to the daily surface: the
    entry carries a ``review_id`` (plus optional ``semantic_hash`` and
    ``approved_authoring_tier``) so the migration is version-bound and auditable.
    The legacy ``DAILY_FACTOR_MIGRATED`` frozenset is kept in sync so existing
    consumers that import it directly continue to see the operator.
    """
    REVIEWED_MIGRATION_MANIFEST[canonical] = {
        "review_id": review_id,
        "semantic_hash": semantic_hash,
        "approved_authoring_tier": approved_authoring_tier,
    }
    global DAILY_FACTOR_MIGRATED
    DAILY_FACTOR_MIGRATED = frozenset(set(DAILY_FACTOR_MIGRATED) | {canonical})


def daily_factor_migrated() -> frozenset[str]:
    """Live read of the reviewed daily-migration surface (manifest keys)."""
    return frozenset(REVIEWED_MIGRATION_MANIFEST)


RESEARCH_ONLY_CANONICALS=frozenset({"holder_concentration_change","holder_count_change_rate"});LEGACY_ONLY_CANONICALS=frozenset({"cube"});INTERNAL_ONLY_CANONICALS=frozenset({"constant","identity","protected_div"})
HIDDEN_DAILY_NAMES=frozenset({"cube","cumulative_max","cumulative_mean","cumulative_min","fmax","fmin","inv","reciprocal","sqr"})
# =====================================================================
# #310: three ORTHOGONAL dimensions.
# ---------------------------------------------------------------------
#   * AuthoringTier          — which authoring surface an operator is authored
#                              on (surface membership; ``classify_canonical``).
#   * ProductionCertification— whether the operator is evidence-certified for
#                              production (six-gate composite; decoupled from
#                              which surface list the canonical appears in).
#   * BackendCapability      — what a specific runtime backend is capable of
#                              for this canonical (backend-granular flags).
# An operator can be authored EXTENDED while its certification is PENDING and a
# given backend is merely *available*; the three never conflate.
# =====================================================================
class AuthoringTier(enum.Enum):
    DAILY = "daily"
    EXTENDED = "extended"
    RESEARCH = "research"
    INTERNAL = "internal"
    # Backward-compatible non-production authoring surfaces.
    UNSAFE = "unsafe"
    LEGACY = "legacy"
    UNCLASSIFIED = "unclassified"


class ProductionCertification(enum.Enum):
    CERTIFIED = "certified"
    PENDING = "pending"
    DENIED = "denied"


@dataclass(frozen=True)
class BackendCapability:
    backend: str
    available: bool
    production_certified: bool
    reference_backend: bool
    certification_tier: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "available": self.available,
            "production_certified": self.production_certified,
            "reference_backend": self.reference_backend,
            "certification_tier": self.certification_tier,
        }


def authoring_tier(canonical: str) -> AuthoringTier:
    """Return the authoring tier enum for ``canonical`` (surface membership)."""
    return AuthoringTier(classify_canonical(canonical))


def production_certification(canonical: str) -> ProductionCertification:
    """Return the operator's production certification (evidence/policy driven).

    #310: certification is derived from the operator's ACTUAL certification
    fields — the six-gate ``production_certified`` composite written by
    ``reconcile_operator_certification`` (the single certification authority) —
    never from which surface list the canonical happens to appear in.
    """
    from cleaned_operators.registry import OperatorRegistry

    resolved = OperatorRegistry._aliases.get(canonical, canonical)
    catalog = OperatorRegistry._catalog.get(resolved, {})
    # Six-gate composite is the only production-certification authority.
    if catalog.get("production_certified") is True:
        return ProductionCertification.CERTIFIED
    # Explicitly blocked / never intended for production.
    if catalog.get("status") in (
        "denied", "rejected", "deprecated", "stub", "doc_only",
    ):
        return ProductionCertification.DENIED
    try:
        from cleaned_operators.operator_spec import (
            PERMANENTLY_FORBIDDEN_CANONICALS,
            is_production_denied,
        )
        if resolved in PERMANENTLY_FORBIDDEN_CANONICALS or is_production_denied(resolved):
            return ProductionCertification.DENIED
    except Exception:
        pass
    if not catalog:
        # Unknown / unregistered operator fails closed.
        return ProductionCertification.DENIED
    # Not a production target -> no certification is ever granted.
    try:
        from cleaned_operators.production_hardening import factor_production_targets
        if resolved not in factor_production_targets():
            return ProductionCertification.DENIED
    except Exception:
        pass
    # Registered, reviewed target, but evidence not yet bound -> awaiting
    # certification (PENDING), which is distinct from a hard DENIED.
    return ProductionCertification.PENDING


def backend_capability(canonical: str, backend: str) -> BackendCapability:
    """Return per-backend capability flags for ``canonical``."""
    from cleaned_operators.registry import OperatorRegistry

    resolved = OperatorRegistry._aliases.get(canonical, canonical)
    backends = OperatorRegistry.backends_for(resolved)
    meta = dict(
        (OperatorRegistry._catalog.get(resolved, {}).get("backend_meta") or {}).get(backend) or {}
    )
    return BackendCapability(
        backend=backend,
        available=backend in backends,
        production_certified=bool(meta.get("production_certified")),
        reference_backend=bool(meta.get("reference_backend")),
        certification_tier=str(meta.get("certification_tier") or ""),
    )


def classify_canonical(canonical:str)->str:
    if canonical in INTERNAL_ONLY_CANONICALS:return "internal"
    if canonical in DAILY_CANONICALS or canonical in daily_factor_migrated():return "daily"
    if canonical in EXTENDED_ONLY_CANONICALS:return "extended"
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
