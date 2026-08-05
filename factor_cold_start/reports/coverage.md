# Factor cold-start coverage

- Total factors: **4628**
- Catalogs: **4**
- Existing GTJA/Week2 formulas excluded structurally: **453**

## Catalog summary

| Catalog | Factors | Families | Fields | Eligible operators | Coverage |
|---|---:|---:|---:|---:|---:|
| `ashare_daily` | 819 | 20 | 13 | 79/79 | 100.0% |
| `ashare_extended` | 1363 | 13 | 11 | 91/392 | 23.21% |
| `us_daily` | 1007 | 25 | 26 | 79/79 | 100.0% |
| `us_extended` | 1439 | 13 | 13 | 90/391 | 23.02% |

## `ashare_daily`

### Families

| Family | Count |
|---|---:|
| `price_volume_interaction` | 144 |
| `liquidity_activity` | 116 |
| `return_momentum` | 95 |
| `volatility` | 84 |
| `trend_location` | 60 |
| `liquidity_impact` | 56 |
| `nonlinear_robust` | 41 |
| `candle_vwap` | 40 |
| `adjusted_price` | 32 |
| `reversal` | 28 |
| `serial_dependence` | 24 |
| `trend_quality` | 24 |
| `conditional_regime` | 22 |
| `cross_sectional_state` | 18 |
| `turnover` | 18 |
| `group_relative` | 8 |
| `adjustment_event` | 3 |
| `data_quality` | 3 |
| `risk_scaling` | 2 |
| `discrete_regime` | 1 |

### Availability tiers

| Tier | Count |
|---|---:|
| `core` | 793 |
| `enriched` | 26 |

### Intentionally excluded operators

- `period_average`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_cagr`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_change`: fundamental fiscal-period operator; excluded from price-volume cold start
- `quarter_from_cumulative`: fundamental statement transformation
- `ttm_from_cumulative`: fundamental statement transformation
- `ttm_from_quarterly`: fundamental statement transformation
- `yoy_by_period`: fundamental statement transformation

## `ashare_extended`

### Families

| Family | Count |
|---|---:|
| `distribution_tail` | 448 |
| `rolling_regression` | 305 |
| `decay_trend` | 192 |
| `conditional_history` | 174 |
| `technical` | 81 |
| `nonlinear_experimental` | 66 |
| `price_deviation` | 32 |
| `compounded_return` | 24 |
| `advanced_cross_sectional` | 14 |
| `data_quality` | 10 |
| `liquidity_activity` | 8 |
| `cross_sectional_state` | 5 |
| `turnover` | 4 |

### Availability tiers

| Tier | Count |
|---|---:|
| `core` | 1345 |
| `enriched` | 18 |

### Missing eligible operators

`ADL`, `CMF`, `CMO`, `ChaikinOscillator`, `DEMA`, `DMI_minus`, `DMI_plus`, `DX`, `EaseOfMovement`, `ForceIndex`, `KAMA`, `KeltnerLower`, `KeltnerMid`, `KeltnerPosition`, `KeltnerUpper`, `MFI`, `NATR`, `PPO`, `PPO_hist`, `PPO_signal`, `PSAR`, `PVO`, `PVO_hist`, `PVO_signal`, `Supertrend`, `SupertrendDirection`, `TEMA`, `TSI`, `TSI_signal`, `UltimateOscillator`, `VortexMinus`, `VortexPlus`, `abnormal_turnover`, `abnormal_volume`, `abs_return_volume_corr`, `adv`, `amihud_illiquidity`, `average_turnover`, `benchmark_excess_return`, `benchmark_relative_price`, `bollinger_pct_b`, `bollinger_width`, `book_to_price`, `bounded_nvi`, `bounded_pvi`, `candle_abs_body`, `candle_body`, `candle_body_percentile`, `candle_body_position`, `candle_body_ratio`, `candle_body_zscore`, `candle_close_location`, `candle_close_strength`, `candle_direction`, `candle_gap`, `candle_gap_atr`, `candle_gap_pct`, `candle_inside_ratio`, `candle_lower_shadow`, `candle_lower_shadow_ratio`, `candle_lower_shadow_zscore`, `candle_overlap_ratio`, `candle_range`, `candle_range_atr`, `candle_range_percentile`, `candle_range_zscore`, `candle_rejection_lower`, `candle_rejection_upper`, `candle_upper_shadow`, `candle_upper_shadow_ratio`, `candle_upper_shadow_zscore`, `candlestick_pattern`, `cdl_dark_cloud_cover`, `cdl_doji`, `cdl_dragonfly_doji`, `cdl_engulfing`, `cdl_evening_star`, `cdl_gravestone_doji`, `cdl_hammer`, `cdl_hanging_man`, `cdl_harami`, `cdl_harami_cross`, `cdl_inside_bar`, `cdl_inverted_hammer`, `cdl_marubozu`, `cdl_morning_star`, `cdl_outside_bar`, `cdl_piercing`, `cdl_shooting_star`, `cdl_spinning_top`, `cdl_three_black_crows`, `cdl_three_white_soldiers`, `cdl_tweezer_bottom`, `cdl_tweezer_top`, `choppiness_index`, `corwin_schultz_spread`, `coskewness_to_market`, `digital_count`, `dollar_volume`, `dollar_volume_zscore`, `donchian_lower`, `donchian_mid`, `donchian_position`, `donchian_upper`, `down_volume_ratio`, `earnings_yield`, `efficiency_ratio`, `expanding_rank`, `fin_accrual_ratio`, `fin_average_balance`, `fin_cagr`, `fin_cash_conversion`, `fin_cash_earnings_gap`, `fin_common_size`, `fin_cv`, `fin_days_since_update`, `fin_diff`, `fin_divergence`, `fin_growth`, `fin_growth_acceleration`, `fin_growth_change`, `fin_growth_persistence`, `fin_growth_stability`, `fin_growth_volatility`, `fin_lag`, `fin_log_change`, `fin_mad`, `fin_monotonicity`, `fin_negative_streak`, `fin_pct_change`, `fin_percentile_history`, `fin_positive_streak`, `fin_qoq`, `fin_range`, `fin_ratio`, `fin_restated_flag`, `fin_revision_count`, `fin_revision_delta`, `fin_revision_direction`, `fin_revision_magnitude`, `fin_revision_pct`, `fin_sign_change_count`, `fin_stability`, `fin_staleness`, `fin_std`, `fin_trend_acceleration`, `fin_trend_r2`, `fin_trend_slope`, `fin_trend_tstat`, `fin_ttm`, `fin_turnover`, `fin_working_capital_change`, `fin_yoy`, `fin_zscore_history`, `float_share_ratio`, `free_float_share_ratio`, `garman_klass_vol`, `group_decay_linear`, `high_low_spread_proxy`, `holder_concentration`, `holder_concentration_change`, `holder_count_change_rate`, `hump_decay`, `ichimoku_cloud_position`, `ichimoku_cloud_width`, `ichimoku_kijun`, `ichimoku_senkou_a`, `ichimoku_senkou_b`, `ichimoku_tenkan`, `idio_skew`, `idio_vol`, `industry_size_neutralize`, `intraday_volatility`, `intraday_vwap_deviation`, `limit_down_state`, `limit_up_state`, `lqtp_historical_cvar`, `overnight_volatility`, `parkinson_vol`, `pattern_ascending_triangle`, `pattern_bear_flag`, `pattern_broadening`, `pattern_bull_flag`, `pattern_descending_triangle`, `pattern_double_bottom`, `pattern_double_top`, `pattern_falling_channel`, `pattern_falling_wedge`, `pattern_head_shoulders`, `pattern_inverse_head_shoulders`, `pattern_rectangle`, `pattern_rising_channel`, `pattern_rising_wedge`, `pattern_sym_triangle`, `price_impact`, `price_turnover_divergence`, `price_volume_divergence`, `range_volatility`, `rank_corr`, `relative_volume`, `residual_momentum_capm`, `return_per_turnover`, `return_turnover_beta`, `return_volume_beta`, `return_volume_corr`, `rogers_satchell_vol`, `roll_spread_proxy`, `rolling_beta_to_market`, `rolling_obv`, `rolling_pvt`, `rolling_vwap`, `signed_dollar_volume`, `signed_volume`, `signed_volume_imbalance`, `size_neutralize`, `tail_beta`, `tradable_state`, `trade_when`, `true_turnover_rate`, `ts_breakdown_low`, `ts_breakout_high`, `ts_channel_position`, `ts_channel_width`, `ts_channel_width_atr`, `ts_channel_width_pct`, `ts_channel_width_slope`, `ts_confirmed_pivot_high`, `ts_confirmed_pivot_low`, `ts_consolidation_slope`, `ts_consolidation_volume_decay`, `ts_consolidation_width`, `ts_days_since_high`, `ts_days_since_low`, `ts_distance_to_high`, `ts_distance_to_low`, `ts_distance_to_resistance`, `ts_distance_to_support`, `ts_impulse_return`, `ts_impulse_strength`, `ts_impulse_volume`, `ts_last_pivot_high`, `ts_last_pivot_low`, `ts_line_convergence`, `ts_line_parallelism`, `ts_max_buildup`, `ts_moment`, `ts_new_high`, `ts_new_low`, `ts_nth_pivot_high`, `ts_nth_pivot_high_age`, `ts_nth_pivot_low`, `ts_nth_pivot_low_age`, `ts_pattern_symmetry`, `ts_pivot_high_age`, `ts_pivot_high_count`, `ts_pivot_high_spacing`, `ts_pivot_low_age`, `ts_pivot_low_count`, `ts_pivot_low_spacing`, `ts_poly2_coeff`, `ts_poly2_resid`, `ts_prev_high`, `ts_prev_low`, `ts_range_expansion`, `ts_resistance_break`, `ts_resistance_fit_r2`, `ts_resistance_level`, `ts_resistance_slope`, `ts_sma_cn`, `ts_sum_decay`, `ts_support_break`, `ts_support_fit_r2`, `ts_support_level`, `ts_support_slope`, `ts_swing_amplitude`, `ts_swing_amplitude_atr`, `ts_swing_amplitude_pct`, `ts_swing_duration`, `ts_swing_velocity`, `turnover_acceleration`, `turnover_adjusted_volatility`, `turnover_autocorr`, `turnover_momentum`, `turnover_shock`, `turnover_volatility`, `turnover_zscore`, `ulcer_index`, `up_down_volume_ratio`, `up_volume_ratio`, `volume_acceleration`, `volume_autocorr`, `volume_momentum`, `volume_shock`, `volume_to_range`, `volume_volatility`, `volume_weighted_momentum`, `volume_weighted_return`, `volume_zscore`, `vwap_deviation`, `yang_zhang_vol`, `zero_return_ratio`

### Intentionally excluded operators

- `fundamental_staleness`: fundamental availability diagnostic
- `period_lag`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_stability`: fundamental fiscal-period operator; excluded from price-volume cold start
- `revision_delta`: fundamental revision diagnostic

## `us_daily`

### Families

| Family | Count |
|---|---:|
| `price_volume_interaction` | 144 |
| `liquidity_activity` | 116 |
| `return_momentum` | 95 |
| `volatility` | 84 |
| `us_derived` | 70 |
| `trend_location` | 60 |
| `liquidity_impact` | 56 |
| `overnight_intraday` | 56 |
| `nonlinear_robust` | 41 |
| `candle_vwap` | 40 |
| `microstructure` | 33 |
| `adjusted_price` | 32 |
| `reversal` | 28 |
| `short_flow` | 28 |
| `serial_dependence` | 24 |
| `trend_quality` | 24 |
| `conditional_regime` | 22 |
| `cross_sectional_state` | 18 |
| `turnover` | 18 |
| `group_relative` | 8 |
| `adjustment_event` | 3 |
| `data_quality` | 3 |
| `risk_scaling` | 2 |
| `data_consistency` | 1 |
| `discrete_regime` | 1 |

### Availability tiers

| Tier | Count |
|---|---:|
| `core` | 793 |
| `derived` | 127 |
| `enriched` | 54 |
| `microstructure` | 33 |

### Intentionally excluded operators

- `period_average`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_cagr`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_change`: fundamental fiscal-period operator; excluded from price-volume cold start
- `quarter_from_cumulative`: fundamental statement transformation
- `ttm_from_cumulative`: fundamental statement transformation
- `ttm_from_quarterly`: fundamental statement transformation
- `yoy_by_period`: fundamental statement transformation

## `us_extended`

### Families

| Family | Count |
|---|---:|
| `distribution_tail` | 448 |
| `rolling_regression` | 305 |
| `decay_trend` | 192 |
| `conditional_history` | 174 |
| `technical` | 81 |
| `overnight_intraday` | 80 |
| `nonlinear_experimental` | 66 |
| `price_deviation` | 32 |
| `compounded_return` | 24 |
| `advanced_cross_sectional` | 14 |
| `data_quality` | 10 |
| `liquidity_activity` | 8 |
| `cross_sectional_state` | 5 |

### Availability tiers

| Tier | Count |
|---|---:|
| `core` | 1345 |
| `derived` | 80 |
| `enriched` | 14 |

### Missing eligible operators

`ADL`, `CMF`, `CMO`, `ChaikinOscillator`, `DEMA`, `DMI_minus`, `DMI_plus`, `DX`, `EaseOfMovement`, `ForceIndex`, `KAMA`, `KeltnerLower`, `KeltnerMid`, `KeltnerPosition`, `KeltnerUpper`, `MFI`, `NATR`, `PPO`, `PPO_hist`, `PPO_signal`, `PSAR`, `PVO`, `PVO_hist`, `PVO_signal`, `Supertrend`, `SupertrendDirection`, `TEMA`, `TSI`, `TSI_signal`, `UltimateOscillator`, `VortexMinus`, `VortexPlus`, `abnormal_turnover`, `abnormal_volume`, `abs_return_volume_corr`, `adv`, `amihud_illiquidity`, `average_turnover`, `benchmark_excess_return`, `benchmark_relative_price`, `bollinger_pct_b`, `bollinger_width`, `book_to_price`, `bounded_nvi`, `bounded_pvi`, `candle_abs_body`, `candle_body`, `candle_body_percentile`, `candle_body_position`, `candle_body_ratio`, `candle_body_zscore`, `candle_close_location`, `candle_close_strength`, `candle_direction`, `candle_gap`, `candle_gap_atr`, `candle_gap_pct`, `candle_inside_ratio`, `candle_lower_shadow`, `candle_lower_shadow_ratio`, `candle_lower_shadow_zscore`, `candle_overlap_ratio`, `candle_range`, `candle_range_atr`, `candle_range_percentile`, `candle_range_zscore`, `candle_rejection_lower`, `candle_rejection_upper`, `candle_upper_shadow`, `candle_upper_shadow_ratio`, `candle_upper_shadow_zscore`, `candlestick_pattern`, `cdl_dark_cloud_cover`, `cdl_doji`, `cdl_dragonfly_doji`, `cdl_engulfing`, `cdl_evening_star`, `cdl_gravestone_doji`, `cdl_hammer`, `cdl_hanging_man`, `cdl_harami`, `cdl_harami_cross`, `cdl_inside_bar`, `cdl_inverted_hammer`, `cdl_marubozu`, `cdl_morning_star`, `cdl_outside_bar`, `cdl_piercing`, `cdl_shooting_star`, `cdl_spinning_top`, `cdl_three_black_crows`, `cdl_three_white_soldiers`, `cdl_tweezer_bottom`, `cdl_tweezer_top`, `choppiness_index`, `corwin_schultz_spread`, `coskewness_to_market`, `digital_count`, `dollar_volume`, `dollar_volume_zscore`, `donchian_lower`, `donchian_mid`, `donchian_position`, `donchian_upper`, `down_volume_ratio`, `earnings_yield`, `efficiency_ratio`, `expanding_rank`, `fin_accrual_ratio`, `fin_average_balance`, `fin_cagr`, `fin_cash_conversion`, `fin_cash_earnings_gap`, `fin_common_size`, `fin_cv`, `fin_days_since_update`, `fin_diff`, `fin_divergence`, `fin_growth`, `fin_growth_acceleration`, `fin_growth_change`, `fin_growth_persistence`, `fin_growth_stability`, `fin_growth_volatility`, `fin_lag`, `fin_log_change`, `fin_mad`, `fin_monotonicity`, `fin_negative_streak`, `fin_pct_change`, `fin_percentile_history`, `fin_positive_streak`, `fin_qoq`, `fin_range`, `fin_ratio`, `fin_restated_flag`, `fin_revision_count`, `fin_revision_delta`, `fin_revision_direction`, `fin_revision_magnitude`, `fin_revision_pct`, `fin_sign_change_count`, `fin_stability`, `fin_staleness`, `fin_std`, `fin_trend_acceleration`, `fin_trend_r2`, `fin_trend_slope`, `fin_trend_tstat`, `fin_ttm`, `fin_turnover`, `fin_working_capital_change`, `fin_yoy`, `fin_zscore_history`, `float_share_ratio`, `free_float_share_ratio`, `garman_klass_vol`, `group_decay_linear`, `high_low_spread_proxy`, `holder_concentration`, `holder_concentration_change`, `holder_count_change_rate`, `hump_decay`, `ichimoku_cloud_position`, `ichimoku_cloud_width`, `ichimoku_kijun`, `ichimoku_senkou_a`, `ichimoku_senkou_b`, `ichimoku_tenkan`, `idio_skew`, `idio_vol`, `industry_size_neutralize`, `intraday_volatility`, `intraday_vwap_deviation`, `limit_down_state`, `limit_up_state`, `lqtp_historical_cvar`, `overnight_volatility`, `parkinson_vol`, `pattern_ascending_triangle`, `pattern_bear_flag`, `pattern_broadening`, `pattern_bull_flag`, `pattern_descending_triangle`, `pattern_double_bottom`, `pattern_double_top`, `pattern_falling_channel`, `pattern_falling_wedge`, `pattern_head_shoulders`, `pattern_inverse_head_shoulders`, `pattern_rectangle`, `pattern_rising_channel`, `pattern_rising_wedge`, `pattern_sym_triangle`, `price_impact`, `price_turnover_divergence`, `price_volume_divergence`, `range_volatility`, `rank_corr`, `relative_volume`, `residual_momentum_capm`, `return_per_turnover`, `return_turnover_beta`, `return_volume_beta`, `return_volume_corr`, `rogers_satchell_vol`, `roll_spread_proxy`, `rolling_beta_to_market`, `rolling_obv`, `rolling_pvt`, `rolling_vwap`, `signed_dollar_volume`, `signed_volume`, `signed_volume_imbalance`, `size_neutralize`, `tail_beta`, `tradable_state`, `trade_when`, `true_turnover_rate`, `ts_breakdown_low`, `ts_breakout_high`, `ts_channel_position`, `ts_channel_width`, `ts_channel_width_atr`, `ts_channel_width_pct`, `ts_channel_width_slope`, `ts_confirmed_pivot_high`, `ts_confirmed_pivot_low`, `ts_consolidation_slope`, `ts_consolidation_volume_decay`, `ts_consolidation_width`, `ts_days_since_high`, `ts_days_since_low`, `ts_distance_to_high`, `ts_distance_to_low`, `ts_distance_to_resistance`, `ts_distance_to_support`, `ts_impulse_return`, `ts_impulse_strength`, `ts_impulse_volume`, `ts_last_pivot_high`, `ts_last_pivot_low`, `ts_line_convergence`, `ts_line_parallelism`, `ts_max_buildup`, `ts_moment`, `ts_new_high`, `ts_new_low`, `ts_nth_pivot_high`, `ts_nth_pivot_high_age`, `ts_nth_pivot_low`, `ts_nth_pivot_low_age`, `ts_pattern_symmetry`, `ts_pivot_high_age`, `ts_pivot_high_count`, `ts_pivot_high_spacing`, `ts_pivot_low_age`, `ts_pivot_low_count`, `ts_pivot_low_spacing`, `ts_poly2_coeff`, `ts_poly2_resid`, `ts_prev_high`, `ts_prev_low`, `ts_range_expansion`, `ts_resistance_break`, `ts_resistance_fit_r2`, `ts_resistance_level`, `ts_resistance_slope`, `ts_sma_cn`, `ts_sum_decay`, `ts_support_break`, `ts_support_fit_r2`, `ts_support_level`, `ts_support_slope`, `ts_swing_amplitude`, `ts_swing_amplitude_atr`, `ts_swing_amplitude_pct`, `ts_swing_duration`, `ts_swing_velocity`, `turnover_acceleration`, `turnover_adjusted_volatility`, `turnover_autocorr`, `turnover_momentum`, `turnover_shock`, `turnover_volatility`, `turnover_zscore`, `ulcer_index`, `up_down_volume_ratio`, `up_volume_ratio`, `volume_acceleration`, `volume_autocorr`, `volume_momentum`, `volume_shock`, `volume_to_range`, `volume_volatility`, `volume_weighted_momentum`, `volume_weighted_return`, `volume_zscore`, `vwap_deviation`, `yang_zhang_vol`, `zero_return_ratio`

### Intentionally excluded operators

- `fundamental_staleness`: fundamental availability diagnostic
- `period_lag`: fundamental fiscal-period operator; excluded from price-volume cold start
- `period_stability`: fundamental fiscal-period operator; excluded from price-volume cold start
- `real_turnover_rate`: requires a verified free-float share-count field; the current US contract does not guarantee one
- `revision_delta`: fundamental revision diagnostic
