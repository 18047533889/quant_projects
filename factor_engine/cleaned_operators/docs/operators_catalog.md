# Operators Catalog（自动生成）

> 从 `load_all()` 去重后的最终 runtime registry 生成。
> 日常因子 DSL 仅使用 `surface=daily`；其他工具见 `research_operators/`。

## 摘要

- canonical 总数：541
- daily：86
- research：2
- unsafe：7
- legacy：1

| canonical | surface | backends | aliases | pit_safe | scope | lookback | min_periods | lag |
|---|---|---|---|---|---|---|---|---|
| ADL | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ADX | extended | pandas_numpy, polars | ts_adx | True | ts | None | 2 | 0 |
| ATR_WILDER | extended | pandas_numpy, polars, sql | ts_atr_wilder | True | ts | None | 1 | 0 |
| CMF | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| CMO | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ChaikinOscillator | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| DEMA | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| DMI_minus | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| DMI_plus | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| DX | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| EaseOfMovement | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ForceIndex | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| KAMA | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| KeltnerLower | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| KeltnerMid | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| KeltnerPosition | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| KeltnerUpper | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| MACD_hist | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| MACD_line | extended | pandas_numpy, polars | MACD | True | ts | None | 1 | 0 |
| MACD_signal | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| MFI | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| NATR | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| PPO | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| PPO_hist | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| PPO_signal | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| PSAR | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| PVO | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| PVO_hist | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| PVO_signal | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| RSI_WILDER | extended | pandas_numpy, polars, sql | ts_rsi_wilder | True | ts | None | 1 | 0 |
| Supertrend | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| SupertrendDirection | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| TEMA | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| TSI | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| TSI_signal | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| UltimateOscillator | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| VortexMinus | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| VortexPlus | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| abnormal_turnover | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| abnormal_volume | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| abs | daily | pandas_numpy, polars, sql | ABS | True | elementwise | None | None | 0 |
| abs_return_volume_corr | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| acos | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| add | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| adv | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| amihud_illiquidity | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| and_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| arg | unsafe | pandas_numpy |  | True | elementwise | None | None | 0 |
| asin | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| atan | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| atan2 | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| average_turnover | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| benchmark_excess_return | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| benchmark_relative_price | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| bollinger_pct_b | extended | pandas_numpy |  | True | elementwise | None | 1 | 0 |
| bollinger_width | extended | pandas_numpy |  | True | elementwise | None | 1 | 0 |
| book_to_price | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| bounded_nvi | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| bounded_pvi | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_abs_body | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| candle_body | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| candle_body_percentile | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_body_position | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_body_ratio | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| candle_body_zscore | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_close_location | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| candle_close_strength | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_direction | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| candle_gap | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_gap_atr | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_gap_pct | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_inside_ratio | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_lower_shadow | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| candle_lower_shadow_ratio | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| candle_lower_shadow_zscore | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_overlap_ratio | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_range | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| candle_range_atr | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_range_percentile | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_range_zscore | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_rejection_lower | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_rejection_upper | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candle_upper_shadow | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| candle_upper_shadow_ratio | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| candle_upper_shadow_zscore | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| candlestick_pattern | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cash_flow_lifecycle_stage | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| cbrt | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| cdl_dark_cloud_cover | extended | pandas_numpy |  | True | ts | None | 1 | 1 |
| cdl_doji | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_dragonfly_doji | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_engulfing | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_evening_star | extended | pandas_numpy |  | True | ts | None | 1 | 2 |
| cdl_gravestone_doji | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_hammer | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_hanging_man | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_harami | extended | pandas_numpy |  | True | ts | None | 1 | 1 |
| cdl_harami_cross | extended | pandas_numpy |  | True | ts | None | 1 | 1 |
| cdl_inside_bar | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_inverted_hammer | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_marubozu | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_morning_star | extended | pandas_numpy |  | True | ts | None | 1 | 2 |
| cdl_outside_bar | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_piercing | extended | pandas_numpy |  | True | ts | None | 1 | 1 |
| cdl_shooting_star | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_spinning_top | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cdl_three_black_crows | extended | pandas_numpy |  | True | ts | None | 1 | 2 |
| cdl_three_white_soldiers | extended | pandas_numpy |  | True | ts | None | 1 | 2 |
| cdl_tweezer_bottom | extended | pandas_numpy |  | True | ts | None | 1 | 1 |
| cdl_tweezer_top | extended | pandas_numpy |  | True | ts | None | 1 | 1 |
| ceil | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| choppiness_index | extended | pandas_numpy |  | True | elementwise | None | 1 | 0 |
| clip | daily | pandas_numpy, polars, sql | CLIP, cap, clamp | True | elementwise | None | None | 0 |
| coalesce | daily | pandas_numpy, polars, sql | COALESCE | True | elementwise | None | None | 0 |
| constant | internal | pandas_numpy |  | True | elementwise | None | None | 0 |
| corwin_schultz_spread | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| cos | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| cosh | unsafe | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| coskewness_to_market | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| cot | unsafe | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| cs_bucket | extended | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_count | daily | pandas_numpy, polars, sql | c_count | True | cs | None | None | 0 |
| cs_demean | daily | pandas_numpy, polars, sql | CS_DEMEAN, c_demean | True | cs | None | None | 0 |
| cs_fill_mean | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_fill_median | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_mad | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_mad_zscore | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_mean | daily | pandas_numpy, polars, sql | c_mean | True | cs | None | None | 0 |
| cs_multi_resid | extended | pandas_numpy, sql |  | True | cs | None | None | 0 |
| cs_neutralize | extended | pandas_numpy |  | True | cs | None | None | 0 |
| cs_pct_rank | daily | pandas_numpy, polars, sql | rank_pct | True | cs | None | None | 0 |
| cs_quantile | extended | pandas_numpy, sql | c_percentile, quantile | True | cs | None | None | 0 |
| cs_rank_gaussian | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_regression | extended | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_resid | extended | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_std | daily | pandas_numpy, polars, sql | c_std | True | cs | None | None | 0 |
| cs_sum | daily | pandas_numpy, polars, sql | c_sum | True | cs | None | None | 0 |
| cs_weighted_demean | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_weighted_mean | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_weighted_zscore | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_wls_resid | extended | pandas_numpy, sql |  | True | cs | None | None | 0 |
| csc | unsafe | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| cube | legacy | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| date_diff_days | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| digital_count | extended | pandas_numpy, polars |  | True | ts | None | 1 | 1 |
| divide | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| dollar_volume | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| dollar_volume_zscore | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| donchian_lower | extended | pandas_numpy |  | True | elementwise | None | 1 | 0 |
| donchian_mid | extended | pandas_numpy |  | True | elementwise | None | 1 | 0 |
| donchian_position | extended | pandas_numpy |  | True | elementwise | None | 1 | 0 |
| donchian_upper | extended | pandas_numpy |  | True | elementwise | None | 1 | 0 |
| down_volume_ratio | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| earnings_yield | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| efficiency_ratio | extended | pandas_numpy |  | True | elementwise | None | 1 | 0 |
| eq | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| exp | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| exp_neg | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| expanding_rank | extended | pandas_numpy, polars | cum_rank | True | ts | None | 1 | 0 |
| ffill_limit | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| fillna_const | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| fin_accrual_ratio | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_actual_expectation_divergence | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_average_balance | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_beat_streak | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_cagr | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_cash_conversion | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_cash_earnings_gap | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_common_size | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_cv | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_days_since_expectation_revision | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_days_since_update | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_diff | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_divergence | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_expectation_dispersion | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_expectation_revision | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_expectation_revision_count | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_expectation_revision_magnitude | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_expectation_revision_pct | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_expectation_revision_speed | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_growth | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_growth_acceleration | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_growth_change | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_growth_persistence | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_growth_stability | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_growth_volatility | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_lag | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_log_change | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_mad | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_miss_streak | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_monotonicity | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_negative_streak | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_pct_change | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_percentile_history | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_positive_streak | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_qoq | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_quarter_from_cumulative | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_range | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_ratio | extended | pandas_numpy, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_restated_flag | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_revision_count | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_revision_delta | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_revision_direction | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_revision_magnitude | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_revision_pct | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_seasonal_percentile | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_seasonal_zscore | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_sign_change_count | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_stability | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_staleness | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_std | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_surprise | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_surprise_event_percentile | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_surprise_event_zscore | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_surprise_zscore | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_trend_acceleration | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_trend_r2 | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_trend_slope | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_trend_tstat | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_ttm | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_ttm_cumulative | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_ttm_quarterly | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_turnover | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_working_capital_change | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_yoy | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fin_zscore_history | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_accrual_quality | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_ar_resid_std | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_asymmetric_elasticity | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_autocorr | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_change_direction_agreement | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_direction_consistency | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_pair_direction_agreement | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_perpetual_inventory | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_regression_resid_std | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_reversal_ratio | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_sign_agreement | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_sign_consistency | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_standardized_surprise | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fiscal_true_streak | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| fix | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| flex_max | extended | pandas_numpy, polars | max | True | elementwise | None | None | 0 |
| flex_min | extended | pandas_numpy, polars | min | True | elementwise | None | None | 0 |
| float_share_ratio | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| floor | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| free_float_share_ratio | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| fundamental_staleness | extended | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| garman_klass_vol | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ge | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| group_count | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_decay_linear | extended | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_max | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_mean | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_min | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_neutralize | daily | pandas_numpy, polars, sql | INDUSTRY_NEUTRAL, INDUSTRY_NEUTRALIZE, IND_NEUTRALIZE, NEUTRALIZE, c_neutralize, group_demean, ind_neutralize, industry_neutral, industry_neutralize, neutralize, panel_neutralize | True | group | None | None | 0 |
| group_normalize | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_percentile | extended | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_rank | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_std | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_sum | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_weighted_mean | extended | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_weighted_zscore | extended | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_winsorize | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_zscore | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| gt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| high_low_spread_proxy | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| holder_concentration | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| holder_concentration_change | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| holder_count_change_rate | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| hump_decay | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ichimoku_cloud_position | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ichimoku_cloud_width | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ichimoku_kijun | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ichimoku_senkou_a | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ichimoku_senkou_b | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ichimoku_tenkan | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| identity | internal | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| idio_skew | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| idio_vol | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| industry_size_neutralize | extended | pandas_numpy, polars, sql | size_industry_neutralize | True | elementwise | None | None | 0 |
| intraday_volatility | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| intraday_vwap_deviation | extended | pandas_numpy |  | True | session_intraday | None | 1 | 0 |
| inverse | daily | pandas_numpy, polars, sql | inv, reciprocal | True | elementwise | None | None | 0 |
| is_finite | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| is_infinite | daily | pandas_numpy, polars, sql | IS_INFINITE, is_inf | True | elementwise | None | None | 0 |
| is_nan | extended | pandas_numpy, polars, sql | IS_NAN | True | elementwise | None | None | 0 |
| is_not_null | daily | pandas_numpy, polars, sql | IS_NOT_NULL | True | elementwise | None | None | 0 |
| is_null | daily | pandas_numpy, polars, sql | IS_NULL | True | elementwise | None | None | 0 |
| le | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| lerp | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| limit_down_close | extended | pandas_numpy | limit_down_state | True | elementwise | None | None | 0 |
| limit_up_close | extended | pandas_numpy | limit_up_state | True | elementwise | None | None | 0 |
| log | daily | pandas_numpy, polars, sql | LOG, ln | True | elementwise | None | None | 0 |
| log10 | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| log2 | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| log_abs | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| lqtp_historical_cvar | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| lt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| maximum | daily | pandas_numpy, polars, sql | fmax | True | elementwise | None | None | 0 |
| minimum | daily | pandas_numpy, polars, sql | fmin | True | elementwise | None | None | 0 |
| multiply | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ne | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| neg | daily | pandas_numpy, polars, sql | negate, reverse | True | elementwise | None | None | 0 |
| normalize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| not_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| or_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| overnight_volatility | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| parkinson_vol | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_123_bear | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_123_bull | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_ascending_triangle | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_bear_flag | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_bear_pennant | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_breakdown_retest | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_breakout_retest | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_broadening | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_bull_flag | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_bull_pennant | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_cup | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_cup_handle | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_descending_triangle | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_double_bottom | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_double_top | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_falling_channel | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_falling_wedge | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_head_shoulders | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_inverse_head_shoulders | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_rectangle | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_rising_channel | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_rising_wedge | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_rounding_bottom | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_rounding_top | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_sym_triangle | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_triple_bottom | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| pattern_triple_top | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| period_average | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| period_cagr | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| period_change | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| period_lag | extended | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| period_stability | extended | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| power | daily | pandas_numpy, polars, sql | POWER, pow | True | elementwise | None | None | 0 |
| price_impact | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| price_spread_deviation | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| price_turnover_divergence | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| price_volume_divergence | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| protected_div | internal | pandas_numpy, sql |  | True | elementwise | None | None | 0 |
| quarter_from_cumulative | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| range_volatility | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| rank | daily | pandas_numpy, polars, sql | CS_RANK, RANK, c_rank, cs_rank, cs_rank_01, panel_rank | True | cs | None | None | 0 |
| rank_corr | extended | pandas_numpy, polars | RANKCORR, RANK_CORR, rankcorr | True | ts | None | 1 | 0 |
| real_turnover_rate | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| relative_volume | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| residual_momentum_capm | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| return_per_turnover | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| return_turnover_beta | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| return_volume_beta | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| return_volume_corr | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| revision_delta | extended | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| rogers_satchell_vol | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| roll_spread_proxy | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| rolling_beta_to_market | extended | pandas_numpy, polars | FP_BETA, ROLLING_BETA_TO_MARKET, fp_beta | True | ts | None | 2 | 0 |
| rolling_obv | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| rolling_pvt | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| rolling_vwap | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| round | extended | pandas_numpy, polars | ROUND | True | elementwise | None | None | 0 |
| row_sum_skipna | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| safe_div_null | daily | pandas_numpy, polars, sql | div_or_null, safe_div | True | elementwise | None | None | 0 |
| saturate | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| scale | extended | pandas_numpy, polars, sql | SCALE, c_scale | True | cs | None | None | 0 |
| sec | unsafe | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| sigmoid | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| sign | daily | pandas_numpy, polars, sql | SIGN | True | elementwise | None | None | 0 |
| signed_dollar_volume | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| signed_log | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| signed_power | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| signed_sqrt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| signed_volume | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| signed_volume_imbalance | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| sin | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| sinh | unsafe | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| size_neutralize | extended | pandas_numpy, polars, sql | CAP_NEUTRALIZE, MARKET_CAP_NEUTRALIZE, SIZE_NEUTRALIZE, cap_neutralize, market_cap_neutralize | True | elementwise | None | None | 0 |
| sqrt | daily | pandas_numpy, polars, sql | SQRT | True | elementwise | None | None | 0 |
| sqrt_abs | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| square | extended | pandas_numpy, polars | sqr | True | elementwise | None | None | 0 |
| subtract | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| tail_beta | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| tan | unsafe | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| tanh | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| tradable_state | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| trade_when | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| true_range | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| true_turnover_rate | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| truncate | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ts_argmax | extended | pandas_numpy, polars, sql | m_argmax, ts_arg_max | True | ts | None | 1 | 0 |
| ts_argmin | extended | pandas_numpy, polars, sql | m_argmin, ts_arg_min | True | ts | None | 1 | 0 |
| ts_autocorr | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 1 |
| ts_average_volume | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_beta | daily | pandas_numpy, polars, sql | beta, m_beta, rolling_beta | True | ts | None | 1 | 0 |
| ts_bottomk_mean | extended | pandas_numpy, polars | ts_bottom_n_avg | True | ts | None | 1 | 0 |
| ts_bottomk_std | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_bottomk_sum | extended | pandas_numpy, polars | ts_bottom_n_sum | True | ts | None | 1 | 0 |
| ts_breakdown_low | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_breakout_high | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_channel_position | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_channel_width | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_channel_width_atr | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_channel_width_pct | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_channel_width_slope | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_confirmed_pivot_high | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_confirmed_pivot_low | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_consolidation_slope | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_consolidation_volume_decay | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_consolidation_width | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_corr | daily | pandas_numpy, polars, sql | Corr, TS_CORR, corr, correlation, m_cor, ts_correlation | True | ts | None | 1 | 0 |
| ts_count_if | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_cov | daily | pandas_numpy, polars, sql | Cov, Covariance, TS_COV, cov, m_cov, ts_covariance | True | ts | None | 1 | 0 |
| ts_days_since | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_days_since_high | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_days_since_low | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_decay_exp_window | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_decay_linear | extended | pandas_numpy, polars, sql | DECAY_LINEAR, TS_DECAY_LINEAR, WMA, decay_linear, ts_decay | True | ts | None | 1 | 0 |
| ts_delay | daily | pandas_numpy, polars, sql | DELAY, Delay, Ref, delay, m_delay, prev, shift | True | ts | None | 1 | 1 |
| ts_delta | daily | pandas_numpy, polars, sql | Delta, Diff, TS_DELTA, delta, deltas | True | ts | None | 1 | 1 |
| ts_distance_to_high | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_distance_to_low | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_distance_to_resistance | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_distance_to_support | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_ema | extended | pandas_numpy, polars, sql | EMA, ema, ewm, ewm_mean | True | ts | None | 1 | 0 |
| ts_ewm_corr | extended | pandas_numpy, sql | ewm_corr | True | ts | None | 1 | 0 |
| ts_ewm_cov | extended | pandas_numpy, sql | ewm_cov | True | ts | None | 1 | 0 |
| ts_ewm_std | extended | pandas_numpy, polars, sql | ewm_std | True | ts | None | 1 | 0 |
| ts_ewm_var | extended | pandas_numpy, polars, sql | ewm_var | True | ts | None | 1 | 0 |
| ts_impulse_return | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_impulse_strength | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_impulse_volume | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_kurt | extended | pandas_numpy, polars | Kurt, TS_KURT, kurt, m_kurt, ts_kurtosis | True | ts | None | 1 | 0 |
| ts_last_if | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_last_pivot_high | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_last_pivot_low | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_line_convergence | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_line_parallelism | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_log_return | daily | pandas_numpy, polars, sql | log_returns | True | ts | None | 1 | 1 |
| ts_mad | extended | pandas_numpy, polars, sql | Mad, m_mad, mad | True | ts | None | 1 | 0 |
| ts_max | daily | pandas_numpy, polars, sql | Max, TS_MAX, m_max, window_max | True | ts | None | 1 | 0 |
| ts_max_buildup | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_max_drawdown | extended | pandas_numpy, sql |  | True | ts | None | 1 | 0 |
| ts_mean | daily | pandas_numpy, polars, sql | Mean, SMA, TS_MEAN, m_avg, ma, mean, move, running_mean, window_mean | True | ts | None | 1 | 0 |
| ts_mean_if | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_median | daily | pandas_numpy, polars, sql | Median, m_median, median | True | ts | None | 1 | 0 |
| ts_min | daily | pandas_numpy, polars, sql | Min, TS_MIN, m_min, window_min | True | ts | None | 1 | 0 |
| ts_moment | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_new_high | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_new_low | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_nth_pivot_high | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_nth_pivot_high_age | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_nth_pivot_low | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_nth_pivot_low_age | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_nth_value | extended | pandas_numpy, sql |  | True | ts | None | 1 | 0 |
| ts_partial_corr | extended | pandas_numpy, sql |  | True | ts | None | 3 | 0 |
| ts_pattern_symmetry | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_pct | daily | pandas_numpy, polars, sql | TS_PCT, m_pct_change, pct_change, returns, ts_return | True | ts | None | 1 | 1 |
| ts_pivot_high_age | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_pivot_high_count | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_pivot_high_spacing | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_pivot_low_age | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_pivot_low_count | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_pivot_low_spacing | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_poly2_coeff | extended | pandas_numpy |  | True | ts | None | 3 | 0 |
| ts_poly2_resid | extended | pandas_numpy |  | True | ts | None | 3 | 0 |
| ts_prev_high | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_prev_low | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_product | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_quantile | extended | pandas_numpy, polars, sql | Percentile, TS_QUANTILE, m_percentile, percentile | True | ts | None | 1 | 0 |
| ts_range_expansion | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_rank | daily | pandas_numpy, polars, sql | TS_RANK, m_rank | True | ts | None | 1 | 0 |
| ts_ratio | extended | pandas_numpy | ratios | True | ts | None | 1 | 1 |
| ts_regression_intercept | extended | pandas_numpy | intercept | True | ts | None | 3 | 0 |
| ts_regression_r2 | extended | pandas_numpy |  | True | ts | None | 3 | 0 |
| ts_regression_resid | extended | pandas_numpy | rolling_residual | True | ts | None | 3 | 0 |
| ts_regression_slope | extended | pandas_numpy, sql | TS_REGRESSION_SLOPE, ts_regression | True | ts | None | 3 | 0 |
| ts_regression_tstat | extended | pandas_numpy, sql |  | True | ts | None | 3 | 0 |
| ts_resistance_break | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_resistance_fit_r2 | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_resistance_level | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_resistance_slope | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_sharpe | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_skew | extended | pandas_numpy, polars, sql | Skew, TS_SKEW, m_skew, skew, ts_skewness | True | ts | None | 1 | 0 |
| ts_sma_cn | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_std | daily | pandas_numpy, polars, sql | Std, TS_STD, m_std, running_std, std, std_n, ts_std_dev, ts_stddev, window_std | True | ts | None | 1 | 0 |
| ts_std_if | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_sum | daily | pandas_numpy, polars, sql | Sum, TS_SUM, m_sum, running_sum, sum, sum_n, window_sum | True | ts | None | 1 | 0 |
| ts_sum_decay | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_sum_if | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_support_break | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_support_fit_r2 | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_support_level | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_support_slope | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_swing_amplitude | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_swing_amplitude_atr | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_swing_amplitude_pct | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_swing_duration | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_swing_velocity | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_tail_mean | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_time_slope | extended | pandas_numpy, polars, sql | Slope, TS_TIME_SLOPE, slope | True | ts | None | 1 | 0 |
| ts_topk_mean | extended | pandas_numpy, polars | tm_top_n_avg, ts_top_n_avg | True | ts | None | 1 | 0 |
| ts_topk_std | extended | pandas_numpy, polars | ts_top_n_std | True | ts | None | 1 | 0 |
| ts_topk_sum | extended | pandas_numpy, polars | TS_TOPK_SUM, m_top_n_sum, tm_top_n_sum | True | ts | None | 1 | 0 |
| ts_trend_tstat | extended | pandas_numpy, sql |  | True | ts | None | 3 | 0 |
| ts_true_streak | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_var | daily | pandas_numpy, polars, sql | Var, m_var, var | True | ts | None | 1 | 0 |
| ts_zscore | daily | pandas_numpy, polars, sql | m_zscore | True | ts | None | 1 | 0 |
| ttm_from_cumulative | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| ttm_from_quarterly | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| turnover_acceleration | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| turnover_adjusted_volatility | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| turnover_autocorr | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| turnover_momentum | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| turnover_shock | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| turnover_volatility | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| turnover_zscore | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ulcer_index | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| unitize | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| up_down_volume_ratio | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| up_volume_ratio | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| volume_acceleration | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| volume_autocorr | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| volume_momentum | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| volume_shock | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| volume_to_range | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| volume_volatility | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| volume_weighted_momentum | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| volume_weighted_return | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| volume_zscore | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| vwap_deviation | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| where | daily | pandas_numpy, polars, sql | IIF, WHERE, if, if_else, iif | True | elementwise | None | None | 0 |
| winsorize | daily | pandas_numpy, polars, sql | WINSORIZE, c_winsorize | True | cs | None | None | 0 |
| winsorize_mean | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| yang_zhang_vol | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| yoy_by_period | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| zero_return_ratio | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| zscore | daily | pandas_numpy, polars, sql | CS_ZSCORE, ZSCORE, c_zscore, cs_zscore, panel_standardize, panel_zscore, standardize | True | cs | None | None | 0 |
