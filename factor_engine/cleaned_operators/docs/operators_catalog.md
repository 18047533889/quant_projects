# Operators Catalog（自动生成）

> 从 `load_all()` 去重后的最终 runtime registry 生成。
> 日常因子 DSL 仅使用 `surface=daily`；其他工具见 `research_operators/`。

## 摘要

- canonical 总数：1737
- daily：1238
- research：8
- unsafe：7
- legacy：1

| canonical | surface | backends | aliases | pit_safe | scope | lookback | min_periods | lag |
|---|---|---|---|---|---|---|---|---|
| ADX | daily | pandas_numpy, polars, sql | ts_adx | True | ts | None | 2 | 0 |
| ALMA | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ATR_WILDER | daily | pandas_numpy, polars, sql | ts_atr_wilder | True | ts | None | 2 | 0 |
| CMF | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| CMO | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ChaikinOscillator | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| CoppockCurve | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| DEMA | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| DMI_minus | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| DMI_plus | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| DX | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| EaseOfMovement | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ElderRay | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| FisherTransform | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ForceIndex | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| HMA | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| KAMA | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| KeltnerLower | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| KeltnerMid | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| KeltnerPosition | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| KeltnerUpper | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| MACD_hist | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| MACD_line | daily | pandas_numpy, polars, sql | MACD | True | ts | None | 1 | 0 |
| MACD_signal | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| MFI | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| NATR | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| PPO | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| PPO_hist | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| PPO_signal | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| PSAR | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| PVO | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| PVO_hist | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| PVO_signal | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| QQE | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| RSI_WILDER | daily | pandas_numpy, polars, sql | ts_rsi_wilder | True | ts | None | 2 | 0 |
| RSX | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| Supertrend | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| SupertrendDirection | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| TEMA | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| TSI | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| TSI_signal | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| UltimateOscillator | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| VortexMinus | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| VortexPlus | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| WMA | extended | pandas_numpy, polars, sql | ts_wma, wma | True | ts | None | 1 | 0 |
| a_share_cap_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| abnormal_turnover | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| abnormal_volume | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| abs | daily | pandas_numpy, polars, sql | ABS | True | elementwise | None | None | 0 |
| abs_return_volume_corr | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| accounting_comparability_score | extended | pandas_numpy, polars |  | False | fundamental_period | None | 12 | 0 |
| acos | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| acos_bounded | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| add | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| adv | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| altman_z_score | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| amihud_illiquidity | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| and_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| aq1_accrual_ratio_dispersion | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| aq1_accrual_stability | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| aq1_cash_conversion_strength | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| aq1_cash_flow_volatility | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| aq1_working_capital_accrual | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| arg | unsafe | pandas_numpy |  | True | elementwise | None | None | 0 |
| ashare_days_since_limit_down | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_days_since_limit_up | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_failed_limit_count | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_limit_asymmetry | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_limit_distance | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ashare_limit_down_streak | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_limit_down_touch | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| ashare_limit_down_volume_ratio | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_limit_event_density | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_limit_failed | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| ashare_limit_one_price | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| ashare_limit_open_down_streak | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_limit_open_failed | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| ashare_limit_open_up_streak | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_limit_touch_count | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_limit_up_streak | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_limit_up_touch | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| ashare_limit_up_volume_ratio | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_one_price_limit_streak | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ashare_open_at_upper_limit | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| ashare_suspension_episode_length | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| asin | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| asin_bounded | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| atan | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| atan2 | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| atr_acceleration | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| atr_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| atr_percentile | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| atr_short_long_ratio | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| atr_zscore | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| average_turnover | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| baseline_scaled_wasserstein_distance | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| benchmark_excess_return | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| benchmark_relative_price | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| beta_divergence_pct | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| beta_residual_z | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| bollinger_pct_b | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| bollinger_width | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| book_to_price | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| bounded_nvi | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| bounded_pvi | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| bvc_imbalance_ma | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| bvc_sign_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| calendar_day_diff | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| candle_abs_body | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| candle_body | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| candle_body_percentile | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_body_position | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_body_ratio | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| candle_body_strength | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| candle_body_zscore | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_close_location | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| candle_close_strength | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_direction | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| candle_gap | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_gap_atr | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_gap_pct | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_inside_ratio | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_lower_shadow | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| candle_lower_shadow_ratio | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| candle_lower_shadow_zscore | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_overlap_ratio | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_pattern_count | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| candle_range | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| candle_range_atr | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_range_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| candle_range_percentile | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_range_zscore | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_rejection_lower | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_rejection_upper | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_upper_shadow | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| candle_upper_shadow_ratio | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| candle_upper_shadow_zscore | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| candle_wick_balance | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| candlestick_pattern | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| capital_change_age | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| capital_change_magnitude | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| cash_flow_lifecycle_stage | daily | pandas_numpy, polars |  | True | fundamental_period | None | 3 | 0 |
| category_age | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| category_age_lower_bound | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| category_frequency | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| category_transition_rate | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| category_transition_surprise | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| cbrt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| cdl_dark_cloud_cover | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 1 |
| cdl_doji | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cdl_dragonfly_doji | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cdl_engulfing | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cdl_evening_star | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 2 |
| cdl_gravestone_doji | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cdl_hammer | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| cdl_hanging_man | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| cdl_harami | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 1 |
| cdl_harami_cross | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 1 |
| cdl_inside_bar | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cdl_inverted_hammer | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cdl_marubozu | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cdl_morning_star | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 2 |
| cdl_outside_bar | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cdl_piercing | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 1 |
| cdl_shooting_star | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cdl_spinning_top | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cdl_three_black_crows | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 2 |
| cdl_three_white_soldiers | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 2 |
| cdl_tweezer_bottom | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 1 |
| cdl_tweezer_top | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 1 |
| ceil | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| chikou_distance_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| choppiness_index | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| circulating_cap_ratio_change | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| circulating_cap_unlock_proxy | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| clip | daily | pandas_numpy, polars, sql | CLIP, cap, clamp | True | elementwise | None | None | 0 |
| coalesce | daily | pandas_numpy, polars, sql | COALESCE | True | elementwise | None | None | 0 |
| composition_aitchison_distance | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| composition_clr_component | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| composition_entropy | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| composition_ilr_balance | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| composition_js_divergence | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| composition_normalized_entropy | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| consolidation_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| consolidation_range_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| constant | internal | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| corwin_schultz_spread | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| cos | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| cos_phase | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| cosh | unsafe | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| coskewness_to_market | daily | pandas_numpy, polars, sql |  | True | ts | None | 5 | 0 |
| cot | unsafe | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| cross_event | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| cs1_monthly_lag_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cs1_seasonal_relative_rank | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cs1_seasonal_residual_smoothness | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cs1_week_cycle_phase | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cs1_weekday_anomaly | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cs1_weekday_effect_strength | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cs1_weekday_lag_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cs1_weekday_lag_zscore | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cs1_weekly_harmonic_power | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cs_actual_lof_score | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_autoencoder_reconstruction_error | extended | pandas_numpy, polars |  | False | cs | None | None | 0 |
| cs_bucket | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_bucket_fixed | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_bucket_historical | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_count | daily | pandas_numpy, polars, sql | c_count | True | cs | None | None | 0 |
| cs_coverage_ratio | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_demean | daily | pandas_numpy, polars, sql | CS_DEMEAN, c_demean | True | cs | None | None | 0 |
| cs_empirical_bayes_shrinkage | extended | pandas_numpy, polars |  | False | cs | None | 1 | 0 |
| cs_factor_bucket_return | extended | pandas_numpy, polars |  | False | cs | None | 1 | 0 |
| cs_fill_mean | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_fill_median | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_hartigan_dip | daily | pandas_numpy, polars |  | True | cs | None | 100 | 0 |
| cs_huber_resid | daily | pandas_numpy, polars |  | False | cs | None | None | 0 |
| cs_impute_mean | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_impute_median | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_isolation | daily | pandas_numpy, polars |  | True | cs | None | 1 | 0 |
| cs_isolation_forest_score | extended | pandas_numpy, polars |  | False | cs | None | 1 | 0 |
| cs_isotonic_residual | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_isotonic_residual_lagged_direction | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_knn_distance | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_knn_graph_dirichlet_energy | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_knn_local_gradient_norm | extended | pandas_numpy, polars |  | True | cs | None | 2 | 0 |
| cs_knn_local_linear_residual | extended | pandas_numpy, polars |  | True | cs | None | 2 | 0 |
| cs_knn_local_moran | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_knn_neighbor_retention | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_knn_peer_mean_ex_self | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_knn_tangent_residual | extended | pandas_numpy, polars |  | False | cs | None | 2 | 0 |
| cs_lad_resid | daily | pandas_numpy, polars |  | False | cs | None | None | 0 |
| cs_local_curvature | daily | pandas_numpy, polars |  | True | cs | None | 1 | 0 |
| cs_local_density | daily | pandas_numpy, polars |  | True | cs | None | 1 | 0 |
| cs_local_density_score | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_mad | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_mad_zscore | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_mahalanobis_distance | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_mean | daily | pandas_numpy, polars, sql | c_mean | True | cs | None | None | 0 |
| cs_multi_resid | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_multi_ridge_resid | extended | pandas_numpy, polars | cs_multi_robust_resid | True | cs | None | None | 0 |
| cs_neighbor_gap | daily | pandas_numpy, polars |  | True | cs | None | 1 | 0 |
| cs_neutralize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_pct_rank | daily | pandas_numpy, polars, sql | rank_pct | True | cs | None | None | 0 |
| cs_physical_panel_coverage | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_predictability_mosaic_score | extended | pandas_numpy, polars |  | True | cs | None | 1 | 0 |
| cs_quantile | daily | pandas_numpy, polars, sql | c_percentile, quantile | True | cs | None | None | 0 |
| cs_quantile_resid | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_rank_churn | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_rank_combined_churn | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_rank_composition_churn | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_rank_copula_entropy | extended | pandas_numpy, polars |  | False | cs | None | 2 | 0 |
| cs_rank_copula_mi | extended | pandas_numpy, polars |  | False | cs | None | 2 | 0 |
| cs_rank_gaussian | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_regression | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_relative_density_ratio | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_resid | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_residual_percentile | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_ridge_resid | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_robust_mahalanobis_mad | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_shrink_to_group_mean | extended | pandas_numpy, polars |  | False | group | None | 1 | 0 |
| cs_shrinkage_mahalanobis | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_sliced_wasserstein_copula_shift | extended | pandas_numpy, polars |  | True | cs | None | 2 | 0 |
| cs_spline_resid | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_std | daily | pandas_numpy, polars, sql | c_std | True | cs | None | None | 0 |
| cs_sum | daily | pandas_numpy, polars, sql | c_sum | True | cs | None | None | 0 |
| cs_tail_breadth | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_tail_retention | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_trimmed_ols_resid | extended | pandas_numpy, polars | cs_robust_resid | False | cs | None | None | 0 |
| cs_universe_coverage | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| cs_valid_count | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_weighted_demean | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_weighted_mean | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_weighted_percentile_rank | daily | pandas_numpy, polars, sql |  | True | cs | None | 1 | 0 |
| cs_weighted_zscore | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_wls_resid | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| csc | unsafe | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| cube | legacy | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| date_diff_days | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| dema_distance_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| digital_count | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 1 |
| directional_change_extent | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| directional_change_state | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| divide | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| dollar_volume | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| dollar_volume_zscore | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| donchian_breakout_down | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| donchian_breakout_up | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| donchian_channel_position | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| donchian_lower | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| donchian_mid | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| donchian_position | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| donchian_upper | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| donchian_width_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| down_volume_ratio | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| earnings_yield | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| efficiency_ratio | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| ema_crossover | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ema_distance_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ep1_earnings_autocorr | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ep1_earnings_consistency | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ep1_earnings_surprise_decay | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ep1_roa_stability | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| eq | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| erd_burst_duration | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| erd_cross_events_spacing | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| erd_event_rate_decay_slope | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| erd_event_response_amplitude | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| erd_event_window_return_gradient | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| erd_half_life_decay_count | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| erd_marked_event_decay | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| erd_post_event_hazard | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| erd_recency_decay | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| erd_sign_consistent_decay | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| es1_earnings_cv | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| es1_earnings_mean_reversion_speed | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| es1_earnings_smoothness | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| es1_negative_earnings_streak | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| es1_revenue_earnings_divergence | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| event_abnormal_return_past | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| event_active_count | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_allan_factor | daily | pandas_numpy, polars |  | True | ts | None | 8 | 0 |
| event_allan_log_mean | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_allan_scaling_slope | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_arithmetic_return_sum | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_cluster_count | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| event_cluster_duration | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| event_cluster_mean_size | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| event_cluster_score | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| event_cumulative_return_past | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| event_decay_asof | daily | pandas_numpy, polars | event_decay | True | elementwise | None | None | 0 |
| event_decay_window | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| event_direction_imbalance | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| event_direction_persistence | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| event_fano_excess | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_fano_factor | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_flip_density | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| event_frequency | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| event_hawkes_branching_ratio_proxy | extended | pandas_numpy, polars | event_hawkes_branching_ratio | False | ts | None | 5 | 1 |
| event_historical_response_mean | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| event_historical_response_sign_balance | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| event_interval_mark_coupling | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| event_interval_memory | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_level_survival_share | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_local_variation | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_log_return_sum | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_mark_autocorr | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| event_rate_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| event_recency_z | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| event_refractory | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| event_response_decay_rate | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| event_response_dispersion | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| event_response_effective_events | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_response_overlap_ratio | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| event_response_peak_lag | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| event_response_reversal_strength | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| event_return_since_last | daily | pandas_numpy, polars | event_compounded_return | True | elementwise | None | None | 0 |
| event_streak | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| ewm_corr | extended | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ewm_cov | extended | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ex_self_mad_z | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ex_self_mean_gap | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ex_self_rank_pct | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ex_self_zscore | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| exp | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| exp_neg | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| expanding_rank | daily | pandas_numpy, polars, sql | cum_rank | True | ts | None | 1 | 0 |
| ffill_limit | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| fillna_const | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| fin_accrual_ratio | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_acquisition_cash_intensity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_actual_expectation_divergence | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_announcement_lag | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| fin_applicability_mask | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| fin_average_balance | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_beat_streak | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_borrowing_intensity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_cagr | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_capex_growth | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_capex_intensity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_cash_burn_runway | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_cash_conversion | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_cash_earnings_gap | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_cash_sales_divergence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_cashflow_persistence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_common_size | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_component_score | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| fin_comprehensive_income_gap | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_contract_asset_growth | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_contract_asset_intensity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_contract_asset_liability_gap | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_contract_liability_growth | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_contract_liability_intensity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_core_earnings_ratio | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_cv | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_days_since_expectation_revision | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_days_since_update | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_debt_repayment_intensity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_debt_service_coverage_proxy | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_deferred_tax_gap | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_delta_noa | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_diff | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_discontinued_operation_ratio | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_divergence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_earnings_cash_gap_volatility | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_earnings_persistence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_earnings_smoothness | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_equity_capital_growth | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_expectation_dispersion | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_expectation_revision | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_expectation_revision_count | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_expectation_revision_magnitude | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_expectation_revision_pct | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_expectation_revision_speed | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_expense_sales_divergence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_fair_value_income_dependence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_financing_gap | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_fundamental_strength_coverage | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_fundamental_strength_score | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_goodwill_intensity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_goodwill_risk_score | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_growth | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_growth_acceleration | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_growth_change | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_growth_persistence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_growth_stability | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_growth_volatility | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_impairment_intensity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_interest_coverage_proxy | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_inventory_sales_divergence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_investment_income_dependence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_lag | daily | pandas_numpy, polars, sql | report_lag | True | fundamental_period | None | 1 | 0 |
| fin_lease_asset_liability_gap | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_lease_intensity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_log_change | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_margin_persistence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_mean_abs_deviation | extended | pandas_numpy, polars, sql | fin_mad | True | fundamental_period | None | 1 | 0 |
| fin_median_abs_deviation | extended | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_minority_profit_share | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_miss_streak | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_monotonicity | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_negative_streak | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_net_borrowing_cashflow | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_net_debt_issuance | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_noncore_income_ratio | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_oci_to_equity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_other_earnings_dependence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_pct_change | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_percentile_history | daily | pandas_numpy, polars | report_rank | True | fundamental_period | None | 1 | 0 |
| fin_percentile_vs_prior_history | extended | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_period_restated | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_period_revision_age | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_period_revision_count | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_positive_streak | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_qoq | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_quarter_from_cumulative | daily | pandas_numpy, polars | report_single_quarter | True | fundamental_period | None | 1 | 0 |
| fin_range | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_ratio | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_rd_capitalization_ratio | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_rd_total_intensity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_receivable_sales_divergence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_restated_flag | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_revision_count | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_revision_delta | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_revision_direction | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_revision_magnitude | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_revision_pct | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_roe_cash_gap | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_seasonal_percentile | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_seasonal_zscore | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_sign_change_count | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_stability | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_staleness | daily | pandas_numpy, polars | report_age | False | elementwise | None | None | 0 |
| fin_std | daily | pandas_numpy, polars, sql | report_rolling_std | True | fundamental_period | None | 1 | 0 |
| fin_surprise | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_surprise_event_percentile | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_surprise_event_zscore | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_surprise_zscore | daily | pandas_numpy, polars | report_surprise_to_trend | False | elementwise | None | None | 0 |
| fin_total_operating_accruals | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| fin_trend_acceleration | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_trend_r2 | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_trend_slope | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_trend_tstat | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_ttm | extended | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| fin_ttm_cumulative | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_ttm_quarterly | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_turnover | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_working_capital_accruals | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_working_capital_change | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_yoy | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| fin_zscore_history | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fin_zscore_vs_prior_history | extended | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_acceleration | extended | pandas_numpy, polars |  | True | fundamental_period | None | 3 | 0 |
| fiscal_accrual_quality | daily | pandas_numpy, polars |  | True | fundamental_period | None | 4 | 0 |
| fiscal_ar_resid_std | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_asymmetric_elasticity | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_asymmetric_timeliness | extended | pandas_numpy, polars |  | False | fundamental_period | None | None | 0 |
| fiscal_autocorr | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_change_direction_agreement | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_direction_consistency | daily | pandas_numpy, polars |  | True | fundamental_period | None | 3 | 0 |
| fiscal_pair_direction_agreement | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_pct_change | extended | pandas_numpy, polars |  | True | fundamental_period | None | 2 | 0 |
| fiscal_perpetual_inventory | daily | pandas_numpy, polars |  | True | fundamental_period | None | 8 | 0 |
| fiscal_regression_resid_std | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_reversal_ratio | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_rolling_std | extended | pandas_numpy, polars |  | True | fundamental_period | None | 3 | 0 |
| fiscal_sign_agreement | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_sign_consistency | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_standardized_surprise | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fiscal_true_streak | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| fix | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| flex_max | daily | pandas_numpy, polars, sql | max | True | elementwise | None | None | 0 |
| flex_min | daily | pandas_numpy, polars, sql | min | True | elementwise | None | None | 0 |
| float_share_ratio | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| floor | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| free_float_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| free_float_share_ratio | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| free_float_turnover | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| free_to_circulating_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| fundamental_staleness | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| garman_klass_vol | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ge | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| group_corr_mst_length | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_count | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_current_members_tail_coexceedance | daily | pandas_numpy, polars | group_tail_coexceedance_density | True | group | None | None | 0 |
| group_decay_linear | daily | pandas_numpy, polars, sql | group_rank_linear_weighted_value | True | group | None | None | 0 |
| group_distribution_js_divergence | daily | pandas_numpy, polars |  | True | group | None | 1 | 0 |
| group_ex_self_mad | daily | pandas_numpy, polars |  | True | group | None | 1 | 0 |
| group_ex_self_mean | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_ex_self_quantile | daily | pandas_numpy, polars |  | True | group | None | 1 | 0 |
| group_ex_self_std | daily | pandas_numpy, polars |  | True | group | None | 1 | 0 |
| group_ex_self_weighted_mean | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_feature_coverage_ratio | extended | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_feature_effective_rank | daily | pandas_numpy, polars | group_corr_effective_rank | True | group | None | None | 0 |
| group_feature_mode_localization | daily | pandas_numpy, polars | group_corr_mode_localization | True | group | None | None | 0 |
| group_feature_mode_share | daily | pandas_numpy, polars | group_corr_mode_share | True | group | None | None | 0 |
| group_feature_second_mode_localization | daily | pandas_numpy, polars | group_corr_second_mode_localization | True | group | None | None | 0 |
| group_feature_spectral_gap | daily | pandas_numpy, polars | group_corr_spectral_gap | True | group | None | None | 0 |
| group_feature_valid_member_count | extended | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_impute_median | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_kurtosis | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_leader_laggard_exposure | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_max | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_mean | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_min | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_multi_level_rank_consistency | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_multi_resid | extended | pandas_numpy, polars |  | False | group | None | None | 0 |
| group_neutralize | daily | pandas_numpy, polars, sql | INDUSTRY_NEUTRAL, INDUSTRY_NEUTRALIZE, IND_NEUTRALIZE, NEUTRALIZE, c_neutralize, group_demean, ind_neutralize, industry_neutral, industry_neutralize, neutralize, panel_neutralize | True | group | None | None | 0 |
| group_normalize | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_peer_beta_deviation | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_peer_deviation_index | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_peer_information_diffusion | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_percentile | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_quantile_spread | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_rank | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_rank_weighted_value | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_return_dispersion_exposure | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_signal_attraction_share | extended | pandas_numpy, polars | relation_pagerank_centrality | False | group | None | 2 | 0 |
| group_skewness | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_spd_feature_structure_shift | extended | pandas_numpy, polars |  | True | group | None | 2 | 0 |
| group_std | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_sum | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_tail_centrality | extended | pandas_numpy, polars |  | True | group | None | 5 | 0 |
| group_tail_lead_score | extended | pandas_numpy, polars |  | False | group | None | 5 | 0 |
| group_tail_ratio | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_topk_mean | daily | pandas_numpy, polars, sql |  | True | group | None | 1 | 0 |
| group_ts_decay_linear | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_valid_count | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_wasserstein_barycenter_distance | extended | pandas_numpy, polars |  | True | group | None | 4 | 0 |
| group_weighted_mean | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_weighted_zscore | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| group_winsorize | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_zscore | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| gt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| hfl_hurst_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| hfl_variance_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| hierarchical_group_neutralize | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| high_low_spread_proxy | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| holder_class_entropy | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_class_js_shift | extended | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| holder_common_holding_peer_return | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_company_ownership_hhi | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| holder_concentration | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| holder_concentration_acceleration | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_concentration_change | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| holder_concentration_slope | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_count_change_rate | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| holder_disclosure_count | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| holder_disclosure_coverage | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| holder_entry_share | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_exit_share | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_float_concentration_gap | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_freeze_concentration | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_freeze_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_id_matched_churn | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_id_matched_entry_share | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_id_matched_exit_share | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_id_overlap_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_locked_share_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_nature_entropy | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_net_entry_share | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_observed_topk_hhi | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| holder_peer_return_breadth | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_pledge_change | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_pledge_churn | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| holder_pledge_concentration | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_pledge_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_pledged_holder_count | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_rank_stability | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_share_weighted_rank_migration | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_shareholder_network_centrality | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_shareholder_overlap_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| holder_topk_share_sum | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| holder_weighted_churn | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| hump_decay | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ichimoku_cloud_position | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ichimoku_cloud_width | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ichimoku_kijun | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ichimoku_senkou_a | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ichimoku_senkou_b | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ichimoku_tenkan | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| identity | internal | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| idio_skew | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| idio_vol | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| index_entry_exit_event | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| index_event_decay | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| index_member | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| index_membership_age | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| index_reconstitution_churn | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| index_weight | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| index_weight_change | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| index_weight_gap_to_free_float | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| industry_fiscal_resid | extended | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| industry_rolling_pca_loading | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| industry_size_neutralize | daily | pandas_numpy, polars, sql | size_industry_neutralize | True | cs | None | None | 0 |
| intra_abs_return_profile_cosine | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_amihud | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_amount_profile_cosine | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_amount_profile_jsd | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_bar_range_deviation | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_bar_range_persistence | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_beta_asymmetry | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_bipower_variation | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_close_participation | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_common_trading_intensity | extended | pandas_numpy, polars |  | False | session_intraday | None | 3 | 0 |
| intra_concentration | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_consolidation_quality | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_continuous_variance | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_down_down_semibeta | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_down_up_semibeta | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_drawdown_depth | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_drawdown_duration | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_drawdown_recovery_half_life | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_dynamic_stock_graph_features | extended | pandas_numpy, polars |  | False | session_intraday | None | 5 | 0 |
| intra_entropy | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_eod_reversal_decomposition | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_event_pre_post_contrast | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_event_window_reduce | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_extreme_bar_return | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_high_low_affinity | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_high_time | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_idiosyncratic_kurtosis | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_idiosyncratic_kurtosis_ex_self | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_idiosyncratic_skewness | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_idiosyncratic_skewness_ex_self | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_idiosyncratic_variance | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| intra_idiosyncratic_variance_ex_self | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_impulse_event_detector | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_industry_lead_lag_ex_self | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_interval_amount_share | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_interval_illiquidity | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_interval_realized_variance | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_interval_return | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_interval_volume_share | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_interval_vwap_deviation | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_jump_clustering | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_jump_concentration | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_jump_count | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_jump_first_time | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_jump_last_time | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_jump_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_jump_variation | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_kyle_lambda_proxy | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_limit_duration | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_limit_first_hit_time | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_limit_pre_hit_pressure_profile | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_limit_reopen_count | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_liquidity_resilience_curve_fit | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_local_conditional_entropy | extended | pandas_numpy, polars |  | False | session_intraday | None | 10 | 0 |
| intra_longest_above_vwap_streak | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_longest_below_vwap_streak | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_low_time | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_lunch_gap_return | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_market_lead_lag_ex_self | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_market_model_r2 | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_market_model_r2_ex_self | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_max_drawdown | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_max_drawup | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_multiresolution_resample_reduce | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_negative_jump_variation | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| intra_negative_tail_variation | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_neighbor_event_class | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_path_efficiency | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_positive_jump_variation | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| intra_positive_tail_variation | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_post_impulse_response | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_price_delay | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_price_vwap_max_negative_excursion | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_price_vwap_max_positive_excursion | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_probe_outcome_score | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_profile_earth_mover_distance | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_range_gap_flag | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_realized_beta | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| intra_realized_beta_ex_self | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_realized_correlation | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| intra_realized_correlation_ex_self | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_realized_kurtosis | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_realized_quarticity | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_realized_semivariance | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_realized_skewness | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_realized_variance | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_response_curve_features | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_return_activity_corr | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_return_profile_cosine | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| intra_round_price_barrier_response | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_round_price_clustering_share | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_same_slot_momentum | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_same_slot_reversal | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_same_slot_zscore | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_segment_amount_share | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_segment_realized_vol | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_segment_return | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_segment_volume_share | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_segment_vwap_deviation | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_session_boundary_jump | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_session_mean_reversion | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_session_return_asymmetry | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_signed_imbalance_proxy | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_signed_jump_ratio | daily | pandas_numpy, polars | intraday_signed_jump_balance | False | elementwise | None | None | 0 |
| intra_signed_return_profile_cosine | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_signed_tail_variation_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_slice_mask_pair_reduce | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_slice_mask_reduce | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_slot_amount_surprise | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_slot_volatility_surprise | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_slot_volume_surprise | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_smart_money_vwap_ratio | extended | pandas_numpy, polars |  | False | session_intraday | None | 4 | 0 |
| intra_state_count | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_state_dwell_stats | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_state_follow_beta | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_state_follow_corr | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_state_follow_ratio | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_state_interval_moment | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_state_pair_same_slot_corr | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_state_sum | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_state_transition_entropy | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_state_vwap | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_supply_absorption_score | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_tail_event_count | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_tail_volume_share | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_time_above_vwap | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_tripower_quarticity | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_up_down_semibeta | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_up_up_semibeta | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_ute_high | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_ute_low | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_volume_at_price_profile | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_volume_imbalance | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_volume_price_alignment | daily | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_volume_profile_cosine | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_volume_profile_jsd | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_volume_profile_peak_geometry | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_volume_profile_supply_structure | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_volume_profile_value_area | extended | pandas_numpy, polars |  | True | session_intraday | None | 1 | 0 |
| intra_vwap_above_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_vwap_cross_count | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_vwap_path_curvature | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_vwap_path_curvature_pct | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_vwap_path_slope | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_vwap_path_slope_pct | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intra_vwap_reversion_speed | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intraday_activity_duration_curvature | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| intraday_barrier_approach_acceleration | extended | pandas_numpy, polars |  | True | session_intraday | None | 20 | 0 |
| intraday_bvc_imbalance | daily | pandas_numpy, polars |  | True | session_intraday | None | 20 | 0 |
| intraday_impact_asymmetry | daily | pandas_numpy, polars |  | True | session_intraday | None | 20 | 0 |
| intraday_impact_beta | daily | pandas_numpy, polars |  | True | session_intraday | None | 20 | 0 |
| intraday_impact_decay_rate | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| intraday_jump_test_stat | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intraday_medrv | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intraday_minrv | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intraday_profile_pca_residual | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| intraday_profile_phase_shift | extended | pandas_numpy, polars |  | False | session_intraday | None | 20 | 0 |
| intraday_profile_surprise_energy | extended | pandas_numpy, polars |  | True | session_intraday | None | 20 | 0 |
| intraday_quantile_curve_pca_residual | extended | pandas_numpy, polars |  | False | session_intraday | None | 20 | 0 |
| intraday_quantile_curve_pca_score | extended | pandas_numpy, polars |  | False | session_intraday | None | 20 | 0 |
| intraday_realized_power_variation | extended | pandas_numpy, polars |  | False | session_intraday | None | 20 | 0 |
| intraday_realized_semivariance_balance | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| intraday_return_wasserstein_shift | daily | pandas_numpy, polars |  | True | session_intraday | None | 30 | 0 |
| intraday_rv_signature_curvature | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intraday_rv_signature_slope | daily | pandas_numpy, polars |  | True | session_intraday | None | 2 | 0 |
| intraday_session_shape_novelty | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| intraday_subsampled_rv_dispersion | extended | pandas_numpy, polars |  | True | session_intraday | None | 20 | 0 |
| intraday_volatility | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| intraday_volatility_concentration | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| intraday_volatility_entropy | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| intraday_volatility_signature_slope | extended | pandas_numpy, polars |  | True | session_intraday | None | 20 | 0 |
| intraday_volatility_time_centroid | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| intraday_volume_clock_path_efficiency | daily | pandas_numpy, polars |  | True | session_intraday | None | 2 | 0 |
| intraday_volume_clock_roughness | daily | pandas_numpy, polars |  | True | session_intraday | None | 2 | 0 |
| intraday_vwap_deviation | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| intraday_wasserstein_pair_distance | extended | pandas_numpy, polars |  | True | session_intraday | None | 20 | 0 |
| inverse | daily | pandas_numpy, polars, sql | inv, reciprocal | True | elementwise | None | None | 0 |
| is_finite | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| is_infinite | daily | pandas_numpy, polars, sql | IS_INFINITE, is_inf | True | elementwise | None | None | 0 |
| is_nan | daily | pandas_numpy, polars, sql | IS_NAN | True | elementwise | None | None | 0 |
| is_not_null | daily | pandas_numpy, polars, sql | IS_NOT_NULL | True | elementwise | None | None | 0 |
| is_null | daily | pandas_numpy, polars, sql | IS_NULL | True | elementwise | None | None | 0 |
| kama_distance_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| keltner_breakout_strength | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| keltner_compression | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| keltner_width_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| le | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| lerp | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| lf1_amihud_own | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| lf1_herfindahl_volume | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| lf1_liquidity_decay_base | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| lf1_market_share_turnover | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| lf1_mean_variance_turnover | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| lf1_own_volume_share | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| lf1_ret_volume_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| lf1_turnover_distribution_skew | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| lf1_volume_amihud_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| lf1_volume_amplification | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| lf1_volume_concentration | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| limit_down_close | daily | pandas_numpy, polars, sql | limit_down_state | True | elementwise | None | None | 0 |
| limit_up_close | daily | pandas_numpy, polars, sql | limit_up_state | True | elementwise | None | None | 0 |
| listing_age | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| log | daily | pandas_numpy, polars, sql | LOG, ln | True | elementwise | None | None | 0 |
| log10 | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| log2 | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| log_abs | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| log_positive_or_nan | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| lqtp_historical_cvar | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| lt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| m1_corr_momentum | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| m1_cs_momentum_dispersion | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| m1_momentum_regime | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| m1_momentum_speed_change | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| m1_momentum_stability | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| m1_momentum_strength | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| m1_rank_momentum_gap | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| m1_ranked_momentum | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| m1_volume_adjusted_momentum | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ma_slope_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| market_cap_free_cap_gap | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| maximum | daily | pandas_numpy, polars, sql | fmax | True | elementwise | None | None | 0 |
| micro_bvc_vpin | research | pandas_numpy, polars |  | True | session_intraday | None | 20 | 0 |
| minimum | daily | pandas_numpy, polars, sql | fmin | True | elementwise | None | None | 0 |
| multi_index_entry_intensity | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| multiply | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ne | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| neg | daily | pandas_numpy, polars, sql | negate, reverse | True | elementwise | None | None | 0 |
| negative_event_age | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| negative_event_rate | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| nonfinite_to_num | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| normalize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| not_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ofi_abs_imbalance_trend | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ofi_dominant_direction | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ofi_imbalance_agreement | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ofi_imbalance_cv | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ofi_imbalance_persistence | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ofi_reversal_rate | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ofi_volume_flow_regime | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ofi_volume_imbalance | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ofi_zero_flow_balance | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ohlc_corwin_schultz_spread | extended | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| open_close_return | daily | pandas_numpy, polars, sql | intraday_return | True | elementwise | None | None | 0 |
| open_to_vwap_return | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| or_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| overnight_return | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 1 |
| overnight_volatility | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| panel_async_beta_ex_self | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| panel_factor_pocket_strength | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| panel_mixture_of_experts_score | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| panel_peer_graph_aggregate | extended | pandas_numpy, polars |  | False | cs | None | 1 | 0 |
| panel_predictability_mosaic_score | extended | pandas_numpy, polars |  | True | cs | None | 1 | 0 |
| panel_regime_conditioned_forecast | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| panel_rolling_elastic_net_forecast | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| panel_rolling_pca_explained_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| panel_rolling_pca_loading | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| panel_rolling_pca_resid | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| panel_rolling_pca_resid_momentum | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| panel_rolling_pca_resid_vol | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| panel_rolling_pcr_forecast | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| panel_rolling_pls_forecast | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| parkinson_vol | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| pattern_123_bear | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_123_bull | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_ascending_triangle | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_bear_flag | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_bear_pennant | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_breakdown_retest | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_breakout_retest | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_broadening | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_bull_flag | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_bull_pennant | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_cup | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_cup_handle | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_descending_triangle | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_double_bottom | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_double_top | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_falling_channel | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_falling_wedge | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_head_shoulders | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_inverse_head_shoulders | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_rectangle | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_rising_channel | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_rising_wedge | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_rounding_bottom | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_rounding_top | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_sym_triangle | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_triple_bottom | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| pattern_triple_top | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| period_average | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| period_cagr | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 2 | 0 |
| period_change | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| period_lag | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| period_stability | daily | pandas_numpy, polars |  | True | fundamental_period | None | 2 | 0 |
| piotroski_f_score | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| piotroski_f_score_tolerant | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| piotroski_observed_count | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| piotroski_partial_score | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| positive_event_age | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| positive_event_rate | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| power | daily | pandas_numpy, polars, sql | POWER, pow | True | elementwise | None | None | 0 |
| price_impact | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| price_spread_deviation | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| price_turnover_divergence | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| price_volume_divergence | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| protected_div | internal | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| psar_days_since_flip | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| psar_direction | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| psar_distance_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| psar_flip | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| quarter_from_cumulative | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| range_volatility | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| rank | daily | pandas_numpy, polars, sql | CS_RANK, RANK, c_rank, cs_rank, cs_rank_01, panel_rank | True | cs | None | None | 0 |
| rank_corr | daily | pandas_numpy, polars, sql | RANKCORR, RANK_CORR, rankcorr | True | ts | None | 2 | 0 |
| real_turnover_rate | daily | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| reg_forecast_error_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| reg_r2_trailing | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| reg_residual_zscore | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| reg_slope_tstat | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| relation_category_share | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| relation_category_signed_contribution | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| relation_concentration_acceleration | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_diffusion_score | extended | pandas_numpy, polars |  | False | group | None | 2 | 0 |
| relation_distinct_count | extended | pandas_numpy, polars |  | False | cs | None | None | 0 |
| relation_distribution_excess_kurtosis | extended | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_distribution_pearson_kurtosis | daily | pandas_numpy, polars | relation_distribution_kurtosis | True | cs | None | None | 0 |
| relation_distribution_skew | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_entropy | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_entropy_change | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_entry_count | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| relation_exit_count | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| relation_hhi | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_hhi_change | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_jaccard | extended | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| relation_overlap_ratio | extended | pandas_numpy, polars |  | False | cs | None | None | 0 |
| relation_peer_weighted_mean_ex_self | daily | pandas_numpy, polars |  | True | group | None | None | 0 |
| relation_rank_entity_mobility | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| relation_rank_mobility | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_rank_weighted_sum | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_share_mobility | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_topk_concentration | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_topk_sum | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| relation_weighted_change | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| relation_weighted_std_ex_self | daily | pandas_numpy, polars |  | True | group | None | 1 | 0 |
| relative_strength_group_pct | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| relative_volume | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| report_benford_js_divergence | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| report_change_breadth | daily | pandas_numpy, polars |  | True | fundamental_period | None | 5 | 0 |
| report_change_coherence | daily | pandas_numpy, polars |  | True | fundamental_period | None | 5 | 0 |
| report_filing_delay_surprise | daily | pandas_numpy, polars |  | True | fundamental_period | None | 5 | 0 |
| report_revision_magnitude | extended | pandas_numpy, polars |  | False | ts | None | 5 | 0 |
| report_rolling_mean | daily | pandas_numpy, polars |  | True | fundamental_period | None | 2 | 0 |
| report_yoy_lag | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| residual_momentum_capm | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| return_per_turnover | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| return_turnover_beta | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| return_volume_beta | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| return_volume_corr | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| revision_delta | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| rogers_satchell_vol | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| roll_spread_proxy | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| rolling_adl_flow | daily | pandas_numpy, polars, sql | ADL | True | ts | None | 1 | 0 |
| rolling_beta_to_market | extended | pandas_numpy, polars | FP_BETA, ROLLING_BETA_TO_MARKET, fp_beta | False | ts | None | 2 | 0 |
| rolling_obv | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| rolling_pvt | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| rolling_vwap | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| round | daily | pandas_numpy, polars, sql | ROUND | True | elementwise | None | None | 0 |
| row_sum_skipna | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | None | 0 |
| safe_div_null | daily | pandas_numpy, polars, sql | div_or_null, safe_div | True | elementwise | None | None | 0 |
| same_clock_lag | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| saturate | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| scale | daily | pandas_numpy, polars, sql | SCALE, c_scale | True | cs | None | None | 0 |
| sec | unsafe | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| senkou_span_causal_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| session_event_recovery_score | daily | pandas_numpy, polars |  | True | session_intraday | None | 2 | 0 |
| sigmoid | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| sign | daily | pandas_numpy, polars, sql | SIGN | True | elementwise | None | None | 0 |
| signed_dollar_volume | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| signed_event_decay | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| signed_event_rate | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| signed_log | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| signed_power | daily | pandas_numpy, polars, sql |  | True | elementwise | None | 1 | 0 |
| signed_sqrt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| signed_volume | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| signed_volume_imbalance | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| sin | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| sin_phase | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| sinh | unsafe | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| size_neutralize | daily | pandas_numpy, polars, sql | CAP_NEUTRALIZE, MARKET_CAP_NEUTRALIZE, SIZE_NEUTRALIZE, cap_neutralize, market_cap_neutralize | True | cs | None | None | 0 |
| sma_distance_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| spectral_energy_ratio | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| spectral_trend_share | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| sqrt | daily | pandas_numpy, polars, sql | SQRT | True | elementwise | None | None | 0 |
| sqrt_abs | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| square | daily | pandas_numpy, polars, sql | sqr | True | elementwise | None | None | 0 |
| sr_distance_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| sr_touch_count | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_adaptive_deadband | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_adaptive_slew_limit | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_age | extended | pandas_numpy, polars | state_episode_duration | True | elementwise | None | 1 | 0 |
| state_confidence_weighted_ema | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_cost_aware_deadband | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_cost_aware_slew | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_deadband | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_dwell_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_episode_age | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| state_episode_age_capped | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| state_episode_age_lower_bound | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| state_episode_censored_flag | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| state_episode_efficiency | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| state_episode_excursion_balance | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| state_episode_mae | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| state_episode_mfe | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| state_episode_retrace_ratio | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| state_ewm_if | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_flip_age | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| state_flip_density | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| state_hold | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_l1_turnover_prox | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_l2_partial_adjustment | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_latch | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_persistence | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| state_quantile_hysteresis | extended | pandas_numpy, polars |  | True | group | None | None | 0 |
| state_rank_deadband | extended | pandas_numpy, polars |  | True | group | None | None | 0 |
| state_since_count | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| state_since_last | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| state_since_mean | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| state_since_sum | extended | pandas_numpy, polars | state_since_reduce | True | elementwise | None | None | 0 |
| state_since_trend_tstat | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| state_slew_limit | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_transition_count | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| state_transition_rate | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| state_transition_surprise | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| state_uncertainty_deadband | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| subtract | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| supertrend_days_since_flip | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| supertrend_direction | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| supertrend_distance_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| supertrend_flip | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| suspension_frequency | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| suspension_status_coverage | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| sv_net_flow_direction | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| sv_own_flow_fraction | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| sv_self_relative_change | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| sv_signed_beta_market | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| sv_signed_shock_persistence | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| sv_signed_volume_volatility | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| tail_beta | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| tan | unsafe | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| tanh | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| tema_distance_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| tenkan_kijun_cross | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| tod_close_midday_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| tod_edge_activity | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| tod_intraday_range_position | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| tod_open_midday_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| tod_overnight_activity_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| tradable_state | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| trade_when | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| trading_day_diff | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| true_range | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| true_range_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| true_range_surprise | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| true_range_zscore | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| true_turnover_rate | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| truncate | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ts_abdi_ranaldo_spread | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_abs_concentration | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_abs_entropy | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_abs_entropy_nats | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_abs_entropy_normalized | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_active_information_storage | extended | pandas_numpy, polars |  | False | ts | None | 5 | 1 |
| ts_activity_clock_age | daily | pandas_numpy, polars |  | True | ts | None | 2 | 1 |
| ts_activity_clock_lagged_value | daily | pandas_numpy, polars |  | True | ts | None | 2 | 1 |
| ts_activity_clock_lagged_value_prior | extended | pandas_numpy, polars | activity_clock_lagged_value_prior | True | ts | None | 1 | 0 |
| ts_adaptive_noise_kalman | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_alpha_beta_filter | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ar_coeff_stability | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ar_coefficient | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ar_fitted_value | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_ar_forecast | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_ar_in_sample_resid | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_ar_innovation | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_ar_innovation_z | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_ar_prior_coeff | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ar_prior_forecast | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ar_prior_innovation | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ar_prior_innovation_z | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_argmax | daily | pandas_numpy, polars, sql | m_argmax, ts_arg_max | True | ts | None | 1 | 0 |
| ts_argmax_age | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_argmax_index_from_oldest | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_argmin | daily | pandas_numpy, polars, sql | m_argmin, ts_arg_min | True | ts | None | 1 | 0 |
| ts_argmin_age | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_argmin_index_from_oldest | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_autocorr | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 1 |
| ts_autocorr_decay_half_life | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_autocorrelation_time | daily | pandas_numpy, polars | ts_integrated_autocorrelation_time | True | ts | None | 1 | 0 |
| ts_autocorrelation_time_initial_positive_sequence | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_average_volume | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_bds_statistic | extended | pandas_numpy, polars |  | True | ts | None | 30 | 0 |
| ts_bessel_lowpass_causal | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_best_lag_corr_excess | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_best_lag_corr_raw | extended | pandas_numpy, polars | ts_best_lag_corr | False | ts | None | 1 | 0 |
| ts_beta | daily | pandas_numpy, polars, sql | beta, m_beta, rolling_beta | True | ts | None | 2 | 0 |
| ts_beta_break_score | extended | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_beta_if | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_betti_1_max_persistence | extended | pandas_numpy, polars |  | False | ts | None | 2 | 0 |
| ts_bicoherence_top_decile_excess | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_bicoherence_top_decile_mean | extended | pandas_numpy, polars | ts_bicoherence_max | True | ts | None | 16 | 0 |
| ts_binned_response_curvature | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_binned_response_monotonicity | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_bottomk_mean | daily | pandas_numpy, polars, sql | ts_bottom_n_avg | True | ts | None | 1 | 0 |
| ts_bottomk_std | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_bottomk_sum | daily | pandas_numpy, polars, sql | ts_bottom_n_sum | True | ts | None | 1 | 0 |
| ts_breakdown_low | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_breakout_high | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_bures_corr_shift | extended | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_butterworth_lowpass_causal | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_causal_local_linear_smoother | extended | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_causal_savgol_endpoint | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_change_point_probability | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_channel_position | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_channel_width | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_channel_width_atr | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_channel_width_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_channel_width_slope | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_chatterjee_xi | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_chord_excursion_area | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_conditional_mutual_information | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_conditional_transfer_entropy | extended | pandas_numpy, polars |  | False | ts | None | 3 | 0 |
| ts_confirmed_pivot_high | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_confirmed_pivot_low | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_consolidation_slope | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_consolidation_volume_decay | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_consolidation_width | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_copula_central_asymmetry | extended | pandas_numpy, polars |  | False | ts | None | 5 | 0 |
| ts_corr | daily | pandas_numpy, polars, sql | Corr, TS_CORR, corr, correlation, m_cor, ts_correlation | True | ts | None | 2 | 0 |
| ts_corr_if | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_count_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_cov | daily | pandas_numpy, polars, sql | Cov, Covariance, TS_COV, cov, m_cov, ts_covariance | True | ts | None | 2 | 0 |
| ts_cov_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_coverage_ratio | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_cpt_value | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_cross_extremogram | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_cross_quantilogram | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_cross_spectral_coherence | daily | pandas_numpy, polars |  | True | ts | None | 8 | 0 |
| ts_cross_spectral_phase | daily | pandas_numpy, polars |  | True | ts | None | 8 | 0 |
| ts_crossing_acceleration | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_crossing_speed | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_cumulative_deviation_score | extended | pandas_numpy, polars | ts_cusum_break_score | True | ts | None | 1 | 0 |
| ts_current_drawdown_area | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_current_drawdown_duration | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_cusum_pressure | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_cusum_vol_break_score | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_days_since | daily | pandas_numpy, polars, sql | event_age | True | ts | None | 1 | 0 |
| ts_days_since_high | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_days_since_low | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_dc_duration_asymmetry | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_dc_event_rate | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_dc_overshoot_asymmetry | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_dc_overshoot_ratio | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_decay_exp_window | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_decay_linear | daily | pandas_numpy, polars, sql | DECAY_LINEAR, TS_DECAY_LINEAR, decay_linear, ts_decay | True | ts | None | 1 | 0 |
| ts_delay | daily | pandas_numpy, polars, sql | DELAY, Delay, Ref, delay, m_delay, prev, shift | True | ts | None | 1 | 1 |
| ts_delay_intrinsic_dimension | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_delta | daily | pandas_numpy, polars, sql | Delta, Diff, TS_DELTA, delta, deltas | True | ts | None | 1 | 1 |
| ts_detrended_level_spectral_entropy | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_dfa_hurst | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_distance_corr | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_distance_correlation_partial_proxy | daily | pandas_numpy, polars | ts_partial_distance_correlation | True | ts | None | 1 | 0 |
| ts_distance_cov | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_distance_to_high | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_distance_to_low | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_distance_to_resistance | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_distance_to_support | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_dmd_dominant_frequency | research | pandas_numpy, polars |  | True | ts | None | 12 | 0 |
| ts_dmd_dominant_growth_rate | research | pandas_numpy, polars |  | True | ts | None | 12 | 0 |
| ts_dmd_level_dominant_frequency | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_dmd_level_dominant_growth_rate | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_dmd_level_mode_concentration | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_dmd_mode_concentration | research | pandas_numpy, polars |  | True | ts | None | 12 | 0 |
| ts_dmd_return_dominant_frequency | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_dmd_return_dominant_growth_rate | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_dmd_return_mode_concentration | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_dominant_cycle_period | daily | pandas_numpy, polars |  | True | ts | None | 16 | 0 |
| ts_downside_deviation | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_edge_effective_spread | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_effective_transfer_entropy | extended | pandas_numpy, polars |  | False | ts | None | 2 | 0 |
| ts_effective_turning_rate | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ema | daily | pandas_numpy, polars, sql | EMA, ema, ewm, ewm_mean | True | ts | None | 1 | 0 |
| ts_endpoint_deviation | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_energy_break_score | daily | pandas_numpy, polars |  | True | ts | None | 10 | 1 |
| ts_envelope_boundary_dwell | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_envelope_compression | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_envelope_pressure | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_event_spacing_cv | daily | pandas_numpy, polars | event_interval_cv | True | ts | None | 1 | 0 |
| ts_event_spacing_mean | daily | pandas_numpy, polars | event_interval_mean | True | ts | None | 1 | 0 |
| ts_evt_threshold_stability | extended | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_ewm_corr | daily | polars |  | True |  | None | None | None |
| ts_ewm_cov | daily | polars |  | True |  | None | None | None |
| ts_ewm_std | daily | pandas_numpy, polars, sql | ewm_std | True | ts | None | 2 | 0 |
| ts_ewm_var | daily | pandas_numpy, polars, sql | ewm_var | True | ts | None | 2 | 0 |
| ts_expected_shortfall | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_expected_shortfall_asymmetry | daily | pandas_numpy, polars |  | True | ts | None | 6 | 0 |
| ts_expectile | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_expectile_beta | extended | pandas_numpy, polars |  | True | ts | None | 4 | 0 |
| ts_expectile_beta_spread | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_expectile_regression_coeff | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_expectile_regression_coeff_prior | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_expectile_regression_forecast_error | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_expectile_regression_resid | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_extrema_confirmation_rate | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_extrema_divergence_strength | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_extremal_dependence_decay | extended | pandas_numpy, polars |  | False | ts | None | 4 | 0 |
| ts_extremal_index | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_extreme_cluster_ratio | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_extremogram | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_feature_effective_rank | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_feature_mode_share | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_feature_pca_reconstruction_error | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_feature_subspace_rotation | extended | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_ffill_limited | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_fir_lowpass_causal | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_first_passage_bias | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_first_passage_conditional_time | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_first_passage_hit_probability | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_fisher_information_shift | extended | pandas_numpy, polars |  | False | ts | None | 2 | 0 |
| ts_forbidden_ordinal_pattern_excess | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_forbidden_ordinal_pattern_ratio | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_forbidden_ordinal_pattern_signed_excess | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_fractional_difference | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_fractional_difference_discarded_weight_mass | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_gap_fill_ratio | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_gap_reversion_ratio | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_gap_survival_duration | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_garch_next_vol_forecast | extended | pandas_numpy, polars | ts_garch_vol_forecast | False | ts | None | None | 0 |
| ts_garch_persistence | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_garch_standardized_shock | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_garch_vol_surprise | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_generalized_hurst_exponent | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_generalized_hurst_spread_q1_q4 | daily | pandas_numpy, polars | ts_multifractal_width | True | ts | None | 1 | 0 |
| ts_gjr_garch_vol_forecast | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_gjr_leverage | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_glr_mean_shift_score | daily | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_glr_variance_shift_score | daily | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_gpd_shape_pwm | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_h_infinity_level_filter | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_hampel_filter_causal | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_hankel_effective_rank | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_hankel_singular_gap | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_har_from_return_forecast_error_z | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_har_from_return_next_vol | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_har_rv_forecast_error_z | extended | pandas_numpy, polars | ts_har_rv_innovation_z | False | ts | None | None | 0 |
| ts_har_rv_next_var_forecast | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_har_rv_next_vol_forecast | extended | pandas_numpy, polars | ts_har_rv_forecast, ts_har_rv_next_forecast | False | ts | None | None | 0 |
| ts_hartigan_dip | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_higuchi_fractal_dimension | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_hill_tail_index | daily | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_hodges_lehmann_location | extended | pandas_numpy, polars |  | True | ts | None | 8 | 0 |
| ts_hsic | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_huber_regression_coeff | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_huber_regression_coeff_prior | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_huber_regression_forecast_error | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_huber_regression_forecast_error_z | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_huber_regression_in_sample_resid | extended | pandas_numpy, polars | ts_huber_regression_resid | False | ts | None | 1 | 0 |
| ts_huber_regression_predictive_resid | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_huber_regression_resid_z | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_hurst_dfa | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_hvg_assortativity | extended | pandas_numpy, polars |  | True | ts | None | 8 | 0 |
| ts_hvg_clustering_coefficient | extended | pandas_numpy, polars |  | True | ts | None | 8 | 0 |
| ts_hvg_degree_entropy | daily | pandas_numpy, polars |  | True | ts | None | 8 | 0 |
| ts_hvg_forward_backward_asymmetry | daily | pandas_numpy, polars |  | True | ts | None | 8 | 0 |
| ts_hvg_motif_entropy | extended | pandas_numpy, polars |  | True | ts | None | 8 | 0 |
| ts_hysteresis_age | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_hysteresis_state | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_impulse_return | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_impulse_strength | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_impulse_volume | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_industry_liquidity_beta | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_interval_exploration_efficiency | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_interval_nesting_depth | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_interval_occupancy_entropy | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_interval_occupancy_mode_distance | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_interval_overlap_connected_component_ratio | daily | pandas_numpy, polars | ts_interval_overlap_component_ratio | True | ts | None | 1 | 0 |
| ts_interval_union_coverage | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_joint_energy_shift | daily | pandas_numpy, polars |  | True | ts | None | 10 | 1 |
| ts_jump_bipower_proxy | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_kalman_beta | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_kalman_beta_change | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_kalman_beta_uncertainty | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_kalman_innovation_z | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_kalman_level | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_kalman_trend | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_kama | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_kernel_granger_score | extended | pandas_numpy, polars |  | True | ts | None | 30 | 0 |
| ts_km_diffusion_gradient | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_km_equilibrium_distance | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_km_quasipotential_depth | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_kramers_moyal_diffusion | extended | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_kramers_moyal_drift | extended | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_kramers_moyal_local_stability | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_ks_shift | daily | pandas_numpy, polars |  | True | ts | None | 6 | 0 |
| ts_kurt | daily | pandas_numpy, polars, sql | Kurt, TS_KURT, kurt, m_kurt, ts_kurtosis | True | ts | None | 4 | 0 |
| ts_l1_trend_filter_trailing | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_l_kurtosis | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_l_skewness | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_lag_of_peak_corr | daily | pandas_numpy, polars |  | True | ts | None | 6 | 0 |
| ts_lagged_mutual_information | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_last_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_last_pivot_high | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_last_pivot_low | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_lempel_ziv_complexity | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_level_shift_score | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_leverage_effect | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_line_convergence | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_line_parallelism | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_lo_mackinlay_vr | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_lo_mackinlay_z | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_local_lyapunov_exponent | extended | pandas_numpy, polars |  | False | ts | None | 5 | 0 |
| ts_location_shift | daily | pandas_numpy, polars |  | True | ts | None | 6 | 0 |
| ts_log_return | daily | pandas_numpy, polars, sql | log_returns | True | ts | None | 2 | 1 |
| ts_lower_partial_moment | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_lower_tail_coexceedance_probability | daily | pandas_numpy, polars | ts_lower_tail_dependence | True | ts | None | 1 | 0 |
| ts_lz_complexity | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_mad | daily | pandas_numpy, polars, sql | m_mad | True | ts | None | 1 | 0 |
| ts_market_liquidity_beta | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_markov_committor | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_markov_entropy_production | extended | pandas_numpy, polars |  | False | ts | None | 5 | 0 |
| ts_markov_mean_first_passage_time | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_markov_persistence | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_markov_spectral_gap | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_markov_state_entropy | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_markov_stationary_surprisal | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_markov_transition_surprisal | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_mass_concentration | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_matrix_profile_discord_score | extended | pandas_numpy, polars | ts_matrix_profile_motif_distance | False | ts | None | None | 0 |
| ts_matrix_profile_motif_age | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_matrix_profile_motif_frequency | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_matrix_profile_neighbor_dispersion | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_matrix_profile_novelty | daily | pandas_numpy, polars | ts_multivariate_matrix_profile_novelty | True | ts | None | 1 | 0 |
| ts_max | daily | pandas_numpy, polars, sql | Max, TS_MAX, m_max, window_max | True | ts | None | 1 | 0 |
| ts_max_buildup | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_max_chord_excursion | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_max_drawdown | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_max_drawdown_activity_cost | daily | pandas_numpy, polars |  | True | ts | None | 2 | 1 |
| ts_max_if | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_mean | daily | pandas_numpy, polars, sql | Mean, SMA, TS_MEAN, m_avg, ma, mean, move, running_mean, window_mean | True | ts | None | 1 | 0 |
| ts_mean_abs_deviation | extended | pandas_numpy, polars | Mad, mad, ts_mean_absolute_deviation | True | ts | None | 1 | 0 |
| ts_mean_excess_slope | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_mean_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_mean_reversion_half_life | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_mean_reversion_ou_approx_half_life | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_mean_reversion_ou_approx_half_life_prior | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_median | daily | pandas_numpy, polars, sql | Median, m_median, median | True | ts | None | 1 | 0 |
| ts_median3_causal | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_median_abs_deviation | extended | pandas_numpy, polars | ts_median_absolute_deviation | True | ts | None | 1 | 0 |
| ts_min | daily | pandas_numpy, polars, sql | Min, TS_MIN, m_min, window_min | True | ts | None | 1 | 0 |
| ts_min_if | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_mmd_rbf_shift | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_modwt_band_corr | extended | pandas_numpy, polars |  | False | ts | None | 8 | 0 |
| ts_moment | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_monotonicity | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_motif_recurrence_count | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_multi_regression_adjusted_r2_prior | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_multi_regression_coeff | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_multi_regression_coeff_prior | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_multi_regression_coeff_stability | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_multi_regression_forecast_error | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_multi_regression_forecast_error_z | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_multi_regression_r2 | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_multi_regression_r2_prior | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_multi_regression_resid | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_multi_regression_resid_z | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_multifractal_asymmetry | extended | pandas_numpy, polars |  | True | ts | None | 20 | 0 |
| ts_multifractal_curvature | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_multifractal_spectrum_width | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_multiscale_entropy_slope | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_multiscale_permutation_entropy_slope | extended | pandas_numpy, polars |  | False | ts | None | 5 | 0 |
| ts_multiscale_trend_consensus | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_multiscale_trend_curvature | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_multiscale_trend_dispersion | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_mutual_information | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_nearest_structural_level_distance | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_negative_ratio | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_new_high | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_new_low | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_nth_pivot_high | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_nth_pivot_high_age | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_nth_pivot_low | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_nth_pivot_low_age | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_nth_value | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_opening_mispricing_score | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ordinal_irreversibility | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_overnight_intraday_cov | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_overnight_intraday_sign_agreement | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_overnight_intraday_spread | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_partial_corr | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 0 |
| ts_pastor_stambaugh_liquidity_gamma | extended | pandas_numpy, polars |  | True | ts | None | 20 | 0 |
| ts_path_efficiency | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_path_leadlag_area | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_path_signature_area | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_path_signature_depth2_norm | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_pattern_symmetry | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_pct | daily | pandas_numpy, polars, sql | TS_PCT, m_pct_change, pct_change, returns, ts_return | True | ts | None | 1 | 1 |
| ts_permutation_entropy | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_permutation_transition_entropy | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_persistence_birth_dispersion | extended | pandas_numpy, polars | ts_betti_crocker_bifurcation_score | False | ts | None | 8 | 0 |
| ts_persistence_diagram_shift | extended | pandas_numpy, polars |  | False | ts | None | 2 | 0 |
| ts_persistence_entropy_h0 | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_persistence_entropy_h1 | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_pettitt_change_score | extended | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_pickands_tail_index | extended | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_pivot_high_age | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_pivot_high_count | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_pivot_high_spacing | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_pivot_low_age | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_pivot_low_count | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_pivot_low_spacing | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_poly2_coeff | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_poly2_forecast_error | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_poly2_forecast_error_z | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_poly2_prior_coeff | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_poly2_resid | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_positive_ratio | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_prev_high | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_prev_low | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_price_delay | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_product | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_pseudocount_sample_entropy | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_qn_scale | daily | pandas_numpy, polars |  | True | ts | None | 8 | 0 |
| ts_quantile | daily | pandas_numpy, polars, sql | Percentile, TS_QUANTILE, m_percentile, percentile | True | ts | None | 1 | 0 |
| ts_quantile_beta_spread | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_quantile_beta_spread_prior | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_quantile_crossing_spectral_concentration | extended | pandas_numpy, polars |  | False | ts | None | 8 | 0 |
| ts_quantile_if | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_quantile_kurtosis | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_quantile_range | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_quantile_regression_beta | research | pandas_numpy, polars |  | False | ts | None | 5 | 0 |
| ts_quantile_regression_coeff | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_quantile_regression_coeff_prior | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_quantile_regression_resid | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_quantile_regression_slope | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_quantile_skew | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_quantile_transport_curvature | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_quantile_transport_slope | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_quantilogram | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_range_expansion | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_rank | daily | pandas_numpy, polars, sql | TS_RANK, m_rank | True | ts | None | 1 | 0 |
| ts_rank_if | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_ratio | daily | pandas_numpy, polars, sql | ratios | True | ts | None | 2 | 1 |
| ts_realized_quarticity | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 0 |
| ts_recovery_fraction | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_recurrence_determinism | daily | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_recurrence_diagonal_entropy | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_recurrence_divergence | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_recurrence_laminarity | daily | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_recurrence_longest_vertical_length | extended | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_recurrence_mean_diagonal_length | extended | pandas_numpy, polars |  | True | ts | None | 10 | 0 |
| ts_recurrence_rate | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_recurrence_trapping_time | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_regime_duration | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_regression_forecast_error | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_regression_forecast_error_z | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_regression_in_sample_resid | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_regression_intercept | daily | pandas_numpy, polars | intercept | True | ts | None | 3 | 0 |
| ts_regression_r2 | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_regression_resid | daily | pandas_numpy, polars | rolling_residual | True | ts | None | 3 | 0 |
| ts_regression_resid_if | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_regression_resid_mean | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_regression_slope | daily | pandas_numpy, polars, sql | TS_REGRESSION_SLOPE, ts_regression | True | ts | None | 3 | 0 |
| ts_regression_tstat | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 0 |
| ts_residualized_hsic | extended | pandas_numpy, polars |  | True | ts | None | 24 | 0 |
| ts_resistance_break | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_resistance_fit_r2 | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_resistance_level | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_resistance_log_slope | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_resistance_slope | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_response_slope_asymmetry | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_return_spectral_entropy | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ridge_regression_coeff | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_ridge_regression_coeff_prior | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ridge_regression_forecast_error | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ridge_regression_forecast_error_z | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ridge_regression_in_sample_resid | extended | pandas_numpy, polars | ts_ridge_regression_resid | False | ts | None | 1 | 0 |
| ts_ridge_regression_predictive_resid | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_ridge_regression_resid_z | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_robust_ema | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_robust_zscore_inclusive | extended | pandas_numpy, polars | ts_robust_zscore | False | ts | None | 1 | 0 |
| ts_robust_zscore_prior | extended | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_roll_effective_spread | extended | pandas_numpy, polars |  | False | ts | None | 5 | 0 |
| ts_rolling_median_causal | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_rolling_sr_gaussian_mean_shift_score | extended | pandas_numpy, polars | ts_sr_gaussian_mean_shift_score | True | ts | None | 12 | 0 |
| ts_roughness | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_rqa_determinism_fixed_rr | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_rqa_laminarity_fixed_rr | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_run_concentration | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_run_efficiency | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_run_strength | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_sample_entropy | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_scale_shift | daily | pandas_numpy, polars |  | True | ts | None | 6 | 0 |
| ts_score_rank_weighted_mean | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_semivariance_balance | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_sharpe | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_sign_cluster_index | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_sign_persistence | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_signature_mahalanobis_anomaly | extended | pandas_numpy, polars |  | False | ts | None | 5 | 0 |
| ts_skew | daily | pandas_numpy, polars, sql | Skew, TS_SKEW, m_skew, skew, ts_skewness | True | ts | None | 3 | 0 |
| ts_sma_cn | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_spectral_centroid | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_spectral_entropy | daily | pandas_numpy, polars |  | True | ts | None | 16 | 0 |
| ts_spectral_flatness | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_spectral_low_frequency_ratio | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_spectral_lowpass_trailing | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_spectral_peak_concentration | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_spectral_quality_factor | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ssa_denoise_trailing | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_ssa_prior_reconstruction_error | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_ssa_reconstruction_residual | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_staleness | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_state_age_percentile | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_state_density | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_state_entry_strength | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_state_exit_hazard | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_state_integral | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_state_residual_life | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_std | daily | pandas_numpy, polars, sql | Std, TS_STD, m_std, running_std, std, std_n, ts_std_dev, ts_stddev, window_std | True | ts | None | 2 | 0 |
| ts_std_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_stratified_mean_spread | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_structural_level_density | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_structural_level_strength | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_student_t_kalman_filter | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_sum | daily | pandas_numpy, polars, sql | Sum, TS_SUM, m_sum, running_sum, sum, sum_n, window_sum | True | ts | None | 1 | 0 |
| ts_sum_decay | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_sum_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_super_smoother | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_support_break | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_support_fit_r2 | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_support_level | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_support_log_slope | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_support_slope | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_swing_amplitude | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_swing_amplitude_atr | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_swing_amplitude_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_swing_duration | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_swing_velocity | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_tail_imbalance | daily | pandas_numpy, polars |  | True | ts | None | 4 | 0 |
| ts_tail_mean | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_tail_ratio | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_threshold_cycle_asymmetry | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_threshold_cycle_period | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_time_since_change | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_time_slope | daily | pandas_numpy, polars | Slope, TS_TIME_SLOPE, slope | True | ts | None | 2 | 0 |
| ts_time_under_water | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_topk_mean | daily | pandas_numpy, polars, sql | tm_top_n_avg, ts_top_n_avg | True | ts | None | 1 | 0 |
| ts_topk_std | daily | pandas_numpy, polars, sql | ts_top_n_std | True | ts | None | 2 | 0 |
| ts_topk_sum | daily | pandas_numpy, polars, sql | TS_TOPK_SUM, m_top_n_sum, tm_top_n_sum | True | ts | None | 1 | 0 |
| ts_total_variation_filter_trailing | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_transfer_entropy | extended | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_transfer_entropy_peak_excess | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_transfer_entropy_peak_lag | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_transfer_entropy_peak_strength | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_transition_count | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_transition_intensity | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_trend_break_score | daily | pandas_numpy, polars |  | True | ts | None | 4 | 0 |
| ts_trend_tstat | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 0 |
| ts_trimmed_mean | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_true_streak | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_turning_intensity | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_turning_point_ratio | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_turning_rate | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_turnover_age_dispersion | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_turnover_cost_dispersion | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_turnover_cost_entropy | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_turnover_cost_entropy_vol_scaled | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_turnover_cost_mode_distance | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_turnover_cost_quantile_distance | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_turnover_cost_skew | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_turnover_holding_age | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_turnover_near_cost_mass | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_turnover_old_mass | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_turnover_profit_share | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_turnover_reference_price | daily | pandas_numpy, polars |  | True | ts | None | 5 | 1 |
| ts_two_state_regime_probability | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_upper_partial_moment | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_upper_tail_coexceedance_probability | daily | pandas_numpy, polars | ts_upper_tail_dependence | True | ts | None | 1 | 0 |
| ts_upside_deviation | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_valid_count | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_value_at_argextreme | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_var | daily | pandas_numpy, polars, sql | Var, m_var, var | True | ts | None | 2 | 0 |
| ts_variance_ratio_proxy | extended | pandas_numpy, polars | ts_variance_ratio | True | ts | None | 1 | 0 |
| ts_variance_ratio_slope | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_variogram_slope | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_vector_path_curvature | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_vector_path_efficiency | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_vector_self_intersection_rate | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_vector_state_local_density | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_vector_state_mahalanobis | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_vector_turning_coherence | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_vol_acceleration | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_vol_clustering | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_vol_of_vol | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_vol_pvariation_roughness | daily | pandas_numpy, polars | ts_pvariation_scaling_exponent | True | ts | None | 1 | 0 |
| ts_vol_scaling_break | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_vol_shift_score | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_vol_term_structure | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_wasserstein_shift | daily | pandas_numpy, polars |  | True | ts | None | 6 | 0 |
| ts_wavelet_energy_slope | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_wavelet_entropy | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_wavelet_high_frequency_ratio | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_wavelet_low_frequency_ratio | extended | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_wavelet_lowpass_reconstruct | extended | pandas_numpy, polars |  | False | ts | None | 8 | 0 |
| ts_wavelet_shrinkage_trailing | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_weighted_downside_deviation | extended | pandas_numpy, polars | ts_weighted_semivariance_sqrt | True | ts | None | 1 | 0 |
| ts_weighted_drawdown_area | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_weighted_expected_shortfall | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ts_weighted_permutation_entropy | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_weighted_semivariance | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_weighted_standardized_moment | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_weighted_time_centroid | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ts_zero_ratio | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_zscore | daily | pandas_numpy, polars, sql | m_zscore | True | ts | None | 1 | 0 |
| ttm_from_cumulative | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| ttm_from_quarterly | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| turnover_acceleration | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| turnover_adjusted_volatility | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| turnover_autocorr | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| turnover_chip_age_cost_surface | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| turnover_chip_overhang_surface | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| turnover_momentum | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| turnover_shock | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| turnover_volatility | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| turnover_zscore | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ulcer_index | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| unitize | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| up_down_volume_ratio | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| up_volume_ratio | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| update_acceleration | extended | pandas_numpy, polars |  | True | ts | None | 4 | 0 |
| update_direction_persistence | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| update_path_efficiency | extended | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| update_surprise | extended | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| val1_discount_regime_share | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_earnings_yield_ma_diff | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_earnings_yield_persistence | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_earnings_yield_slope | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_equity_yield_dispersion | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_fcf_yield_growth | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_pe_beta_to_market | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_qmj_quality_rank | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_relative_valuation_gap | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_valuation_percentile_own | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_valuation_stat_spread | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_valuation_z_own | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| val1_valuations_lag_component | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| valuation_cashflow_disagreement | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| valuation_growth_mismatch | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| valuation_pcf_definition_gap | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| valuation_pcf_gap_positive | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| valuation_pcf_gap_signed_log | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| valuation_pe_gap_positive | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| valuation_pe_gap_signed_log | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| valuation_pe_ttm_lyr_gap | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| valuation_quality_mismatch | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| vax_liquidity_adjusted_return | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vax_liquidity_penalty_exposure | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vax_ret_per_liquidity_unit | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| volume_acceleration | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| volume_autocorr | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| volume_momentum | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| volume_price_range_density | daily | pandas_numpy, polars, sql | volume_to_range | True | ts | None | 1 | 0 |
| volume_shock | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| volume_volatility | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| volume_weighted_momentum | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| volume_weighted_return | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| volume_zscore | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| vpin_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| vr1_ewma_range_vol | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vr1_garman_klass_ext | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vr1_parkinson_close_scale | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vr1_range_everage | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vr1_range_to_close_eff | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vr1_rogers_satchell | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vv1_dispersion_vol | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vv1_downside_vol_share | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vv1_fractional_share | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vv1_long_short_vol_beta | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vv1_regime_change_ratio | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vv1_vol_acceleration | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vv1_vol_level_score | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vv1_vol_of_vol | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| vwap_deviation | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| vwap_distance_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| vwap_premium_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| vwap_slope_pct | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| vwap_to_close_return | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| wavelet_detail_energy_ratio | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| where | daily | pandas_numpy, polars, sql | IIF, WHERE, if, if_else, iif | True | elementwise | None | None | 0 |
| winsorize | daily | pandas_numpy, polars, sql | WINSORIZE, c_winsorize | True | cs | None | None | 0 |
| winsorize_mean | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| yang_zhang_vol | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| yoy_by_period | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 5 | 0 |
| zero_return_ratio | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| zmijewski_score | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| zscore | daily | pandas_numpy, polars, sql | CS_ZSCORE, ZSCORE, c_zscore, cs_zscore, panel_standardize, panel_zscore, standardize | True | cs | None | None | 0 |
