# Operators Catalog（自动生成）

> 从 `load_all()` 去重后的最终 runtime registry 生成。
> 日常因子 DSL 仅使用 `surface=daily`；其他工具见 `research_operators/`。

## 摘要

- canonical 总数：220
- daily：180
- research：24
- unsafe：0
- legacy：1

| canonical | surface | backends | aliases | pit_safe | scope | lookback | min_periods | lag |
|---|---|---|---|---|---|---|---|---|
| ADX | daily | pandas_numpy, polars | ts_adx | True | ts | None | None | 0 |
| ATR_WILDER | daily | pandas_numpy, polars, sql | ts_atr_wilder | True | ts | None | 2 | 0 |
| MACD_hist | daily | pandas_numpy, polars |  | True | ts | None | None | 0 |
| MACD_line | daily | pandas_numpy, polars | MACD | True | ts | None | None | 0 |
| MACD_signal | daily | pandas_numpy, polars |  | True | ts | None | None | 0 |
| RSI_WILDER | daily | pandas_numpy, polars, sql | ts_rsi_wilder | True | ts | None | 2 | 0 |
| abs | daily | pandas_numpy, polars, sql | ABS | True | elementwise | None | None | 0 |
| acos | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| add | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| and_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| arg | daily | pandas_numpy |  | False | elementwise | None | None | 0 |
| asin | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| atan | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| atan2 | daily | pandas_numpy |  | False | elementwise | None | None | 0 |
| cbrt | daily | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| ceil | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| clip | daily | pandas_numpy, polars, sql | CLIP, cap, clamp, clip | True | elementwise | None | None | 0 |
| coalesce | daily | pandas_numpy, polars, sql | COALESCE | True | elementwise | None | None | 0 |
| constant | internal | pandas_numpy |  | False | elementwise | None | None | 0 |
| cos | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cosh | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| coskewness_to_market | research | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| cot | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cs_bucket | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_count | daily | pandas_numpy, polars, sql | c_count | False | cs | 0 | 1 | 0 |
| cs_demean | daily | pandas_numpy, polars, sql | CS_DEMEAN, c_demean | True | cs | None | None | 0 |
| cs_fill_mean | daily | pandas_numpy, polars |  | True | cs | 0 | 1 | 0 |
| cs_fill_median | daily | pandas_numpy, polars |  | True | cs | 0 | 1 | 0 |
| cs_mad | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_mad_zscore | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_mean | daily | pandas_numpy, polars, sql | c_mean | False | cs | 0 | 1 | 0 |
| cs_multi_resid | daily | pandas_numpy, sql |  | True | cs | None | None | 0 |
| cs_neutralize | daily | pandas_numpy |  | True | cs | None | None | 0 |
| cs_pct_rank | daily | pandas_numpy, polars, sql | rank_pct | True | cs | None | None | 0 |
| cs_quantile | daily | pandas_numpy, sql | c_percentile, quantile | True | cs | None | None | 0 |
| cs_rank_gaussian | daily | pandas_numpy, polars |  | True | cs | 0 | 1 | 0 |
| cs_regression | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_resid | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_std | daily | pandas_numpy, polars, sql | c_std | False | cs | 0 | 1 | 0 |
| cs_sum | daily | pandas_numpy, polars, sql | c_sum | False | cs | 0 | 1 | 0 |
| cs_weighted_demean | daily | pandas_numpy, polars |  | True | cs | 0 | 1 | 0 |
| cs_weighted_mean | daily | pandas_numpy, polars |  | True | cs | 0 | 1 | 0 |
| cs_weighted_zscore | daily | pandas_numpy, polars |  | True | cs | 0 | 1 | 0 |
| cs_wls_resid | daily | pandas_numpy, sql |  | True | cs | None | None | 0 |
| csc | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cube | legacy | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| digital_count | research | pandas_numpy, polars |  | False | ts | None | None | 0 |
| divide | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| downside_beta | research | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| eq | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ewm | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ewm_corr | daily | pandas_numpy, sql |  | True | ts | None | 1 | 0 |
| ewm_cov | daily | pandas_numpy, sql |  | False | elementwise | None | None | 0 |
| ewm_std | daily | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| ewm_var | daily | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| exp | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| exp_neg | extended | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ffill_limit | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| fillna_const | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| fix | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| flex_max | extended | pandas_numpy, polars | max | True | elementwise | None | None | 0 |
| flex_min | extended | pandas_numpy, polars | min | True | elementwise | None | None | 0 |
| floor | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| fundamental_staleness | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| ge | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| group_count | daily | pandas_numpy, polars, sql |  | False | cs | 0 | 1 | 0 |
| group_decay_linear | research | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_max | daily | pandas_numpy, polars, sql |  | False | cs | 0 | 1 | 0 |
| group_mean | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_min | daily | pandas_numpy, polars, sql |  | False | cs | 0 | 1 | 0 |
| group_neutralize | daily | pandas_numpy, polars, sql | INDUSTRY_NEUTRAL, INDUSTRY_NEUTRALIZE, IND_NEUTRALIZE, NEUTRALIZE, c_neutralize, group_demean, ind_neutralize, industry_neutral, industry_neutralize, neutralize, panel_neutralize | True | cs | None | None | 0 |
| group_normalize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_percentile | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_rank | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_std | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_sum | daily | pandas_numpy, polars, sql |  | False | cs | 0 | 1 | 0 |
| group_weighted_mean | daily | pandas_numpy, polars |  | True | cs | 0 | 1 | 0 |
| group_weighted_zscore | daily | pandas_numpy, polars |  | True | cs | 0 | 1 | 0 |
| group_winsorize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_zscore | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| gt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| identity | internal | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| idio_skew | research | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| idio_vol | research | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| intercept | daily | pandas_numpy, polars | Intercept | False | aggregate | None | 1 | 0 |
| intraday_vwap_deviation | research | pandas_numpy |  | True | ts | None | 1 | 0 |
| inverse | daily | pandas_numpy, polars, sql | inv, reciprocal | True | elementwise | None | None | 0 |
| is_finite | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| is_infinite | daily | pandas_numpy, polars, sql | IS_INFINITE, is_inf | False | elementwise | None | None | 0 |
| is_nan | daily | pandas_numpy, polars, sql | IS_NAN | True | elementwise | None | None | 0 |
| is_not_null | daily | pandas_numpy, polars, sql | IS_NOT_NULL, is_not_null | True | elementwise | None | None | 0 |
| is_null | daily | pandas_numpy, polars, sql | IS_NULL, is_null | True | elementwise | None | None | 0 |
| le | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| lerp | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| log | daily | pandas_numpy, polars, sql | LOG, ln | True | elementwise | None | None | 0 |
| log10 | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| log2 | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| log_abs | daily | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| lt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| maximum | daily | pandas_numpy, polars, sql | fmax | True | elementwise | None | None | 0 |
| micro_amihud_hf | research | pandas_numpy, polars |  | True | ts | None | None | 0 |
| micro_bipower_var | research | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| micro_jump_indicator | research | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| micro_kyle_lambda | research | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| micro_mid_return | research | pandas_numpy, polars |  | True | ts | None | None | 1 |
| micro_realized_vol | research | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| micro_spread | research | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| micro_trade_imbalance | research | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| micro_vpin | research | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| minimum | daily | pandas_numpy, polars, sql | fmin | True | elementwise | None | None | 0 |
| multiply | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ne | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| neg | daily | pandas_numpy, polars, sql | negate, reverse | True | elementwise | None | None | 0 |
| normalize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| not_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| or_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| period_average | daily | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| period_cagr | daily | pandas_numpy |  | True | fundamental_period | None | 2 | 0 |
| period_change | daily | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| period_lag | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| period_stability | daily | pandas_numpy, polars |  | True | fundamental_period | None | 2 | 0 |
| power | daily | pandas_numpy, polars, sql | POWER, pow | True | elementwise | None | None | 0 |
| price_spread_deviation | daily | pandas_numpy |  | False | ts | None | None | 0 |
| protected_div | internal | pandas_numpy, sql |  | True | elementwise | None | None | 0 |
| quarter_from_cumulative | daily | pandas_numpy | quarter_from_cumulative | True | ts | None | 1 | 0 |
| rank | daily | pandas_numpy, sql | CS_RANK, RANK, c_rank, cs_rank, cs_rank_01, panel_rank | True | cs | None | None | 0 |
| rank_corr | research | pandas_numpy, polars | RANKCORR, RANK_CORR, rankcorr | True | ts | None | 2 | 0 |
| real_turnover_rate | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| residual_momentum_capm | research | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| revision_delta | daily | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| rolling_beta_to_market | research | pandas_numpy, polars | FP_BETA, ROLLING_BETA_TO_MARKET, fp_beta | True | ts | None | 2 | 0 |
| round | extended | pandas_numpy, polars | ROUND | False | elementwise | None | None | 0 |
| safe_div_null | daily | pandas_numpy, polars, sql | div_or_null, safe_div, safe_div_null | True | elementwise | None | None | 0 |
| saturate | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| scale | daily | pandas_numpy, polars, sql | SCALE, c_scale | True | cs | None | None | 0 |
| sec | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| sigmoid | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| sign | daily | pandas_numpy, polars, sql | SIGN | True | elementwise | None | None | 0 |
| signed_log | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| signed_power | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| signed_sqrt | daily | pandas_numpy, sql |  | True | elementwise | None | None | 0 |
| sin | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| sinh | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| size_neutralize | daily | pandas_numpy | CAP_NEUTRALIZE, MARKET_CAP_NEUTRALIZE, SIZE_NEUTRALIZE, cap_neutralize, market_cap_neutralize | False | cs | 0 | 1 | 0 |
| sqrt | daily | pandas_numpy, polars, sql | SQRT | True | elementwise | None | None | 0 |
| sqrt_abs | extended | pandas_numpy |  | False | elementwise | None | None | 0 |
| square | extended | pandas_numpy, polars | sqr | True | elementwise | None | None | 0 |
| subtract | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| tail_beta | research | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| tan | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| tanh | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| trade_when | research | pandas_numpy, polars |  | False | ts | None | None | 0 |
| true_range | daily | pandas_numpy, polars |  | True | ts | None | None | 0 |
| truncate | extended | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| ts_argmax | daily | pandas_numpy, polars, sql | m_argmax, ts_arg_max | True | ts | None | 1 | 0 |
| ts_argmin | daily | pandas_numpy, polars, sql | m_argmin, ts_arg_min | True | ts | None | 1 | 0 |
| ts_autocorr | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 1 |
| ts_beta | daily | pandas_numpy, polars, sql | beta, m_beta, rolling_beta | True | ts | None | 2 | 0 |
| ts_bottomk_mean | daily | pandas_numpy, polars | ts_bottom_n_avg | True | ts | None | 1 | 0 |
| ts_bottomk_std | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_bottomk_sum | daily | pandas_numpy, polars | ts_bottom_n_sum | True | ts | None | 1 | 0 |
| ts_corr | daily | pandas_numpy, polars, sql | Corr, TS_CORR, corr, correlation, m_cor, ts_correlation | True | ts | None | 1 | 0 |
| ts_count_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_cov | daily | pandas_numpy, polars, sql | Cov, Covariance, TS_COV, cov, m_cov, ts_covariance | True | ts | None | 1 | 0 |
| ts_days_since | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_decay_exp_window | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_decay_linear | daily | pandas_numpy, polars, sql | DECAY_LINEAR, TS_DECAY_LINEAR, WMA, decay_linear, ts_decay, ts_decay_linear | True | ts | None | 1 | 0 |
| ts_delay | daily | pandas_numpy, polars, sql | DELAY, Delay, Ref, delay, m_delay, prev, shift, ts_delay | True | ts | None | None | 1 |
| ts_delta | daily | pandas_numpy, polars, sql | Delta, Diff, TS_DELTA, delta, deltas | True | ts | None | None | 1 |
| ts_ema | daily | pandas_numpy, polars, sql | EMA, ema, ewm_mean, ts_ema | True | ts | None | 1 | 0 |
| ts_kurt | daily | pandas_numpy, polars | Kurt, TS_KURT, kurt, m_kurt, ts_kurtosis | False | ts | None | None | 0 |
| ts_last_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_log_return | daily | pandas_numpy, sql | log_returns | True | ts | None | 2 | 1 |
| ts_mad | daily | pandas_numpy, polars, sql | Mad, m_mad, mad | True | ts | None | None | 0 |
| ts_max | daily | pandas_numpy, polars, sql | Max, TS_MAX, m_max, window_max | True | ts | None | 1 | 0 |
| ts_max_buildup | daily | pandas_numpy |  | False | ts | None | None | 0 |
| ts_max_drawdown | daily | pandas_numpy, sql |  | True | ts | None | 2 | 0 |
| ts_mean | daily | pandas_numpy, polars, sql | Mean, SMA, TS_MEAN, m_avg, ma, mean, move, running_mean, window_mean | True | ts | None | 1 | 0 |
| ts_mean_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_median | daily | pandas_numpy, polars, sql | Median, m_median, median | True | ts | None | 1 | 0 |
| ts_min | daily | pandas_numpy, polars, sql | Min, TS_MIN, m_min, window_min | True | ts | None | 1 | 0 |
| ts_moment | daily | pandas_numpy |  | False | ts | None | None | 0 |
| ts_nth_value | daily | pandas_numpy, sql |  | True | ts | None | 1 | 0 |
| ts_partial_corr | daily | pandas_numpy, sql |  | True | ts | None | 3 | 0 |
| ts_pct | daily | pandas_numpy, polars, sql | TS_PCT, m_pct_change, pct_change, returns, ts_return | True | ts | None | None | 1 |
| ts_poly2_coeff | research | pandas_numpy |  | True | ts | None | 3 | 0 |
| ts_poly2_resid | research | pandas_numpy |  | True | ts | None | 3 | 0 |
| ts_product | daily | pandas_numpy, polars, sql |  | True | ts | None | None | 0 |
| ts_quantile | daily | pandas_numpy, polars, sql | Percentile, TS_QUANTILE, m_percentile, percentile | False | ts | None | None | 0 |
| ts_rank | daily | pandas_numpy, polars, sql | TS_RANK, m_rank | True | ts | None | 1 | 0 |
| ts_ratio | daily | pandas_numpy | ratios | False | ts | None | None | 0 |
| ts_regression_intercept | daily | pandas_numpy |  | True | ts | None | 3 | 0 |
| ts_regression_r2 | daily | pandas_numpy |  | True | ts | None | 3 | 0 |
| ts_regression_resid | daily | pandas_numpy | rolling_residual | True | ts | None | 3 | 0 |
| ts_regression_slope | daily | pandas_numpy, sql | TS_REGRESSION_SLOPE, ts_regression | True | ts | None | 3 | 0 |
| ts_regression_tstat | daily | pandas_numpy, sql |  | True | ts | None | 3 | 0 |
| ts_sharpe | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_skew | daily | pandas_numpy, polars, sql | Skew, TS_SKEW, m_skew, skew, ts_skewness | False | ts | None | None | 0 |
| ts_std | daily | pandas_numpy, polars, sql | Std, TS_STD, m_std, running_std, std, std_n, ts_std_dev, ts_stddev, window_std | True | ts | None | 1 | 0 |
| ts_std_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_sum | daily | pandas_numpy, polars, sql | Sum, TS_SUM, m_sum, running_sum, sum, sum_n, window_sum | True | ts | None | 1 | 0 |
| ts_sum_decay | research | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_sum_if | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_tail_mean | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| ts_time_slope | daily | pandas_numpy, polars, sql | Slope, TS_TIME_SLOPE, slope | True | ts | None | 2 | 0 |
| ts_topk_mean | daily | pandas_numpy, polars | tm_top_n_avg, ts_top_n_avg | True | ts | None | 1 | 0 |
| ts_topk_std | daily | pandas_numpy, polars | ts_top_n_std | True | ts | None | 1 | 0 |
| ts_topk_sum | daily | pandas_numpy, polars | TS_TOPK_SUM, m_top_n_sum, tm_top_n_sum | True | ts | None | 1 | 0 |
| ts_trend_tstat | daily | pandas_numpy, sql |  | True | ts | None | 3 | 0 |
| ts_true_streak | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_var | daily | pandas_numpy, polars, sql | Var, m_var, var | True | ts | None | 1 | 0 |
| ts_zscore | daily | pandas_numpy, polars, sql | m_zscore | True | ts | None | 1 | 0 |
| ttm_from_cumulative | daily | pandas_numpy | ttm_from_cumulative | True | ts | None | 1 | 0 |
| ttm_from_quarterly | daily | pandas_numpy | ttm_from_quarterly | True | ts | None | 1 | 0 |
| unitize | daily | pandas_numpy |  | False | elementwise | None | None | 0 |
| where | daily | pandas_numpy, polars, sql | IIF, WHERE, if, if_else, iif | True | elementwise | None | None | 0 |
| winsorize | daily | pandas_numpy, polars, sql | WINSORIZE, c_winsorize | True | cs | None | None | 0 |
| winsorize_mean | daily | pandas_numpy |  | False | elementwise | None | None | 0 |
| yoy_by_period | daily | pandas_numpy | yoy_by_period | True | ts | None | 5 | 0 |
| zscore | daily | pandas_numpy, polars, sql | CS_ZSCORE, ZSCORE, c_zscore, cs_zscore, panel_standardize, panel_zscore, standardize | True | cs | None | None | 0 |
