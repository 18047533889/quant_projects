# Operators Catalog（自动生成）

> 从 `load_all()` 去重后的最终 runtime registry 生成。
> 日常因子 DSL 仅使用 `surface=daily`；其他工具见 `research_operators/`。

## 摘要

- canonical 总数：205
- daily：86
- research：14
- unsafe：0
- legacy：1

| canonical | surface | backends | aliases | pit_safe | scope | lookback | min_periods | lag |
|---|---|---|---|---|---|---|---|---|
| ADX | extended | pandas_numpy, polars | ts_adx | True | ts | None | None | 0 |
| ATR_WILDER | extended | pandas_numpy, polars, sql | ts_atr_wilder | True | ts | None | 2 | 0 |
| MACD_hist | extended | pandas_numpy, polars |  | True | ts | None | None | 0 |
| MACD_line | extended | pandas_numpy, polars | MACD | True | ts | None | None | 0 |
| MACD_signal | extended | pandas_numpy, polars |  | True | ts | None | None | 0 |
| RSI_WILDER | extended | pandas_numpy, polars, sql | ts_rsi_wilder | True | ts | None | 2 | 0 |
| abs | daily | pandas_numpy, polars, sql | ABS | True | elementwise | None | None | 0 |
| acos | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| add | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| and_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| arg | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| asin | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| atan | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| atan2 | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| cbrt | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ceil | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| clip | daily | pandas_numpy, polars, sql | CLIP, cap, clamp | True | elementwise | None | None | 0 |
| coalesce | daily | pandas_numpy, polars, sql | COALESCE | True | elementwise | None | None | 0 |
| constant | internal | pandas_numpy |  | True | elementwise | None | None | 0 |
| cos | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| cosh | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| coskewness_to_market | research | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| cot | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| cs_bucket | extended | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_count | daily | pandas_numpy, polars, sql | c_count | True | cs | None | None | 0 |
| cs_demean | daily | pandas_numpy, polars, sql | CS_DEMEAN, c_demean | True | cs | None | None | 0 |
| cs_fill_mean | extended | pandas_numpy, polars |  | True | cs | 0 | 1 | 0 |
| cs_fill_median | extended | pandas_numpy, polars |  | True | cs | 0 | 1 | 0 |
| cs_mad | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_mad_zscore | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_mean | daily | pandas_numpy, polars, sql | c_mean | True | cs | None | None | 0 |
| cs_multi_resid | extended | pandas_numpy, sql |  | True | cs | None | None | 0 |
| cs_neutralize | extended | pandas_numpy |  | True | cs | None | None | 0 |
| cs_pct_rank | daily | pandas_numpy, polars, sql | rank_pct | True | cs | None | None | 0 |
| cs_quantile | extended | pandas_numpy, sql | c_percentile, quantile | True | cs | None | None | 0 |
| cs_rank_gaussian | extended | pandas_numpy |  | True | cs | 0 | 1 | 0 |
| cs_regression | extended | pandas_numpy, sql |  | True | cs | None | None | 0 |
| cs_resid | extended | pandas_numpy, sql |  | True | cs | None | None | 0 |
| cs_std | daily | pandas_numpy, polars, sql | c_std | True | cs | None | None | 0 |
| cs_sum | daily | pandas_numpy, polars, sql | c_sum | True | cs | None | None | 0 |
| cs_weighted_demean | extended | pandas_numpy |  | True | cs | 0 | 1 | 0 |
| cs_weighted_mean | extended | pandas_numpy |  | True | cs | 0 | 1 | 0 |
| cs_weighted_zscore | extended | pandas_numpy |  | True | cs | 0 | 1 | 0 |
| cs_wls_resid | extended | pandas_numpy, sql |  | True | cs | None | None | 0 |
| csc | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| cube | legacy | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| digital_count | research | pandas_numpy, polars |  | True | ts | None | 1 | 1 |
| divide | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| eq | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| exp | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| exp_neg | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| ffill_limit | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| fillna_const | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| fix | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| flex_max | extended | pandas_numpy, polars | max | True | elementwise | None | None | 0 |
| flex_min | extended | pandas_numpy, polars | min | True | elementwise | None | None | 0 |
| floor | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| fundamental_staleness | extended | pandas_numpy, polars |  | True | fundamental_period | None | 1 | 0 |
| ge | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| group_count | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_decay_linear | research | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_max | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_mean | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_min | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_neutralize | daily | pandas_numpy, polars, sql | INDUSTRY_NEUTRAL, INDUSTRY_NEUTRALIZE, IND_NEUTRALIZE, NEUTRALIZE, group_demean, ind_neutralize, industry_neutral, industry_neutralize, neutralize | True | cs | None | None | 0 |
| group_normalize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_percentile | extended | pandas_numpy, sql |  | True | cs | None | None | 0 |
| group_rank | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_std | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_sum | daily | pandas_numpy, polars, sql |  | True | group | None | None | 0 |
| group_weighted_mean | extended | pandas_numpy |  | True | cs | 0 | 1 | 0 |
| group_weighted_zscore | extended | pandas_numpy |  | True | cs | 0 | 1 | 0 |
| group_winsorize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_zscore | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| gt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| identity | internal | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| idio_skew | research | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| intraday_vwap_deviation | research | pandas_numpy |  | True | ts | None | 1 | 0 |
| inverse | daily | pandas_numpy, polars, sql | inv, reciprocal | True | elementwise | None | None | 0 |
| is_finite | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| is_infinite | daily | pandas_numpy, polars, sql | IS_INFINITE, is_inf | True | elementwise | None | None | 0 |
| is_nan | extended | pandas_numpy, polars, sql | IS_NAN | True | elementwise | None | None | 0 |
| is_not_null | daily | pandas_numpy, polars, sql | IS_NOT_NULL | True | elementwise | None | None | 0 |
| is_null | daily | pandas_numpy, polars, sql | IS_NULL | True | elementwise | None | None | 0 |
| le | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| lerp | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| log | daily | pandas_numpy, polars, sql | LOG, ln | True | elementwise | None | None | 0 |
| log10 | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| log2 | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| log_abs | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| lt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| maximum | daily | pandas_numpy, polars, sql | fmax | True | elementwise | None | None | 0 |
| minimum | daily | pandas_numpy, polars, sql | fmin | True | elementwise | None | None | 0 |
| multiply | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ne | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| neg | daily | pandas_numpy, polars, sql | reverse | True | elementwise | None | None | 0 |
| normalize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| not_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| or_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| period_average | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| period_cagr | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 2 | 0 |
| period_change | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| period_lag | extended | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| period_stability | extended | pandas_numpy |  | True | fundamental_period | None | 2 | 0 |
| power | daily | pandas_numpy, polars, sql | POWER, pow | True | elementwise | None | None | 0 |
| price_spread_deviation | extended | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| protected_div | internal | pandas_numpy, sql |  | True | elementwise | None | None | 0 |
| quarter_from_cumulative | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| rank | daily | pandas_numpy, polars, sql | CS_RANK, RANK, c_rank, cs_rank, cs_rank_01 | True | cs | None | None | 0 |
| real_turnover_rate | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| residual_momentum_capm | research | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| revision_delta | extended | pandas_numpy |  | True | fundamental_period | None | 1 | 0 |
| rolling_beta_to_market | research | pandas_numpy, polars | FP_BETA, ROLLING_BETA_TO_MARKET, fp_beta | True | ts | None | 2 | 0 |
| round | extended | pandas_numpy, polars | ROUND | True | elementwise | None | None | 0 |
| safe_div_null | daily | pandas_numpy, polars, sql | div_or_null, safe_div | True | elementwise | None | None | 0 |
| saturate | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| scale | extended | pandas_numpy, polars, sql | SCALE, c_scale | True | cs | None | None | 0 |
| sec | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| sigmoid | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| sign | daily | pandas_numpy, polars, sql | SIGN | True | elementwise | None | None | 0 |
| signed_log | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| signed_power | extended | pandas_numpy, polars |  | True | elementwise | None | 1 | 0 |
| signed_sqrt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| sin | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| sinh | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| sqrt | daily | pandas_numpy, polars, sql | SQRT | True | elementwise | None | None | 0 |
| sqrt_abs | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| square | extended | pandas_numpy, polars | sqr | True | elementwise | None | None | 0 |
| subtract | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| tail_beta | research | pandas_numpy |  | True | ts | None | 3 | 0 |
| tan | extended | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| tanh | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| trade_when | research | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| true_range | extended | pandas_numpy, polars |  | True | ts | None | None | 0 |
| truncate | extended | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ts_argmax | extended | pandas_numpy, sql | m_argmax, ts_arg_max | True | ts | None | 1 | 0 |
| ts_argmin | extended | pandas_numpy, sql | m_argmin, ts_arg_min | True | ts | None | 1 | 0 |
| ts_autocorr | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 1 |
| ts_beta | daily | pandas_numpy, polars, sql | Beta, beta, m_beta, rolling_beta | True | ts | None | 2 | 0 |
| ts_bottomk_mean | extended | pandas_numpy | ts_bottom_n_avg | True | ts | None | 1 | 0 |
| ts_bottomk_std | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_bottomk_sum | extended | pandas_numpy | ts_bottom_n_sum | True | ts | None | 1 | 0 |
| ts_corr | daily | pandas_numpy, polars, sql | TS_CORR, corr, correlation, m_cor, ts_correlation | True | ts | None | 1 | 0 |
| ts_count_if | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_cov | daily | pandas_numpy, polars, sql | TS_COV, cov, m_cov, ts_covariance | True | ts | None | 1 | 0 |
| ts_days_since | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_decay_exp_window | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_decay_linear | extended | pandas_numpy, sql | DECAY_LINEAR, TS_DECAY_LINEAR, WMA, decay_linear, ts_decay | True | ts | None | 1 | 0 |
| ts_delay | daily | pandas_numpy, polars, sql | DELAY, Delay, Ref, delay, m_delay, prev, shift | True | ts | None | None | 1 |
| ts_delta | daily | pandas_numpy, polars, sql | Delta, Diff, TS_DELTA, delta | True | ts | None | None | 1 |
| ts_ema | extended | pandas_numpy, polars, sql | EMA, ema, ewm | True | ts | None | 1 | 0 |
| ts_ewm_corr | extended | pandas_numpy, sql | ewm_corr | True | ts | None | 2 | 0 |
| ts_ewm_cov | extended | pandas_numpy, sql | ewm_cov | True | ts | None | 2 | 0 |
| ts_ewm_std | extended | pandas_numpy, polars, sql | ewm_std | True | ts | None | 2 | 0 |
| ts_ewm_var | extended | pandas_numpy, polars, sql | ewm_var | True | ts | None | 2 | 0 |
| ts_kurt | extended | pandas_numpy | TS_KURT, kurt, m_kurt, ts_kurtosis | True | ts | None | None | 0 |
| ts_last_if | extended | pandas_numpy, sql |  | True | ts | None | 1 | 0 |
| ts_log_return | daily | pandas_numpy, polars, sql | log_returns | True | ts | None | 2 | 1 |
| ts_mad | extended | pandas_numpy, sql | m_mad, mad | True | ts | None | None | 0 |
| ts_max | daily | pandas_numpy, polars, sql | Max, TS_MAX, m_max | True | ts | None | 1 | 0 |
| ts_max_buildup | research | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_max_drawdown | extended | pandas_numpy, sql |  | True | ts | None | 2 | 0 |
| ts_mean | daily | pandas_numpy, polars, sql | Mean, SMA, TS_MEAN, m_avg, ma, mean | True | ts | None | 1 | 0 |
| ts_mean_if | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_median | daily | pandas_numpy, polars, sql | m_median, median | True | ts | None | 1 | 0 |
| ts_min | daily | pandas_numpy, polars, sql | Min, TS_MIN, m_min | True | ts | None | 1 | 0 |
| ts_moment | research | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_nth_value | extended | pandas_numpy, sql |  | True | ts | None | 1 | 0 |
| ts_partial_corr | extended | pandas_numpy, sql |  | True | ts | None | 3 | 0 |
| ts_pct | daily | pandas_numpy, polars, sql | TS_PCT, m_pct_change, pct_change, returns, ts_return | True | ts | None | None | 1 |
| ts_poly2_coeff | research | pandas_numpy |  | True | ts | None | 3 | 0 |
| ts_poly2_resid | research | pandas_numpy |  | True | ts | None | 3 | 0 |
| ts_product | extended | pandas_numpy, sql |  | True | ts | None | None | 0 |
| ts_quantile | extended | pandas_numpy, polars, sql | TS_QUANTILE, m_percentile, percentile | True | ts | None | None | 0 |
| ts_rank | daily | pandas_numpy, polars, sql | TS_RANK, m_rank | True | ts | None | 1 | 0 |
| ts_ratio | extended | pandas_numpy | ratios | True | ts | None | 2 | 1 |
| ts_regression_intercept | extended | pandas_numpy | intercept | True | ts | None | 3 | 0 |
| ts_regression_r2 | extended | pandas_numpy |  | True | ts | None | 3 | 0 |
| ts_regression_resid | extended | pandas_numpy | rolling_residual | True | ts | None | 3 | 0 |
| ts_regression_slope | extended | pandas_numpy, sql | TS_REGRESSION_SLOPE, ts_regression | True | ts | None | 3 | 0 |
| ts_regression_tstat | extended | pandas_numpy, sql |  | True | ts | None | 3 | 0 |
| ts_sharpe | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_skew | extended | pandas_numpy, polars, sql | TS_SKEW, m_skew, skew, ts_skewness | True | ts | None | None | 0 |
| ts_std | daily | pandas_numpy, polars, sql | Std, TS_STD, m_std, std, std_n, ts_std_dev, ts_stddev | True | ts | None | 1 | 0 |
| ts_std_if | extended | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_sum | daily | pandas_numpy, polars, sql | TS_SUM, m_sum, sum, sum_n | True | ts | None | 1 | 0 |
| ts_sum_decay | research | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_sum_if | extended | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ts_tail_mean | extended | pandas_numpy |  | True | ts | None | 1 | 0 |
| ts_time_slope | extended | pandas_numpy, sql | Slope, TS_TIME_SLOPE, slope | True | ts | None | 2 | 0 |
| ts_topk_mean | extended | pandas_numpy | tm_top_n_avg, ts_top_n_avg | True | ts | None | 1 | 0 |
| ts_topk_std | extended | pandas_numpy | ts_top_n_std | True | ts | None | 1 | 0 |
| ts_topk_sum | extended | pandas_numpy | TS_TOPK_SUM, m_top_n_sum, tm_top_n_sum | True | ts | None | 1 | 0 |
| ts_trend_tstat | extended | pandas_numpy, sql |  | True | ts | None | 3 | 0 |
| ts_true_streak | extended | pandas_numpy, sql |  | True | ts | None | 1 | 0 |
| ts_var | daily | pandas_numpy, polars, sql | m_var, var | True | ts | None | 1 | 0 |
| ts_zscore | daily | pandas_numpy, polars, sql | m_zscore | True | ts | None | 1 | 0 |
| ttm_from_cumulative | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| ttm_from_quarterly | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 1 | 0 |
| unitize | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| where | daily | pandas_numpy, polars, sql | IIF, WHERE, if, iif | True | elementwise | None | None | 0 |
| winsorize | daily | pandas_numpy, polars, sql | WINSORIZE, c_winsorize | True | cs | None | None | 0 |
| winsorize_mean | extended | pandas_numpy |  | True | elementwise | None | None | 0 |
| yoy_by_period | daily | pandas_numpy, polars, sql |  | True | fundamental_period | None | 5 | 0 |
| zscore | daily | pandas_numpy, polars, sql | CS_ZSCORE, ZSCORE, c_zscore, cs_zscore | True | cs | None | None | 0 |
