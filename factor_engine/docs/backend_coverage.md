# Backend capability matrix

> Generated from the final `OperatorRegistry`; static policy sets are intersected with active canonicals.
> `implemented` does not mean parity verified; `verified` does not mean production-safe.

## Summary

- active canonical: **205**
- daily surface: **86**
- Polars registered: **134**
- Polars non-bridge implementation: **134**
- Polars parity verified: **71**
- Polars production safe: **71**
- DuckDB SQL emitter implemented: **128**
- DuckDB parity verified: **87**
- DuckDB production safe: **87**
- Polars and DuckDB both production safe: **71**

## Matrix

| canonical | surface | pandas | polars | polars native | polars verified | polars safe | duckdb implemented | duckdb verified | duckdb safe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADX | extended | yes | yes | yes | no | no | no | no | no |
| ATR_WILDER | extended | yes | yes | yes | no | no | yes | no | no |
| MACD_hist | extended | yes | yes | yes | no | no | no | no | no |
| MACD_line | extended | yes | yes | yes | no | no | no | no | no |
| MACD_signal | extended | yes | yes | yes | no | no | no | no | no |
| RSI_WILDER | extended | yes | yes | yes | no | no | yes | no | no |
| abs | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| acos | extended | yes | yes | yes | no | no | no | no | no |
| add | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| and_ | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| arg | extended | yes | no | no | no | no | no | no | no |
| asin | extended | yes | yes | yes | no | no | no | no | no |
| atan | extended | yes | yes | yes | no | no | no | no | no |
| atan2 | extended | yes | no | no | no | no | no | no | no |
| cbrt | extended | yes | yes | yes | no | no | yes | no | no |
| ceil | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| clip | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| coalesce | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| constant | internal | yes | no | no | no | no | no | no | no |
| cos | extended | yes | yes | yes | no | no | no | no | no |
| cosh | extended | yes | yes | yes | no | no | no | no | no |
| coskewness_to_market | research | yes | yes | yes | no | no | no | no | no |
| cot | extended | yes | yes | yes | no | no | no | no | no |
| cs_bucket | extended | yes | yes | yes | no | no | yes | no | no |
| cs_count | daily | yes | yes | yes | no | no | yes | yes | yes |
| cs_demean | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| cs_fill_mean | extended | yes | yes | yes | no | no | no | no | no |
| cs_fill_median | extended | yes | yes | yes | no | no | no | no | no |
| cs_mad | daily | yes | no | no | no | no | yes | yes | yes |
| cs_mad_zscore | daily | yes | no | no | no | no | yes | yes | yes |
| cs_mean | daily | yes | yes | yes | no | no | yes | yes | yes |
| cs_multi_resid | extended | yes | no | no | no | no | yes | no | no |
| cs_neutralize | extended | yes | no | no | no | no | no | no | no |
| cs_pct_rank | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| cs_quantile | extended | yes | no | no | no | no | yes | no | no |
| cs_rank_gaussian | extended | yes | no | no | no | no | no | no | no |
| cs_regression | extended | yes | no | no | no | no | yes | no | no |
| cs_resid | extended | yes | no | no | no | no | yes | no | no |
| cs_std | daily | yes | no | no | no | no | yes | yes | yes |
| cs_sum | daily | yes | yes | yes | no | no | yes | yes | yes |
| cs_weighted_demean | extended | yes | no | no | no | no | no | no | no |
| cs_weighted_mean | extended | yes | no | no | no | no | no | no | no |
| cs_weighted_zscore | extended | yes | no | no | no | no | no | no | no |
| cs_wls_resid | extended | yes | no | no | no | no | yes | no | no |
| csc | extended | yes | yes | yes | no | no | no | no | no |
| cube | legacy | yes | yes | yes | no | no | no | no | no |
| digital_count | research | yes | yes | yes | no | no | no | no | no |
| divide | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| eq | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| exp | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| exp_neg | extended | yes | yes | yes | no | no | no | no | no |
| ffill_limit | extended | yes | yes | yes | no | no | no | no | no |
| fillna_const | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| fix | extended | yes | yes | yes | no | no | no | no | no |
| flex_max | extended | yes | yes | yes | no | no | no | no | no |
| flex_min | extended | yes | yes | yes | no | no | no | no | no |
| floor | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| fundamental_staleness | extended | yes | yes | yes | no | no | no | no | no |
| ge | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| group_count | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| group_decay_linear | research | yes | yes | yes | no | no | yes | no | no |
| group_max | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| group_mean | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| group_min | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| group_neutralize | daily | yes | no | no | no | no | yes | yes | yes |
| group_normalize | daily | yes | no | no | no | no | yes | yes | yes |
| group_percentile | extended | yes | no | no | no | no | yes | no | no |
| group_rank | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| group_std | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| group_sum | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| group_weighted_mean | extended | yes | no | no | no | no | no | no | no |
| group_weighted_zscore | extended | yes | no | no | no | no | no | no | no |
| group_winsorize | daily | yes | no | no | no | no | yes | yes | yes |
| group_zscore | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| gt | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| identity | internal | yes | yes | yes | no | no | no | no | no |
| idio_skew | research | yes | yes | yes | no | no | no | no | no |
| intraday_vwap_deviation | research | yes | no | no | no | no | no | no | no |
| inverse | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| is_finite | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| is_infinite | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| is_nan | extended | yes | yes | yes | no | no | yes | no | no |
| is_not_null | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| is_null | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| le | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| lerp | extended | yes | yes | yes | no | no | no | no | no |
| log | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| log10 | extended | yes | yes | yes | no | no | no | no | no |
| log2 | extended | yes | yes | yes | no | no | no | no | no |
| log_abs | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| lt | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| maximum | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| minimum | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| multiply | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ne | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| neg | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| normalize | daily | yes | no | no | no | no | yes | yes | yes |
| not_ | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| or_ | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| period_average | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| period_cagr | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| period_change | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| period_lag | extended | yes | yes | yes | no | no | yes | no | no |
| period_stability | extended | yes | no | no | no | no | no | no | no |
| power | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| price_spread_deviation | extended | yes | no | no | no | no | no | no | no |
| protected_div | internal | yes | no | no | no | no | yes | yes | yes |
| quarter_from_cumulative | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| rank | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| real_turnover_rate | extended | yes | yes | yes | no | no | no | no | no |
| residual_momentum_capm | research | yes | yes | yes | no | no | no | no | no |
| revision_delta | extended | yes | no | no | no | no | no | no | no |
| rolling_beta_to_market | research | yes | yes | yes | no | no | no | no | no |
| round | extended | yes | yes | yes | no | no | no | no | no |
| safe_div_null | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| saturate | extended | yes | yes | yes | no | no | no | no | no |
| scale | extended | yes | yes | yes | no | no | yes | no | no |
| sec | extended | yes | yes | yes | no | no | no | no | no |
| sigmoid | extended | yes | yes | yes | no | no | no | no | no |
| sign | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| signed_log | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| signed_power | extended | yes | yes | yes | no | no | no | no | no |
| signed_sqrt | daily | yes | no | no | no | no | yes | yes | yes |
| sin | extended | yes | yes | yes | no | no | no | no | no |
| sinh | extended | yes | yes | yes | no | no | no | no | no |
| sqrt | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| sqrt_abs | extended | yes | no | no | no | no | no | no | no |
| square | extended | yes | yes | yes | no | no | no | no | no |
| subtract | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| tail_beta | research | yes | no | no | no | no | no | no | no |
| tan | extended | yes | yes | yes | no | no | no | no | no |
| tanh | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| trade_when | research | yes | yes | yes | no | no | no | no | no |
| true_range | extended | yes | yes | yes | no | no | no | no | no |
| truncate | extended | yes | yes | yes | no | no | yes | no | no |
| ts_argmax | extended | yes | no | no | no | no | yes | no | no |
| ts_argmin | extended | yes | no | no | no | no | yes | no | no |
| ts_autocorr | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_beta | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_bottomk_mean | extended | yes | no | no | no | no | no | no | no |
| ts_bottomk_std | extended | yes | no | no | no | no | no | no | no |
| ts_bottomk_sum | extended | yes | no | no | no | no | no | no | no |
| ts_corr | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_count_if | extended | yes | yes | yes | no | no | yes | no | no |
| ts_cov | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_days_since | extended | yes | yes | yes | no | no | yes | no | no |
| ts_decay_exp_window | extended | yes | no | no | no | no | no | no | no |
| ts_decay_linear | extended | yes | no | no | no | no | yes | no | no |
| ts_delay | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_delta | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_ema | extended | yes | yes | yes | no | no | yes | no | no |
| ts_ewm_corr | extended | yes | no | no | no | no | yes | no | no |
| ts_ewm_cov | extended | yes | no | no | no | no | yes | no | no |
| ts_ewm_std | extended | yes | yes | yes | no | no | yes | no | no |
| ts_ewm_var | extended | yes | yes | yes | no | no | yes | no | no |
| ts_kurt | extended | yes | no | no | no | no | no | no | no |
| ts_last_if | extended | yes | no | no | no | no | yes | no | no |
| ts_log_return | daily | yes | no | no | no | no | yes | yes | yes |
| ts_mad | extended | yes | no | no | no | no | yes | no | no |
| ts_max | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_max_buildup | research | yes | no | no | no | no | no | no | no |
| ts_max_drawdown | extended | yes | no | no | no | no | yes | no | no |
| ts_mean | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_mean_if | extended | yes | yes | yes | no | no | yes | no | no |
| ts_median | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_min | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_moment | research | yes | no | no | no | no | no | no | no |
| ts_nth_value | extended | yes | no | no | no | no | yes | no | no |
| ts_partial_corr | extended | yes | no | no | no | no | yes | no | no |
| ts_pct | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_poly2_coeff | research | yes | no | no | no | no | no | no | no |
| ts_poly2_resid | research | yes | no | no | no | no | no | no | no |
| ts_product | extended | yes | no | no | no | no | yes | no | no |
| ts_quantile | extended | yes | yes | yes | no | no | yes | no | no |
| ts_rank | daily | yes | no | no | no | no | yes | yes | yes |
| ts_ratio | extended | yes | no | no | no | no | no | no | no |
| ts_regression_intercept | extended | yes | no | no | no | no | no | no | no |
| ts_regression_r2 | extended | yes | no | no | no | no | no | no | no |
| ts_regression_resid | extended | yes | no | no | no | no | no | no | no |
| ts_regression_slope | extended | yes | no | no | no | no | yes | no | no |
| ts_regression_tstat | extended | yes | no | no | no | no | yes | no | no |
| ts_sharpe | daily | yes | no | no | no | no | yes | yes | yes |
| ts_skew | extended | yes | yes | yes | no | no | yes | no | no |
| ts_std | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_std_if | extended | yes | yes | yes | no | no | yes | no | no |
| ts_sum | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_sum_decay | research | yes | no | no | no | no | no | no | no |
| ts_sum_if | extended | yes | yes | yes | no | no | yes | no | no |
| ts_tail_mean | extended | yes | no | no | no | no | no | no | no |
| ts_time_slope | extended | yes | no | no | no | no | yes | no | no |
| ts_topk_mean | extended | yes | no | no | no | no | no | no | no |
| ts_topk_std | extended | yes | no | no | no | no | no | no | no |
| ts_topk_sum | extended | yes | no | no | no | no | no | no | no |
| ts_trend_tstat | extended | yes | no | no | no | no | yes | no | no |
| ts_true_streak | extended | yes | no | no | no | no | yes | no | no |
| ts_var | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ts_zscore | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ttm_from_cumulative | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| ttm_from_quarterly | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| unitize | extended | yes | no | no | no | no | no | no | no |
| where | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| winsorize | daily | yes | no | no | no | no | yes | yes | yes |
| winsorize_mean | extended | yes | no | no | no | no | no | no | no |
| yoy_by_period | daily | yes | yes | yes | yes | yes | yes | yes | yes |
| zscore | daily | yes | yes | yes | yes | yes | yes | yes | yes |
