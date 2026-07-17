# Backend capability matrix

> Generated from the final `OperatorRegistry` and exact active canonical names.
> Registration, non-bridge implementation, expression-native implementation, parity evidence and production routing are distinct states.

## Summary

- active canonical: **218**
- daily surface: **50**
- extended surface: **25**
- internal surface: **1**
- legacy surface: **1**
- research surface: **23**
- unclassified surface: **118**
- Pandas runtime: **218**
- Polars registered: **182**
- Polars non-bridge: **181**
- Polars expression-native: **34**
- Polars parity verified: **68**
- Polars production safe: **84**
- DuckDB emitter implemented: **118**
- DuckDB reference parity: **77**
- DuckDB production safe: **66**
- ClickHouse production safe: **0**

## Matrix

| canonical | surface | pandas | polars | no bridge | expression native | polars verified | polars safe | duckdb implemented | duckdb verified | duckdb safe | polars source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ADX | daily | yes | yes | yes | yes | no | no | no | no | no | layer_governance_native_polars |
| ATR_WILDER | unclassified | yes | yes | yes | no | yes | yes | yes | no | no | composite_fastpath_native_polars |
| MACD_hist | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| MACD_line | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| MACD_signal | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| RSI_WILDER | unclassified | yes | yes | yes | no | yes | yes | yes | no | no | composite_fastpath_native_polars |
| abs | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| acos | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| add | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| and_ | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars_auto |
| arg | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| asin | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| atan | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| atan2 | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| cbrt | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| ceil | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| clip | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| coalesce | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars_auto |
| constant | internal | yes | no | no | no | no | no | no | no | no | - |
| cos | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| cosh | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| coskewness_to_market | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| cot | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| cs_bucket | unclassified | yes | yes | yes | yes | no | no | yes | no | no | operator_overhaul_native_polars |
| cs_count | unclassified | yes | yes | yes | no | no | no | yes | no | no | factor_dsl_np |
| cs_demean | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| cs_fill_mean | daily | yes | yes | yes | yes | no | no | no | no | no | layer_governance_native_polars |
| cs_fill_median | daily | yes | yes | yes | yes | no | no | no | no | no | layer_governance_native_polars |
| cs_mad | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| cs_mad_zscore | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| cs_mean | unclassified | yes | yes | yes | no | no | no | yes | no | no | factor_dsl_np |
| cs_multi_resid | unclassified | yes | no | no | no | no | no | yes | no | no | - |
| cs_neutralize | daily | yes | no | no | no | no | no | no | no | no | - |
| cs_pct_rank | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| cs_quantile | unclassified | yes | yes | yes | no | yes | yes | yes | no | no | factor_dsl_polars |
| cs_rank_gaussian | daily | yes | no | no | no | no | no | no | no | no | - |
| cs_regression | unclassified | yes | yes | yes | no | yes | yes | yes | no | no | factor_dsl_polars |
| cs_resid | unclassified | yes | yes | yes | no | yes | yes | yes | no | no | factor_dsl_polars |
| cs_std | unclassified | yes | yes | yes | no | no | no | yes | no | no | factor_dsl_np |
| cs_sum | unclassified | yes | yes | yes | no | no | no | yes | no | no | factor_dsl_np |
| cs_weighted_demean | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| cs_weighted_mean | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| cs_weighted_zscore | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| cs_wls_resid | unclassified | yes | no | no | no | no | no | yes | no | no | - |
| csc | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| cube | legacy | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| digital_count | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| divide | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| downside_beta | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| eq | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars_auto |
| exp | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| exp_neg | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| ffill_limit | daily | yes | yes | yes | yes | no | no | no | no | no | layer_governance_native_polars |
| fillna_const | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| fix | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| flex_max | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| flex_min | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| floor | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| fundamental_staleness | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| ge | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars_auto |
| group_count | unclassified | yes | yes | yes | no | no | no | yes | yes | no | factor_dsl_polars |
| group_decay_linear | unclassified | yes | yes | yes | no | no | no | yes | no | no | factor_dsl_polars |
| group_max | unclassified | yes | yes | yes | no | no | no | yes | yes | no | factor_dsl_polars |
| group_mean | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| group_min | unclassified | yes | yes | yes | no | no | no | yes | yes | no | factor_dsl_polars |
| group_neutralize | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| group_normalize | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| group_percentile | unclassified | yes | yes | yes | no | yes | yes | yes | yes | no | semantic_hardening |
| group_rank | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| group_std | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| group_sum | unclassified | yes | yes | yes | no | no | no | yes | yes | no | factor_dsl_polars |
| group_weighted_mean | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| group_weighted_zscore | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| group_winsorize | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| group_zscore | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| gt | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars_auto |
| identity | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| idio_skew | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| idio_vol | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| intraday_vwap_deviation | research | yes | no | no | no | no | no | no | no | no | - |
| inverse | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| is_finite | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| is_infinite | unclassified | yes | yes | yes | no | no | no | yes | yes | no | factor_dsl_polars |
| is_nan | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| is_not_null | unclassified | yes | yes | yes | no | yes | yes | yes | yes | no | factor_dsl_polars |
| is_null | unclassified | yes | yes | yes | no | yes | yes | yes | yes | no | factor_dsl_polars |
| le | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars_auto |
| lerp | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| log | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| log10 | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| log2 | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| log_abs | unclassified | yes | yes | yes | no | yes | yes | yes | yes | no | factor_dsl_polars_auto |
| lt | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars_auto |
| maximum | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| micro_amihud_hf | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| micro_bipower_var | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| micro_jump_indicator | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| micro_kyle_lambda | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| micro_mid_return | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| micro_realized_vol | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| micro_spread | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| micro_trade_imbalance | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| micro_vpin | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| minimum | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| multiply | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| ne | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars_auto |
| neg | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| normalize | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| not_ | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars_auto |
| or_ | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars_auto |
| period_average | daily | yes | no | no | no | no | no | no | no | no | - |
| period_cagr | daily | yes | no | no | no | no | no | no | no | no | - |
| period_change | daily | yes | no | no | no | no | no | no | no | no | - |
| period_lag | unclassified | yes | no | no | no | no | no | yes | no | no | - |
| period_stability | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| power | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| price_spread_deviation | extended | yes | no | no | no | no | no | no | no | no | - |
| protected_div | unclassified | yes | no | no | no | no | no | yes | yes | yes | - |
| quarter_from_cumulative | daily | yes | no | no | no | no | no | no | no | no | - |
| rank | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| rank_corr | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| real_turnover_rate | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| residual_momentum_capm | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| reverse | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| revision_delta | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| rolling_beta_to_market | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| round | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| safe_div_null | unclassified | yes | yes | yes | yes | yes | yes | yes | yes | yes | final_expression_native_polars |
| saturate | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| scale | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| sec | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| sigmoid | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| sign | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| signed_log | unclassified | yes | yes | yes | no | yes | yes | yes | yes | no | factor_dsl_polars |
| signed_power | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| signed_sqrt | unclassified | yes | no | no | no | no | no | yes | yes | no | - |
| sin | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| sinh | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| sqrt | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| sqrt_abs | daily | yes | no | no | no | no | no | no | no | no | - |
| square | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| subtract | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_polars |
| tail_beta | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| tan | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| tanh | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| trade_when | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars |
| true_range | daily | yes | yes | yes | yes | no | no | no | no | no | operator_overhaul_native_polars |
| truncate | extended | yes | yes | yes | no | no | no | no | no | no | factor_dsl_polars_auto |
| ts_argmax | unclassified | yes | yes | yes | yes | yes | yes | yes | no | no | final_expression_native_polars |
| ts_argmin | unclassified | yes | yes | yes | yes | yes | yes | yes | no | no | final_expression_native_polars |
| ts_autocorr | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| ts_beta | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| ts_bottomk_mean | daily | yes | yes | yes | yes | no | no | no | no | no | final_expression_native_polars |
| ts_bottomk_std | daily | yes | yes | yes | yes | no | no | no | no | no | final_expression_native_polars |
| ts_bottomk_sum | daily | yes | yes | yes | yes | no | no | no | no | no | final_expression_native_polars |
| ts_corr | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| ts_count_if | unclassified | yes | yes | yes | yes | no | no | yes | no | no | operator_overhaul_native_polars |
| ts_cov | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| ts_days_since | unclassified | yes | yes | yes | yes | no | no | yes | no | no | operator_overhaul_native_polars |
| ts_decay_exp_window | daily | yes | yes | yes | no | no | no | no | no | no | factor_dsl_np |
| ts_decay_linear | unclassified | yes | yes | yes | no | yes | yes | yes | no | no | factor_dsl_np |
| ts_delay | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_np |
| ts_delta | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_np |
| ts_ema | unclassified | yes | yes | yes | yes | yes | yes | yes | no | no | final_expression_native_polars |
| ts_ewm_corr | unclassified | yes | yes | yes | no | no | no | yes | no | no | factor_dsl_polars |
| ts_ewm_cov | unclassified | yes | yes | yes | no | no | no | yes | no | no | factor_dsl_polars |
| ts_ewm_std | unclassified | yes | yes | yes | no | no | no | yes | no | no | factor_dsl_polars |
| ts_ewm_var | unclassified | yes | yes | yes | no | no | no | yes | no | no | factor_dsl_polars |
| ts_kurt | daily | yes | yes | yes | no | yes | yes | no | no | no | factor_dsl_np |
| ts_last_if | unclassified | yes | yes | yes | yes | no | no | yes | no | no | final_expression_native_polars |
| ts_log_return | unclassified | yes | no | no | no | no | no | yes | no | no | - |
| ts_mad | unclassified | yes | yes | yes | no | yes | yes | yes | no | no | semantic_hardening |
| ts_max | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_np |
| ts_max_buildup | extended | yes | no | no | no | no | no | no | no | no | - |
| ts_max_drawdown | unclassified | yes | no | no | no | no | no | yes | no | no | - |
| ts_mean | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_np |
| ts_mean_if | unclassified | yes | yes | yes | yes | no | no | yes | no | no | operator_overhaul_native_polars |
| ts_median | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| ts_min | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_np |
| ts_moment | extended | yes | no | no | no | no | no | no | no | no | - |
| ts_nth_value | unclassified | yes | no | no | no | no | no | yes | no | no | - |
| ts_partial_corr | unclassified | yes | no | no | no | no | no | yes | no | no | - |
| ts_pct | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| ts_poly2_coeff | research | yes | no | no | no | no | no | no | no | no | - |
| ts_poly2_resid | research | yes | no | no | no | no | no | no | no | no | - |
| ts_product | unclassified | yes | yes | yes | no | yes | yes | yes | no | no | semantic_hardening |
| ts_quantile | unclassified | yes | yes | yes | no | yes | yes | yes | no | no | factor_dsl_polars |
| ts_rank | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| ts_ratio | extended | yes | no | no | no | no | no | no | no | no | - |
| ts_regression_intercept | daily | yes | no | no | no | no | no | no | no | no | - |
| ts_regression_r2 | daily | yes | no | no | no | no | no | no | no | no | - |
| ts_regression_resid | daily | yes | no | no | no | no | no | no | no | no | - |
| ts_regression_slope | unclassified | yes | yes | no | no | yes | no | yes | no | no | - |
| ts_regression_tstat | unclassified | yes | no | no | no | no | no | yes | no | no | - |
| ts_sharpe | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| ts_skew | unclassified | yes | yes | yes | no | yes | yes | yes | no | no | factor_dsl_np |
| ts_std | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| ts_std_if | unclassified | yes | yes | yes | yes | no | no | yes | no | no | operator_overhaul_native_polars |
| ts_sum | unclassified | yes | yes | yes | no | no | yes | yes | yes | yes | factor_dsl_np |
| ts_sum_decay | research | yes | yes | yes | no | no | no | no | no | no | factor_dsl_np |
| ts_sum_if | unclassified | yes | yes | yes | yes | no | no | yes | no | no | operator_overhaul_native_polars |
| ts_tail_mean | daily | yes | no | no | no | no | no | no | no | no | - |
| ts_time_slope | unclassified | yes | no | no | no | no | no | yes | no | no | - |
| ts_topk_mean | daily | yes | yes | yes | yes | no | no | no | no | no | final_expression_native_polars |
| ts_topk_std | daily | yes | yes | yes | yes | no | no | no | no | no | final_expression_native_polars |
| ts_topk_sum | daily | yes | yes | yes | yes | no | no | no | no | no | final_expression_native_polars |
| ts_trend_tstat | unclassified | yes | no | no | no | no | no | yes | no | no | - |
| ts_true_streak | unclassified | yes | yes | yes | yes | no | no | yes | no | no | final_expression_native_polars |
| ts_var | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| ts_zscore | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| ttm_from_cumulative | daily | yes | no | no | no | no | no | no | no | no | - |
| ttm_from_quarterly | daily | yes | no | no | no | no | no | no | no | no | - |
| unitize | extended | yes | no | no | no | no | no | no | no | no | - |
| where | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_polars |
| winsorize | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
| winsorize_mean | extended | yes | no | no | no | no | no | no | no | no | - |
| yoy_by_period | daily | yes | no | no | no | no | no | no | no | no | - |
| zscore | unclassified | yes | yes | yes | no | yes | yes | yes | yes | yes | factor_dsl_np |
