# Operators Catalog（自动生成）

> 生成日期：2026-07-08
> 由 `scripts/generate_operators_catalog.py` 从注册表 + OperatorPolicy 生成。

## 摘要

- canonical 总数：361

## Implemented 算子

| canonical | backends | pit_safe | scope | lookback | min_periods | lag |
|-----------|----------|----------|-------|----------|-------------|-----|
| ACF | pandas_numpy | True | aggregate | None | 1 | 0 |
| ADX | pandas_numpy, polars | True | unknown | None | None | 0 |
| ADXR | pandas_numpy, polars | True | unknown | None | None | 0 |
| AROON | pandas_numpy, polars | True | unknown | None | None | 0 |
| AROON_down | pandas_numpy, polars | True | unknown | None | None | 0 |
| AROON_up | pandas_numpy, polars | True | unknown | None | None | 0 |
| ATR | pandas_numpy, polars | True | unknown | None | None | 0 |
| Beta | polars | True | aggregate | None | 1 | 0 |
| BollingerBands | pandas_numpy, polars | True | unknown | None | None | 0 |
| BollingerLower | pandas_numpy, polars | True | unknown | None | None | 0 |
| BollingerUpper | pandas_numpy, polars | True | unknown | None | None | 0 |
| CCI | pandas_numpy, polars | True | unknown | None | None | 0 |
| Corr | polars | True | ts | None | 1 | 0 |
| Cov | polars | True | ts | None | None | 0 |
| Covariance | polars | True | ts | None | None | 0 |
| DPO | pandas_numpy, polars | True | unknown | None | None | 0 |
| KAMA | pandas_numpy, polars | True | unknown | None | None | 0 |
| Kurt | polars | True | ts | None | None | 0 |
| Lead | pandas_numpy, polars | False | ts | None | None | -1 |
| MACD | pandas_numpy, polars | True | unknown | None | None | 0 |
| MACD_hist | pandas_numpy, polars | True | unknown | None | None | 0 |
| MACD_line | pandas_numpy, polars | True | unknown | None | None | 0 |
| MACD_signal | pandas_numpy, polars | True | unknown | None | None | 0 |
| MOM | pandas_numpy, polars | True | unknown | None | None | 0 |
| Median | polars | True | ts | None | None | 0 |
| Mode | pandas_numpy, polars | True | aggregate | None | 1 | 0 |
| OBV | pandas_numpy, polars | True | unknown | None | None | 0 |
| ROC | pandas_numpy, polars | True | unknown | None | None | 0 |
| RSI | pandas_numpy, polars | True | unknown | None | None | 0 |
| Skew | polars | True | ts | None | None | 0 |
| Slope | pandas_numpy, polars | True | aggregate | None | 1 | 0 |
| StochasticD | pandas_numpy, polars | True | unknown | None | None | 0 |
| StochasticK | pandas_numpy, polars | True | unknown | None | None | 0 |
| TRIX | pandas_numpy, polars | True | unknown | None | None | 0 |
| Var | polars | True | ts | None | None | 0 |
| WMA | pandas_numpy, polars | True | ts | None | 1 | 0 |
| WilliamsR | pandas_numpy, polars | True | unknown | None | None | 0 |
| abs | pandas_numpy, polars, sql | True | elementwise | None | None | 0 |
| acos | pandas_numpy, polars | True | elementwise | None | None | 0 |
| add | pandas_numpy, polars, sql | True | elementwise | None | None | 0 |
| aggr_top_n | pandas_numpy | True | ts | None | None | 0 |
| and_ | pandas_numpy, polars | True | elementwise | None | None | 0 |
| arg | pandas_numpy | True | elementwise | None | None | 0 |
| asin | pandas_numpy, polars | True | elementwise | None | None | 0 |
| at_imax | pandas_numpy | True | aggregate | None | 1 | 0 |
| at_imin | pandas_numpy | True | aggregate | None | 1 | 0 |
| atan | pandas_numpy, polars | True | elementwise | None | None | 0 |
| atan2 | pandas_numpy, polars | True | elementwise | None | None | 0 |
| autocorr | pandas_numpy, polars | True | aggregate | None | 1 | 0 |
| avg | pandas_numpy, polars | True | aggregate | None | None | 0 |
| avg2 | pandas_numpy | True | unknown | None | None | 0 |
| bartlett_test | pandas_numpy | True | hypothesis | None | None | 0 |
| beta | pandas_numpy, polars | True | aggregate | None | 1 | 0 |
| bfill | pandas_numpy, polars | False | elementwise | None | None | 0 |
| blom_transform | pandas_numpy | True | elementwise | None | None | 0 |
| c_count | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| c_mean | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| c_std | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| c_sum | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| cbrt | pandas_numpy, polars | True | elementwise | None | None | 0 |
| cdf_chi2 | pandas_numpy | True | aggregate | None | 1 | 0 |
| cdf_f | pandas_numpy | True | aggregate | None | 1 | 0 |
| cdf_normal | pandas_numpy | True | aggregate | None | 1 | 0 |
| cdf_t | pandas_numpy | True | aggregate | None | 1 | 0 |
| ceil | pandas_numpy, polars | True | elementwise | None | None | 0 |
| chi_square_test | pandas_numpy | True | hypothesis | None | None | 0 |
| clip | pandas_numpy, polars, sql | True | elementwise | None | None | 0 |
| coalesce | pandas_numpy, polars | True | elementwise | None | None | 0 |
| complex | pandas_numpy | True | elementwise | None | None | 0 |
| conj | pandas_numpy | True | elementwise | None | None | 0 |
| constant | pandas_numpy | True | elementwise | None | None | 0 |
| convolve | pandas_numpy | True | elementwise | None | None | 0 |
| corr_test | pandas_numpy, polars | True | hypothesis | None | None | 0 |
| correlate | pandas_numpy | True | elementwise | None | None | 0 |
| cos | pandas_numpy, polars | True | elementwise | None | None | 0 |
| cosh | pandas_numpy, polars | True | elementwise | None | None | 0 |
| coskewness_to_market | pandas_numpy, polars | True | ts | None | None | 0 |
| cot | pandas_numpy, polars | True | elementwise | None | None | 0 |
| count | pandas_numpy, polars | True | aggregate | None | 1 | 0 |
| cs_demean | pandas_numpy, polars, sql | True | cs | 0 | 1 | 0 |
| cs_regression | pandas_numpy, polars | True | cs | None | None | 0 |
| cs_resid | pandas_numpy, polars | True | cs | None | None | 0 |
| csc | pandas_numpy, polars | True | elementwise | None | None | 0 |
| cube | pandas_numpy, polars | True | elementwise | None | None | 0 |
| cum_count | pandas_numpy, polars | True | ts | None | None | 0 |
| cum_delta | pandas_numpy, polars | True | ts | None | None | 0 |
| cum_first | pandas_numpy, polars | True | ts | None | None | 0 |
| cum_last | polars | True | elementwise | None | None | 0 |
| cum_max | pandas_numpy, polars | True | ts | None | None | 0 |
| cum_min | pandas_numpy, polars | True | ts | None | None | 0 |
| cum_positive_streak | pandas_numpy, polars | True | ts | None | None | 0 |
| cum_prod | pandas_numpy, polars | True | ts | None | None | 0 |
| cum_sum | pandas_numpy, polars | True | ts | None | 1 | 0 |
| cum_top_n_avg | pandas_numpy, polars | True | ts | None | None | 0 |
| cum_top_n_sum | pandas_numpy, polars | True | ts | None | 1 | 0 |
| cumulative_max | pandas_numpy, polars | True | elementwise | None | None | 0 |
| cumulative_mean | pandas_numpy, polars | True | elementwise | None | None | 0 |
| cumulative_min | pandas_numpy, polars | True | elementwise | None | None | 0 |
| cumulative_returns | pandas_numpy, polars | True | unknown | None | None | 0 |
| decimate | pandas_numpy | True | elementwise | None | None | 0 |
| digital_count | pandas_numpy | True | ts | None | None | 0 |
| divide | pandas_numpy, polars, sql | True | elementwise | None | None | 0 |
| downside_beta | pandas_numpy, polars | True | ts | None | None | 0 |
| dropna | pandas_numpy, polars | True | elementwise | None | None | 0 |
| durbin_watson_test | pandas_numpy, polars | True | hypothesis | None | None | 0 |
| eig | pandas_numpy | True | elementwise | None | None | 0 |
| eq | pandas_numpy, polars | True | elementwise | None | None | 0 |
| ewm | pandas_numpy, polars | True | elementwise | None | None | 0 |
| ewm_corr | pandas_numpy, polars | True | ts | None | 1 | 0 |
| ewm_cov | pandas_numpy, polars | True | elementwise | None | None | 0 |
| ewm_mean | pandas_numpy, polars, sql | True | ts | None | 1 | 0 |
| ewm_std | pandas_numpy, polars | True | elementwise | None | None | 0 |
| ewm_var | pandas_numpy, polars | True | elementwise | None | None | 0 |
| exp | pandas_numpy, polars, sql | True | elementwise | None | None | 0 |
| exp_neg | pandas_numpy, polars | True | elementwise | None | None | 0 |
| expanding_max | pandas_numpy, polars | True | elementwise | None | None | 0 |
| expanding_mean | pandas_numpy, polars | True | elementwise | None | None | 0 |
| expanding_min | pandas_numpy, polars | True | elementwise | None | None | 0 |
| expanding_rank | pandas_numpy, polars | True | ts | None | None | 0 |
| expanding_std | pandas_numpy, polars | True | elementwise | None | None | 0 |
| expanding_sum | pandas_numpy, polars | True | elementwise | None | None | 0 |
| expanding_zscore | pandas_numpy, polars | True | ts | None | None | 0 |
| ffill | pandas_numpy, polars | True | elementwise | None | None | 0 |
| fft | pandas_numpy | True | elementwise | None | None | 0 |
| fillna | pandas_numpy, polars | True | elementwise | None | None | 0 |
| fillna_const | pandas_numpy, polars | True | elementwise | None | None | 0 |
| fillna_interpolate | pandas_numpy, polars | False | elementwise | None | None | 0 |
| filter_bandpass | pandas_numpy | True | elementwise | None | None | 0 |
| filter_highpass | pandas_numpy | True | elementwise | None | None | 0 |
| filter_lowpass | pandas_numpy | True | elementwise | None | None | 0 |
| filter_notch | pandas_numpy | True | elementwise | None | None | 0 |
| first_not_null | pandas_numpy | True | aggregate | None | 1 | 0 |
| fix | pandas_numpy, polars | True | elementwise | None | None | 0 |
| flex_max | pandas_numpy | True | elementwise | None | None | 0 |
| flex_min | pandas_numpy | True | elementwise | None | None | 0 |
| floor | pandas_numpy, polars | True | elementwise | None | None | 0 |
| fmax | pandas_numpy, polars | True | elementwise | None | None | 0 |
| fmin | pandas_numpy, polars | True | elementwise | None | None | 0 |
| ge | pandas_numpy, polars | True | elementwise | None | None | 0 |
| geometric_mean | pandas_numpy | True | elementwise | None | None | 0 |
| granger_causality | pandas_numpy | True | hypothesis | None | None | 0 |
| group_decay_linear | pandas_numpy, polars | True | ts | None | 1 | 0 |
| group_mean | pandas_numpy, polars, sql | True | cs | None | None | 0 |
| group_neutralize | pandas_numpy, polars, sql | True | cs | None | None | 0 |
| group_normalize | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| group_percentile | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| group_rank | pandas_numpy, polars, sql | True | cs | None | None | 0 |
| group_std | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| group_winsorize | pandas_numpy, polars, sql | True | cs | None | None | 0 |
| group_zscore | pandas_numpy, polars, sql | True | cs | None | None | 0 |
| gt | pandas_numpy, polars | True | elementwise | None | None | 0 |
| harmonic_mean | pandas_numpy | True | elementwise | None | None | 0 |
| hump_decay | pandas_numpy, polars | True | ts | None | 1 | 0 |
| identity | pandas_numpy, polars | True | elementwise | None | None | 0 |
| idio_skew | pandas_numpy, polars | True | ts | None | None | 0 |
| idio_vol | pandas_numpy, polars | True | ts | None | None | 0 |
| if_else | polars, sql | True | ts | None | None | 0 |
| ifft | pandas_numpy | True | elementwise | None | None | 0 |
| ifnan | pandas_numpy, polars | True | ts | None | None | 0 |
| imag | pandas_numpy | True | elementwise | None | None | 0 |
| intercept | pandas_numpy, polars | True | aggregate | None | 1 | 0 |
| interpolate | pandas_numpy | True | elementwise | None | None | 0 |
| inv | pandas_numpy | True | elementwise | None | None | 0 |
| inverse | pandas_numpy | True | elementwise | None | None | 0 |
| is_finite | pandas_numpy, polars | True | ts | None | None | 0 |
| is_inf | pandas_numpy, polars | True | elementwise | None | None | 0 |
| is_nan | pandas_numpy, polars | True | elementwise | None | None | 0 |
| jarque_bera_test | pandas_numpy | True | hypothesis | None | None | 0 |
| kendall_corr_test | pandas_numpy | True | hypothesis | None | None | 0 |
| kpss_test | pandas_numpy | True | hypothesis | None | None | 0 |
| ks_test | pandas_numpy | True | hypothesis | None | None | 0 |
| lasso | pandas_numpy | True | aggregate | None | 1 | 0 |
| le | pandas_numpy, polars | True | elementwise | None | None | 0 |
| lerp | pandas_numpy | True | elementwise | None | None | 0 |
| levene_test | pandas_numpy | True | hypothesis | None | None | 0 |
| lilliefors_test | pandas_numpy | True | hypothesis | None | None | 0 |
| log | pandas_numpy, polars, sql | True | elementwise | None | None | 0 |
| log10 | pandas_numpy, polars | True | elementwise | None | None | 0 |
| log2 | pandas_numpy, polars | True | elementwise | None | None | 0 |
| log_abs | pandas_numpy, polars | True | elementwise | None | None | 0 |
| log_returns | pandas_numpy, polars | True | unknown | None | None | 0 |
| lt | pandas_numpy, polars | True | elementwise | None | None | 0 |
| lu_decompose | pandas_numpy | True | elementwise | None | None | 0 |
| mat_add | pandas_numpy | True | elementwise | None | None | 0 |
| mat_determinant | pandas_numpy | True | elementwise | None | None | 0 |
| mat_inverse | pandas_numpy | True | elementwise | None | None | 0 |
| mat_multiply | pandas_numpy | True | elementwise | None | None | 0 |
| mat_rank | pandas_numpy | True | elementwise | None | None | 0 |
| mat_subtract | pandas_numpy | True | elementwise | None | None | 0 |
| mat_transpose | pandas_numpy | True | elementwise | None | None | 0 |
| max_drawdown | pandas_numpy, polars | True | unknown | None | None | 0 |
| maximum | pandas_numpy, polars | True | elementwise | None | None | 0 |
| micro_amihud_hf | pandas_numpy, polars | True | ts | None | 2 | 0 |
| micro_bipower_var | pandas_numpy, polars | True | ts | None | 2 | 0 |
| micro_jump_indicator | pandas_numpy, polars | True | ts | None | 2 | 0 |
| micro_kyle_lambda | pandas_numpy, polars | True | ts | None | 2 | 0 |
| micro_mid_return | pandas_numpy, polars | True | ts | None | None | 1 |
| micro_realized_vol | pandas_numpy, polars | True | ts | None | 1 | 0 |
| micro_spread | pandas_numpy, polars | True | elementwise | None | None | 0 |
| micro_trade_imbalance | pandas_numpy, polars | True | ts | None | 1 | 0 |
| micro_vpin | pandas_numpy, polars | True | ts | None | 1 | 0 |
| minimum | pandas_numpy, polars | True | elementwise | None | None | 0 |
| multiply | pandas_numpy, polars, sql | True | elementwise | None | None | 0 |
| nan_to_num | pandas_numpy | True | elementwise | None | None | 0 |
| ne | pandas_numpy, polars | True | elementwise | None | None | 0 |
| neg | pandas_numpy, polars, sql | True | elementwise | None | None | 0 |
| next | pandas_numpy, polars | False | ts | None | None | -1 |
| norm | pandas_numpy | True | elementwise | None | None | 0 |
| norm_l1 | pandas_numpy | True | elementwise | None | None | 0 |
| norm_linf | pandas_numpy | True | elementwise | None | None | 0 |
| normalize | pandas_numpy, polars | True | cs | None | None | 0 |
| not_ | pandas_numpy, polars | True | elementwise | None | None | 0 |
| or_ | pandas_numpy, polars | True | elementwise | None | None | 0 |
| pacf | pandas_numpy | True | aggregate | None | 1 | 0 |
| pca | pandas_numpy | True | elementwise | None | None | 0 |
| pdf_chi2 | pandas_numpy | True | aggregate | None | 1 | 0 |
| pdf_f | pandas_numpy | True | aggregate | None | 1 | 0 |
| pdf_normal | pandas_numpy | True | aggregate | None | 1 | 0 |
| pdf_t | pandas_numpy | True | aggregate | None | 1 | 0 |
| phase | pandas_numpy | True | elementwise | None | None | 0 |
| polar | pandas_numpy | True | elementwise | None | None | 0 |
| power | pandas_numpy, polars | True | elementwise | None | None | 0 |
| prev | pandas_numpy, polars | True | ts | None | None | 0 |
| price_spread_deviation | pandas_numpy | True | ts | None | None | 0 |
| product | pandas_numpy | True | aggregate | None | 1 | 0 |
| protected_div | pandas_numpy | True | unknown | None | None | 0 |
| protected_log | pandas_numpy | True | unknown | None | None | 0 |
| protected_sqrt | pandas_numpy | True | unknown | None | None | 0 |
| qr_decompose | pandas_numpy | True | elementwise | None | None | 0 |
| quantile | pandas_numpy, polars | True | cs | None | None | 0 |
| quantile_normal | pandas_numpy | True | aggregate | None | 1 | 0 |
| quantile_t | pandas_numpy | True | aggregate | None | 1 | 0 |
| quarter | pandas_numpy | True | unknown | None | None | 0 |
| r_squared | pandas_numpy, polars | True | aggregate | None | 1 | 0 |
| rand_exp | pandas_numpy | True | aggregate | None | 1 | 0 |
| rand_lognormal | pandas_numpy | True | aggregate | None | 1 | 0 |
| rand_normal | pandas_numpy | True | aggregate | None | 1 | 0 |
| rand_poisson | pandas_numpy | True | aggregate | None | 1 | 0 |
| rand_uniform | pandas_numpy | True | aggregate | None | 1 | 0 |
| rank | pandas_numpy, polars, sql | True | cs | None | None | 0 |
| rank_corr | pandas_numpy, polars | True | ts | None | 2 | 0 |
| rank_transform | pandas_numpy | True | elementwise | None | None | 0 |
| rankavg_transform | pandas_numpy | True | elementwise | None | None | 0 |
| real | pandas_numpy | True | elementwise | None | None | 0 |
| real_turnover_rate | pandas_numpy, polars | True | ts | None | None | 0 |
| reciprocal | polars | True | elementwise | None | None | 0 |
| regress | pandas_numpy | True | aggregate | None | 1 | 0 |
| residual | pandas_numpy, polars | True | aggregate | None | 1 | 0 |
| residual_momentum_capm | pandas_numpy, polars | True | ts | None | None | 0 |
| reverse | pandas_numpy | True | elementwise | None | None | 0 |
| ridge | pandas_numpy | True | aggregate | None | 1 | 0 |
| rolling_beta_to_market | pandas_numpy, polars | True | ts | None | None | 0 |
| round | pandas_numpy, polars | True | elementwise | None | None | 0 |
| row_avg | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| row_beta | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| row_corr | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| row_count | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| row_kurt | pandas_numpy | True | cs | 0 | 1 | 0 |
| row_max | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| row_median | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| row_min | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| row_prod | pandas_numpy | True | cs | 0 | 1 | 0 |
| row_skew | pandas_numpy | True | cs | 0 | 1 | 0 |
| row_std | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| row_sum | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| row_var | pandas_numpy, polars | True | cs | 0 | 1 | 0 |
| sample | pandas_numpy | True | aggregate | None | 1 | 0 |
| saturate | pandas_numpy, polars | True | ts | None | None | 0 |
| scale | pandas_numpy, polars, sql | True | cs | None | None | 0 |
| sec | pandas_numpy, polars | True | elementwise | None | None | 0 |
| sem | pandas_numpy | True | aggregate | None | 1 | 0 |
| sharpe_ratio | pandas_numpy, polars | True | unknown | None | None | 0 |
| shuffle | pandas_numpy | False | unknown | None | None | 0 |
| sigmoid | pandas_numpy, polars | True | elementwise | None | None | 0 |
| sign | pandas_numpy, polars | True | elementwise | None | None | 0 |
| signed_log | pandas_numpy, polars | True | ts | None | None | 0 |
| signed_power | pandas_numpy, polars | True | ts | None | None | 0 |
| signed_sqrt | pandas_numpy | True | elementwise | None | None | 0 |
| sin | pandas_numpy, polars | True | elementwise | None | None | 0 |
| sinh | pandas_numpy, polars | True | elementwise | None | None | 0 |
| size_neutralize | pandas_numpy | True | cs | 0 | 1 | 0 |
| slope | pandas_numpy | True | aggregate | None | 1 | 0 |
| spearman_corr_test | pandas_numpy | True | hypothesis | None | None | 0 |
| sqr | pandas_numpy | True | elementwise | None | None | 0 |
| sqrt | pandas_numpy, polars, sql | True | elementwise | None | None | 0 |
| sqrt_abs | pandas_numpy | True | elementwise | None | None | 0 |
| square | pandas_numpy, polars | True | elementwise | None | None | 0 |
| stationarity_test | pandas_numpy | True | hypothesis | None | None | 0 |
| std_agg | pandas_numpy | True | aggregate | None | 1 | 0 |
| stdp | pandas_numpy | True | aggregate | None | 1 | 0 |
| subtract | pandas_numpy, polars, sql | True | elementwise | None | None | 0 |
| sum_agg | pandas_numpy | True | aggregate | None | 1 | 0 |
| svd | pandas_numpy | True | elementwise | None | None | 0 |
| tail_beta | pandas_numpy, polars | True | ts | None | None | 0 |
| tan | pandas_numpy, polars | True | elementwise | None | None | 0 |
| tanh | pandas_numpy, polars | True | elementwise | None | None | 0 |
| tm_top_n_avg | pandas_numpy | True | ts | None | None | 0 |
| tm_top_n_sum | pandas_numpy | True | ts | None | 1 | 0 |
| trade_when | pandas_numpy, polars | True | ts | None | None | 0 |
| truncate | pandas_numpy, polars | True | elementwise | None | None | 0 |
| ts_argmax | pandas_numpy, polars | True | ts | None | None | 0 |
| ts_argmin | pandas_numpy, polars | True | ts | None | None | 0 |
| ts_beta | pandas_numpy, polars, sql | True | ts | None | 2 | 0 |
| ts_bottom_n_avg | pandas_numpy | True | ts | None | None | 0 |
| ts_bottom_n_sum | pandas_numpy | True | ts | None | 1 | 0 |
| ts_corr | pandas_numpy, polars, sql | True | ts | None | 1 | 0 |
| ts_cov | pandas_numpy, polars | True | ts | None | None | 0 |
| ts_decay_exp_window | pandas_numpy, polars | True | ts | None | 1 | 0 |
| ts_decay_linear | pandas_numpy, polars | True | ts | None | 1 | 0 |
| ts_delay | pandas_numpy, polars, sql | True | ts | None | None | 1 |
| ts_delta | pandas_numpy, polars, sql | True | ts | None | None | 1 |
| ts_ema | pandas_numpy, polars, sql | True | ts | None | 1 | 0 |
| ts_kurt | pandas_numpy, polars | True | ts | None | None | 0 |
| ts_mad | pandas_numpy, polars, sql | True | ts | None | None | 0 |
| ts_max | pandas_numpy, polars, sql | True | ts | None | 1 | 0 |
| ts_max_buildup | pandas_numpy | True | ts | None | None | 0 |
| ts_mean | pandas_numpy, polars, sql | True | ts | None | 1 | 0 |
| ts_median | pandas_numpy, polars, sql | True | ts | None | None | 0 |
| ts_min | pandas_numpy, polars, sql | True | ts | None | 1 | 0 |
| ts_moment | pandas_numpy | True | ts | None | None | 0 |
| ts_pct | pandas_numpy, polars, sql | True | ts | None | None | 0 |
| ts_poly2_coeff | pandas_numpy | True | ts | None | None | 0 |
| ts_poly2_resid | pandas_numpy | True | ts | None | None | 0 |
| ts_product | pandas_numpy, polars | True | ts | None | None | 0 |
| ts_quantile | pandas_numpy, polars | True | ts | None | None | 0 |
| ts_rank | pandas_numpy, polars, sql | True | ts | None | 1 | 0 |
| ts_ratio | pandas_numpy | True | ts | None | None | 0 |
| ts_regression | pandas_numpy, polars | True | ts | None | None | 0 |
| ts_skew | pandas_numpy, polars | True | ts | None | None | 0 |
| ts_std | pandas_numpy, polars, sql | True | ts | None | 1 | 0 |
| ts_sum | pandas_numpy, polars, sql | True | ts | None | 1 | 0 |
| ts_sum_decay | pandas_numpy, polars | True | ts | None | 1 | 0 |
| ts_top_n_avg | pandas_numpy, polars | True | ts | None | None | 0 |
| ts_top_n_std | pandas_numpy, polars | True | ts | None | 1 | 0 |
| ts_topk_sum | pandas_numpy, polars | True | ts | None | 1 | 0 |
| ts_var | pandas_numpy, polars, sql | True | ts | None | None | 0 |
| ts_zscore | pandas_numpy, polars, sql | True | ts | None | None | 0 |
| ttest_one_sample | pandas_numpy | True | hypothesis | None | None | 0 |
| ttest_paired | pandas_numpy | True | hypothesis | None | None | 0 |
| ttest_two_samples | pandas_numpy | True | hypothesis | None | None | 0 |
| ttm | pandas_numpy | True | unknown | None | None | 0 |
| tukey_transform | pandas_numpy | True | elementwise | None | None | 0 |
| unitize | pandas_numpy | True | elementwise | None | None | 0 |
| unwrap | pandas_numpy | True | elementwise | None | None | 0 |
| van_der_waerden_transform | pandas_numpy | True | elementwise | None | None | 0 |
| varp | pandas_numpy | True | aggregate | None | 1 | 0 |
| volatility | pandas_numpy, polars | True | unknown | None | None | 0 |
| vp_weighted_price | pandas_numpy, polars | True | ts | None | None | 0 |
| vpmacd | pandas_numpy, polars | True | ts | None | None | 0 |
| vpmacd_signal | pandas_numpy, polars | True | ts | None | None | 0 |
| vwap | pandas_numpy, polars | True | unknown | None | None | 0 |
| wavelet | pandas_numpy | True | elementwise | None | None | 0 |
| wavelet_denoise | pandas_numpy | True | elementwise | None | None | 0 |
| wavg | pandas_numpy | True | aggregate | None | 1 | 0 |
| weighted_mean | pandas_numpy | True | elementwise | None | None | 0 |
| where | pandas_numpy, polars, sql | True | ts | None | None | 0 |
| winsorize | pandas_numpy, polars, sql | True | cs | None | None | 0 |
| winsorize_mean | pandas_numpy | True | elementwise | None | None | 0 |
| wsum | pandas_numpy | True | aggregate | None | 1 | 0 |
| yoy | pandas_numpy | True | unknown | None | None | 0 |
| zscore | pandas_numpy, polars, sql | True | cs | None | None | 0 |

## Stub / catalog-only

