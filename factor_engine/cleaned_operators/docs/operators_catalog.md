# Operators Catalog（自动生成）

> 从 `load_all()` 去重后的最终 runtime registry 生成。
> 日常因子 DSL 仅使用 `surface=daily`；其他工具见 `research_operators/`。

## 摘要

- canonical 总数：364
- daily：271
- research：91
- unsafe：0
- legacy：1

| canonical | surface | backends | aliases | pit_safe | scope | lookback | min_periods | lag |
|---|---|---|---|---|---|---|---|---|
| ACF | research | pandas_numpy, polars | acf | False | aggregate | None | 1 | 0 |
| ADX | daily | pandas_numpy, polars | ts_adx | False | unknown | None | None | 0 |
| ADXR | daily | pandas_numpy, polars | ts_adxr | True | ts | None | 2 | 0 |
| AROON | daily | pandas_numpy, polars | ts_aroon | True | ts | None | 2 | 0 |
| AROON_down | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| AROON_up | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| ATR | daily | pandas_numpy, polars | ts_atr | False | unknown | None | None | 0 |
| ATR_WILDER | daily | pandas_numpy, polars, sql | ts_atr_wilder | True | ts | None | 2 | 0 |
| BollingerBands | daily | pandas_numpy, polars | ts_bbands | True | ts | None | 1 | 0 |
| BollingerLower | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| BollingerUpper | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| CCI | daily | pandas_numpy, polars | ts_cci | False | unknown | None | None | 0 |
| DPO | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| KAMA | daily | pandas_numpy, polars |  | False | unknown | None | None | 0 |
| MACD | daily | pandas_numpy, polars | ts_macd | False | unknown | None | None | 0 |
| MACD_hist | daily | pandas_numpy, polars |  | False | unknown | None | None | 0 |
| MACD_line | daily | pandas_numpy, polars |  | False | unknown | None | None | 0 |
| MACD_signal | daily | pandas_numpy, polars |  | False | unknown | None | None | 0 |
| MOM | daily | pandas_numpy, polars | ts_mom | True | ts | None | 1 | 0 |
| Mode | research | pandas_numpy, polars | mode | False | aggregate | None | 1 | 0 |
| OBV | daily | pandas_numpy, polars | ts_obv | True | ts | None | 1 | 0 |
| ROC | daily | pandas_numpy, polars | ts_roc | True | ts | None | 1 | 0 |
| RSI | daily | pandas_numpy, polars | ts_rsi | False | unknown | None | None | 0 |
| RSI_WILDER | daily | pandas_numpy, polars, sql | ts_rsi_wilder | True | ts | None | 2 | 0 |
| Slope | daily | pandas_numpy, polars, sql |  | True | ts | None | None | 0 |
| StochasticD | daily | pandas_numpy, polars | ts_stochf | True | ts | None | 1 | 0 |
| StochasticK | daily | pandas_numpy, polars | ts_stoch | True | ts | None | 1 | 0 |
| TRIX | daily | pandas_numpy, polars | ts_trix | False | unknown | None | None | 0 |
| WMA | daily | pandas_numpy, polars, sql | ts_wma, wma | True | ts | None | 1 | 0 |
| WilliamsR | daily | pandas_numpy, polars | ts_willr | True | ts | None | 1 | 0 |
| abs | daily | pandas_numpy, polars, sql | ABS | True | elementwise | None | None | 0 |
| acos | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| add | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| aggr_top_n | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| and_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| arg | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| asin | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| at_imax | daily | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| at_imin | daily | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| atan | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| atan2 | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| autocorr | research | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| avg | daily | pandas_numpy, polars | mean_agg | True | aggregate | None | None | 0 |
| avg2 | research | pandas_numpy, polars | AVG2 | True | ts | None | 2 | 0 |
| bartlett_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| beta | daily | pandas_numpy, polars | Beta | False | aggregate | None | 1 | 0 |
| blom_transform | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| c_count | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| c_mean | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| c_percentile | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| c_std | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| c_sum | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cap | daily | pandas_numpy, polars, sql | CLIP, cap, clamp, clip | True | elementwise | None | None | 0 |
| causal_linear_extrapolate | research | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| cbrt | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cdf_chi2 | research | pandas_numpy |  | False | aggregate | None | 1 | 0 |
| cdf_f | research | pandas_numpy |  | False | aggregate | None | 1 | 0 |
| cdf_normal | research | pandas_numpy |  | False | aggregate | None | 1 | 0 |
| cdf_t | research | pandas_numpy |  | False | aggregate | None | 1 | 0 |
| ceil | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| chi_square_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| close_gap | daily | pandas_numpy |  | True | elementwise | None | None | 0 |
| coalesce | daily | pandas_numpy, polars, sql | COALESCE | True | elementwise | None | None | 0 |
| complex | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| conj | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| constant | internal | pandas_numpy |  | False | elementwise | None | None | 0 |
| convolve | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| corr_test | research | pandas_numpy, polars |  | True | hypothesis | None | None | 0 |
| correlate | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cos | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cosh | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| coskewness_to_market | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| cot | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| count | daily | pandas_numpy, polars, sql |  | False | aggregate | None | 1 | 0 |
| cs_demean | daily | pandas_numpy, polars, sql | CS_DEMEAN, c_demean | True | cs | None | None | 0 |
| cs_mad | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_mad_zscore | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_pct_rank | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_quantile | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_rank_01 | daily | pandas_numpy |  | True | cs | None | None | 0 |
| cs_regression | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| cs_resid | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| csc | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cube | legacy | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| cum_count | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| cum_delta | daily | pandas_numpy, polars, sql |  | True | ts | None | None | 0 |
| cum_first | daily | pandas_numpy, polars |  | True | ts | None | None | 0 |
| cum_max | daily | pandas_numpy, polars, sql |  | False | ts | None | None | 0 |
| cum_min | daily | pandas_numpy, polars, sql |  | False | ts | None | None | 0 |
| cum_positive_streak | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| cum_prod | daily | pandas_numpy, polars, sql |  | True | ts | None | None | 0 |
| cum_sum | daily | pandas_numpy, polars, sql |  | False | ts | None | 1 | 0 |
| cum_top_n_avg | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| cum_top_n_sum | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| cumulative_returns | daily | pandas_numpy, polars |  | False | unknown | None | None | 0 |
| current_ratio | daily | pandas_numpy |  | True | elementwise | None | None | 0 |
| debt_to_equity | daily | pandas_numpy |  | True | elementwise | None | None | 0 |
| decay_linear | daily | pandas_numpy, polars, sql | DECAY_LINEAR, TS_DECAY_LINEAR, decay_linear, ts_decay, ts_decay_linear | True | ts | None | 1 | 0 |
| decimate | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| delay | daily | pandas_numpy, polars, sql | DELAY, Delay, Ref, delay, m_delay, shift, ts_delay | True | ts | None | None | 1 |
| digital_count | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| div_or_default | daily | pandas_numpy, sql |  | False | unknown | None | None | 0 |
| div_or_null | daily | pandas_numpy |  | False | unknown | None | None | 0 |
| divide | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| downside_beta | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| dropna | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| durbin_watson_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| eig | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| ema | daily | pandas_numpy, polars, sql | EMA, ema, ewm_mean, ts_ema | True | ts | None | 1 | 0 |
| eq | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ewm | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ewm_corr | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| ewm_cov | daily | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| ewm_std | daily | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| ewm_var | daily | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| exp | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| exp_neg | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| expanding_max | daily | pandas_numpy, polars | cumulative_max | False | elementwise | None | None | 0 |
| expanding_mean | daily | pandas_numpy, polars, sql | cum_avg, cumulative_mean | False | elementwise | None | None | 0 |
| expanding_min | daily | pandas_numpy, polars | cumulative_min | False | elementwise | None | None | 0 |
| expanding_rank | daily | pandas_numpy, polars | cum_rank | True | ts | None | None | 0 |
| expanding_std | daily | pandas_numpy, polars, sql | cum_std | False | elementwise | None | None | 0 |
| expanding_sum | daily | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| expanding_zscore | daily | pandas_numpy, polars | cum_standardize | False | ts | None | None | 0 |
| ffill | daily | pandas_numpy, polars, sql | FillForward, cum_last, fillna_forward, last, last_not_null | True | elementwise | None | None | 0 |
| fft | research | pandas_numpy | dft | False | elementwise | None | None | 0 |
| fillna | daily | pandas_numpy, polars, sql | FillNA | True | elementwise | None | None | 0 |
| fillna_const | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| filter_bandpass | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| filter_highpass | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| filter_lowpass | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| filter_notch | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| first_not_null | daily | pandas_numpy, polars | first | False | aggregate | None | 1 | 0 |
| fix | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| flex_max | daily | pandas_numpy, polars | max | True | elementwise | None | None | 0 |
| flex_min | daily | pandas_numpy, polars | min | True | elementwise | None | None | 0 |
| floor | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| ge | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| geometric_mean | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| granger_causality | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| group_count | daily | pandas_numpy, sql |  | False | cs | 0 | 1 | 0 |
| group_decay_linear | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_max | daily | pandas_numpy, sql |  | False | cs | 0 | 1 | 0 |
| group_mean | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_min | daily | pandas_numpy, sql |  | False | cs | 0 | 1 | 0 |
| group_neutralize | daily | pandas_numpy, polars, sql | INDUSTRY_NEUTRAL, INDUSTRY_NEUTRALIZE, IND_NEUTRALIZE, NEUTRALIZE, c_neutralize, group_demean, ind_neutralize, industry_neutral, industry_neutralize, industry_size_neutralize, neutralize, panel_neutralize, size_industry_neutralize | True | cs | None | None | 0 |
| group_normalize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_percentile | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_rank | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_std | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_sum | daily | pandas_numpy, sql |  | False | cs | 0 | 1 | 0 |
| group_winsorize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| group_zscore | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| gt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| harmonic_mean | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| hump_decay | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| identity | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| idio_skew | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| idio_vol | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| ifft | research | pandas_numpy | idft | False | elementwise | None | None | 0 |
| ifnan | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| imag | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| intercept | daily | pandas_numpy, polars | Intercept | False | aggregate | None | 1 | 0 |
| intraday_vwap_deviation | daily | pandas_numpy |  | True | ts | None | 1 | 0 |
| inverse | daily | pandas_numpy, polars, sql | inv, reciprocal | True | elementwise | None | None | 0 |
| is_finite | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| is_inf | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| is_infinite | daily | pandas_numpy, polars, sql | IS_INFINITE | False | elementwise | None | None | 0 |
| is_nan | daily | pandas_numpy, polars, sql | IS_NAN | True | elementwise | None | None | 0 |
| is_not_null | daily | pandas_numpy, polars, sql | IS_NOT_NULL, is_not_null | True | elementwise | None | None | 0 |
| is_null | daily | pandas_numpy, polars, sql | IS_NULL, is_null | True | elementwise | None | None | 0 |
| jarque_bera_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| kendall_corr_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| kpss_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| ks_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| lasso | research | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| le | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| lerp | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| levene_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| lilliefors_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| log | daily | pandas_numpy, polars, sql | LOG, ln | True | elementwise | None | None | 0 |
| log10 | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| log2 | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| log_abs | daily | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| log_fill_invalid | daily | pandas_numpy, sql |  | False | unknown | None | None | 0 |
| log_returns | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 1 |
| lt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| lu_decompose | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| mat_add | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| mat_determinant | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| mat_inverse | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| mat_multiply | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| mat_rank | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| mat_subtract | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| mat_transpose | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| max_drawdown | research | pandas_numpy, polars |  | False | unknown | None | None | 0 |
| maximum | daily | pandas_numpy, polars, sql | fmax | True | elementwise | None | None | 0 |
| micro_amihud_hf | daily | pandas_numpy, polars |  | True | ts | None | None | 0 |
| micro_bipower_var | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| micro_jump_indicator | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| micro_kyle_lambda | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| micro_mid_return | daily | pandas_numpy, polars |  | True | ts | None | None | 1 |
| micro_realized_vol | daily | pandas_numpy, polars |  | True | ts | None | 2 | 0 |
| micro_spread | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| micro_trade_imbalance | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| micro_vpin | daily | pandas_numpy, polars |  | True | ts | None | 1 | 0 |
| minimum | daily | pandas_numpy, polars, sql | fmin | True | elementwise | None | None | 0 |
| multiply | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| nan_to_num | daily | pandas_numpy, polars, sql | NAN_TO_NUM | True | elementwise | None | None | 0 |
| ne | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| neg | daily | pandas_numpy, polars, sql | negate | True | elementwise | None | None | 0 |
| normalize | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| not_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| open_gap | daily | pandas_numpy |  | True | ts | None | None | 1 |
| operating_margin | daily | pandas_numpy |  | True | elementwise | None | None | 0 |
| or_ | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| pacf | research | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| pca | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| pdf_chi2 | research | pandas_numpy |  | False | aggregate | None | 1 | 0 |
| pdf_f | research | pandas_numpy |  | False | aggregate | None | 1 | 0 |
| pdf_normal | research | pandas_numpy |  | False | aggregate | None | 1 | 0 |
| pdf_t | research | pandas_numpy |  | False | aggregate | None | 1 | 0 |
| phase | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| polar | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| power | daily | pandas_numpy, polars, sql | POWER, pow | True | elementwise | None | None | 0 |
| prev | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| price_spread_deviation | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| product | daily | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| protected_div | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| protected_log | daily | pandas_numpy, polars, sql | log_clip_domain | True | elementwise | None | None | 0 |
| protected_sqrt | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| qr_decompose | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| quantile | daily | pandas_numpy, polars |  | True | cs | None | None | 0 |
| quantile_normal | research | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| quantile_t | research | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| quarter | research | pandas_numpy, polars | QUARTER | True | ts | None | 1 | 0 |
| quarter_from_cumulative | research | pandas_numpy | quarter_from_cumulative | True | ts | None | 1 | 0 |
| quick_ratio | daily | pandas_numpy |  | True | elementwise | None | None | 0 |
| r_squared | research | pandas_numpy, polars | R2 | False | aggregate | None | 1 | 0 |
| rank | daily | pandas_numpy, polars, sql | CS_RANK, RANK, c_rank, cs_rank, panel_rank | True | cs | None | None | 0 |
| rank_corr | daily | pandas_numpy, polars | RANKCORR, RANK_CORR, rankcorr | True | ts | None | 2 | 0 |
| rank_pct | daily | pandas_numpy, polars, sql |  | True | cs | None | None | 0 |
| rank_transform | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| rankavg_transform | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| real | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| real_turnover_rate | daily | pandas_numpy, polars |  | True | elementwise | None | None | 0 |
| regress | research | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| residual | research | pandas_numpy, polars | Residual | False | aggregate | None | 1 | 0 |
| residual_momentum_capm | daily | pandas_numpy, polars |  | True | ts | None | 5 | 0 |
| reverse | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ridge | research | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| rolling_beta | daily | pandas_numpy, polars, sql | ROLLING_BETA, ts_rolling_beta | True | ts | None | 2 | 0 |
| rolling_beta_to_market | daily | pandas_numpy, polars | FP_BETA, ROLLING_BETA_TO_MARKET, fp_beta | True | ts | None | 2 | 0 |
| round | daily | pandas_numpy, polars | ROUND | False | elementwise | None | None | 0 |
| row_avg | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_beta | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_corr | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_count | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_kurt | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_max | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_median | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_min | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_prod | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_skew | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_std | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_sum | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| row_var | research | pandas_numpy, polars |  | False | cs | 0 | 1 | 0 |
| safe_div | daily | pandas_numpy, polars, sql | safe_div, safe_div_null | True | elementwise | None | None | 0 |
| saturate | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| scale | daily | pandas_numpy, polars, sql | SCALE, c_scale | True | cs | None | None | 0 |
| sec | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| sem | research | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| sharpe_ratio | research | pandas_numpy, polars |  | False | unknown | None | None | 0 |
| sigmoid | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| sign | daily | pandas_numpy, polars, sql | SIGN | True | elementwise | None | None | 0 |
| signed_log | daily | pandas_numpy, polars, sql |  | False | ts | None | None | 0 |
| signed_power | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| signed_sqrt | daily | pandas_numpy, polars, sql |  | False | elementwise | None | None | 0 |
| sin | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| sinh | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| size_neutralize | daily | pandas_numpy, polars | CAP_NEUTRALIZE, MARKET_CAP_NEUTRALIZE, SIZE_NEUTRALIZE, cap_neutralize, market_cap_neutralize | False | cs | 0 | 1 | 0 |
| slope | daily | pandas_numpy, polars |  | True | ts | None | None | 0 |
| spearman_corr_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| sqrt | daily | pandas_numpy, polars, sql | SQRT | True | elementwise | None | None | 0 |
| sqrt_abs | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| square | daily | pandas_numpy, polars | sqr | False | elementwise | None | None | 0 |
| stationarity_test | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| std_agg | daily | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| stdp | daily | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| subtract | daily | pandas_numpy, polars, sql |  | True | elementwise | None | None | 0 |
| sum_agg | daily | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| svd | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| tail_beta | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| tan | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| tanh | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| tm_top_n_avg | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| tm_top_n_sum | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| trade_when | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| truncate | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| ts_argmax | daily | pandas_numpy, polars, sql | m_argmax, ts_arg_max | True | ts | None | None | 0 |
| ts_argmin | daily | pandas_numpy, polars, sql | m_argmin, ts_arg_min | True | ts | None | None | 0 |
| ts_autocorr | daily | pandas_numpy, polars, sql |  | True | ts | None | 3 | 1 |
| ts_beta | daily | pandas_numpy, polars, sql | m_beta | True | ts | None | 2 | 0 |
| ts_bottom_n_avg | daily | pandas_numpy, polars | m_bottom_n_avg | False | ts | None | None | 0 |
| ts_bottom_n_sum | daily | pandas_numpy, polars | m_bottom_n_sum | False | ts | None | 1 | 0 |
| ts_corr | daily | pandas_numpy, polars, sql | Corr, TS_CORR, corr, correlation, m_cor, ts_correlation | True | ts | None | 1 | 0 |
| ts_cov | daily | pandas_numpy, polars, sql | Cov, Covariance, TS_COV, cov, m_cov, ts_covariance | True | ts | None | 1 | 0 |
| ts_decay_exp_window | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_delta | daily | pandas_numpy, polars, sql | Delta, Diff, TS_DELTA, delta, deltas | True | ts | None | None | 1 |
| ts_kurt | daily | pandas_numpy, polars | Kurt, TS_KURT, kurt, m_kurt, ts_kurtosis | False | ts | None | None | 0 |
| ts_log_return | daily | pandas_numpy |  | True | ts | None | 2 | 1 |
| ts_mad | daily | pandas_numpy, polars, sql | Mad, m_mad, mad | False | ts | None | None | 0 |
| ts_max | daily | pandas_numpy, polars, sql | Max, TS_MAX, m_max, window_max | True | ts | None | 1 | 0 |
| ts_max_buildup | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_mean | daily | pandas_numpy, polars, sql | Mean, SMA, TS_MEAN, m_avg, ma, mean, move, running_mean, window_mean | True | ts | None | 1 | 0 |
| ts_median | daily | pandas_numpy, polars, sql | Median, m_median, median | True | ts | None | 1 | 0 |
| ts_min | daily | pandas_numpy, polars, sql | Min, TS_MIN, m_min, window_min | True | ts | None | 1 | 0 |
| ts_moment | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| ts_pct | daily | pandas_numpy, polars, sql | TS_PCT, m_pct_change, pct_change, returns, ts_return | True | ts | None | None | 1 |
| ts_poly2_coeff | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_poly2_resid | daily | pandas_numpy, polars |  | True | ts | None | 3 | 0 |
| ts_product | daily | pandas_numpy, polars, sql |  | True | ts | None | None | 0 |
| ts_quantile | daily | pandas_numpy, polars, sql | Percentile, TS_QUANTILE, m_percentile, percentile | False | ts | None | None | 0 |
| ts_rank | daily | pandas_numpy, polars, sql | TS_RANK, m_rank | True | ts | None | 1 | 0 |
| ts_ratio | daily | pandas_numpy, polars | ratios | False | ts | None | None | 0 |
| ts_regression_slope | daily | pandas_numpy, polars, sql | TS_REGRESSION_SLOPE, ts_regression | True | ts | None | None | 0 |
| ts_sharpe | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| ts_skew | daily | pandas_numpy, polars, sql | Skew, TS_SKEW, m_skew, skew, ts_skewness | False | ts | None | None | 0 |
| ts_std | daily | pandas_numpy, polars, sql | Std, TS_STD, m_std, running_std, std, std_n, ts_std_dev, ts_stddev, window_std | True | ts | None | 1 | 0 |
| ts_sum | daily | pandas_numpy, polars, sql | Sum, TS_SUM, m_sum, running_sum, sum, sum_n, window_sum | True | ts | None | 1 | 0 |
| ts_sum_decay | daily | pandas_numpy, polars |  | False | ts | None | 1 | 0 |
| ts_time_slope | daily | pandas_numpy, sql | Slope, TS_TIME_SLOPE, slope | True | ts | None | None | 0 |
| ts_top_n_avg | daily | pandas_numpy, polars | m_top_n_avg | False | ts | None | None | 0 |
| ts_top_n_std | daily | pandas_numpy, polars | m_top_n_std | False | ts | None | 1 | 0 |
| ts_topk_sum | daily | pandas_numpy, polars | TS_TOPK_SUM, m_top_n_sum | True | ts | None | 1 | 0 |
| ts_var | daily | pandas_numpy, polars, sql | Var, m_var, var | True | ts | None | 1 | 0 |
| ts_zscore | daily | pandas_numpy, polars, sql | m_zscore | True | ts | None | 1 | 0 |
| ttest_one_sample | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| ttest_paired | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| ttest_two_samples | research | pandas_numpy, polars |  | False | hypothesis | None | None | 0 |
| ttm | research | pandas_numpy, polars | TTM | True | ts | None | 1 | 0 |
| ttm_from_cumulative | research | pandas_numpy | ttm_from_cumulative | True | ts | None | 1 | 0 |
| ttm_from_quarterly | research | pandas_numpy | ttm_from_quarterly | True | ts | None | 1 | 0 |
| tukey_transform | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| unitize | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| unwrap | research | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| van_der_waerden_transform | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| varp | daily | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| volatility | daily | pandas_numpy, polars, sql |  | True | ts | None | 2 | 0 |
| vp_weighted_price | daily | pandas_numpy, polars |  | True | ts | None | None | 0 |
| vpmacd | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| vpmacd_signal | daily | pandas_numpy, polars |  | False | ts | None | None | 0 |
| vwap | daily | pandas_numpy, polars, sql |  | True | ts | None | 1 | 0 |
| wavelet | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| wavelet_denoise | research | pandas_numpy |  | False | elementwise | None | None | 0 |
| wavg | daily | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| weighted_mean | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| where | daily | pandas_numpy, polars, sql | IIF, WHERE, if, if_else, iif | True | elementwise | None | None | 0 |
| winsorize | daily | pandas_numpy, polars, sql | WINSORIZE, c_winsorize | True | cs | None | None | 0 |
| winsorize_mean | daily | pandas_numpy, polars |  | False | elementwise | None | None | 0 |
| wsum | daily | pandas_numpy, polars |  | False | aggregate | None | 1 | 0 |
| yoy | research | pandas_numpy, polars | YOY | True | ts | None | 5 | 0 |
| yoy_by_period | research | pandas_numpy | yoy_by_period | True | ts | None | 5 | 0 |
| zscore | daily | pandas_numpy, polars, sql | CS_ZSCORE, ZSCORE, c_zscore, cs_zscore, panel_standardize, panel_zscore, standardize | True | cs | None | None | 0 |
