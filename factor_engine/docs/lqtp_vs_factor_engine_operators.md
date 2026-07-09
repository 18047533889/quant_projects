# LQTP-backtest vs factor_engine 算子对照表

> 对照基准：LQTP-backtest 因子服务用户手册（2026-06-18 更新）  
> factor_engine：`cleaned_operators` DSL 白名单 **512** 名 / **378** 规范算子  
> 重新生成：`cd factor_engine && PYTHONPATH=. python3 scripts/generate_lqtp_comparison.py`

## 图例

| 符号 | 含义 |
|------|------|
| ✓ | 该侧手册/白名单明确列出或可直接使用 |
| — | 该侧未列出 / 无等价实现 |
| ✅ | **factor_engine 有，LQTP 手册未列** |

## 一、LQTP 手册算子 → factor_engine 覆盖情况

| 分类 | LQTP 手册算子 | LQTP | FE | 备注 |
|------|---------------|:----:|:--:|------|
| 截面 | rank / cs_rank | ✓ | ✓ |  |
| 截面 | zscore / cs_zscore | ✓ | ✓ |  |
| 截面 | cs_demean | ✓ | ✓ |  |
| 截面 | scale | ✓ | ✓ |  |
| 截面 | winsorize | ✓ | ✓ |  |
| 截面 | cs_resid | ✓ | ✓ SQL |  |
| 截面 | cs_regression(y,x[,mode]) | ✓ | ✓ SQL | mode 0/1/2 |
| 标量 | abs | ✓ | ✓ |  |
| 标量 | log | ✓ | ✓ |  |
| 标量 | sqrt | ✓ | ✓ |  |
| 标量 | sign | ✓ | ✓ |  |
| 标量 | round | ✓ | ✓ |  |
| 标量 | signed_sqrt | ✓ | ✓ |  |
| 标量 | sigmoid | ✓ | ✓ |  |
| 标量 | power | ✓ | ✓ |  |
| 标量 | cap | ✓ | ✓ |  |
| 标量 | where / iif | ✓ | ✓ |  |
| 标量 | is_null / is_nan | ✓ | ✓ |  |
| 标量 | nan_to_num | ✓ | ✓ |  |
| 标量 | coalesce | ✓ | ✓ |  |
| 时序 | ts_mean | ✓ | ✓ |  |
| 时序 | ts_sum | ✓ | ✓ |  |
| 时序 | ts_std | ✓ | ✓ |  |
| 时序 | ts_max | ✓ | ✓ |  |
| 时序 | ts_min | ✓ | ✓ |  |
| 时序 | ts_rank | ✓ | ✓ |  |
| 时序 | ts_delta | ✓ | ✓ |  |
| 时序 | ts_pct | ✓ | ✓ |  |
| 时序 | delay | ✓ | ✓ |  |
| 时序 | decay_linear / ts_decay_linear | ✓ | ✓ |  |
| 时序 | ema / EMA | ✓ | ✓ |  |
| 时序 | price_spread_deviation | ✓ | — |  |
| 时序 | ts_corr | ✓ | ✓ |  |
| 时序 | ts_cov | ✓ | ✓ |  |
| 时序 | ts_regression_slope | ✓ | ✓ |  |
| 时序 | ts_quantile | ✓ | ✓ |  |
| 时序 | ts_skew | ✓ | ✓ |  |
| 时序 | ts_kurt | ✓ | ✓ |  |
| 时序 | ts_moment | ✓ | — |  |
| 时序 | ts_topk_sum | ✓ | ✓ |  |
| 时序 | ts_rank_corr / rankcorr | ✓ | — |  |
| 时序 | ts_poly2_coeff | ✓ | — |  |
| 时序 | ts_poly2_resid | ✓ | — |  |
| 时序 | digital_count | ✓ | — |  |
| 时序 | ts_max_buildup | ✓ | — |  |
| 时序 | ts_argmax | ✓ | ✓ |  |
| 时序 | ts_argmin | ✓ | ✓ |  |
| 市场 | benchmark_index(index) | ✓ | — | FE 无内置数据源/聚合入口 |
| 市场 | rolling_beta_to_market | ✓ | ✓ |  |
| 市场 | fp_beta | ✓ | ✓ |  |
| 市场 | downside_beta | ✓ | ✓ |  |
| 市场 | tail_beta | ✓ | ✓ |  |
| 市场 | residual_momentum_capm | ✓ | ✓ |  |
| 市场 | coskewness_to_market | ✓ | ✓ |  |
| 市场 | idio_vol | ✓ | ✓ |  |
| 市场 | idio_skew | ✓ | ✓ |  |
| 财报 | ttm | ✓ | ✓ |  |
| 财报 | quarter | ✓ | ✓ |  |
| 财报 | yoy | ✓ | ✓ |  |
| 财报 | avg2 | ✓ | ✓ |  |
| 中性化 | industry_neutralize 及别名 | ✓ | ✓ |  |
| 中性化 | size_neutralize 及别名 | ✓ | ✓ |  |
| 中性化 | neutralize 组合 | ✓ | ✓ |  |
| 分钟/L2 | minute_bar | ✓ | — | FE 无内置数据源/聚合入口 |
| 分钟/L2 | l2_sum / l2_sum_if | ✓ | — | FE 无内置数据源/聚合入口 |
| 分钟/L2 | l2_count / l2_count_if | ✓ | — | FE 无内置数据源/聚合入口 |
| 分钟/L2 | real_turnover_rate | ✓ | ✓ |  |
| YAML模板 | safe_div | ✓ | ✓ |  |
| YAML模板 | nullif_zero | ✓ | — | LQTP functions.yaml 预置；FE 需手写 |
| YAML模板 | safe_log | ✓ | ✓ |  |
| YAML模板 | clean | ✓ | ✓ |  |
| YAML模板 | ma / sum_n / std_n / delta / pct_change | ✓ | ✓ |  |
| YAML模板 | daily_return 等收益模板 | ✓ | ✓ |  |
| YAML模板 | momentum / reversal 等 | ✓ | ✓ |  |
| YAML模板 | atr / realized_vol / downside_vol | ✓ | ✓ |  |
| YAML模板 | volume_ma / volume_ratio 等 | ✓ | ✓ |  |
| YAML模板 | vwap_gap / illiquidity / 量价相关模板 | ✓ | — | LQTP functions.yaml 预置；FE 需手写 |
| YAML模板 | return_rank / volume_rank 等 | ✓ | ✓ |  |
| YAML模板 | quality_mask / neutral_return 等 | ✓ | — | LQTP functions.yaml 预置；FE 需手写 |
| YAML模板 | L2 成交模板 | ✓ | — | LQTP functions.yaml 预置；FE 需手写 |

**小结**：LQTP 手册 **80** 条；FE 覆盖 **63** 条，缺口 **17** 条。

## 二、factor_engine 有而 LQTP 手册未列的算子（✅）

**合计 301 个 DSL 名**（含别名）。

| factor_engine 分类 | FE独有算子（节选） | 数量 |
|-------------------|-------------------|:----:|
| 技术信号 | ✅ `ADX`, `ADXR`, `AROON`, `AROON_down`, `AROON_up`, `BollingerBands` …（+24） | 30 |
| 时序滚动 | ✅ `aggr_top_n`, `cum_top_n_avg`, `cum_top_n_sum`, `m_beta`, `m_bottom_n_avg`, `m_bottom_n_sum` …（+13） | 19 |
| 统计与回归 | ✅ `ACF`, `autocorr`, `avg`, `bartlett_test`, `Beta`, `cdf_chi2` …（+61） | 67 |
| 截面变换 | ✅ `c_count`, `c_mean`, `c_percentile`, `c_std`, `c_sum`, `row_avg` …（+12） | 18 |
| 分组中性化 | ✅ `deltas`, `group_decay_linear`, `group_demean`, `group_mean`, `group_normalize`, `group_percentile` …（+10） | 16 |
| 价量衍生 | ✅ `max_drawdown`, `sharpe_ratio`, `vwap` | 3 |
| 数据清洗 | ✅ `bfill`, `dropna`, `ewm`, `ewm_corr`, `ewm_cov`, `ewm_mean` …（+17） | 23 |
| 滞后 / 差分 / 累计 | ✅ `cum_avg`, `cum_count`, `cum_delta`, `cum_first`, `cum_last`, `cum_max` …（+10） | 16 |
| 元素级数学 | ✅ `acos`, `add`, `and_`, `arg`, `asin`, `atan` …（+100） | 106 |

### 技术信号（30）

| 算子 | FE独有 |
|------|:------:|
| `ADX` | ✅ |
| `ADXR` | ✅ |
| `AROON` | ✅ |
| `AROON_down` | ✅ |
| `AROON_up` | ✅ |
| `BollingerBands` | ✅ |
| `BollingerLower` | ✅ |
| `BollingerUpper` | ✅ |
| `CCI` | ✅ |
| `DPO` | ✅ |
| `hump_decay` | ✅ |
| `ifnan` | ✅ |
| `KAMA` | ✅ |
| `MACD` | ✅ |
| `MACD_hist` | ✅ |
| `MACD_line` | ✅ |
| `MACD_signal` | ✅ |
| `OBV` | ✅ |
| `ROC` | ✅ |
| `RSI` | ✅ |
| `saturate` | ✅ |
| `signed_log` | ✅ |
| `signed_power` | ✅ |
| `StochasticD` | ✅ |
| `StochasticK` | ✅ |
| `TRIX` | ✅ |
| `ts_willr` | ✅ |
| `vp_weighted_price` | ✅ |
| `vpmacd` | ✅ |
| `vpmacd_signal` | ✅ |

### 时序滚动（19）

| 算子 | FE独有 |
|------|:------:|
| `aggr_top_n` | ✅ |
| `cum_top_n_avg` | ✅ |
| `cum_top_n_sum` | ✅ |
| `m_beta` | ✅ |
| `m_bottom_n_avg` | ✅ |
| `m_bottom_n_sum` | ✅ |
| `m_mad` | ✅ |
| `m_median` | ✅ |
| `m_top_n_avg` | ✅ |
| `m_top_n_std` | ✅ |
| `m_var` | ✅ |
| `m_zscore` | ✅ |
| `tm_top_n_avg` | ✅ |
| `ts_decay` | ✅ |
| `ts_decay_exp_window` | ✅ |
| `ts_product` | ✅ |
| `ts_sum_decay` | ✅ |
| `ts_zscore` | ✅ |
| `WMA` | ✅ |

### 统计与回归（67）

| 算子 | FE独有 |
|------|:------:|
| `ACF` | ✅ |
| `autocorr` | ✅ |
| `avg` | ✅ |
| `bartlett_test` | ✅ |
| `Beta` | ✅ |
| `cdf_chi2` | ✅ |
| `cdf_f` | ✅ |
| `cdf_normal` | ✅ |
| `cdf_t` | ✅ |
| `chi_square_test` | ✅ |
| `corr_test` | ✅ |
| `count` | ✅ |
| `Covariance` | ✅ |
| `durbin_watson_test` | ✅ |
| `first` | ✅ |
| `first_not_null` | ✅ |
| `granger_causality` | ✅ |
| `Intercept` | ✅ |
| `jarque_bera_test` | ✅ |
| `kendall_corr_test` | ✅ |
| `kpss_test` | ✅ |
| `ks_test` | ✅ |
| `lasso` | ✅ |
| `last` | ✅ |
| `last_not_null` | ✅ |
| `levene_test` | ✅ |
| `lilliefors_test` | ✅ |
| `Mad` | ✅ |
| `mean_agg` | ✅ |
| `Median` | ✅ |
| `Mode` | ✅ |
| `pacf` | ✅ |
| `pdf_chi2` | ✅ |
| `pdf_f` | ✅ |
| `pdf_normal` | ✅ |
| `pdf_t` | ✅ |
| `Percentile` | ✅ |
| `product` | ✅ |
| `quantile_normal` | ✅ |
| `quantile_t` | ✅ |
| `R2` | ✅ |
| `r_squared` | ✅ |
| `rand_exp` | ✅ |
| `rand_lognormal` | ✅ |
| `rand_normal` | ✅ |
| `rand_poisson` | ✅ |
| `rand_uniform` | ✅ |
| `regress` | ✅ |
| `Residual` | ✅ |
| `ridge` | ✅ |
| `sample` | ✅ |
| `sem` | ✅ |
| `shuffle` | ✅ |
| `Slope` | ✅ |
| `spearman_corr_test` | ✅ |
| `stationarity_test` | ✅ |
| `std_agg` | ✅ |
| `stdp` | ✅ |
| `Sum` | ✅ |
| `sum_agg` | ✅ |
| `ttest_one_sample` | ✅ |
| `ttest_paired` | ✅ |
| `ttest_two_samples` | ✅ |
| `Var` | ✅ |
| `varp` | ✅ |
| `wavg` | ✅ |
| `wsum` | ✅ |

### 截面变换（18）

| 算子 | FE独有 |
|------|:------:|
| `c_count` | ✅ |
| `c_mean` | ✅ |
| `c_percentile` | ✅ |
| `c_std` | ✅ |
| `c_sum` | ✅ |
| `row_avg` | ✅ |
| `row_beta` | ✅ |
| `row_corr` | ✅ |
| `row_count` | ✅ |
| `row_kurt` | ✅ |
| `row_max` | ✅ |
| `row_median` | ✅ |
| `row_min` | ✅ |
| `row_prod` | ✅ |
| `row_skew` | ✅ |
| `row_std` | ✅ |
| `row_sum` | ✅ |
| `row_var` | ✅ |

### 分组中性化（16）

| 算子 | FE独有 |
|------|:------:|
| `deltas` | ✅ |
| `group_decay_linear` | ✅ |
| `group_demean` | ✅ |
| `group_mean` | ✅ |
| `group_normalize` | ✅ |
| `group_percentile` | ✅ |
| `group_rank` | ✅ |
| `group_std` | ✅ |
| `group_winsorize` | ✅ |
| `group_zscore` | ✅ |
| `move` | ✅ |
| `panel_neutralize` | ✅ |
| `panel_rank` | ✅ |
| `panel_standardize` | ✅ |
| `panel_zscore` | ✅ |
| `ratios` | ✅ |

### 价量衍生（3）

| 算子 | FE独有 |
|------|:------:|
| `max_drawdown` | ✅ |
| `sharpe_ratio` | ✅ |
| `vwap` | ✅ |

### 数据清洗（23）

| 算子 | FE独有 |
|------|:------:|
| `bfill` | ✅ |
| `dropna` | ✅ |
| `ewm` | ✅ |
| `ewm_corr` | ✅ |
| `ewm_cov` | ✅ |
| `ewm_mean` | ✅ |
| `ewm_std` | ✅ |
| `ewm_var` | ✅ |
| `expanding_max` | ✅ |
| `expanding_mean` | ✅ |
| `expanding_min` | ✅ |
| `expanding_rank` | ✅ |
| `expanding_std` | ✅ |
| `expanding_sum` | ✅ |
| `ffill` | ✅ |
| `fillna_const` | ✅ |
| `fillna_interpolate` | ✅ |
| `is_inf` | ✅ |
| `protected_sqrt` | ✅ |
| `window_max` | ✅ |
| `window_min` | ✅ |
| `window_std` | ✅ |
| `window_sum` | ✅ |

### 滞后 / 差分 / 累计（16）

| 算子 | FE独有 |
|------|:------:|
| `cum_avg` | ✅ |
| `cum_count` | ✅ |
| `cum_delta` | ✅ |
| `cum_first` | ✅ |
| `cum_last` | ✅ |
| `cum_max` | ✅ |
| `cum_min` | ✅ |
| `cum_positive_streak` | ✅ |
| `cum_prod` | ✅ |
| `cum_rank` | ✅ |
| `cum_standardize` | ✅ |
| `cum_std` | ✅ |
| `cum_sum` | ✅ |
| `Lead` | ✅ |
| `next` | ✅ |
| `prev` | ✅ |

### 元素级数学（106）

| 算子 | FE独有 |
|------|:------:|
| `acos` | ✅ |
| `add` | ✅ |
| `and_` | ✅ |
| `arg` | ✅ |
| `asin` | ✅ |
| `atan` | ✅ |
| `atan2` | ✅ |
| `blom_transform` | ✅ |
| `cbrt` | ✅ |
| `ceil` | ✅ |
| `complex` | ✅ |
| `conj` | ✅ |
| `constant` | ✅ |
| `convolve` | ✅ |
| `correlate` | ✅ |
| `cos` | ✅ |
| `cosh` | ✅ |
| `cot` | ✅ |
| `csc` | ✅ |
| `cube` | ✅ |
| `cumulative_max` | ✅ |
| `cumulative_mean` | ✅ |
| `cumulative_min` | ✅ |
| `decimate` | ✅ |
| `dft` | ✅ |
| `divide` | ✅ |
| `eig` | ✅ |
| `eq` | ✅ |
| `exp` | ✅ |
| `exp_neg` | ✅ |
| `fft` | ✅ |
| `filter_bandpass` | ✅ |
| `filter_highpass` | ✅ |
| `filter_lowpass` | ✅ |
| `filter_notch` | ✅ |
| `fix` | ✅ |
| `floor` | ✅ |
| `fmax` | ✅ |
| `fmin` | ✅ |
| `ge` | ✅ |
| `geometric_mean` | ✅ |
| `gt` | ✅ |
| `harmonic_mean` | ✅ |
| `identity` | ✅ |
| `idft` | ✅ |
| `ifft` | ✅ |
| `imag` | ✅ |
| `interpolate` | ✅ |
| `inv` | ✅ |
| `inverse` | ✅ |
| `le` | ✅ |
| `lerp` | ✅ |
| `log10` | ✅ |
| `log2` | ✅ |
| `log_abs` | ✅ |
| `lt` | ✅ |
| `lu_decompose` | ✅ |
| `mat_add` | ✅ |
| `mat_determinant` | ✅ |
| `mat_inverse` | ✅ |
| `mat_multiply` | ✅ |
| `mat_rank` | ✅ |
| `mat_subtract` | ✅ |
| `mat_transpose` | ✅ |
| `multiply` | ✅ |
| `ne` | ✅ |
| `neg` | ✅ |
| `negate` | ✅ |
| `norm` | ✅ |
| `norm_l1` | ✅ |
| `norm_linf` | ✅ |
| `normalize` | ✅ |
| `not_` | ✅ |
| `or_` | ✅ |
| `pca` | ✅ |
| `phase` | ✅ |
| `polar` | ✅ |
| `qr_decompose` | ✅ |
| `rank_transform` | ✅ |
| `rankavg_transform` | ✅ |
| `real` | ✅ |
| `reciprocal` | ✅ |
| `reverse` | ✅ |
| `running_mean` | ✅ |
| `running_std` | ✅ |
| `running_sum` | ✅ |
| `sec` | ✅ |
| `sin` | ✅ |
| `sinh` | ✅ |
| `sqr` | ✅ |
| `sqrt_abs` | ✅ |
| `square` | ✅ |
| `standardize` | ✅ |
| `subtract` | ✅ |
| `svd` | ✅ |
| `tan` | ✅ |
| `tanh` | ✅ |
| `truncate` | ✅ |
| `tukey_transform` | ✅ |
| `unitize` | ✅ |
| `unwrap` | ✅ |
| `van_der_waerden_transform` | ✅ |
| `wavelet` | ✅ |
| `wavelet_denoise` | ✅ |
| `weighted_mean` | ✅ |
| `winsorize_mean` | ✅ |

