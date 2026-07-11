# 算子库清洗汇总目录

> 可独立交付目录；三源并集、语义去重；runtime 与 catalog 分层标注。
>
> **第 31 版说明**：`selected_source` 列中 **`api/operators`** 为 **历史迁移前路径**（该包已删除）。当前 runtime 均在 **`cleaned_operators/`**；**投递白名单**以 [`../docs/dsl_allowlist.json`](../docs/dsl_allowlist.json) / [`../docs/dsl_operators_reference.md`](../docs/dsl_operators_reference.md) 为准（`status=stub` 与 `api_expr_only` 条目 **不在** `build_dsl_allowlist()`）。

## 审计摘要

| 指标 | 数量 |
|------|------|
| 三源并集条目（去重后） | 579 |
| implemented（runtime 有实现） | 441 |
| alias（仅别名） | 116 |
| stub（api 占位） | 79 |
| api_expr_only（DSL Expr 构造） | 59 |
| doc_only（真实未覆盖） | 0 |
| 已实现 canonical 算子（业务表） | 392 |
| 注册别名总数 | 116 |

## 覆盖检查

| 检查项 | 结果 |
|--------|------|
| factor_dsl_np 遗漏数 | 0 |
| 文档遗漏数 | 0 |
| LQTP 遗漏数 | 0 |
| 重复 canonical+backend | 0 |

## 状态说明

| 状态 | 含义 |
|------|------|
| implemented | cleaned_operators 有 pandas_numpy 和/或 polars runtime 实现 |
| alias | 作为 canonical 的别名保留，不单独实现 |
| stub | catalog 语义预留，无 runtime，**不在 DSL 白名单** |
| api_expr_only | **历史标注**（原 api/operators Expr-only）；第 31 版后多数已迁入 cleaned 或不再入白名单 |
| doc_only | 文档/注册提及，且不属于 implemented/alias/stub/api_expr_only |

## 各业务分类 implemented

### elementwise_math

**基础数学与元素函数**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `abs` | `ABS` | pandas_numpy | `factor_dsl_np:math/elementary.py:abs` | - | implemented |
| `acos` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:acos` | - | implemented |
| `add` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `and_` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `arg` | - | pandas_numpy | `factor_dsl_np:math/complex_ops.py:arg` | - | implemented |
| `asin` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:asin` | - | implemented |
| `atan` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:atan` | - | implemented |
| `atan2` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:atan2` | - | implemented |
| `blom_transform` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:blom_transform` | - | implemented |
| `cap` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `cbrt` | - | pandas_numpy | `factor_dsl_np:math/elementary.py:cbrt` | - | implemented |
| `ceil` | - | pandas_numpy | `factor_dsl_np:math/rounding.py:ceil` | - | implemented |
| `clip` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:clip` | - | implemented |
| `coalesce` | `COALESCE` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `complex` | - | pandas_numpy | `factor_dsl_np:math/complex_ops.py:complex` | - | implemented |
| `conj` | - | pandas_numpy | `factor_dsl_np:math/complex_ops.py:conj` | - | implemented |
| `constant` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:constant` | - | implemented |
| `convolve` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:convolve` | - | implemented |
| `correlate` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:correlate` | - | implemented |
| `cos` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:cos` | - | implemented |
| `cosh` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:cosh` | - | implemented |
| `cot` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:cot` | - | implemented |
| `csc` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:csc` | - | implemented |
| `cube` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:cube` | - | implemented |
| `cumulative_max` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:cumulative_max` | - | implemented |
| `cumulative_mean` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:cumulative_mean` | - | implemented |
| `cumulative_min` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:cumulative_min` | - | implemented |
| `decimate` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:decimate` | - | implemented |
| `dft` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:dft` | - | implemented |
| `divide` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `eig` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:eig` | - | implemented |
| `eq` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `exp` | - | pandas_numpy | `factor_dsl_np:math/elementary.py:exp` | - | implemented |
| `exp_neg` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:exp_neg` | - | implemented |
| `fft` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:fft` | - | implemented |
| `filter_bandpass` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:filter_bandpass` | - | implemented |
| `filter_highpass` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:filter_highpass` | - | implemented |
| `filter_lowpass` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:filter_lowpass` | - | implemented |
| `filter_notch` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:filter_notch` | - | implemented |
| `fix` | - | pandas_numpy | `factor_dsl_np:math/rounding.py:fix` | - | implemented |
| `floor` | - | pandas_numpy | `factor_dsl_np:math/rounding.py:floor` | - | implemented |
| `fmax` | - | pandas_numpy | `factor_dsl_np:math/special.py:fmax` | - | implemented |
| `fmin` | - | pandas_numpy | `factor_dsl_np:math/special.py:fmin` | - | implemented |
| `ge` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `geometric_mean` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:geometric_mean` | - | implemented |
| `gt` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `harmonic_mean` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:harmonic_mean` | - | implemented |
| `identity` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:identity` | - | implemented |
| `idft` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:idft` | - | implemented |
| `ifft` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:ifft` | - | implemented |
| `imag` | - | pandas_numpy | `factor_dsl_np:math/complex_ops.py:imag` | - | implemented |
| `interpolate` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:interpolate` | - | implemented |
| `inv` | - | pandas_numpy | `factor_dsl_np:math/elementary.py:inv` | - | implemented |
| `inverse` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `le` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `lerp` | - | pandas_numpy | `factor_dsl_np:math/special.py:lerp` | - | implemented |
| `log` | `LOG`, `ln` | pandas_numpy | `factor_dsl_np:math/elementary.py:log` | `ln` | implemented |
| `log10` | - | pandas_numpy | `factor_dsl_np:math/elementary.py:log10` | - | implemented |
| `log2` | - | pandas_numpy | `factor_dsl_np:math/elementary.py:log2` | - | implemented |
| `log_abs` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:log_abs` | - | implemented |
| `lt` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `lu_decompose` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:lu_decompose` | - | implemented |
| `mat_add` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:mat_add` | - | implemented |
| `mat_determinant` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:mat_determinant` | - | implemented |
| `mat_inverse` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:mat_inverse` | - | implemented |
| `mat_multiply` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:mat_multiply` | - | implemented |
| `mat_rank` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:mat_rank` | - | implemented |
| `mat_subtract` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:mat_subtract` | - | implemented |
| `mat_transpose` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:mat_transpose` | - | implemented |
| `multiply` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `ne` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `neg` | - | pandas_numpy | `factor_dsl_np:math/elementary.py:neg` | - | implemented |
| `negate` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:negate` | - | implemented |
| `norm` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:norm` | - | implemented |
| `norm_l1` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:norm_l1` | - | implemented |
| `norm_linf` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:norm_linf` | - | implemented |
| `normalize` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:normalize` | - | implemented |
| `not_` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `or_` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `pca` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:pca` | - | implemented |
| `phase` | - | pandas_numpy | `factor_dsl_np:math/complex_ops.py:phase` | - | implemented |
| `polar` | - | pandas_numpy | `factor_dsl_np:math/complex_ops.py:polar` | - | implemented |
| `power` | `POWER` | pandas_numpy | `factor_dsl_np:math/elementary.py:pow` | - | implemented |
| `qr_decompose` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:qr_decompose` | - | implemented |
| `rank_transform` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:rank_transform` | - | implemented |
| `rankavg_transform` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:rankavg_transform` | - | implemented |
| `real` | - | pandas_numpy | `factor_dsl_np:math/complex_ops.py:real` | - | implemented |
| `reciprocal` | - | pandas_numpy | `factor_dsl_np:math/elementary.py:reciprocal` | - | implemented |
| `reverse` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `round` | `ROUND` | pandas_numpy | `factor_dsl_np:math/rounding.py:round` | - | implemented |
| `running_mean` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:running_mean` | - | implemented |
| `running_std` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:running_std` | - | implemented |
| `running_sum` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:running_sum` | - | implemented |
| `sec` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:sec` | - | implemented |
| `sigmoid` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `sign` | `SIGN` | pandas_numpy | `factor_dsl_np:math/elementary.py:sign` | - | implemented |
| `signed_sqrt` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `sin` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:sin` | - | implemented |
| `sinh` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:sinh` | - | implemented |
| `sqr` | - | pandas_numpy | `factor_dsl_np:math/elementary.py:sqr` | - | implemented |
| `sqrt` | `SQRT` | pandas_numpy | `factor_dsl_np:math/elementary.py:sqrt` | - | implemented |
| `sqrt_abs` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:sqrt_abs` | - | implemented |
| `square` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:square` | - | implemented |
| `standardize` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:standardize` | - | implemented |
| `subtract` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `svd` | - | pandas_numpy | `factor_dsl_np:math/matrix_ops.py:svd` | - | implemented |
| `tan` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:tan` | - | implemented |
| `tanh` | - | pandas_numpy | `factor_dsl_np:math/trigonometric.py:tanh` | - | implemented |
| `truncate` | - | pandas_numpy | `factor_dsl_np:math/rounding.py:truncate` | - | implemented |
| `tukey_transform` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:tukey_transform` | - | implemented |
| `unitize` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:unitize` | - | implemented |
| `unwrap` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:unwrap` | - | implemented |
| `van_der_waerden_transform` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:van_der_waerden_transform` | - | implemented |
| `wavelet` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:wavelet` | - | implemented |
| `wavelet_denoise` | - | pandas_numpy | `factor_dsl_np:math/fourier_ops.py:wavelet_denoise` | - | implemented |
| `weighted_mean` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:weighted_mean` | - | implemented |
| `winsorize` | `WINSORIZE` | pandas_numpy, polars | `factor_dsl_np:math/utility_ops.py:winsorize` | `c_winsorize` | implemented |
| `winsorize_mean` | - | pandas_numpy | `factor_dsl_np:math/utility_ops.py:winsorize_mean` | - | implemented |

### time_series

**时序滚动**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `SMA` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:SMA` | - | implemented |
| `WMA` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:WMA` | - | implemented |
| `aggr_top_n` | - | pandas_numpy | `factor_dsl_np:time_series/topn_ops.py:aggr_top_n` | - | implemented |
| `cum_top_n_avg` | - | pandas_numpy | `factor_dsl_np:time_series/topn_ops.py:cum_top_n_avg` | - | implemented |
| `cum_top_n_sum` | - | pandas_numpy | `factor_dsl_np:time_series/topn_ops.py:cum_top_n_sum` | - | implemented |
| `decay_linear` | `DECAY_LINEAR`, `TS_DECAY_LINEAR` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_decay_linear` | - | implemented |
| `digital_count` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ema` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:EMA` | - | implemented |
| `m_argmax` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:m_argmax` | - | implemented |
| `m_argmin` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:m_argmin` | - | implemented |
| `m_beta` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:m_beta` | - | implemented |
| `m_bottom_n_avg` | - | pandas_numpy | `factor_dsl_np:time_series/topn_ops.py:m_bottom_n_avg` | - | implemented |
| `m_bottom_n_sum` | - | pandas_numpy | `factor_dsl_np:time_series/topn_ops.py:m_bottom_n_sum` | - | implemented |
| `m_mad` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:m_mad` | - | implemented |
| `m_median` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:m_median` | - | implemented |
| `m_pct_change` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:m_pct_change` | - | implemented |
| `m_percentile` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:m_percentile` | - | implemented |
| `m_top_n_avg` | - | pandas_numpy | `factor_dsl_np:time_series/topn_ops.py:m_top_n_avg` | - | implemented |
| `m_top_n_std` | - | pandas_numpy | `factor_dsl_np:time_series/topn_ops.py:m_top_n_std` | - | implemented |
| `m_top_n_sum` | - | pandas_numpy | `factor_dsl_np:time_series/topn_ops.py:m_top_n_sum` | - | implemented |
| `m_var` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:m_var` | - | implemented |
| `m_zscore` | - | pandas_numpy | `factor_dsl_np:time_series/m_ops.py:m_zscore` | - | implemented |
| `price_spread_deviation` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `rank_corr` | `RANKCORR`, `RANK_CORR`, `rankcorr` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `tm_top_n_avg` | - | pandas_numpy | `factor_dsl_np:time_series/topn_ops.py:tm_top_n_avg` | - | implemented |
| `tm_top_n_sum` | - | pandas_numpy | `factor_dsl_np:time_series/topn_ops.py:tm_top_n_sum` | - | implemented |
| `ts_argmax` | `ts_arg_max` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ts_argmin` | `ts_arg_min` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ts_corr` | `TS_CORR`, `correlation`, `m_cor`, `ts_correlation` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_corr` | `correlation`, `m_cor`, `ts_correlation` | implemented |
| `ts_cov` | `TS_COV`, `m_cov`, `ts_covariance` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_cov` | `m_cov`, `ts_covariance` | implemented |
| `ts_decay` | - | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_decay` | - | implemented |
| `ts_decay_exp_window` | - | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_decay_exp_window` | - | implemented |
| `ts_delay` | `DELAY`, `Delay`, `Ref`, `delay`, `m_delay`, `shift` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_delay` | `Delay`, `Ref`, `m_delay`, `shift` | implemented |
| `ts_delta` | `Delta`, `Diff`, `TS_DELTA`, `pct_change` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_delta` | `Delta`, `pct_change` | implemented |
| `ts_kurt` | `TS_KURT` | pandas_numpy, polars | `factor_dsl_np:time_series/m_ops.py:m_kurt` | `ts_kurtosis` | implemented |
| `ts_max` | `Max`, `TS_MAX`, `m_max`, `max` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_max` | `Max`, `m_max`, `max` | implemented |
| `ts_max_buildup` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ts_mean` | `Mean`, `TS_MEAN`, `m_avg`, `mean` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_mean` | `Mean`, `m_avg` | implemented |
| `ts_min` | `Min`, `TS_MIN`, `m_min`, `min` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_min` | `Min`, `m_min`, `min` | implemented |
| `ts_moment` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ts_pct` | `TS_PCT` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ts_poly2_coeff` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ts_poly2_resid` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ts_product` | - | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_product` | - | implemented |
| `ts_quantile` | `TS_QUANTILE` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ts_rank` | `TS_RANK`, `m_rank` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_rank` | `m_rank` | implemented |
| `ts_regression` | `TS_REGRESSION_SLOPE`, `ts_regression_slope` | pandas_numpy | `factor_dsl_np:time_series/ts_ops.py:ts_regression` | - | implemented |
| `ts_skew` | `TS_SKEW` | pandas_numpy, polars | `factor_dsl_np:time_series/m_ops.py:m_skew` | `ts_skewness` | implemented |
| `ts_std` | `Std`, `TS_STD`, `m_std`, `std`, `ts_std_dev`, `ts_stddev` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_std` | `Std`, `m_std`, `ts_std_dev`, `ts_stddev` | implemented |
| `ts_sum` | `TS_SUM`, `m_sum` | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_sum` | `m_sum` | implemented |
| `ts_sum_decay` | - | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_sum_decay` | - | implemented |
| `ts_topk_sum` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ts_zscore` | - | pandas_numpy, polars | `factor_dsl_np:time_series/ts_ops.py:ts_zscore` | - | implemented |

### shift_diff_cum

**滞后/差分/累计**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `Lead` | - | pandas_numpy | `factor_dsl_np:time_series/shift_ops.py:Lead` | - | implemented |
| `cum_avg` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_avg` | - | implemented |
| `cum_count` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_count` | - | implemented |
| `cum_delta` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_delta` | - | implemented |
| `cum_first` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_first` | - | implemented |
| `cum_last` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_last` | - | implemented |
| `cum_max` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_max` | - | implemented |
| `cum_min` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_min` | - | implemented |
| `cum_positive_streak` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_positive_streak` | - | implemented |
| `cum_prod` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_prod` | - | implemented |
| `cum_rank` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_rank` | - | implemented |
| `cum_standardize` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_standardize` | - | implemented |
| `cum_std` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_std` | - | implemented |
| `cum_sum` | - | pandas_numpy | `factor_dsl_np:time_series/cum_ops.py:cum_sum` | - | implemented |
| `next` | - | pandas_numpy | `factor_dsl_np:time_series/shift_ops.py:next` | - | implemented |
| `prev` | - | pandas_numpy | `factor_dsl_np:time_series/shift_ops.py:prev` | - | implemented |

### cross_sectional

**截面变换**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `c_count` | - | pandas_numpy, polars | `factor_dsl_np:cross_sectional/c_ops.py:c_count` | - | implemented |
| `c_mean` | - | pandas_numpy, polars | `factor_dsl_np:cross_sectional/c_ops.py:c_mean` | - | implemented |
| `c_percentile` | - | pandas_numpy, polars | `factor_dsl_np:cross_sectional/c_ops.py:c_percentile` | - | implemented |
| `c_std` | - | pandas_numpy, polars | `factor_dsl_np:cross_sectional/c_ops.py:c_std` | - | implemented |
| `c_sum` | - | pandas_numpy, polars | `factor_dsl_np:cross_sectional/c_ops.py:c_sum` | - | implemented |
| `cs_demean` | `CS_DEMEAN` | pandas_numpy, polars | `factor_dsl_np:cross_sectional/c_ops.py:c_demean` | - | implemented |
| `cs_regression` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `cs_resid` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `neutralize` | `NEUTRALIZE`, `group_neutralize`, `industry_size_neutralize`, `size_industry_neutralize` | pandas_numpy, polars | `factor_dsl_np:cross_sectional/c_ops.py:c_neutralize` | - | implemented |
| `rank` | `CS_RANK`, `RANK`, `c_rank`, `cs_rank` | pandas_numpy, polars | `factor_dsl_np:cross_sectional/c_ops.py:rank` | `c_rank` | implemented |
| `row_avg` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_avg` | - | implemented |
| `row_beta` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_beta` | - | implemented |
| `row_corr` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_corr` | - | implemented |
| `row_count` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_count` | - | implemented |
| `row_kurt` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_kurt` | - | implemented |
| `row_max` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_max` | - | implemented |
| `row_median` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_median` | - | implemented |
| `row_min` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_min` | - | implemented |
| `row_prod` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_prod` | - | implemented |
| `row_skew` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_skew` | - | implemented |
| `row_std` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_std` | - | implemented |
| `row_sum` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_sum` | - | implemented |
| `row_var` | - | pandas_numpy | `factor_dsl_np:cross_sectional/row_ops.py:row_var` | - | implemented |
| `scale` | `SCALE`, `c_scale` | pandas_numpy, polars | `factor_dsl_np:cross_sectional/c_ops.py:scale` | `c_scale` | implemented |
| `zscore` | `CS_ZSCORE`, `ZSCORE`, `c_zscore`, `cs_zscore` | pandas_numpy, polars | `factor_dsl_np:cross_sectional/c_ops.py:zscore` | `c_zscore` | implemented |

### group_neutralization

**分组与中性化**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `deltas` | - | pandas_numpy | `factor_dsl_np:time_series/panel_ops.py:deltas` | - | implemented |
| `group_decay_linear` | - | pandas_numpy | `factor_dsl_np:cross_sectional/group_ops.py:group_decay_linear` | - | implemented |
| `group_demean` | - | pandas_numpy | `factor_dsl_np:cross_sectional/group_ops.py:group_demean` | - | implemented |
| `group_mean` | - | pandas_numpy | `factor_dsl_np:cross_sectional/group_ops.py:group_mean` | - | implemented |
| `group_normalize` | - | pandas_numpy | `factor_dsl_np:cross_sectional/group_ops.py:group_normalize` | - | implemented |
| `group_percentile` | - | pandas_numpy | `factor_dsl_np:cross_sectional/group_ops.py:group_percentile` | - | implemented |
| `group_rank` | - | pandas_numpy | `factor_dsl_np:cross_sectional/group_ops.py:group_rank` | - | implemented |
| `group_std` | - | pandas_numpy | `factor_dsl_np:cross_sectional/group_ops.py:group_std` | - | implemented |
| `group_winsorize` | - | pandas_numpy | `factor_dsl_np:cross_sectional/group_ops.py:group_winsorize` | - | implemented |
| `group_zscore` | - | pandas_numpy | `factor_dsl_np:cross_sectional/group_ops.py:group_zscore` | - | implemented |
| `industry_neutralize` | `INDUSTRY_NEUTRAL`, `INDUSTRY_NEUTRALIZE`, `IND_NEUTRALIZE` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `move` | - | pandas_numpy | `factor_dsl_np:time_series/panel_ops.py:move` | - | implemented |
| `panel_neutralize` | - | pandas_numpy | `factor_dsl_np:time_series/panel_ops.py:panel_neutralize` | - | implemented |
| `panel_rank` | - | pandas_numpy | `factor_dsl_np:time_series/panel_ops.py:panel_rank` | - | implemented |
| `panel_standardize` | - | pandas_numpy | `factor_dsl_np:time_series/panel_ops.py:panel_standardize` | - | implemented |
| `panel_zscore` | - | pandas_numpy | `factor_dsl_np:time_series/panel_ops.py:panel_zscore` | - | implemented |
| `ratios` | - | pandas_numpy | `factor_dsl_np:time_series/panel_ops.py:ratios` | - | implemented |
| `size_neutralize` | `CAP_NEUTRALIZE`, `MARKET_CAP_NEUTRALIZE`, `SIZE_NEUTRALIZE` | pandas_numpy | `lqtp_numpy` | - | implemented |

### data_cleaning

**数据清洗**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `bfill` | `FillBackward`, `fillna_backward` | pandas_numpy | `factor_dsl_np:data_handling/missing_values.py:bfill` | `fillna_backward` | implemented |
| `dropna` | - | pandas_numpy | `factor_dsl_np:data_handling/missing_values.py:dropna` | - | implemented |
| `ewm` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:ewm` | - | implemented |
| `ewm_corr` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:ewm_corr` | - | implemented |
| `ewm_cov` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:ewm_cov` | - | implemented |
| `ewm_mean` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:ewm_mean` | - | implemented |
| `ewm_std` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:ewm_std` | - | implemented |
| `ewm_var` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:ewm_var` | - | implemented |
| `expanding_max` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:expanding_max` | - | implemented |
| `expanding_mean` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:expanding_mean` | - | implemented |
| `expanding_min` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:expanding_min` | - | implemented |
| `expanding_rank` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:expanding_rank` | - | implemented |
| `expanding_std` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:expanding_std` | - | implemented |
| `expanding_sum` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:expanding_sum` | - | implemented |
| `ffill` | `FillForward`, `fillna_forward` | pandas_numpy | `factor_dsl_np:data_handling/missing_values.py:ffill` | `fillna_forward` | implemented |
| `fillna` | `FillNA` | pandas_numpy | `factor_dsl_np:data_handling/missing_values.py:fillna` | - | implemented |
| `fillna_const` | - | pandas_numpy | `factor_dsl_np:data_handling/missing_values.py:fillna_const` | - | implemented |
| `fillna_interpolate` | - | pandas_numpy | `factor_dsl_np:data_handling/missing_values.py:fillna_interpolate` | - | implemented |
| `is_inf` | - | pandas_numpy | `factor_dsl_np:data_handling/missing_values.py:is_inf` | - | implemented |
| `is_nan` | `IS_NAN`, `IS_NULL`, `is_null` | pandas_numpy | `factor_dsl_np:data_handling/missing_values.py:is_nan` | - | implemented |
| `nan_to_num` | `NAN_TO_NUM` | pandas_numpy | `factor_dsl_np:data_handling/missing_values.py:nan_to_num` | - | implemented |
| `protected_div` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `protected_log` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `protected_sqrt` | - | pandas_numpy | `basic_runtime` | - | implemented |
| `window_max` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:window_max` | - | implemented |
| `window_mean` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:window_mean` | - | implemented |
| `window_min` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:window_min` | - | implemented |
| `window_std` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:window_std` | - | implemented |
| `window_sum` | - | pandas_numpy | `factor_dsl_np:data_handling/window_ops.py:window_sum` | - | implemented |

### statistics_regression

**统计与回归**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `ACF` | `acf` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:ACF` | - | implemented |
| `Beta` | - | pandas_numpy | `factor_dsl_np:statistics/regression.py:Beta` | - | implemented |
| `Corr` | `corr` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:Corr` | - | implemented |
| `Cov` | `cov` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:Cov` | - | implemented |
| `Covariance` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:Covariance` | - | implemented |
| `Intercept` | - | pandas_numpy | `factor_dsl_np:statistics/regression.py:Intercept` | - | implemented |
| `Kurt` | `kurt` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:Kurt` | - | implemented |
| `Mad` | `mad` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:Mad` | - | implemented |
| `Median` | `median` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:Median` | - | implemented |
| `Mode` | `mode` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:Mode` | - | implemented |
| `Percentile` | `percentile` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:Percentile` | - | implemented |
| `R2` | - | pandas_numpy | `factor_dsl_np:statistics/regression.py:R2` | - | implemented |
| `Residual` | - | pandas_numpy | `factor_dsl_np:statistics/regression.py:Residual` | - | implemented |
| `Skew` | `skew` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:Skew` | - | implemented |
| `Slope` | - | pandas_numpy | `factor_dsl_np:statistics/regression.py:Slope` | - | implemented |
| `Sum` | `sum` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:Sum` | - | implemented |
| `Var` | `var` | pandas_numpy | `factor_dsl_np:statistics/basic_stats.py:Var` | `var` | implemented |
| `at_imax` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:at_imax` | - | implemented |
| `at_imin` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:at_imin` | - | implemented |
| `autocorr` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:autocorr` | - | implemented |
| `avg` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:avg` | - | implemented |
| `bartlett_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:bartlett_test` | - | implemented |
| `beta` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:beta` | - | implemented |
| `cdf_chi2` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:cdf_chi2` | - | implemented |
| `cdf_f` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:cdf_f` | - | implemented |
| `cdf_normal` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:cdf_normal` | - | implemented |
| `cdf_t` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:cdf_t` | - | implemented |
| `chi_square_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:chi_square_test` | - | implemented |
| `corr_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:corr_test` | - | implemented |
| `count` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:count` | - | implemented |
| `durbin_watson_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:durbin_watson_test` | - | implemented |
| `first` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:first` | - | implemented |
| `first_not_null` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:first_not_null` | - | implemented |
| `granger_causality` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:granger_causality` | - | implemented |
| `intercept` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:intercept` | - | implemented |
| `jarque_bera_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:jarque_bera_test` | - | implemented |
| `kendall_corr_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:kendall_corr_test` | - | implemented |
| `kpss_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:kpss_test` | - | implemented |
| `ks_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:ks_test` | - | implemented |
| `lasso` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:lasso` | - | implemented |
| `last` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:last` | - | implemented |
| `last_not_null` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:last_not_null` | - | implemented |
| `levene_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:levene_test` | - | implemented |
| `lilliefors_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:lilliefors_test` | - | implemented |
| `mean_agg` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:mean_agg` | - | implemented |
| `pacf` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:pacf` | - | implemented |
| `pdf_chi2` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:pdf_chi2` | - | implemented |
| `pdf_f` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:pdf_f` | - | implemented |
| `pdf_normal` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:pdf_normal` | - | implemented |
| `pdf_t` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:pdf_t` | - | implemented |
| `product` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:product` | - | implemented |
| `quantile` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:quantile` | - | implemented |
| `quantile_normal` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:quantile_normal` | - | implemented |
| `quantile_t` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:quantile_t` | - | implemented |
| `r_squared` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:r_squared` | - | implemented |
| `rand_exp` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:rand_exp` | - | implemented |
| `rand_lognormal` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:rand_lognormal` | - | implemented |
| `rand_normal` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:rand_normal` | - | implemented |
| `rand_poisson` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:rand_poisson` | - | implemented |
| `rand_uniform` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:rand_uniform` | - | implemented |
| `regress` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:regress` | - | implemented |
| `residual` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:residual` | - | implemented |
| `ridge` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:ridge` | - | implemented |
| `sample` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:sample` | - | implemented |
| `sem` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:sem` | - | implemented |
| `shuffle` | - | pandas_numpy | `factor_dsl_np:statistics/probability_ops.py:shuffle` | - | implemented |
| `slope` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:slope` | - | implemented |
| `spearman_corr_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:spearman_corr_test` | - | implemented |
| `stationarity_test` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:stationarity_test` | - | implemented |
| `std_agg` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:std_agg` | - | implemented |
| `stdp` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:stdp` | - | implemented |
| `sum_agg` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:sum_agg` | - | implemented |
| `ttest_one_sample` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:ttest_one_sample` | - | implemented |
| `ttest_paired` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:ttest_paired` | - | implemented |
| `ttest_two_samples` | - | pandas_numpy | `factor_dsl_np:statistics/hypothesis_ops.py:ttest_two_samples` | - | implemented |
| `varp` | - | pandas_numpy | `factor_dsl_np:statistics/aggregate_ops.py:varp` | - | implemented |
| `wavg` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:wavg` | - | implemented |
| `wsum` | - | pandas_numpy | `factor_dsl_np:statistics/regression_ex.py:wsum` | - | implemented |

### price_volume

**量价与风险**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `coskewness_to_market` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `cumulative_returns` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:cumulative_returns` | - | implemented |
| `downside_beta` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `idio_skew` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `idio_vol` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `max_drawdown` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:max_drawdown` | - | implemented |
| `residual_momentum_capm` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `returns` | `log_returns` | pandas_numpy | `factor_dsl_np:financial/__init__.py:returns` | `log_returns` | implemented |
| `rolling_beta_to_market` | `FP_BETA`, `ROLLING_BETA_TO_MARKET`, `fp_beta` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `sharpe_ratio` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:sharpe_ratio` | - | implemented |
| `tail_beta` | - | pandas_numpy | `lqtp_numpy` | - | implemented |
| `volatility` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:volatility` | - | implemented |
| `vwap` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:vwap` | - | implemented |

### technical_signal

**技术指标与信号**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `ADX` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:ADX` | - | implemented |
| `ADXR` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:ADXR` | - | implemented |
| `AROON` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:AROON` | - | implemented |
| `AROON_down` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:AROON_down` | - | implemented |
| `AROON_up` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:AROON_up` | - | implemented |
| `ATR` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:ATR` | - | implemented |
| `BollingerBands` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:BollingerBands` | - | implemented |
| `BollingerLower` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:BollingerLower` | - | implemented |
| `BollingerUpper` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:BollingerUpper` | - | implemented |
| `CCI` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:CCI` | - | implemented |
| `DPO` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:DPO` | - | implemented |
| `KAMA` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:KAMA` | - | implemented |
| `MACD` | `ts_macd` | pandas_numpy | `factor_dsl_np:financial/__init__.py:MACD` | - | implemented |
| `MACD_hist` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:MACD_hist` | - | implemented |
| `MACD_line` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:MACD_line` | - | implemented |
| `MACD_signal` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:MACD_signal` | - | implemented |
| `MOM` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:MOM` | - | implemented |
| `OBV` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:OBV` | - | implemented |
| `ROC` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:ROC` | - | implemented |
| `RSI` | `ts_rsi` | pandas_numpy | `factor_dsl_np:financial/__init__.py:RSI` | - | implemented |
| `StochasticD` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:StochasticD` | - | implemented |
| `StochasticK` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:StochasticK` | - | implemented |
| `TRIX` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:TRIX` | - | implemented |
| `WilliamsR` | - | pandas_numpy | `factor_dsl_np:financial/__init__.py:WilliamsR` | - | implemented |
| `clamp` | - | pandas_numpy | `factor_dsl_np:signal/__init__.py:clamp` | - | implemented |
| `hump_decay` | - | pandas_numpy | `factor_dsl_np:signal/__init__.py:hump_decay` | - | implemented |
| `if_else` | - | pandas_numpy | `factor_dsl_np:signal/__init__.py:if_else` | - | implemented |
| `ifnan` | - | pandas_numpy | `factor_dsl_np:signal/__init__.py:ifnan` | - | implemented |
| `is_finite` | - | pandas_numpy | `factor_dsl_np:signal/__init__.py:is_finite` | - | implemented |
| `saturate` | - | pandas_numpy | `factor_dsl_np:signal/__init__.py:saturate` | - | implemented |
| `signed_log` | - | pandas_numpy | `factor_dsl_np:signal/__init__.py:signed_log` | - | implemented |
| `signed_power` | - | pandas_numpy | `factor_dsl_np:signal/__init__.py:signed_power` | - | implemented |
| `trade_when` | - | pandas_numpy | `factor_dsl_np:signal/__init__.py:trade_when` | - | implemented |
| `vp_weighted_price` | - | pandas_numpy | `factor_dsl_np:signal/vpmacd_ops.py:vp_weighted_price` | - | implemented |
| `vpmacd` | - | pandas_numpy | `factor_dsl_np:signal/vpmacd_ops.py:vpmacd` | - | implemented |
| `vpmacd_signal` | - | pandas_numpy | `factor_dsl_np:signal/vpmacd_ops.py:vpmacd_signal` | - | implemented |
| `where` | `IIF`, `WHERE`, `if`, `iif` | pandas_numpy | `factor_dsl_np:signal/__init__.py:where` | - | implemented |

### fundamental

**基本面与财报**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `avg2` | `AVG2` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `quarter` | `QUARTER` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `ttm` | `TTM` | pandas_numpy | `lqtp_numpy` | - | implemented |
| `yoy` | `YOY` | pandas_numpy | `lqtp_numpy` | - | implemented |

### intraday_microstructure

**日内与微观结构**

| canonical | aliases | backends | selected_source | other_candidates | status |
|---------|---------|----------|-----------------|------------------|--------|
| `real_turnover_rate` | - | pandas_numpy | `lqtp_numpy` | - | implemented |

## stub 全表

| name | canonical | sources | status |
|------|-----------|---------|--------|
| `_make_stub` | `_make_stub` | `api/operators` | stub |
| `alt_8k_item_stub` | `alt_8k_item_stub` | `api/operators` | stub |
| `alt_app_rating_stub` | `alt_app_rating_stub` | `api/operators` | stub |
| `alt_carbon_intensity_stub` | `alt_carbon_intensity_stub` | `api/operators` | stub |
| `alt_credit_spread_stub` | `alt_credit_spread_stub` | `api/operators` | stub |
| `alt_customer_concentration_stub` | `alt_customer_concentration_stub` | `api/operators` | stub |
| `alt_earnings_call_tone_stub` | `alt_earnings_call_tone_stub` | `api/operators` | stub |
| `alt_esg_controversy_stub` | `alt_esg_controversy_stub` | `api/operators` | stub |
| `alt_esg_score_stub` | `alt_esg_score_stub` | `api/operators` | stub |
| `alt_job_posting_stub` | `alt_job_posting_stub` | `api/operators` | stub |
| `alt_litigation_stub` | `alt_litigation_stub` | `api/operators` | stub |
| `alt_news_sentiment_x_volume_stub` | `alt_news_sentiment_x_volume_stub` | `api/operators` | stub |
| `alt_news_volume_stub` | `alt_news_volume_stub` | `api/operators` | stub |
| `alt_patent_citation_stub` | `alt_patent_citation_stub` | `api/operators` | stub |
| `alt_satellite_activity_stub` | `alt_satellite_activity_stub` | `api/operators` | stub |
| `alt_sentiment_delta_stub` | `alt_sentiment_delta_stub` | `api/operators` | stub |
| `alt_sentiment_ema_stub` | `alt_sentiment_ema_stub` | `api/operators` | stub |
| `alt_sentiment_stub` | `alt_sentiment_stub` | `api/operators` | stub |
| `alt_sentiment_vol_stub` | `alt_sentiment_vol_stub` | `api/operators` | stub |
| `alt_social_buzz_stub` | `alt_social_buzz_stub` | `api/operators` | stub |
| `alt_supply_chain_exposure_stub` | `alt_supply_chain_exposure_stub` | `api/operators` | stub |
| `alt_web_traffic_stub` | `alt_web_traffic_stub` | `api/operators` | stub |
| `analyst_dispersion_stub` | `analyst_dispersion_stub` | `api/operators` | stub |
| `analyst_revision_30d_stub` | `analyst_revision_30d_stub` | `api/operators` | stub |
| `days_since_filing_stub` | `days_since_filing_stub` | `api/operators` | stub |
| `days_since_forecast_stub` | `days_since_forecast_stub` | `api/operators` | stub |
| `event_window_mask_stub` | `event_window_mask_stub` | `api/operators` | stub |
| `fundamental_accruals_stub` | `fundamental_accruals_stub` | `api/operators` | stub |
| `fundamental_altman_z_stub` | `fundamental_altman_z_stub` | `api/operators` | stub |
| `fundamental_asset_growth_stub` | `fundamental_asset_growth_stub` | `api/operators` | stub |
| `fundamental_cagr_stub` | `fundamental_cagr_stub` | `api/operators` | stub |
| `fundamental_cf_accruals_stub` | `fundamental_cf_accruals_stub` | `api/operators` | stub |
| `fundamental_current_ratio_stub` | `fundamental_current_ratio_stub` | `api/operators` | stub |
| `fundamental_goodwill_ratio_stub` | `fundamental_goodwill_ratio_stub` | `api/operators` | stub |
| `fundamental_gross_margin_stub` | `fundamental_gross_margin_stub` | `api/operators` | stub |
| `fundamental_interest_coverage_stub` | `fundamental_interest_coverage_stub` | `api/operators` | stub |
| `fundamental_inv_growth_stub` | `fundamental_inv_growth_stub` | `api/operators` | stub |
| `fundamental_lag_quarter_stub` | `fundamental_lag_quarter_stub` | `api/operators` | stub |
| `fundamental_leverage_stub` | `fundamental_leverage_stub` | `api/operators` | stub |
| `fundamental_net_margin_stub` | `fundamental_net_margin_stub` | `api/operators` | stub |
| `fundamental_no_stub` | `fundamental_no_stub` | `api/operators` | stub |
| `fundamental_oper_margin_stub` | `fundamental_oper_margin_stub` | `api/operators` | stub |
| `fundamental_payout_stub` | `fundamental_payout_stub` | `api/operators` | stub |
| `fundamental_qoq_stub` | `fundamental_qoq_stub` | `api/operators` | stub |
| `fundamental_quick_ratio_stub` | `fundamental_quick_ratio_stub` | `api/operators` | stub |
| `fundamental_rec_growth_stub` | `fundamental_rec_growth_stub` | `api/operators` | stub |
| `fundamental_report_delay_stub` | `fundamental_report_delay_stub` | `api/operators` | stub |
| `fundamental_revision_stub` | `fundamental_revision_stub` | `api/operators` | stub |
| `fundamental_rnd_intensity_stub` | `fundamental_rnd_intensity_stub` | `api/operators` | stub |
| `fundamental_roa_stub` | `fundamental_roa_stub` | `api/operators` | stub |
| `fundamental_roe_stub` | `fundamental_roe_stub` | `api/operators` | stub |
| `fundamental_surprise_stub` | `fundamental_surprise_stub` | `api/operators` | stub |
| `fundamental_tax_rate_stub` | `fundamental_tax_rate_stub` | `api/operators` | stub |
| `fundamental_ttm_stub` | `fundamental_ttm_stub` | `api/operators` | stub |
| `fundamental_yoy_stub` | `fundamental_yoy_stub` | `api/operators` | stub |
| `insider_net_buy_stub` | `insider_net_buy_stub` | `api/operators` | stub |
| `institutional_ownership_chg_stub` | `institutional_ownership_chg_stub` | `api/operators` | stub |
| `lob_ofi_stub` | `lob_ofi_stub` | `api/operators` | stub |
| `micro_amihud_hf_stub` | `micro_amihud_hf_stub` | `api/operators` | stub |
| `micro_avg_trade_size_stub` | `micro_avg_trade_size_stub` | `api/operators` | stub |
| `micro_bipower_var_stub` | `micro_bipower_var_stub` | `api/operators` | stub |
| `micro_book_slope_stub` | `micro_book_slope_stub` | `api/operators` | stub |
| `micro_cancel_trade_ratio_stub` | `micro_cancel_trade_ratio_stub` | `api/operators` | stub |
| `micro_depth_imbalance_stub` | `micro_depth_imbalance_stub` | `api/operators` | stub |
| `micro_effective_spread_stub` | `micro_effective_spread_stub` | `api/operators` | stub |
| `micro_jump_indicator_stub` | `micro_jump_indicator_stub` | `api/operators` | stub |
| `micro_kyle_lambda_stub` | `micro_kyle_lambda_stub` | `api/operators` | stub |
| `micro_large_trade_ratio_stub` | `micro_large_trade_ratio_stub` | `api/operators` | stub |
| `micro_mid_return_stub` | `micro_mid_return_stub` | `api/operators` | stub |
| `micro_quote_update_rate_stub` | `micro_quote_update_rate_stub` | `api/operators` | stub |
| `micro_realized_vol_stub` | `micro_realized_vol_stub` | `api/operators` | stub |
| `micro_spread_stub` | `micro_spread_stub` | `api/operators` | stub |
| `micro_tick_rule_agreement_stub` | `micro_tick_rule_agreement_stub` | `api/operators` | stub |
| `micro_trade_count_intensity_stub` | `micro_trade_count_intensity_stub` | `api/operators` | stub |
| `micro_trade_imbalance_stub` | `micro_trade_imbalance_stub` | `api/operators` | stub |
| `micro_vpin_stub` | `micro_vpin_stub` | `api/operators` | stub |
| `universe_reit_stub` | `universe_reit_stub` | `api/operators` | stub |
| `vec_avg` | `vec_avg` | `api/operators` | stub |
| `vec_sum` | `vec_sum` | `api/operators` | stub |


## api_expr_only 全表（历史审计）

> **第 31 版**：下列多数算子已迁入 `cleaned_operators` 并在 `build_dsl_allowlist()` 中可用；本表保留迁移前 `api_expr_only` 标注，`sources` 列中的 `api/operators/*` 为历史路径。

| name | canonical | sources | status |
|------|-----------|---------|--------|
| `bucket` | `bucket` | `api/operator_registry.py`, `api/operators/transformational.py` | api_expr_only |
| `change_instrument` | `change_instrument` | `api/operator_registry.py`, `api/operators/context.py` | api_expr_only |
| `col` | `col` | `api/operator_registry.py`, `api/columns.py` | api_expr_only |
| `days_from_last_change` | `days_from_last_change` | `api/operator_registry.py`, `api/operators/ts.py` | api_expr_only |
| `densify` | `densify` | `api/operator_registry.py`, `api/operators/arithmetic.py` | api_expr_only |
| `group_backfill` | `group_backfill` | `api/operator_registry.py`, `api/operators/group.py` | api_expr_only |
| `group_scale` | `group_scale` | `api/operator_registry.py`, `api/operators/group.py` | api_expr_only |
| `hump` | `hump` | `api/operator_registry.py`, `api/operators/ts.py` | api_expr_only |
| `kth_element` | `kth_element` | `api/operator_registry.py`, `api/operators/ts.py` | api_expr_only |
| `last_diff_value` | `last_diff_value` | `api/operator_registry.py`, `api/operators/ts.py` | api_expr_only |
| `orthogonalize` | `orthogonalize` | `api/operator_registry.py`, `api/operators/context.py` | api_expr_only |
| `pasteurize` | `pasteurize` | `api/operator_registry.py`, `api/operators/cleaning.py` | api_expr_only |
| `tail` | `tail` | `api/operator_registry.py`, `api/operators/cleaning.py` | api_expr_only |
| `ts_ad` | `ts_ad` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_adosc` | `ts_adosc` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_adx` | `ts_adx` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_adxr` | `ts_adxr` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_apo` | `ts_apo` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_aroon` | `ts_aroon` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_atr` | `ts_atr` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_av_diff` | `ts_av_diff` | `api/operator_registry.py`, `api/operators/ts.py` | api_expr_only |
| `ts_backfill` | `ts_backfill` | `api/operator_registry.py`, `api/operators/ts.py` | api_expr_only |
| `ts_bbands` | `ts_bbands` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_bop` | `ts_bop` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_cci` | `ts_cci` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_cmo` | `ts_cmo` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_count_nans` | `ts_count_nans` | `api/operator_registry.py`, `api/operators/ts.py` | api_expr_only |
| `ts_dema` | `ts_dema` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_donchian` | `ts_donchian` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_dx` | `ts_dx` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_ema` | `ts_ema` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_kama` | `ts_kama` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_keltner` | `ts_keltner` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_linearreg_angle` | `ts_linearreg_angle` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_linearreg_slope` | `ts_linearreg_slope` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_ma_envelope` | `ts_ma_envelope` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_mfi` | `ts_mfi` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_mom` | `ts_mom` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_natr` | `ts_natr` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_obv` | `ts_obv` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_ppo` | `ts_ppo` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_roc` | `ts_roc` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_rocr` | `ts_rocr` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_rocr100` | `ts_rocr100` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_sar` | `ts_sar` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_scale` | `ts_scale` | `api/operator_registry.py`, `api/operators/ts.py` | api_expr_only |
| `ts_sma` | `ts_sma` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_step` | `ts_step` | `api/operator_registry.py`, `api/operators/ts.py` | api_expr_only |
| `ts_stoch` | `ts_stoch` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_stochf` | `ts_stochf` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_stochrsi` | `ts_stochrsi` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_t3` | `ts_t3` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_tema` | `ts_tema` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_trange` | `ts_trange` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_trima` | `ts_trima` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_trix` | `ts_trix` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_ultosc` | `ts_ultosc` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_willr` | `ts_willr` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |
| `ts_wma` | `ts_wma` | `api/operator_registry.py`, `api/operators/technical.py` | api_expr_only |

## doc_only 全表

| name | canonical | sources | status |
|------|-----------|---------|--------|
| （无） | - | - | - |
