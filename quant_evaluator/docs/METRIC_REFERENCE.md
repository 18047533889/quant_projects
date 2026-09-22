# QuantEvaluator 全部注册指标计算手册

与 [统一口径与取舍](METRIC_CONVENTIONS.md) 配套。正文使用数学公式与中文解释，源码只作为核对链接。

注册 ID 共 **164** 个，别名不重复计数。下面逐项列出全部 ID，包括实验性或不能单独执行的项目。
默认参数是函数层默认；公开入口额外构建策略见统一口径，尤其 IC 的每日20配对、分桶人数及观察期。
公式按当前实际实现编写，非仅按指标名称套用教科书定义。输入合同与公开入口可能比低层函数施加更严格的限制。
None/NaN/unsupported不代表0；状态stable也不代表生产可交易或GPU已验收。

## 公共符号与阅读规则

除逐项另有定义：$`t`$ 为时间，$`i`$ 为资产，$`f`$ 为因子，$`x`$ 为因子值，$`y`$ 为预测标签，$`r`$ 为单期收益，$`w`$ 为权重；$`T,N,Q`$ 分别为有效期数、资产数、桶数。


```math
\bar z=\frac{1}{n}\sum_{j=1}^{n}z_j,\qquad s(z)=\sqrt{\frac{\sum_{j=1}^{n}(z_j-\bar z)^2}{n-1}}
```


$`\mathbf 1(\cdot)`$ 是条件成立取1、否则取0的指示函数；$`\mathrm{rank}`$ 默认使用平均并列秩；$`\mathrm{Corr}`$ 是相关系数。有限值集合及有效掩码按各项定义筛选；没有足够样本时为不可用，不自动补0。某些分布指标使用总体矩或其他分母，以该项公式为准。

GitHub 渲染数学公式；若使用本地 Markdown 阅读器，请开启 LaTeX/MathJax 数学显示。

## 完整目录

| ID | 名称 | 状态 | 输出 |
|---|---|---|---|
| [adaptive_quantile_count](#metric-adaptive_quantile_count) | Adaptive Quantile Count | experimental | scalar |
| [autocorrelation_ic](#metric-autocorrelation_ic) | autocorrelation_ic | stable | series |
| [benjamini_hochberg_correction](#metric-benjamini_hochberg_correction) | Benjamini-Hochberg Correction | stable | scalar |
| [beta_exposure](#metric-beta_exposure) | Beta Exposure | experimental | scalar |
| [block_bootstrap_ci](#metric-block_bootstrap_ci) | Block Bootstrap Confidence Interval | stable | scalar |
| [bonferroni_correction](#metric-bonferroni_correction) | Bonferroni Correction | stable | scalar |
| [bottom_quantile_cliff](#metric-bottom_quantile_cliff) | Bottom Quantile Cliff | experimental | scalar |
| [bottom_quantile_cliff_robust](#metric-bottom_quantile_cliff_robust) | Bottom Quantile Cliff (Robust) | experimental | scalar |
| [bottom_tail_slope](#metric-bottom_tail_slope) | Bottom Tail Slope | experimental | scalar |
| [calmar_ratio](#metric-calmar_ratio) | calmar_ratio | stable | scalar |
| [change_point_score](#metric-change_point_score) | Change-Point Score | experimental | scalar |
| [coverage](#metric-coverage) | Coverage Rate | stable | scalar |
| [coverage_stability](#metric-coverage_stability) | coverage_stability | stable | scalar |
| [cross_section_cardinality](#metric-cross_section_cardinality) | Cross-Section Cardinality | experimental | scalar |
| [cusum_break_score](#metric-cusum_break_score) | CUSUM Break Score | experimental | scalar |
| [cvar_95](#metric-cvar_95) | cvar_95 | stable | scalar |
| [cvar_99](#metric-cvar_99) | cvar_99 | stable | scalar |
| [cvar_expected_shortfall](#metric-cvar_expected_shortfall) | CVaR Expected Shortfall | experimental | scalar |
| [daily_quantile_monotonicity_rate](#metric-daily_quantile_monotonicity_rate) | Daily Quantile Monotonicity Rate | experimental | scalar |
| [daily_quantile_monotonicity_series](#metric-daily_quantile_monotonicity_series) | Daily Quantile Monotonicity Series | experimental | series |
| [distinct_level_ratio](#metric-distinct_level_ratio) | Distinct Level Ratio | experimental | scalar |
| [downside_deviation](#metric-downside_deviation) | Downside Deviation | experimental | scalar |
| [drawdown_duration](#metric-drawdown_duration) | drawdown_duration | stable | scalar |
| [effective_n](#metric-effective_n) | Effective N | experimental | scalar |
| [exposure_drift](#metric-exposure_drift) | Exposure Drift | experimental | scalar |
| [factor_coverage](#metric-factor_coverage) | factor_coverage | stable | scalar |
| [factor_turnover_rate](#metric-factor_turnover_rate) | Factor Turnover Rate | stable | scalar |
| [hac_pvalue](#metric-hac_pvalue) | HAC p-value | stable | scalar |
| [hac_tstat](#metric-hac_tstat) | HAC t-statistic | stable | scalar |
| [half_life](#metric-half_life) | IC Temporal Persistence Half-Life | stable | scalar |
| [hhi_concentration](#metric-hhi_concentration) | hhi_concentration | stable | scalar |
| [hhi_effective_n](#metric-hhi_effective_n) | hhi_effective_n | stable | scalar |
| [holm_bonferroni_correction](#metric-holm_bonferroni_correction) | Holm-Bonferroni Correction | stable | scalar |
| [ic_autocorr_lag1](#metric-ic_autocorr_lag1) | IC Autocorrelation (Lag 1) | stable | scalar |
| [ic_decay](#metric-ic_decay) | ic_decay | stable | series |
| [ic_ir](#metric-ic_ir) | IC Information Ratio | stable | scalar |
| [ic_median](#metric-ic_median) | Median IC | stable | scalar |
| [ic_positive_ratio](#metric-ic_positive_ratio) | IC Positive Ratio | experimental | scalar |
| [ic_recent_vs_history_delta](#metric-ic_recent_vs_history_delta) | IC Recent vs History Delta | experimental | scalar |
| [ic_serial_autocorrelation_lags_1_5_10_20](#metric-ic_serial_autocorrelation_lags_1_5_10_20) | IC Serial Autocorrelation (lags 1/5/10/20) | experimental | scalar |
| [ic_sign_consistency](#metric-ic_sign_consistency) | IC Sign Consistency | experimental | scalar |
| [ic_sign_flip_rate](#metric-ic_sign_flip_rate) | IC Sign Flip Rate | experimental | scalar |
| [ic_stability](#metric-ic_stability) | ic_stability | stable | scalar |
| [ic_std](#metric-ic_std) | IC Standard Deviation | stable | scalar |
| [ic_summary](#metric-ic_summary) | ic_summary | stable | distribution |
| [industry_exposure](#metric-industry_exposure) | Industry Exposure | experimental | scalar |
| [information_ratio](#metric-information_ratio) | Information Ratio | stable | scalar |
| [inverted_u_score](#metric-inverted_u_score) | Inverted-U Shape Score | experimental | scalar |
| [joint_coverage](#metric-joint_coverage) | joint_coverage | stable | scalar |
| [kurtosis](#metric-kurtosis) | kurtosis | stable | scalar |
| [label_maturity](#metric-label_maturity) | Label Maturity | experimental | scalar |
| [left_right_asymmetry](#metric-left_right_asymmetry) | Left-Right Asymmetry | experimental | scalar |
| [linear_trend_score](#metric-linear_trend_score) | Linear Trend Score | experimental | scalar |
| [liquidity_exposure](#metric-liquidity_exposure) | Liquidity Exposure | experimental | scalar |
| [long_short_returns](#metric-long_short_returns) | long_short_returns | stable | series |
| [max_absolute_style_exposure](#metric-max_absolute_style_exposure) | Max Absolute Style Exposure | experimental | scalar |
| [max_drawdown](#metric-max_drawdown) | max_drawdown | stable | scalar |
| [max_underwater_duration](#metric-max_underwater_duration) | Max Underwater Duration | experimental | scalar |
| [mean_ic](#metric-mean_ic) | Mean IC | stable | scalar |
| [mean_investment_fraction](#metric-mean_investment_fraction) | Mean Investment Fraction | stable | scalar |
| [mean_underwater_duration](#metric-mean_underwater_duration) | Mean Underwater Duration | experimental | scalar |
| [missing_ratio](#metric-missing_ratio) | Missing Ratio | experimental | scalar |
| [missing_timeline](#metric-missing_timeline) | Missing Timeline | experimental | scalar |
| [momentum_exposure](#metric-momentum_exposure) | Momentum Exposure | experimental | scalar |
| [month_consistency](#metric-month_consistency) | Month Consistency | experimental | scalar |
| [monthly_rank_ic](#metric-monthly_rank_ic) | Monthly Rank IC | experimental | scalar |
| [neutralized_rank_ic](#metric-neutralized_rank_ic) | Neutralized Rank IC | experimental | scalar |
| [outlier_ratio](#metric-outlier_ratio) | Outlier Ratio | experimental | scalar |
| [parameter_generalization](#metric-parameter_generalization) | Parameter Generalization | experimental | scalar |
| [pearson_ic](#metric-pearson_ic) | Mean Pearson IC | stable | scalar |
| [pearson_ic_ir](#metric-pearson_ic_ir) | Pearson IC Information Ratio | stable | scalar |
| [pearson_ic_series](#metric-pearson_ic_series) | Daily Pearson IC Series | stable | scalar |
| [pearson_ic_std](#metric-pearson_ic_std) | Pearson IC Standard Deviation | stable | scalar |
| [purity_ratio](#metric-purity_ratio) | Purity Ratio | experimental | scalar |
| [quantile_adjacent_spread](#metric-quantile_adjacent_spread) | Quantile Adjacent Spread | experimental | scalar |
| [quantile_curvature](#metric-quantile_curvature) | Quantile Curvature | experimental | scalar |
| [quantile_extreme_cliff](#metric-quantile_extreme_cliff) | Quantile Extreme Cliff | experimental | scalar |
| [quantile_monotonicity](#metric-quantile_monotonicity) | Adjacent Quantile Increase Fraction | experimental | scalar |
| [quantile_rank_monotonicity](#metric-quantile_rank_monotonicity) | Signed Quantile Rank Monotonicity | experimental | scalar |
| [quantile_returns](#metric-quantile_returns) | quantile_returns | stable | series |
| [quantile_returns_daily](#metric-quantile_returns_daily) | Daily quantile returns and counts | experimental | scalar |
| [quantile_returns_full](#metric-quantile_returns_full) | Full Quantile Returns | stable | scalar |
| [quantile_spread](#metric-quantile_spread) | Top-Bottom Quantile Spread | stable | scalar |
| [quantile_stability](#metric-quantile_stability) | quantile_stability | stable | series |
| [quantile_tail_asymmetry](#metric-quantile_tail_asymmetry) | Quantile Tail Asymmetry | experimental | scalar |
| [quarter_consistency](#metric-quarter_consistency) | Quarter Consistency | experimental | scalar |
| [quarterly_rank_ic](#metric-quarterly_rank_ic) | Quarterly Rank IC | experimental | scalar |
| [rank_ic](#metric-rank_ic) | Mean Rank IC | stable | scalar |
| [rank_ic_cross_section](#metric-rank_ic_cross_section) | rank_ic_cross_section | stable | series |
| [rank_ic_decay_h01_h05_h10_h20](#metric-rank_ic_decay_h01_h05_h10_h20) | IC Serial Autocorrelation (lags 1/5/10/20; legacy ID) | experimental | scalar |
| [rank_ic_positive_ratio](#metric-rank_ic_positive_ratio) | Rank IC Positive Ratio | experimental | scalar |
| [rank_ic_series](#metric-rank_ic_series) | Daily Rank IC Series | stable | scalar |
| [rank_ic_time_series](#metric-rank_ic_time_series) | rank_ic_time_series | stable | series |
| [rank_stability](#metric-rank_stability) | Rank Stability | stable | scalar |
| [recent_12m_rank_ic](#metric-recent_12m_rank_ic) | Recent 12-Month Rank IC | experimental | scalar |
| [recent_3m_rank_ic](#metric-recent_3m_rank_ic) | Recent 3-Month Rank IC | experimental | scalar |
| [recent_6m_rank_ic](#metric-recent_6m_rank_ic) | Recent 6-Month Rank IC | experimental | scalar |
| [recent_degradation_score](#metric-recent_degradation_score) | Recent Degradation Score | experimental | scalar |
| [regime_conditional_ic](#metric-regime_conditional_ic) | Regime Conditional IC | experimental | scalar |
| [regime_dispersion](#metric-regime_dispersion) | Regime Dispersion | experimental | scalar |
| [regime_sign_consistency](#metric-regime_sign_consistency) | Regime Sign Consistency | experimental | scalar |
| [regime_worst_ic](#metric-regime_worst_ic) | Regime Worst IC | experimental | scalar |
| [relative_max_drawdown](#metric-relative_max_drawdown) | Relative Max Drawdown | stable | scalar |
| [residual_rank_ic](#metric-residual_rank_ic) | Residual Rank IC | experimental | scalar |
| [return_coverage](#metric-return_coverage) | return_coverage | stable | scalar |
| [return_skew](#metric-return_skew) | Return Skew | experimental | scalar |
| [rolling_1y_sharpe_min](#metric-rolling_1y_sharpe_min) | Rolling 1Y Sharpe Min | experimental | scalar |
| [rolling_1y_sharpe_q10](#metric-rolling_1y_sharpe_q10) | Rolling 1Y Sharpe Q10 | experimental | scalar |
| [rolling_ic](#metric-rolling_ic) | rolling_ic | stable | series |
| [rolling_ic_drawdown](#metric-rolling_ic_drawdown) | Rolling IC Drawdown | experimental | scalar |
| [rolling_ic_volatility](#metric-rolling_ic_volatility) | Rolling IC Volatility | experimental | scalar |
| [rolling_rank_ic_ir](#metric-rolling_rank_ic_ir) | Rolling Rank IC IR | experimental | scalar |
| [rolling_rank_ic_mean](#metric-rolling_rank_ic_mean) | Rolling Rank IC Mean | experimental | scalar |
| [shape_bootstrap_confidence](#metric-shape_bootstrap_confidence) | Shape Bootstrap Rank Agreement (legacy ID) | experimental | scalar |
| [shape_bootstrap_rank_agreement](#metric-shape_bootstrap_rank_agreement) | Block Bootstrap Rank Agreement | experimental | scalar |
| [shape_regime_stability](#metric-shape_regime_stability) | Shape Regime Stability | experimental | scalar |
| [shape_stability](#metric-shape_stability) | Shape Stability | experimental | scalar |
| [sharpe_ratio](#metric-sharpe_ratio) | sharpe_ratio | stable | scalar |
| [sidak_correction](#metric-sidak_correction) | Sidak Correction | stable | scalar |
| [size_exposure](#metric-size_exposure) | Size Exposure | experimental | scalar |
| [skewness](#metric-skewness) | skewness | stable | scalar |
| [sortino_ratio](#metric-sortino_ratio) | sortino_ratio | stable | scalar |
| [spearman_ic](#metric-spearman_ic) | spearman_ic | stable | scalar |
| [staleness](#metric-staleness) | Staleness | experimental | scalar |
| [subsample_stability](#metric-subsample_stability) | Subsample IC Stability | stable | scalar |
| [tail_vs_middle_contrast](#metric-tail_vs_middle_contrast) | Tail vs Middle Contrast | experimental | scalar |
| [tie_ratio](#metric-tie_ratio) | Tie Ratio | experimental | scalar |
| [time_to_recovery](#metric-time_to_recovery) | Time to Recovery | experimental | scalar |
| [top_quantile_cliff](#metric-top_quantile_cliff) | Top Quantile Cliff | experimental | scalar |
| [top_quantile_cliff_robust](#metric-top_quantile_cliff_robust) | Top Quantile Cliff (Robust) | experimental | scalar |
| [top_tail_slope](#metric-top_tail_slope) | Top Tail Slope | experimental | scalar |
| [tracking_error](#metric-tracking_error) | Tracking Error | stable | scalar |
| [tradable_coverage](#metric-tradable_coverage) | Tradable Coverage | experimental | scalar |
| [train_predictive_dimension](#metric-train_predictive_dimension) | Train Predictive Dimension | experimental | scalar |
| [train_validation_icir_delta](#metric-train_validation_icir_delta) | Train-Validation ICIR Delta | experimental | scalar |
| [train_validation_rankic_delta](#metric-train_validation_rankic_delta) | Train-Validation RankIC Delta | experimental | scalar |
| [train_validation_shape_delta](#metric-train_validation_shape_delta) | Train-Validation Shape Delta | experimental | scalar |
| [train_validation_sharpe_delta](#metric-train_validation_sharpe_delta) | Train-Validation Sharpe Delta | experimental | scalar |
| [turnover](#metric-turnover) | Portfolio Turnover | stable | scalar |
| [turnover_adjusted_ic](#metric-turnover_adjusted_ic) | turnover_adjusted_ic | stable | scalar |
| [turnover_cost](#metric-turnover_cost) | turnover_cost | stable | scalar |
| [turnover_rate](#metric-turnover_rate) | turnover_rate | stable | scalar |
| [turnover_stability](#metric-turnover_stability) | turnover_stability | stable | scalar |
| [u_shape_score](#metric-u_shape_score) | U-Shape Score | experimental | scalar |
| [universe_churn](#metric-universe_churn) | Universe Churn | experimental | scalar |
| [validation_predictive_dimension](#metric-validation_predictive_dimension) | Validation Predictive Dimension | experimental | scalar |
| [validation_retention](#metric-validation_retention) | Validation Retention | experimental | scalar |
| [var_95](#metric-var_95) | var_95 | stable | scalar |
| [var_99](#metric-var_99) | var_99 | stable | scalar |
| [volatility_exposure](#metric-volatility_exposure) | Volatility Exposure | experimental | scalar |
| [win_rate](#metric-win_rate) | win_rate | stable | scalar |
| [worst_12m](#metric-worst_12m) | Worst Rolling 252 Periods (legacy ID) | experimental | scalar |
| [worst_calendar_month](#metric-worst_calendar_month) | worst_calendar_month | stable | scalar |
| [worst_calendar_quarter](#metric-worst_calendar_quarter) | worst_calendar_quarter | stable | scalar |
| [worst_calendar_year](#metric-worst_calendar_year) | worst_calendar_year | stable | scalar |
| [worst_month](#metric-worst_month) | Worst Rolling 21 Periods (legacy ID) | experimental | scalar |
| [worst_quarter](#metric-worst_quarter) | Worst Rolling 63 Periods (legacy ID) | experimental | scalar |
| [worst_quarter_rank_ic](#metric-worst_quarter_rank_ic) | Worst-Quarter Rank IC | experimental | scalar |
| [worst_rolling_21d](#metric-worst_rolling_21d) | Worst 21 trading periods (rolling) | stable | scalar |
| [worst_rolling_252d](#metric-worst_rolling_252d) | Worst 252 trading periods (rolling) | stable | scalar |
| [worst_rolling_63d](#metric-worst_rolling_63d) | Worst 63 trading periods (rolling) | stable | scalar |
| [worst_year_rank_ic](#metric-worst_year_rank_ic) | Worst-Year Rank IC | experimental | scalar |
| [year_consistency](#metric-year_consistency) | Year Consistency | experimental | scalar |
| [yearly_rank_ic](#metric-yearly_rank_ic) | Yearly Rank IC | experimental | scalar |

<a id="metric-adaptive_quantile_count"></a>
## adaptive_quantile_count — Adaptive Quantile Count

Tie-aware ACTUAL fixed quantile-bin count feasible per factor on every date (plan §14.1 adaptive 20 -> (10, 5) fallback). Uses the canonical quantile assignment policy and records per-date distinct levels, occupied-bucket minima, and fallback reasons. NaN when no candidate is feasible (never 0 bins).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`count`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_adaptive_quantile_count`。

### 数学公式与计算口径

实现是逐日可行性选择，不是连续公式。候选桶数按策略顺序（默认 20，再退化到 10、5）检验；选中的 Q 必须在每个日期都有 Q 个非空桶，且每桶有效样本达到策略下限：




```math
Q^{\ast}=\max_{Q\in\mathcal Q}\{Q:\ \forall t,\ B_t(Q)=Q,\ \min_b n_{t,b}\ge n_{\min}\}
```




其中 $`B_t(Q)`$ 为日期 $`t`$ 实际占用桶数。任一日期缺失或无候选可行时为 NaN；直接传入既有二维分位收益矩阵时，仅当该因子整列全为有限值才返回矩阵行数，否则 NaN。并列处理默认 `tie_policy=max`。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `policy` | `None` |
| `tie_policy` | `'max'` |

实现核对：[函数定义](../metrics/shape_evidence.py#L245)；`quant_evaluator.metrics.shape_evidence.compute_adaptive_quantile_count`。

<a id="metric-autocorrelation_ic"></a>
## autocorrelation_ic — autocorrelation_ic

Autocorrelation of IC values at specified lags

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.temporal.compute_ic_autocorrelation`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，因此通过统一注册执行器不可用，不能据此生成实测值。下式只说明注册所指向源码函数的计算语义；必须由显式上游制品或专门入口提供输入。

返回每日 IC 序列在原始时间轴上的自相关函数（含 0 阶）：




```math
\rho_k=\frac{\sum_{t\in P_k}(x_t-\bar x_k^{(1)})(x_{t-k}-\bar x_k^{(0)})}{\sqrt{\sum_{t\in P_k}(x_t-\bar x_k^{(1)})^2\sum_{t\in P_k}(x_{t-k}-\bar x_k^{(0)})^2}},\quad k=0,\ldots,20
```




$`P_k`$ 只含原时间轴上两端都有限的配对，均值也按该阶配对分别计算；不会压缩 NaN 后重建滞后。默认 `max_lag=20,min_obs=30`；总长度或某阶配对不足、任一侧零方差则该值 NaN，0 阶在证据充分时为 1。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-benjamini_hochberg_correction"></a>
## benjamini_hochberg_correction — Benjamini-Hochberg Correction

Benjamini-Hochberg FDR correction: controls the false discovery rate (spec §34).

- 版本：`4.0.0`；状态：`stable`；层级：`research`。
- 输入依赖：`p_values`。
- 输出：`scalar`；单位：`pvalue`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.multiple_testing.benjamini_hochberg_correction`。

### 数学公式与计算口径

对 $`m`$ 个有限 p 值升序为 $`p_{(1)}\le\cdots\le p_{(m)}`$，实际返回保持单调的 BH 调整值：




```math
q_{(i)}=\min\!\left(1,\min_{j\ge i}\frac{m}{j}p_{(j)}\right)
```




再映射回原顺序。非有限输入保留为 NaN，不作为有效检验；这是调整后的 p 值，不是拒绝指示。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `alpha` | `0.05` |

实现核对：[函数定义](../metrics/multiple_testing.py#L119)；`quant_evaluator.metrics.multiple_testing.benjamini_hochberg_correction`。

<a id="metric-beta_exposure"></a>
## beta_exposure — Beta Exposure

Signed mean beta-style exposure of the factor (typed per-style field). NaN when style absent or no finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_beta_exposure`。

### 数学公式与计算口径

逐日用因子、全部风格暴露均有限且权重严格为正的共同支持做含截距 WLS，得到原始斜率 $`b_{t,k}`$。在同一支持上令 $`\tilde w_{t,i}=w_{t,i}/\sum_jw_{t,j}`$，并用加权标准差把斜率标准化：




```math
\tilde\beta_{t,k}=b_{t,k}\frac{\sqrt{\sum_i\tilde w_{t,i}(Z_{t,i,k}-\bar Z_{t,k}^{w})^2}}{\sqrt{\sum_i\tilde w_{t,i}(x_{t,i}-\bar x_t^{w})^2}},\qquad E_{beta}=\mathrm{MeanFinite}_t(\tilde\beta_{t,beta})
```




默认 `min_obs=10`；ExposurePanel 已绑定权重时沿用，未绑定则等权。回归不可估、$`R^2`$ 非有限、因子加权标准差为 0 或 beta 风格加权标准差为 0 时当日载荷为 NaN；缺少 `beta` typed field 或无有限日也为 NaN。输出是有符号标准化暴露。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `factor_values` | `None` |
| `min_obs` | `10` |
| `weights` | `None` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L627)；`quant_evaluator.metrics.exposure_evidence.compute_beta_exposure`。

<a id="metric-block_bootstrap_ci"></a>
## block_bootstrap_ci — Block Bootstrap Confidence Interval

95% confidence interval half-width for mean IC via block bootstrap

- 版本：`2.0.0`；状态：`stable`；层级：`research`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`60`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`block_bootstrap_ci`。

### 数学公式与计算口径

注册适配器返回置信区间半宽，而不是上下界本身。完整共同日历上以长度 $`L`$ 的连续块有放回抽样，得到均值 $`\bar x^{\ast (b)}`$ 后：




```math
H=\frac{Q_{(1+c)/2}(\bar x^{\ast})-Q_{(1-c)/2}(\bar x^{\ast})}{2}
```




默认 `min_periods=60,block_length=10,num_bootstrap=1000,confidence_level=0.95,random_seed=0`。同一请求各因子共享抽样起点；列中任何 NaN/Inf 使底层区间为 NaN，$`T\lt 2L`$ 也为 NaN；注册层有限 IC 少于 60 期时强制 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `60` |
| `block_length` | `10` |
| `num_bootstrap` | `1000` |
| `confidence_level` | `0.95` |
| `random_seed` | `0` |

实现核对：[函数定义](../metrics/registry_adapters.py#L271)；`quant_evaluator.metrics.registry_adapters.compute_block_bootstrap_ci_value`。

<a id="metric-bonferroni_correction"></a>
## bonferroni_correction — Bonferroni Correction

Bonferroni multiple-testing correction: adjusted p = min(p * n, 1). Controls the family-wise error rate (spec §34).

- 版本：`4.0.0`；状态：`stable`；层级：`research`。
- 输入依赖：`p_values`。
- 输出：`scalar`；单位：`pvalue`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.multiple_testing.bonferroni_correction`。

### 数学公式与计算口径

对 $`m`$ 个有效检验逐个调整：




```math
p_i^{adj}=\min(1,mp_i)
```




$`m`$ 按实现中的有效 p 值集合计数；非有限输入保持 NaN。输出为调整 p 值，不是显著性布尔值。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `alpha` | `0.05` |

实现核对：[函数定义](../metrics/multiple_testing.py#L72)；`quant_evaluator.metrics.multiple_testing.bonferroni_correction`。

<a id="metric-bottom_quantile_cliff"></a>
## bottom_quantile_cliff — Bottom Quantile Cliff

Bottom-quantile cliff: ret[1] - ret[0], per factor. The jump in return from the lowest to the second-lowest quantile (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_bottom_quantile_cliff`。

### 数学公式与计算口径

对从低因子值到高因子值排列的分位收益 $`r_1,\ldots,r_Q`$，实现的一格底部悬崖为：




```math
C_{bottom}=r_2-r_1
```




至少需要两个分位，且所需两格必须有限，否则 NaN。正值表示从最低桶进入次低桶时收益上升。


实现核对：[函数定义](../metrics/quantile_shape.py#L193)；`quant_evaluator.metrics.quantile_shape.compute_bottom_quantile_cliff`。

<a id="metric-bottom_quantile_cliff_robust"></a>
## bottom_quantile_cliff_robust — Bottom Quantile Cliff (Robust)

ROBUST bottom cliff per factor: Q_1 - mean(Q_2..Q_4) (plan §14.4 mirror). Direction higher_is_better (big positive step out of the bottom bucket). NaN when the bottom 4 quantile returns are not all finite.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_bottom_quantile_cliff_robust`。

### 数学公式与计算口径

实现与名称旁注严格一致，取最低四桶但计算为：




```math
C_{bottom}^{robust}=r_2-\frac{r_2+r_3+r_4}{3}
```




注意它不是 $`r_1-\mathrm{mean}(r_2,r_3,r_4)`$；这是当前源码的实际索引语义。少于四桶或这四格任一非有限则 NaN。


实现核对：[函数定义](../metrics/shape_evidence.py#L747)；`quant_evaluator.metrics.shape_evidence.compute_bottom_quantile_cliff_robust`。

<a id="metric-bottom_tail_slope"></a>
## bottom_tail_slope — Bottom Tail Slope

Mean adjacent return difference over the BOTTOM segment of the quantile profile as drawn (quantiles 0..2), per factor. Direction NEUTRAL (orientation informative, never assumed, plan §14.4).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_bottom_tail_slope`。

### 数学公式与计算口径

底部三格上的两个相邻差的均值：




```math
S_{bottom}=\frac{(r_2-r_1)+(r_3-r_2)}{2}=\frac{r_3-r_1}{2}
```




分位按因子值由低到高排列。少于三桶或前三格任一非有限时 NaN；正值表示底部曲线向右上升。


实现核对：[函数定义](../metrics/shape_evidence.py#L439)；`quant_evaluator.metrics.shape_evidence.compute_bottom_tail_slope`。

<a id="metric-calmar_ratio"></a>
## calmar_ratio — calmar_ratio

Calmar ratio: explicit arithmetic (legacy default) or CAGR annualization / max drawdown

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_calmar_ratio`。

### 数学公式与计算口径

先按复利财富路径求正的最大回撤幅度 $`MDD`$；默认分子是 CAGR：




```math
Calmar=\frac{(\prod_{t=1}^{T}(1+r_t))^{P/T}-1}{MDD}
```




默认 `periods_per_year=252,min_periods=20,annualization=cagr,missing_return_policy=unknown`。因此任何非有限收益都会令默认结果 NaN（不会悄悄压缩日历）；样本不足、财富路径无效或 $`MDD\le0`$ 时 NaN。显式 `annualization=arithmetic` 才改用 $`P\bar r`$。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `periods_per_year` | `252` |
| `min_periods` | `20` |
| `annualization` | `'cagr'` |
| `missing_return_policy` | `'unknown'` |

实现核对：[函数定义](../metrics/portfolio_stats.py#L498)；`quant_evaluator.metrics.portfolio_stats.compute_calmar_ratio`。

<a id="metric-change_point_score"></a>
## change_point_score — Change-Point Score

CUSUM-based change-point score: max |cumulative deviation from the mean IC|, per factor. A large score indicates a structural break in the IC level (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_change_point_score`。

### 数学公式与计算口径

注册的实际实现是未标准化 CUSUM 最大偏离（不是相邻 60 日窗口版本）：




```math
S=\max_t\left|\sum_{s\le t}(IC_s-\overline{IC})\mathbf 1_{IC_s\ finite}\right|
```




缺失位置贡献 0，但均值只用有限 IC。默认 `min_periods=20`，不足则 NaN；常数序列得到 0。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/stability_regime.py#L195)；`quant_evaluator.metrics.stability_regime.compute_change_point_score`。

<a id="metric-coverage"></a>
## coverage — Coverage Rate

Per-factor fraction of the (T, N) panel with jointly valid factor and label values (never averaged across factor columns; canonical family: coverage)

- 版本：`0.1.0`；状态：`stable`；层级：`core`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`coverage`。

### 数学公式与计算口径

逐因子统计因子值与标签在有效性掩码下共同有限的单元格比例：




```math
Coverage_f=\frac{\sum_{t,n}\mathbf 1\{x_{tnf},y_{tn}\text{ jointly valid}\}}{TN}
```




默认 `min_assets=10` 只用于报告中的有效日/低样本日诊断，不改变该比例分子。空面板按底层报告规则返回 0；不会跨因子平均。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_assets` | `10` |

实现核对：[函数定义](../metrics/registry_adapters.py#L326)；`quant_evaluator.metrics.registry_adapters.compute_coverage_value`。

<a id="metric-coverage_stability"></a>
## coverage_stability — coverage_stability

Variance of factor coverage across periods

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`variance`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quality.compute_per_time_coverage`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，因此通过统一注册执行器不可用，不能据此生成实测值。注册描述要求的是逐日覆盖率方差；若由显式上游入口提供逐日覆盖率 $`c_{t,f}`$，目标归约为：




```math
V_f=\frac1T\sum_{t=1}^{T}(c_{t,f}-\bar c_f)^2
```




其中 $`c_{t,f}=N^{-1}\sum_n\mathbf1\{x_{tnf},y_{tn}\text{共同有效}\}`$。但注册仅定位到返回 $`T\times F`$ 逐日覆盖矩阵的函数，并未绑定把矩阵归约成方差的可调用适配器；`ddof` 也未在注册中落实，故不得声称已有可执行标量结果。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-cross_section_cardinality"></a>
## cross_section_cardinality — Cross-Section Cardinality

Mean number of distinct values per day (cross-section cardinality), per factor (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`count`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_cross_section_cardinality`。

### 数学公式与计算口径

逐日计算有限因子值的不同取值数，再对有至少一个有限值的日期取均值：




```math
C_f=\frac1{|D_f|}\sum_{t\in D_f}\left|\{x_{tnf}:x_{tnf}\ finite\}\right|
```




validity=false 先转为 NaN；全日期均无有限值时 NaN。


实现核对：[函数定义](../metrics/data_quality.py#L152)；`quant_evaluator.metrics.data_quality.compute_cross_section_cardinality`。

<a id="metric-cusum_break_score"></a>
## cusum_break_score — CUSUM Break Score

CUSUM break score: max |cumulative deviation| normalised by std*sqrt(T), per factor. A standardised change-point statistic; larger values flag a stronger break (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`zscore`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_cusum_break_score`。

### 数学公式与计算口径

标准化 CUSUM 断点分数：




```math
S=\frac{\max_t\left|\sum_{s\le t}(IC_s-\bar{IC})\mathbf1_{IC_s\ finite}\right|}{s_{IC}\sqrt n}
```




$`n`$ 与样本标准差 $`s_{IC}`$ 只由有限值计算，缺失时点在累积和中贡献 0。默认 `min_periods=20`；样本不足或 $`s_{IC}\le10^{-12}`$ 时 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/stability_regime.py#L211)；`quant_evaluator.metrics.stability_regime.compute_cusum_break_score`。

<a id="metric-cvar_95"></a>
## cvar_95 — cvar_95

Conditional Value at Risk (Expected Shortfall) at 95%

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.risk.var_cvar.compute_cvar`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，统一注册执行器不可用，不能生成实测值。注册定位的源码函数在显式调用并取置信度 $`c=0.95`$ 时语义如下。

设有限收益数为 $`n`$，将损失 $`\ell_i=-r_i`$ 从大到小排序，尾部质量 $`m=(1-c)n`$、$`k=\lfloor m\rfloor`$、$`a=m-k`$。源码用固定尾部质量和边界分数权重：




```math
ES_c^{signed}=\frac{\sum_{i=1}^{k}\ell_{(i)}+a\ell_{(k+1)}}{m},\qquad CVaR_c=\max(0,ES_c^{signed})
```




这不是简单对 $`r\le q`$ 的样本平均；当尾部质量不足 1 个观测时实际退化为最坏损失。底层默认 `method=historical,min_periods=20`，只使用有限收益。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-cvar_99"></a>
## cvar_99 — cvar_99

Conditional Value at Risk (Expected Shortfall) at 99%

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.risk.var_cvar.compute_cvar`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，统一注册执行器不可用，不能生成实测值。注册定位的源码函数在显式调用并取置信度 $`c=0.99`$ 时语义如下。

设有限收益数为 $`n`$，将损失 $`\ell_i=-r_i`$ 从大到小排序，尾部质量 $`m=(1-c)n`$、$`k=\lfloor m\rfloor`$、$`a=m-k`$。源码用固定尾部质量和边界分数权重：




```math
ES_c^{signed}=\frac{\sum_{i=1}^{k}\ell_{(i)}+a\ell_{(k+1)}}{m},\qquad CVaR_c=\max(0,ES_c^{signed})
```




这不是简单对 $`r\le q`$ 的样本平均；当尾部质量不足 1 个观测时实际退化为最坏损失。底层默认 `method=historical,min_periods=20`，只使用有限收益。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-cvar_expected_shortfall"></a>
## cvar_expected_shortfall — CVaR Expected Shortfall

Historical CVaR / expected shortfall at 95% of the probe daily PnL series (delegates to risk.var_cvar.compute_cvar — single numeric authority). Positive loss magnitude; NaN when insufficient data.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_cvar_expected_shortfall`。

### 数学公式与计算口径

该可调用 ID 是一维 95% 历史 ES 适配器。设有限收益数为 $`n`$，将损失 $`\ell_i=-r_i`$ 从大到小排序，尾部质量 $`m=(1-c)n`$、$`k=\lfloor m\rfloor`$、$`a=m-k`$。源码用固定尾部质量和边界分数权重：




```math
ES_c^{signed}=\frac{\sum_{i=1}^{k}\ell_{(i)}+a\ell_{(k+1)}}{m},\qquad CVaR_c=\max(0,ES_c^{signed})
```




这不是简单对 $`r\le q`$ 的样本平均；当尾部质量不足 1 个观测时实际退化为最坏损失。适配器默认 `confidence_level=0.95`；底层 `method=historical,min_periods=20`，只使用有限收益。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `confidence_level` | `0.95` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/underwater.py#L259)；`quant_evaluator.metrics.underwater.compute_cvar_expected_shortfall`。

<a id="metric-daily_quantile_monotonicity_rate"></a>
## daily_quantile_monotonicity_rate — Daily Quantile Monotonicity Rate

Mean of valid per-date increasing-adjacent-pair fractions; missing pairs and dates are excluded from declared denominators. Distinct from quantile_monotonicity.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.registry_adapters.compute_daily_quantile_monotonicity_rate_value`。

### 数学公式与计算口径

先逐日计算有效相邻分位对中收益上升的比例，再对有效日期取均值；名称中的 rate 不是达到某阈值的日期占比：




```math
m_{t,f}=\frac{\sum_{q=1}^{Q-1}\mathbf1(r_{t,q+1,f}\gt r_{t,q,f})\mathbf1_{pair}}{\sum_{q=1}^{Q-1}\mathbf1_{pair}},\qquad Rate_f=\frac1{|D_f|}\sum_{t\in D_f}m_{t,f}
```




默认 `n_quantiles=5,min_assets=10,min_periods=20`。非有限相邻对不进入当日分母；当日无有效对则 $`m_t`$ 为 NaN；有效日期少于 20 时最终 NaN，相等不算上升。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `n_quantiles` | `5` |
| `min_assets` | `10` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/registry_adapters.py#L257)；`quant_evaluator.metrics.registry_adapters.compute_daily_quantile_monotonicity_rate_value`。

<a id="metric-daily_quantile_monotonicity_series"></a>
## daily_quantile_monotonicity_series — Daily Quantile Monotonicity Series

Per-date fraction of increasing finite adjacent quantile-return pairs. This is daily-profile evidence, never the formal long-run mean-profile score.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`series`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`1`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.registry_adapters.compute_daily_quantile_monotonicity_series_value`。

### 数学公式与计算口径

输出逐日有效相邻分位对中的收益上升比例：




```math
m_{t,f}=\frac{\sum_{q=1}^{Q-1}\mathbf1(r_{t,q+1,f}\gt r_{t,q,f})\mathbf1\{r_{t,q,f},r_{t,q+1,f}\ finite\}}{\sum_{q=1}^{Q-1}\mathbf1\{r_{t,q,f},r_{t,q+1,f}\ finite\}}
```




默认 `n_quantiles=5,min_assets=10`。$`r_{t,q,f}`$ 是当日分位标签均值；非有限相邻对逐对排除而非要求整条曲线完整，分母为 0 的日期 NaN，相等不算上升。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `n_quantiles` | `5` |
| `min_assets` | `10` |

实现核对：[函数定义](../metrics/registry_adapters.py#L235)；`quant_evaluator.metrics.registry_adapters.compute_daily_quantile_monotonicity_series_value`。

<a id="metric-distinct_level_ratio"></a>
## distinct_level_ratio — Distinct Level Ratio

Mean fraction of distinct values among finite factor values per day, per factor. A low ratio indicates heavy duplication / coarse factor values (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_distinct_level_ratio`。

### 数学公式与计算口径

逐日以有限截面计算不同值占比，再跨有效日期平均：




```math
D_f=\frac1{|\mathcal T_f|}\sum_{t\in\mathcal T_f}\frac{|\mathrm{unique}(x_{t,:,f}^{finite})|}{n_{t,f}}
```




validity=false 先变 NaN；空截面日期跳过，若所有日期均空则 NaN。


实现核对：[函数定义](../metrics/data_quality.py#L122)；`quant_evaluator.metrics.data_quality.compute_distinct_level_ratio`。

<a id="metric-downside_deviation"></a>
## downside_deviation — Downside Deviation

Annualized downside deviation (RMS of negative excess returns) of the probe daily PnL series, matching the portfolio_stats.sortino semantics. NaN when insufficient data or no negative excess returns.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`return`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_downside_deviation`。

### 数学公式与计算口径

只在负的超额收益子集上计算 RMS，并年化：




```math
DD=\sqrt{P}\sqrt{\frac1{n_-}\sum_{r_t-r_f/P\lt 0}(r_t-r_f/P)^2}
```




默认 `risk_free_rate=0,periods_per_year=252,min_periods=20`。先仅保留有限收益；不足 20 个或没有负超额收益时 NaN。分母是负收益个数，不是全部观测数。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `risk_free_rate` | `0.0` |
| `periods_per_year` | `252` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/underwater.py#L234)；`quant_evaluator.metrics.underwater.compute_downside_deviation`。

<a id="metric-drawdown_duration"></a>
## drawdown_duration — drawdown_duration

Duration (in periods) of the longest drawdown

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`periods`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.risk.drawdown_analysis.compute_drawdown_duration`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，因此通过统一注册执行器不可用，不能据此生成实测值。下式只说明注册所指向源码函数的计算语义；必须由显式上游制品或专门入口提供输入。

由财富曲线识别每段从跌破历史峰值到恢复峰值的水下区间；该注册 ID 返回最长持续期：




```math
D_{max}=\max_j(e_j-s_j+1)
```




默认 `min_periods=10`。无回撤返回 0；有限收益不足或财富路径出现无效估值（如导致不可定义的复利路径）时 NaN。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-effective_n"></a>
## effective_n — Effective N

Mean number of finite factor values per day, per factor. A measure of the effective universe size (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`count`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_effective_n`。

### 数学公式与计算口径

每个日期统计有限因子值个数，然后对全部日期（包括计数为 0 的日期）取均值：




```math
N_{eff,f}=\frac1T\sum_{t=1}^T\sum_{n=1}^N\mathbf1(x_{tnf}\ finite)
```




validity=false 的单元先视为 NaN。它不是 Kish 有效样本量。


实现核对：[函数定义](../metrics/data_quality.py#L114)；`quant_evaluator.metrics.data_quality.compute_effective_n`。

<a id="metric-exposure_drift"></a>
## exposure_drift — Exposure Drift

Mean absolute change of the per-style exposure panel between adjacent periods (persistence / stability measure). NaN when fewer than 2 periods or no finite adjacent pair.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`2`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_exposure_drift`。

### 数学公式与计算口径

逐日用因子、全部风格暴露均有限且权重严格为正的共同支持做含截距 WLS，得到原始斜率 $`b_{t,k}`$。在同一支持上令 $`\tilde w_{t,i}=w_{t,i}/\sum_jw_{t,j}`$，并用加权标准差把斜率标准化：相邻变化也基于该标准化载荷，而非原始回归斜率：




```math
Drift=\mathrm{mean}_{t:J_t\ne\varnothing}\left[\frac1{|J_t|}\sum_{k\in J_t}|\tilde\beta_{t+1,k}-\tilde\beta_{t,k}|\right],\qquad \tilde\beta_{t,k}=b_{t,k}\frac{s^w_{t,Z_k}}{s^w_{t,x}}
```




$`J_t`$ 是相邻两日都有限的风格集合，$`s^w`$ 使用同日归一化正权重。默认 `min_obs=10`；任一侧回归无效或因子/该风格加权标准差为 0，则对应标准化载荷缺失并不进入该对；少于两期或完全没有共同有限相邻项时 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `factor_values` | `None` |
| `min_obs` | `10` |
| `weights` | `None` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L501)；`quant_evaluator.metrics.exposure_evidence.compute_exposure_drift`。

<a id="metric-factor_coverage"></a>
## factor_coverage — factor_coverage

Fraction of universe with non-null factor values

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quality.compute_coverage`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，因此通过统一注册执行器不可用，不能据此生成实测值。下式只说明注册所指向源码函数的计算语义；必须由显式上游制品或专门入口提供输入。

该稳定 ID 的实际语义是逐因子有效因子单元占原始 $`T\times N`$ 面板的比例：




```math
Coverage_f^{factor}=\frac1{TN}\sum_{t,n}\mathbf1\{x_{tnf}\ finite\ \land\ validity_{tnf}\}
```




不要求标签共同有效。空输入或完全无证据按实现的缺失政策处理；不会跨因子平均。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-factor_turnover_rate"></a>
## factor_turnover_rate — Factor Turnover Rate

Turnover rate of top/bottom quantile membership

- 版本：`2.0.0`；状态：`stable`；层级：`research`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`30`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`factor_turnover_rate`。

### 数学公式与计算口径

逐日取因子截面的顶部集合（默认 `quantile=0.9`），计算相邻日成员变化率后取有限均值：




```math
u_t=\frac{|A_t\triangle A_{t+1}|}{|V_t\cup V_{t+1}|},\qquad Turnover_f=\mathrm{mean}_{t:u_t\ finite}u_t
```




$`V_t`$ 是当日因子有限的资产集。默认 `measure=universe_membership_change,min_periods=30`；两日各至少 10 个有限资产，且若前日入选资产次日信号缺失则该对日期为 NaN。有效相邻值不足 30 个时最终 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `30` |
| `quantile` | `0.9` |

实现核对：[函数定义](../metrics/registry_adapters.py#L297)；`quant_evaluator.metrics.registry_adapters.compute_factor_turnover_rate_value`。

<a id="metric-hac_pvalue"></a>
## hac_pvalue — HAC p-value

Two-sided HAC-robust p-value for mean(IC) != 0 per factor (canonical alias ic.rank.hac_p)

- 版本：`4.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`30`。
- 别名： `ic.rank.hac_p` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`hac_pvalue`。

### 数学公式与计算口径

先按 Newey–West 得到均值的 HAC 方差，再用标准正态近似给双侧 p 值：




```math
\widehat V(\bar x)=\frac1n\left[\gamma_0+2\sum_{k=1}^{L}w_k\gamma_k\right],\quad t=\frac{\bar x}{\sqrt{\widehat V(\bar x)}},\quad p=2\Phi(-|t|)
```




默认 `min_periods=30,max_lag=5,kernel=bartlett`，$`w_k=1-k/(L+1)`$。只裁掉两端缺失；内部 NaN/Inf 使该列无证据。连续样本至少需 $`L+10`$，且注册层有限观测少于 30 时 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `30` |
| `max_lag` | `5` |
| `kernel` | `'bartlett'` |

实现核对：[函数定义](../metrics/registry_adapters.py#L416)；`quant_evaluator.metrics.registry_adapters.compute_hac_pvalue_value`。

<a id="metric-hac_tstat"></a>
## hac_tstat — HAC t-statistic

Heteroskedasticity and autocorrelation consistent t-statistic for IC

- 版本：`4.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`30`。
- 别名： `ic.rank.hac_t` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`hac_tstat`。

### 数学公式与计算口径

Newey–West HAC t 统计量：




```math
t_{HAC}=\frac{\bar x}{\sqrt{n^{-1}(\gamma_0+2\sum_{k=1}^{L}w_k\gamma_k)}}
```




默认 `min_periods=30,max_lag=5,kernel=bartlett`，$`w_k=1-k/(L+1)`$，各自协方差分母均为 $`n`$。底层只裁掉空的首尾；内部缺失或 Inf 返回 NaN，且连续样本至少需 $`L+10`$；注册层有限观测少于 30 也强制 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `30` |
| `max_lag` | `5` |
| `kernel` | `'bartlett'` |

实现核对：[函数定义](../metrics/registry_adapters.py#L87)；`quant_evaluator.metrics.registry_adapters.compute_hac_tstat_value`。

<a id="metric-half_life"></a>
## half_life — IC Temporal Persistence Half-Life

Centered AR(1) IC temporal persistence; not predictive horizon decay

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`60`。
- 别名： `ic_temporal_persistence` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`half_life`。

### 数学公式与计算口径

在原始相邻时间配对上拟合带截距 AR(1)（默认 `model=centered_ar1`）：




```math
IC_t=\alpha+\phi IC_{t-1}+\varepsilon_t,\qquad h_{1/2}=-\frac{\log2}{\log\phi}
```




只用两端都有限的真实相邻对，不压缩缺失。默认 `min_periods=60`；有限值或相邻对不足、回归退化、$`\phi\le0`$ 或 $`\phi\ge1`$ 时 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `60` |

实现核对：[函数定义](../metrics/registry_adapters.py#L127)；`quant_evaluator.metrics.registry_adapters.compute_half_life_value`。

<a id="metric-hhi_concentration"></a>
## hhi_concentration — hhi_concentration

Herfindahl-Hirschman Index of factor value concentration

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`dimensionless`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure.compute_concentration_hhi`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，统一注册执行器不可用，不能生成实测标量。注册定位的 helper 实际逐日返回总绝对暴露的 HHI：




```math
s_{t,i}=\frac{|w_{t,i}x_{t,i}|}{\sum_j|w_{t,j}x_{t,j}|},\qquad HHI_t=\sum_i s_{t,i}^2
```




默认权重为 1；仅保留因子与权重有限且权重严格为正的资产，权重先归一化（不改变上述份额）。总绝对暴露为 0 时该日 NaN。注册没有绑定把逐日序列归约为 scalar 的规则，故不得擅自取均值。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-hhi_effective_n"></a>
## hhi_effective_n — hhi_effective_n

Effective number of groups (1/HHI) for factor concentration

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`count`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure.compute_concentration_hhi`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，且定位到的 helper 只返回逐日 HHI，没有实现或绑定 `1/HHI` 的标量归约；因此统一注册执行器不可用，实际结果应记为 NaN/不可用。名称所声明但尚未绑定的目标关系仅为：




```math
N_{eff,t}=\frac1{HHI_t},\qquad HHI_t=\sum_i\left(\frac{|w_{t,i}x_{t,i}|}{\sum_j|w_{t,j}x_{t,j}|}\right)^2
```




不得把这条目标关系当作当前已执行结果，也不得擅自决定跨日期的 reducer。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-holm_bonferroni_correction"></a>
## holm_bonferroni_correction — Holm-Bonferroni Correction

Holm-Bonferroni step-down correction: more powerful than Bonferroni, controls the family-wise error rate (spec §34).

- 版本：`4.0.0`；状态：`stable`；层级：`research`。
- 输入依赖：`p_values`。
- 输出：`scalar`；单位：`pvalue`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.multiple_testing.holm_bonferroni_correction`。

### 数学公式与计算口径

将 $`m`$ 个有限 p 值升序，做 Holm 逐步调整并保证单调：




```math
q_{(i)}=\min\left(1,\max_{j\le i}(m-j+1)p_{(j)}\right)
```




再映射回原顺序。非有限值保持 NaN 且不进入有效检验数；输出是调整 p 值。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `alpha` | `0.05` |

实现核对：[函数定义](../metrics/multiple_testing.py#L200)；`quant_evaluator.metrics.multiple_testing.holm_bonferroni_correction`。

<a id="metric-ic_autocorr_lag1"></a>
## ic_autocorr_lag1 — IC Autocorrelation (Lag 1)

First-order autocorrelation of IC series

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`30`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`ic_autocorr_lag1`。

### 数学公式与计算口径

每日 IC 在原始时间轴上滞后 1 的 Pearson 相关：




```math
\rho_1=Corr(IC_t,IC_{t-1})
```




只使用真实相邻且两端有限的配对，配对内分别中心化；默认底层 ACF `min_obs=30`。配对不足或任一侧零方差时 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `30` |
| `max_lag` | `20` |

实现核对：[函数定义](../metrics/registry_adapters.py#L99)；`quant_evaluator.metrics.registry_adapters.compute_ic_autocorr_lag1_value`。

<a id="metric-ic_decay"></a>
## ic_decay — ic_decay

IC decay: correlation at increasing forward horizons

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic_summary.compute_ic_decay`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，因此通过统一注册执行器不可用，不能据此生成实测值。下式只说明注册所指向源码函数的计算语义；必须由显式上游制品或专门入口提供输入。

对每个输入预测期限 $`h`$ 分别计算逐日 IC，再对日期取有限均值，输出期限×因子矩阵：




```math
Decay_{h,f}=\mathrm{mean}_{t:\,IC_{t,f}^{(h)}\ finite}IC_{t,f}^{(h)}
```




默认 `method=pearson,min_assets=10`。它不是对期限拟合指数衰减率；某期限无有效日时该格 NaN。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-ic_ir"></a>
## ic_ir — IC Information Ratio

Mean IC divided by IC standard deviation per factor (canonical alias ic.rank.ir). Spearman-family: the pearson.ir alias is its own spec (pearson_ic_ir).

- 版本：`3.0.0`；状态：`stable`；层级：`core`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名： `ic.rank.ir` ,  `icir` ,  `rank_ic_ir` ,  `rank_icir` ,  `rank_icir_raw` ,  `rankicir` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`ic_ir`。

### 数学公式与计算口径

IC 信息比率是有限日 IC 的均值除以样本标准差：




```math
ICIR_f=\frac{\bar{IC}_f}{s_f},\qquad s_f^2=\frac1{n_f-1}\sum_t(IC_{t,f}-\bar{IC}_f)^2
```




默认 `min_periods=20`。不足、常数序列或结果非有限时 NaN；不会对很小但非零的真实标准差做人为截断，也不年化。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/registry_adapters.py#L79)；`quant_evaluator.metrics.registry_adapters.compute_ic_ir_value`。

<a id="metric-ic_median"></a>
## ic_median — Median IC

Time-median of the daily IC series per factor (canonical alias ic.rank.median)

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名： `ic.rank.median` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`ic_median`。

### 数学公式与计算口径

有限每日 IC 的中位数：




```math
MedIC_f=\mathrm{median}\{IC_{t,f}:IC_{t,f}\ finite\}
```




默认注册适配器要求至少 20 个有限期；不足返回 NaN。缺失不按 0 计。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/registry_adapters.py#L400)；`quant_evaluator.metrics.registry_adapters.compute_ic_median_value`。

<a id="metric-ic_positive_ratio"></a>
## ic_positive_ratio — IC Positive Ratio

Fraction of finite daily IC values that are strictly positive, per factor. A value near 1.0 indicates the factor's IC is consistently positive (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_ic_positive_ratio`。

### 数学公式与计算口径

有限每日 IC 中严格大于 0 的比例：




```math
P_f^+=\frac{\sum_t\mathbf1(IC_{t,f}\gt 0)}{\sum_t\mathbf1(IC_{t,f}\ finite)}
```




默认 `min_periods=20`；不足则 NaN。恰等于 0 不计为正。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L155)；`quant_evaluator.metrics.predictive.compute_ic_positive_ratio`。

<a id="metric-ic_recent_vs_history_delta"></a>
## ic_recent_vs_history_delta — IC Recent vs History Delta

(recent mean IC - full-history mean IC) / full-history std, per factor. Positive means recent IC is stronger than the historical average; a strongly negative value flags recent degradation (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`zscore`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_ic_recent_vs_history_delta`。

### 数学公式与计算口径

最近窗口均值相对全历史均值的标准化差：




```math
\Delta_f=\frac{\overline{IC}_{last\ 63}-\overline{IC}_{all}}{s_{all}}
```




默认 `recent_days=63,min_periods=20`，均值和样本标准差均忽略 NaN。全历史有限数不足或 $`s_{all}\le10^{-12}`$ 时 NaN；最近窗口可短于 63 个轴位置。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `recent_days` | `63` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L300)；`quant_evaluator.metrics.predictive.compute_ic_recent_vs_history_delta`。

<a id="metric-ic_serial_autocorrelation_lags_1_5_10_20"></a>
## ic_serial_autocorrelation_lags_1_5_10_20 — IC Serial Autocorrelation (lags 1/5/10/20)

Mean serial autocorrelation of one IC series; not predictive IC across label horizons

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_rank_ic_decay`。

### 数学公式与计算口径

分别在原始时间轴计算 $`k\in\{1,5,10,20\}`$ 的配对 Pearson 自相关，并取有限阶均值：




```math
S_f=\mathrm{nanmean}_{k\in\{1,5,10,20\}}Corr(IC_t,IC_{t-k})
```




默认 `min_periods=20`（按全列有限 IC 检查）。每阶只用两端有限的原位置配对，不压缩缺失；不可定义的阶为 NaN，全部不可定义则结果 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `horizons` | `(1, 5, 10, 20)` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L270)；`quant_evaluator.metrics.predictive.compute_rank_ic_decay`。

<a id="metric-ic_sign_consistency"></a>
## ic_sign_consistency — IC Sign Consistency

Fraction of finite daily IC values sharing the sign of the mean IC, per factor. A value near 1.0 indicates the factor's IC sign is highly consistent (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_ic_sign_consistency`。

### 数学公式与计算口径

先求有限 IC 的均值符号，再计算同号比例：




```math
C_f=\frac1{n_f}\sum_{t:IC_t\ finite}\mathbf1\{sign(IC_t)=sign(\bar{IC})\}
```




默认 `min_periods=20`，不足则 NaN。若均值恰为 0，只有 IC 恰为 0 的日期计为一致。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L287)；`quant_evaluator.metrics.predictive.compute_ic_sign_consistency`。

<a id="metric-ic_sign_flip_rate"></a>
## ic_sign_flip_rate — IC Sign Flip Rate

Fraction of adjacent finite IC pairs whose sign flips, per factor. Lower values indicate a more persistent (stable) IC sign (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_ic_sign_flip_rate`。

### 数学公式与计算口径

原始相邻日期中两端均有限且符号不同的比例：




```math
Flip_f=\frac{\sum_t\mathbf1\{sign(IC_t)\ne sign(IC_{t-1})\}\mathbf1_{pair}}{\sum_t\mathbf1_{pair}}
```




默认 `min_periods=20`，因此至少需 19 个有效相邻对；缺失不会被压缩跨越。0 与正/负之间算符号变化。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/stability_regime.py#L181)；`quant_evaluator.metrics.stability_regime.compute_ic_sign_flip_rate`。

<a id="metric-ic_stability"></a>
## ic_stability — ic_stability

Rolling correlation of IC values across sub-periods

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic_summary.compute_ic_stability`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，因此通过统一注册执行器不可用，不能据此生成实测值。下式只说明注册所指向源码函数的计算语义；必须由显式上游制品或专门入口提供输入。

当前稳定 ID 使用滚动窗口前半与后半的相关作为弱稳定性代理：




```math
S_{w,f}=Corr(IC_{w:w+H-1,f},IC_{w+H:w+2H-1,f})
```




默认 `window_size=60`、每半 `min_periods=20`，两半按相同相对位置成对删除缺失；任一半零方差则窗口 NaN。输出为窗口序列/其适配结果，单一标量解读需以注册适配器为准。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-ic_std"></a>
## ic_std — IC Standard Deviation

Standard deviation of the daily IC series per factor (canonical alias ic.rank.std). Spearman-family: the pearson.std alias is its own spec (pearson_ic_std).

- 版本：`3.0.0`；状态：`stable`；层级：`core`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名： `ic.rank.std` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`ic_std`。

### 数学公式与计算口径

有限每日 IC 的样本标准差：




```math
s_f=\sqrt{\frac1{n_f-1}\sum_t(IC_{t,f}-\bar{IC}_f)^2}
```




默认 `min_periods=20`，不足返回 NaN；采用 `ddof=1`，缺失不计入 $`n_f`$。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `valid_counts` | `None` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/ic.py#L273)；`quant_evaluator.metrics.ic.compute_ic_std`。

<a id="metric-ic_summary"></a>
## ic_summary — ic_summary

Summary statistics (mean, std, skew, kurtosis) of IC time series

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`distribution`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic_summary.compute_rolling_ic_stats`。

### 数学公式与计算口径

当前注册项没有可直接调用的 `compute_fn`，因此通过统一注册执行器不可用，不能据此生成实测值。下式只说明注册所指向源码函数的计算语义；必须由显式上游制品或专门入口提供输入。

这是分布型汇总产物而非单一数值公式；对同一每日 IC 列汇集实际实现的统计量，核心包括：




```math
\bar{IC},\quad s_{IC},\quad ICIR=\bar{IC}/s_{IC},\quad t=\bar{IC}/(s_{IC}/\sqrt n),\quad p=2F_{t,n-1}(-|t|)
```




各字段只用有限 IC，并遵循各子统计默认最低 20 期与常数序列缺失规则。消费者不应把该 distribution 输出当作一个 scalar。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-industry_exposure"></a>
## industry_exposure — Industry Exposure

Signed mean industry-style exposure of the factor (typed per-style field, from the injected ExposurePanel). NaN when style absent or no finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_industry_exposure`。

### 数学公式与计算口径

逐日用因子、全部风格暴露均有限且权重严格为正的共同支持做含截距 WLS，得到原始斜率 $`b_{t,k}`$。在同一支持上令 $`\tilde w_{t,i}=w_{t,i}/\sum_jw_{t,j}`$，并用加权标准差把斜率标准化：




```math
\tilde\beta_{t,k}=b_{t,k}\frac{\sqrt{\sum_i\tilde w_{t,i}(Z_{t,i,k}-\bar Z_{t,k}^{w})^2}}{\sqrt{\sum_i\tilde w_{t,i}(x_{t,i}-\bar x_t^{w})^2}},\qquad E_{industry}=\mathrm{MeanFinite}_t(\tilde\beta_{t,industry})
```




默认 `min_obs=10`；未显式绑定权重时等权。回归不可估、$`R^2`$ 非有限、因子或 industry 风格的加权标准差为 0 时该日 NaN；缺少 `industry` typed field 或无有限日时 NaN。输出是有符号标准化暴露。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `factor_values` | `None` |
| `min_obs` | `10` |
| `weights` | `None` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L617)；`quant_evaluator.metrics.exposure_evidence.compute_industry_exposure`。

<a id="metric-information_ratio"></a>
## information_ratio — Information Ratio

Benchmark/invested-capital evidence from an explicitly bound execution trajectory leg

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`annualized_ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.long_only.compute_information_ratio`。

### 数学公式与计算口径

对收益序列计算年化信息比率（实现中基准为 0）：




```math
IR=\sqrt P\frac{\bar r}{s_r}
```




默认 `periods_per_year=252,min_periods=2`，$`s_r`$ 为有限收益的样本标准差。样本不足、标准差为 0 或结果非有限时 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `periods_per_year` | `252` |
| `min_periods` | `2` |

实现核对：[函数定义](../metrics/long_only.py#L32)；`quant_evaluator.metrics.long_only.compute_information_ratio`。

<a id="metric-inverted_u_score"></a>
## inverted_u_score — Inverted-U Shape Score

Inverted-U (hill) score in [0, 1] per factor. Mirror of u_shape_score with concave curvature and an inverted-U template; template fit must exceed the monotone fit.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`score`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_inverted_u_score`。

### 数学公式与计算口径

令分位坐标 $`x_q\in[0,1]`$，倒 U 模板 $`z_q=-(x_q-0.5)^2`$。分别做带截距的一元 OLS 得 $`R_I^2`$ 与线性模板 $`R_L^2`$，并计算负二阶差分占比 $`c_-=mean[\Delta^2r_q\lt 0]`$。实际得分为：




```math
Score=0.5\max(R_I^2,0)+0.3c_-+0.2(\max(R_I^2,0)-R_L^2)
```




但若 $`c_-\lt 0.5`$ 则返回 0；少于 4 个有限分位、无有限内部三点或回归不可定义则 NaN；常数曲线返回 0。实现以凹曲率区分倒 U，因为自由斜率使正负二次模板本身具有同样拟合能力。


实现核对：[函数定义](../metrics/shape_evidence.py#L185)；`quant_evaluator.metrics.shape_evidence.compute_inverted_u_score`。

<a id="metric-joint_coverage"></a>
## joint_coverage — joint_coverage

Fraction of universe with both factor and return available

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quality.compute_coverage_per_factor`。

### 数学公式与计算口径


```math
C_f=\frac{\sum_{t,i}\mathbf1[\mathrm{valid}(x_{tif})\land\mathrm{valid}(y_{ti})]}{TN}
```




valid 使用 FactorBatch 与 LabelBundle 的联合有效掩码，不只是 NaN 判断。实现默认 min_assets=10；该阈值不改变覆盖率分子分母，只决定 valid_days（当日联合有效资产数至少 10）及 days_below_min_assets。空面板实现覆盖率为 0.0；注册层无直接 compute_fn，运行时须绑定该覆盖报告。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-kurtosis"></a>
## kurtosis — kurtosis

Excess kurtosis of the return distribution

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.distribution.compute_kurtosis`。

### 数学公式与计算口径


```math
K=\mathrm{Kurtosis}_{\mathrm{unbiased}}(r)-3
```




实际实现调用 scipy.stats.kurtosis，默认 axis=0、min_obs=10、excess=True（fisher=True）、bias=False、nan_policy=omit；omit 只忽略 NaN，并不忽略正负无穷。门槛计数使用 isfinite，少于 10 个有限观测强制 NaN；即使有限计数达标，数组中的无穷仍可使结果 NaN。excess=False 时不减 3。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-label_maturity"></a>
## label_maturity — Label Maturity

Fraction of (T, N) cells with a finite forward-return label, per factor. Measures how much of the factor's universe has a mature (available) label for evaluation (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_label_maturity`。

### 数学公式与计算口径


```math
M_f=\frac1{TN}\sum_{t,i}\mathbf1[\mathrm{finite}(x_{tif})\land\mathrm{finite}(y_{ti})]
```




分母始终是完整 $`T\times N`$；因子或前瞻收益任一非有限都不计分子。


实现核对：[函数定义](../metrics/data_quality.py#L170)；`quant_evaluator.metrics.data_quality.compute_label_maturity`。

<a id="metric-left_right_asymmetry"></a>
## left_right_asymmetry — Left-Right Asymmetry

Mean right-minus-left mirrored quantile contrast; symmetric U and inverted-U are zero. Descriptive signed asymmetry, not curvature.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_left_right_asymmetry`。

### 数学公式与计算口径


```math
A_f=\frac1m\sum_{j=1}^m(q_{Q+1-j,f}-q_{j,f}),\quad m=\lfloor Q/2\rfloor
```




奇数中央桶排除；要求 $`Q\ge4`$ 且所有镜像桶有限，否则 NaN。


实现核对：[函数定义](../metrics/shape_evidence.py#L491)；`quant_evaluator.metrics.shape_evidence.compute_left_right_asymmetry`。

<a id="metric-linear_trend_score"></a>
## linear_trend_score — Linear Trend Score

Pearson correlation of the quantile profile with the linear quantile coordinate per factor in [-1, 1]. +1 = perfectly monotone increasing, -1 = decreasing, ~0 = flat or U-shaped (separate the U with u_shape_score).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_linear_trend_score`。

### 数学公式与计算口径


```math
L_f=\mathrm{Corr}(z_j,q_{j,f})
```




$`z_j`$ 是线性桶坐标；仅用有限桶且至少 3 个。收益全相等返回 0，相关非有限返回 NaN。


实现核对：[函数定义](../metrics/shape_evidence.py#L514)；`quant_evaluator.metrics.shape_evidence.compute_linear_trend_score`。

<a id="metric-liquidity_exposure"></a>
## liquidity_exposure — Liquidity Exposure

Signed mean liquidity-style exposure of the factor (typed per-style field). NaN when style absent or no finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_liquidity_exposure`。

### 数学公式与计算口径


```math
x_{ti}=\alpha_t+\sum_k b_{tk}Z_{tik}+\varepsilon_{ti},\qquad E_{liq}=\mathrm{MeanFinite}_t\!\left(b_{t,liq}\frac{s_{Z,t,liq}}{s_{x,t}}\right)
```




每日在因子、全部风格和正权重共同有限的同一支持上做含截距 WLS；括号内是标准化载荷。默认 min_obs=10；因子零方差、该风格零方差、回归无效、风格缺失或无有限日均返回 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `factor_values` | `None` |
| `min_obs` | `10` |
| `weights` | `None` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L632)；`quant_evaluator.metrics.exposure_evidence.compute_liquidity_exposure`。

<a id="metric-long_short_returns"></a>
## long_short_returns — long_short_returns

Time series of long-minus-short portfolio returns, shape (T,) or (T, F). Portfolio construction: long bucket = factor values above the long_threshold quantile (default 0.8), short bucket = values below short_threshold (default 0.2); equal-weight mean of forward returns per bucket per day, long minus short. Missing-return policy defaults to 'zero_fill' (NaN returns contribute 0).

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`zero_fill`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_long_short_returns`。

### 数学公式与计算口径


```math
R^{LS}_{tf}=\sum_i w_{tif}y_{ti},\qquad w_{tif}=\frac{\mathbf1[i\in L_{tf}]-\mathbf1[i\in S_{tf}]}{|L_{tf}|+|S_{tf}|}
```




默认上下阈值 0.8/0.2、tie_policy=max，至少 2 个有限因子值且两桶非空；这是总毛敞口 100% 的逐资产权重，并非多头均值减空头均值。missing_return_policy 默认 zero_fill；drop 在任何已选收益缺失时令整日 NaN（不删除资产、不重配权），fail 报错。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `long_threshold` | `0.8` |
| `short_threshold` | `0.2` |
| `validity_mask` | `None` |
| `missing_return_policy` | `'zero_fill'` |
| `tie_policy` | `'max'` |

实现核对：[函数定义](../metrics/portfolio_stats.py#L193)；`quant_evaluator.metrics.portfolio_stats.compute_long_short_returns`。

<a id="metric-max_absolute_style_exposure"></a>
## max_absolute_style_exposure — Max Absolute Style Exposure

The style dimension with the largest mean absolute exposure (dict: style / value / absolute_mean / counts). NaN when no style has enough finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_max_absolute_style_exposure`。

### 数学公式与计算口径


```math
a_s=\mathrm{MeanFinite}_t|\tilde\beta_{ts}|,\qquad s^{\ast}=\arg\max_{s:n_s\ge m}a_s
```




$`\tilde\beta_{ts}=b_{ts}s_{Z,ts}/s_{x,t}`$ 来自同支持含截距 WLS。默认 min_finite=5、回归 min_obs=10；胜出风格按最大平均绝对载荷选，但返回的 value 是该风格带符号标准化载荷的时间均值，另返回 absolute_mean 与 counts。无合格风格为 unknown/NaN/0。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_finite` | `5` |
| `factor_values` | `None` |
| `min_obs` | `10` |
| `weights` | `None` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L467)；`quant_evaluator.metrics.exposure_evidence.compute_max_absolute_style_exposure`。

<a id="metric-max_drawdown"></a>
## max_drawdown — max_drawdown

Maximum compounded portfolio NAV drawdown including initial capital and default

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_maximum_drawdown`。

### 数学公式与计算口径


```math
D_{max}=-\min_t d_t,\qquad d_t=\frac{W_t}{\max_{0\le u\le t}W_u}-1,\quad W_0=1
```




返回的是非负回撤幅度，同时返回负值回撤序列和峰值索引；初始本金纳入高水位。默认 missing_return_policy=unknown：任一非有限收益通常使最大回撤未知，破产收益 -1 则为 100%；zero_fill 才将缺失按 0，fail 报错。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `missing_return_policy` | `'unknown'` |

实现核对：[函数定义](../metrics/portfolio_stats.py#L390)；`quant_evaluator.metrics.portfolio_stats.compute_maximum_drawdown`。

<a id="metric-max_underwater_duration"></a>
## max_underwater_duration — Max Underwater Duration

Longest continuous stretch (periods) of the probe daily PnL series staying below its running-max wealth (waterline). Computed by metrics.underwater.compute_max_underwater_duration on a (T,) dot-frequency series (probe cohort pnl). NaN when insufficient data — never a fabricated 0.

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`periods`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_max_underwater_duration`。

### 数学公式与计算口径


```math
U_{max}=\max_e|e|,\quad e:\; W_t\lt \max_{u\le t}W_u\text{ 的连续区间}
```




默认 min_periods=10；有限收益不足为 NaN，从未水下为 0。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `10` |

实现核对：[函数定义](../metrics/underwater.py#L86)；`quant_evaluator.metrics.underwater.compute_max_underwater_duration`。

<a id="metric-mean_ic"></a>
## mean_ic — Mean IC

Time-averaged Pearson information coefficient: the time-mean of daily Pearson IC between factor values and labels (canonical alias ic.pearson.mean). Observation_count is the number of finite daily IC observations under the same pair validity and minimum-assets policy.

- 版本：`1`；状态：`stable`；层级：`core`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`mean_ic`。

### 数学公式与计算口径


```math
\bar{IC}_f=\mathrm{nanmean}_t(IC^P_{tf})
```




实现以非 NaN（不是 isfinite）计有效期，默认 min_periods=20；nanmean 只忽略 NaN。正负无穷会被计期并传播，最终非有限均值被置为 NaN。valid_counts 参数虽存在但实际未使用。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `valid_counts` | `None` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/ic.py#L259)；`quant_evaluator.metrics.ic.compute_mean_ic_value`。

<a id="metric-mean_investment_fraction"></a>
## mean_investment_fraction — Mean Investment Fraction

Benchmark/invested-capital evidence from an explicitly bound execution trajectory leg

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`fraction`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.long_only.compute_mean_investment_fraction`。

### 数学公式与计算口径


```math
\bar I_f=\frac1{n_f}\sum_{t:I_{tf}\text{ finite}}I_{tf}
```




输入必须是显式绑定的 investment_fraction 执行轨迹腿；compute_fn 为 compute_mean_investment_fraction，默认 min_periods=1。有限样本不足为 NaN，任一被采用的投资比例小于 0 会报错；全现金的显式 0 是有效观测并返回 0，不是缺失。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `1` |

实现核对：[函数定义](../metrics/long_only.py#L61)；`quant_evaluator.metrics.long_only.compute_mean_investment_fraction`。

<a id="metric-mean_underwater_duration"></a>
## mean_underwater_duration — Mean Underwater Duration

Mean length (periods) of underwater episodes of the probe daily PnL series. NaN when insufficient data; 0.0 when never underwater.

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`periods`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_mean_underwater_duration`。

### 数学公式与计算口径


```math
\bar U=E^{-1}\sum_{e=1}^E|e|
```




默认 min_periods=10；有限收益不足为 NaN，从未水下明确返回 0.0。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `10` |

实现核对：[函数定义](../metrics/underwater.py#L94)；`quant_evaluator.metrics.underwater.compute_mean_underwater_duration`。

<a id="metric-missing_ratio"></a>
## missing_ratio — Missing Ratio

Fraction of (T, N) cells with non-finite factor values, per factor (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_missing_ratio`。

### 数学公式与计算口径


```math
R_f=(TN)^{-1}\sum_{t,i}\mathbf1[\neg\mathrm{finite}(x_{tif})]
```




NaN 与正负无穷均算缺失，分母为完整面板。


实现核对：[函数定义](../metrics/data_quality.py#L48)；`quant_evaluator.metrics.data_quality.compute_missing_ratio`。

<a id="metric-missing_timeline"></a>
## missing_timeline — Missing Timeline

Fraction of time periods with any missing factor value, per factor. 1.0 means every day has at least one missing asset (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_missing_timeline`。

### 数学公式与计算口径


```math
R_f=T^{-1}\sum_t\mathbf1[\exists i:\neg\mathrm{finite}(x_{tif})]
```




某日任一资产缺失，该日即计 1。


实现核对：[函数定义](../metrics/data_quality.py#L56)；`quant_evaluator.metrics.data_quality.compute_missing_timeline`。

<a id="metric-momentum_exposure"></a>
## momentum_exposure — Momentum Exposure

Signed mean momentum-style exposure of the factor (typed per-style field). NaN when style absent or no finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_momentum_exposure`。

### 数学公式与计算口径


```math
x_{ti}=\alpha_t+\sum_k b_{tk}Z_{tik}+\varepsilon_{ti},\qquad E_{mom}=\mathrm{MeanFinite}_t\!\left(b_{t,mom}\frac{s_{Z,t,mom}}{s_{x,t}}\right)
```




每日在因子、全部风格和正权重共同有限的同一支持上做含截距 WLS。默认 min_obs=10；零方差、无效回归、风格缺失或无有限日均为 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `factor_values` | `None` |
| `min_obs` | `10` |
| `weights` | `None` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L642)；`quant_evaluator.metrics.exposure_evidence.compute_momentum_exposure`。

<a id="metric-month_consistency"></a>
## month_consistency — Month Consistency

Fraction of months whose mean IC matches the overall IC sign, per factor (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_month_consistency`。

### 数学公式与计算口径


```math
C_f=|G_f|^{-1}\sum_{g\in G_f}\mathbf1[\mathrm{sign}(\bar{IC}_{gf})=\mathrm{sign}(\bar{IC}_f)]
```




若 time_index 长度正确且可由 pandas 转换，则按日历月；否则按轴位置连续 21 期分块，末块以 NaN 补齐后求块均值。只在有限月均值中计比例，默认至少 20 个有限日度 IC，否则 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `time_index` | `None` |

实现核对：[函数定义](../metrics/stability_regime.py#L110)；`quant_evaluator.metrics.stability_regime.compute_month_consistency`。

<a id="metric-monthly_rank_ic"></a>
## monthly_rank_ic — Monthly Rank IC

Mean of the per-month mean rank IC, per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_monthly_rank_ic`。

### 数学公式与计算口径


```math
M_f=|G_f|^{-1}\sum_{g\in G_f}\bar{IC}^{rank}_{gf}
```




先求月均再按有效月份等权。time_index 可用时按日历月；缺失、长度不符或转换异常时按连续 21 期分块，末块 NaN 补齐。默认至少 20 个有限日度 IC，否则 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `time_index` | `None` |

实现核对：[函数定义](../metrics/predictive.py#L181)；`quant_evaluator.metrics.predictive.compute_monthly_rank_ic`。

<a id="metric-neutralized_rank_ic"></a>
## neutralized_rank_ic — Neutralized Rank IC

Cross-sectional residual (neutralized) rank IC: per date regress the factor on the exposure panel (OLS intercept + styles), Spearman-correlate residuals with forward returns, time-mean. A factor whose IC survives neutralization has alpha orthogonal to style exposures. CPU reference; GPU optional (kernels/gpu/exposure_evidence.py).

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, forward_returns, exposure_panel`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_neutralized_rank_ic`。

### 数学公式与计算口径


```math
NIC=\mathrm{Mean}_t\rho_S(e_t,y_t),\qquad e_t=x_t-[\mathbf1,Z_t]\hat\gamma_t
```




第一阶段在因子、全部暴露和正回归权重共同有效的支持上做含截距投影；ExposurePanel 若绑定 regression_weights 就使用加权、秩感知的载荷实现，否则等权。第二阶段再取残差与标签的共同有限支持计算 Spearman，因此并非全程同一支持；回归和最终相关均要求默认 min_obs=10，无有效日则 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_obs` | `10` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L541)；`quant_evaluator.metrics.exposure_evidence.compute_neutralized_rank_ic`。

<a id="metric-outlier_ratio"></a>
## outlier_ratio — Outlier Ratio

Fraction of finite factor values that are z-score outliers (|z|>3), per factor (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_outlier_ratio`。

### 数学公式与计算口径


```math
O_f=|V_f|^{-1}\sum_{(t,i)\in V_f}\mathbf1[|z_{tif}|\gt c]
```




仅有限值进入分母；默认 c=3、min_obs=10。样本不足或标准差无效为 NaN，边界严格大于。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `threshold` | `3.0` |
| `min_obs` | `10` |

实现核对：[函数定义](../metrics/data_quality.py#L86)；`quant_evaluator.metrics.data_quality.compute_outlier_ratio`。

<a id="metric-parameter_generalization"></a>
## parameter_generalization — Parameter Generalization

Parameter-generalization summary per factor (plan §13.6): the robust retention mean where the train denominator is stable, NaN when no factor has a stable denominator (missing evidence - never 0). Versioned anchors in RetentionPolicy.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_validation_retention`。

### 数学公式与计算口径


```math
PG_f=R_f=\frac{v_f}{t_f}\quad\text{仅当 }|t_f|\ge\tau\text{ 且未触发近零反号保护}
```




$`t_f,v_f`$ 是因子 $`f`$ 的训练/验证预测维度证据，$`\tau=`$ RetentionPolicy.min_abs_train，反号保护使用 sign_flip_guard。任一端非有限记 insufficient_data，训练近零记 train_near_zero，保护触发记 sign_flip_guard；这些情形该因子结果均为 NaN/None。运行时 TrainVsValidationArtifact.parameter_generalization 直接保存 tuple(retentions)，注册 compute_fn 仅将该逐因子数组透传，绝不跨因子平均；底层 compute_validation_retention 的跨因子均值辅助结果不是此注册指标的公开输出。


实现核对：[函数定义](../registry/metrics.py#L4085)；`quant_evaluator.registry.metrics.<lambda>`。

<a id="metric-pearson_ic"></a>
## pearson_ic — Mean Pearson IC

Time-mean of daily Pearson IC between factor values and labels (canonical alias ic.pearson.mean; same kernel as mean_ic). observation_count is finite daily IC days. Default daily minimum is 20 paired assets, shared with ICIR; mean needs 1 day, IR 20.

- 版本：`2.0.0`；状态：`stable`；层级：`core`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`drop_pair`；数值政策：`finite`；注册最低期数：`None`。
- 别名： `ic.pearson.mean` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic.compute_daily_ic`。

### 数学公式与计算口径


```math
PIC_f=n_f^{-1}\sum_{t\in V_f}\mathrm{Corr}_P(x_{tf},y_t)
```




每日只用同时有限配对，默认 min_assets=20；跨日默认 min_periods=1。样本不足或常数截面为 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `1` |
| `min_assets` | `20` |

实现核对：[函数定义](../metrics/registry_adapters.py#L344)；`quant_evaluator.metrics.registry_adapters.compute_pearson_ic_value`。

<a id="metric-pearson_ic_ir"></a>
## pearson_ic_ir — Pearson IC Information Ratio

Mean Pearson IC divided by Pearson IC standard deviation per factor (canonical alias ic.pearson.ir). Fully separated from the Spearman-family ic_ir (ic.rank.ir).

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名： `ic.pearson.ir` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`pearson_ic_ir`。

### 数学公式与计算口径


```math
IR_f=\bar{IC}^P_f/s(IC^P_{tf})
```




只用有限值，默认 min_periods=20；不足或标准差为 0/非有限时 NaN。不同于 Spearman 的 ic_ir。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/registry_adapters.py#L79)；`quant_evaluator.metrics.registry_adapters.compute_ic_ir_value`。

<a id="metric-pearson_ic_series"></a>
## pearson_ic_series — Daily Pearson IC Series

Daily Pearson IC per factor over time, shape (T, F) (canonical alias ic.pearson.daily)

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名： `ic.pearson.daily` 。
- 增量模式：`APPEND_EXACT`；注册实现定位：`pearson_ic_series`。

### 数学公式与计算口径


```math
IC^P_{tf}=\frac{\sum_{i\in V}(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_{i\in V}(x_i-\bar x)^2\sum_{i\in V}(y_i-\bar y)^2}}
```




$`V`$ 是同时有限配对；默认 min_assets=20。不足或常数截面为 NaN；输出 $`T\times F`$。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_assets` | `20` |

实现核对：[函数定义](../metrics/registry_adapters.py#L376)；`quant_evaluator.metrics.registry_adapters.compute_pearson_ic_series_value`。

<a id="metric-pearson_ic_std"></a>
## pearson_ic_std — Pearson IC Standard Deviation

Standard deviation of the daily Pearson IC series per factor (canonical alias ic.pearson.std). Fully separated from the Spearman-family ic_std (ic.rank.std).

- 版本：`0.1.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名： `ic.pearson.std` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`pearson_ic_std`。

### 数学公式与计算口径


```math
s_f=\sqrt{(n_f-1)^{-1}\sum_{t\in V_f}(IC^P_{tf}-\bar{IC}^P_f)^2}
```




仅有限日度 IC，默认 min_periods=20；不足为 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `valid_counts` | `None` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/ic.py#L273)；`quant_evaluator.metrics.ic.compute_ic_std`。

<a id="metric-purity_ratio"></a>
## purity_ratio — Purity Ratio

Time mean of 1 - R-squared from same-support weighted factor-on-risk regression with intercept. Constant factors and insufficient degrees of freedom are undefined. SAME_DATE_DESCRIPTIVE, not OOS evidence.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_purity_ratio`。

### 数学公式与计算口径


```math
P=\mathrm{Mean}_t(1-R_t^2),\quad x_t=\alpha_t+Z_t\gamma_t+e_t
```




同支持加权回归且含截距。默认 min_finite=5、min_obs=10；常数因子或自由度不足为 NaN；是同日描述而非 OOS。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_finite` | `5` |
| `factor_values` | `None` |
| `min_obs` | `10` |
| `weights` | `None` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L527)；`quant_evaluator.metrics.exposure_evidence.compute_purity_ratio`。

<a id="metric-quantile_adjacent_spread"></a>
## quantile_adjacent_spread — Quantile Adjacent Spread

Mean absolute return difference between adjacent quantiles, per factor. A measure of how smooth (vs step-like) the quantile profile is (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_adjacent_spread`。

### 数学公式与计算口径


```math
S_f=|A_f|^{-1}\sum_{q\in A_f}|r_{q+1,f}-r_{q,f}|
```




$`A_f`$ 仅含两端均有限的相邻对；无有效对为 NaN。


实现核对：[函数定义](../metrics/quantile_shape.py#L133)；`quant_evaluator.metrics.quantile_shape.compute_quantile_adjacent_spread`。

<a id="metric-quantile_curvature"></a>
## quantile_curvature — Quantile Curvature

Signed curvature of the quantile-return profile (mean second difference), per factor. Positive = convex (accelerating) profile; negative = concave (decelerating) (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_curvature`。

### 数学公式与计算口径


```math
C_f=|I_f|^{-1}\sum_{q\in I_f}(r_{q+1,f}-2r_{q,f}+r_{q-1,f})
```




仅连续三桶都有限的内部位置；至少 3 桶且有有效三元组，否则 NaN。


实现核对：[函数定义](../metrics/quantile_shape.py#L87)；`quant_evaluator.metrics.quantile_shape.compute_quantile_curvature`。

<a id="metric-quantile_extreme_cliff"></a>
## quantile_extreme_cliff — Quantile Extreme Cliff

Mean of the top and bottom quantile cliffs, per factor. A large value means the extreme quantiles carry most of the spread (a cliff profile) (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_extreme_cliff`。

### 数学公式与计算口径


```math
E_f=[(r_Q-r_{Q-1})+(r_2-r_1)]/2
```




顶部与底部两桶均须有限且至少 2 桶，否则 NaN。


实现核对：[函数定义](../metrics/quantile_shape.py#L155)；`quant_evaluator.metrics.quantile_shape.compute_quantile_extreme_cliff`。

<a id="metric-quantile_monotonicity"></a>
## quantile_monotonicity — Adjacent Quantile Increase Fraction

Fraction of adjacent quantile steps that are monotone increasing, per factor. 1.0 = perfectly monotone (higher factor value -> higher return); 0.0 = no increasing adjacent pairs (flat OR decreasing). Not signed Spearman monotonicity; a >=0 gate has no direction filter.

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_monotonicity`。

### 数学公式与计算口径


```math
M_f=|A_f|^{-1}\sum_{q\in A_f}\mathbf1[r_{q+1,f}\gt r_{q,f}]
```




仅有限相邻对；严格大于才算，持平不算。无有效对为 NaN。


实现核对：[函数定义](../metrics/quantile_shape.py#L42)；`quant_evaluator.metrics.quantile_shape.compute_quantile_monotonicity`。

<a id="metric-quantile_rank_monotonicity"></a>
## quantile_rank_monotonicity — Signed Quantile Rank Monotonicity

Spearman(bucket index, mean bucket return), [-1,1]; requires all buckets finite and at least three. Flat profile is unavailable. Distinct from adjacent increase fraction.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`complete_profile`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_rank_monotonicity`。

### 数学公式与计算口径


```math
M_f=\rho_S((1,\ldots,Q),(r_{1f},\ldots,r_{Qf}))
```




至少 3 桶且全部有限；并列用平均秩。完全平坦或部分缺桶为 NaN。


实现核对：[函数定义](../metrics/quantile_shape.py#L67)；`quant_evaluator.metrics.quantile_shape.compute_quantile_rank_monotonicity`。

<a id="metric-quantile_returns"></a>
## quantile_returns — quantile_returns

Average forward return per quantile bucket

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile.compute_quantile_returns`。

### 数学公式与计算口径


```math
r_{tqf}=|B_{tqf}|^{-1}\sum_{i\in B_{tqf}}y_{ti}
```




注册项无可直接调用 compute_fn；桶数、并列、最小资产与缺失收益规则须由上游制品给出，空桶为 NaN。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-quantile_returns_daily"></a>
## quantile_returns_daily — Daily quantile returns and counts

Unreduced T×Q×F diagnostic returns, counts and masks; no implicit scalar objective

- 版本：`1.0.0`；状态：`experimental`；层级：`research`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.registry_adapters.build_daily_quantile_return_artifact`。

### 数学公式与计算口径


```math
r_{tqf}=n_{tqf}^{-1}\sum_{i\in B_{tqf}}y_{ti},\quad v_{tqf}=\mathbf1[\mathrm{finite}(r_{tqf})\land n_{tqf}\ge m]
```




输出未聚合收益、计数、掩码。默认 Q=5、m=10、min_periods=20；后者只记录溯源，不在构建时删单元。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `n_quantiles` | `5` |
| `min_assets` | `10` |
| `min_periods` | `20` |
| `tie_status_ref` | `None` |
| `tradability_ref` | `None` |
| `risk_exposure_ref` | `None` |
| `producer_version` | `'1.0.0'` |
| `split_ref` | `None` |
| `config_hash` | `None` |

实现核对：[函数定义](../metrics/registry_adapters.py#L30)；`quant_evaluator.metrics.registry_adapters.build_daily_quantile_return_artifact`。

<a id="metric-quantile_returns_full"></a>
## quantile_returns_full — Full Quantile Returns

Per-quantile time-averaged returns as a VECTOR per factor — shape (n_quantiles, F), NOT a scalar; wrap with metrics.registry_adapters.compute_quantile_returns_full_artifact for the typed VectorMetricArtifact

- 版本：`0.1.0`；状态：`stable`；层级：`research`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quantile_returns_full`。

### 数学公式与计算口径


```math
\bar r_{qf}=n_{qf}^{-1}\sum_{t:r_{tqf}\mathrm{finite}}r_{tqf}
```




默认 Q=5、min_periods=20；每个桶-因子有限日不足则 NaN。输出 $`Q\times F`$，不是标量。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `n_quantiles` | `5` |

实现核对：[函数定义](../metrics/registry_adapters.py#L219)；`quant_evaluator.metrics.registry_adapters.compute_quantile_returns_full_value`。

<a id="metric-quantile_spread"></a>
## quantile_spread — Top-Bottom Quantile Spread

Return spread between top and bottom quantiles

- 版本：`3.0.0`；状态：`stable`；层级：`core`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile.compute_top_bottom_spread`。

### 数学公式与计算口径


```math
S_f=n_f^{-1}\sum_{t\in V_f}(r_{tQf}-r_{t1f})
```




默认 Q=5、min_periods=20；只计顶底差有限日期，不足为 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `n_quantiles` | `5` |

实现核对：[函数定义](../metrics/registry_adapters.py#L183)；`quant_evaluator.metrics.registry_adapters.compute_quantile_spread_value`。

<a id="metric-quantile_stability"></a>
## quantile_stability — quantile_stability

Stability of quantile return rankings across time

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic_summary.compute_ic_stability`。

### 数学公式与计算口径


```math
\mathrm{quantile\_stability}=\mathrm{unavailable}
```




该 ID 没有可直接调用的 compute_fn，且 implementation_id 实际指向 IC 序列的 compute_ic_stability（滚动窗口前后半段 Pearson 相关），并不实现注册描述所称的“分位收益排序跨时稳定性”。因此不能套用该函数或按名称臆造公式；须有专用实现/上游制品后才能给值，当前无证据为 NaN。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-quantile_tail_asymmetry"></a>
## quantile_tail_asymmetry — Quantile Tail Asymmetry

Asymmetry between the top and bottom quantile tails, per factor. Positive = top tail is stronger than the bottom tail (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_tail_asymmetry`。

### 数学公式与计算口径


```math
A_f=(r_{Qf}-r_{mf})-(r_{mf}-r_{1f}),\quad m=\lfloor Q/2\rfloor+1
```




实现以零基 Q//2 选中央桶；至少 3 桶且顶、底、中均有限，否则 NaN。


实现核对：[函数定义](../metrics/quantile_shape.py#L112)；`quant_evaluator.metrics.quantile_shape.compute_quantile_tail_asymmetry`。

<a id="metric-quarter_consistency"></a>
## quarter_consistency — Quarter Consistency

Fraction of quarters whose mean IC matches the overall IC sign, per factor (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_quarter_consistency`。

### 数学公式与计算口径


```math
C_f=|G_f|^{-1}\sum_{g\in G_f}\mathbf1[\mathrm{sign}(\bar{IC}_{gf})=\mathrm{sign}(\bar{IC}_f)]
```




若 time_index 长度正确且可转换则按日历季度；否则按连续 63 期分块，末块 NaN 补齐。只比较有限季度均值，默认至少 20 个有限日度 IC，否则 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `time_index` | `None` |

实现核对：[函数定义](../metrics/stability_regime.py#L99)；`quant_evaluator.metrics.stability_regime.compute_quarter_consistency`。

<a id="metric-quarterly_rank_ic"></a>
## quarterly_rank_ic — Quarterly Rank IC

Mean of the per-quarter mean rank IC, per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_quarterly_rank_ic`。

### 数学公式与计算口径


```math
QIC_f=|G_f|^{-1}\sum_{g\in G_f}\bar{IC}^{rank}_{gf}
```




先求季度均值再按有效季度等权。time_index 可用时按日历季度；缺失、长度不符或转换异常时按连续 63 期分块，末块 NaN 补齐。默认至少 20 个有限日度 IC，否则 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `time_index` | `None` |

实现核对：[函数定义](../metrics/predictive.py#L192)；`quant_evaluator.metrics.predictive.compute_quarterly_rank_ic`。

<a id="metric-rank_ic"></a>
## rank_ic — Mean Rank IC

rank_ic has exactly ONE meaning: the time-mean of daily Spearman rank IC between factor values and labels (canonical alias ic.rank.mean)

- 版本：`4.0.0`；状态：`stable`；层级：`core`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`drop_pair`；数值政策：`finite`；注册最低期数：`None`。
- 别名： `ic.rank.mean` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic.compute_daily_ic`。

### 数学公式与计算口径


```math
IC_t=\mathrm{corr}(\mathrm{rank}_{avg}f_{t,i},\mathrm{rank}_{avg}y_{t,i})
```





```math
RankIC={1\over n}\sum_{t\in T^{\ast}}IC_t
```


 每日保留成对有限值，默认 min_assets=20；再以 min_periods=1 检查有效日。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `1` |
| `min_assets` | `20` |

实现核对：[函数定义](../metrics/registry_adapters.py#L358)；`quant_evaluator.metrics.registry_adapters.compute_rank_ic_value`。

<a id="metric-rank_ic_cross_section"></a>
## rank_ic_cross_section — rank_ic_cross_section

Rank IC computed cross-sectionally for each date

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`drop_pair`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic.compute_daily_ic`。

### 数学公式与计算口径


```math
IC_t=\mathrm{corr}(\mathrm{rank}_{avg}f_{t,i},\mathrm{rank}_{avg}y_{t,i})
```


 输出逐日截面序列；注册项无 compute_fn，只声明 daily-IC 上游制品，不能独立执行。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-rank_ic_decay_h01_h05_h10_h20"></a>
## rank_ic_decay_h01_h05_h10_h20 — IC Serial Autocorrelation (lags 1/5/10/20; legacy ID)

Mean IC serial autocorrelation across lags {1, 5, 10, 20}; NOT predictive horizon decay. trading days, per factor. A single scalar summarising how quickly the IC series loses autocorrelation (decays) at increasing lags (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_rank_ic_decay`。

### 数学公式与计算口径


```math
a_h={\sum_{(t,t-h)\in P_h}(IC_t-\bar I_h^+)(IC_{t-h}-\bar I_h^-)\over\sqrt{\sum( IC_t-\bar I_h^+)^2\sum( IC_{t-h}-\bar I_h^-)^2}},\quad M={1\over |H^{\ast}|}\sum_{h\in H^{\ast}}a_h
```


 $`H=(1,5,10,20)`$；各滞后仅用原时钟上成对有限值，至少 2 对且非零方差；总体默认 min_periods=20。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `horizons` | `(1, 5, 10, 20)` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L270)；`quant_evaluator.metrics.predictive.compute_rank_ic_decay`。

<a id="metric-rank_ic_positive_ratio"></a>
## rank_ic_positive_ratio — Rank IC Positive Ratio

Fraction of finite daily rank-IC values that are strictly positive, per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_rank_ic_positive_ratio`。

### 数学公式与计算口径


```math
M={\sum_t\mathbf1(IC_t\gt 0,\ IC_t\ finite)\over\sum_t\mathbf1(IC_t\ finite)}
```


 零不算正值；默认 min_periods=20。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L165)；`quant_evaluator.metrics.predictive.compute_rank_ic_positive_ratio`。

<a id="metric-rank_ic_series"></a>
## rank_ic_series — Daily Rank IC Series

Daily Spearman rank IC per factor over time, shape (T, F) (canonical alias ic.rank.daily)

- 版本：`4.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名： `ic.rank.daily` 。
- 增量模式：`APPEND_EXACT`；注册实现定位：`rank_ic_series`。

### 数学公式与计算口径


```math
IC_t=\mathrm{corr}(\mathrm{rank}_{avg}f_{t,i},\mathrm{rank}_{avg}y_{t,i})
```


 输出逐日 $`IC_t`$，不做时间聚合；默认 min_assets=20，资产不足或常数秩截面为 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_assets` | `20` |

实现核对：[函数定义](../metrics/registry_adapters.py#L388)；`quant_evaluator.metrics.registry_adapters.compute_rank_ic_series_value`。

<a id="metric-rank_ic_time_series"></a>
## rank_ic_time_series — rank_ic_time_series

Rank IC computed per time slice, returned as a time series

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`drop_pair`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic.compute_daily_ic`。

### 数学公式与计算口径


```math
IC_t=\mathrm{corr}(\mathrm{rank}_{avg}f_{t,i},\mathrm{rank}_{avg}y_{t,i})
```


 输出逐时点序列；注册项无 compute_fn，只能读取显式 daily-IC 制品，不能独立执行。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-rank_stability"></a>
## rank_stability — Rank Stability

Spearman correlation of factor ranks across time

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`rank_stability`。

### 数学公式与计算口径


```math
s_t=\rho(f_{t,\cdot},f_{t-\ell,\cdot}),\qquad M={1\over |T^{\ast}|}\sum_ts_t
```


 默认 lag=1、method='spearman'、min_periods=20；每对日期只用共同有限资产。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `lag` | `1` |
| `method` | `'spearman'` |

实现核对：[函数定义](../metrics/registry_adapters.py#L111)；`quant_evaluator.metrics.registry_adapters.compute_rank_stability_value`。

<a id="metric-recent_12m_rank_ic"></a>
## recent_12m_rank_ic — Recent 12-Month Rank IC

Mean rank IC over the most recent ~12 months (252 trading days), per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_recent_12m_rank_ic`。

### 数学公式与计算口径


```math
M={1\over n_W}\sum_{t\in\{T-251,\ldots,T\}\cap T^{\ast}}IC_t
```


 尾窗为 252 个原始位置；默认全序列有限 IC 总数 min_periods=20，尾窗均值忽略 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L241)；`quant_evaluator.metrics.predictive.compute_recent_12m_rank_ic`。

<a id="metric-recent_3m_rank_ic"></a>
## recent_3m_rank_ic — Recent 3-Month Rank IC

Mean rank IC over the most recent ~3 months (63 trading days), per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_recent_3m_rank_ic`。

### 数学公式与计算口径


```math
M={1\over n_W}\sum_{t\in\{T-62,\ldots,T\}\cap T^{\ast}}IC_t
```


 尾窗为 63 个原始位置；默认全序列有限 IC 总数 min_periods=20，尾窗均值忽略 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L227)；`quant_evaluator.metrics.predictive.compute_recent_3m_rank_ic`。

<a id="metric-recent_6m_rank_ic"></a>
## recent_6m_rank_ic — Recent 6-Month Rank IC

Mean rank IC over the most recent ~6 months (126 trading days), per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_recent_6m_rank_ic`。

### 数学公式与计算口径


```math
M={1\over n_W}\sum_{t\in\{T-125,\ldots,T\}\cap T^{\ast}}IC_t
```


 尾窗为 126 个原始位置；默认全序列有限 IC 总数 min_periods=20，尾窗均值忽略 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L234)；`quant_evaluator.metrics.predictive.compute_recent_6m_rank_ic`。

<a id="metric-recent_degradation_score"></a>
## recent_degradation_score — Recent Degradation Score

Recent degradation: (full mean IC - recent mean IC) / full std, per factor. Positive values indicate the factor's recent IC is weaker than its historical average (degradation) (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`zscore`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_recent_degradation_score`。

### 数学公式与计算口径


```math
M={\bar I_{all}-\bar I_{last\ 63}\over s_{all}},\qquad s_{all}=\mathrm{std}(IC_t,ddof=1)
```


 默认 recent_days=63、min_periods=20；$`s_{all}\le10^{-12}`$ 或证据不足为 NaN；正值才表示近期退化。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `recent_days` | `63` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/stability_regime.py#L230)；`quant_evaluator.metrics.stability_regime.compute_recent_degradation_score`。

<a id="metric-regime_conditional_ic"></a>
## regime_conditional_ic — Regime Conditional IC

Mean IC in the late (recent) regime, per factor. Measures the factor's current predictive power (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_regime_conditional_ic`。

### 数学公式与计算口径


```math
m=\lfloor T/2\rfloor,\qquad M=\mathrm{mean}_{t=m}^{T-1}IC_t
```


 regime 实际固定为原序列前/后半段，此项返回后半段均值；默认 min_periods=20（检查全序列有限数）。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/stability_regime.py#L260)；`quant_evaluator.metrics.stability_regime.compute_regime_conditional_ic`。

<a id="metric-regime_dispersion"></a>
## regime_dispersion — Regime Dispersion

Absolute difference between early and late regime mean IC, per factor. Larger values indicate the factor's predictive power changed across regimes (instability) (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_regime_dispersion`。

### 数学公式与计算口径


```math
M=|e-l|,\quad e=\mathrm{mean}_{t\lt m}IC_t,\ l=\mathrm{mean}_{t\ge m}IC_t
```


 固定前后半段，均值忽略 NaN；默认 min_periods=20。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/stability_regime.py#L280)；`quant_evaluator.metrics.stability_regime.compute_regime_dispersion`。

<a id="metric-regime_sign_consistency"></a>
## regime_sign_consistency — Regime Sign Consistency

1.0 if early and late regime mean IC share the same sign, else 0.0, per factor (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_regime_sign_consistency`。

### 数学公式与计算口径


```math
M=\mathbf1[\mathrm{sign}(e)=\mathrm{sign}(l)]
```


 $`e,l`$ 为前后半段有限 IC 均值；任一不可定义则 NaN，默认 min_periods=20。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/stability_regime.py#L295)；`quant_evaluator.metrics.stability_regime.compute_regime_sign_consistency`。

<a id="metric-regime_worst_ic"></a>
## regime_worst_ic — Regime Worst IC

Minimum of the early/late regime mean IC, per factor. The weaker of the two regime averages (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_regime_worst_ic`。

### 数学公式与计算口径


```math
e=\mathrm{mean}_{t\lt m}IC_t,\quad l=\mathrm{mean}_{t\ge m}IC_t,\quad M=\min(e,l)
```


 固定前后半段，不是财富或外部市场状态；默认 min_periods=20。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/stability_regime.py#L269)；`quant_evaluator.metrics.stability_regime.compute_regime_worst_ic`。

<a id="metric-relative_max_drawdown"></a>
## relative_max_drawdown — Relative Max Drawdown

Benchmark/invested-capital evidence from an explicitly bound execution trajectory leg

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.long_only.compute_relative_max_drawdown`。

### 数学公式与计算口径


```math
W_0=1,\qquad W_t=W_{t-1}(1+r_t^{rel}),\qquad M=\max_t\left(1-{W_t\over\max_{0\le u\le t}W_u}\right)
```


 输入必须是 probe_pnl 明确绑定的 relative_return 轨迹腿，即相对财富的收益增量，不是 active_return（组合收益减基准收益）或直接传入的财富水平。返回正回撤幅度；默认 min_periods=1、missing_return_policy='unknown'，任何非有限增量使该列结果未知，$`r_t^{rel}=-1`$ 则财富归零并返回 1。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `1` |

实现核对：[函数定义](../metrics/long_only.py#L49)；`quant_evaluator.metrics.long_only.compute_relative_max_drawdown`。

<a id="metric-residual_rank_ic"></a>
## residual_rank_ic — Residual Rank IC

Named alias of neutralized_rank_ic (residual version of the rank IC, residualized against style exposures). Same kernel, registered under its own id for downstream reporting.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, forward_returns, exposure_panel`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_residual_rank_ic`。

### 数学公式与计算口径


```math
f_{t,i}=\alpha_t+X_{t,i,\cdot}\beta_t+e_{t,i},\qquad M={1\over |T^{\ast}|}\sum_{t\in T^{\ast}}\rho_S(e_{t,i},y_{t,i})
```


 每日回归支持集先要求因子、全部暴露及回归权重有效且正权重；若 ExposurePanel 绑定 regression_weights 则做带截距 WLS，否则等权。残差生成后再与 forward return 取共同有限支持做 Spearman，因此回归支持与最终标签配对支持不同；两步均默认 min_obs=10。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_obs` | `10` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L596)；`quant_evaluator.metrics.exposure_evidence.compute_residual_rank_ic`。

<a id="metric-return_coverage"></a>
## return_coverage — return_coverage

Fraction of universe with non-null forward returns

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quality.compute_coverage`。

### 数学公式与计算口径


```math
C={\#\{(t,i):y_{t,i}\ finite\}\over\#\{(t,i):i\in U_t\}}
```


 注册项无 compute_fn，只声明 quality coverage 上游制品；实际 universe 分母必须由上游给出，不可独立猜算。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-return_skew"></a>
## return_skew — Return Skew

Sample skewness of the probe daily PnL return series (adjusted Fisher-Pearson, bias=False). NaN when insufficient data or zero std.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_return_skew`。

### 数学公式与计算口径


```math
g_1={1\over n}\sum_t((r_t-\bar r)/\sigma_0)^3,\qquad M={\sqrt{n(n-1)}\over n-2}g_1
```


 默认 bias=False、min_periods=20；$`n\lt \max(20,3)`$ 或总体标准差 $`\sigma_0\le10^{-12}`$ 为 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `bias` | `False` |

实现核对：[函数定义](../metrics/underwater.py#L215)；`quant_evaluator.metrics.underwater.compute_return_skew`。

<a id="metric-rolling_1y_sharpe_min"></a>
## rolling_1y_sharpe_min — Rolling 1Y Sharpe Min

Minimum rolling-252-period annualized Sharpe of the probe daily PnL series (worst observed 1y window). NaN when the series is shorter than one window.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`60`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail`。

### 数学公式与计算口径


```math
S_t=\sqrt{252}\,\bar r_t/s_t,\quad M=\min_t S_t
```


 默认 window=252、min_periods=30、risk_free_rate=0、periods_per_year=252；时间轴必须先具有至少 252 个位置；每个长度 252 的完整对齐位置窗口只要求至少 30 个有限收益，其余 NaN 由 Sharpe 内部剔除，$`s_t`$ 为 ddof=1；无合格窗口返回 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `window` | `252` |
| `quantile` | `0.1` |
| `min_periods` | `30` |
| `periods_per_year` | `252` |

实现核对：[函数定义](../metrics/underwater.py#L171)；`quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail`。

<a id="metric-rolling_1y_sharpe_q10"></a>
## rolling_1y_sharpe_q10 — Rolling 1Y Sharpe Q10

10th-percentile rolling-252-period annualized Sharpe of the probe daily PnL series (tail floor of the 1y rolling Sharpe distribution). NaN when the series is shorter than one window.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`60`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail`。

### 数学公式与计算口径


```math
S_t=\sqrt{252}\,\bar r_t/s_t,\quad M=Q_{0.1}(\{S_t\})
```


 默认 window=252、min_periods=30、risk_free_rate=0、periods_per_year=252；时间轴必须先具有至少 252 个位置；每个长度 252 的完整对齐位置窗口只要求至少 30 个有限收益，其余 NaN 由 Sharpe 内部剔除，$`s_t`$ 为 ddof=1；无合格窗口返回 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `window` | `252` |
| `quantile` | `0.1` |
| `min_periods` | `30` |
| `periods_per_year` | `252` |

实现核对：[函数定义](../metrics/underwater.py#L171)；`quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail`。

<a id="metric-rolling_ic"></a>
## rolling_ic — rolling_ic

Rolling window IC values over time

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic_summary.compute_rolling_ic_stats`。

### 数学公式与计算口径


```math
\mu_t={1\over n_t}\sum_{j=\max(0,t-w+1)}^tIC_j
```


 注册项无 compute_fn，声明为 rolling-IC timeseries 上游制品；窗口 $`w`$ 与输出字段必须由制品给出，不能自行设为 60 或求总均值。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-rolling_ic_drawdown"></a>
## rolling_ic_drawdown — Rolling IC Drawdown

Mean of the rolling-window IC drawdown (peak-to-trough), per factor. A more negative value indicates deeper IC drawdowns (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_rolling_ic_drawdown`。

### 数学公式与计算口径


```math
\mu_t={\sum_{j=\max(0,t-59)}^tIC_j\mathbf1_{finite}\over n_t},\quad C_t=\sum_{u\le t}\tilde\mu_u,\quad d_t=C_t-\max_{v\le t}C_v,\quad M=\mathrm{mean}_td_t
```


 默认 window=60、min_periods=20；$`n_t\lt 20`$ 时 $`\mu_t`$ 为 NaN，累加时以 0 代替该 NaN；这是累计滚动均值的加法回撤，不是财富回撤。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `window` | `60` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/stability_regime.py#L151)；`quant_evaluator.metrics.stability_regime.compute_rolling_ic_drawdown`。

<a id="metric-rolling_ic_volatility"></a>
## rolling_ic_volatility — Rolling IC Volatility

Time-mean of the rolling-window IC standard deviation, per factor. Lower values indicate a more stable IC series (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_rolling_ic_volatility`。

### 数学公式与计算口径


```math
s_t=\sqrt{{\sum_{j\in W_t}(IC_j-\bar I_t)^2\over n_t-1}},\qquad M=\mathrm{mean}_{t:s_t\ finite}s_t
```


 默认 window=60、min_periods=20；窗口使用有限 IC，输出是滚动样本标准差的时间均值。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `window` | `60` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/stability_regime.py#L121)；`quant_evaluator.metrics.stability_regime.compute_rolling_ic_volatility`。

<a id="metric-rolling_rank_ic_ir"></a>
## rolling_rank_ic_ir — Rolling Rank IC IR

Time-mean of the rolling-window IC information ratio, per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_rolling_rank_ic_ir`。

### 数学公式与计算口径


```math
IR_t={\mu_t\over s_t},\qquad M=\mathrm{mean}_{t:IR_t\ finite}IR_t
```


 默认 window=60、min_periods=20；$`s_t`$ 是窗口样本标准差（ddof=1），零方差窗口无效，不年化。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `window` | `60` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L215)；`quant_evaluator.metrics.predictive.compute_rolling_rank_ic_ir`。

<a id="metric-rolling_rank_ic_mean"></a>
## rolling_rank_ic_mean — Rolling Rank IC Mean

Time-mean of the rolling-window mean rank IC, per factor. A smoother estimate of average predictive power (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_rolling_rank_ic_mean`。

### 数学公式与计算口径


```math
\mu_t={1\over n_t}\sum_{j\in W_t}IC_j,\qquad M=\mathrm{mean}_{t:\mu_t\ finite}\mu_t
```


 默认 window=60、min_periods=20；输出是滚动均值序列的时间均值，不是序列本身。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `window` | `60` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/predictive.py#L203)；`quant_evaluator.metrics.predictive.compute_rolling_rank_ic_mean`。

<a id="metric-shape_bootstrap_confidence"></a>
## shape_bootstrap_confidence — Shape Bootstrap Rank Agreement (legacy ID)

Descriptive moving-block bootstrap RANK AGREEMENT, not U-shape probability, per factor: fraction of window-resamples whose order reproduces the overall mean profile order (W >= 3). 1 = shape ordering reproduced in every resample. Deterministic (seeded). NaN for single-window.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_shape_bootstrap_confidence`。

### 数学公式与计算口径


```math
\bar q=\mathrm{mean}_{w=1}^Wq_w,\quad \bar q^{(b)}=\mathrm{mean}_{w\in B_b}q_w,\quad M={\sum_{b\in B^{\ast}}\mathbf1[\rho_S(\bar q^{(b)},\bar q)\ge\tau]\over|B^{\ast}|}
```


 两 ID 调同一实现；默认 block_length=2、resamples=100、seed=0、$`\tau=0.5`$；移动块采样窗口，$`W\lt 3`$ 或可比有限分位少于 3 为 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `block_length` | `2` |
| `resamples` | `100` |
| `random_seed` | `0` |
| `agreement_threshold` | `0.5` |

实现核对：[函数定义](../metrics/shape_evidence.py#L655)；`quant_evaluator.metrics.shape_evidence.compute_shape_bootstrap_confidence`。

<a id="metric-shape_bootstrap_rank_agreement"></a>
## shape_bootstrap_rank_agreement — Block Bootstrap Rank Agreement

Descriptive moving-block resampling agreement with the observed mean rank profile; not U-family success probability

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_shape_bootstrap_confidence`。

### 数学公式与计算口径


```math
\bar q=\mathrm{mean}_{w=1}^Wq_w,\quad \bar q^{(b)}=\mathrm{mean}_{w\in B_b}q_w,\quad M={\sum_{b\in B^{\ast}}\mathbf1[\rho_S(\bar q^{(b)},\bar q)\ge\tau]\over|B^{\ast}|}
```


 两 ID 调同一实现；默认 block_length=2、resamples=100、seed=0、$`\tau=0.5`$；移动块采样窗口，$`W\lt 3`$ 或可比有限分位少于 3 为 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `block_length` | `2` |
| `resamples` | `100` |
| `random_seed` | `0` |
| `agreement_threshold` | `0.5` |

实现核对：[函数定义](../metrics/shape_evidence.py#L655)；`quant_evaluator.metrics.shape_evidence.compute_shape_bootstrap_confidence`。

<a id="metric-shape_regime_stability"></a>
## shape_regime_stability — Shape Regime Stability

Regime version of shape stability: inverse-Fisher correlation of CONSECUTIVE window profiles per factor (W >= 3 windows). Drops a single common shape that is stable overall but regime-uncorrelated. NaN when fewer than 3 windows.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_shape_regime_stability`。

### 数学公式与计算口径


```math
r_w=\mathrm{corr}(q_w,q_{w+1}),\qquad M=\tanh\left({1\over K}\sum_w\mathrm{arctanh}(r_w)\right)
```


 实际比较连续窗口的 Pearson profile 相关；至少 3 个窗口、每对至少 3 个共同有限分位；单 profile/证据不足返回 NaN。


实现核对：[函数定义](../metrics/shape_evidence.py#L612)；`quant_evaluator.metrics.shape_evidence.compute_shape_regime_stability`。

<a id="metric-shape_stability"></a>
## shape_stability — Shape Stability

Inverse-Fisher aggregated window-vs-leave-one-out correlation of the quantile profile across W windows per factor. Consumes a (W, n_quantiles, F) windowed profile panel. 1 = identical shape in every window. NaN for a single-window profile (stability is undefined - missing evidence, never 0/1).

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_shape_stability`。

### 数学公式与计算口径


```math
r_w=\mathrm{corr}(q_w,\mathrm{mean}_{u\ne w}q_u),\qquad M=\tanh\left({1\over K}\sum_{w\in K}\mathrm{arctanh}(r_w)\right)
```


 需要至少 2 个窗口；每次至少 3 个共同有限分位且两 profile 非常数；单 profile 明确返回 NaN。


实现核对：[函数定义](../metrics/shape_evidence.py#L576)；`quant_evaluator.metrics.shape_evidence.compute_shape_stability`。

<a id="metric-sharpe_ratio"></a>
## sharpe_ratio — sharpe_ratio

Annualized Sharpe ratio of a return series per factor: mean(excess return) / std(excess return, ddof=1) * sqrt(periods_per_year), with excess return = return - risk_free_rate / periods_per_year. NaN when fewer than min_periods finite returns or when the return standard deviation is not positive-finite. period: daily (252).

- 版本：`1.0.1`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_sharpe_ratio`。

### 数学公式与计算口径


```math
x_t=r_t-r_f/A,\qquad M=\sqrt A\,{\bar x\over s_x}
```


 默认 $`r_f=0`$、$`A=252`$、min_periods=20；$`s_x`$ 为 ddof=1，非有限收益剔除，$`s_x\le10^{-10}`$ 为 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `risk_free_rate` | `0.0` |
| `periods_per_year` | `252` |
| `min_periods` | `20` |

实现核对：[函数定义](../metrics/portfolio_stats.py#L334)；`quant_evaluator.metrics.portfolio_stats.compute_sharpe_ratio`。

<a id="metric-sidak_correction"></a>
## sidak_correction — Sidak Correction

Sidak multiple-testing correction: adjusted p = 1 - (1 - p)^n. Assumes independence, slightly less conservative than Bonferroni (spec §34).

- 版本：`4.0.0`；状态：`stable`；层级：`research`。
- 输入依赖：`p_values`。
- 输出：`scalar`；单位：`pvalue`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.multiple_testing.sidak_correction`。

### 数学公式与计算口径


```math
m=\#\{p_i\ finite\},\quad \alpha_S=1-(1-\alpha)^{1/m},\quad p_i'=\min\{1,1-(1-p_i)^m\},\quad reject_i=\mathbf1[p_i\le\alpha_S]
```


 默认 alpha=0.05；非有限 p 保持 NaN 且 reject=False，有限 p 必须在 [0,1]。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `alpha` | `0.05` |

实现核对：[函数定义](../metrics/multiple_testing.py#L283)；`quant_evaluator.metrics.multiple_testing.sidak_correction`。

<a id="metric-size_exposure"></a>
## size_exposure — Size Exposure

Signed mean size-style exposure of the factor (typed per-style field). NaN when style absent or no finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_size_exposure`。

### 数学公式与计算口径


```math
z_{t,size}=\beta_{t,size}{sd_w(X_{t,size})\over sd_w(f_t)},\qquad M={1\over |T^{\ast}|}\sum_{t\in T^{\ast}}z_{t,size}
```


 实际先逐日以截距和全部 style 暴露对因子做加权最小二乘，再标准化 size 系数，最后取其有限时间均值；不是证券暴露的加权平均，也不是 raw OLS 系数。默认 min_obs=10、weights=None；已有 FactorLoadingSeries 已绑定因子和权重，缺少 size 字段返回 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `factor_values` | `None` |
| `min_obs` | `10` |
| `weights` | `None` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L622)；`quant_evaluator.metrics.exposure_evidence.compute_size_exposure`。

<a id="metric-skewness"></a>
## skewness — skewness

Skewness of the return distribution

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.distribution.compute_skewness`。

### 数学公式与计算口径


```math
g_1={1\over n}\sum((x-\bar x)/\sigma_0)^3,\qquad M={\sqrt{n(n-1)}\over n-2}g_1
```


 实现为 scipy.stats.skew(nan_policy='omit', bias=False)，默认 axis=0、min_obs=10；注册项无 compute_fn，只能使用该上游 distribution 制品。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-sortino_ratio"></a>
## sortino_ratio — sortino_ratio

Annualized Sortino ratio of a return series per factor: mean(excess return) / downside deviation * sqrt(periods_per_year), where downside deviation is the root mean square of negative excess returns only (no ddof). NaN when fewer than min_periods finite returns, when there are no negative excess returns, or when the downside deviation is not positive-finite.

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_sortino_ratio`。

### 数学公式与计算口径


```math
x_t=r_t-h,\quad d=\sqrt{{\sum_{x_t\lt 0}x_t^2\over D}},\quad M={\bar x\over d}\times\begin{cases}\sqrt A,&annualization='sqrt_frequency'\\1,&'none'\end{cases}
```


 默认 $`A=252`$、risk_free_rate=0、mar=None、min_periods=20、downside_denominator='negative'，故 $`h=r_f/A`$、$`D`$ 为负 excess 个数；若 denominator='all' 则 $`D=n`$。无下行或 $`d\le10^{-12}`$ 为 NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `risk_free_rate` | `0.0` |
| `periods_per_year` | `252` |
| `min_periods` | `20` |
| `downside_denominator` | `'negative'` |
| `mar` | `None` |
| `annualization` | `'sqrt_frequency'` |

实现核对：[函数定义](../metrics/portfolio_stats.py#L566)；`quant_evaluator.metrics.portfolio_stats.compute_sortino_ratio`。

<a id="metric-spearman_ic"></a>
## spearman_ic — spearman_ic

Spearman rank correlation between factor values and forward returns

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`drop_pair`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic.compute_daily_ic`。

### 数学公式与计算口径


```math
IC_t=\mathrm{corr}(\mathrm{rank}_{avg}f_{t,i},\mathrm{rank}_{avg}y_{t,i})
```


 注册项无 compute_fn，只声明 daily-IC 上游实现；逐日删除非有限配对并使用平均秩，不能独立执行或擅自再做时间均值。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-staleness"></a>
## staleness — Staleness

Mean fraction of assets whose value is unchanged from the prior day, per factor. A high staleness ratio indicates a slow-moving / sticky factor (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`2`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_staleness`。

### 数学公式与计算口径


```math
M={\sum_{t=1}^{T-1}\sum_i\mathbf1[f_{t,i}=f_{t-1,i},\ f_{t,i},f_{t-1,i}\ finite]\over\sum_{t=1}^{T-1}\sum_i\mathbf1[f_{t,i},f_{t-1,i}\ finite]}
```


 按因子对所有相邻日/资产共同有限配对汇总；$`T\lt 2`$ 返回 NaN，共同有限数为 0 时实现分母钳到 1、结果为 0。


实现核对：[函数定义](../metrics/data_quality.py#L68)；`quant_evaluator.metrics.data_quality.compute_staleness`。

<a id="metric-subsample_stability"></a>
## subsample_stability — Subsample IC Stability

Standard deviation of mean IC across bootstrap subsamples

- 版本：`0.1.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`40`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`subsample_stability`。

### 数学公式与计算口径


```math
m=\max(1,\lfloor0.8T\rfloor),\quad S_b\sim\mathrm{SampleWithoutReplacement}(\{1,\ldots,T\},m),\quad \bar I_b=\mathrm{nanmean}_{t\in S_b}IC_t,\quad M=\sqrt{{\sum_{b\in B^{\ast}}(\bar I_b-\bar{\bar I})^2\over |B^{\ast}|-1}}
```


 实际返回 100 个无放回随机子样本均值之间的样本标准差（ddof=1），非 $`1-std`$、也不是有放回 bootstrap；默认 num_subsamples=100、subsample_fraction=0.8、random_seed=0。外层先要求原 IC 序列有限值数 min_periods=40；子样本均值忽略 NaN，最终标准差也忽略非有限子样本均值。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `40` |
| `num_subsamples` | `100` |
| `subsample_fraction` | `0.8` |
| `random_seed` | `0` |

实现核对：[函数定义](../metrics/registry_adapters.py#L201)；`quant_evaluator.metrics.registry_adapters.compute_subsample_stability_value`。

<a id="metric-tail_vs_middle_contrast"></a>
## tail_vs_middle_contrast — Tail vs Middle Contrast

Mean |tail returns - middle return| per factor (tails = outer quartiles of the quantile range). High for U/inverted-U profiles, low for flat. Always >= 0. NaN when profile too small or tail/middle returns not finite.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_tail_vs_middle_contrast`。

### 数学公式与计算口径

令桶均值为 $`g_0,\ldots,g_{Q-1}`$，采用零基索引。设 $`m=\lfloor Q/2\rfloor`$，$`\ell=\max(1,\min(m-1,\mathrm{round}(Q/4)))`$，$`h=\min(Q-1,\max(m+1,\mathrm{round}(3Q/4)))`$，尾桶集合 $`A=\{0,\ldots,\ell-1,h,\ldots,Q-1\}`$。



```math
C=\frac1{|A|}\sum_{q\in A}|g_q-g_m|
```



至少6桶，中央桶及所有尾桶必须有限，否则NaN。round 使用最接近整数、半整数到偶数规则；当边界不在中央两侧时改为 $`\ell=m-1,h=m+1`$。结果非负，U形与倒U形都可能高，不能据此判断方向。


实现核对：[函数定义](../metrics/shape_evidence.py#L457)；`quant_evaluator.metrics.shape_evidence.compute_tail_vs_middle_contrast`。

<a id="metric-tie_ratio"></a>
## tie_ratio — Tie Ratio

Mean fraction of finite factor values that are tied with another value, per factor. The complement of the distinct level ratio (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_tie_ratio`。

### 数学公式与计算口径

令 $`n_t`$ 为当日有限因子值数、$`u_t`$ 为其中不同值数，$`D=\{t:n_t\gt 0\}`$。



```math
Tie=1-\frac1{|D|}\sum_{t\in D}\frac{u_t}{n_t}
```



先应用因子 validity；全空日期跳过，全部日期为空返回NaN。这是“1−不同取值占比”，不是“所有属于重复组的资产占比”：例如 $`(1,1,2)`$ 得 $`1/3`$，而不是 $`2/3`$。


实现核对：[函数定义](../metrics/data_quality.py#L143)；`quant_evaluator.metrics.data_quality.compute_tie_ratio`。

<a id="metric-time_to_recovery"></a>
## time_to_recovery — Time to Recovery

Mean time (periods) from an underwater episode's trough back to a new wealth high. Only completed recoveries are averaged; censored ages are not recoveries. NaN for no completed event or unknown valuation.

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`periods`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_time_to_recovery`。

### 数学公式与计算口径

对财富曲线识别回撤事件。设事件 $`e`$ 的谷底位置为 $`b_e`$、首次恢复原峰值的位置为 $`r_e`$，$`E`$ 仅包括已恢复事件。



```math
TTR=\frac1{|E|}\sum_{e\in E}(r_e-b_e)
```



单位为输入轴上的期数，**从谷底起算**，不是从峰值起算。默认至少10个有限收益；未知估值导致不可确定路径时NaN。未恢复事件不伪造恢复时间；无已恢复事件也为NaN。可选 max_recovery_lookback 按事件过滤超过上限的已恢复间隔。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `10` |
| `max_recovery_lookback` | `None` |

实现核对：[函数定义](../metrics/underwater.py#L102)；`quant_evaluator.metrics.underwater.compute_time_to_recovery`。

<a id="metric-top_quantile_cliff"></a>
## top_quantile_cliff — Top Quantile Cliff

Top-quantile cliff: ret[top] - ret[top-1], per factor. The jump in return from the second-highest to the highest quantile (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_top_quantile_cliff`。

### 数学公式与计算口径

桶序由低因子到高因子，$`g_q`$ 为第 $`q`$ 桶的平均收益。



```math
C_{\rm top}=g_Q-g_{Q-1}
```



至少2桶，最高两桶都有限，否则NaN。正值表示最高桶相对次高桶跳升。


实现核对：[函数定义](../metrics/quantile_shape.py#L179)；`quant_evaluator.metrics.quantile_shape.compute_top_quantile_cliff`。

<a id="metric-top_quantile_cliff_robust"></a>
## top_quantile_cliff_robust — Top Quantile Cliff (Robust)

ROBUST top cliff per factor: Q_K - mean(Q_(K-3)..Q_(K-1)) (plan §14.4 - contrast the top bucket against the mean of the three PRIOR buckets, not the noisy one-bin Q_K - Q_(K-1)). Direction higher_is_better (big positive jump into the top bucket). NaN when the top 4 quantile returns are not all finite.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_top_quantile_cliff_robust`。

### 数学公式与计算口径


```math
C_{\rm top,robust}=g_Q-\frac{g_{Q-1}+g_{Q-2}+g_{Q-3}}3
```



$`g_q`$ 为按因子升序排列的桶平均收益；至少4桶，最高四桶都有限。这里对照的是三个不同的先前桶，不包含最高桶；缺失则NaN。


实现核对：[函数定义](../metrics/shape_evidence.py#L724)；`quant_evaluator.metrics.shape_evidence.compute_top_quantile_cliff_robust`。

<a id="metric-top_tail_slope"></a>
## top_tail_slope — Top Tail Slope

Mean adjacent return difference over the TOP segment of the quantile profile as drawn (quantiles K-3..K-1), per factor. Positive for a positively inclined factor; large = top-tail cliff. Direction NEUTRAL: the metric measures the drawn profile's top segment; orientation is informative, never assumed (plan §14.4).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_top_tail_slope`。

### 数学公式与计算口径


```math
S_{\rm top}=\frac{(g_Q-g_{Q-1})+(g_{Q-1}-g_{Q-2})}{2}=\frac{g_Q-g_{Q-2}}2
```



$`g_q`$ 为第 $`q`$ 桶收益。至少3桶，最高三桶全有限，否则NaN。单位为每跨一个桶的收益变化，不是对时间回归的斜率。


实现核对：[函数定义](../metrics/shape_evidence.py#L418)；`quant_evaluator.metrics.shape_evidence.compute_top_tail_slope`。

<a id="metric-tracking_error"></a>
## tracking_error — Tracking Error

Benchmark/invested-capital evidence from an explicitly bound execution trajectory leg

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`annualized_return`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.long_only.compute_tracking_error`。

### 数学公式与计算口径

对已绑定的净主动收益 $`a_t`$（策略净收益减基准）：



```math
TE=\sqrt{A}\sqrt{\frac{\sum_{t\in V}(a_t-\bar a)^2}{|V|-1}}
```



$`V`$ 是有限收益集合，$`A`$ 默认252；默认至少2期。函数自身不再减一次基准，必须由输入轨迹保证主动收益口径。常数序列的TE为0；样本不足为NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `periods_per_year` | `252` |
| `min_periods` | `2` |

实现核对：[函数定义](../metrics/long_only.py#L17)；`quant_evaluator.metrics.long_only.compute_tracking_error`。

<a id="metric-tradable_coverage"></a>
## tradable_coverage — Tradable Coverage

Fraction of days with at least ``min_assets`` jointly valid cells, per factor. A tradable day is one where the factor and label are both available for enough assets (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_tradable_coverage`。

### 数学公式与计算口径

设 $`v_{tif}=1`$ 当且仅当应用 validity 后因子和标签共同有限，$`T`$ 为全部日期数。



```math
C_f=\frac1T\sum_{t=1}^T\mathbf1\left(\sum_i v_{tif}\ge m\right),\qquad m=10
```



这是达到最低有效配对数的“日期比例”，不是资产单元覆盖率，也不自动证明这些股票能真实成交。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_assets` | `10` |

实现核对：[函数定义](../metrics/data_quality.py#L188)；`quant_evaluator.metrics.data_quality.compute_tradable_coverage`。

<a id="metric-train_predictive_dimension"></a>
## train_predictive_dimension — Train Predictive Dimension

Train-side predictive dimension of the authorized train-vs-validation comparison (plan §13.6), per factor (e.g. mean daily rank IC on the train evaluation).

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.train_predictive_dimension`。

### 数学公式与计算口径


```math
D_f^{\mathrm{train}}=A.\mathrm{train\_predictive\_dimension}_f
```



这里 $`A`$ 为训练—验证证据制品，$`D`$ 是显式绑定的预测维度（例如该分区的平均RankIC），并非“有效因子个数”。输入来自已授权、绑定相同因子ID/版本及指标实例的训练—验证制品；当前注册函数仅投影制品字段，不重新拟合。缺失字段为NaN，不可从密封测试集补造。


实现核对：[函数定义](../registry/metrics.py#L3917)；`quant_evaluator.registry.metrics.<lambda>`。

<a id="metric-train_validation_icir_delta"></a>
## train_validation_icir_delta — Train-Validation ICIR Delta

Absolute ICIR delta (validation - train) per factor (plan §13.6). Delta, never a ratio.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`ratio`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas`。

### 数学公式与计算口径


```math
\Delta_f=ICIR_{\mathrm{validation},f}-ICIR_{\mathrm{train},f}
```



这是验证减训练的**有符号差值**，不是取绝对值，也不是比率。任一侧非有限则NaN；接近零的训练值并不阻止计算差值。输入来自已授权、绑定相同因子ID/版本及指标实例的训练—验证制品；当前注册函数仅投影制品字段，不重新拟合。缺失字段为NaN，不可从密封测试集补造。


实现核对：[函数定义](../registry/metrics.py#L4014)；`quant_evaluator.registry.metrics.<lambda>`。

<a id="metric-train_validation_rankic_delta"></a>
## train_validation_rankic_delta — Train-Validation RankIC Delta

Absolute rank-IC delta (validation - train) per factor (plan §13.6). A DELTA, never a ratio: well-defined even when train is near zero. NaN when either side is not finite.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas`。

### 数学公式与计算口径


```math
\Delta_f=\overline{RankIC}_{\mathrm{validation},f}-\overline{RankIC}_{\mathrm{train},f}
```



这是验证减训练的**有符号差值**，不是取绝对值，也不是比率。任一侧非有限则NaN；接近零的训练值并不阻止计算差值。输入来自已授权、绑定相同因子ID/版本及指标实例的训练—验证制品；当前注册函数仅投影制品字段，不重新拟合。缺失字段为NaN，不可从密封测试集补造。


实现核对：[函数定义](../registry/metrics.py#L3991)；`quant_evaluator.registry.metrics.<lambda>`。

<a id="metric-train_validation_shape_delta"></a>
## train_validation_shape_delta — Train-Validation Shape Delta

Absolute shape-evidence delta (validation - train) per factor (plan §13.6). Delta, never a ratio.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`score`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas`。

### 数学公式与计算口径


```math
\Delta_f=Shape_{\mathrm{validation},f}-Shape_{\mathrm{train},f}
```



这是验证减训练的**有符号差值**，不是取绝对值，也不是比率。任一侧非有限则NaN；接近零的训练值并不阻止计算差值。输入来自已授权、绑定相同因子ID/版本及指标实例的训练—验证制品；当前注册函数仅投影制品字段，不重新拟合。缺失字段为NaN，不可从密封测试集补造。


实现核对：[函数定义](../registry/metrics.py#L4060)；`quant_evaluator.registry.metrics.<lambda>`。

<a id="metric-train_validation_sharpe_delta"></a>
## train_validation_sharpe_delta — Train-Validation Sharpe Delta

Absolute long/short Sharpe delta (validation - train) per factor (plan §13.6). Delta, never a ratio.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`ratio`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas`。

### 数学公式与计算口径


```math
\Delta_f=Sharpe_{\mathrm{validation},f}-Sharpe_{\mathrm{train},f}
```



这是验证减训练的**有符号差值**，不是取绝对值，也不是比率。任一侧非有限则NaN；接近零的训练值并不阻止计算差值。输入来自已授权、绑定相同因子ID/版本及指标实例的训练—验证制品；当前注册函数仅投影制品字段，不重新拟合。缺失字段为NaN，不可从密封测试集补造。


实现核对：[函数定义](../registry/metrics.py#L4037)；`quant_evaluator.registry.metrics.<lambda>`。

<a id="metric-turnover"></a>
## turnover — Portfolio Turnover

Average turnover rate for factor-based portfolios

- 版本：`3.0.0`；状态：`stable`；层级：`core`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`2`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`turnover`。

### 数学公式与计算口径

当日至少有2个有效因子值时，先将有效值转为平均并列秩 $`R_{ti}`$；无信号资产的代理权重明确设0：



```math
w_{ti}=\frac{R_{ti}}{\sum_{j\in V_t}R_{tj}}\ (i\in V_t),\quad w_{ti}=0\ (i\notin V_t),\qquad \tau_t=\frac12\sum_i|w_{ti}-w_{t-1,i}|,\qquad TO=\frac1{|D|}\sum_{t\in D}\tau_t
```



$`D`$ 是两日权重均已知的相邻转换集合。人数不足的日期整行未知，涉及该日的换手为NaN。默认 min_periods=2，因此至少1个有效相邻转换。它是排名代理权重的半L1换手，**不是 $`(1-\rho_S)/2`$，也不是真实成交额**。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `2` |

实现核对：[函数定义](../metrics/registry_adapters.py#L135)；`quant_evaluator.metrics.registry_adapters.compute_turnover_value`。

<a id="metric-turnover_adjusted_ic"></a>
## turnover_adjusted_ic — turnover_adjusted_ic

IC adjusted for turnover-induced transaction costs

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.turnover.compute_turnover_contribution`。

### 数学公式与计算口径


```math
M=\mathrm{UNAVAILABLE}
```



该注册ID没有直接 compute_fn，不能输出实测“换手调整IC”。implementation_id 指向的低层函数实际计算逐资产换手贡献，并未落实IC调整公式；不能猜成 $`IC/TO`$。必须由显式上游合同补足定义后另行接线。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-turnover_cost"></a>
## turnover_cost — turnover_cost

Mean realized per-period declared execution cost drag in basis points

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`bps`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.turnover_cost.compute_turnover_cost`。

### 数学公式与计算口径


```math
Cost_{\rm bp}=10^4\frac1{|V|}\sum_{t\in V}c_t
```



$`c_t`$ 必须来自带类型的 cost_drag 成本轨迹，是非负成本率，不能传收益、毛净收益差或因子排名换手。NaN视为未观测，负数和无穷报错；默认至少1个观测。结果单位为基点。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `1` |

实现核对：[函数定义](../metrics/turnover_cost.py#L8)；`quant_evaluator.metrics.turnover_cost.compute_turnover_cost`。

<a id="metric-turnover_rate"></a>
## turnover_rate — turnover_rate

Average rate of change in factor ranking between periods

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.turnover.compute_turnover`。

### 数学公式与计算口径


```math
\tau=\frac12\sum_i|w_i^{\rm new}-w_i^{\rm old}|
```



这是该ID所定位的低层权重换手函数的实际公式，输入为同形一维权重，任一未知/无穷坐标使本次换手NaN。 当前注册项无直接 compute_fn，统一执行器不可独立计算此ID；低层公式说明不等于已接线。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-turnover_stability"></a>
## turnover_stability — turnover_stability

Variance of turnover rate across periods

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`variance`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.turnover.compute_turnover`。

### 数学公式与计算口径


```math
\tau=\frac12\sum_i|w_i^{\rm new}-w_i^{\rm old}|
```



这是该ID所定位的低层权重换手函数的实际公式，输入为同形一维权重，任一未知/无穷坐标使本次换手NaN。但并未绑定跨期“稳定性”的归约公式；不能把上述单次换手称为标准差或稳定度。 当前注册项无直接 compute_fn，统一执行器不可独立计算此ID；低层公式说明不等于已接线。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-u_shape_score"></a>
## u_shape_score — U-Shape Score

U-shape score in [0, 1] per factor (plan §14.3). NOT RankIC-about-0 detection: combines U-template R^2 (middle underperforms both tails, convex), the fraction of positive (convex) interior second differences, and the requirement that the U-template fit EXCEEDS the monotone-linear template fit. 1 = textbook U, 0 = not U.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`score`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_u_shape_score`。

### 数学公式与计算口径

将桶位置映射到 $`x_q\in[0,1]`$，分别对桶收益 $`g_q`$ 拟合带截距的一元回归：U模板 $`(x_q-0.5)^2`$ 与线性模板 $`x_q`$，得到 $`R_U^2,R_L^2`$。令 $`c_+`$ 为有效连续三桶的二阶差分严格为正的比例。



```math
U=\begin{cases}0,&R_U^2\le R_L^2\ \text{或}\ c_+\lt 0.5,\\0.5\max(R_U^2,0)+0.3c_++0.2(\max(R_U^2,0)-R_L^2),&\text{其他情况}.\end{cases}
```



至少4个有限桶，且有连续三桶可计算曲率；否则NaN。全平曲线为0。各回归 $`R^2=1-\sum(g-\hat g)^2/\sum(g-\bar g)^2`$。此分数综合模板拟合和曲率，不是“RankIC接近0就算U形”。


实现核对：[函数定义](../metrics/shape_evidence.py#L125)；`quant_evaluator.metrics.shape_evidence.compute_u_shape_score`。

<a id="metric-universe_churn"></a>
## universe_churn — Universe Churn

Mean fraction of the tradable universe that changes membership per day, per factor. Churn = fraction of assets tradable on exactly one of two adjacent days (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`2`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_universe_churn`。

### 数学公式与计算口径

令 $`v_{tif}`$ 表示因子与标签共同有效，原资产轴大小为 $`N`$：



```math
Churn_f=\frac1{T-1}\sum_{t=2}^{T}\frac{\sum_i\mathbf1(v_{tif}\ne v_{t-1,i,f})}{N}
```



validity先应用，缺失代表不在该日共同有效集合。少于2期NaN；分母是完整资产轴 $`N`$，不是两日有效集合的并集。


实现核对：[函数定义](../metrics/data_quality.py#L208)；`quant_evaluator.metrics.data_quality.compute_universe_churn`。

<a id="metric-validation_predictive_dimension"></a>
## validation_predictive_dimension — Validation Predictive Dimension

Validation-side predictive dimension of the authorized train-vs-validation comparison (plan §13.6), per factor.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.validation_predictive_dimension`。

### 数学公式与计算口径


```math
D_f^{\mathrm{validation}}=A.\mathrm{validation\_predictive\_dimension}_f
```



这里 $`A`$ 为训练—验证证据制品，$`D`$ 是显式绑定的预测维度（例如该分区的平均RankIC），并非“有效因子个数”。输入来自已授权、绑定相同因子ID/版本及指标实例的训练—验证制品；当前注册函数仅投影制品字段，不重新拟合。缺失字段为NaN，不可从密封测试集补造。


实现核对：[函数定义](../registry/metrics.py#L3940)；`quant_evaluator.registry.metrics.<lambda>`。

<a id="metric-validation_retention"></a>
## validation_retention — Validation Retention

Robust validation/train retention per factor (plan §13.6): validation/train where the train denominator is stable, NaN with an explicit reason (train_near_zero / sign_flip_guard / insufficient_data) otherwise - never a blind division by a tiny train value. Grade anchors are versioned policy constants in RetentionPolicy.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_validation_retention_array`。

### 数学公式与计算口径

设训练预测维度为 $`a_f`$、验证维度为 $`b_f`$。令不稳定条件 $`U_f`$ 为：两者异号且 $`|a_f|\lt g\max(|a_f|,|b_f|,10^{-9})`$。



```math
R_f=\begin{cases}b_f/a_f,&a_f,b_f\ \text{有限},\ |a_f|\ge\epsilon,\ \neg U_f,\\\mathrm{NaN},&\text{其他情况}.\end{cases}
```



默认版本化策略 $`\epsilon=0.05,g=0.5`$。该指标逐因子返回保留率，允许负值；不跨因子平均。记录分母接近零、符号保护或缺失原因；训练为零时不盲目相除。输入来自已授权、绑定相同因子ID/版本及指标实例的训练—验证制品；当前注册函数仅投影制品字段，不重新拟合。缺失字段为NaN，不可从密封测试集补造。


实现核对：[函数定义](../registry/metrics.py#L3967)；`quant_evaluator.registry.metrics.<lambda>`。

<a id="metric-var_95"></a>
## var_95 — var_95

Value at Risk at 95% confidence level

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.risk.var_cvar.compute_var`。

### 数学公式与计算口径


```math
VaR_c=\max(0,-Q_{1-c}(r)),\qquad c=0.95
```



这是所定位低层函数在**显式给定该置信度、historical方法**下的公式；$`Q`$ 是有限收益样本的线性插值经验分位数，默认最低20期，输出正损失幅度。当前注册ID无直接 compute_fn，未落实独立调用；低层 compute_var 自身默认置信度0.95，不能仅凭 var_99 名称认为已自动传0.99。其他参数化方法不是此历史法公式。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-var_99"></a>
## var_99 — var_99

Value at Risk at 99% confidence level

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.risk.var_cvar.compute_var`。

### 数学公式与计算口径


```math
VaR_c=\max(0,-Q_{1-c}(r)),\qquad c=0.99
```



这是所定位低层函数在**显式给定该置信度、historical方法**下的公式；$`Q`$ 是有限收益样本的线性插值经验分位数，默认最低20期，输出正损失幅度。当前注册ID无直接 compute_fn，未落实独立调用；低层 compute_var 自身默认置信度0.95，不能仅凭 var_99 名称认为已自动传0.99。其他参数化方法不是此历史法公式。


没有可直接调用的compute_fn；不得编造实测值。需满足显式证据/上游制品入口。

<a id="metric-volatility_exposure"></a>
## volatility_exposure — Volatility Exposure

Signed mean volatility-style exposure of the factor (typed per-style field). NaN when style absent or no finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_volatility_exposure`。

### 数学公式与计算口径

每日在因子与全部暴露共同有效且回归权重为正的支持上做含截距WLS。设原始斜率为 $`b_{tk}`$、归一化观测权重为 $`\omega_{ti}`$：



```math
b_t=\arg\min_b\sum_i\omega_{ti}(x_{ti}-Z_{ti}b)^2,\quad s_{\omega}(z)=\sqrt{\sum_i\omega_i(z_i-\bar z_\omega)^2},\quad \beta_{tk}=b_{tk}\frac{s_\omega(Z_k)}{s_\omega(x)},\quad E_{\rm vol}=\frac1{|D|}\sum_{t\in D}\beta_{t,\rm volatility}
```



$`Z`$ 包含截距列；标准化只用于非截距风格斜率。$`D`$ 是目标标准化载荷有限的日期；默认每期至少10个共同有效样本，并满足回归自由度/秩条件。返回 volatility 风格的**标准化有符号载荷**均值，不是原始回归系数或证券暴露均值。零方差、缺少字段或无有效载荷NaN；输入已绑定载荷制品时直接复用，不重复估计。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `factor_values` | `None` |
| `min_obs` | `10` |
| `weights` | `None` |

实现核对：[函数定义](../metrics/exposure_evidence.py#L637)；`quant_evaluator.metrics.exposure_evidence.compute_volatility_exposure`。

<a id="metric-win_rate"></a>
## win_rate — win_rate

Win rate of a return series per factor: fraction of finite returns that are strictly positive, in [0, 1]. NaN when there are no finite returns. (Returns equal to exactly 0 count as neither wins nor losses.)

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_win_rate`。

### 数学公式与计算口径


```math
WinRate=\frac{\sum_{t\in V}\mathbf1(r_t\gt 0)}{|V|}
```



$`V`$ 只含有限收益；等于0不算胜。至少一个有限值即可由低层函数计算，全无有效收益NaN。


实现核对：[函数定义](../metrics/portfolio_stats.py#L645)；`quant_evaluator.metrics.portfolio_stats.compute_win_rate`。

<a id="metric-worst_12m"></a>
## worst_12m — Worst Rolling 252 Periods (legacy ID)

Worst fixed 252-period (trading) block compounded return of the probe daily PnL series. NaN when the series is shorter than one 12m block.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`252`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_worst_period_return`。

### 数学公式与计算口径


```math
R_{\rm worst}(L)=\min_{0\le s\le T-L}\left[\prod_{j=s}^{s+L-1}(1+r_j)-1\right]
```



这是固定长度交易期的滚动最差复利收益，不是自然12个月。底层 period='month'/'quarter'/'year' 对应 $`L=21/63/252`$，min_periods 默认10（实际检查输入轴长度），不足一个完整窗口NaN。**当前该ID直接绑定同一未预设参数的函数，默认仍是 period='month'（21期），不能按名称推断为252期。** 应显式传相应 period，或优先使用对应 worst_rolling_* 指标。 原时间轴不压缩，任一NaN污染窗口后最终普通min也可能返回NaN；不是忽略坏窗口的新版滚动接口。 公共注册层另有最短时间轴门槛252期；不要将函数层默认10与注册门槛混为一谈。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `period` | `'month'` |
| `min_periods` | `10` |

实现核对：[函数定义](../metrics/underwater.py#L131)；`quant_evaluator.metrics.underwater.compute_worst_period_return`。

<a id="metric-worst_calendar_month"></a>
## worst_calendar_month — worst_calendar_month

Worst calendar-period compounded probe return; explicit DA calendar and partial-period policy

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_calendar_month`。

### 数学公式与计算口径


```math
R_{\rm worst}=\min_{p\in P_{\rm eligible}}\left[\prod_{t\in S_p}(1+r_t)-1\right]
```



$`S_p`$ 是绑定 CalendarSnapshot 按本地时区划分的自然月交易日集合。默认 partial_policy='exclude'：日历快照需在该周期两端都有外侧交易日作边界证明，观测覆盖全部预期交易日且收益全部有限，才纳入 $`P_{\rm eligible}`$。显式 include 可以纳入部分周期，但所有实际观测仍须有限。无合格周期NaN；不以21/63/252期滚动窗口代替自然周期。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `partial_policy` | `'exclude'` |

实现核对：[函数定义](../metrics/calendar_returns.py#L202)；`quant_evaluator.metrics.calendar_returns.compute_worst_calendar_month`。

<a id="metric-worst_calendar_quarter"></a>
## worst_calendar_quarter — worst_calendar_quarter

Worst calendar-period compounded probe return; explicit DA calendar and partial-period policy

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_calendar_quarter`。

### 数学公式与计算口径


```math
R_{\rm worst}=\min_{p\in P_{\rm eligible}}\left[\prod_{t\in S_p}(1+r_t)-1\right]
```



$`S_p`$ 是绑定 CalendarSnapshot 按本地时区划分的自然季度交易日集合。默认 partial_policy='exclude'：日历快照需在该周期两端都有外侧交易日作边界证明，观测覆盖全部预期交易日且收益全部有限，才纳入 $`P_{\rm eligible}`$。显式 include 可以纳入部分周期，但所有实际观测仍须有限。无合格周期NaN；不以21/63/252期滚动窗口代替自然周期。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `partial_policy` | `'exclude'` |

实现核对：[函数定义](../metrics/calendar_returns.py#L207)；`quant_evaluator.metrics.calendar_returns.compute_worst_calendar_quarter`。

<a id="metric-worst_calendar_year"></a>
## worst_calendar_year — worst_calendar_year

Worst calendar-period compounded probe return; explicit DA calendar and partial-period policy

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_calendar_year`。

### 数学公式与计算口径


```math
R_{\rm worst}=\min_{p\in P_{\rm eligible}}\left[\prod_{t\in S_p}(1+r_t)-1\right]
```



$`S_p`$ 是绑定 CalendarSnapshot 按本地时区划分的自然年交易日集合。默认 partial_policy='exclude'：日历快照需在该周期两端都有外侧交易日作边界证明，观测覆盖全部预期交易日且收益全部有限，才纳入 $`P_{\rm eligible}`$。显式 include 可以纳入部分周期，但所有实际观测仍须有限。无合格周期NaN；不以21/63/252期滚动窗口代替自然周期。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `partial_policy` | `'exclude'` |

实现核对：[函数定义](../metrics/calendar_returns.py#L212)；`quant_evaluator.metrics.calendar_returns.compute_worst_calendar_year`。

<a id="metric-worst_month"></a>
## worst_month — Worst Rolling 21 Periods (legacy ID)

Worst fixed 21-period (trading) block compounded return of the probe daily PnL series. Uses the codebase's fixed trading-period calendar (21/period month); NaN when the series is shorter than one block.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`21`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_worst_period_return`。

### 数学公式与计算口径


```math
R_{\rm worst}(L)=\min_{0\le s\le T-L}\left[\prod_{j=s}^{s+L-1}(1+r_j)-1\right]
```



这是固定长度交易期的滚动最差复利收益，不是自然月。底层 period='month'/'quarter'/'year' 对应 $`L=21/63/252`$，min_periods 默认10（实际检查输入轴长度），不足一个完整窗口NaN。该ID默认 period='month'，即21期。 原时间轴不压缩，任一NaN污染窗口后最终普通min也可能返回NaN；不是忽略坏窗口的新版滚动接口。 公共注册层另有最短时间轴门槛21期；不要将函数层默认10与注册门槛混为一谈。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `period` | `'month'` |
| `min_periods` | `10` |

实现核对：[函数定义](../metrics/underwater.py#L131)；`quant_evaluator.metrics.underwater.compute_worst_period_return`。

<a id="metric-worst_quarter"></a>
## worst_quarter — Worst Rolling 63 Periods (legacy ID)

Worst fixed 63-period (trading) block compounded return of the probe daily PnL series. NaN when the series is shorter than one block.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`63`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_worst_period_return`。

### 数学公式与计算口径


```math
R_{\rm worst}(L)=\min_{0\le s\le T-L}\left[\prod_{j=s}^{s+L-1}(1+r_j)-1\right]
```



这是固定长度交易期的滚动最差复利收益，不是自然季度。底层 period='month'/'quarter'/'year' 对应 $`L=21/63/252`$，min_periods 默认10（实际检查输入轴长度），不足一个完整窗口NaN。**当前该ID直接绑定同一未预设参数的函数，默认仍是 period='month'（21期），不能按名称推断为63期。** 应显式传相应 period，或优先使用对应 worst_rolling_* 指标。 原时间轴不压缩，任一NaN污染窗口后最终普通min也可能返回NaN；不是忽略坏窗口的新版滚动接口。 公共注册层另有最短时间轴门槛63期；不要将函数层默认10与注册门槛混为一谈。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `period` | `'month'` |
| `min_periods` | `10` |

实现核对：[函数定义](../metrics/underwater.py#L131)；`quant_evaluator.metrics.underwater.compute_worst_period_return`。

<a id="metric-worst_quarter_rank_ic"></a>
## worst_quarter_rank_ic — Worst-Quarter Rank IC

Minimum per-quarter mean rank IC (the factor's worst quarter), per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_worst_quarter_rank_ic`。

### 数学公式与计算口径


```math
\mu_p=\frac1{|V_p|}\sum_{t\in V_p}RankIC_t,\qquad M=\min_{p:|V_p|\gt 0}\mu_p
```



$`V_p`$ 为季度组中的有效日集合，结果是最差组的平均RankIC，不是最差一天。有可解析且长度匹配的 time_index 时按自然周期分组，否则低层实现退回从起点划分的固定交易期块（季度63、年度252，末尾不足块保留）。每组只平均有限IC；总体至少20个有限IC，否则NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `time_index` | `None` |

实现核对：[函数定义](../metrics/predictive.py#L259)；`quant_evaluator.metrics.predictive.compute_worst_quarter_rank_ic`。

<a id="metric-worst_rolling_21d"></a>
## worst_rolling_21d — Worst 21 trading periods (rolling)

Worst fully matured fixed-length compounded return; never a calendar period

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return`。

### 数学公式与计算口径


```math
R_{\rm worst}=\min_{s\in D_L}\left[\prod_{j=s}^{s+L-1}(1+r_j)-1\right],\qquad L=21
```



$`D_L`$ 只含原时间轴上已经完整成熟、每期收益均有限的长度 $`L`$ 窗口。默认 min_periods=1，指至少一个**有效完整窗口**，不许可扩展前缀。未知窗口跳过但不压缩日历，等未知行移出后窗口可恢复有效；输入有限收益低于−100%报错。没有有效窗口NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `window` | `21` |
| `min_periods` | `1` |

绑定参数：`args=(), kwargs={'window': 21}`。

实现核对：[函数定义](../metrics/calendar_returns.py#L39)；`quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return`。

<a id="metric-worst_rolling_252d"></a>
## worst_rolling_252d — Worst 252 trading periods (rolling)

Worst fully matured fixed-length compounded return; never a calendar period

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return`。

### 数学公式与计算口径


```math
R_{\rm worst}=\min_{s\in D_L}\left[\prod_{j=s}^{s+L-1}(1+r_j)-1\right],\qquad L=252
```



$`D_L`$ 只含原时间轴上已经完整成熟、每期收益均有限的长度 $`L`$ 窗口。默认 min_periods=1，指至少一个**有效完整窗口**，不许可扩展前缀。未知窗口跳过但不压缩日历，等未知行移出后窗口可恢复有效；输入有限收益低于−100%报错。没有有效窗口NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `window` | `252` |
| `min_periods` | `1` |

绑定参数：`args=(), kwargs={'window': 252}`。

实现核对：[函数定义](../metrics/calendar_returns.py#L39)；`quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return`。

<a id="metric-worst_rolling_63d"></a>
## worst_rolling_63d — Worst 63 trading periods (rolling)

Worst fully matured fixed-length compounded return; never a calendar period

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return`。

### 数学公式与计算口径


```math
R_{\rm worst}=\min_{s\in D_L}\left[\prod_{j=s}^{s+L-1}(1+r_j)-1\right],\qquad L=63
```



$`D_L`$ 只含原时间轴上已经完整成熟、每期收益均有限的长度 $`L`$ 窗口。默认 min_periods=1，指至少一个**有效完整窗口**，不许可扩展前缀。未知窗口跳过但不压缩日历，等未知行移出后窗口可恢复有效；输入有限收益低于−100%报错。没有有效窗口NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `window` | `63` |
| `min_periods` | `1` |

绑定参数：`args=(), kwargs={'window': 63}`。

实现核对：[函数定义](../metrics/calendar_returns.py#L39)；`quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return`。

<a id="metric-worst_year_rank_ic"></a>
## worst_year_rank_ic — Worst-Year Rank IC

Minimum per-year mean rank IC (the factor's worst year), per factor. A robustness check on how bad the factor's worst annual performance is (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_worst_year_rank_ic`。

### 数学公式与计算口径


```math
\mu_p=\frac1{|V_p|}\sum_{t\in V_p}RankIC_t,\qquad M=\min_{p:|V_p|\gt 0}\mu_p
```



$`V_p`$ 为年度组中的有效日集合，结果是最差组的平均RankIC，不是最差一天。有可解析且长度匹配的 time_index 时按自然周期分组，否则低层实现退回从起点划分的固定交易期块（季度63、年度252，末尾不足块保留）。每组只平均有限IC；总体至少20个有限IC，否则NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `time_index` | `None` |

实现核对：[函数定义](../metrics/predictive.py#L248)；`quant_evaluator.metrics.predictive.compute_worst_year_rank_ic`。

<a id="metric-year_consistency"></a>
## year_consistency — Year Consistency

Fraction of years whose mean IC matches the overall IC sign, per factor. A robustness check on how consistently the factor works across years (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_year_consistency`。

### 数学公式与计算口径


```math
\mu_p=\mathrm{mean}_{t\in V_p}IC_t,\qquad C=\frac{\sum_{p\in P}\mathbf1(\mathrm{sign}\mu_p=\mathrm{sign}\bar{IC})}{|P|}
```



$`P`$ 为有有限组均值的年度组；$`\bar{IC}`$ 为全体有效日的等权均值。零与零同号，零与正负不同号。有可解析且长度匹配的 time_index 时按自然周期分组，否则低层实现退回从起点划分的固定交易期块（季度63、年度252，末尾不足块保留）。每组只平均有限IC；总体至少20个有限IC，否则NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `time_index` | `None` |

实现核对：[函数定义](../metrics/stability_regime.py#L88)；`quant_evaluator.metrics.stability_regime.compute_year_consistency`。

<a id="metric-yearly_rank_ic"></a>
## yearly_rank_ic — Yearly Rank IC

Mean of the per-year mean rank IC, per factor. A robust annual average that down-weights any single strong year (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_yearly_rank_ic`。

### 数学公式与计算口径


```math
\mu_p=\mathrm{mean}_{t\in V_p}RankIC_t,\qquad Y=\frac1{|P|}\sum_{p\in P}\mu_p
```



先求各年均值，再对有数据的年份等权；不同年份有效日数不同，结果不等于全体日期直接等权均值。有可解析且长度匹配的 time_index 时按自然周期分组，否则低层实现退回从起点划分的固定交易期块（季度63、年度252，末尾不足块保留）。每组只平均有限IC；总体至少20个有限IC，否则NaN。

### 函数层默认参数

| 参数 | 默认值 |
|---|---|
| `min_periods` | `20` |
| `time_index` | `None` |

实现核对：[函数定义](../metrics/predictive.py#L170)；`quant_evaluator.metrics.predictive.compute_yearly_rank_ic`。

## 实现核对索引（可选）

正文不要求阅读代码。下列链接仅用于核对共享计算函数及掩码细节。

<details>
<summary>展开共享实现链接</summary>

- [quant_evaluator.metrics.calendar_returns._calendar_metric](../metrics/calendar_returns.py#L103)
- [quant_evaluator.metrics.calendar_returns._local_session_dates](../metrics/calendar_returns.py#L70)
- [quant_evaluator.metrics.calendar_returns._period_id](../metrics/calendar_returns.py#L95)
- [quant_evaluator.metrics.calendar_returns._period_key](../metrics/calendar_returns.py#L87)
- [quant_evaluator.metrics.calendar_returns._validated_returns](../metrics/calendar_returns.py#L26)
- [quant_evaluator.metrics.data_quality._factor_values](../metrics/data_quality.py#L36)
- [quant_evaluator.metrics.data_quality._labels](../metrics/data_quality.py#L43)
- [quant_evaluator.metrics.exposure.compute_factor_loadings](../metrics/exposure.py#L75)
- [quant_evaluator.metrics.exposure.rank_aware_projection](../metrics/exposure.py#L11)
- [quant_evaluator.metrics.exposure_evidence._as_factor_loadings](../metrics/exposure_evidence.py#L407)
- [quant_evaluator.metrics.exposure_evidence._panel_arrays](../metrics/exposure_evidence.py#L417)
- [quant_evaluator.metrics.exposure_evidence._select_style](../metrics/exposure_evidence.py#L608)
- [quant_evaluator.metrics.exposure_evidence.build_factor_loading_series](../metrics/exposure_evidence.py#L365)
- [quant_evaluator.metrics.exposure_evidence.compute_style_exposure_evidence](../metrics/exposure_evidence.py#L431)
- [quant_evaluator.metrics.ic._pairwise_finite_mask](../metrics/ic.py#L20)
- [quant_evaluator.metrics.ic._pearson_correlation](../metrics/ic.py#L34)
- [quant_evaluator.metrics.ic._reject_boolean_ic_series](../metrics/ic.py#L190)
- [quant_evaluator.metrics.ic._spearman_rank_correlation](../metrics/ic.py#L90)
- [quant_evaluator.metrics.ic.compute_daily_ic](../metrics/ic.py#L122)
- [quant_evaluator.metrics.ic.compute_mean_ic](../metrics/ic.py#L213)
- [quant_evaluator.metrics.ic_summary.compute_icir](../metrics/ic_summary.py#L17)
- [quant_evaluator.metrics.label_panel.normalize_label_panel](../metrics/label_panel.py#L11)
- [quant_evaluator.metrics.long_only._matrix](../metrics/long_only.py#L8)
- [quant_evaluator.metrics.multiple_testing._validate_alpha](../metrics/multiple_testing.py#L59)
- [quant_evaluator.metrics.multiple_testing._validate_p_values](../metrics/multiple_testing.py#L27)
- [quant_evaluator.metrics.portfolio_stats._validate_missing_return_policy](../metrics/portfolio_stats.py#L162)
- [quant_evaluator.metrics.portfolio_stats.equal_gross_long_short_returns](../metrics/portfolio_stats.py#L114)
- [quant_evaluator.metrics.portfolio_stats.equal_gross_weights](../metrics/portfolio_stats.py#L97)
- [quant_evaluator.metrics.predictive._as_series](../metrics/predictive.py#L43)
- [quant_evaluator.metrics.predictive._autocorr_lag](../metrics/predictive.py#L132)
- [quant_evaluator.metrics.predictive._mean_of_period_means](../metrics/predictive.py#L91)
- [quant_evaluator.metrics.predictive._period_means](../metrics/predictive.py#L55)
- [quant_evaluator.metrics.predictive._recent_mean](../metrics/predictive.py#L103)
- [quant_evaluator.metrics.predictive._rolling_mean_ir](../metrics/predictive.py#L109)
- [quant_evaluator.metrics.predictive._valid_counts](../metrics/predictive.py#L51)
- [quant_evaluator.metrics.predictive._worst_period](../metrics/predictive.py#L97)
- [quant_evaluator.metrics.quality._valid_pair_mask](../metrics/quality.py#L17)
- [quant_evaluator.metrics.quality.compute_coverage_per_factor](../metrics/quality.py#L94)
- [quant_evaluator.metrics.quantile._percentile_boundaries](../metrics/quantile.py#L198)
- [quant_evaluator.metrics.quantile._searchsorted_bins](../metrics/quantile.py#L240)
- [quant_evaluator.metrics.quantile._validate_quantile_count](../metrics/quantile.py#L77)
- [quant_evaluator.metrics.quantile.assign_quantiles_batch](../metrics/quantile.py#L258)
- [quant_evaluator.metrics.quantile.compute_quantile_returns](../metrics/quantile.py#L316)
- [quant_evaluator.metrics.quantile.compute_quantile_returns_fast](../metrics/quantile.py#L42)
- [quant_evaluator.metrics.quantile_numba.compute_quantile_returns_numba](../metrics/quantile_numba.py#L232)
- [quant_evaluator.metrics.quantile_shape._as_matrix](../metrics/quantile_shape.py#L29)
- [quant_evaluator.metrics.quantile_shape._finite_columns](../metrics/quantile_shape.py#L37)
- [quant_evaluator.metrics.risk.drawdown_analysis.compute_drawdown_series](../metrics/risk/drawdown_analysis.py#L65)
- [quant_evaluator.metrics.risk.drawdown_analysis.drawdown_events](../metrics/risk/drawdown_analysis.py#L12)
- [quant_evaluator.metrics.risk.var_cvar._validate_confidence_level](../metrics/risk/var_cvar.py#L14)
- [quant_evaluator.metrics.risk.var_cvar.compute_cvar](../metrics/risk/var_cvar.py#L221)
- [quant_evaluator.metrics.risk.var_cvar.empirical_expected_shortfall](../metrics/risk/var_cvar.py#L327)
- [quant_evaluator.metrics.robustness._contiguous_sample](../metrics/robustness.py#L115)
- [quant_evaluator.metrics.robustness._validate_bootstrap_policy](../metrics/robustness.py#L98)
- [quant_evaluator.metrics.robustness._validate_hac_policy](../metrics/robustness.py#L91)
- [quant_evaluator.metrics.robustness.compute_block_bootstrap_ci](../metrics/robustness.py#L258)
- [quant_evaluator.metrics.robustness.compute_hac_tstat](../metrics/robustness.py#L197)
- [quant_evaluator.metrics.robustness.compute_hac_variance](../metrics/robustness.py#L123)
- [quant_evaluator.metrics.robustness.compute_subsample_ic](../metrics/robustness.py#L13)
- [quant_evaluator.metrics.robustness.compute_subsample_ic_std](../metrics/robustness.py#L61)
- [quant_evaluator.metrics.shape_evidence._as_matrix](../metrics/shape_evidence.py#L75)
- [quant_evaluator.metrics.shape_evidence._bottom_tail_slope_value](../metrics/shape_evidence.py#L407)
- [quant_evaluator.metrics.shape_evidence._fit_variance_explained](../metrics/shape_evidence.py#L108)
- [quant_evaluator.metrics.shape_evidence._per_window_profile](../metrics/shape_evidence.py#L568)
- [quant_evaluator.metrics.shape_evidence._template_fit_x](../metrics/shape_evidence.py#L101)
- [quant_evaluator.metrics.shape_evidence._top_tail_slope_value](../metrics/shape_evidence.py#L392)
- [quant_evaluator.metrics.shape_evidence._windows_or_single](../metrics/shape_evidence.py#L555)
- [quant_evaluator.metrics.stability_regime._as_series](../metrics/stability_regime.py#L37)
- [quant_evaluator.metrics.stability_regime._period_consistency](../metrics/stability_regime.py#L75)
- [quant_evaluator.metrics.stability_regime._period_means](../metrics/stability_regime.py#L48)
- [quant_evaluator.metrics.stability_regime._regime_split](../metrics/stability_regime.py#L253)
- [quant_evaluator.metrics.stability_regime._valid_counts](../metrics/stability_regime.py#L44)
- [quant_evaluator.metrics.temporal.compute_autocorrelation](../metrics/temporal.py#L18)
- [quant_evaluator.metrics.temporal.compute_factor_turnover_rate](../metrics/temporal.py#L225)
- [quant_evaluator.metrics.temporal.compute_half_life](../metrics/temporal.py#L294)
- [quant_evaluator.metrics.temporal.compute_ic_autocorrelation](../metrics/temporal.py#L117)
- [quant_evaluator.metrics.temporal.compute_mean_rank_stability](../metrics/temporal.py#L194)
- [quant_evaluator.metrics.temporal.compute_rank_stability](../metrics/temporal.py#L138)
- [quant_evaluator.metrics.turnover.compute_turnover_series](../metrics/turnover.py#L125)
- [quant_evaluator.metrics.underwater._as_1d](../metrics/underwater.py#L65)
- [quant_evaluator.metrics.underwater._path_events](../metrics/underwater.py#L73)

</details>

## 定义完整性指纹

生成器检查全部注册指标都有数学口径；实现指纹变化时仍须人工复核公式，指纹本身不证明数学说明正确。

<details>
<summary>展开实现指纹</summary>

| 函数 | SHA-256（源公式） |
|---|---|
| `quant_evaluator.metrics.calendar_returns._calendar_metric` | `103fa02cf478fdd6bfe75205a377146293379ec91678369829d10313d9d8de5d` |
| `quant_evaluator.metrics.calendar_returns._local_session_dates` | `55e8dbd214f87e15f3234eb83979bc82f0447566619b53e13cff2d9826c7336e` |
| `quant_evaluator.metrics.calendar_returns._period_id` | `ce81da132ea0fa3aa42f137802435d678df57c18f7f334263312e2530c255654` |
| `quant_evaluator.metrics.calendar_returns._period_key` | `3265bef3999380c3219372bdf23fee2d83cb3e3dca05d07611cb73827f52f364` |
| `quant_evaluator.metrics.calendar_returns._validated_returns` | `891fd71ce565f28f6d6ba29de4b379267ca1483148e45744aba5816be90c832e` |
| `quant_evaluator.metrics.calendar_returns.compute_worst_calendar_month` | `378588358c580b2badc08f0542bb8d2c4fdc96932d31ccbb196531b14c9b5ad2` |
| `quant_evaluator.metrics.calendar_returns.compute_worst_calendar_quarter` | `d790b2f450f10fef0c91627eaad33b7a5caa4bdb696dc864b30a9c16c2b19c27` |
| `quant_evaluator.metrics.calendar_returns.compute_worst_calendar_year` | `59d3cb3c82cb94d89ac77ae00b04e13c0e9ea4b1616d7e43f432f13e4868bc35` |
| `quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return` | `8355bf5944bbf7b7936ad0acc3c98b04041779c86235b5bea362ad43338db60e` |
| `quant_evaluator.metrics.data_quality._factor_values` | `28f6a3ae67a896a9cac8bc5d6829e106d9dfae4c85062308a9375fbc874d0242` |
| `quant_evaluator.metrics.data_quality._labels` | `befd306e8fde4bf86790eb60eb778bd4778f51a4c97478f7498c527c02a1dd04` |
| `quant_evaluator.metrics.data_quality.compute_cross_section_cardinality` | `c246630d68f9e60c73618b46b3cac8a5d3b45e7239c733f4b0c081472d895ac7` |
| `quant_evaluator.metrics.data_quality.compute_distinct_level_ratio` | `cb52cd1787c8b465e1939a70fa5983a2f76b9b8166c9d039eda76f3f4348f885` |
| `quant_evaluator.metrics.data_quality.compute_effective_n` | `7cd127797fb770eb6dee3a483dff6dbb033eceb9ff78957aaa7d717e52226242` |
| `quant_evaluator.metrics.data_quality.compute_label_maturity` | `3ec3dc7d7d2c8940421a88ae3ebabe6c27779ba043e07a7595ff9e7472a31776` |
| `quant_evaluator.metrics.data_quality.compute_missing_ratio` | `ac8a521fe940cedb0df29af7cd9e609ef52abf7be3f6b416b7771dd211b144cf` |
| `quant_evaluator.metrics.data_quality.compute_missing_timeline` | `3e13fbff80aee55e7e07de67e1ea9d974c874c4e9f21f5c2c8d425df48b0c265` |
| `quant_evaluator.metrics.data_quality.compute_outlier_ratio` | `58b6d72605b9b6c862256b3e9c668955d477c719ad485aec0d2bd9b7c2007d1f` |
| `quant_evaluator.metrics.data_quality.compute_staleness` | `32ae306943e16e5d6a23325b65c379da0abdfdf65b2525d316e289831cedefb8` |
| `quant_evaluator.metrics.data_quality.compute_tie_ratio` | `960079db68aaecf24a0a3dc65c72c76be58db6a5df1a1466ae487e3089f92a19` |
| `quant_evaluator.metrics.data_quality.compute_tradable_coverage` | `f5a4bdf5e92cc638eb126f31d99af32103fabd7c7e8bd4b03d0a515cab76218d` |
| `quant_evaluator.metrics.data_quality.compute_universe_churn` | `bd7290dd8698cffc2dd0505b15e98c65c85c8b036d3d6cc2e0dc49af97cb286d` |
| `quant_evaluator.metrics.exposure.compute_factor_loadings` | `aaed7707c773d62d72e15f44a4608c32e66a51b48b2a8607c60fa925915031b1` |
| `quant_evaluator.metrics.exposure.rank_aware_projection` | `e640d1e095e3625f5daf0726df5c136f97c546100de851884dcffd9c1295103a` |
| `quant_evaluator.metrics.exposure_evidence._as_factor_loadings` | `29e05d0b4303e15d25158b5a277fbae5b163c2fea474eb8d043fdbffbfbee380` |
| `quant_evaluator.metrics.exposure_evidence._panel_arrays` | `7fd55e737add00a26709892db7ee6a927d264938668b2a155f5b6797814cd4b3` |
| `quant_evaluator.metrics.exposure_evidence._select_style` | `6f1411762e0260b304f8f51b1a0ec9ad768e6d91e34bfaa0a9b9f02c4a1d6015` |
| `quant_evaluator.metrics.exposure_evidence.build_factor_loading_series` | `5f7a550c42910e00db31c3f144c8d2fdc2f65e78b48440dee2d59a8f7e4a379b` |
| `quant_evaluator.metrics.exposure_evidence.compute_beta_exposure` | `3286648c78943affd40590bf5964cf065ae3108d9da5f4d54083637ec43203f3` |
| `quant_evaluator.metrics.exposure_evidence.compute_exposure_drift` | `69a419abf67576478eb675e96f28ad2e12b704636bb62764be04df51dafcea31` |
| `quant_evaluator.metrics.exposure_evidence.compute_industry_exposure` | `61086a6825a3338d49d744e375e51704499ed7ca1c0b35a340d4759397ca38b1` |
| `quant_evaluator.metrics.exposure_evidence.compute_liquidity_exposure` | `0904e99e268fe4fc50d1418fe53c9b5a00dcdfd36332592ce3ddf305e518595b` |
| `quant_evaluator.metrics.exposure_evidence.compute_max_absolute_style_exposure` | `1f27bbe844b1b5e398a80bda091fbd1a2c89b7cd2342efc3951787be0082cd72` |
| `quant_evaluator.metrics.exposure_evidence.compute_momentum_exposure` | `a560897662311881ee86f0ee94a99db4bee93fc09afdfa4a01a711af99275ecd` |
| `quant_evaluator.metrics.exposure_evidence.compute_neutralized_rank_ic` | `a2fd22db4f44c299837c1c57a3249f374a53009b53ed8e0b7d916ed53b1cc6e3` |
| `quant_evaluator.metrics.exposure_evidence.compute_purity_ratio` | `dab62ab7b2f4efd0bef349113e7a107ab92c85a4fcf531e9e58af368072fa1b0` |
| `quant_evaluator.metrics.exposure_evidence.compute_residual_rank_ic` | `e7a02ebb34308374cccf27256e13ede601f384e59ddf2012ca6657eee7ee12ec` |
| `quant_evaluator.metrics.exposure_evidence.compute_size_exposure` | `581226de1b2583df9ba1d1b520a3a985db702ee347c9f99f61038cfd4fc64472` |
| `quant_evaluator.metrics.exposure_evidence.compute_style_exposure_evidence` | `8489b5d105ac339a1566602ef60c35d2c92f2a29fd6dc4523f8aa228413b53d5` |
| `quant_evaluator.metrics.exposure_evidence.compute_volatility_exposure` | `c686f6a4b47075a132e79caf213143e227212a0af63392513a8707727fc9a8e5` |
| `quant_evaluator.metrics.ic._pairwise_finite_mask` | `f9943a8e1f717199085d02df95c5f4b86ed466fc0c1b28bc381e83daec72a4b5` |
| `quant_evaluator.metrics.ic._pearson_correlation` | `1e4fc5004d3a36274e67073bfab0e098445bb585147e920fbc66109912ad7762` |
| `quant_evaluator.metrics.ic._reject_boolean_ic_series` | `906995c840d5cdbe79ea225fbc3f6fe61253d993e24ddc92643accdbea08b3ec` |
| `quant_evaluator.metrics.ic._spearman_rank_correlation` | `4e9ef965007cf5bbd49a0f7b4cba971ef6d41f351b3f57ac49c7bb7817dcb532` |
| `quant_evaluator.metrics.ic.compute_daily_ic` | `d612b8eb06284ccf0709e21394a312bbe5e6c8f9f8cda5a4952fd21f349c1009` |
| `quant_evaluator.metrics.ic.compute_ic_std` | `d61c306ec97b6b117ccf93933512552e29fc881140f1cfe8dbfcd8b7c1013f37` |
| `quant_evaluator.metrics.ic.compute_mean_ic` | `eaf6b80b8bfe3b7dae1838c5fece7d7f12d7057c102102b5286d0cadbd4e5c85` |
| `quant_evaluator.metrics.ic.compute_mean_ic_value` | `8c8639913d4d91672fadd341f7c84a23911faaeabce313ca136956f0ba925627` |
| `quant_evaluator.metrics.ic_summary.compute_icir` | `513cc3fda0d2c2b9d8a7deab8a7c77b093882be51950b96148e9e6bfa8d110b6` |
| `quant_evaluator.metrics.label_panel.normalize_label_panel` | `623ee37a5bf5de87790a0475178e8df0469437a3dd9c41c0f3b0859b5d3f1600` |
| `quant_evaluator.metrics.long_only._matrix` | `24d9629100c137f5690085ad6f2cf805bb10fd4d6aa17e7f341041ee8a02dbdc` |
| `quant_evaluator.metrics.long_only.compute_information_ratio` | `86e342c72efeb769d8737f1eb9aa215d844163c6e8ad50cb94791271cf375cfb` |
| `quant_evaluator.metrics.long_only.compute_mean_investment_fraction` | `cc8f419ed4d473b782a97f0e6dc81e13d62d82465c0b9c7f2ed9970db45c01e6` |
| `quant_evaluator.metrics.long_only.compute_relative_max_drawdown` | `3ea3a1ed78b1e833901facca869a1faaa1709d7675cf1dcb20fdcde92ea1d622` |
| `quant_evaluator.metrics.long_only.compute_tracking_error` | `bc70bdaf7f2052cf2c1dfc4ecac43bb6a826eec37c28aac6bf3e875099f7797d` |
| `quant_evaluator.metrics.multiple_testing._validate_alpha` | `353731ccf5bb186d1336d57cd1f2f3d27da29977fe7c2604be93a79b7d370767` |
| `quant_evaluator.metrics.multiple_testing._validate_p_values` | `3b200301d2ede20ba5a1214a1923bd34165241823975ca9185cdd42288fda480` |
| `quant_evaluator.metrics.multiple_testing.benjamini_hochberg_correction` | `077d7dc917113c445528beb558548deda10ba33bfde52ddd0ba5689794762014` |
| `quant_evaluator.metrics.multiple_testing.bonferroni_correction` | `8cd5bf9dd42eedd0eacbe4cd287855a606ab9f3850eec57a9850866972f8b5f4` |
| `quant_evaluator.metrics.multiple_testing.holm_bonferroni_correction` | `4c39def08b3e93ab31466b51aedf9355d3bd88eb1545cbc415d899b170574bc4` |
| `quant_evaluator.metrics.multiple_testing.sidak_correction` | `baab41028ef224afe30e891c418ca4a81fca4eb0d9c6d67a3cfd3e2d3b793046` |
| `quant_evaluator.metrics.portfolio_stats._validate_missing_return_policy` | `3dfef08beb67cfaa3783ec41ecbe0c9a8b48d00c87a1d38ea2c7547cb5828a43` |
| `quant_evaluator.metrics.portfolio_stats.compute_calmar_ratio` | `60d8305ed95f6a200983c18adba8c730eaa785cab630aa80751513f0794aaf32` |
| `quant_evaluator.metrics.portfolio_stats.compute_long_short_returns` | `1a88d4e15c59d49b17106f84e468bdfeedd291a60a44ba366f42e2f31d4ee894` |
| `quant_evaluator.metrics.portfolio_stats.compute_maximum_drawdown` | `9f79e65f449fe2354efd09d3bb30d0586560af0482c30028b8c9b8c81022ba61` |
| `quant_evaluator.metrics.portfolio_stats.compute_sharpe_ratio` | `bc3c1fb9f7857b1a7b97fbd1403d71b7e1f70a9761b1978a4dbecf22974abd2e` |
| `quant_evaluator.metrics.portfolio_stats.compute_sortino_ratio` | `e059188f067c6571d48326f6ec31fc879779a65e9836f0755d8947acfa958425` |
| `quant_evaluator.metrics.portfolio_stats.compute_win_rate` | `e7b0cd50717a8af9ed1ad2246f768ac5ea20e2b1ba29053f931c289c86d622a3` |
| `quant_evaluator.metrics.portfolio_stats.equal_gross_long_short_returns` | `58044ced9028fa23ddcdba26cd3894b3c8f4fdbd393607c6f139ed4d03e16653` |
| `quant_evaluator.metrics.portfolio_stats.equal_gross_weights` | `bd21f497ac7f3bb3bab5b8e949cd5cc84b71d231f1897a4361d53d7efd12f3ce` |
| `quant_evaluator.metrics.predictive._as_series` | `64761316903f12723e1e1942e0a28a2f60e8ddabbeb12a26ab9b1be4250ea939` |
| `quant_evaluator.metrics.predictive._autocorr_lag` | `381e58a6c532930a2ef3eb3107ffd50f859049d8f5c8546f5e2cce96973bf5ee` |
| `quant_evaluator.metrics.predictive._mean_of_period_means` | `35de9030a191f3f42a0b9560d016372b97e134b38ecb42f35d60c80c0954ecef` |
| `quant_evaluator.metrics.predictive._period_means` | `2ac2dc6168281d9047c50d37e294a38753d00855a10f03f51ec573e51c8adc32` |
| `quant_evaluator.metrics.predictive._recent_mean` | `e92755ae21393c79fa146f5417f139093f010d20051ae00db32a79b3627b17fd` |
| `quant_evaluator.metrics.predictive._rolling_mean_ir` | `5e185e60888241ef7285402627ba2e1319bbfd7dc1d1aa4b26be2d8f933a1264` |
| `quant_evaluator.metrics.predictive._valid_counts` | `29c09f90d2517d4a0958fdb169b0880169f5038032c652e7f4ae8e0214c9e024` |
| `quant_evaluator.metrics.predictive._worst_period` | `4138a74f4a2f93b40232fbb693b92b662a83afa1beb34e6597124ef2603bd92b` |
| `quant_evaluator.metrics.predictive.compute_ic_positive_ratio` | `4109cbac42e8e932b32f97ac72d6f34213b5890f22a7585da25e70f6e66c42a0` |
| `quant_evaluator.metrics.predictive.compute_ic_recent_vs_history_delta` | `350301183f00a21a47266b163868abf80b2dbe8ffc3e85bd0dad914bf8ac163f` |
| `quant_evaluator.metrics.predictive.compute_ic_sign_consistency` | `32f04e02d66f725ec20c7c1eb2813c736f94cc1e5d453af953e6e44eb1a52b85` |
| `quant_evaluator.metrics.predictive.compute_monthly_rank_ic` | `a724952763b3186f6100a98d9d7704a08627902390280aed05f565eae571827b` |
| `quant_evaluator.metrics.predictive.compute_quarterly_rank_ic` | `dfe9d3712d6c04fe9d48bee67cad998fb05fe30ca25691f1e24378e9f0243f37` |
| `quant_evaluator.metrics.predictive.compute_rank_ic_decay` | `0e9ccc05e119d328222b0486ca7e9cd5040fd3374168b0292acd4fa9244df1ea` |
| `quant_evaluator.metrics.predictive.compute_rank_ic_positive_ratio` | `69dacc294cd4107ddd8c9cd45a6c122d4aa0ef321483d0a10fb90e5431168e0c` |
| `quant_evaluator.metrics.predictive.compute_recent_12m_rank_ic` | `15542500df68d6a0d4a9fc339fac7aadff88fdb6d865e90afaa4ad53c30443f8` |
| `quant_evaluator.metrics.predictive.compute_recent_3m_rank_ic` | `59d4a8b0a73a0238d48bffb9c62401202b7cd1ab07675da6a4c1ba95f520d14f` |
| `quant_evaluator.metrics.predictive.compute_recent_6m_rank_ic` | `3186d5fef5051e5471b0ac6856f22350c5a4a40fc386e9dd8f0749c689d177db` |
| `quant_evaluator.metrics.predictive.compute_rolling_rank_ic_ir` | `c89d146098b91a44bf305cb1330509404566b7bc6020e438f47285328ee99b88` |
| `quant_evaluator.metrics.predictive.compute_rolling_rank_ic_mean` | `61ed82a654daa60e741d44abaf6768bb0109efa8e7b65598acd42f10013a60fd` |
| `quant_evaluator.metrics.predictive.compute_worst_quarter_rank_ic` | `f28664252ecc516061f3a672e8ffc2287414ac0b7b246d420ef0567b82816882` |
| `quant_evaluator.metrics.predictive.compute_worst_year_rank_ic` | `88523f86be5b6a47dd84ad0a44f00566308b30809a351cd80cead8432aab7015` |
| `quant_evaluator.metrics.predictive.compute_yearly_rank_ic` | `3bcfb31c9e1d4aab95241fee6b7da6b70319f6204e6b92974a6eb9793a86077a` |
| `quant_evaluator.metrics.quality._valid_pair_mask` | `d59adb953f5c638c642cf2182f64b0fbc47284a545f3f8944860bb52e27bfc4a` |
| `quant_evaluator.metrics.quality.compute_coverage_per_factor` | `5d8c23026037e9481ee3a7cd3b6ed1df255ee7a5e0c21ed7eb718fa83344411d` |
| `quant_evaluator.metrics.quantile._percentile_boundaries` | `54b193352f71f621617f3f9e0885d91048bf74821c901d06590665a77182b4ba` |
| `quant_evaluator.metrics.quantile._searchsorted_bins` | `14ea6c1f029264e10fd45d2396c34b2f7773096e89dd3ed8985d7fd3de90f155` |
| `quant_evaluator.metrics.quantile._validate_quantile_count` | `2034e129494bc9950df02d152b2d1de1f360662b7c65caeefc1c7f1cb755aae2` |
| `quant_evaluator.metrics.quantile.assign_quantiles_batch` | `9317b2dc6d614ad166e129412a8038461aeae918c17c5dd373a121cbeae717b1` |
| `quant_evaluator.metrics.quantile.compute_quantile_returns` | `1b5b6ba86fd2baacdceeef54cb6c7ce99bc1e9b9bf862fa889d88316298ce6a9` |
| `quant_evaluator.metrics.quantile.compute_quantile_returns_fast` | `a333a80bd310c5334f771b45bef42c409a63e7586a7004dd0c4d50ce6ba0fc24` |
| `quant_evaluator.metrics.quantile_numba.compute_quantile_returns_numba` | `a4bbec682e4fc712865a43bafb0809df3cd8fbe169b8648c696c78a84524f90e` |
| `quant_evaluator.metrics.quantile_shape._as_matrix` | `179ebe17a5e5133b07c132e0131404c618b147f4c4d032d2197f0855e4e76aa5` |
| `quant_evaluator.metrics.quantile_shape._finite_columns` | `f61eb31021df766312ed7f37577bae6ab43d39ad9ca792d5610fc124dd905318` |
| `quant_evaluator.metrics.quantile_shape.compute_bottom_quantile_cliff` | `258f1edbcd73f1e00bce67629b21f4f595df02afac325e8441b73928aeaaff7c` |
| `quant_evaluator.metrics.quantile_shape.compute_quantile_adjacent_spread` | `0d989520353e45a48ca629a6c942e97b4c9fbe65362cc739d9eb5c4bfff559fb` |
| `quant_evaluator.metrics.quantile_shape.compute_quantile_curvature` | `b113407562475aed34a88ee467aa9cc31474e344ecf802e2b5d303ac2b9ec653` |
| `quant_evaluator.metrics.quantile_shape.compute_quantile_extreme_cliff` | `b2a32ce7d1c06220ea67581366f8baa1ad03ea3084044a8fccf3a3d5e44089b3` |
| `quant_evaluator.metrics.quantile_shape.compute_quantile_monotonicity` | `24f10583948698c67ca7eb1a85056a098e3ed73a17628e91e89dd8cf75d9cb73` |
| `quant_evaluator.metrics.quantile_shape.compute_quantile_rank_monotonicity` | `55b187be9ed79752514f5a376b6b519dc9a89e758750c02eb4885543d7eb57d2` |
| `quant_evaluator.metrics.quantile_shape.compute_quantile_tail_asymmetry` | `ce011cc032643fbc31e25dd04b3e39ff20e16dbab3d6328e17087461e9642a78` |
| `quant_evaluator.metrics.quantile_shape.compute_top_quantile_cliff` | `79387a799cb9537e00f72d9b571f4730386c69a004a4a88722cfdf31a5dc2d67` |
| `quant_evaluator.metrics.registry_adapters.build_daily_quantile_return_artifact` | `ff8ad9b3271ab3bdfe9bb02e8d44ef137493c8bb07f324d0fe6dd22c8473502c` |
| `quant_evaluator.metrics.registry_adapters.compute_block_bootstrap_ci_value` | `978a56ff6af23e5f5e713e3fe8b013ad6f552a6d81133bcac09962d04cc35a33` |
| `quant_evaluator.metrics.registry_adapters.compute_coverage_value` | `197f4aadae48af2e23ddb886d7e4ef8ef99297fd948d27f98dcdbabe346d740d` |
| `quant_evaluator.metrics.registry_adapters.compute_daily_quantile_monotonicity_rate_value` | `508c49fba200a51439943ec9d3c34d5535bf82e6dc35c95d05d2b271304cfb54` |
| `quant_evaluator.metrics.registry_adapters.compute_daily_quantile_monotonicity_series_value` | `ef9d289ae8446240612d033c56231cf3c22b3114939ae3b4b0f4cbd0a52d2ed8` |
| `quant_evaluator.metrics.registry_adapters.compute_factor_turnover_rate_value` | `e5e8410f90aedb103f89ef6983347c0b3810b4bf6d3050b98a97aa3b31471c3e` |
| `quant_evaluator.metrics.registry_adapters.compute_hac_pvalue_value` | `a18f93b7904a2e274d20d79b8a3d98f9d10d9696141d352525a2ed20882bf0be` |
| `quant_evaluator.metrics.registry_adapters.compute_hac_tstat_value` | `a5e8899e8cba91a7501f5c884289374dadde3cf3699da9665271fcb48d2db469` |
| `quant_evaluator.metrics.registry_adapters.compute_half_life_value` | `e42bed114e40ca1445af9d6fd3ed6d8264c4094c4deaba4bcdf0048b6adee29e` |
| `quant_evaluator.metrics.registry_adapters.compute_ic_autocorr_lag1_value` | `d4bc5cc32ca8e2c7c8c29a92fc981c9ada34cf9f877f7caa35fc4c5713105bf3` |
| `quant_evaluator.metrics.registry_adapters.compute_ic_ir_value` | `88ca0d3d630b89cd62f8a998126fb1c6d9e8945210f9b25425eaa24275dd8a73` |
| `quant_evaluator.metrics.registry_adapters.compute_ic_median_value` | `c23a21e950cff5ffc5c60f93b54a906713e793bfd087ddcf44aceee5f28829de` |
| `quant_evaluator.metrics.registry_adapters.compute_pearson_ic_series_value` | `32795368e0635031d30087387eee3888d702e6fb1f9c852791265da45795c8d1` |
| `quant_evaluator.metrics.registry_adapters.compute_pearson_ic_value` | `e3c31bc8093e948dcd88b6f7780223efac1445ff5c38d4d7a8ebe5fbbf85f8b5` |
| `quant_evaluator.metrics.registry_adapters.compute_quantile_returns_full_value` | `205ea7b0b98d2ee03a653382fad3f2c061e18bb45f7d1c5a8a5de9d877dddae8` |
| `quant_evaluator.metrics.registry_adapters.compute_quantile_spread_value` | `5c183a3f26e0cf69112795ef961b9242a8eaad814ca68b42301ad7e93fc4d58a` |
| `quant_evaluator.metrics.registry_adapters.compute_rank_ic_series_value` | `b4fc89ebe6b033e2232b0afa479928fda00d8a015c88c965d76183ef0df6af73` |
| `quant_evaluator.metrics.registry_adapters.compute_rank_ic_value` | `6f6265b28abd476d16151366891fed14960419ca1cdbd9d779cf6dfcd3078fe1` |
| `quant_evaluator.metrics.registry_adapters.compute_rank_stability_value` | `39d524e66ce4ab37789d9825d328f41e8971af703e5f7b5e63de971d305177b8` |
| `quant_evaluator.metrics.registry_adapters.compute_subsample_stability_value` | `489216040cc6db11db8d63cff048db0b2d4a84959f5b13a335e39d564eb08828` |
| `quant_evaluator.metrics.registry_adapters.compute_turnover_value` | `98c39f7e184453db3d4f4a822b0c29d50b4b0178b9148f06ab8a603cfbef0b44` |
| `quant_evaluator.metrics.risk.drawdown_analysis.compute_drawdown_series` | `669f4917b2984564f35e072086ec84c3214e6214b61b2ea136c9580743ac52f6` |
| `quant_evaluator.metrics.risk.drawdown_analysis.drawdown_events` | `7423a0057053609bdff08b921cb2f1063ff7c35e24a2a2790298127d13d56231` |
| `quant_evaluator.metrics.risk.var_cvar._validate_confidence_level` | `c40d00fc14e5083340089c42dbf8f55fcb74b779baf95a53cc0c362573b4c7f1` |
| `quant_evaluator.metrics.risk.var_cvar.compute_cvar` | `d89681c0b25f45aadb748a4a0c50aacb176a5d4d3510d5cbcd7fd53839bab945` |
| `quant_evaluator.metrics.risk.var_cvar.empirical_expected_shortfall` | `9de90307746c22c31bcc6021fa4b293477b4eb66c25195128464c8f55be7ce8e` |
| `quant_evaluator.metrics.robustness._contiguous_sample` | `2363c7c4f334782505bc0d65e8c301e5c0121025d0c1c28651f6d751aade1e15` |
| `quant_evaluator.metrics.robustness._validate_bootstrap_policy` | `dcf9a3b7b26abcb8225b32304b88cd5a1d7e28515b9f6196940a39cf914a8903` |
| `quant_evaluator.metrics.robustness._validate_hac_policy` | `570d3d84faf7cad3e2f4487fcbe15e720f808d12f84ff21e453cd8bf8064f2f3` |
| `quant_evaluator.metrics.robustness.compute_block_bootstrap_ci` | `70690a295aa60e5668eca596f273f038437a3c17300da91182d32f6c8a6e7234` |
| `quant_evaluator.metrics.robustness.compute_hac_tstat` | `0efc235981a2937e80b431e494f4a308ae24965612511b6aeef9e111f095ecdc` |
| `quant_evaluator.metrics.robustness.compute_hac_variance` | `48f6545852f5228d681caf534938deeeeba50baf7d99f46000685f3e34596824` |
| `quant_evaluator.metrics.robustness.compute_subsample_ic` | `a360bcd41a10fdad30b2cf37d948fe1b10a1881d1fcf122d173f3802af0411be` |
| `quant_evaluator.metrics.robustness.compute_subsample_ic_std` | `548c9d1de546f6cc886e106fbec094b00b42ee00b383e0641e2b587ab13451d8` |
| `quant_evaluator.metrics.shape_evidence._as_matrix` | `179ebe17a5e5133b07c132e0131404c618b147f4c4d032d2197f0855e4e76aa5` |
| `quant_evaluator.metrics.shape_evidence._bottom_tail_slope_value` | `4c6a901c2fa05e18d612a8cafbc0db7908b372204c624e204a8edb79f3d56619` |
| `quant_evaluator.metrics.shape_evidence._fit_variance_explained` | `277f1a2da0bb9782b4df460f9b5f9ad85e49c714c57915dc13a231232100bc44` |
| `quant_evaluator.metrics.shape_evidence._per_window_profile` | `7affdef2077b607a6f0dbd290ce9dbae9597d2f96f20b4184afa81f892dba995` |
| `quant_evaluator.metrics.shape_evidence._template_fit_x` | `78f9c015cc8cebcb5ded3b73c7ad2198e003353236a4391561645fecee2dc38a` |
| `quant_evaluator.metrics.shape_evidence._top_tail_slope_value` | `beb6824da30528c86663f4683165757f955c3641adde42586db2d8c97a25ff0a` |
| `quant_evaluator.metrics.shape_evidence._windows_or_single` | `f7a2be5a1d37eb155aa42654fe9fd12a55b3a8173ad2c76fedc0b63070147fb4` |
| `quant_evaluator.metrics.shape_evidence.compute_adaptive_quantile_count` | `2a9f264cf865cd75e3ffbf37c5e967fe68d9dde170f783172f0e30e833138f90` |
| `quant_evaluator.metrics.shape_evidence.compute_bottom_quantile_cliff_robust` | `921fa20437b25896124ccd996f00beb14bb702a9662fc1dd45c9e90a2e58be87` |
| `quant_evaluator.metrics.shape_evidence.compute_bottom_tail_slope` | `bc255a5790167c9ca770ce7a193ce83803a05eb0fdac98d656a4d0e1732a7129` |
| `quant_evaluator.metrics.shape_evidence.compute_inverted_u_score` | `c601546fdbd0bba7d294410c396e0ae96e7ea2eede06fe1ff676c83654a36f56` |
| `quant_evaluator.metrics.shape_evidence.compute_left_right_asymmetry` | `650635b50694eb1df56778a4e4d8366e027514f5a719f71b4b9839e6be00b0e8` |
| `quant_evaluator.metrics.shape_evidence.compute_linear_trend_score` | `e6b733cd884bb92e2c6222f5f715a6e93348b608277ef1ca1b38e60829e7b480` |
| `quant_evaluator.metrics.shape_evidence.compute_shape_bootstrap_confidence` | `9021965656f6334fbc93bca8c6b7a613f68c61be064ef0c14f520287b8d5d0bb` |
| `quant_evaluator.metrics.shape_evidence.compute_shape_regime_stability` | `1c5789573d1625324172c04204d3aa0577f5e4225bcc79247d3e4d19bd5db5f7` |
| `quant_evaluator.metrics.shape_evidence.compute_shape_stability` | `1ccc91d5a8d52c42d37d7c633b7e3e3668997a00cad9c850dc2a681210666c54` |
| `quant_evaluator.metrics.shape_evidence.compute_tail_vs_middle_contrast` | `238dafb17e7d731a4dab207673d908aea1c711bc68ce9bf52c315eaafc439612` |
| `quant_evaluator.metrics.shape_evidence.compute_top_quantile_cliff_robust` | `0e5219863f1524a3b109f7527758669a3021abe63ae24152fe6db6e5ffd06605` |
| `quant_evaluator.metrics.shape_evidence.compute_top_tail_slope` | `1d03b960dabb932c2d341f7d20ee2aaf7e55e5a9974b11b0e2d9b20ab061c303` |
| `quant_evaluator.metrics.shape_evidence.compute_u_shape_score` | `f1cc830ff4ec46891118e6f7957813c147c49f1313b78b2b61409439b4c28ee1` |
| `quant_evaluator.metrics.stability_regime._as_series` | `8a9a98b5c01ab754406daf3a64f9aeb2f2530a60c2a34800712311ff7229d145` |
| `quant_evaluator.metrics.stability_regime._period_consistency` | `63af5be96352577e035fb5029936123c7ffa819b44eac1aeb42ccdb1c18e4368` |
| `quant_evaluator.metrics.stability_regime._period_means` | `750357cbf7b63ee09920f91d2b42efcd4b2977c1c7e0226c148fb9de800e569b` |
| `quant_evaluator.metrics.stability_regime._regime_split` | `de3e6f41edbdc64d32cd71d0361aaf1f4246fb3205789ce311a62b4b03673c1a` |
| `quant_evaluator.metrics.stability_regime._valid_counts` | `29c09f90d2517d4a0958fdb169b0880169f5038032c652e7f4ae8e0214c9e024` |
| `quant_evaluator.metrics.stability_regime.compute_change_point_score` | `0ba77d3f6e34f586054ee4d1e5c034e370c2b00161d67c3c9815141eaf2984e7` |
| `quant_evaluator.metrics.stability_regime.compute_cusum_break_score` | `e829c1004b71c34c0ba2e42d3670ad2c34e05720517dd948bdcec245ef02ceaa` |
| `quant_evaluator.metrics.stability_regime.compute_ic_sign_flip_rate` | `764de92ae608e2dcfdf57d843ee8ade5b9d21e4b8d6c75fc59ed0f0c42429266` |
| `quant_evaluator.metrics.stability_regime.compute_month_consistency` | `58b4d49d8207d9e10efa7a3169b1444aa10b8a5909d14f354fa0b76b1cd59d6c` |
| `quant_evaluator.metrics.stability_regime.compute_quarter_consistency` | `edfb37e8384b5a987e416b297292341a13fcf50c4ac779f6ada1929507670b1e` |
| `quant_evaluator.metrics.stability_regime.compute_recent_degradation_score` | `efd80c42e888c69c2beebb6e80311641587a0374724a0cc2f4af857873556e7c` |
| `quant_evaluator.metrics.stability_regime.compute_regime_conditional_ic` | `4600f6ec28e019ee55bac81a0a7142270fbf6665f08efeb81484e98e6ec9d5aa` |
| `quant_evaluator.metrics.stability_regime.compute_regime_dispersion` | `3d52e4f2c37d744cbe8762851e864c5e6d7deb7330b72653acf0272893461463` |
| `quant_evaluator.metrics.stability_regime.compute_regime_sign_consistency` | `0c2d9a9f5b1438ed1675d42633b2092ad7243cbaab43dd04e7ddaa23da856968` |
| `quant_evaluator.metrics.stability_regime.compute_regime_worst_ic` | `6862dbf02845711a23317365b17c6cf23967588aba9a9d6507178f4fad2d9532` |
| `quant_evaluator.metrics.stability_regime.compute_rolling_ic_drawdown` | `856ff2a101262941ff232edb042ef7bf0fc76fe4e8320c48067ec26b29c91909` |
| `quant_evaluator.metrics.stability_regime.compute_rolling_ic_volatility` | `d2c8d6fc3a0dd6d70a1cdc0417901608ae93d78a479c58cdbb064c0ff2a725f5` |
| `quant_evaluator.metrics.stability_regime.compute_year_consistency` | `6d4aa93103d4d084d5cd0b2f279cdab4eee2b89a75c52f32e89ab6d24ae88d6d` |
| `quant_evaluator.metrics.temporal.compute_autocorrelation` | `f5e8a2976ef20231db20cd3c7fff3e4271f25e840b992952ab502c02cddb8fb6` |
| `quant_evaluator.metrics.temporal.compute_factor_turnover_rate` | `ba51d30cdd5aadef18924b263d46564f95842d087b862b1a07f2d94e17f40c93` |
| `quant_evaluator.metrics.temporal.compute_half_life` | `041ad9702c3776062c946d0027ff78f5af5c2de3806b7fc8233542f1e1a5589a` |
| `quant_evaluator.metrics.temporal.compute_ic_autocorrelation` | `2041044885a25f9294b85059466759d55a9d63d9770dcde7e94646d018e70a86` |
| `quant_evaluator.metrics.temporal.compute_mean_rank_stability` | `88bb6005dd8f4e979212c8ffe343ad826e470747631f7ec69956805bfec94860` |
| `quant_evaluator.metrics.temporal.compute_rank_stability` | `a29ac205be5bef3cdf82a83cbaed318e3dffe88f74ca2192c6ae8cec12a63fe1` |
| `quant_evaluator.metrics.turnover.compute_turnover_series` | `bb63754a8a8f7167f4395ab85004135c5f29215c87c648f42cf8935f61884327` |
| `quant_evaluator.metrics.turnover_cost.compute_turnover_cost` | `4393d560afd261dbd62c3125b68d735ef23cb7ef52b6b7bbad6ea92346eb3d6a` |
| `quant_evaluator.metrics.underwater._as_1d` | `ab58eec4852a0657ae29653d7e83673e82e220fde3315fe1eacda1095cf442f6` |
| `quant_evaluator.metrics.underwater._path_events` | `def3f0eee4c144e32872e1fa7fa87f90c7fb9870222887a6099fc715d663be38` |
| `quant_evaluator.metrics.underwater.compute_cvar_expected_shortfall` | `52b96219b3353639e244593345b0a1c26ad9fa94b34d1d32b55a4a5e5f904155` |
| `quant_evaluator.metrics.underwater.compute_downside_deviation` | `8709385959df597408c8be00692dd4d9ebad572a412aff25011189071e261121` |
| `quant_evaluator.metrics.underwater.compute_max_underwater_duration` | `477840ee34c9ebb1843e8fc161c123202d0d1d89b532d959d5ae4f56bf3fc02a` |
| `quant_evaluator.metrics.underwater.compute_mean_underwater_duration` | `ff09d9c05ce2483c7eb176213bfba4313c10a9d218f244907a170d7a63a48118` |
| `quant_evaluator.metrics.underwater.compute_return_skew` | `5fb9ae0836c09901fe99ae2c637c02bdcf5fe63d04e13ed289eaf4c2b62549ac` |
| `quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail` | `53ae866389a4049f77ebcd40030f75e38eb42fe472b2644bf57a549ebeebaf77` |
| `quant_evaluator.metrics.underwater.compute_time_to_recovery` | `ed7c042e255922f755904cbf200c06df3796eb44f290df61eb9d92f6ffd1411f` |
| `quant_evaluator.metrics.underwater.compute_worst_period_return` | `ea08291c31af370abca0a5646299598d96ad82d7b930c2106ed274cce224322b` |
| `quant_evaluator.registry.metrics.<lambda>` | `dc185263c1d0e833a4280026b7ba4d72fc22224dc852299a582f75dee485f7b0` |

</details>
