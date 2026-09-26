# quant_evaluator 全指标 CPU vs GPU 对锦标赛

> 2026-09-26 时效性说明：下文是 9 月 25 日当时源码的测量快照，
> 不能当作当前逐指标自动路由表。现行公共 evaluate 已将 backend="gpu"
> 路由到 CUDA；quantile_stability 也已分块向量化并修正分块前的
> 日期对临时数组膨胀。新的 1250 日×5000 股合成压力测试和
> 701 日×5314 股注册真实行情 IC 内核 A/B 见
> [2026-09-26 后端测量](benchmarks/backend_tournament_20260926.md)。
> 两种样本中 Pearson/Spearman IC 的 Numba 内核最快；该结果不能
> 推论所有 164 个注册指标，也尚未替公共 evaluate 接入逐指标路由。

最后更新：2026-09-25（Asia/Hong_Kong）

> **声明：本文档只是测量与建议，未改动任何接线。** 全部数据来自只读基准脚本
> `/home/sunhaiwei/qe_opt/gpu_tournament.py`（复用 `audit_all_metrics.py` 的 fixture），
> 结果 JSON 在 `/home/sunhaiwei/qe_opt/gpu_tournament.json`，context 级错误补充在
> `/home/sunhaiwei/qe_opt/gpu_tournament_ctx0_errors.json`。接线决策留给主 agent。

## 1. 测量环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA L20（46068 MiB），驱动 580.126.20 |
| cupy | 14.2.0 |
| CPU | 32 核 |
| Python | /home/sunhaiwei/quant_projects/.venv（python3.12） |
| 正式树 | /home/sunhaiwei/quant_projects（2026-09-25/26 夜间状态；比 9/24 audit 时多注册了一批 ic_summary 家族指标） |
| fixture | S=T250/N50，L=T1250/N300；4 组 context（probe/holding/cost/calendar），每指标取第一个成功 context，cpu/gpu/cuda 三后端 context 完全一致（0 错配） |
| 计时口径 | 每指标每后端 1 次未计时探测调用（兼作 GPU warmup 与 context 选择）+ 2 次计时取最优；单调用 60s watchdog |
| parity 家规 | rtol≤1e-8，atol≤1e-10，equal_nan；对 cuda 加速 ≥2× 的指标做 cpu-vs-cuda metric value 对拍 |
| 总耗时 | 9.7 min（140 指标 × 2 档 × 3 后端） |

## 2. 关键发现（先读这个）

1. **`backend="gpu"` 是静默 CPU 回退，不是 GPU。** runtime 分发只认 `"cuda"`/`"cuda_strict"`
   （`runtime/evaluator.py:1548-1550`），全仓库 runtime/api/contracts 中不存在 `"gpu"` 这个值的归一化。
   别名审计（rank_ic，3 次取最优，单位 ms）：

   | scale | backend=None | "cpu" | **"gpu"** | "cuda" |
   |---|---|---|---|---|
   | S | 5.68 | 5.63 | **5.81** | 4.96 |
   | L | 69.3 | 68.6 | **68.5** | 28.6 |

   全量 139 个可测指标中 `gpu_ms` 与 `cpu_ms` 分布一致（偏差只出现在小指标计时噪声内）。
   **任何把 `backend="gpu"` 当 GPU 用的调用方实际都在跑 CPU。** 真正的 GPU 路由值是 `"cuda"`。
2. **cuda 实现覆盖率 27/140（约 19%）。** 113 个指标 cuda 路径全失败，其中 112 个触及计划校验
   `UnsupportedMetricError: CUDA has no implementation for [...]`（fail-closed，属预期守卫，不算缺陷），
   ic_decay 被 pre-branch 契约守卫拦截（见 §5.2）。
3. **exposure 套件在 cuda 上慢 10–25×**（L 档 ~2.7–3.0s vs CPU ~0.3s），`worst_rolling_*` 也慢 2–3×。
   这些指标即使已接线 cuda 也**不应该**用。
4. **GPU 加速随规模衰减（crossover 现象）**：S 档 `quantile_spread` ×6.87、
   `daily_quantile_monotonicity_rate` ×5.61、`quantile_monotonicity` ×3.86；
   到 L 档同样指标只剩 ×1.7 左右，没有任何指标达到 ×2。CPU 矩阵变大后向量化摊薄、
   而 cuda 路径固定开销（会话/拷贝/planning）占比上升。**当前 cuda 路径只在小面板（S 档量级）有明显优势。**
5. **`quantile_stability` 在 L 档 CPU 上病态慢（>60s watchdog 超时）**：内核
   `metrics/ic_summary.py:934 compute_quantile_rank_stability` 逐窗口 `np.corrcoef` 循环
   （T=1250/N300 下 >60s；S 档 129ms 正常）。cuda 无实现。该指标 9/24 之后才注册。
   锦标赛表中 L 档它的计时来自 watchdog 超时回退到 ctx3（小日历 fixture，T≈262/N4，23.6ms），
   **不可与其它 L 档数据直接比较**。
6. **`ic_decay` 在 CPU 上就不可测**：`requires unavailable artifact builder HorizonMeanIC`
   —— 无论 backend，这是 artifact builder 接线缺口（4 组 context 都没有可用的 HorizonMeanIC）。
7. 9/24 audit 时未注册的 10 个指标中 9 个（autocorrelation_ic、ic_summary、ic_stability 等）现已注册可测
   （cpu_ok=139/140），但全部无 cuda 实现。

## 3. 锦标赛表

cuda 加速比 = cpu_ms / cuda_ms；`GPU可用=✅` 表示 cuda 路径返回了数值。
parity 列只对加速 ≥2× 的指标做了对拍。表按各档加速比降序。

### 3.1 S 档（T=250, N=50）

| metric | cpu_ms | gpu_ms | cuda_ms | cuda加速比 | GPU可用 | parity(≥2×) |
|---|---|---|---|---|---|---|
| quantile_spread | 52.83 | 36.01 | 7.69 | 6.87 | ✅ | OK |
| daily_quantile_monotonicity_rate | 54.11 | 36.90 | 9.64 | 5.61 | ✅ | OK |
| quantile_monotonicity | 31.25 | 28.01 | 8.10 | 3.86 | ✅ | OK |
| pearson_ic | 10.31 | 10.10 | 5.51 | 1.87 | ✅ | — |
| daily_quantile_monotonicity_series | 19.30 | 17.86 | 11.01 | 1.75 | ✅ | — |
| rank_ic | 8.96 | 8.77 | 8.63 | 1.04 | ✅ | — |
| adaptive_quantile_count | 21.99 | 21.75 | 21.99 | 1.00 | ✅ | — |
| worst_calendar_year | 6.21 | 5.47 | 6.26 | 0.99 | ✅ | — |
| worst_rolling_252d | 4.60 | 4.59 | 4.68 | 0.98 | ✅ | — |
| win_rate | 4.47 | 4.43 | 4.92 | 0.91 | ✅ | — |
| sortino_ratio | 4.90 | 4.59 | 5.49 | 0.89 | ✅ | — |
| worst_calendar_quarter | 6.01 | 6.00 | 7.04 | 0.85 | ✅ | — |
| sharpe_ratio | 4.57 | 4.50 | 5.40 | 0.85 | ✅ | — |
| max_drawdown | 4.57 | 4.56 | 5.98 | 0.76 | ✅ | — |
| worst_calendar_month | 5.88 | 5.88 | 8.85 | 0.66 | ✅ | — |
| calmar_ratio | 4.75 | 5.83 | 7.44 | 0.64 | ✅ | — |
| worst_rolling_63d | 6.24 | 6.28 | 23.91 | 0.26 | ✅ | — |
| worst_rolling_21d | 6.95 | 7.00 | 28.20 | 0.25 | ✅ | — |
| exposure_drift | 20.35 | 21.86 | 506.55 | 0.04 | ✅ | — |
| size_exposure | 19.11 | 19.61 | 506.31 | 0.04 | ✅ | — |
| momentum_exposure | 18.93 | 19.43 | 503.94 | 0.04 | ✅ | — |
| max_absolute_style_exposure | 19.27 | 19.25 | 519.90 | 0.04 | ✅ | — |
| volatility_exposure | 18.81 | 18.44 | 511.97 | 0.04 | ✅ | — |
| purity_ratio | 18.46 | 18.35 | 504.01 | 0.04 | ✅ | — |
| beta_exposure | 18.25 | 18.22 | 499.21 | 0.04 | ✅ | — |
| liquidity_exposure | 19.41 | 19.50 | 532.70 | 0.04 | ✅ | — |
| industry_exposure | 18.72 | 18.84 | 534.15 | 0.04 | ✅ | — |
| quantile_stability | 129.18 | 105.18 | — | — | ❌ | — |
| residual_rank_ic | 33.71 | 33.12 | — | — | ❌ | — |
| neutralized_rank_ic | 33.27 | 33.00 | — | — | ❌ | — |
| shape_bootstrap_rank_agreement | 33.12 | 23.43 | — | — | ❌ | — |
| quantile_curvature | 31.00 | 31.44 | — | — | ❌ | — |
| top_quantile_cliff_robust | 30.60 | 23.55 | — | — | ❌ | — |
| top_tail_slope | 30.17 | 33.41 | — | — | ❌ | — |
| quantile_adjacent_spread | 28.97 | 31.88 | — | — | ❌ | — |
| quantile_returns | 28.87 | 36.58 | — | — | ❌ | — |
| linear_trend_score | 28.86 | 31.23 | — | — | ❌ | — |
| quantile_tail_asymmetry | 28.72 | 28.74 | — | — | ❌ | — |
| tail_vs_middle_contrast | 28.45 | 25.44 | — | — | ❌ | — |
| shape_stability | 27.81 | 18.90 | — | — | ❌ | — |
| shape_bootstrap_confidence | 26.78 | 30.54 | — | — | ❌ | — |
| quantile_extreme_cliff | 25.89 | 26.92 | — | — | ❌ | — |
| bottom_tail_slope | 24.32 | 30.49 | — | — | ❌ | — |
| bottom_quantile_cliff_robust | 23.95 | 23.08 | — | — | ❌ | — |
| quantile_rank_monotonicity | 23.74 | 28.61 | — | — | ❌ | — |
| top_quantile_cliff | 22.92 | 19.89 | — | — | ❌ | — |
| shape_regime_stability | 22.83 | 28.64 | — | — | ❌ | — |
| u_shape_score | 22.62 | 27.14 | — | — | ❌ | — |
| left_right_asymmetry | 15.91 | 29.53 | — | — | ❌ | — |
| inverted_u_score | 15.76 | 18.09 | — | — | ❌ | — |
| bottom_quantile_cliff | 15.67 | 15.00 | — | — | ❌ | — |
| turnover_rate | 15.31 | 14.64 | — | — | ❌ | — |
| quarter_consistency | 12.05 | 9.47 | — | — | ❌ | — |
| ic_summary | 11.33 | 10.76 | — | — | ❌ | — |
| ic_stability | 10.95 | 10.98 | — | — | ❌ | — |
| turnover_adjusted_ic | 10.79 | 11.19 | — | — | ❌ | — |
| rolling_ic | 10.34 | 10.25 | — | — | ❌ | — |
| autocorrelation_ic | 10.22 | 10.31 | — | — | ❌ | — |
| year_consistency | 9.60 | 9.07 | — | — | ❌ | — |
| ic_positive_ratio | 8.79 | 8.59 | — | — | ❌ | — |
| yearly_rank_ic | 8.77 | 8.99 | — | — | ❌ | — |
| change_point_score | 8.77 | 8.91 | — | — | ❌ | — |
| ic_sign_consistency | 8.75 | 8.78 | — | — | ❌ | — |
| ic_recent_vs_history_delta | 8.73 | 8.69 | — | — | ❌ | — |
| distinct_level_ratio | 8.73 | 5.87 | — | — | ❌ | — |
| quarterly_rank_ic | 8.71 | 8.83 | — | — | ❌ | — |
| cusum_break_score | 8.65 | 8.55 | — | — | ❌ | — |
| spearman_ic | 8.65 | 9.03 | — | — | ❌ | — |
| recent_degradation_score | 8.63 | 8.36 | — | — | ❌ | — |
| rolling_ic_volatility | 8.62 | 8.54 | — | — | ❌ | — |
| worst_quarter_rank_ic | 8.62 | 8.66 | — | — | ❌ | — |
| ic_sign_flip_rate | 8.60 | 8.71 | — | — | ❌ | — |
| rolling_rank_ic_ir | 8.56 | 8.49 | — | — | ❌ | — |
| sidak_correction | 8.55 | 8.38 | — | — | ❌ | — |
| ic_serial_autocorrelation_lags_1_5_10_20 | 8.55 | 8.74 | — | — | ❌ | — |
| recent_6m_rank_ic | 8.54 | 8.46 | — | — | ❌ | — |
| rank_ic_decay_h01_h05_h10_h20 | 8.51 | 8.52 | — | — | ❌ | — |
| month_consistency | 8.49 | 8.38 | — | — | ❌ | — |
| holm_bonferroni_correction | 8.48 | 9.10 | — | — | ❌ | — |
| monthly_rank_ic | 8.46 | 9.37 | — | — | ❌ | — |
| rolling_ic_drawdown | 8.46 | 8.35 | — | — | ❌ | — |
| recent_3m_rank_ic | 8.46 | 8.49 | — | — | ❌ | — |
| rolling_rank_ic_mean | 8.45 | 8.43 | — | — | ❌ | — |
| benjamini_hochberg_correction | 8.45 | 8.37 | — | — | ❌ | — |
| regime_sign_consistency | 8.44 | 9.23 | — | — | ❌ | — |
| worst_year_rank_ic | 8.42 | 8.35 | — | — | ❌ | — |
| regime_dispersion | 8.41 | 8.43 | — | — | ❌ | — |
| bonferroni_correction | 8.38 | 8.35 | — | — | ❌ | — |
| regime_worst_ic | 8.38 | 8.32 | — | — | ❌ | — |
| recent_12m_rank_ic | 8.35 | 8.45 | — | — | ❌ | — |
| rank_ic_positive_ratio | 8.29 | 8.25 | — | — | ❌ | — |
| regime_conditional_ic | 8.27 | 8.40 | — | — | ❌ | — |
| rank_ic_cross_section | 8.22 | 8.41 | — | — | ❌ | — |
| tie_ratio | 8.22 | 6.32 | — | — | ❌ | — |
| rank_ic_time_series | 8.21 | 8.18 | — | — | ❌ | — |
| hhi_concentration | 7.94 | 7.82 | — | — | ❌ | — |
| hhi_effective_n | 7.77 | 7.76 | — | — | ❌ | — |
| universe_churn | 7.04 | 7.37 | — | — | ❌ | — |
| cvar_expected_shortfall | 6.12 | 4.54 | — | — | ❌ | — |
| cross_section_cardinality | 5.89 | 5.91 | — | — | ❌ | — |
| long_short_returns | 5.54 | 5.40 | — | — | ❌ | — |
| turnover_stability | 5.14 | 5.10 | — | — | ❌ | — |
| staleness | 5.06 | 4.49 | — | — | ❌ | — |
| kurtosis | 4.93 | 5.04 | — | — | ❌ | — |
| validation_predictive_dimension | 4.85 | 4.65 | — | — | ❌ | — |
| time_to_recovery | 4.79 | 4.74 | — | — | ❌ | — |
| skewness | 4.78 | 5.03 | — | — | ❌ | — |
| label_maturity | 4.75 | 4.76 | — | — | ❌ | — |
| drawdown_duration | 4.72 | 4.74 | — | — | ❌ | — |
| downside_deviation | 4.65 | 4.62 | — | — | ❌ | — |
| outlier_ratio | 4.64 | 4.58 | — | — | ❌ | — |
| effective_n | 4.62 | 4.68 | — | — | ❌ | — |
| max_underwater_duration | 4.61 | 4.58 | — | — | ❌ | — |
| cvar_95 | 4.59 | 4.53 | — | — | ❌ | — |
| joint_coverage | 4.59 | 4.63 | — | — | ❌ | — |
| train_validation_shape_delta | 4.59 | 4.59 | — | — | ❌ | — |
| mean_underwater_duration | 4.57 | 4.52 | — | — | ❌ | — |
| cvar_99 | 4.56 | 6.01 | — | — | ❌ | — |
| validation_retention | 4.54 | 4.49 | — | — | ❌ | — |
| worst_month | 4.54 | 4.51 | — | — | ❌ | — |
| return_coverage | 4.53 | 4.43 | — | — | ❌ | — |
| train_validation_rankic_delta | 4.53 | 4.59 | — | — | ❌ | — |
| worst_quarter | 4.52 | 4.48 | — | — | ❌ | — |
| train_validation_sharpe_delta | 4.51 | 4.59 | — | — | ❌ | — |
| var_95 | 4.51 | 4.46 | — | — | ❌ | — |
| worst_12m | 4.51 | 4.52 | — | — | ❌ | — |
| parameter_generalization | 4.49 | 4.49 | — | — | ❌ | — |
| train_validation_icir_delta | 4.49 | 4.53 | — | — | ❌ | — |
| return_skew | 4.48 | 4.56 | — | — | ❌ | — |
| coverage_stability | 4.47 | 4.49 | — | — | ❌ | — |
| train_predictive_dimension | 4.47 | 4.53 | — | — | ❌ | — |
| factor_coverage | 4.46 | 4.75 | — | — | ❌ | — |
| var_99 | 4.45 | 4.48 | — | — | ❌ | — |
| rolling_1y_sharpe_min | 4.45 | 4.42 | — | — | ❌ | — |
| tradable_coverage | 4.41 | 4.35 | — | — | ❌ | — |
| missing_timeline | 4.40 | 4.38 | — | — | ❌ | — |
| rolling_1y_sharpe_q10 | 4.40 | 4.51 | — | — | ❌ | — |
| missing_ratio | 4.39 | 4.43 | — | — | ❌ | — |
| turnover_cost | 1.95 | 1.82 | — | — | ❌ | — |
| ic_decay | — | — | — | — | ❌ | — |

### 3.2 L 档（T=1250, N=300）

| metric | cpu_ms | gpu_ms | cuda_ms | cuda加速比 | GPU可用 | parity(≥2×) |
|---|---|---|---|---|---|---|
| daily_quantile_monotonicity_rate | 162.14 | 161.03 | 93.04 | 1.74 | ✅ | — |
| quantile_spread | 149.95 | 142.24 | 86.71 | 1.73 | ✅ | — |
| daily_quantile_monotonicity_series | 122.80 | 126.33 | 90.56 | 1.36 | ✅ | — |
| quantile_monotonicity | 122.12 | 122.99 | 93.39 | 1.31 | ✅ | — |
| pearson_ic | 116.03 | 117.38 | 89.17 | 1.30 | ✅ | — |
| rank_ic | 136.74 | 143.11 | 113.09 | 1.21 | ✅ | — |
| win_rate | 88.93 | 88.96 | 86.81 | 1.02 | ✅ | — |
| sharpe_ratio | 87.14 | 87.59 | 86.05 | 1.01 | ✅ | — |
| adaptive_quantile_count | 189.29 | 192.54 | 191.47 | 0.99 | ✅ | — |
| sortino_ratio | 95.58 | 95.92 | 96.76 | 0.99 | ✅ | — |
| max_drawdown | 84.16 | 84.72 | 85.21 | 0.99 | ✅ | — |
| calmar_ratio | 87.76 | 90.08 | 89.28 | 0.98 | ✅ | — |
| worst_calendar_year | 5.37 | 5.39 | 6.11 | 0.88 | ✅ | — |
| worst_calendar_quarter | 5.55 | 5.53 | 6.74 | 0.82 | ✅ | — |
| worst_calendar_month | 5.78 | 5.79 | 8.63 | 0.67 | ✅ | — |
| worst_rolling_252d | 94.80 | 94.48 | 167.15 | 0.57 | ✅ | — |
| worst_rolling_21d | 95.89 | 96.17 | 205.19 | 0.47 | ✅ | — |
| worst_rolling_63d | 74.46 | 76.28 | 182.66 | 0.41 | ✅ | — |
| exposure_drift | 338.37 | 310.04 | 2758.94 | 0.12 | ✅ | — |
| beta_exposure | 301.27 | 316.49 | 2763.96 | 0.11 | ✅ | — |
| liquidity_exposure | 299.93 | 305.33 | 2752.27 | 0.11 | ✅ | — |
| industry_exposure | 296.58 | 299.29 | 2740.95 | 0.11 | ✅ | — |
| momentum_exposure | 301.52 | 301.75 | 2788.08 | 0.11 | ✅ | — |
| volatility_exposure | 297.67 | 305.91 | 2766.16 | 0.11 | ✅ | — |
| purity_ratio | 304.27 | 302.23 | 2846.04 | 0.11 | ✅ | — |
| size_exposure | 307.92 | 296.57 | 2908.38 | 0.11 | ✅ | — |
| max_absolute_style_exposure | 314.22 | 314.13 | 2978.33 | 0.11 | ✅ | — |
| residual_rank_ic | 366.49 | 383.30 | — | — | ❌ | — |
| neutralized_rank_ic | 357.78 | 368.91 | — | — | ❌ | — |
| spearman_ic | 146.49 | 149.47 | — | — | ❌ | — |
| turnover_rate | 146.15 | 147.97 | — | — | ❌ | — |
| change_point_score | 143.45 | 139.67 | — | — | ❌ | — |
| shape_bootstrap_confidence | 143.28 | 142.99 | — | — | ❌ | — |
| cusum_break_score | 142.31 | 142.83 | — | — | ❌ | — |
| rolling_ic_drawdown | 141.69 | 135.24 | — | — | ❌ | — |
| regime_worst_ic | 140.12 | 132.17 | — | — | ❌ | — |
| recent_degradation_score | 140.03 | 155.84 | — | — | ❌ | — |
| shape_bootstrap_rank_agreement | 138.80 | 136.91 | — | — | ❌ | — |
| rank_ic_cross_section | 134.31 | 135.95 | — | — | ❌ | — |
| rank_ic_time_series | 133.68 | 136.02 | — | — | ❌ | — |
| quantile_tail_asymmetry | 133.49 | 129.75 | — | — | ❌ | — |
| quarterly_rank_ic | 132.64 | 130.77 | — | — | ❌ | — |
| turnover_adjusted_ic | 130.39 | 130.50 | — | — | ❌ | — |
| regime_conditional_ic | 130.15 | 134.00 | — | — | ❌ | — |
| regime_sign_consistency | 129.62 | 136.34 | — | — | ❌ | — |
| quarter_consistency | 129.33 | 128.09 | — | — | ❌ | — |
| rank_ic_decay_h01_h05_h10_h20 | 129.10 | 127.31 | — | — | ❌ | — |
| regime_dispersion | 129.01 | 140.32 | — | — | ❌ | — |
| rolling_rank_ic_mean | 128.76 | 125.79 | — | — | ❌ | — |
| ic_summary | 128.56 | 116.94 | — | — | ❌ | — |
| month_consistency | 128.45 | 137.32 | — | — | ❌ | — |
| monthly_rank_ic | 128.31 | 127.88 | — | — | ❌ | — |
| rolling_ic_volatility | 128.10 | 126.25 | — | — | ❌ | — |
| rank_ic_positive_ratio | 127.97 | 127.86 | — | — | ❌ | — |
| recent_12m_rank_ic | 127.91 | 127.88 | — | — | ❌ | — |
| recent_3m_rank_ic | 127.87 | 127.96 | — | — | ❌ | — |
| ic_positive_ratio | 127.61 | 126.28 | — | — | ❌ | — |
| ic_serial_autocorrelation_lags_1_5_10_20 | 127.33 | 125.70 | — | — | ❌ | — |
| rolling_rank_ic_ir | 126.69 | 126.67 | — | — | ❌ | — |
| tail_vs_middle_contrast | 126.64 | 118.19 | — | — | ❌ | — |
| bonferroni_correction | 126.63 | 126.42 | — | — | ❌ | — |
| ic_sign_consistency | 126.62 | 127.57 | — | — | ❌ | — |
| recent_6m_rank_ic | 126.38 | 140.02 | — | — | ❌ | — |
| ic_sign_flip_rate | 126.26 | 126.98 | — | — | ❌ | — |
| holm_bonferroni_correction | 126.18 | 127.42 | — | — | ❌ | — |
| ic_recent_vs_history_delta | 126.10 | 126.80 | — | — | ❌ | — |
| inverted_u_score | 125.89 | 118.80 | — | — | ❌ | — |
| sidak_correction | 125.87 | 127.94 | — | — | ❌ | — |
| shape_stability | 125.63 | 128.42 | — | — | ❌ | — |
| quantile_extreme_cliff | 124.42 | 119.43 | — | — | ❌ | — |
| benjamini_hochberg_correction | 124.23 | 125.75 | — | — | ❌ | — |
| worst_quarter_rank_ic | 124.14 | 123.93 | — | — | ❌ | — |
| bottom_quantile_cliff_robust | 123.62 | 144.81 | — | — | ❌ | — |
| bottom_quantile_cliff | 123.40 | 125.05 | — | — | ❌ | — |
| left_right_asymmetry | 123.36 | 122.24 | — | — | ❌ | — |
| bottom_tail_slope | 123.32 | 120.39 | — | — | ❌ | — |
| top_quantile_cliff_robust | 122.85 | 120.94 | — | — | ❌ | — |
| rolling_ic | 121.41 | 119.20 | — | — | ❌ | — |
| top_quantile_cliff | 121.40 | 115.36 | — | — | ❌ | — |
| linear_trend_score | 121.07 | 131.31 | — | — | ❌ | — |
| shape_regime_stability | 121.05 | 142.49 | — | — | ❌ | — |
| u_shape_score | 119.83 | 119.24 | — | — | ❌ | — |
| quantile_returns | 119.28 | 114.06 | — | — | ❌ | — |
| quantile_curvature | 118.22 | 115.33 | — | — | ❌ | — |
| ic_stability | 118.08 | 117.62 | — | — | ❌ | — |
| autocorrelation_ic | 116.97 | 115.26 | — | — | ❌ | — |
| quantile_adjacent_spread | 116.50 | 115.45 | — | — | ❌ | — |
| top_tail_slope | 116.16 | 115.46 | — | — | ❌ | — |
| quantile_rank_monotonicity | 115.91 | 114.92 | — | — | ❌ | — |
| hhi_effective_n | 112.48 | 106.27 | — | — | ❌ | — |
| hhi_concentration | 108.94 | 105.72 | — | — | ❌ | — |
| skewness | 107.01 | 100.19 | — | — | ❌ | — |
| year_consistency | 104.69 | 103.53 | — | — | ❌ | — |
| worst_year_rank_ic | 102.14 | 102.66 | — | — | ❌ | — |
| yearly_rank_ic | 101.05 | 102.95 | — | — | ❌ | — |
| turnover_stability | 100.87 | 100.55 | — | — | ❌ | — |
| cross_section_cardinality | 98.19 | 98.32 | — | — | ❌ | — |
| long_short_returns | 97.43 | 95.61 | — | — | ❌ | — |
| tie_ratio | 97.15 | 96.74 | — | — | ❌ | — |
| distinct_level_ratio | 97.11 | 96.27 | — | — | ❌ | — |
| staleness | 96.22 | 93.33 | — | — | ❌ | — |
| rolling_1y_sharpe_q10 | 93.74 | 86.99 | — | — | ❌ | — |
| rolling_1y_sharpe_min | 92.91 | 101.62 | — | — | ❌ | — |
| factor_coverage | 92.40 | 91.00 | — | — | ❌ | — |
| cvar_95 | 91.82 | 86.61 | — | — | ❌ | — |
| joint_coverage | 91.22 | 90.45 | — | — | ❌ | — |
| outlier_ratio | 90.34 | 88.82 | — | — | ❌ | — |
| coverage_stability | 90.22 | 88.71 | — | — | ❌ | — |
| missing_timeline | 89.87 | 98.98 | — | — | ❌ | — |
| cvar_expected_shortfall | 88.83 | 88.79 | — | — | ❌ | — |
| label_maturity | 88.76 | 88.83 | — | — | ❌ | — |
| max_underwater_duration | 88.73 | 85.36 | — | — | ❌ | — |
| kurtosis | 88.70 | 90.84 | — | — | ❌ | — |
| time_to_recovery | 88.65 | 88.38 | — | — | ❌ | — |
| validation_predictive_dimension | 88.47 | 86.75 | — | — | ❌ | — |
| effective_n | 88.31 | 86.53 | — | — | ❌ | — |
| worst_12m | 87.87 | 87.51 | — | — | ❌ | — |
| cvar_99 | 87.82 | 89.17 | — | — | ❌ | — |
| universe_churn | 87.70 | 87.73 | — | — | ❌ | — |
| train_validation_shape_delta | 87.62 | 87.12 | — | — | ❌ | — |
| return_coverage | 87.61 | 87.60 | — | — | ❌ | — |
| return_skew | 87.39 | 107.91 | — | — | ❌ | — |
| drawdown_duration | 87.38 | 88.65 | — | — | ❌ | — |
| var_99 | 87.13 | 88.19 | — | — | ❌ | — |
| var_95 | 87.11 | 87.45 | — | — | ❌ | — |
| tradable_coverage | 87.06 | 86.99 | — | — | ❌ | — |
| train_validation_sharpe_delta | 86.99 | 87.30 | — | — | ❌ | — |
| train_validation_rankic_delta | 86.92 | 86.87 | — | — | ❌ | — |
| downside_deviation | 86.91 | 88.14 | — | — | ❌ | — |
| train_validation_icir_delta | 86.88 | 87.22 | — | — | ❌ | — |
| parameter_generalization | 86.78 | 86.70 | — | — | ❌ | — |
| train_predictive_dimension | 86.29 | 86.56 | — | — | ❌ | — |
| worst_quarter | 85.49 | 85.62 | — | — | ❌ | — |
| worst_month | 85.47 | 85.96 | — | — | ❌ | — |
| mean_underwater_duration | 84.94 | 85.15 | — | — | ❌ | — |
| validation_retention | 84.07 | 88.23 | — | — | ❌ | — |
| missing_ratio | 83.67 | 83.80 | — | — | ❌ | — |
| quantile_stability | 23.59 | 25.37 | — | — | ❌ | — |
| turnover_cost | 22.33 | 22.15 | — | — | ❌ | — |
| ic_decay | — | — | — | — | ❌ | — |

## 4. Parity 结论

对 cuda 加速 ≥2× 的 3 个指标（全部在 S 档）做了 cpu-vs-cuda 对拍（rtol≤1e-8/atol≤1e-10/equal_nan），
**3/3 全部 OK，无数值偏差**：

| scale | metric | cpu_ms | cuda_ms | 加速比 | parity |
|---|---|---|---|---|---|
| S | quantile_spread | 52.83 | 7.69 | 6.87 | OK |
| S | daily_quantile_monotonicity_rate | 54.11 | 9.64 | 5.61 | OK |
| S | quantile_monotonicity | 31.25 | 8.10 | 3.86 | OK |

L 档无指标达到 ≥2×（最高 ×1.74），按口径未做对拍。

## 5. GPU 路径报错清单（错误原样记录）

### 5.1 cuda：`UnsupportedMetricError: CUDA has no implementation for ['<mid>']`（fail-closed 计划校验）

113 个指标在 cuda 路径全部 context 失败。其中 **112 个**触及 cuda 计划校验
`UnsupportedMetricError`（101 个在所有 context 都是此错误；11 个 generalization/exposure 类指标
在 ctx0/ctx1（证据齐备、守卫放行）报此错误；turnover_cost 仅在 ctx2（cost leg 齐备）报此错误。
ctx2/ctx3 上它们的报错被 **pre-branch 契约守卫提前拦截**（缺 exposure_panel /
缺 typed TrainVsValidationArtifact / 缺 cost leg），这与 ctx 无关的结论一致：
**这 112 个指标没有任何 cuda 内核**。101 个全-context no-impl 指标清单（× 2 档 = 202 条）：


autocorrelation_ic, benjamini_hochberg_correction, bonferroni_correction
bottom_quantile_cliff, bottom_quantile_cliff_robust, bottom_tail_slope, change_point_score
coverage_stability, cross_section_cardinality, cusum_break_score, cvar_95, cvar_99
cvar_expected_shortfall, distinct_level_ratio, downside_deviation, drawdown_duration
effective_n, factor_coverage, hhi_concentration, hhi_effective_n
holm_bonferroni_correction, ic_positive_ratio, ic_recent_vs_history_delta
ic_serial_autocorrelation_lags_1_5_10_20, ic_sign_consistency, ic_sign_flip_rate
ic_stability, ic_summary, inverted_u_score, joint_coverage, kurtosis, label_maturity
left_right_asymmetry, linear_trend_score, long_short_returns, max_underwater_duration
mean_underwater_duration, missing_ratio, missing_timeline, month_consistency
monthly_rank_ic, outlier_ratio, quantile_adjacent_spread, quantile_curvature
quantile_extreme_cliff, quantile_rank_monotonicity, quantile_returns, quantile_stability
quantile_tail_asymmetry, quarter_consistency, quarterly_rank_ic, rank_ic_cross_section
rank_ic_decay_h01_h05_h10_h20, rank_ic_positive_ratio, rank_ic_time_series
recent_12m_rank_ic, recent_3m_rank_ic, recent_6m_rank_ic, recent_degradation_score
regime_conditional_ic, regime_dispersion, regime_sign_consistency, regime_worst_ic
return_coverage, return_skew, rolling_1y_sharpe_min, rolling_1y_sharpe_q10, rolling_ic
rolling_ic_drawdown, rolling_ic_volatility, rolling_rank_ic_ir, rolling_rank_ic_mean
shape_bootstrap_confidence, shape_bootstrap_rank_agreement, shape_regime_stability
shape_stability, sidak_correction, skewness, spearman_ic, staleness
tail_vs_middle_contrast, tie_ratio, time_to_recovery, top_quantile_cliff
top_quantile_cliff_robust, top_tail_slope, tradable_coverage, turnover_adjusted_ic
turnover_rate, turnover_stability, u_shape_score, universe_churn, var_95, var_99
worst_12m, worst_month, worst_quarter, worst_quarter_rank_ic, worst_year_rank_ic
year_consistency, yearly_rank_ic

（共 101 个指标，cuda 路径 fail-closed。
（其中 neutralized_rank_ic / residual_rank_ic / parameter_generalization 及 generalization 家族、
quantile_stability 在 CPU 上可测、cuda 无实现，是「值得补 cuda 内核」的直接候选；
其余多为整组家族缺口。）

### 5.2 契约/fixture 类错误（逐 context 矩阵见 gpu_tournament_ctx0_errors.json）

| metric | ctx0 | ctx2 | 说明 |
|---|---|---|---|
| ic_decay | `InvalidContractError: Metric ic_decay requires unavailable artifact builder HorizonMeanIC` | 同左 | **CPU 也全 context 失败**，与 backend 无关；唯一真正的 artifact-builder 接线缺口 |
| turnover_cost | `InvalidContractError: turnover_cost requires explicitly bound portfolio leg cost_drag` | `UnsupportedMetricError: CUDA has no implementation for ['turnover_cost']` | ctx2（cost leg 齐备）守卫放行后才暴露 cuda 无实现 |
| neutralized_rank_ic / residual_rank_ic | `UnsupportedMetricError: CUDA has no implementation` | `InvalidContractError: ... requires missing exposure_panel input` | ctx0/1 到达 cuda 计划校验；ctx2/3 被 pre-branch 守卫拦截 |
| parameter_generalization / train_predictive_dimension / train_validation_{icir,rankic,shape,sharpe}_delta / validation_predictive_dimension / validation_retention（共 8 个） | `UnsupportedMetricError: CUDA has no implementation` | `InvalidContractError: Generalization metrics require typed TrainVsValidationArtifact` | 同上分层 |

> JSON 中每指标记录的是「最后一个失败 context」的错误，因此这 12 个指标在
> gpu_tournament.json 里显示为契约错误；上表给出了完整的分层视图。

### 5.3 CPU/gpu 路径报错

仅 `ic_decay` 一条（见 5.2）。**无 watchdog 超时**（唯一一次病态慢发生在首跑、
watchdog 引入之前，即 `quantile_stability` L 档；重跑中被 60s watchdog 封顶转为
context 回退，最终在 ctx3 成功）。

### 5.4 9 个指标本次 CPU 也可测（相对 9/24 audit 的变化）

autocorrelation_ic, hhi_concentration, hhi_effective_n, ic_stability, ic_summary,
quantile_stability, return_coverage, turnover_adjusted_ic, turnover_stability
（9/24 时 `Metric function not registered`；现已注册。ic_decay 仍不可测，见 5.2。）

## 6. 结论与建议

### 6.1 GPU（cuda）值得用的指标

只有 S 档量级（T≈250, N≈50）才有 ≥2× 收益：

| metric | S 加速比 | L 加速比 | parity | 建议 |
|---|---|---|---|---|
| quantile_spread | ×6.87 | ×1.73 | OK | S 档用 cuda；crossover 在 S–L 之间 |
| daily_quantile_monotonicity_rate | ×5.61 | ×1.74 | OK | 同上 |
| quantile_monotonicity | ×3.86 | ×1.31 | OK | 同上 |
| pearson_ic | ×1.87 | ×1.30 | 未对拍（<2×） | 边际，小面板可考虑 |
| daily_quantile_monotonicity_series | ×1.75 | ×1.36 | 未对拍（<2×） | 边际 |

**crossover/阈值建议**：把面板规模（T×N 或 T·N·F 元素数）作为路由判据。
本次单点数据只能给出方向：S（250×50=12.5k 元素）GPU 赢 4–7×，L（1250×300=375k 元素）
GPU 优势收敛到 ≤1.8×。建议主 agent 在 250×50 与 1250×300 之间补 2–3 个规模点
（如 500×100、1000×200）标定 crossover 曲线后再定路由阈值；在那之前，
「T×N ≤ ~50k 才走 cuda」是一个保守可用的一级近似。

### 6.2 CPU 更快 / 不该走 GPU 的指标

- **exposure 全家（9 个：beta/size/momentum/volatility/liquidity/industry/purity_ratio/
  max_absolute_style_exposure/exposure_drift）**：cuda 慢 10–25×（L 档 ~2.8s vs ~0.3s，
  S 档 ~0.5s vs ~0.02s）。即使已在 GPUExecutor 白名单里，也应路由回 CPU。
- **worst_rolling_21d/63d/252d**：cuda 慢 2–4×（L 档 167–205ms vs 75–96ms）。
- **worst_calendar_month/quarter/year、calmar_ratio、max_drawdown、sharpe_ratio、
  sortino_ratio、win_rate、adaptive_quantile_count、rank_ic（L 档 ×1.21）**：
  cuda ≈ 或略慢于 CPU（×0.6–1.0），无收益。
- **`backend="gpu"` 字面值**：见发现 1，任何指标用它都等于 CPU。若上游确有传
  `"gpu"` 的调用方，这是第一个要修的接线缺口（加别名归一化或直接改传 `"cuda"`）。

### 6.3 修内核/接线优先级建议（供主 agent 参考，本文档不执行）

1. `quantile_stability` CPU 内核 L 档病态慢（>60s）——向量化或截断窗口数（metrics/ic_summary.py:934）。
2. `backend="gpu"` 别名静默回退——要么支持要么 fail-closed 报错，当前最危险。
3. `ic_decay` 缺 HorizonMeanIC artifact builder——CPU 全 context 不可测。
4. exposure 家族的 cuda 路径若保留白名单，需要重新 tile/降拷贝，否则应从 cuda 计划中摘除。
5. cuda 值得补实现的 CPU 热点：neutralized_rank_ic、residual_rank_ic、spearman_ic、
   rank_ic_time_series/rolling 家族（本次 L 档 CPU 100–160ms 且调用频次高）。

## 7. 口径备注与诚实声明

- gpu_ms 与 cpu_ms 几乎处处相等是**路由事实**（gpu≡CPU 路径）而非复制粘贴；
  少数小指标上 >20% 的相对差是 2 次取最优的计时噪声（S 档 <35ms 的指标），方向随机。
- L 档 quantile_stability 的 cpu/gpu 计时来自 ctx3 回退（见发现 5），仅 23.6/25.4ms，
  **不代表 L 档面板上的真实耗时（>60s）**。
- 每指标只计时 2 次取最优（任务口径），未做更多重复；绝对值有小样本噪声，
  但 ×0.1/×0.12/×5–7 量级的结论远超噪声区间。
- 本文未修改任何仓库代码；gpu_tournament.py、JSON、本文档均为新增产物。
