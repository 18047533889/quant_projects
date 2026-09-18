# QuantEvaluator 全部注册指标计算手册

与 [统一口径与取舍](METRIC_CONVENTIONS.md) 配套。由实际注册表、函数说明与源公式生成；不得手工只改数字。

注册 ID 共 **164** 个，别名不重复计数。下面逐项列出全部 ID，包括实验性或不能单独执行的项目。
默认参数是函数层默认；公开入口额外构建策略见统一口径，尤其 IC 的每日20配对、分桶人数及观察期。
实现源码是精确定义的一部分：保留掩码、分母、边界分支，避免将自定义指标写成名称相近的标准公式。
None/NaN/unsupported不代表0；状态stable也不代表生产可交易或GPU已验收。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_adaptive_quantile_count(factor_batch, policy=None, tie_policy='max')
```

Resolve a tie-aware fixed quantile count independently per factor.

The public path accepts a :class:`FactorBatch`, applies the canonical
quantile tie rule on every date, and returns a ``ScalarMetricArtifact``
whose provenance contains the complete per-date feasibility evidence.
A candidate Q is selected only when every date has all Q occupied buckets
with the policy's minimum effective names.  Missing dates are recorded and
make the fixed-Q comparison insufficient; they are never dropped.

Plain quantile profiles remain accepted for backwards-compatible direct
helper use, where this function only reports their already-built row count.

### 精确计算公式（实际实现）

```python
def compute_adaptive_quantile_count(factor_batch, policy=None, tie_policy="max"):
    """Resolve a tie-aware fixed quantile count independently per factor.

    The public path accepts a :class:`FactorBatch`, applies the canonical
    quantile tie rule on every date, and returns a ``ScalarMetricArtifact``
    whose provenance contains the complete per-date feasibility evidence.
    A candidate Q is selected only when every date has all Q occupied buckets
    with the policy's minimum effective names.  Missing dates are recorded and
    make the fixed-Q comparison insufficient; they are never dropped.

    Plain quantile profiles remain accepted for backwards-compatible direct
    helper use, where this function only reports their already-built row count.
    """
    if not (hasattr(factor_batch, "time_axis") and hasattr(factor_batch, "validity")):
        qr_or_artifact = factor_batch
        if hasattr(qr_or_artifact, "n_quantiles"):
            nq = int(qr_or_artifact.n_quantiles)
            values = np.asarray(qr_or_artifact.values, dtype=np.float64)
        else:
            values = np.asarray(qr_or_artifact, dtype=np.float64)
            if values.ndim == 1:
                values = values[:, None]
            nq = int(values.shape[0])
        if values.ndim != 2 or values.shape[0] == 0:
            return np.array([np.nan], dtype=np.float64)
        return np.where(np.all(np.isfinite(values), axis=0), float(nq), np.nan)

    from quant_evaluator.contracts.adaptive_bins_policy import (
        AdaptiveBinsDateEvidence,
        AdaptiveBinsFactorEvidence,
        AdaptiveBinsPolicy,
        AdaptiveBinsResolution,
    )
    from quant_evaluator.contracts.axis_refs import FactorAxisRef
    from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact
    from quant_evaluator.contracts.quantile_policy import validate_tie_policy
    from quant_evaluator.metrics.quantile import assign_quantiles_batch

    if policy is None:
        policy = AdaptiveBinsPolicy()
    elif not isinstance(policy, AdaptiveBinsPolicy):
        from collections.abc import Mapping
        if isinstance(policy, Mapping):
            policy = AdaptiveBinsPolicy.from_dict(policy)
    if not isinstance(policy, AdaptiveBinsPolicy):
        raise TypeError("policy must be AdaptiveBinsPolicy")
    tie_policy_value = validate_tie_policy(tie_policy).value
    values = np.asarray(factor_batch.values, dtype=np.float64)
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)

    candidates = policy.candidate_bin_counts()
    assignments = {
        q: assign_quantiles_batch(values, n_quantiles=q, method=tie_policy_value)
        for q in candidates
    }
    if values.shape[2] == 1:
        assignments = {q: a[:, :, None] for q, a in assignments.items()}

    output = np.full(factor_batch.num_factors, np.nan, dtype=np.float64)
    factor_evidence = []
    observation_counts = []
    for f, factor_id in enumerate(factor_batch.factor_ids):
        date_rows = []
        candidate_minima = {q: [] for q in candidates}
        for t in range(factor_batch.num_times):
            finite = np.isfinite(values[t, :, f])
            finite_names = int(finite.sum())
            distinct = int(np.unique(values[t, finite, f]).size) if finite_names else 0
            observed = []
            for q in candidates:
                assigned = assignments[q][t, :, f]
                valid_bins = assigned[assigned >= 0]
                minimum = (
                    int(np.bincount(valid_bins, minlength=q).min())
                    if valid_bins.size else None
                )
                observed.append((q, minimum))
                candidate_minima[q].append(minimum)
            applicable = finite_names > 0
            date_rows.append(AdaptiveBinsDateEvidence(
                date_index=t,
                applicable=applicable,
                finite_names=finite_names,
                distinct_levels=distinct,
                candidate_min_bucket_counts=tuple(observed),
                reason="evaluated" if applicable else "missing_factor_values",
            ))

        selected = None
        selected_minimum = None
        for q in candidates:
            counts = candidate_minima[q]
            if counts and all(
                count is not None and count >= policy.min_effective_names_per_bin
                for count in counts
            ):
                selected = q
                selected_minimum = float(min(counts))
                break
        reason = (
            "preferred" if selected == policy.preferred_bins
            else "fallback" if selected is not None
            else "insufficient"
        )
        resolution = AdaptiveBinsResolution(
            bin_count=selected,
            reason=reason,
            min_names_per_bin=selected_minimum,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
        )
        if selected is not None:
            output[f] = float(selected)
        observation_counts.append(sum(row.applicable for row in date_rows))
        factor_evidence.append(AdaptiveBinsFactorEvidence(
            factor_id=factor_id,
            resolution=resolution,
            tie_policy=tie_policy_value,
            comparison_policy="largest_fixed_q_feasible_on_every_date",
            dates=tuple(date_rows),
        ))

    return ScalarMetricArtifact(
        metric_id="adaptive_quantile_count",
        domain="quantile_shape",
        values=output,
        factor_axis=FactorAxisRef(tuple(factor_batch.factor_ids)),
        provenance={
            "adaptive_bins_policy": {
                "policy_id": policy.policy_id,
                "policy_version": policy.policy_version,
                "preferred_bins": policy.preferred_bins,
                "fallback_bins": tuple(policy.fallback_bins),
                "min_effective_names_per_bin": policy.min_effective_names_per_bin,
            },
            "adaptive_bins_coverage": tuple(row.to_dict() for row in factor_evidence),
            "observation_counts": tuple(observation_counts),
        },
    )
```

<a id="metric-autocorrelation_ic"></a>
## autocorrelation_ic — autocorrelation_ic

Autocorrelation of IC values at specified lags

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.temporal.compute_ic_autocorrelation`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.multiple_testing.benjamini_hochberg_correction(p_values: numpy.ndarray, alpha: float = 0.05) -> Tuple[numpy.ndarray, numpy.ndarray, int]
```

Apply Benjamini-Hochberg FDR correction for multiple testing.

Args:
    p_values: Array of p-values, any shape. Non-finite entries are
        treated as missing tests (see module docstring); finite
        entries must lie in [0, 1].
    alpha: False discovery rate, strictly inside (0, 1)

Returns:
    (adjusted_p_values, reject_mask, n_discoveries)
    adjusted_p_values: BH-adjusted p-values
    reject_mask: Boolean mask where null hypothesis is rejected
    n_discoveries: Number of discoveries (rejections)

Raises:
    ValueError: If ``p_values`` is empty, contains finite values
        outside [0, 1], or ``alpha`` is not strictly inside (0, 1)

### 精确计算公式（实际实现）

```python
def benjamini_hochberg_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.

    Args:
        p_values: Array of p-values, any shape. Non-finite entries are
            treated as missing tests (see module docstring); finite
            entries must lie in [0, 1].
        alpha: False discovery rate, strictly inside (0, 1)

    Returns:
        (adjusted_p_values, reject_mask, n_discoveries)
        adjusted_p_values: BH-adjusted p-values
        reject_mask: Boolean mask where null hypothesis is rejected
        n_discoveries: Number of discoveries (rejections)

    Raises:
        ValueError: If ``p_values`` is empty, contains finite values
            outside [0, 1], or ``alpha`` is not strictly inside (0, 1)
    """
    alpha = _validate_alpha(alpha)
    p_values = _validate_p_values(p_values)

    # Flatten for processing
    original_shape = p_values.shape
    p_flat = p_values.ravel()

    # Extract valid p-values with their indices
    valid_mask = np.isfinite(p_flat)
    valid_indices = np.where(valid_mask)[0]
    p_valid = p_flat[valid_mask]

    n_tests = len(p_valid)

    if n_tests == 0:
        return np.full_like(p_values, np.nan), np.zeros_like(p_values, dtype=bool), 0

    # Sort p-values and track original indices
    sort_idx = np.argsort(p_valid)
    p_sorted = p_valid[sort_idx]
    original_idx = valid_indices[sort_idx]

    # Compute BH critical values
    ranks = np.arange(1, n_tests + 1)
    bh_critical = (ranks / n_tests) * alpha

    # Find largest i where p[i] <= (i/m) * alpha
    comparisons = p_sorted <= bh_critical
    if np.any(comparisons):
        max_idx = np.where(comparisons)[0][-1]
        n_discoveries = max_idx + 1
    else:
        n_discoveries = 0

    # Compute BH adjusted p-values. In sorted order, they must be
    # non-decreasing, so propagate each smaller value toward lower ranks.
    adjusted_sorted = np.minimum.accumulate(
        (p_sorted * n_tests / ranks)[::-1]
    )[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Map back to original positions
    adjusted_flat = np.full_like(p_flat, np.nan)
    adjusted_flat[original_idx] = adjusted_sorted

    # Rejection mask
    reject_flat = np.zeros_like(p_flat, dtype=bool)
    if n_discoveries > 0:
        reject_indices = original_idx[:n_discoveries]
        reject_flat[reject_indices] = True

    return (
        adjusted_flat.reshape(original_shape),
        reject_flat.reshape(original_shape),
        n_discoveries,
    )
```

<a id="metric-beta_exposure"></a>
## beta_exposure — Beta Exposure

Signed mean beta-style exposure of the factor (typed per-style field). NaN when style absent or no finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_beta_exposure`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_beta_exposure(panel: 'FactorLoadingSeries', *, factor_values=None, min_obs=10, weights=None) -> 'float'
```

Signed mean beta-style exposure of the factor (one typed field).

### 精确计算公式（实际实现）

```python
def compute_beta_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean beta-style exposure of the factor (one typed field)."""
    return _select_style(panel, "beta",factor_values=factor_values,min_obs=min_obs,weights=weights)
```

<a id="metric-block_bootstrap_ci"></a>
## block_bootstrap_ci — Block Bootstrap Confidence Interval

95% confidence interval half-width for mean IC via block bootstrap

- 版本：`2.0.0`；状态：`stable`；层级：`research`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`60`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`block_bootstrap_ci`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_block_bootstrap_ci_value(ic_series: numpy.ndarray, min_periods: int = 60, block_length: int = 10, num_bootstrap: int = 1000, confidence_level: float = 0.95, random_seed: int = 0) -> numpy.ndarray
```

Return the block-bootstrap CI half-width per factor.

compute_block_bootstrap_ci returns a (lower, upper) tuple; the registry
exposes the scalar half-width (upper - lower) / 2, NaN below min_periods
(matching the other ic_series adapters' contract).

### 精确计算公式（实际实现）

```python
def compute_block_bootstrap_ci_value(
    ic_series: np.ndarray,
    min_periods: int = 60,
    block_length: int = 10,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> np.ndarray:
    """Return the block-bootstrap CI half-width per factor.

    compute_block_bootstrap_ci returns a (lower, upper) tuple; the registry
    exposes the scalar half-width (upper - lower) / 2, NaN below min_periods
    (matching the other ic_series adapters' contract).
    """
    ci_lower, ci_upper = compute_block_bootstrap_ci(
        ic_series,
        block_length=block_length,
        num_bootstrap=num_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    valid_periods = np.sum(np.isfinite(ic_series), axis=0)
    with np.errstate(invalid="ignore"):
        half_width = (ci_upper - ci_lower) / 2.0
    return np.where(valid_periods >= min_periods, half_width, np.nan)
```

<a id="metric-bonferroni_correction"></a>
## bonferroni_correction — Bonferroni Correction

Bonferroni multiple-testing correction: adjusted p = min(p * n, 1). Controls the family-wise error rate (spec §34).

- 版本：`4.0.0`；状态：`stable`；层级：`research`。
- 输入依赖：`p_values`。
- 输出：`scalar`；单位：`pvalue`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.multiple_testing.bonferroni_correction`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.multiple_testing.bonferroni_correction(p_values: numpy.ndarray, alpha: float = 0.05) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Apply Bonferroni correction for multiple testing.

Args:
    p_values: Array of p-values, any shape. Non-finite entries are
        treated as missing tests (see module docstring); finite
        entries must lie in [0, 1].
    alpha: Family-wise error rate, strictly inside (0, 1)

Returns:
    (adjusted_p_values, reject_mask)
    adjusted_p_values: Bonferroni-adjusted p-values (min(p * n_tests, 1.0))
    reject_mask: Boolean mask where null hypothesis is rejected

Raises:
    ValueError: If ``p_values`` is empty, contains finite values
        outside [0, 1], or ``alpha`` is not strictly inside (0, 1)

### 精确计算公式（实际实现）

```python
def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction for multiple testing.

    Args:
        p_values: Array of p-values, any shape. Non-finite entries are
            treated as missing tests (see module docstring); finite
            entries must lie in [0, 1].
        alpha: Family-wise error rate, strictly inside (0, 1)

    Returns:
        (adjusted_p_values, reject_mask)
        adjusted_p_values: Bonferroni-adjusted p-values (min(p * n_tests, 1.0))
        reject_mask: Boolean mask where null hypothesis is rejected

    Raises:
        ValueError: If ``p_values`` is empty, contains finite values
            outside [0, 1], or ``alpha`` is not strictly inside (0, 1)
    """
    alpha = _validate_alpha(alpha)
    p_values = _validate_p_values(p_values)

    # Flatten for processing
    original_shape = p_values.shape
    p_flat = p_values.ravel()

    # Count valid (non-NaN) p-values
    valid_mask = np.isfinite(p_flat)
    n_tests = np.sum(valid_mask)

    if n_tests == 0:
        return np.full_like(p_values, np.nan), np.zeros_like(p_values, dtype=bool)

    # Adjust p-values
    adjusted = np.full_like(p_flat, np.nan)
    adjusted[valid_mask] = np.minimum(p_flat[valid_mask] * n_tests, 1.0)

    # Rejection mask
    reject = np.zeros_like(p_flat, dtype=bool)
    reject[valid_mask] = adjusted[valid_mask] <= alpha

    return adjusted.reshape(original_shape), reject.reshape(original_shape)
```

<a id="metric-bottom_quantile_cliff"></a>
## bottom_quantile_cliff — Bottom Quantile Cliff

Bottom-quantile cliff: ret[1] - ret[0], per factor. The jump in return from the lowest to the second-lowest quantile (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_bottom_quantile_cliff`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.quantile_shape.compute_bottom_quantile_cliff(qr: 'np.ndarray') -> 'np.ndarray'
```

Bottom-quantile cliff: ret[1] - ret[0], (F,).

### 精确计算公式（实际实现）

```python
def compute_bottom_quantile_cliff(qr: np.ndarray) -> np.ndarray:
    """Bottom-quantile cliff: ret[1] - ret[0], (F,)."""
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 2:
        return out
    for f in range(F):
        col = m[:, f]
        if np.isfinite(col[1]) and np.isfinite(col[0]):
            out[f] = col[1] - col[0]
    return out
```

<a id="metric-bottom_quantile_cliff_robust"></a>
## bottom_quantile_cliff_robust — Bottom Quantile Cliff (Robust)

ROBUST bottom cliff per factor: Q_1 - mean(Q_2..Q_4) (plan §14.4 mirror). Direction higher_is_better (big positive step out of the bottom bucket). NaN when the bottom 4 quantile returns are not all finite.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_bottom_quantile_cliff_robust`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_bottom_quantile_cliff_robust(qr: 'np.ndarray') -> 'np.ndarray'
```

Robust bottom cliff: ``ret[1] - mean(ret[1:4])`` per factor, (F,).

Plan §14.4 symmetric variant: contrast the FIRST bucket against the mean
of the three next buckets.  Direction ``higher_is_better``.  NaN when
the bottom 4 quantile returns are not all finite.

### 精确计算公式（实际实现）

```python
def compute_bottom_quantile_cliff_robust(qr: np.ndarray) -> np.ndarray:
    """Robust bottom cliff: ``ret[1] - mean(ret[1:4])`` per factor, (F,).

    Plan §14.4 symmetric variant: contrast the FIRST bucket against the mean
    of the three next buckets.  Direction ``higher_is_better``.  NaN when
    the bottom 4 quantile returns are not all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 4:
        return out
    for f in range(F):
        col = m[:, f]
        seg = col[:4]
        if not np.all(np.isfinite(seg)):
            continue
        out[f] = seg[1] - float(np.mean(seg[1:4]))
    return out
```

<a id="metric-bottom_tail_slope"></a>
## bottom_tail_slope — Bottom Tail Slope

Mean adjacent return difference over the BOTTOM segment of the quantile profile as drawn (quantiles 0..2), per factor. Direction NEUTRAL (orientation informative, never assumed, plan §14.4).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_bottom_tail_slope`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_bottom_tail_slope(qr: 'np.ndarray') -> 'np.ndarray'
```

Bottom-tail slope per factor, (F,).

Mean adjacent return difference over the BOTTOM segment of the curve as
drawn (quantile 0..2).  For a positively inclined factor this is
POSITIVE (returns rise out of the bottom bucket).  NaN when the first
3 quantile returns are not all finite.

### 精确计算公式（实际实现）

```python
def compute_bottom_tail_slope(qr: np.ndarray) -> np.ndarray:
    """Bottom-tail slope per factor, (F,).

    Mean adjacent return difference over the BOTTOM segment of the curve as
    drawn (quantile 0..2).  For a positively inclined factor this is
    POSITIVE (returns rise out of the bottom bucket).  NaN when the first
    3 quantile returns are not all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 3:
        return out
    for f in range(F):
        out[f] = _bottom_tail_slope_value(m[:, f], n_adj=2)
    return out
```

<a id="metric-calmar_ratio"></a>
## calmar_ratio — calmar_ratio

Calmar ratio: explicit arithmetic (legacy default) or CAGR annualization / max drawdown

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_calmar_ratio`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.portfolio_stats.compute_calmar_ratio(returns: numpy.ndarray, periods_per_year: int = 252, min_periods: int = 20, annualization: str = 'cagr', missing_return_policy: str = 'unknown') -> numpy.ndarray
```

Compute Calmar ratio with explicit arithmetic (legacy) or CAGR numerator.

Args:
    returns: Return series (T,) or (T, F)
    periods_per_year: Number of periods per year
    min_periods: Minimum periods required

Returns:
    Calmar ratio, scalar or shape (F,)

### 精确计算公式（实际实现）

```python
def compute_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    min_periods: int = 20,
    annualization: str = "cagr",
    missing_return_policy: str = "unknown",
) -> np.ndarray:
    """
    Compute Calmar ratio with explicit arithmetic (legacy) or CAGR numerator.

    Args:
        returns: Return series (T,) or (T, F)
        periods_per_year: Number of periods per year
        min_periods: Minimum periods required

    Returns:
        Calmar ratio, scalar or shape (F,)
    """
    if annualization not in {"arithmetic", "cagr"}:
        raise ValueError("annualization must be arithmetic or cagr")
    if missing_return_policy not in {"unknown", "zero_fill", "fail"}:
        raise ValueError("invalid missing_return_policy")
    if missing_return_policy == "fail" and np.any(~np.isfinite(returns)):
        raise ValueError("nonfinite returns with missing_return_policy='fail'")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive finite")
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    calmar = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        if missing_return_policy == "unknown" and n_valid != T:
            continue
        if missing_return_policy == "zero_fill":
            ret_valid = np.where(valid, ret_f, 0.0)
            n_valid = T

        # Annualized return
        mean_ret = np.mean(ret_valid)
        ann_ret = mean_ret * periods_per_year
        if annualization == "cagr":
            ann_ret = np.prod(1.0 + ret_valid) ** (periods_per_year / n_valid) - 1.0

        # Maximum drawdown
        max_dd, _, _ = compute_maximum_drawdown(ret_valid, missing_return_policy=missing_return_policy)

        if not np.isfinite(max_dd) or max_dd <= 1e-12:
            continue

        calmar[f] = ann_ret / max_dd

    return calmar[0] if squeeze else calmar
```

<a id="metric-change_point_score"></a>
## change_point_score — Change-Point Score

CUSUM-based change-point score: max |cumulative deviation from the mean IC|, per factor. A large score indicates a structural break in the IC level (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_change_point_score`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_change_point_score(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

CUSUM-based change-point score: max |cumulative deviation|, (F,).

A large score indicates a structural break in the IC level.

### 精确计算公式（实际实现）

```python
def compute_change_point_score(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """CUSUM-based change-point score: max |cumulative deviation|, (F,).

    A large score indicates a structural break in the IC level.
    """
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(s, axis=0)
    dev = np.where(valid, s - mean[None, :], 0.0)
    cum = np.cumsum(dev, axis=0)
    score = np.max(np.abs(cum), axis=0)
    return np.where(n >= min_periods, score, np.nan)
```

<a id="metric-coverage"></a>
## coverage — Coverage Rate

Per-factor fraction of the (T, N) panel with jointly valid factor and label values (never averaged across factor columns; canonical family: coverage)

- 版本：`0.1.0`；状态：`stable`；层级：`core`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`coverage`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_coverage_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, label_bundle: quant_evaluator.contracts.label_bundle.LabelBundle, min_assets: int = 10) -> numpy.ndarray
```

Return the coverage fraction per factor, shape (F,).

Uses :func:`quant_evaluator.metrics.quality.compute_coverage_per_factor`
(the single truth for coverage semantics): no aggregation across factor
columns, and ``min_assets`` is honoured for day-level diagnostics.

### 精确计算公式（实际实现）

```python
def compute_coverage_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> np.ndarray:
    """Return the coverage fraction per factor, shape (F,).

    Uses :func:`quant_evaluator.metrics.quality.compute_coverage_per_factor`
    (the single truth for coverage semantics): no aggregation across factor
    columns, and ``min_assets`` is honoured for day-level diagnostics.
    """
    report = compute_coverage_per_factor(factor_batch, label_bundle, min_assets=min_assets)
    return np.asarray(
        [report[factor_id]["coverage"] for factor_id in factor_batch.factor_ids],
        dtype=np.float64,
    )
```

<a id="metric-coverage_stability"></a>
## coverage_stability — coverage_stability

Variance of factor coverage across periods

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`variance`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quality.compute_per_time_coverage`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_cross_section_cardinality(factor_batch: 'FactorBatch') -> 'np.ndarray'
```

Mean number of distinct values per day (cross-section cardinality), (F,).

### 精确计算公式（实际实现）

```python
def compute_cross_section_cardinality(factor_batch: FactorBatch) -> np.ndarray:
    """Mean number of distinct values per day (cross-section cardinality), (F,)."""
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    out = np.full(F, np.nan)
    for f in range(F):
        counts = []
        for t in range(T):
            row = values[t, :, f]
            finite = row[np.isfinite(row)]
            if finite.size == 0:
                continue
            counts.append(np.unique(finite).size)
        if counts:
            out[f] = float(np.mean(counts))
    return out
```

<a id="metric-cusum_break_score"></a>
## cusum_break_score — CUSUM Break Score

CUSUM break score: max |cumulative deviation| normalised by std*sqrt(T), per factor. A standardised change-point statistic; larger values flag a stronger break (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`zscore`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_cusum_break_score`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_cusum_break_score(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

CUSUM break score: max cumulative deviation normalised by std, (F,).

``max|cumsum(ic - mean)| / (std * sqrt(T))`` — a standardised change-point
statistic.  Larger values flag a stronger break.

### 精确计算公式（实际实现）

```python
def compute_cusum_break_score(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """CUSUM break score: max cumulative deviation normalised by std, (F,).

    ``max|cumsum(ic - mean)| / (std * sqrt(T))`` — a standardised change-point
    statistic.  Larger values flag a stronger break.
    """
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(s, axis=0)
        std = np.nanstd(s, axis=0, ddof=1)
    dev = np.where(valid, s - mean[None, :], 0.0)
    cum = np.cumsum(dev, axis=0)
    score = np.max(np.abs(cum), axis=0) / np.maximum(std * np.sqrt(n), 1e-12)
    score = np.where(std > 1e-12, score, np.nan)
    return np.where(n >= min_periods, score, np.nan)
```

<a id="metric-cvar_95"></a>
## cvar_95 — cvar_95

Conditional Value at Risk (Expected Shortfall) at 95%

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.risk.var_cvar.compute_cvar`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_cvar_expected_shortfall(returns: 'np.ndarray', confidence_level: 'float' = 0.95, min_periods: 'int' = 20) -> 'float'
```

Historical CVaR / expected shortfall at ``confidence_level`` (95%).

Reuses the existing ``risk/var_cvar.compute_cvar`` historical method; a
wrapper so the evidence output has its own registered id while the
numeric semantics stay the single-source policy of ``var_cvar.py``.
Returns positive loss magnitude; 0.0 when no returns fall at/below the
VaR threshold; NaN when insufficient observations.

### 精确计算公式（实际实现）

```python
def compute_cvar_expected_shortfall(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    min_periods: int = 20,
) -> float:
    """Historical CVaR / expected shortfall at ``confidence_level`` (95%).

    Reuses the existing ``risk/var_cvar.compute_cvar`` historical method; a
    wrapper so the evidence output has its own registered id while the
    numeric semantics stay the single-source policy of ``var_cvar.py``.
    Returns positive loss magnitude; 0.0 when no returns fall at/below the
    VaR threshold; NaN when insufficient observations.
    """
    from quant_evaluator.metrics.risk.var_cvar import compute_cvar

    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    val = compute_cvar(
        ret,
        confidence_level=confidence_level,
        method="historical",
        min_periods=min_periods,
    )
    return float(val)
```

<a id="metric-daily_quantile_monotonicity_rate"></a>
## daily_quantile_monotonicity_rate — Daily Quantile Monotonicity Rate

Mean of valid per-date increasing-adjacent-pair fractions; missing pairs and dates are excluded from declared denominators. Distinct from quantile_monotonicity.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.registry_adapters.compute_daily_quantile_monotonicity_rate_value`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_daily_quantile_monotonicity_rate_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, label_bundle: quant_evaluator.contracts.label_bundle.LabelBundle, n_quantiles: int = 5, min_assets: int = 10, min_periods: int = 20) -> numpy.ndarray
```

Mean valid daily-profile monotonicity fraction per factor.

### 精确计算公式（实际实现）

```python
def compute_daily_quantile_monotonicity_rate_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
    min_periods: int = 20,
) -> np.ndarray:
    """Mean valid daily-profile monotonicity fraction per factor."""
    series = compute_daily_quantile_monotonicity_series_value(
        factor_batch, label_bundle, n_quantiles=n_quantiles, min_assets=min_assets)
    counts = np.isfinite(series).sum(axis=0)
    means = np.nansum(series, axis=0) / np.maximum(counts, 1)
    return np.where(counts >= min_periods, means, np.nan)
```

<a id="metric-daily_quantile_monotonicity_series"></a>
## daily_quantile_monotonicity_series — Daily Quantile Monotonicity Series

Per-date fraction of increasing finite adjacent quantile-return pairs. This is daily-profile evidence, never the formal long-run mean-profile score.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`series`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`1`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.registry_adapters.compute_daily_quantile_monotonicity_series_value`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_daily_quantile_monotonicity_series_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, label_bundle: quant_evaluator.contracts.label_bundle.LabelBundle, n_quantiles: int = 5, min_assets: int = 10) -> numpy.ndarray
```

Per-date increasing-adjacent-pair fraction, shape ``(T,F)``.

This is deliberately distinct from ``quantile_monotonicity``, which is
computed once on the long-run mean profile. Non-finite adjacent pairs do
not enter a date's denominator; a date with no valid pair is NaN.

### 精确计算公式（实际实现）

```python
def compute_daily_quantile_monotonicity_series_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
) -> np.ndarray:
    """Per-date increasing-adjacent-pair fraction, shape ``(T,F)``.

    This is deliberately distinct from ``quantile_monotonicity``, which is
    computed once on the long-run mean profile. Non-finite adjacent pairs do
    not enter a date's denominator; a date with no valid pair is NaN.
    """
    daily, _ = compute_quantile_returns_fast(
        factor_batch, label_bundle, n_quantiles=n_quantiles, min_assets=min_assets)
    pairs = np.isfinite(daily[:, :-1, :]) & np.isfinite(daily[:, 1:, :])
    denominator = pairs.sum(axis=1)
    increasing = ((daily[:, 1:, :] > daily[:, :-1, :]) & pairs).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = increasing / denominator
    return np.where(denominator > 0, values, np.nan)
```

<a id="metric-distinct_level_ratio"></a>
## distinct_level_ratio — Distinct Level Ratio

Mean fraction of distinct values among finite factor values per day, per factor. A low ratio indicates heavy duplication / coarse factor values (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_distinct_level_ratio`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_distinct_level_ratio(factor_batch: 'FactorBatch') -> 'np.ndarray'
```

Mean fraction of distinct values among finite factor values per day, (F,).

A low ratio indicates heavy duplication / coarse factor values.

### 精确计算公式（实际实现）

```python
def compute_distinct_level_ratio(factor_batch: FactorBatch) -> np.ndarray:
    """Mean fraction of distinct values among finite factor values per day, (F,).

    A low ratio indicates heavy duplication / coarse factor values.
    """
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    out = np.full(F, np.nan)
    for f in range(F):
        ratios = []
        for t in range(T):
            row = values[t, :, f]
            finite = row[np.isfinite(row)]
            if finite.size == 0:
                continue
            ratios.append(np.unique(finite).size / finite.size)
        if ratios:
            out[f] = float(np.mean(ratios))
    return out
```

<a id="metric-downside_deviation"></a>
## downside_deviation — Downside Deviation

Annualized downside deviation (RMS of negative excess returns) of the probe daily PnL series, matching the portfolio_stats.sortino semantics. NaN when insufficient data or no negative excess returns.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`return`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_downside_deviation`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_downside_deviation(returns: 'np.ndarray', risk_free_rate: 'float' = 0.0, periods_per_year: 'int' = 252, min_periods: 'int' = 20) -> 'float'
```

Annualized downside deviation (semicovariance) of the return series.

Definition matches the existing ``portfolio_stats.compute_sortino_ratio``
family: RMS of *negative excess* returns only (no ddof), annualized by
sqrt(periods_per_year).  NaN when there are fewer than ``min_periods``
finite returns or no negative excess returns.

### 精确计算公式（实际实现）

```python
def compute_downside_deviation(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _PERIODS_PER_YEAR,
    min_periods: int = 20,
) -> float:
    """Annualized downside deviation (semicovariance) of the return series.

    Definition matches the existing ``portfolio_stats.compute_sortino_ratio``
    family: RMS of *negative excess* returns only (no ddof), annualized by
    sqrt(periods_per_year).  NaN when there are fewer than ``min_periods``
    finite returns or no negative excess returns.
    """
    ret = _as_1d(returns)
    ret = ret[np.isfinite(ret)]
    if ret.size < min_periods:
        return np.nan
    rf_per = risk_free_rate / periods_per_year
    excess = ret - rf_per
    downside = excess[excess < 0.0]
    if downside.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year))
```

<a id="metric-drawdown_duration"></a>
## drawdown_duration — drawdown_duration

Duration (in periods) of the longest drawdown

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`periods`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.risk.drawdown_analysis.compute_drawdown_duration`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_effective_n(factor_batch: 'FactorBatch') -> 'np.ndarray'
```

Mean number of finite factor values per day, (F,).

### 精确计算公式（实际实现）

```python
def compute_effective_n(factor_batch: FactorBatch) -> np.ndarray:
    """Mean number of finite factor values per day, (F,)."""
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    n = np.sum(np.isfinite(values), axis=1)  # (T, F)
    return np.mean(n, axis=0)
```

<a id="metric-exposure_drift"></a>
## exposure_drift — Exposure Drift

Mean absolute change of the per-style exposure panel between adjacent periods (persistence / stability measure). NaN when fewer than 2 periods or no finite adjacent pair.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`2`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_exposure_drift`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_exposure_drift(panel: 'FactorLoadingSeries', *, factor_values=None, min_obs=10, weights=None) -> 'float'
```

Mean absolute change of the per-style exposure series between adjacent
periods (a persistence / stability measure).

``drift = mean_{t,k} |panel[t+1,k] - panel[t,k]|`` over jointly-finite
adjacent cells.  NaN when the panel has fewer than 2 periods or no finite
adjacent pair.

### 精确计算公式（实际实现）

```python
def compute_exposure_drift(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Mean absolute change of the per-style exposure series between adjacent
    periods (a persistence / stability measure).

    ``drift = mean_{t,k} |panel[t+1,k] - panel[t,k]|`` over jointly-finite
    adjacent cells.  NaN when the panel has fewer than 2 periods or no finite
    adjacent pair.
    """
    panel=_as_factor_loadings(panel,factor_values,min_obs,weights)
    arr=panel.values
    T,K=arr.shape
    if T < 2:
        return np.nan
    diffs: list[float] = []
    for t in range(T - 1):
        a = arr[t]
        b = arr[t + 1]
        joint = np.isfinite(a) & np.isfinite(b)
        if not np.any(joint):
            continue
        diffs.append(float(np.mean(np.abs(a[joint] - b[joint]))))
    if not diffs:
        return np.nan
    return float(np.mean(diffs))
```

<a id="metric-factor_coverage"></a>
## factor_coverage — factor_coverage

Fraction of universe with non-null factor values

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quality.compute_coverage`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_factor_turnover_rate_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, min_periods: int = 30, quantile: float = 0.9) -> numpy.ndarray
```

Return mean top-quantile membership turnover per factor.

compute_factor_turnover_rate returns a (T-1, F) series; the registry
exposes the time-averaged scalar per factor, NaN below min_periods.

### 精确计算公式（实际实现）

```python
def compute_factor_turnover_rate_value(
    factor_batch: FactorBatch,
    min_periods: int = 30,
    quantile: float = 0.9,
) -> np.ndarray:
    """Return mean top-quantile membership turnover per factor.

    compute_factor_turnover_rate returns a (T-1, F) series; the registry
    exposes the time-averaged scalar per factor, NaN below min_periods.
    """
    values = np.asarray(factor_batch.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("factor_batch.values must be (T, N, F)")
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    turnover_series = compute_factor_turnover_rate(values, quantile=quantile)
    if turnover_series.size == 0:
        return np.full(values.shape[2], np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(turnover_series, axis=0)
    valid_counts = np.sum(np.isfinite(turnover_series), axis=0)
    return np.where(valid_counts >= min_periods, means, np.nan)
```

<a id="metric-hac_pvalue"></a>
## hac_pvalue — HAC p-value

Two-sided HAC-robust p-value for mean(IC) != 0 per factor (canonical alias ic.rank.hac_p)

- 版本：`4.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`30`。
- 别名： `ic.rank.hac_p` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`hac_pvalue`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_hac_pvalue_value(ic_series: numpy.ndarray, min_periods: int = 30, max_lag: int = 5, kernel: str = 'bartlett') -> numpy.ndarray
```

Return the two-sided HAC p-value for mean(IC) != 0 per factor.

Uses the HAC t-statistic with a Gaussian null approximation; NaN below
min_periods (matching the hac_tstat adapter contract).

### 精确计算公式（实际实现）

```python
def compute_hac_pvalue_value(
    ic_series: np.ndarray,
    min_periods: int = 30,
    max_lag: int = 5,
    kernel: str = "bartlett",
) -> np.ndarray:
    """Return the two-sided HAC p-value for mean(IC) != 0 per factor.

    Uses the HAC t-statistic with a Gaussian null approximation; NaN below
    min_periods (matching the hac_tstat adapter contract).
    """
    from scipy import stats as _stats

    t_stat, _ = compute_hac_tstat(ic_series, max_lag=max_lag, kernel=kernel)
    valid_periods = np.sum(np.isfinite(ic_series), axis=0)
    with np.errstate(invalid="ignore"):
        p_values = 2.0 * _stats.norm.sf(np.abs(t_stat))
    p_values = np.where(np.isfinite(p_values), p_values, np.nan)
    return np.where(valid_periods >= min_periods, p_values, np.nan)
```

<a id="metric-hac_tstat"></a>
## hac_tstat — HAC t-statistic

Heteroskedasticity and autocorrelation consistent t-statistic for IC

- 版本：`4.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`30`。
- 别名： `ic.rank.hac_t` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`hac_tstat`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_hac_tstat_value(ic_series: numpy.ndarray, min_periods: int = 30, max_lag: int = 5, kernel: str = 'bartlett') -> numpy.ndarray
```

Return only the HAC t-statistic component per factor.

### 精确计算公式（实际实现）

```python
def compute_hac_tstat_value(
    ic_series: np.ndarray,
    min_periods: int = 30,
    max_lag: int = 5,
    kernel: str = "bartlett",
) -> np.ndarray:
    """Return only the HAC t-statistic component per factor."""
    t_stat, _ = compute_hac_tstat(ic_series, max_lag=max_lag, kernel=kernel)
    valid_periods = np.sum(np.isfinite(ic_series), axis=0)
    return np.where(valid_periods >= min_periods, t_stat, np.nan)
```

<a id="metric-half_life"></a>
## half_life — IC Temporal Persistence Half-Life

Centered AR(1) IC temporal persistence; not predictive horizon decay

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`60`。
- 别名： `ic_temporal_persistence` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`half_life`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_half_life_value(ic_series: numpy.ndarray, min_periods: int = 60) -> numpy.ndarray
```

Return estimated IC half-life per factor.

### 精确计算公式（实际实现）

```python
def compute_half_life_value(
    ic_series: np.ndarray,
    min_periods: int = 60,
) -> np.ndarray:
    """Return estimated IC half-life per factor."""
    return compute_half_life(ic_series, min_periods=min_periods)
```

<a id="metric-hhi_concentration"></a>
## hhi_concentration — hhi_concentration

Herfindahl-Hirschman Index of factor value concentration

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`dimensionless`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure.compute_concentration_hhi`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.multiple_testing.holm_bonferroni_correction(p_values: numpy.ndarray, alpha: float = 0.05) -> Tuple[numpy.ndarray, numpy.ndarray, int]
```

Apply Holm-Bonferroni step-down correction.

More powerful than Bonferroni, controls family-wise error rate.

Args:
    p_values: Array of p-values, any shape. Non-finite entries are
        treated as missing tests (see module docstring); finite
        entries must lie in [0, 1].
    alpha: Family-wise error rate, strictly inside (0, 1)

Returns:
    (adjusted_p_values, reject_mask, n_discoveries)
    adjusted_p_values: Holm-adjusted p-values
    reject_mask: Boolean mask where null hypothesis is rejected
    n_discoveries: Number of discoveries (rejections)

Raises:
    ValueError: If ``p_values`` is empty, contains finite values
        outside [0, 1], or ``alpha`` is not strictly inside (0, 1)

### 精确计算公式（实际实现）

```python
def holm_bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Apply Holm-Bonferroni step-down correction.

    More powerful than Bonferroni, controls family-wise error rate.

    Args:
        p_values: Array of p-values, any shape. Non-finite entries are
            treated as missing tests (see module docstring); finite
            entries must lie in [0, 1].
        alpha: Family-wise error rate, strictly inside (0, 1)

    Returns:
        (adjusted_p_values, reject_mask, n_discoveries)
        adjusted_p_values: Holm-adjusted p-values
        reject_mask: Boolean mask where null hypothesis is rejected
        n_discoveries: Number of discoveries (rejections)

    Raises:
        ValueError: If ``p_values`` is empty, contains finite values
            outside [0, 1], or ``alpha`` is not strictly inside (0, 1)
    """
    alpha = _validate_alpha(alpha)
    p_values = _validate_p_values(p_values)

    # Flatten for processing
    original_shape = p_values.shape
    p_flat = p_values.ravel()

    # Extract valid p-values
    valid_mask = np.isfinite(p_flat)
    valid_indices = np.where(valid_mask)[0]
    p_valid = p_flat[valid_mask]

    n_tests = len(p_valid)

    if n_tests == 0:
        return np.full_like(p_values, np.nan), np.zeros_like(p_values, dtype=bool), 0

    # Sort p-values
    sort_idx = np.argsort(p_valid)
    p_sorted = p_valid[sort_idx]
    original_idx = valid_indices[sort_idx]

    # Compute Holm critical values (step-down)
    ranks = np.arange(1, n_tests + 1)
    holm_critical = alpha / (n_tests - ranks + 1)

    # Find rejections (sequential)
    n_discoveries = 0
    for i in range(n_tests):
        if p_sorted[i] <= holm_critical[i]:
            n_discoveries = i + 1
        else:
            break

    # Compute Holm adjusted p-values. In sorted order, adjusted values must
    # be non-decreasing, so each rank inherits the largest prior value.
    adjusted_sorted = np.minimum(
        np.maximum.accumulate(p_sorted * (n_tests - ranks + 1)),
        1.0,
    )

    # Map back to original positions
    adjusted_flat = np.full_like(p_flat, np.nan)
    adjusted_flat[original_idx] = adjusted_sorted

    # Rejection mask
    reject_flat = np.zeros_like(p_flat, dtype=bool)
    if n_discoveries > 0:
        reject_indices = original_idx[:n_discoveries]
        reject_flat[reject_indices] = True

    return (
        adjusted_flat.reshape(original_shape),
        reject_flat.reshape(original_shape),
        n_discoveries,
    )
```

<a id="metric-ic_autocorr_lag1"></a>
## ic_autocorr_lag1 — IC Autocorrelation (Lag 1)

First-order autocorrelation of IC series

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`30`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`ic_autocorr_lag1`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_ic_autocorr_lag1_value(ic_series: numpy.ndarray, min_periods: int = 30, max_lag: int = 20) -> numpy.ndarray
```

Return lag-one IC autocorrelation per factor.

### 精确计算公式（实际实现）

```python
def compute_ic_autocorr_lag1_value(
    ic_series: np.ndarray,
    min_periods: int = 30,
    max_lag: int = 20,
) -> np.ndarray:
    """Return lag-one IC autocorrelation per factor."""
    acf = compute_ic_autocorrelation(
        ic_series, max_lag=max_lag, min_obs=min_periods
    )
    return acf[1] if acf.shape[0] > 1 else np.full(ic_series.shape[1], np.nan)
```

<a id="metric-ic_decay"></a>
## ic_decay — ic_decay

IC decay: correlation at increasing forward horizons

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic_summary.compute_ic_decay`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_ic_ir_value(ic_series: numpy.ndarray, min_periods: int = 20) -> numpy.ndarray
```

Return the scalar IC information ratio per factor.

### 精确计算公式（实际实现）

```python
def compute_ic_ir_value(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> np.ndarray:
    """Return the scalar IC information ratio per factor."""
    return compute_icir(ic_series, min_periods=min_periods)
```

<a id="metric-ic_median"></a>
## ic_median — Median IC

Time-median of the daily IC series per factor (canonical alias ic.rank.median)

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名： `ic.rank.median` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`ic_median`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_ic_median_value(ic_series: numpy.ndarray, min_periods: int = 20) -> numpy.ndarray
```

Return the time-median IC per factor, shape (F,).

### 精确计算公式（实际实现）

```python
def compute_ic_median_value(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> np.ndarray:
    """Return the time-median IC per factor, shape (F,)."""
    import warnings

    valid_periods = np.sum(np.isfinite(ic_series), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            medians = np.nanmedian(ic_series, axis=0)
    medians = np.where(np.isfinite(medians), medians, np.nan)
    return np.where(valid_periods >= min_periods, medians, np.nan)
```

<a id="metric-ic_positive_ratio"></a>
## ic_positive_ratio — IC Positive Ratio

Fraction of finite daily IC values that are strictly positive, per factor. A value near 1.0 indicates the factor's IC is consistently positive (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_ic_positive_ratio`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_ic_positive_ratio(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

Fraction of finite daily IC values that are strictly positive, (F,).

### 精确计算公式（实际实现）

```python
def compute_ic_positive_ratio(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Fraction of finite daily IC values that are strictly positive, (F,)."""
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    pos = np.sum((s > 0) & valid, axis=0)
    ratio = pos / np.maximum(n, 1)
    return np.where(n >= min_periods, ratio, np.nan)
```

<a id="metric-ic_recent_vs_history_delta"></a>
## ic_recent_vs_history_delta — IC Recent vs History Delta

(recent mean IC - full-history mean IC) / full-history std, per factor. Positive means recent IC is stronger than the historical average; a strongly negative value flags recent degradation (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`zscore`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_ic_recent_vs_history_delta`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_ic_recent_vs_history_delta(ic_series: 'np.ndarray', recent_days: 'int' = 63, min_periods: 'int' = 20) -> 'np.ndarray'
```

(recent mean IC - full-history mean IC) / full-history std, (F,).

Positive means recent IC is stronger than the historical average; a
strongly negative value flags recent degradation.

### 精确计算公式（实际实现）

```python
def compute_ic_recent_vs_history_delta(
    ic_series: np.ndarray,
    recent_days: int = 63,
    min_periods: int = 20,
) -> np.ndarray:
    """(recent mean IC - full-history mean IC) / full-history std, (F,).

    Positive means recent IC is stronger than the historical average; a
    strongly negative value flags recent degradation.
    """
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore"):
        hist_mean = np.nanmean(s, axis=0)
        hist_std = np.nanstd(s, axis=0, ddof=1)
        recent_mean = _recent_mean(s, recent_days)
    delta = (recent_mean - hist_mean) / np.maximum(hist_std, 1e-12)
    delta = np.where(hist_std > 1e-12, delta, np.nan)
    return np.where(n >= min_periods, delta, np.nan)
```

<a id="metric-ic_serial_autocorrelation_lags_1_5_10_20"></a>
## ic_serial_autocorrelation_lags_1_5_10_20 — IC Serial Autocorrelation (lags 1/5/10/20)

Mean serial autocorrelation of one IC series; not predictive IC across label horizons

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_rank_ic_decay`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_rank_ic_decay(ic_series: 'np.ndarray', horizons=(1, 5, 10, 20), min_periods: 'int' = 20) -> 'np.ndarray'
```

Mean IC autocorrelation across the decay horizons {1,5,10,20}, (F,).

A single scalar per factor summarising how quickly the IC series loses
autocorrelation (decays) at increasing lags.

### 精确计算公式（实际实现）

```python
def compute_rank_ic_decay(
    ic_series: np.ndarray,
    horizons=(1, 5, 10, 20),
    min_periods: int = 20,
) -> np.ndarray:
    """Mean IC autocorrelation across the decay horizons {1,5,10,20}, (F,).

    A single scalar per factor summarising how quickly the IC series loses
    autocorrelation (decays) at increasing lags.
    """
    s = _as_series(ic_series)
    acfs = [_autocorr_lag(s, h) for h in horizons]
    with np.errstate(invalid="ignore"):
        out = np.nanmean(np.stack(acfs, axis=0), axis=0)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-ic_sign_consistency"></a>
## ic_sign_consistency — IC Sign Consistency

Fraction of finite daily IC values sharing the sign of the mean IC, per factor. A value near 1.0 indicates the factor's IC sign is highly consistent (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_ic_sign_consistency`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_ic_sign_consistency(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

Fraction of finite daily IC values sharing the sign of the mean IC, (F,).

### 精确计算公式（实际实现）

```python
def compute_ic_sign_consistency(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Fraction of finite daily IC values sharing the sign of the mean IC, (F,)."""
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(s, axis=0)
    sign = np.sign(mean)
    same = np.sum((np.sign(s) == sign[None, :]) & valid, axis=0)
    ratio = same / np.maximum(n, 1)
    return np.where(n >= min_periods, ratio, np.nan)
```

<a id="metric-ic_sign_flip_rate"></a>
## ic_sign_flip_rate — IC Sign Flip Rate

Fraction of adjacent finite IC pairs whose sign flips, per factor. Lower values indicate a more persistent (stable) IC sign (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_ic_sign_flip_rate`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_ic_sign_flip_rate(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

Fraction of adjacent finite IC pairs whose sign flips, (F,).

Lower values indicate a more persistent (stable) IC sign.

### 精确计算公式（实际实现）

```python
def compute_ic_sign_flip_rate(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Fraction of adjacent finite IC pairs whose sign flips, (F,).

    Lower values indicate a more persistent (stable) IC sign.
    """
    s = _as_series(ic_series)
    finite = np.isfinite(s)
    pair = finite[1:, :] & finite[:-1, :]
    n = np.sum(pair, axis=0)
    flip = np.sum((np.sign(s[1:, :]) != np.sign(s[:-1, :])) & pair, axis=0)
    rate = flip / np.maximum(n, 1)
    return np.where(n >= min_periods - 1, rate, np.nan)
```

<a id="metric-ic_stability"></a>
## ic_stability — ic_stability

Rolling correlation of IC values across sub-periods

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic_summary.compute_ic_stability`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.ic.compute_ic_std(ic_series: numpy.ndarray, valid_counts: Optional[numpy.ndarray] = None, min_periods: int = 20) -> numpy.ndarray
```

Compute only the IC standard-deviation component.

### 精确计算公式（实际实现）

```python
def compute_ic_std(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> np.ndarray:
    """Compute only the IC standard-deviation component."""
    _, ic_std = compute_mean_ic(
        ic_series,
        valid_counts=valid_counts,
        min_periods=min_periods,
    )
    return ic_std
```

<a id="metric-ic_summary"></a>
## ic_summary — ic_summary

Summary statistics (mean, std, skew, kurtosis) of IC time series

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`distribution`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic_summary.compute_rolling_ic_stats`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_industry_exposure(panel: 'FactorLoadingSeries', *, factor_values=None, min_obs=10, weights=None) -> 'float'
```

Signed mean industry-style exposure of the factor (one typed field).

### 精确计算公式（实际实现）

```python
def compute_industry_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean industry-style exposure of the factor (one typed field)."""
    return _select_style(panel, "industry",factor_values=factor_values,min_obs=min_obs,weights=weights)
```

<a id="metric-information_ratio"></a>
## information_ratio — Information Ratio

Benchmark/invested-capital evidence from an explicitly bound execution trajectory leg

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`annualized_ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.long_only.compute_information_ratio`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.long_only.compute_information_ratio(returns, periods_per_year: 'int' = 252, min_periods: 'int' = 2)
```

Annualized mean(active)/sample-std(active); zero risk is undefined.

### 精确计算公式（实际实现）

```python
def compute_information_ratio(returns, periods_per_year: int = 252, min_periods: int = 2):
    """Annualized mean(active)/sample-std(active); zero risk is undefined."""
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 2:
        raise ValueError("min_periods must be an integer >= 2")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    x, squeeze = _matrix(returns)
    out = np.full(x.shape[1], np.nan)
    for f in range(x.shape[1]):
        v = x[np.isfinite(x[:, f]), f]
        if len(v) >= min_periods:
            scale = np.std(v, ddof=1)
            if np.isfinite(scale) and scale > 0:
                out[f] = np.mean(v) / scale * np.sqrt(periods_per_year)
    return out[0] if squeeze else out
```

<a id="metric-inverted_u_score"></a>
## inverted_u_score — Inverted-U Shape Score

Inverted-U (hill) score in [0, 1] per factor. Mirror of u_shape_score with concave curvature and an inverted-U template; template fit must exceed the monotone fit.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`score`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_inverted_u_score`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_inverted_u_score(qr: 'np.ndarray') -> 'np.ndarray'
```

Inverted-U score per factor, (F,).

Mirrors :func:`compute_u_shape_score` with the opposite curvature
(concave: interior second differences NEGATIVE) and the opposite
template (``-U(x)`` inverted-U template, concave).  NaN when fewer than
4 finite quantile returns.

NOTE on template fitting with a free scale: fitting ``a + b*T`` to ``y``
with ``T = -(x-0.5)^2`` also fits ``-T`` perfectly when ``b`` is free
(``b`` flips sign), so an inverted-U TEMPLATE fit alone cannot separate
U from inverted-U — the concave-curvature term is the separator.  A pure
U profile therefore reports a non-zero inverted-U score only from the
template term; the concave fraction keeps it below the U score.

### 精确计算公式（实际实现）

```python
def compute_inverted_u_score(qr: np.ndarray) -> np.ndarray:
    """Inverted-U score per factor, (F,).

    Mirrors :func:`compute_u_shape_score` with the opposite curvature
    (concave: interior second differences NEGATIVE) and the opposite
    template (``-U(x)`` inverted-U template, concave).  NaN when fewer than
    4 finite quantile returns.

    NOTE on template fitting with a free scale: fitting ``a + b*T`` to ``y``
    with ``T = -(x-0.5)^2`` also fits ``-T`` perfectly when ``b`` is free
    (``b`` flips sign), so an inverted-U TEMPLATE fit alone cannot separate
    U from inverted-U — the concave-curvature term is the separator.  A pure
    U profile therefore reports a non-zero inverted-U score only from the
    template term; the concave fraction keeps it below the U score.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 4:
        return out
    x = _template_fit_x(nq)
    it = -(x - 0.5) ** 2
    for f in range(F):
        col = m[:, f]
        finite = np.isfinite(col)
        if np.sum(finite) < 4:
            continue
        y = col[finite]
        xi = it[finite]
        xm = x[finite]
        if np.ptp(y) == 0.0:
            out[f] = 0.0
            continue
        r2_i = _fit_variance_explained(y, xi)
        r2_m = _fit_variance_explained(y, xm)
        if not (np.isfinite(r2_i) and np.isfinite(r2_m)):
            continue
        interior = finite[1:-1] & finite[:-2] & finite[2:]
        if not np.any(interior):
            continue
        d2 = col[2:][interior] - 2.0 * col[1:-1][interior] + col[:-2][interior]
        curv = float(np.mean(d2 < 0.0))
        # A U profile is convex; without concave curvature the inverted-U
        # label must not win even though the sign-flipped template fits.
        if curv < 0.5:
            out[f] = 0.0
            continue
        if r2_i <= r2_m:
            out[f] = 0.0
            continue
        inv_fit = max(r2_i, 0.0)
        out[f] = float(0.5 * inv_fit + 0.3 * curv + 0.2 * (inv_fit - r2_m))
    return out
```

<a id="metric-joint_coverage"></a>
## joint_coverage — joint_coverage

Fraction of universe with both factor and return available

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quality.compute_coverage_per_factor`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_label_maturity(factor_batch: 'FactorBatch', label_bundle: 'LabelBundle') -> 'np.ndarray'
```

Fraction of (T, N) cells with a finite forward-return label, (F,).

Measures how much of the factor's universe has a mature (available)
label for evaluation.

### 精确计算公式（实际实现）

```python
def compute_label_maturity(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> np.ndarray:
    """Fraction of (T, N) cells with a finite forward-return label, (F,).

    Measures how much of the factor's universe has a mature (available)
    label for evaluation.
    """
    values = _factor_values(factor_batch)
    labels = _labels(label_bundle, factor_batch.num_assets)
    T, N, F = values.shape
    label_finite = np.isfinite(labels)  # (T, N)
    factor_finite = np.isfinite(values)  # (T, N, F)
    mature = factor_finite & label_finite[:, :, None]
    return np.sum(mature, axis=(0, 1)) / (T * N)
```

<a id="metric-left_right_asymmetry"></a>
## left_right_asymmetry — Left-Right Asymmetry

Mean right-minus-left mirrored quantile contrast; symmetric U and inverted-U are zero. Descriptive signed asymmetry, not curvature.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_left_right_asymmetry`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_left_right_asymmetry(qr: 'np.ndarray') -> 'np.ndarray'
```

Left-right asymmetry of the quantile profile per factor, (F,).

Mean right-minus-left mirrored bucket contrast. Symmetric U and inverted-U
profiles are zero; curvature is not asymmetry. The central bucket (odd Q)
is excluded. All mirrored pairs must be finite; direction is descriptive.

### 精确计算公式（实际实现）

```python
def compute_left_right_asymmetry(qr: np.ndarray) -> np.ndarray:
    """Left-right asymmetry of the quantile profile per factor, (F,).

    Mean right-minus-left mirrored bucket contrast. Symmetric U and inverted-U
    profiles are zero; curvature is not asymmetry. The central bucket (odd Q)
    is excluded. All mirrored pairs must be finite; direction is descriptive.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 4:
        return out
    mid = nq // 2
    for f in range(F):
        col = m[:, f]
        bottom = col[:mid]
        top = col[-mid:][::-1]
        if not np.all(np.isfinite(np.concatenate([bottom, top]))):
            continue
        out[f] = float(np.mean(top - bottom))
    return out
```

<a id="metric-linear_trend_score"></a>
## linear_trend_score — Linear Trend Score

Pearson correlation of the quantile profile with the linear quantile coordinate per factor in [-1, 1]. +1 = perfectly monotone increasing, -1 = decreasing, ~0 = flat or U-shaped (separate the U with u_shape_score).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_linear_trend_score`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_linear_trend_score(qr: 'np.ndarray') -> 'np.ndarray'
```

Linear-trend score of the quantile profile per factor, (F,).

Pearson correlation of the profile with the linear quantile coordinate,
bounded to [-1, 1].  ``+1`` = perfectly monotone increasing, ``-1`` =
perfectly monotone decreasing, ~0 = flat or U-shaped (the U-shape kite
is separated by ``u_shape_score``).  Direction ``higher_is_better``.
NaN when fewer than 3 finite quantile returns.

### 精确计算公式（实际实现）

```python
def compute_linear_trend_score(qr: np.ndarray) -> np.ndarray:
    """Linear-trend score of the quantile profile per factor, (F,).

    Pearson correlation of the profile with the linear quantile coordinate,
    bounded to [-1, 1].  ``+1`` = perfectly monotone increasing, ``-1`` =
    perfectly monotone decreasing, ~0 = flat or U-shaped (the U-shape kite
    is separated by ``u_shape_score``).  Direction ``higher_is_better``.
    NaN when fewer than 3 finite quantile returns.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 3:
        return out
    x = _template_fit_x(nq)
    for f in range(F):
        col = m[:, f]
        finite = np.isfinite(col)
        if np.sum(finite) < 3:
            continue
        y = col[finite]
        xf = x[finite]
        if np.ptp(y) == 0.0:
            out[f] = 0.0
            continue
        r = np.corrcoef(xf, y)[0, 1]
        out[f] = float(r) if np.isfinite(r) else np.nan
    return out
```

<a id="metric-liquidity_exposure"></a>
## liquidity_exposure — Liquidity Exposure

Signed mean liquidity-style exposure of the factor (typed per-style field). NaN when style absent or no finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_liquidity_exposure`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_liquidity_exposure(panel: 'FactorLoadingSeries', *, factor_values=None, min_obs=10, weights=None) -> 'float'
```

Signed mean liquidity-style exposure of the factor (one typed field).

### 精确计算公式（实际实现）

```python
def compute_liquidity_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean liquidity-style exposure of the factor (one typed field)."""
    return _select_style(panel, "liquidity",factor_values=factor_values,min_obs=min_obs,weights=weights)
```

<a id="metric-long_short_returns"></a>
## long_short_returns — long_short_returns

Time series of long-minus-short portfolio returns, shape (T,) or (T, F). Portfolio construction: long bucket = factor values above the long_threshold quantile (default 0.8), short bucket = values below short_threshold (default 0.2); equal-weight mean of forward returns per bucket per day, long minus short. Missing-return policy defaults to 'zero_fill' (NaN returns contribute 0).

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`zero_fill`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_long_short_returns`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.portfolio_stats.compute_long_short_returns(factor_values: numpy.ndarray, forward_returns: numpy.ndarray, long_threshold: float = 0.8, short_threshold: float = 0.2, validity_mask: Optional[numpy.ndarray] = None, missing_return_policy: str = 'zero_fill', tie_policy: str = 'max') -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
```

Compute long/short portfolio returns based on factor quantiles.

Missing-return policy (QE-METRIC P0-10):

- ``"zero_fill"`` (default, back-compat): forward returns that are NaN
  are treated as 0 for the assets selected into the long/short buckets.
  This affects bucket means (a NaN-return asset contributes 0 instead
  of being excluded) and is documented here precisely because it can
  bias portfolio returns toward 0 in sparse universes.
- ``"drop"``: assets with non-finite forward returns are excluded from
  the bucket means; if a bucket ends up empty, that period's return is
  NaN (never 0).
- ``"fail"``: raise ValueError if any forward return is non-finite.

Args:
    factor_values: Factor values (T, N) or (T, N, F)
    forward_returns: Forward returns (T, N)
    long_threshold: Quantile threshold for long positions (default 0.8 = top 20%)
    short_threshold: Quantile threshold for short positions (default 0.2 = bottom 20%)
    validity_mask: Optional boolean mask (T, N) or (T, N, F)
    missing_return_policy: "zero_fill" | "drop" | "fail" (see above)

Returns:
    (long_returns, short_returns, long_short_returns)
    Each shape (T,) or (T, F) for time series of portfolio returns

### 精确计算公式（实际实现）

```python
def compute_long_short_returns(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    long_threshold: float = 0.8,
    short_threshold: float = 0.2,
    validity_mask: Optional[np.ndarray] = None,
    missing_return_policy: str = "zero_fill",
    tie_policy: str = "max",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute long/short portfolio returns based on factor quantiles.

    Missing-return policy (QE-METRIC P0-10):

    - ``"zero_fill"`` (default, back-compat): forward returns that are NaN
      are treated as 0 for the assets selected into the long/short buckets.
      This affects bucket means (a NaN-return asset contributes 0 instead
      of being excluded) and is documented here precisely because it can
      bias portfolio returns toward 0 in sparse universes.
    - ``"drop"``: assets with non-finite forward returns are excluded from
      the bucket means; if a bucket ends up empty, that period's return is
      NaN (never 0).
    - ``"fail"``: raise ValueError if any forward return is non-finite.

    Args:
        factor_values: Factor values (T, N) or (T, N, F)
        forward_returns: Forward returns (T, N)
        long_threshold: Quantile threshold for long positions (default 0.8 = top 20%)
        short_threshold: Quantile threshold for short positions (default 0.2 = bottom 20%)
        validity_mask: Optional boolean mask (T, N) or (T, N, F)
        missing_return_policy: "zero_fill" | "drop" | "fail" (see above)

    Returns:
        (long_returns, short_returns, long_short_returns)
        Each shape (T,) or (T, F) for time series of portfolio returns
    """
    _validate_missing_return_policy(missing_return_policy)
    from .quantile import _searchsorted_bins
    from quant_evaluator.contracts.quantile_policy import validate_tie_policy
    policy = validate_tie_policy(tie_policy)
    factor_values = np.asarray(factor_values)
    forward_returns = np.asarray(forward_returns)
    if factor_values.ndim not in (2, 3) or forward_returns.shape != factor_values.shape[:2]:
        raise ValueError("factor_values must be (T,N[,F]) and forward_returns must match (T,N)")
    if not (np.isfinite(short_threshold) and np.isfinite(long_threshold)
            and 0 < short_threshold < long_threshold < 1):
        raise ValueError("thresholds require 0 < short_threshold < long_threshold < 1")
    if validity_mask is not None:
        validity_mask = np.asarray(validity_mask)
        if validity_mask.dtype != np.dtype(bool) or validity_mask.shape not in (factor_values.shape, factor_values.shape[:2]):
            raise ValueError("validity_mask must be boolean with matching panel or factor shape")
    if missing_return_policy == "fail" and np.any(~np.isfinite(forward_returns)):
        n_missing = int(np.sum(~np.isfinite(forward_returns)))
        raise ValueError(
            f"forward_returns contains {n_missing} non-finite value(s) and "
            "missing_return_policy='fail'"
        )

    factor_values, forward_returns = np.asarray(factor_values), np.asarray(forward_returns)
    if factor_values.ndim not in (2, 3) or forward_returns.shape != factor_values.shape[:2]:
        raise ValueError("factor and label axes must match (T,N[,F]) and (T,N)")
    if not 0 <= short_threshold < long_threshold <= 1:
        raise ValueError("require 0 <= short_threshold < long_threshold <= 1")
    if validity_mask is not None:
        validity_mask = np.asarray(validity_mask)
        if validity_mask.dtype != np.bool_ or validity_mask.shape not in (factor_values.shape[:2], factor_values.shape):
            raise ValueError("validity_mask must be boolean (T,N) or match factor axes")
    # Handle 3D factor values
    if factor_values.ndim == 3:
        T, N, F = factor_values.shape
        long_rets = np.full((T, F), np.nan)
        short_rets = np.full((T, F), np.nan)
        ls_rets = np.full((T, F), np.nan)

        for f in range(F):
            fv = factor_values[:, :, f]
            vm = (validity_mask[:, :, f] if validity_mask.ndim == 3 else validity_mask) if validity_mask is not None else None
            long_rets[:, f], short_rets[:, f], ls_rets[:, f] = compute_long_short_returns(
                fv, forward_returns, long_threshold, short_threshold, vm,
                missing_return_policy=missing_return_policy,
                tie_policy=tie_policy,
            )
        return long_rets, short_rets, ls_rets

    # 2D case
    T, N = factor_values.shape
    long_returns = np.full(T, np.nan)
    short_returns = np.full(T, np.nan)
    long_short_returns = np.full(T, np.nan)

    for t in range(T):
        factor_t = factor_values[t, :]
        ret_t = forward_returns[t, :]

        # Apply validity mask
        if validity_mask is not None:
            valid = validity_mask[t, :]
            factor_t = np.where(valid, factor_t, np.nan)

        # Filter finite factor values. Missing-return policy:
        # - "zero_fill": buckets are formed on finite factors only; NaN
        #   forward returns contribute 0 to the bucket mean (documented).
        # - "drop": assets with non-finite returns are excluded entirely.
        finite_mask = np.isfinite(factor_t)
        if missing_return_policy == "zero_fill":
            ret_t = np.where(np.isfinite(ret_t), ret_t, 0.0)

        if np.sum(finite_mask) < 2:
            # QE-R2 (P0-FA-015 hardening): a long/short bucket needs at least
            # two valid cross-sectional observations (a single asset cannot
            # form a top-20%/bottom-20% bucket pair).  Leave the period NaN —
            # an empty bucket is never a fabricated 0.
            continue

        factor_valid = factor_t[finite_mask]
        ret_valid = ret_t[finite_mask]

        # Compute quantiles
        long_cutoff = np.quantile(factor_valid, long_threshold)
        short_cutoff = np.quantile(factor_valid, short_threshold)

        # Select long/short positions
        bins = _searchsorted_bins(np.array([short_cutoff, long_cutoff]), factor_valid, 3, policy)
        long_mask = bins == 2
        short_mask = bins == 0
        # Ex-post missing labels affect measured return, never membership.

        if np.sum(long_mask) > 0:
            long_returns[t] = np.mean(ret_valid[long_mask])

        if np.sum(short_mask) > 0:
            short_returns[t] = np.mean(ret_valid[short_mask])

        if np.sum(long_mask) > 0 and np.sum(short_mask) > 0:
            long_short_returns[t] = equal_gross_long_short_returns(
                long_mask[None, :], short_mask[None, :], ret_valid[None, :],
                missing_return_policy=missing_return_policy)[0]

    return long_returns, short_returns, long_short_returns
```

<a id="metric-max_absolute_style_exposure"></a>
## max_absolute_style_exposure — Max Absolute Style Exposure

The style dimension with the largest mean absolute exposure (dict: style / value / absolute_mean / counts). NaN when no style has enough finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_max_absolute_style_exposure`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_max_absolute_style_exposure(panel: 'FactorLoadingSeries', min_finite: 'int' = 5, *, factor_values=None, min_obs=10, weights=None) -> 'Dict[str, Any]'
```

The style dimension with the largest mean *absolute* exposure.

Returns a plain dict (``style`` / ``value`` / ``absolute_mean`` /
``counts``) — the ``max_absolute_style_exposure`` evidence payload.
``value`` is the signed mean exposure of the winning style; NaN when no
style has at least ``min_finite`` finite observations.

### 精确计算公式（实际实现）

```python
def compute_max_absolute_style_exposure(
    panel: FactorLoadingSeries,
    min_finite: int = 5,
    *, factor_values=None, min_obs=10, weights=None,
) -> Dict[str, Any]:
    """The style dimension with the largest mean *absolute* exposure.

    Returns a plain dict (``style`` / ``value`` / ``absolute_mean`` /
    ``counts``) — the ``max_absolute_style_exposure`` evidence payload.
    ``value`` is the signed mean exposure of the winning style; NaN when no
    style has at least ``min_finite`` finite observations.
    """
    panel=_as_factor_loadings(panel,factor_values,min_obs,weights)
    if isinstance(min_finite,bool) or not isinstance(min_finite,(int,np.integer)) or min_finite<1:
        raise ValueError("min_finite must be a positive integer")
    ev = compute_style_exposure_evidence(panel, absolute=True)
    valid = np.isfinite(ev.values) & (ev.counts >= min_finite)
    if not np.any(valid):
        return {
            "style": ExposureStyle.UNKNOWN.value,
            "value": np.nan,
            "absolute_mean": np.nan,
            "counts": 0,
        }
    idx = int(np.argmax(np.where(valid, ev.values, -np.inf)))
    signed = compute_style_exposure_evidence(panel, absolute=False)
    return {
        "style": str(panel.style_names[idx]),
        "value": float(signed.values[idx]),
        "absolute_mean": float(ev.values[idx]),
        "counts": int(ev.counts[idx]),
    }
```

<a id="metric-max_drawdown"></a>
## max_drawdown — max_drawdown

Maximum compounded portfolio NAV drawdown including initial capital and default

- 版本：`2.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_maximum_drawdown`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.portfolio_stats.compute_maximum_drawdown(returns: numpy.ndarray, missing_return_policy: str = 'unknown') -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
```

Compute maximum drawdown from return series.

This is the drawdown authority alongside
``metrics/risk/drawdown_analysis.py``; zero wealth is an absorbing 100%
loss and returns below -100% require a separate capital contract.

Unknown valuation remains unknown by default. Explicit ``zero_fill`` is
a legacy research assumption; ``fail`` rejects a nonfinite return.

Args:
    returns: Return series (T,) or (T, F)
    missing_return_policy: "unknown", explicit "zero_fill", or "fail".

Returns:
    (max_drawdown, drawdown_series, peak_indices)
    max_drawdown: Maximum drawdown magnitude (positive), shape () or (F,)
    drawdown_series: Drawdown at each time step, shape (T,) or (T, F);
        -1 from the first zero wealth onward (observed default)
    peak_indices: Index of the PEAK (last index where the running
        maximum is attained at or before the maximum-drawdown trough),
        shape () or (F,); -1 denotes initial capital before the first return.

### 精确计算公式（实际实现）

```python
def compute_maximum_drawdown(
    returns: np.ndarray,
    missing_return_policy: str = "unknown",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute maximum drawdown from return series.

    This is the drawdown authority alongside
    ``metrics/risk/drawdown_analysis.py``; zero wealth is an absorbing 100%
    loss and returns below -100% require a separate capital contract.

    Unknown valuation remains unknown by default. Explicit ``zero_fill`` is
    a legacy research assumption; ``fail`` rejects a nonfinite return.

    Args:
        returns: Return series (T,) or (T, F)
        missing_return_policy: "unknown", explicit "zero_fill", or "fail".

    Returns:
        (max_drawdown, drawdown_series, peak_indices)
        max_drawdown: Maximum drawdown magnitude (positive), shape () or (F,)
        drawdown_series: Drawdown at each time step, shape (T,) or (T, F);
            -1 from the first zero wealth onward (observed default)
        peak_indices: Index of the PEAK (last index where the running
            maximum is attained at or before the maximum-drawdown trough),
            shape () or (F,); -1 denotes initial capital before the first return.
    """
    if missing_return_policy not in ("unknown", "zero_fill", "fail"):
        raise ValueError(
            f"missing_return_policy must be 'unknown', 'zero_fill' or 'fail', "
            f"got {missing_return_policy!r}"
        )

    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape

    if missing_return_policy == "fail" and np.any(~np.isfinite(returns)):
        n_missing = int(np.sum(~np.isfinite(returns)))
        raise ValueError(
            f"returns contains {n_missing} non-finite value(s) and "
            "missing_return_policy='fail'"
        )

    if T == 0:
        if squeeze:
            return float("nan"), np.empty(0, dtype=np.float64), -1
        return np.full(F, np.nan), np.empty((0, F)), np.full(F, -1, dtype=np.int64)
    from .risk.drawdown_analysis import compute_drawdown_series
    drawdown_series, cum_returns, running_max = compute_drawdown_series(returns)
    if missing_return_policy == "unknown":
        unknown = np.maximum.accumulate(~np.isfinite(returns), axis=0)
        bankrupt = np.maximum.accumulate(returns == -1.0, axis=0)
        drawdown_series = np.where(unknown & ~bankrupt, np.nan, drawdown_series)

    # Maximum drawdown per factor (most negative, converted to positive).
    max_dd = -np.min(np.where(np.isfinite(drawdown_series), drawdown_series, np.inf), axis=0)
    max_dd = np.where(np.isfinite(max_dd), max_dd, np.nan)
    if missing_return_policy == "unknown":
        max_dd = np.where(np.any(~np.isfinite(returns), axis=0), np.nan, max_dd)
        max_dd = np.where(np.any(returns == -1.0, axis=0), 1.0, max_dd)

    # Trough index per factor: first occurrence of the minimum drawdown,
    # NaN-safe (an all-NaN column has no
    # defined trough; np.nanargmin would raise on it).
    trough_indices = np.empty(F, dtype=np.int64)
    for f in range(F):
        col = drawdown_series[:, f]
        finite_idx = np.nonzero(np.isfinite(col))[0]
        if finite_idx.size == 0:
            trough_indices[f] = 0
            continue
        vals = col[finite_idx]
        min_val = np.min(vals)
        trough_indices[f] = int(finite_idx[np.nonzero(vals == min_val)[0][0]])

    # Peak index: last index at or before the trough where the wealth curve
    # attains its running maximum (i.e. cum_returns == running_max). This is
    # the true peak of the maximum drawdown episode, not the trough.
    peak_indices = np.empty(F, dtype=np.int64)
    for f in range(F):
        trough = int(trough_indices[f])
        col = drawdown_series[: trough + 1, f]
        finite_idx = np.nonzero(np.isfinite(col))[0]
        if finite_idx.size == 0:
            # Entire prefix nonfinite: peak undefined,
            # use index 0.
            peak_indices[f] = 0
            continue
        trough_eff = int(finite_idx[-1])
        # Exact high-water convention, identical to the drawdown magnitude.
        at_max = cum_returns[: trough_eff + 1, f] == running_max[trough_eff, f]
        if not np.any(at_max):
            peak_indices[f] = -1
            continue
        # Last index where wealth equals the running max at the trough.
        peak_indices[f] = int(np.nonzero(at_max)[0][-1])

    if squeeze:
        return max_dd[0], drawdown_series[:, 0], int(peak_indices[0])
    else:
        return max_dd, drawdown_series, peak_indices
```

<a id="metric-max_underwater_duration"></a>
## max_underwater_duration — Max Underwater Duration

Longest continuous stretch (periods) of the probe daily PnL series staying below its running-max wealth (waterline). Computed by metrics.underwater.compute_max_underwater_duration on a (T,) dot-frequency series (probe cohort pnl). NaN when insufficient data — never a fabricated 0.

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`periods`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_max_underwater_duration`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_max_underwater_duration(returns: 'np.ndarray', min_periods: 'int' = 10) -> 'float'
```

Longest underwater grid span; unknown valuation paths return NaN.

### 精确计算公式（实际实现）

```python
def compute_max_underwater_duration(returns: np.ndarray, min_periods: int = 10) -> float:
    """Longest underwater grid span; unknown valuation paths return NaN."""
    events = _path_events(returns, min_periods)
    if events is None:
        return np.nan
    return float(max((e["duration"] for e in events), default=0))
```

<a id="metric-mean_ic"></a>
## mean_ic — Mean IC

Time-averaged Pearson information coefficient: the time-mean of daily Pearson IC between factor values and labels (canonical alias ic.pearson.mean). Observation_count is the number of finite daily IC observations under the same pair validity and minimum-assets policy.

- 版本：`1`；状态：`stable`；层级：`core`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`mean_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.ic.compute_mean_ic_value(ic_series: numpy.ndarray, valid_counts: Optional[numpy.ndarray] = None, min_periods: int = 20) -> numpy.ndarray
```

Compute only the mean IC component for registry execution.

### 精确计算公式（实际实现）

```python
def compute_mean_ic_value(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> np.ndarray:
    """Compute only the mean IC component for registry execution."""
    mean_ic, _ = compute_mean_ic(
        ic_series,
        valid_counts=valid_counts,
        min_periods=min_periods,
    )
    return mean_ic
```

<a id="metric-mean_investment_fraction"></a>
## mean_investment_fraction — Mean Investment Fraction

Benchmark/invested-capital evidence from an explicitly bound execution trajectory leg

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`fraction`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.long_only.compute_mean_investment_fraction`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.long_only.compute_mean_investment_fraction(returns, min_periods: 'int' = 1)
```

Mean actual invested-capital fraction; all-cash is explicitly 0, not missing.

### 精确计算公式（实际实现）

```python
def compute_mean_investment_fraction(returns, min_periods: int = 1):
    """Mean actual invested-capital fraction; all-cash is explicitly 0, not missing."""
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 1:
        raise ValueError("min_periods must be an integer >= 1")
    x, squeeze = _matrix(returns)
    out = np.full(x.shape[1], np.nan)
    for f in range(x.shape[1]):
        v = x[np.isfinite(x[:, f]), f]
        if len(v) >= min_periods:
            if np.any(v < 0):
                raise ValueError("investment fraction must be nonnegative")
            out[f] = np.mean(v)
    return out[0] if squeeze else out
```

<a id="metric-mean_underwater_duration"></a>
## mean_underwater_duration — Mean Underwater Duration

Mean length (periods) of underwater episodes of the probe daily PnL series. NaN when insufficient data; 0.0 when never underwater.

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`periods`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_mean_underwater_duration`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_mean_underwater_duration(returns: 'np.ndarray', min_periods: 'int' = 10) -> 'float'
```

Mean observed underwater span, including explicitly censored events.

### 精确计算公式（实际实现）

```python
def compute_mean_underwater_duration(returns: np.ndarray, min_periods: int = 10) -> float:
    """Mean observed underwater span, including explicitly censored events."""
    events = _path_events(returns, min_periods)
    if events is None:
        return np.nan
    return float(np.mean([e["duration"] for e in events])) if events else 0.0
```

<a id="metric-missing_ratio"></a>
## missing_ratio — Missing Ratio

Fraction of (T, N) cells with non-finite factor values, per factor (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_missing_ratio`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_missing_ratio(factor_batch: 'FactorBatch') -> 'np.ndarray'
```

Fraction of (T, N) cells with non-finite factor values, (F,).

### 精确计算公式（实际实现）

```python
def compute_missing_ratio(factor_batch: FactorBatch) -> np.ndarray:
    """Fraction of (T, N) cells with non-finite factor values, (F,)."""
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    missing = np.sum(~np.isfinite(values), axis=(0, 1))
    return missing / (T * N)
```

<a id="metric-missing_timeline"></a>
## missing_timeline — Missing Timeline

Fraction of time periods with any missing factor value, per factor. 1.0 means every day has at least one missing asset (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_missing_timeline`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_missing_timeline(factor_batch: 'FactorBatch') -> 'np.ndarray'
```

Fraction of time periods with any missing factor value, (F,).

A value of 1.0 means every day has at least one missing asset; 0.0 means
the factor is fully populated on every day.

### 精确计算公式（实际实现）

```python
def compute_missing_timeline(factor_batch: FactorBatch) -> np.ndarray:
    """Fraction of time periods with any missing factor value, (F,).

    A value of 1.0 means every day has at least one missing asset; 0.0 means
    the factor is fully populated on every day.
    """
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    any_missing = np.any(~np.isfinite(values), axis=1)  # (T, F)
    return np.sum(any_missing, axis=0) / T
```

<a id="metric-momentum_exposure"></a>
## momentum_exposure — Momentum Exposure

Signed mean momentum-style exposure of the factor (typed per-style field). NaN when style absent or no finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_momentum_exposure`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_momentum_exposure(panel: 'FactorLoadingSeries', *, factor_values=None, min_obs=10, weights=None) -> 'float'
```

Signed mean momentum-style exposure of the factor (one typed field).

### 精确计算公式（实际实现）

```python
def compute_momentum_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean momentum-style exposure of the factor (one typed field)."""
    return _select_style(panel, "momentum",factor_values=factor_values,min_obs=min_obs,weights=weights)
```

<a id="metric-month_consistency"></a>
## month_consistency — Month Consistency

Fraction of months whose mean IC matches the overall IC sign, per factor (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_month_consistency`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_month_consistency(ic_series: 'np.ndarray', min_periods: 'int' = 20, time_index: 'Optional[Sequence]' = None) -> 'np.ndarray'
```

Fraction of months whose mean IC matches the overall IC sign, (F,).

### 精确计算公式（实际实现）

```python
def compute_month_consistency(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Fraction of months whose mean IC matches the overall IC sign, (F,)."""
    s = _as_series(ic_series)
    out = _period_consistency(s, "month", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-monthly_rank_ic"></a>
## monthly_rank_ic — Monthly Rank IC

Mean of the per-month mean rank IC, per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_monthly_rank_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_monthly_rank_ic(ic_series: 'np.ndarray', min_periods: 'int' = 20, time_index: 'Optional[Sequence]' = None) -> 'np.ndarray'
```

Mean of the per-month mean rank IC, (F,).

### 精确计算公式（实际实现）

```python
def compute_monthly_rank_ic(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Mean of the per-month mean rank IC, (F,)."""
    s = _as_series(ic_series)
    out = _mean_of_period_means(s, "month", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-neutralized_rank_ic"></a>
## neutralized_rank_ic — Neutralized Rank IC

Cross-sectional residual (neutralized) rank IC: per date regress the factor on the exposure panel (OLS intercept + styles), Spearman-correlate residuals with forward returns, time-mean. A factor whose IC survives neutralization has alpha orthogonal to style exposures. CPU reference; GPU optional (kernels/gpu/exposure_evidence.py).

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, forward_returns, exposure_panel`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_neutralized_rank_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_neutralized_rank_ic(factor_values: 'np.ndarray', forward_returns: 'np.ndarray', panel: 'ExposurePanel', min_obs: 'int' = 10) -> 'float'
```

Cross-sectional residual (neutralized) rank IC.

Per date t: regress the factor cross-section on the exposure panel
(intercept + K styles) via ``metrics/exposure.compute_factor_loadings``,
take the OLS residuals, then Spearman-rank-correlate the residuals with
forward returns.  The time-mean of the daily residual IC is the
neutralized rank IC.  A factor whose IC survives neutralization has alpha
orthogonal to the style exposures.

CPU reference implementation (GPU optional; parity checked in tests).
NaN when fewer than ``min_obs`` jointly-finite assets on every date, or
when the factor/label/panel shapes are inconsistent.

### 精确计算公式（实际实现）

```python
def compute_neutralized_rank_ic(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    panel: ExposurePanel,
    min_obs: int = 10,
) -> float:
    """Cross-sectional residual (neutralized) rank IC.

    Per date t: regress the factor cross-section on the exposure panel
    (intercept + K styles) via ``metrics/exposure.compute_factor_loadings``,
    take the OLS residuals, then Spearman-rank-correlate the residuals with
    forward returns.  The time-mean of the daily residual IC is the
    neutralized rank IC.  A factor whose IC survives neutralization has alpha
    orthogonal to the style exposures.

    CPU reference implementation (GPU optional; parity checked in tests).
    NaN when fewer than ``min_obs`` jointly-finite assets on every date, or
    when the factor/label/panel shapes are inconsistent.
    """
    fv = np.asarray(factor_values, dtype=np.float64)
    fwd = np.asarray(forward_returns, dtype=np.float64)
    if fv.ndim != 2 or fwd.ndim != 2:
        raise ValueError("factor_values / forward_returns must be (T, N)")
    if fv.shape != fwd.shape:
        raise ValueError(
            f"factor_values {fv.shape} and forward_returns {fwd.shape} must match"
        )
    arr, _ = _panel_arrays(panel)
    T, N, K = arr.shape
    if (T, N) != fv.shape:
        raise ValueError(
            f"panel (T,N)=({T},{N}) must match factor (T,N)={fv.shape}"
        )
    # Regress per date with intercept; residuals (T, N) — NaN where invalid.
    _, _, residuals = compute_factor_loadings(fv, arr, intercept=True, min_obs=min_obs,weights=panel.regression_weights)

    daily_ics: list[float] = []
    for t in range(T):
        y = fwd[t]
        resid = residuals[t]
        joint = np.isfinite(y) & np.isfinite(resid)
        if np.sum(joint) < 2:
            continue
        # ``min_obs`` is the declared per-date evidence floor for this metric,
        # and applies to the final residual/label pair as well as the OLS fit.
        rho = _spearman_rank_correlation(
            resid[joint], y[joint], min_obs=min_obs
        )
        if np.isfinite(rho):
            daily_ics.append(float(rho))
    if not daily_ics:
        return np.nan
    return float(np.mean(daily_ics))
```

<a id="metric-outlier_ratio"></a>
## outlier_ratio — Outlier Ratio

Fraction of finite factor values that are z-score outliers (|z|>3), per factor (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_outlier_ratio`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_outlier_ratio(factor_batch: 'FactorBatch', threshold: 'float' = 3.0, min_obs: 'int' = 10) -> 'np.ndarray'
```

Fraction of finite factor values that are z-score outliers, (F,).

Outliers are defined per factor over the full (T, N) panel using a
z-score threshold (default 3.0).

### 精确计算公式（实际实现）

```python
def compute_outlier_ratio(
    factor_batch: FactorBatch,
    threshold: float = 3.0,
    min_obs: int = 10,
) -> np.ndarray:
    """Fraction of finite factor values that are z-score outliers, (F,).

    Outliers are defined per factor over the full (T, N) panel using a
    z-score threshold (default 3.0).
    """
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    out = np.full(F, np.nan)
    for f in range(F):
        col = values[:, :, f]
        finite = np.isfinite(col)
        n = np.sum(finite)
        if n < min_obs:
            continue
        mean = np.nanmean(col)
        std = np.nanstd(col, ddof=1)
        if not np.isfinite(std) or std == 0:
            continue
        z = np.abs((col - mean) / std)
        out[f] = np.sum((z > threshold) & finite) / n
    return out
```

<a id="metric-parameter_generalization"></a>
## parameter_generalization — Parameter Generalization

Parameter-generalization summary per factor (plan §13.6): the robust retention mean where the train denominator is stable, NaN when no factor has a stable denominator (missing evidence - never 0). Versioned anchors in RetentionPolicy.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_validation_retention`。

### 计算定义与默认参数

```python
quant_evaluator.registry.metrics.<lambda>(v)
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
```

<a id="metric-pearson_ic"></a>
## pearson_ic — Mean Pearson IC

Time-mean of daily Pearson IC between factor values and labels (canonical alias ic.pearson.mean; same kernel as mean_ic). observation_count is finite daily IC days. Default daily minimum is 20 paired assets, shared with ICIR; mean needs 1 day, IR 20.

- 版本：`2.0.0`；状态：`stable`；层级：`core`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`drop_pair`；数值政策：`finite`；注册最低期数：`None`。
- 别名： `ic.pearson.mean` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic.compute_daily_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_pearson_ic_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, label_bundle: quant_evaluator.contracts.label_bundle.LabelBundle, min_periods: int = 1, min_assets: int = 20) -> numpy.ndarray
```

Return the time-mean daily Pearson IC per factor, shape (F,).

### 精确计算公式（实际实现）

```python
def compute_pearson_ic_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_periods: int = 1,
    min_assets: int = 20,
) -> np.ndarray:
    """Return the time-mean daily Pearson IC per factor, shape (F,)."""
    ic_series, _ = compute_daily_ic(
        factor_batch, label_bundle, method="pearson", min_assets=min_assets
    )
    mean, _ = compute_mean_ic(ic_series, min_periods=min_periods)
    return mean
```

<a id="metric-pearson_ic_ir"></a>
## pearson_ic_ir — Pearson IC Information Ratio

Mean Pearson IC divided by Pearson IC standard deviation per factor (canonical alias ic.pearson.ir). Fully separated from the Spearman-family ic_ir (ic.rank.ir).

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名： `ic.pearson.ir` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`pearson_ic_ir`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_ic_ir_value(ic_series: numpy.ndarray, min_periods: int = 20) -> numpy.ndarray
```

Return the scalar IC information ratio per factor.

### 精确计算公式（实际实现）

```python
def compute_ic_ir_value(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> np.ndarray:
    """Return the scalar IC information ratio per factor."""
    return compute_icir(ic_series, min_periods=min_periods)
```

<a id="metric-pearson_ic_series"></a>
## pearson_ic_series — Daily Pearson IC Series

Daily Pearson IC per factor over time, shape (T, F) (canonical alias ic.pearson.daily)

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名： `ic.pearson.daily` 。
- 增量模式：`APPEND_EXACT`；注册实现定位：`pearson_ic_series`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_pearson_ic_series_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, label_bundle: quant_evaluator.contracts.label_bundle.LabelBundle, min_assets: int = 20) -> numpy.ndarray
```

Return the daily Pearson IC series per factor, shape (T, F).

### 精确计算公式（实际实现）

```python
def compute_pearson_ic_series_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 20,
) -> np.ndarray:
    """Return the daily Pearson IC series per factor, shape (T, F)."""
    ic_series, _ = compute_daily_ic(
        factor_batch, label_bundle, method="pearson", min_assets=min_assets
    )
    return ic_series
```

<a id="metric-pearson_ic_std"></a>
## pearson_ic_std — Pearson IC Standard Deviation

Standard deviation of the daily Pearson IC series per factor (canonical alias ic.pearson.std). Fully separated from the Spearman-family ic_std (ic.rank.std).

- 版本：`0.1.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名： `ic.pearson.std` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`pearson_ic_std`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.ic.compute_ic_std(ic_series: numpy.ndarray, valid_counts: Optional[numpy.ndarray] = None, min_periods: int = 20) -> numpy.ndarray
```

Compute only the IC standard-deviation component.

### 精确计算公式（实际实现）

```python
def compute_ic_std(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> np.ndarray:
    """Compute only the IC standard-deviation component."""
    _, ic_std = compute_mean_ic(
        ic_series,
        valid_counts=valid_counts,
        min_periods=min_periods,
    )
    return ic_std
```

<a id="metric-purity_ratio"></a>
## purity_ratio — Purity Ratio

Time mean of 1 - R-squared from same-support weighted factor-on-risk regression with intercept. Constant factors and insufficient degrees of freedom are undefined. SAME_DATE_DESCRIPTIVE, not OOS evidence.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_purity_ratio`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_purity_ratio(panel: 'FactorLoadingSeries', min_finite: 'int' = 5, *, factor_values=None, min_obs=10, weights=None) -> 'float'
```

Time mean of residual/total weighted variance, exactly 1-R².

Same factor, joint support, weights and intercept regression as its
loadings. Constant factor, saturated model or no explanatory variation is
undefined, not perfect purity. This is in-sample descriptive evidence.

### 精确计算公式（实际实现）

```python
def compute_purity_ratio(panel: FactorLoadingSeries, min_finite: int = 5, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Time mean of residual/total weighted variance, exactly 1-R².

    Same factor, joint support, weights and intercept regression as its
    loadings. Constant factor, saturated model or no explanatory variation is
    undefined, not perfect purity. This is in-sample descriptive evidence.
    """
    panel=_as_factor_loadings(panel,factor_values,min_obs,weights)
    if isinstance(min_finite,bool) or not isinstance(min_finite,(int,np.integer)) or min_finite<1:
        raise ValueError("min_finite must be a positive integer")
    finite=panel.r_squared[np.isfinite(panel.r_squared)]
    return float(np.mean(1.-finite)) if len(finite)>=min_finite else float("nan")
```

<a id="metric-quantile_adjacent_spread"></a>
## quantile_adjacent_spread — Quantile Adjacent Spread

Mean absolute return difference between adjacent quantiles, per factor. A measure of how smooth (vs step-like) the quantile profile is (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_adjacent_spread`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.quantile_shape.compute_quantile_adjacent_spread(qr: 'np.ndarray') -> 'np.ndarray'
```

Mean absolute return difference between adjacent quantiles, (F,).

A measure of how smooth (vs step-like) the quantile profile is.  NaN
when fewer than 2 finite quantile returns.

### 精确计算公式（实际实现）

```python
def compute_quantile_adjacent_spread(qr: np.ndarray) -> np.ndarray:
    """Mean absolute return difference between adjacent quantiles, (F,).

    A measure of how smooth (vs step-like) the quantile profile is.  NaN
    when fewer than 2 finite quantile returns.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    ok = _finite_columns(m)
    if nq < 2:
        return out
    for f in np.where(ok)[0]:
        col = m[:, f]
        finite = np.isfinite(col)
        pairs = finite[:-1] & finite[1:]
        if not np.any(pairs):
            continue
        out[f] = float(np.mean(np.abs(col[1:][pairs] - col[:-1][pairs])))
    return out
```

<a id="metric-quantile_curvature"></a>
## quantile_curvature — Quantile Curvature

Signed curvature of the quantile-return profile (mean second difference), per factor. Positive = convex (accelerating) profile; negative = concave (decelerating) (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_curvature`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.quantile_shape.compute_quantile_curvature(qr: 'np.ndarray') -> 'np.ndarray'
```

Signed curvature of the quantile-return profile, (F,).

Estimated as the second difference of the mean quantile returns
(``ret[q+1] - 2*ret[q] + ret[q-1]``) averaged over interior quantiles.
Positive curvature = convex (accelerating) profile; negative = concave
(decelerating).  NaN when fewer than 3 finite quantile returns.

### 精确计算公式（实际实现）

```python
def compute_quantile_curvature(qr: np.ndarray) -> np.ndarray:
    """Signed curvature of the quantile-return profile, (F,).

    Estimated as the second difference of the mean quantile returns
    (``ret[q+1] - 2*ret[q] + ret[q-1]``) averaged over interior quantiles.
    Positive curvature = convex (accelerating) profile; negative = concave
    (decelerating).  NaN when fewer than 3 finite quantile returns.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 3:
        return out
    for f in range(F):
        col = m[:, f]
        finite = np.isfinite(col)
        # interior positions with all three neighbours finite
        interior = finite[1:-1] & finite[:-2] & finite[2:]
        if not np.any(interior):
            continue
        d2 = col[2:][interior] - 2.0 * col[1:-1][interior] + col[:-2][interior]
        out[f] = float(np.mean(d2))
    return out
```

<a id="metric-quantile_extreme_cliff"></a>
## quantile_extreme_cliff — Quantile Extreme Cliff

Mean of the top and bottom quantile cliffs, per factor. A large value means the extreme quantiles carry most of the spread (a cliff profile) (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_extreme_cliff`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.quantile_shape.compute_quantile_extreme_cliff(qr: 'np.ndarray') -> 'np.ndarray'
```

Mean of the top and bottom quantile cliffs, (F,).

``(cliff_top + cliff_bottom) / 2`` where cliff_top = ret[top] - ret[top-1]
and cliff_bottom = ret[1] - ret[0].  A large value means the extreme
quantiles carry most of the spread (a cliff profile).  NaN when the
required quantile returns are not finite.

### 精确计算公式（实际实现）

```python
def compute_quantile_extreme_cliff(qr: np.ndarray) -> np.ndarray:
    """Mean of the top and bottom quantile cliffs, (F,).

    ``(cliff_top + cliff_bottom) / 2`` where cliff_top = ret[top] - ret[top-1]
    and cliff_bottom = ret[1] - ret[0].  A large value means the extreme
    quantiles carry most of the spread (a cliff profile).  NaN when the
    required quantile returns are not finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 2:
        return out
    for f in range(F):
        col = m[:, f]
        if not (np.isfinite(col[0]) and np.isfinite(col[1])
                and np.isfinite(col[-1]) and np.isfinite(col[-2])):
            continue
        cliff_top = col[-1] - col[-2]
        cliff_bottom = col[1] - col[0]
        out[f] = (cliff_top + cliff_bottom) / 2.0
    return out
```

<a id="metric-quantile_monotonicity"></a>
## quantile_monotonicity — Adjacent Quantile Increase Fraction

Fraction of adjacent quantile steps that are monotone increasing, per factor. 1.0 = perfectly monotone (higher factor value -> higher return); 0.0 = no increasing adjacent pairs (flat OR decreasing). Not signed Spearman monotonicity; a >=0 gate has no direction filter.

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_monotonicity`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.quantile_shape.compute_quantile_monotonicity(qr: 'np.ndarray') -> 'np.ndarray'
```

Fraction of adjacent quantile steps that are monotone increasing, (F,).

For each factor, count adjacent pairs (q, q+1) where
``ret[q+1] > ret[q]`` over the finite pairs, divided by the number of
finite adjacent pairs.  NaN when fewer than 2 finite quantile returns.

### 精确计算公式（实际实现）

```python
def compute_quantile_monotonicity(qr: np.ndarray) -> np.ndarray:
    """Fraction of adjacent quantile steps that are monotone increasing, (F,).

    For each factor, count adjacent pairs (q, q+1) where
    ``ret[q+1] > ret[q]`` over the finite pairs, divided by the number of
    finite adjacent pairs.  NaN when fewer than 2 finite quantile returns.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    ok = _finite_columns(m)
    if nq < 2:
        return out
    for f in np.where(ok)[0]:
        col = m[:, f]
        finite = np.isfinite(col)
        pairs = finite[:-1] & finite[1:]
        n_pairs = int(np.sum(pairs))
        if n_pairs == 0:
            continue
        inc = np.sum((col[1:][pairs] > col[:-1][pairs]))
        out[f] = inc / n_pairs
    return out
```

<a id="metric-quantile_rank_monotonicity"></a>
## quantile_rank_monotonicity — Signed Quantile Rank Monotonicity

Spearman(bucket index, mean bucket return), [-1,1]; requires all buckets finite and at least three. Flat profile is unavailable. Distinct from adjacent increase fraction.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`complete_profile`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_quantile_rank_monotonicity`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.quantile_shape.compute_quantile_rank_monotonicity(qr: 'np.ndarray') -> 'np.ndarray'
```

Signed Spearman correlation of bucket index and mean bucket return.

Requires every bucket to be finite and at least three buckets. Average
ranks handle ties; an exactly flat profile is undefined (NaN). Range
[-1, 1]. Unlike adjacent increase fraction this retains direction and
refuses to score a partially observed quantile profile.

### 精确计算公式（实际实现）

```python
def compute_quantile_rank_monotonicity(qr: np.ndarray) -> np.ndarray:
    """Signed Spearman correlation of bucket index and mean bucket return.

    Requires every bucket to be finite and at least three buckets. Average
    ranks handle ties; an exactly flat profile is undefined (NaN). Range
    [-1, 1]. Unlike adjacent increase fraction this retains direction and
    refuses to score a partially observed quantile profile.
    """
    from scipy.stats import spearmanr
    m = _as_matrix(qr)
    out = np.full(m.shape[1], np.nan)
    if m.shape[0] < 3:
        return out
    for f in range(m.shape[1]):
        col = m[:, f]
        if np.isfinite(col).all() and np.any(col != col[0]):
            out[f] = spearmanr(np.arange(len(col)), col).statistic
    return out
```

<a id="metric-quantile_returns"></a>
## quantile_returns — quantile_returns

Average forward return per quantile bucket

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile.compute_quantile_returns`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.build_daily_quantile_return_artifact(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, label_bundle: quant_evaluator.contracts.label_bundle.LabelBundle, n_quantiles: int = 5, min_assets: int = 10, min_periods: int = 20, *, tie_status_ref: str | None = None, tradability_ref: str | None = None, risk_exposure_ref: str | None = None, producer_version: str = '1.0.0', split_ref: str | None = None, config_hash: str | None = None) -> quant_evaluator.contracts.artifact_types.DailyQuantileReturnArtifact
```

Build the non-aggregated daily TQF quantile evidence artifact.

### 精确计算公式（实际实现）

```python
def build_daily_quantile_return_artifact(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
    min_periods: int = 20,
    *,
    tie_status_ref: str | None = None,
    tradability_ref: str | None = None,
    risk_exposure_ref: str | None = None,
    producer_version: str = "1.0.0",
    split_ref: str | None = None,
    config_hash: str | None = None,
) -> DailyQuantileReturnArtifact:
    """Build the non-aggregated daily TQF quantile evidence artifact."""
    if min_periods < 1:
        raise ValueError("min_periods must be positive")
    returns, counts = compute_quantile_returns_fast(
        factor_batch, label_bundle, n_quantiles=n_quantiles,
        min_assets=min_assets,
    )
    counts = np.asarray(counts)
    valid = np.isfinite(returns) & (counts >= min_assets)
    time_axis = tuple(label_bundle.observation_time or label_bundle.decision_time)
    return DailyQuantileReturnArtifact(
        values=returns,
        counts=counts.astype(np.int64, copy=False),
        valid_mask=valid,
        time_axis=time_axis,
        quantile_axis=tuple(range(n_quantiles)),
        factor_axis=tuple(factor_batch.factor_ids),
        tie_status_ref=tie_status_ref,
        tradability_ref=tradability_ref,
        risk_exposure_ref=risk_exposure_ref,
        producer_version=producer_version,
        provenance={
            "label_id": label_bundle.target_id,
            "label_content_hash": label_bundle.content_hash,
            "factor_value_hash": factor_batch.value_hash,
            "n_quantiles": n_quantiles,
            "min_assets": min_assets,
            "min_periods": min_periods,
            "split_ref": split_ref,
            "config_hash": config_hash,
            "valid_period_counts_qf": np.sum(valid, axis=0),
        },
    )
```

<a id="metric-quantile_returns_full"></a>
## quantile_returns_full — Full Quantile Returns

Per-quantile time-averaged returns as a VECTOR per factor — shape (n_quantiles, F), NOT a scalar; wrap with metrics.registry_adapters.compute_quantile_returns_full_artifact for the typed VectorMetricArtifact

- 版本：`0.1.0`；状态：`stable`；层级：`research`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quantile_returns_full`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_quantile_returns_full_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, label_bundle: quant_evaluator.contracts.label_bundle.LabelBundle, min_periods: int = 20, n_quantiles: int = 5) -> numpy.ndarray
```

Return per-quantile time-averaged returns, shape (n_quantiles, F).

### 精确计算公式（实际实现）

```python
def compute_quantile_returns_full_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_periods: int = 20,
    n_quantiles: int = 5,
) -> np.ndarray:
    """Return per-quantile time-averaged returns, shape (n_quantiles, F)."""
    quantile_returns, _ = compute_quantile_returns_fast(
        factor_batch, label_bundle, n_quantiles=n_quantiles
    )
    with np.errstate(invalid="ignore"):
        means = np.nanmean(quantile_returns, axis=0)  # (n_quantiles, F)
    valid_counts = np.sum(np.isfinite(quantile_returns), axis=0)  # (n_quantiles, F)
    return np.where(valid_counts >= min_periods, means, np.nan)
```

<a id="metric-quantile_spread"></a>
## quantile_spread — Top-Bottom Quantile Spread

Return spread between top and bottom quantiles

- 版本：`3.0.0`；状态：`stable`；层级：`core`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile.compute_top_bottom_spread`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_quantile_spread_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, label_bundle: quant_evaluator.contracts.label_bundle.LabelBundle, min_periods: int = 20, n_quantiles: int = 5) -> numpy.ndarray
```

Return time-averaged top-minus-bottom quantile return spread per factor.

### 精确计算公式（实际实现）

```python
def compute_quantile_spread_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_periods: int = 20,
    n_quantiles: int = 5,
) -> np.ndarray:
    """Return time-averaged top-minus-bottom quantile return spread per factor."""
    quantile_returns, _ = compute_quantile_returns_fast(
        factor_batch, label_bundle, n_quantiles=n_quantiles
    )
    # (T, n_quantiles, F) -> mean over time of Q_top - Q_bottom per factor.
    with np.errstate(invalid="ignore"):
        spread_series = quantile_returns[:, -1, :] - quantile_returns[:, 0, :]
    valid_counts = np.sum(np.isfinite(spread_series), axis=0)
    means = np.nanmean(spread_series, axis=0) if spread_series.size else np.array([])
    return np.where(valid_counts >= min_periods, means, np.nan)
```

<a id="metric-quantile_stability"></a>
## quantile_stability — quantile_stability

Stability of quantile return rankings across time

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic_summary.compute_ic_stability`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.quantile_shape.compute_quantile_tail_asymmetry(qr: 'np.ndarray') -> 'np.ndarray'
```

Asymmetry between the top and bottom quantile tails, (F,).

``(ret[top] - ret[mid]) - (ret[mid] - ret[bottom])`` where mid is the
median quantile index.  Positive = top tail is stronger than the bottom
tail.  NaN when the top/bottom/mid quantile returns are not all finite.

### 精确计算公式（实际实现）

```python
def compute_quantile_tail_asymmetry(qr: np.ndarray) -> np.ndarray:
    """Asymmetry between the top and bottom quantile tails, (F,).

    ``(ret[top] - ret[mid]) - (ret[mid] - ret[bottom])`` where mid is the
    median quantile index.  Positive = top tail is stronger than the bottom
    tail.  NaN when the top/bottom/mid quantile returns are not all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 3:
        return out
    mid = nq // 2
    for f in range(F):
        col = m[:, f]
        if not (np.isfinite(col[0]) and np.isfinite(col[-1]) and np.isfinite(col[mid])):
            continue
        out[f] = (col[-1] - col[mid]) - (col[mid] - col[0])
    return out
```

<a id="metric-quarter_consistency"></a>
## quarter_consistency — Quarter Consistency

Fraction of quarters whose mean IC matches the overall IC sign, per factor (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_quarter_consistency`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_quarter_consistency(ic_series: 'np.ndarray', min_periods: 'int' = 20, time_index: 'Optional[Sequence]' = None) -> 'np.ndarray'
```

Fraction of quarters whose mean IC matches the overall IC sign, (F,).

### 精确计算公式（实际实现）

```python
def compute_quarter_consistency(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Fraction of quarters whose mean IC matches the overall IC sign, (F,)."""
    s = _as_series(ic_series)
    out = _period_consistency(s, "quarter", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-quarterly_rank_ic"></a>
## quarterly_rank_ic — Quarterly Rank IC

Mean of the per-quarter mean rank IC, per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_quarterly_rank_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_quarterly_rank_ic(ic_series: 'np.ndarray', min_periods: 'int' = 20, time_index: 'Optional[Sequence]' = None) -> 'np.ndarray'
```

Mean of the per-quarter mean rank IC, (F,).

### 精确计算公式（实际实现）

```python
def compute_quarterly_rank_ic(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Mean of the per-quarter mean rank IC, (F,)."""
    s = _as_series(ic_series)
    out = _mean_of_period_means(s, "quarter", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-rank_ic"></a>
## rank_ic — Mean Rank IC

rank_ic has exactly ONE meaning: the time-mean of daily Spearman rank IC between factor values and labels (canonical alias ic.rank.mean)

- 版本：`4.0.0`；状态：`stable`；层级：`core`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`drop_pair`；数值政策：`finite`；注册最低期数：`None`。
- 别名： `ic.rank.mean` 。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic.compute_daily_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_rank_ic_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, label_bundle: quant_evaluator.contracts.label_bundle.LabelBundle, min_periods: int = 1, min_assets: int = 20) -> numpy.ndarray
```

Return the time-mean daily Spearman (rank) IC per factor, shape (F,).

``rank_ic`` has exactly ONE meaning in this package: the time-average of
daily Spearman rank IC between factor values and labels.

### 精确计算公式（实际实现）

```python
def compute_rank_ic_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_periods: int = 1,
    min_assets: int = 20,
) -> np.ndarray:
    """Return the time-mean daily Spearman (rank) IC per factor, shape (F,).

    ``rank_ic`` has exactly ONE meaning in this package: the time-average of
    daily Spearman rank IC between factor values and labels.
    """
    ic_series, _ = compute_daily_ic(
        factor_batch, label_bundle, method="spearman", min_assets=min_assets
    )
    mean, _ = compute_mean_ic(ic_series, min_periods=min_periods)
    return mean
```

<a id="metric-rank_ic_cross_section"></a>
## rank_ic_cross_section — rank_ic_cross_section

Rank IC computed cross-sectionally for each date

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`drop_pair`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic.compute_daily_ic`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_rank_ic_decay(ic_series: 'np.ndarray', horizons=(1, 5, 10, 20), min_periods: 'int' = 20) -> 'np.ndarray'
```

Mean IC autocorrelation across the decay horizons {1,5,10,20}, (F,).

A single scalar per factor summarising how quickly the IC series loses
autocorrelation (decays) at increasing lags.

### 精确计算公式（实际实现）

```python
def compute_rank_ic_decay(
    ic_series: np.ndarray,
    horizons=(1, 5, 10, 20),
    min_periods: int = 20,
) -> np.ndarray:
    """Mean IC autocorrelation across the decay horizons {1,5,10,20}, (F,).

    A single scalar per factor summarising how quickly the IC series loses
    autocorrelation (decays) at increasing lags.
    """
    s = _as_series(ic_series)
    acfs = [_autocorr_lag(s, h) for h in horizons]
    with np.errstate(invalid="ignore"):
        out = np.nanmean(np.stack(acfs, axis=0), axis=0)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-rank_ic_positive_ratio"></a>
## rank_ic_positive_ratio — Rank IC Positive Ratio

Fraction of finite daily rank-IC values that are strictly positive, per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_rank_ic_positive_ratio`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_rank_ic_positive_ratio(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

Fraction of finite daily rank-IC values that are strictly positive, (F,).

### 精确计算公式（实际实现）

```python
def compute_rank_ic_positive_ratio(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Fraction of finite daily rank-IC values that are strictly positive, (F,)."""
    return compute_ic_positive_ratio(ic_series, min_periods=min_periods)
```

<a id="metric-rank_ic_series"></a>
## rank_ic_series — Daily Rank IC Series

Daily Spearman rank IC per factor over time, shape (T, F) (canonical alias ic.rank.daily)

- 版本：`4.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名： `ic.rank.daily` 。
- 增量模式：`APPEND_EXACT`；注册实现定位：`rank_ic_series`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_rank_ic_series_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, label_bundle: quant_evaluator.contracts.label_bundle.LabelBundle, min_assets: int = 20) -> numpy.ndarray
```

Return the daily Spearman (rank) IC series per factor, shape (T, F).

### 精确计算公式（实际实现）

```python
def compute_rank_ic_series_value(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 20,
) -> np.ndarray:
    """Return the daily Spearman (rank) IC series per factor, shape (T, F)."""
    ic_series, _ = compute_daily_ic(
        factor_batch, label_bundle, method="spearman", min_assets=min_assets
    )
    return ic_series
```

<a id="metric-rank_ic_time_series"></a>
## rank_ic_time_series — rank_ic_time_series

Rank IC computed per time slice, returned as a time series

- 版本：`3.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`drop_pair`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic.compute_daily_ic`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_rank_stability_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, min_periods: int = 20, lag: int = 1, method: str = 'spearman') -> numpy.ndarray
```

Return time-averaged rank stability per factor.

### 精确计算公式（实际实现）

```python
def compute_rank_stability_value(
    factor_batch: FactorBatch,
    min_periods: int = 20,
    lag: int = 1,
    method: str = "spearman",
) -> np.ndarray:
    """Return time-averaged rank stability per factor."""
    return compute_mean_rank_stability(
        np.where(factor_batch.validity, factor_batch.values, np.nan)
        if factor_batch.validity is not None else factor_batch.values,
        lag=lag,
        method=method,
        min_periods=min_periods,
    )
```

<a id="metric-recent_12m_rank_ic"></a>
## recent_12m_rank_ic — Recent 12-Month Rank IC

Mean rank IC over the most recent ~12 months (252 trading days), per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_recent_12m_rank_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_recent_12m_rank_ic(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

Mean rank IC over the most recent ~12 months (252 trading days), (F,).

### 精确计算公式（实际实现）

```python
def compute_recent_12m_rank_ic(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Mean rank IC over the most recent ~12 months (252 trading days), (F,)."""
    s = _as_series(ic_series)
    out = _recent_mean(s, 252)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-recent_3m_rank_ic"></a>
## recent_3m_rank_ic — Recent 3-Month Rank IC

Mean rank IC over the most recent ~3 months (63 trading days), per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_recent_3m_rank_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_recent_3m_rank_ic(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

Mean rank IC over the most recent ~3 months (63 trading days), (F,).

### 精确计算公式（实际实现）

```python
def compute_recent_3m_rank_ic(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Mean rank IC over the most recent ~3 months (63 trading days), (F,)."""
    s = _as_series(ic_series)
    out = _recent_mean(s, 63)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-recent_6m_rank_ic"></a>
## recent_6m_rank_ic — Recent 6-Month Rank IC

Mean rank IC over the most recent ~6 months (126 trading days), per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_recent_6m_rank_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_recent_6m_rank_ic(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

Mean rank IC over the most recent ~6 months (126 trading days), (F,).

### 精确计算公式（实际实现）

```python
def compute_recent_6m_rank_ic(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Mean rank IC over the most recent ~6 months (126 trading days), (F,)."""
    s = _as_series(ic_series)
    out = _recent_mean(s, 126)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-recent_degradation_score"></a>
## recent_degradation_score — Recent Degradation Score

Recent degradation: (full mean IC - recent mean IC) / full std, per factor. Positive values indicate the factor's recent IC is weaker than its historical average (degradation) (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`zscore`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_recent_degradation_score`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_recent_degradation_score(ic_series: 'np.ndarray', recent_days: 'int' = 63, min_periods: 'int' = 20) -> 'np.ndarray'
```

Recent degradation: (recent mean IC - full mean IC) / full std, (F,).

Negative values indicate the factor's recent IC is weaker than its
historical average (degradation).  This is the negative of the
``ic_recent_vs_history_delta`` predictive metric.

### 精确计算公式（实际实现）

```python
def compute_recent_degradation_score(
    ic_series: np.ndarray,
    recent_days: int = 63,
    min_periods: int = 20,
) -> np.ndarray:
    """Recent degradation: (recent mean IC - full mean IC) / full std, (F,).

    Negative values indicate the factor's recent IC is weaker than its
    historical average (degradation).  This is the negative of the
    ``ic_recent_vs_history_delta`` predictive metric.
    """
    s = _as_series(ic_series)
    valid = np.isfinite(s)
    n = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore"):
        hist_mean = np.nanmean(s, axis=0)
        hist_std = np.nanstd(s, axis=0, ddof=1)
        recent_mean = np.nanmean(s[-recent_days:, :], axis=0)
    score = (hist_mean - recent_mean) / np.maximum(hist_std, 1e-12)
    score = np.where(hist_std > 1e-12, score, np.nan)
    return np.where(n >= min_periods, score, np.nan)
```

<a id="metric-regime_conditional_ic"></a>
## regime_conditional_ic — Regime Conditional IC

Mean IC in the late (recent) regime, per factor. Measures the factor's current predictive power (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_regime_conditional_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_regime_conditional_ic(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

Mean IC in the late (recent) regime, (F,).

### 精确计算公式（实际实现）

```python
def compute_regime_conditional_ic(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Mean IC in the late (recent) regime, (F,)."""
    s = _as_series(ic_series)
    _, late = _regime_split(s)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(late, axis=0)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-regime_dispersion"></a>
## regime_dispersion — Regime Dispersion

Absolute difference between early and late regime mean IC, per factor. Larger values indicate the factor's predictive power changed across regimes (instability) (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_regime_dispersion`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_regime_dispersion(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

Absolute difference between early and late regime mean IC, (F,).

Larger values indicate the factor's predictive power changed across
regimes (instability).

### 精确计算公式（实际实现）

```python
def compute_regime_dispersion(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Absolute difference between early and late regime mean IC, (F,).

    Larger values indicate the factor's predictive power changed across
    regimes (instability).
    """
    s = _as_series(ic_series)
    early, late = _regime_split(s)
    with np.errstate(invalid="ignore"):
        e = np.nanmean(early, axis=0)
        l = np.nanmean(late, axis=0)
    out = np.abs(e - l)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-regime_sign_consistency"></a>
## regime_sign_consistency — Regime Sign Consistency

1.0 if early and late regime mean IC share the same sign, else 0.0, per factor (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_regime_sign_consistency`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_regime_sign_consistency(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

1.0 if early and late regime mean IC share the same sign, else 0.0, (F,).

### 精确计算公式（实际实现）

```python
def compute_regime_sign_consistency(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """1.0 if early and late regime mean IC share the same sign, else 0.0, (F,)."""
    s = _as_series(ic_series)
    early, late = _regime_split(s)
    with np.errstate(invalid="ignore"):
        e = np.nanmean(early, axis=0)
        l = np.nanmean(late, axis=0)
    out = np.where(np.sign(e) == np.sign(l), 1.0, 0.0)
    out = np.where(np.isfinite(e) & np.isfinite(l), out, np.nan)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-regime_worst_ic"></a>
## regime_worst_ic — Regime Worst IC

Minimum of the early/late regime mean IC, per factor. The weaker of the two regime averages (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_regime_worst_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_regime_worst_ic(ic_series: 'np.ndarray', min_periods: 'int' = 20) -> 'np.ndarray'
```

Minimum of the early/late regime mean IC, (F,).

### 精确计算公式（实际实现）

```python
def compute_regime_worst_ic(ic_series: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Minimum of the early/late regime mean IC, (F,)."""
    s = _as_series(ic_series)
    early, late = _regime_split(s)
    with np.errstate(invalid="ignore"):
        e = np.nanmean(early, axis=0)
        l = np.nanmean(late, axis=0)
    out = np.minimum(e, l)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-relative_max_drawdown"></a>
## relative_max_drawdown — Relative Max Drawdown

Benchmark/invested-capital evidence from an explicitly bound execution trajectory leg

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.long_only.compute_relative_max_drawdown`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.long_only.compute_relative_max_drawdown(returns, min_periods: 'int' = 1)
```

Positive drawdown magnitude from relative-wealth return increments.

### 精确计算公式（实际实现）

```python
def compute_relative_max_drawdown(returns, min_periods: int = 1):
    """Positive drawdown magnitude from relative-wealth return increments."""
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 1:
        raise ValueError("min_periods must be an integer >= 1")
    x, squeeze = _matrix(returns); out = np.full(x.shape[1], np.nan)
    for f in range(x.shape[1]):
        v = x[:, f]
        if np.count_nonzero(np.isfinite(v)) >= min_periods:
            out[f] = np.asarray(compute_maximum_drawdown(v, missing_return_policy="unknown")[0]).item()
    return out[0] if squeeze else out
```

<a id="metric-residual_rank_ic"></a>
## residual_rank_ic — Residual Rank IC

Named alias of neutralized_rank_ic (residual version of the rank IC, residualized against style exposures). Same kernel, registered under its own id for downstream reporting.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, forward_returns, exposure_panel`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_residual_rank_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_residual_rank_ic(factor_values: 'np.ndarray', forward_returns: 'np.ndarray', panel: 'ExposurePanel', min_obs: 'int' = 10) -> 'float'
```

Named alias for ``compute_neutralized_rank_ic`` (residual_rank_ic id).

### 精确计算公式（实际实现）

```python
def compute_residual_rank_ic(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    panel: ExposurePanel,
    min_obs: int = 10,
) -> float:
    """Named alias for ``compute_neutralized_rank_ic`` (residual_rank_ic id)."""
    return compute_neutralized_rank_ic(
        factor_values, forward_returns, panel, min_obs=min_obs
    )
```

<a id="metric-return_coverage"></a>
## return_coverage — return_coverage

Fraction of universe with non-null forward returns

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quality.compute_coverage`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_return_skew(returns: 'np.ndarray', min_periods: 'int' = 20, *, bias: 'bool' = False) -> 'float'
```

Sample skewness of the return series (population ``scipy.stats.skew``
convention, bias=False; for the tail-risk registry family the existing
``metrics/distribution.compute_skewness`` is the same statistic).

NaN when fewer than ``min_periods`` finite returns or when std is 0.

### 精确计算公式（实际实现）

```python
def compute_return_skew(returns: np.ndarray, min_periods: int = 20, *, bias: bool = False) -> float:
    """Sample skewness of the return series (population ``scipy.stats.skew``
    convention, bias=False; for the tail-risk registry family the existing
    ``metrics/distribution.compute_skewness`` is the same statistic).

    NaN when fewer than ``min_periods`` finite returns or when std is 0.
    """
    ret = _as_1d(returns)
    ret = ret[np.isfinite(ret)]
    if ret.size < max(min_periods, 3):
        return np.nan
    mu = np.mean(ret)
    std = np.std(ret, ddof=0)
    if std <= EPS:
        return np.nan
    skew = float(np.mean(((ret - mu) / std) ** 3))
    return skew if bias else float(np.sqrt(ret.size * (ret.size - 1)) / (ret.size - 2) * skew)
```

<a id="metric-rolling_1y_sharpe_min"></a>
## rolling_1y_sharpe_min — Rolling 1Y Sharpe Min

Minimum rolling-252-period annualized Sharpe of the probe daily PnL series (worst observed 1y window). NaN when the series is shorter than one window.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`60`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail(returns: 'np.ndarray', window: 'int' = 252, quantile: 'float' = 0.1, min_periods: 'int' = 30, periods_per_year: 'int' = 252) -> 'float'
```

Rolling-window annualized Sharpe tail statistic.

Computes the rolling-window Sharpe (``quant_evaluator.metrics.
portfolio_stats.compute_sharpe_ratio``) over every aligned window with at
least ``min_periods`` finite returns, then returns the requested quantile:

- ``quantile=0.0``   -> ``rolling_1y_sharpe_min`` (minimum attained);
- ``quantile=0.10``  -> ``rolling_1y_sharpe_q10``.

NaN when there are no valid rolling windows.

### 精确计算公式（实际实现）

```python
def compute_rolling_sharpe_tail(
    returns: np.ndarray,
    window: int = _PERIODS_PER_YEAR,
    quantile: float = 0.10,
    min_periods: int = 30,
    periods_per_year: int = _PERIODS_PER_YEAR,
) -> float:
    """Rolling-window annualized Sharpe tail statistic.

    Computes the rolling-window Sharpe (``quant_evaluator.metrics.
    portfolio_stats.compute_sharpe_ratio``) over every aligned window with at
    least ``min_periods`` finite returns, then returns the requested quantile:

    - ``quantile=0.0``   -> ``rolling_1y_sharpe_min`` (minimum attained);
    - ``quantile=0.10``  -> ``rolling_1y_sharpe_q10``.

    NaN when there are no valid rolling windows.
    """
    from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio

    ret = _as_1d(returns)
    if ret.size < window:
        return np.nan
    roll: list[float] = []
    for t in range(window - 1, ret.size):
        seg = ret[t - window + 1 : t + 1]
        if np.sum(np.isfinite(seg)) >= min_periods:
            val = float(
                compute_sharpe_ratio(
                    seg,
                    risk_free_rate=0.0,
                    periods_per_year=periods_per_year,
                    min_periods=min_periods,
                )
            )
            if np.isfinite(val):
                roll.append(val)
    if not roll:
        return np.nan
    if quantile == 0.0:
        return float(min(roll))
    return float(np.quantile(roll, quantile))
```

<a id="metric-rolling_1y_sharpe_q10"></a>
## rolling_1y_sharpe_q10 — Rolling 1Y Sharpe Q10

10th-percentile rolling-252-period annualized Sharpe of the probe daily PnL series (tail floor of the 1y rolling Sharpe distribution). NaN when the series is shorter than one window.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`60`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_rolling_sharpe_tail(returns: 'np.ndarray', window: 'int' = 252, quantile: 'float' = 0.1, min_periods: 'int' = 30, periods_per_year: 'int' = 252) -> 'float'
```

Rolling-window annualized Sharpe tail statistic.

Computes the rolling-window Sharpe (``quant_evaluator.metrics.
portfolio_stats.compute_sharpe_ratio``) over every aligned window with at
least ``min_periods`` finite returns, then returns the requested quantile:

- ``quantile=0.0``   -> ``rolling_1y_sharpe_min`` (minimum attained);
- ``quantile=0.10``  -> ``rolling_1y_sharpe_q10``.

NaN when there are no valid rolling windows.

### 精确计算公式（实际实现）

```python
def compute_rolling_sharpe_tail(
    returns: np.ndarray,
    window: int = _PERIODS_PER_YEAR,
    quantile: float = 0.10,
    min_periods: int = 30,
    periods_per_year: int = _PERIODS_PER_YEAR,
) -> float:
    """Rolling-window annualized Sharpe tail statistic.

    Computes the rolling-window Sharpe (``quant_evaluator.metrics.
    portfolio_stats.compute_sharpe_ratio``) over every aligned window with at
    least ``min_periods`` finite returns, then returns the requested quantile:

    - ``quantile=0.0``   -> ``rolling_1y_sharpe_min`` (minimum attained);
    - ``quantile=0.10``  -> ``rolling_1y_sharpe_q10``.

    NaN when there are no valid rolling windows.
    """
    from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio

    ret = _as_1d(returns)
    if ret.size < window:
        return np.nan
    roll: list[float] = []
    for t in range(window - 1, ret.size):
        seg = ret[t - window + 1 : t + 1]
        if np.sum(np.isfinite(seg)) >= min_periods:
            val = float(
                compute_sharpe_ratio(
                    seg,
                    risk_free_rate=0.0,
                    periods_per_year=periods_per_year,
                    min_periods=min_periods,
                )
            )
            if np.isfinite(val):
                roll.append(val)
    if not roll:
        return np.nan
    if quantile == 0.0:
        return float(min(roll))
    return float(np.quantile(roll, quantile))
```

<a id="metric-rolling_ic"></a>
## rolling_ic — rolling_ic

Rolling window IC values over time

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`series`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic_summary.compute_rolling_ic_stats`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_rolling_ic_drawdown(ic_series: 'np.ndarray', window: 'int' = 60, min_periods: 'int' = 20) -> 'np.ndarray'
```

Mean of the rolling-window IC drawdown (peak-to-trough), (F,).

Computed on the cumulative sum of the rolling mean IC within each window.

### 精确计算公式（实际实现）

```python
def compute_rolling_ic_drawdown(
    ic_series: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> np.ndarray:
    """Mean of the rolling-window IC drawdown (peak-to-trough), (F,).

    Computed on the cumulative sum of the rolling mean IC within each window.
    """
    s = _as_series(ic_series)
    T, F = s.shape
    finite = np.isfinite(s)
    x = np.where(finite, s, 0.0)
    pref = np.concatenate([np.zeros((1, F)), np.cumsum(x, axis=0)], axis=0)
    fpref = np.concatenate([np.zeros((1, F)), np.cumsum(finite, axis=0)], axis=0)
    ends = np.arange(1, T + 1)
    start = np.maximum(ends - window, 0)
    cnt = fpref[ends] - fpref[start]
    ssum = pref[ends] - pref[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ssum / np.maximum(cnt, 1.0)
    mean = np.where(cnt >= min_periods, mean, np.nan)
    # cumulative sum of the rolling mean (NaN -> 0), then peak-to-trough
    cum = np.cumsum(np.where(np.isfinite(mean), mean, 0.0), axis=0)
    running_max = np.maximum.accumulate(cum, axis=0)
    dd = cum - running_max
    with np.errstate(invalid="ignore"):
        return np.nanmean(dd, axis=0)
```

<a id="metric-rolling_ic_volatility"></a>
## rolling_ic_volatility — Rolling IC Volatility

Time-mean of the rolling-window IC standard deviation, per factor. Lower values indicate a more stable IC series (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_rolling_ic_volatility`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_rolling_ic_volatility(ic_series: 'np.ndarray', window: 'int' = 60, min_periods: 'int' = 20) -> 'np.ndarray'
```

Time-mean of the rolling-window IC standard deviation, (F,).

Lower values indicate a more stable IC series.

### 精确计算公式（实际实现）

```python
def compute_rolling_ic_volatility(
    ic_series: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> np.ndarray:
    """Time-mean of the rolling-window IC standard deviation, (F,).

    Lower values indicate a more stable IC series.
    """
    s = _as_series(ic_series)
    T, F = s.shape
    finite = np.isfinite(s)
    x = np.where(finite, s, 0.0)
    pref = np.concatenate([np.zeros((1, F)), np.cumsum(x, axis=0)], axis=0)
    fpref = np.concatenate([np.zeros((1, F)), np.cumsum(finite, axis=0)], axis=0)
    pref_sq = np.concatenate([np.zeros((1, F)), np.cumsum(x * x, axis=0)], axis=0)
    ends = np.arange(1, T + 1)
    start = np.maximum(ends - window, 0)
    cnt = fpref[ends] - fpref[start]
    ssum = pref[ends] - pref[start]
    ssq = pref_sq[ends] - pref_sq[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ssum / np.maximum(cnt, 1.0)
        var = ssq - cnt * mean * mean
        std = np.sqrt(np.maximum(var, 0.0) / np.maximum(cnt - 1, 1.0))
    std = np.where(cnt >= min_periods, std, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(std, axis=0)
```

<a id="metric-rolling_rank_ic_ir"></a>
## rolling_rank_ic_ir — Rolling Rank IC IR

Time-mean of the rolling-window IC information ratio, per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_rolling_rank_ic_ir`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_rolling_rank_ic_ir(ic_series: 'np.ndarray', window: 'int' = 60, min_periods: 'int' = 20) -> 'np.ndarray'
```

Time-mean of the rolling-window IC information ratio, (F,).

### 精确计算公式（实际实现）

```python
def compute_rolling_rank_ic_ir(
    ic_series: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> np.ndarray:
    """Time-mean of the rolling-window IC information ratio, (F,)."""
    s = _as_series(ic_series)
    _, ir = _rolling_mean_ir(s, window, min_periods)
    with np.errstate(invalid="ignore"):
        return np.nanmean(ir, axis=0)
```

<a id="metric-rolling_rank_ic_mean"></a>
## rolling_rank_ic_mean — Rolling Rank IC Mean

Time-mean of the rolling-window mean rank IC, per factor. A smoother estimate of average predictive power (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_rolling_rank_ic_mean`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_rolling_rank_ic_mean(ic_series: 'np.ndarray', window: 'int' = 60, min_periods: 'int' = 20) -> 'np.ndarray'
```

Time-mean of the rolling-window mean rank IC, (F,).

### 精确计算公式（实际实现）

```python
def compute_rolling_rank_ic_mean(
    ic_series: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> np.ndarray:
    """Time-mean of the rolling-window mean rank IC, (F,)."""
    s = _as_series(ic_series)
    mean, _ = _rolling_mean_ir(s, window, min_periods)
    with np.errstate(invalid="ignore"):
        return np.nanmean(mean, axis=0)
```

<a id="metric-shape_bootstrap_confidence"></a>
## shape_bootstrap_confidence — Shape Bootstrap Rank Agreement (legacy ID)

Descriptive moving-block bootstrap RANK AGREEMENT, not U-shape probability, per factor: fraction of window-resamples whose order reproduces the overall mean profile order (W >= 3). 1 = shape ordering reproduced in every resample. Deterministic (seeded). NaN for single-window.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_shape_bootstrap_confidence`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_shape_bootstrap_confidence(qr: 'np.ndarray', block_length: 'int' = 2, resamples: 'int' = 100, random_seed: 'int' = 0, agreement_threshold: 'float' = 0.5) -> 'np.ndarray'
```

Descriptive moving-block bootstrap rank-agreement frequency, not U probability.

For a 3-D ``(W, n_quantiles, F)`` input: fraction of bootstrap resamples
(across the W windows) whose quantile-rank order (Spearman of the
profile) reproduces the overall mean profile's rank order.  ``1`` = the
shape's ordering is reproduced in every resample (high confidence),
~0.5 = noisy.  Cost is CHEAP (W is small).  Deterministic via
``random_seed``.  NaN for single-profile input.

### 精确计算公式（实际实现）

```python
def compute_shape_bootstrap_confidence(qr: np.ndarray, block_length: int = 2,
                                      resamples: int = 100, random_seed: int = 0,
                                      agreement_threshold: float = .5) -> np.ndarray:
    """Descriptive moving-block bootstrap rank-agreement frequency, not U probability.

    For a 3-D ``(W, n_quantiles, F)`` input: fraction of bootstrap resamples
    (across the W windows) whose quantile-rank order (Spearman of the
    profile) reproduces the overall mean profile's rank order.  ``1`` = the
    shape's ordering is reproduced in every resample (high confidence),
    ~0.5 = noisy.  Cost is CHEAP (W is small).  Deterministic via
    ``random_seed``.  NaN for single-profile input.
    """
    m = np.asarray(qr, dtype=np.float64)
    windows, multi = _windows_or_single(m)
    nw = windows.shape[0]
    if isinstance(block_length, bool) or not isinstance(block_length, (int, np.integer)) or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not np.isfinite(agreement_threshold) or not -1 <= agreement_threshold <= 1:
        raise ValueError("agreement_threshold must be in [-1,1]")
    rng = np.random.default_rng(random_seed)
    if nw < 3:
        return np.full(windows.shape[2], np.nan, dtype=np.float64)
    if block_length > nw:
        raise ValueError("block_length cannot exceed window count")
    from scipy.stats import rankdata
    # One request-level draw schedule for every factor: adding/reordering other
    # factors cannot change a factor's evidence through RNG consumption.
    starts = rng.integers(0, nw-block_length+1, size=(resamples,int(np.ceil(nw/block_length))))
    bootstrap_indices = (starts[:,:,None]+np.arange(block_length)).reshape(resamples,-1)[:,:nw]
    F = windows.shape[2]
    out = np.full(F, np.nan)
    mean_profile = _per_window_profile(windows)
    for f in range(F):
        mp = mean_profile[:, f]
        if np.sum(np.isfinite(mp)) < 3:
            continue
        agreed = 0
        drawn = 0
        for idx in bootstrap_indices:
            sample = windows[idx, :, f]
            with np.errstate(invalid="ignore"):
                sp = np.nanmean(sample, axis=0)
            finite = np.isfinite(sp) & np.isfinite(mp)
            if np.sum(finite) < 3:
                continue
            if np.ptp(sp[finite]) == 0.0 or np.ptp(mp[finite]) == 0.0:
                continue
            r = np.corrcoef(
                rankdata(sp[finite],method="average"),
                rankdata(mp[finite],method="average"),
            )[0, 1]
            drawn += 1
            if np.isfinite(r) and r >= agreement_threshold:
                agreed += 1
        if drawn:
            out[f] = agreed / drawn
    return out
```

<a id="metric-shape_bootstrap_rank_agreement"></a>
## shape_bootstrap_rank_agreement — Block Bootstrap Rank Agreement

Descriptive moving-block resampling agreement with the observed mean rank profile; not U-family success probability

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_shape_bootstrap_confidence`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_shape_bootstrap_confidence(qr: 'np.ndarray', block_length: 'int' = 2, resamples: 'int' = 100, random_seed: 'int' = 0, agreement_threshold: 'float' = 0.5) -> 'np.ndarray'
```

Descriptive moving-block bootstrap rank-agreement frequency, not U probability.

For a 3-D ``(W, n_quantiles, F)`` input: fraction of bootstrap resamples
(across the W windows) whose quantile-rank order (Spearman of the
profile) reproduces the overall mean profile's rank order.  ``1`` = the
shape's ordering is reproduced in every resample (high confidence),
~0.5 = noisy.  Cost is CHEAP (W is small).  Deterministic via
``random_seed``.  NaN for single-profile input.

### 精确计算公式（实际实现）

```python
def compute_shape_bootstrap_confidence(qr: np.ndarray, block_length: int = 2,
                                      resamples: int = 100, random_seed: int = 0,
                                      agreement_threshold: float = .5) -> np.ndarray:
    """Descriptive moving-block bootstrap rank-agreement frequency, not U probability.

    For a 3-D ``(W, n_quantiles, F)`` input: fraction of bootstrap resamples
    (across the W windows) whose quantile-rank order (Spearman of the
    profile) reproduces the overall mean profile's rank order.  ``1`` = the
    shape's ordering is reproduced in every resample (high confidence),
    ~0.5 = noisy.  Cost is CHEAP (W is small).  Deterministic via
    ``random_seed``.  NaN for single-profile input.
    """
    m = np.asarray(qr, dtype=np.float64)
    windows, multi = _windows_or_single(m)
    nw = windows.shape[0]
    if isinstance(block_length, bool) or not isinstance(block_length, (int, np.integer)) or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not np.isfinite(agreement_threshold) or not -1 <= agreement_threshold <= 1:
        raise ValueError("agreement_threshold must be in [-1,1]")
    rng = np.random.default_rng(random_seed)
    if nw < 3:
        return np.full(windows.shape[2], np.nan, dtype=np.float64)
    if block_length > nw:
        raise ValueError("block_length cannot exceed window count")
    from scipy.stats import rankdata
    # One request-level draw schedule for every factor: adding/reordering other
    # factors cannot change a factor's evidence through RNG consumption.
    starts = rng.integers(0, nw-block_length+1, size=(resamples,int(np.ceil(nw/block_length))))
    bootstrap_indices = (starts[:,:,None]+np.arange(block_length)).reshape(resamples,-1)[:,:nw]
    F = windows.shape[2]
    out = np.full(F, np.nan)
    mean_profile = _per_window_profile(windows)
    for f in range(F):
        mp = mean_profile[:, f]
        if np.sum(np.isfinite(mp)) < 3:
            continue
        agreed = 0
        drawn = 0
        for idx in bootstrap_indices:
            sample = windows[idx, :, f]
            with np.errstate(invalid="ignore"):
                sp = np.nanmean(sample, axis=0)
            finite = np.isfinite(sp) & np.isfinite(mp)
            if np.sum(finite) < 3:
                continue
            if np.ptp(sp[finite]) == 0.0 or np.ptp(mp[finite]) == 0.0:
                continue
            r = np.corrcoef(
                rankdata(sp[finite],method="average"),
                rankdata(mp[finite],method="average"),
            )[0, 1]
            drawn += 1
            if np.isfinite(r) and r >= agreement_threshold:
                agreed += 1
        if drawn:
            out[f] = agreed / drawn
    return out
```

<a id="metric-shape_regime_stability"></a>
## shape_regime_stability — Shape Regime Stability

Regime version of shape stability: inverse-Fisher correlation of CONSECUTIVE window profiles per factor (W >= 3 windows). Drops a single common shape that is stable overall but regime-uncorrelated. NaN when fewer than 3 windows.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_shape_regime_stability`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_shape_regime_stability(qr: 'np.ndarray') -> 'np.ndarray'
```

Shape stability across windows, aggregated within/against the mean, (F,).

## This function serves a separate metric id from "shape_stability"
## (id ``shape_regime_stability``, plan §13.4 "shape appears across
## multiple windows").  Implementation note: this metric is currently
## registered as ``UNSUPPORTED`` configuration in the presence of a
## single profile (like ``shape_stability``).  The kernel contract below
## IS the CPU reference for the multi-window case and matches
## ``compute_shape_stability``'s value distribution so both resolve the
## same "shape across windows" question with different cost/eyeballing.

For a 3-D ``(W, n_quantiles, F)`` input: mean pairwise correlation of
consecutive windows (regime version of shape stability) — measures the
coherence of the profile as the regime rolls, damping a single-common
shape that is stable overall but regime-uncorrelated.  NaN for a
single-profile input.

### 精确计算公式（实际实现）

```python
def compute_shape_regime_stability(qr: np.ndarray) -> np.ndarray:
    """Shape stability across windows, aggregated within/against the mean, (F,).

    ## This function serves a separate metric id from "shape_stability"
    ## (id ``shape_regime_stability``, plan §13.4 "shape appears across
    ## multiple windows").  Implementation note: this metric is currently
    ## registered as ``UNSUPPORTED`` configuration in the presence of a
    ## single profile (like ``shape_stability``).  The kernel contract below
    ## IS the CPU reference for the multi-window case and matches
    ## ``compute_shape_stability``'s value distribution so both resolve the
    ## same "shape across windows" question with different cost/eyeballing.

    For a 3-D ``(W, n_quantiles, F)`` input: mean pairwise correlation of
    consecutive windows (regime version of shape stability) — measures the
    coherence of the profile as the regime rolls, damping a single-common
    shape that is stable overall but regime-uncorrelated.  NaN for a
    single-profile input.
    """
    m = np.asarray(qr, dtype=np.float64)
    windows, multi = _windows_or_single(m)
    nw = windows.shape[0]
    if nw < 3:
        return np.full(windows.shape[2], np.nan, dtype=np.float64)
    F = windows.shape[2]
    out = np.full(F, np.nan)
    for f in range(F):
        corrs = []
        for w in range(nw - 1):
            a = windows[w, :, f]
            b = windows[w + 1, :, f]
            finite = np.isfinite(a) & np.isfinite(b)
            if np.sum(finite) < 3:
                continue
            if np.ptp(a[finite]) == 0.0 or np.ptp(b[finite]) == 0.0:
                continue
            r = np.corrcoef(a[finite], b[finite])[0, 1]
            if np.isfinite(r):
                corrs.append(r)
        if corrs:
            out[f] = float(np.tanh(np.mean(np.arctanh(np.clip(corrs, -1 + 1e-9, 1 - 1e-9)))))
    return out
```

<a id="metric-shape_stability"></a>
## shape_stability — Shape Stability

Inverse-Fisher aggregated window-vs-leave-one-out correlation of the quantile profile across W windows per factor. Consumes a (W, n_quantiles, F) windowed profile panel. 1 = identical shape in every window. NaN for a single-window profile (stability is undefined - missing evidence, never 0/1).

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_shape_stability`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_shape_stability(qr: 'np.ndarray') -> 'np.ndarray'
```

Quantile-shape stability across windows per factor, (F,).

For a 3-D ``(W, n_quantiles, F)`` input: mean Fisher-z-corrected
correlation of each window's profile against the leave-one-window-out
mean profile, inverse transformed to correlation units. ``1`` = stable,
low = shape churns.  For a 2-D single profile input this returns NaN
with an honest ``unsupported`` (single-profile stability is undefined —
there is no second observation), never 0 or 1.

### 精确计算公式（实际实现）

```python
def compute_shape_stability(qr: np.ndarray) -> np.ndarray:
    """Quantile-shape stability across windows per factor, (F,).

    For a 3-D ``(W, n_quantiles, F)`` input: mean Fisher-z-corrected
    correlation of each window's profile against the leave-one-window-out
    mean profile, inverse transformed to correlation units. ``1`` = stable,
    low = shape churns.  For a 2-D single profile input this returns NaN
    with an honest ``unsupported`` (single-profile stability is undefined —
    there is no second observation), never 0 or 1.
    """
    m = np.asarray(qr, dtype=np.float64)
    windows, multi = _windows_or_single(m)
    nw = windows.shape[0]
    if nw < 2:
        # Single profile: stability is a missing-evidence scalar.
        return np.full(windows.shape[2], np.nan, dtype=np.float64)
    F = windows.shape[2]
    out = np.full(F, np.nan)
    for f in range(F):
        corrs = []
        for w in range(nw):
            mp = _per_window_profile(np.delete(windows, w, axis=0))[:, f]
            wp = windows[w, :, f]
            finite = np.isfinite(wp) & np.isfinite(mp)
            if np.sum(finite) < 3:
                continue
            if np.ptp(wp[finite]) == 0.0 or np.ptp(mp[finite]) == 0.0:
                continue
            r = np.corrcoef(wp[finite], mp[finite])[0, 1]
            if np.isfinite(r):
                corrs.append(r)
        if corrs:
            out[f] = float(np.tanh(np.mean(np.arctanh(np.clip(corrs, -1 + 1e-9, 1 - 1e-9)))))
    return out
```

<a id="metric-sharpe_ratio"></a>
## sharpe_ratio — sharpe_ratio

Annualized Sharpe ratio of a return series per factor: mean(excess return) / std(excess return, ddof=1) * sqrt(periods_per_year), with excess return = return - risk_free_rate / periods_per_year. NaN when fewer than min_periods finite returns or when the return standard deviation is not positive-finite. period: daily (252).

- 版本：`1.0.1`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`ratio`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_sharpe_ratio`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.portfolio_stats.compute_sharpe_ratio(returns: numpy.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252, min_periods: int = 20) -> numpy.ndarray
```

Compute annualized Sharpe ratio.

Args:
    returns: Return series (T,) or (T, F)
    risk_free_rate: Annual risk-free rate (default 0.0)
    periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
    min_periods: Minimum periods required

Returns:
    Sharpe ratio, scalar or shape (F,)

### 精确计算公式（实际实现）

```python
def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Return series (T,) or (T, F)
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        min_periods: Minimum periods required

    Returns:
        Sharpe ratio, scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape

    if T == 0:
        return float("nan") if squeeze else np.full(F, np.nan, dtype=np.float64)
    sharpe = np.full(F, np.nan)

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]

        # Compute excess returns
        rf_per_period = risk_free_rate / periods_per_year
        excess_ret = ret_valid - rf_per_period

        mean_excess = np.mean(excess_ret)
        std_excess = np.std(excess_ret, ddof=1)

        if not np.isfinite(std_excess) or std_excess <= 1e-10:
            continue

        # Annualize
        sharpe[f] = mean_excess / std_excess * np.sqrt(periods_per_year)

    return sharpe[0] if squeeze else sharpe
```

<a id="metric-sidak_correction"></a>
## sidak_correction — Sidak Correction

Sidak multiple-testing correction: adjusted p = 1 - (1 - p)^n. Assumes independence, slightly less conservative than Bonferroni (spec §34).

- 版本：`4.0.0`；状态：`stable`；层级：`research`。
- 输入依赖：`p_values`。
- 输出：`scalar`；单位：`pvalue`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.multiple_testing.sidak_correction`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.multiple_testing.sidak_correction(p_values: numpy.ndarray, alpha: float = 0.05) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Apply Šidák correction for multiple testing.

Assumes independence, slightly less conservative than Bonferroni.

Args:
    p_values: Array of p-values, any shape. Non-finite entries are
        treated as missing tests (see module docstring); finite
        entries must lie in [0, 1].
    alpha: Family-wise error rate, strictly inside (0, 1)

Returns:
    (adjusted_p_values, reject_mask)
    adjusted_p_values: Šidák-adjusted p-values
    reject_mask: Boolean mask where null hypothesis is rejected

Raises:
    ValueError: If ``p_values`` is empty, contains finite values
        outside [0, 1], or ``alpha`` is not strictly inside (0, 1)

### 精确计算公式（实际实现）

```python
def sidak_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Šidák correction for multiple testing.

    Assumes independence, slightly less conservative than Bonferroni.

    Args:
        p_values: Array of p-values, any shape. Non-finite entries are
            treated as missing tests (see module docstring); finite
            entries must lie in [0, 1].
        alpha: Family-wise error rate, strictly inside (0, 1)

    Returns:
        (adjusted_p_values, reject_mask)
        adjusted_p_values: Šidák-adjusted p-values
        reject_mask: Boolean mask where null hypothesis is rejected

    Raises:
        ValueError: If ``p_values`` is empty, contains finite values
            outside [0, 1], or ``alpha`` is not strictly inside (0, 1)
    """
    alpha = _validate_alpha(alpha)
    p_values = _validate_p_values(p_values)

    # Flatten for processing
    original_shape = p_values.shape
    p_flat = p_values.ravel()

    # Count valid p-values
    valid_mask = np.isfinite(p_flat)
    n_tests = np.sum(valid_mask)

    if n_tests == 0:
        return np.full_like(p_values, np.nan), np.zeros_like(p_values, dtype=bool)

    # Adjusted alpha for Šidák
    alpha_sidak = -np.expm1(np.log1p(-alpha) / n_tests)

    # Adjust p-values: 1 - (1 - p)^m
    adjusted = np.full_like(p_flat, np.nan)
    with np.errstate(divide="ignore"):
        adjusted[valid_mask] = -np.expm1(n_tests * np.log1p(-p_flat[valid_mask]))
    adjusted[valid_mask] = np.minimum(adjusted[valid_mask], 1.0)

    # Rejection mask
    reject = np.zeros_like(p_flat, dtype=bool)
    reject[valid_mask] = p_flat[valid_mask] <= alpha_sidak

    return adjusted.reshape(original_shape), reject.reshape(original_shape)
```

<a id="metric-size_exposure"></a>
## size_exposure — Size Exposure

Signed mean size-style exposure of the factor (typed per-style field). NaN when style absent or no finite cells.

- 版本：`4.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_values, exposure_panel`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.exposure_evidence.compute_size_exposure`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_size_exposure(panel: 'FactorLoadingSeries', *, factor_values=None, min_obs=10, weights=None) -> 'float'
```

Signed mean size-style exposure of the factor (one typed field).

### 精确计算公式（实际实现）

```python
def compute_size_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean size-style exposure of the factor (one typed field)."""
    return _select_style(panel, "size",factor_values=factor_values,min_obs=min_obs,weights=weights)
```

<a id="metric-skewness"></a>
## skewness — skewness

Skewness of the return distribution

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`dimensionless`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.distribution.compute_skewness`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.portfolio_stats.compute_sortino_ratio(returns: numpy.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252, min_periods: int = 20, downside_denominator: str = 'negative', mar: float | None = None, annualization: str = 'sqrt_frequency') -> numpy.ndarray
```

Compute Sortino with explicit target, downside denominator and annualization.

``mar`` is a periodic minimum acceptable return; ``risk_free_rate`` is
annual and converted arithmetically when MAR is absent. ``negative``
preserves the historical conditional RMS; ``all`` uses full-sample
semideviation. No observed downside always returns NaN, never a large
finite substitute. ``none`` returns periodic rather than annualized units.

Args:
    returns: Return series (T,) or (T, F)
    risk_free_rate: Annual risk-free rate
    periods_per_year: Number of periods per year
    min_periods: Minimum periods required

Returns:
    Sortino ratio, scalar or shape (F,)

### 精确计算公式（实际实现）

```python
def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: int = 20,
    downside_denominator: str = "negative",
    mar: float | None = None,
    annualization: str = "sqrt_frequency",
) -> np.ndarray:
    """
    Compute Sortino with explicit target, downside denominator and annualization.

    ``mar`` is a periodic minimum acceptable return; ``risk_free_rate`` is
    annual and converted arithmetically when MAR is absent. ``negative``
    preserves the historical conditional RMS; ``all`` uses full-sample
    semideviation. No observed downside always returns NaN, never a large
    finite substitute. ``none`` returns periodic rather than annualized units.

    Args:
        returns: Return series (T,) or (T, F)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        min_periods: Minimum periods required

    Returns:
        Sortino ratio, scalar or shape (F,)
    """
    if downside_denominator not in {"negative", "all"}:
        raise ValueError("downside_denominator must be negative or all")
    if annualization not in {"sqrt_frequency", "none"}:
        raise ValueError("annualization must be sqrt_frequency or none")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive finite")
    if not np.isfinite(risk_free_rate) or (mar is not None and not np.isfinite(mar)):
        raise ValueError("target return must be finite")
    if mar is not None and risk_free_rate != 0:
        raise ValueError("supply periodic mar or annual risk_free_rate, not both")
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    sortino = np.full(F, np.nan)

    rf_per_period = risk_free_rate / periods_per_year if mar is None else mar

    for f in range(F):
        ret_f = returns[:, f]
        valid = np.isfinite(ret_f)
        n_valid = np.sum(valid)

        if n_valid < min_periods:
            continue

        ret_valid = ret_f[valid]
        excess_ret = ret_valid - rf_per_period

        mean_excess = np.mean(excess_ret)

        # Downside deviation (only negative excess returns)
        downside_ret = excess_ret[excess_ret < 0]
        if len(downside_ret) == 0:
            continue

        denominator = len(downside_ret) if downside_denominator == "negative" else n_valid
        downside_std = np.sqrt(np.sum(downside_ret ** 2) / denominator)

        if not np.isfinite(downside_std) or downside_std <= 1e-12:
            continue

        # Annualize
        scale = np.sqrt(periods_per_year) if annualization == "sqrt_frequency" else 1.0
        sortino[f] = mean_excess / downside_std * scale

    return sortino[0] if squeeze else sortino
```

<a id="metric-spearman_ic"></a>
## spearman_ic — spearman_ic

Spearman rank correlation between factor values and forward returns

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`drop_pair`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.ic.compute_daily_ic`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_staleness(factor_batch: 'FactorBatch') -> 'np.ndarray'
```

Mean fraction of assets whose value is unchanged from the prior day, (F,).

A high staleness ratio indicates the factor is slow-moving / sticky.

### 精确计算公式（实际实现）

```python
def compute_staleness(factor_batch: FactorBatch) -> np.ndarray:
    """Mean fraction of assets whose value is unchanged from the prior day, (F,).

    A high staleness ratio indicates the factor is slow-moving / sticky.
    """
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    if T < 2:
        return np.full(F, np.nan)
    prev = values[:-1, :, :]
    curr = values[1:, :, :]
    both = np.isfinite(prev) & np.isfinite(curr)
    unchanged = (prev == curr) & both
    n = np.sum(both, axis=(0, 1))
    same = np.sum(unchanged, axis=(0, 1))
    return same / np.maximum(n, 1)
```

<a id="metric-subsample_stability"></a>
## subsample_stability — Subsample IC Stability

Standard deviation of mean IC across bootstrap subsamples

- 版本：`0.1.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`40`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`subsample_stability`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_subsample_stability_value(ic_series: numpy.ndarray, min_periods: int = 40, num_subsamples: int = 100, subsample_fraction: float = 0.8, random_seed: int = 0) -> numpy.ndarray
```

Return the std of mean IC across bootstrap subsamples per factor.

### 精确计算公式（实际实现）

```python
def compute_subsample_stability_value(
    ic_series: np.ndarray,
    min_periods: int = 40,
    num_subsamples: int = 100,
    subsample_fraction: float = 0.8,
    random_seed: int = 0,
) -> np.ndarray:
    """Return the std of mean IC across bootstrap subsamples per factor."""
    valid_periods = np.sum(np.isfinite(ic_series), axis=0)
    stability = compute_subsample_ic_std(
        ic_series,
        num_subsamples=num_subsamples,
        subsample_fraction=subsample_fraction,
        random_seed=random_seed,
    )
    return np.where(valid_periods >= min_periods, stability, np.nan)
```

<a id="metric-tail_vs_middle_contrast"></a>
## tail_vs_middle_contrast — Tail vs Middle Contrast

Mean |tail returns - middle return| per factor (tails = outer quartiles of the quantile range). High for U/inverted-U profiles, low for flat. Always >= 0. NaN when profile too small or tail/middle returns not finite.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_tail_vs_middle_contrast`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_tail_vs_middle_contrast(qr: 'np.ndarray') -> 'np.ndarray'
```

Tail-versus-middle contrast per factor, (F,).

``mean(|tail returns - middle return|)`` where the middle return is the
median quantile's return and tails are the top and bottom quartiles of
the quantile range.  A U-shaped profile has HIGH contrast (tails away
from the middle); a flat profile has LOW contrast.  Always >= 0; NaN
when the middle/tail quantile returns are not all finite.

### 精确计算公式（实际实现）

```python
def compute_tail_vs_middle_contrast(qr: np.ndarray) -> np.ndarray:
    """Tail-versus-middle contrast per factor, (F,).

    ``mean(|tail returns - middle return|)`` where the middle return is the
    median quantile's return and tails are the top and bottom quartiles of
    the quantile range.  A U-shaped profile has HIGH contrast (tails away
    from the middle); a flat profile has LOW contrast.  Always >= 0; NaN
    when the middle/tail quantile returns are not all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 6:
        return out
    mid = nq // 2
    # Tails: quantile indices strictly BELOW lo and strictly ABOVE hi,
    # defined so that a U (or inverted-U) puts its extremal mass at the
    # edges.  ``mid`` is the interior reference point (median quantile).
    lo = max(1, min(mid - 1, int(round(nq * 0.25))))
    hi = min(nq - 1, max(mid + 1, int(round(nq * 0.75))))
    # Guarantee non-empty tails on both sides of the middle.
    if lo >= mid or hi <= mid:
        lo = mid - 1
        hi = mid + 1
    for f in range(F):
        col = m[:, f]
        if not (np.isfinite(col[mid]) and np.all(np.isfinite(col[:lo]))
                and np.all(np.isfinite(col[hi:]))):
            continue
        tail_vals = np.concatenate([col[:lo], col[hi:]])
        out[f] = float(np.mean(np.abs(tail_vals - col[mid])))
    return out
```

<a id="metric-tie_ratio"></a>
## tie_ratio — Tie Ratio

Mean fraction of finite factor values that are tied with another value, per factor. The complement of the distinct level ratio (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_tie_ratio`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_tie_ratio(factor_batch: 'FactorBatch') -> 'np.ndarray'
```

Mean fraction of finite factor values that are tied with another value, (F,).

Computed as ``1 - distinct_level_ratio`` (the complement of the distinct
level ratio).

### 精确计算公式（实际实现）

```python
def compute_tie_ratio(factor_batch: FactorBatch) -> np.ndarray:
    """Mean fraction of finite factor values that are tied with another value, (F,).

    Computed as ``1 - distinct_level_ratio`` (the complement of the distinct
    level ratio).
    """
    return 1.0 - compute_distinct_level_ratio(factor_batch)
```

<a id="metric-time_to_recovery"></a>
## time_to_recovery — Time to Recovery

Mean time (periods) from an underwater episode's trough back to a new wealth high. Only completed recoveries are averaged; censored ages are not recoveries. NaN for no completed event or unknown valuation.

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`periods`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`10`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_time_to_recovery`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_time_to_recovery(returns: 'np.ndarray', min_periods: 'int' = 10, max_recovery_lookback: 'Optional[int]' = None) -> 'float'
```

Mean trough-to-recovery intervals of completed events only.

Censored age is available separately in drawdown_events; it never becomes
a measured recovery. The optional limit is applied per completed event.

### 精确计算公式（实际实现）

```python
def compute_time_to_recovery(
    returns: np.ndarray,
    min_periods: int = 10,
    max_recovery_lookback: Optional[int] = None,
) -> float:
    """Mean trough-to-recovery intervals of completed events only.

    Censored age is available separately in drawdown_events; it never becomes
    a measured recovery. The optional limit is applied per completed event.
    """
    if max_recovery_lookback is not None and (
        isinstance(max_recovery_lookback, bool)
        or not isinstance(max_recovery_lookback, (int, np.integer))
        or max_recovery_lookback < 1
    ):
        raise ValueError("max_recovery_lookback must be a positive integer")
    events = _path_events(returns, min_periods)
    if events is None:
        return np.nan
    recovered = [
        e["recovery_idx"] - e["trough_idx"] for e in events
        if not e["censored"] and (
            max_recovery_lookback is None
            or e["recovery_idx"] - e["trough_idx"] <= max_recovery_lookback
        )
    ]
    return float(np.mean(recovered)) if recovered else np.nan
```

<a id="metric-top_quantile_cliff"></a>
## top_quantile_cliff — Top Quantile Cliff

Top-quantile cliff: ret[top] - ret[top-1], per factor. The jump in return from the second-highest to the highest quantile (spec §29).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.quantile_shape.compute_top_quantile_cliff`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.quantile_shape.compute_top_quantile_cliff(qr: 'np.ndarray') -> 'np.ndarray'
```

Top-quantile cliff: ret[top] - ret[top-1], (F,).

### 精确计算公式（实际实现）

```python
def compute_top_quantile_cliff(qr: np.ndarray) -> np.ndarray:
    """Top-quantile cliff: ret[top] - ret[top-1], (F,)."""
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 2:
        return out
    for f in range(F):
        col = m[:, f]
        if np.isfinite(col[-1]) and np.isfinite(col[-2]):
            out[f] = col[-1] - col[-2]
    return out
```

<a id="metric-top_quantile_cliff_robust"></a>
## top_quantile_cliff_robust — Top Quantile Cliff (Robust)

ROBUST top cliff per factor: Q_K - mean(Q_(K-3)..Q_(K-1)) (plan §14.4 - contrast the top bucket against the mean of the three PRIOR buckets, not the noisy one-bin Q_K - Q_(K-1)). Direction higher_is_better (big positive jump into the top bucket). NaN when the top 4 quantile returns are not all finite.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_top_quantile_cliff_robust`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_top_quantile_cliff_robust(qr: 'np.ndarray') -> 'np.ndarray'
```

Robust top cliff: ``ret[top] - mean(ret[K-3:K-1])`` per factor, (F,).

Plan §14.4: single-bin cliffs are noisy; the robust variant contrasts
the top bucket against the mean of the three DIFFERENT prior buckets
(not ``Q_K - Q_(K-1)``).  Direction ``higher_is_better`` (big positive
jump into the top bucket).  NaN when the top 4 quantile returns are not
all finite.

### 精确计算公式（实际实现）

```python
def compute_top_quantile_cliff_robust(qr: np.ndarray) -> np.ndarray:
    """Robust top cliff: ``ret[top] - mean(ret[K-3:K-1])`` per factor, (F,).

    Plan §14.4: single-bin cliffs are noisy; the robust variant contrasts
    the top bucket against the mean of the three DIFFERENT prior buckets
    (not ``Q_K - Q_(K-1)``).  Direction ``higher_is_better`` (big positive
    jump into the top bucket).  NaN when the top 4 quantile returns are not
    all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 4:
        return out
    for f in range(F):
        col = m[:, f]
        seg = col[-4:]
        if not np.all(np.isfinite(seg)):
            continue
        out[f] = seg[-1] - float(np.mean(seg[:-1]))
    return out
```

<a id="metric-top_tail_slope"></a>
## top_tail_slope — Top Tail Slope

Mean adjacent return difference over the TOP segment of the quantile profile as drawn (quantiles K-3..K-1), per factor. Positive for a positively inclined factor; large = top-tail cliff. Direction NEUTRAL: the metric measures the drawn profile's top segment; orientation is informative, never assumed (plan §14.4).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`QuantileReturnArtifact`。
- 输出：`scalar`；单位：`return`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.shape_evidence.compute_top_tail_slope`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_top_tail_slope(qr: 'np.ndarray') -> 'np.ndarray'
```

Top-tail slope per factor, (F,).

Mean adjacent return difference over the TOP segment of the curve as
drawn (quantile ``K-1-k .. K-1``).  For a positively inclined factor
this is POSITIVE (returns keep rising into the top bucket); for a
factor with a top-tail cliff it is LARGE.  Direction metadata is on the
registry spec (``neutral`` — the metric measures the profile's top
segment, sign is informative, not good/bad).  NaN when the top 3
quantile returns are not all finite.

### 精确计算公式（实际实现）

```python
def compute_top_tail_slope(qr: np.ndarray) -> np.ndarray:
    """Top-tail slope per factor, (F,).

    Mean adjacent return difference over the TOP segment of the curve as
    drawn (quantile ``K-1-k .. K-1``).  For a positively inclined factor
    this is POSITIVE (returns keep rising into the top bucket); for a
    factor with a top-tail cliff it is LARGE.  Direction metadata is on the
    registry spec (``neutral`` — the metric measures the profile's top
    segment, sign is informative, not good/bad).  NaN when the top 3
    quantile returns are not all finite.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 3:
        return out
    for f in range(F):
        out[f] = _top_tail_slope_value(m[:, f], n_adj=2)
    return out
```

<a id="metric-tracking_error"></a>
## tracking_error — Tracking Error

Benchmark/invested-capital evidence from an explicitly bound execution trajectory leg

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`annualized_return`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.long_only.compute_tracking_error`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.long_only.compute_tracking_error(returns, periods_per_year: 'int' = 252, min_periods: 'int' = 2)
```

Annualized sample standard deviation of net active returns.

### 精确计算公式（实际实现）

```python
def compute_tracking_error(returns, periods_per_year: int = 252, min_periods: int = 2):
    """Annualized sample standard deviation of net active returns."""
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 2:
        raise ValueError("min_periods must be an integer >= 2")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    x, squeeze = _matrix(returns)
    out = np.full(x.shape[1], np.nan)
    for f in range(x.shape[1]):
        v = x[np.isfinite(x[:, f]), f]
        if len(v) >= min_periods:
            out[f] = np.std(v, ddof=1) * np.sqrt(periods_per_year)
    return out[0] if squeeze else out
```

<a id="metric-tradable_coverage"></a>
## tradable_coverage — Tradable Coverage

Fraction of days with at least ``min_assets`` jointly valid cells, per factor. A tradable day is one where the factor and label are both available for enough assets (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_tradable_coverage`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_tradable_coverage(factor_batch: 'FactorBatch', label_bundle: 'LabelBundle', min_assets: 'int' = 10) -> 'np.ndarray'
```

Fraction of days with at least ``min_assets`` jointly valid cells, (F,).

A tradable day is one where the factor and label are both available for
at least ``min_assets`` assets.

### 精确计算公式（实际实现）

```python
def compute_tradable_coverage(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> np.ndarray:
    """Fraction of days with at least ``min_assets`` jointly valid cells, (F,).

    A tradable day is one where the factor and label are both available for
    at least ``min_assets`` assets.
    """
    values = _factor_values(factor_batch)
    labels = _labels(label_bundle, factor_batch.num_assets)
    T, N, F = values.shape
    label_finite = np.isfinite(labels)
    factor_finite = np.isfinite(values)
    valid = factor_finite & label_finite[:, :, None]  # (T, N, F)
    per_day = np.sum(valid, axis=1)  # (T, F)
    return np.sum(per_day >= min_assets, axis=0) / T
```

<a id="metric-train_predictive_dimension"></a>
## train_predictive_dimension — Train Predictive Dimension

Train-side predictive dimension of the authorized train-vs-validation comparison (plan §13.6), per factor (e.g. mean daily rank IC on the train evaluation).

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.train_predictive_dimension`。

### 计算定义与默认参数

```python
quant_evaluator.registry.metrics.<lambda>(v)
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
```

<a id="metric-train_validation_icir_delta"></a>
## train_validation_icir_delta — Train-Validation ICIR Delta

Absolute ICIR delta (validation - train) per factor (plan §13.6). Delta, never a ratio.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`ratio`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas`。

### 计算定义与默认参数

```python
quant_evaluator.registry.metrics.<lambda>(v)
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
```

<a id="metric-train_validation_rankic_delta"></a>
## train_validation_rankic_delta — Train-Validation RankIC Delta

Absolute rank-IC delta (validation - train) per factor (plan §13.6). A DELTA, never a ratio: well-defined even when train is near zero. NaN when either side is not finite.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas`。

### 计算定义与默认参数

```python
quant_evaluator.registry.metrics.<lambda>(v)
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
```

<a id="metric-train_validation_shape_delta"></a>
## train_validation_shape_delta — Train-Validation Shape Delta

Absolute shape-evidence delta (validation - train) per factor (plan §13.6). Delta, never a ratio.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`score`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas`。

### 计算定义与默认参数

```python
quant_evaluator.registry.metrics.<lambda>(v)
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
```

<a id="metric-train_validation_sharpe_delta"></a>
## train_validation_sharpe_delta — Train-Validation Sharpe Delta

Absolute long/short Sharpe delta (validation - train) per factor (plan §13.6). Delta, never a ratio.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`ratio`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_generalization_deltas`。

### 计算定义与默认参数

```python
quant_evaluator.registry.metrics.<lambda>(v)
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
```

<a id="metric-turnover"></a>
## turnover — Portfolio Turnover

Average turnover rate for factor-based portfolios

- 版本：`3.0.0`；状态：`stable`；层级：`core`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`注册表未标注，见函数公式`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`2`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`turnover`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.registry_adapters.compute_turnover_value(factor_batch: quant_evaluator.contracts.factor_batch.FactorBatch, min_periods: int = 2) -> numpy.ndarray
```

Return mean cross-sectional turnover per factor.

Ranks are used as proxy weights so the value is well defined for a raw
factor batch without a portfolio construction step. Weights are
average-tie ranks (``scipy.stats.rankdata(method="average")``)
normalized to sum 1 per row, so the value is universe-size invariant
and conforms to the canonical turnover definition in
``metrics/turnover.py``.

### 精确计算公式（实际实现）

```python
def compute_turnover_value(
    factor_batch: FactorBatch,
    min_periods: int = 2,
) -> np.ndarray:
    """Return mean cross-sectional turnover per factor.

    Ranks are used as proxy weights so the value is well defined for a raw
    factor batch without a portfolio construction step. Weights are
    average-tie ranks (``scipy.stats.rankdata(method="average")``)
    normalized to sum 1 per row, so the value is universe-size invariant
    and conforms to the canonical turnover definition in
    ``metrics/turnover.py``.
    """
    values = np.asarray(factor_batch.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("factor_batch.values must be (T, N, F)")
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    n_factors = values.shape[2]
    result = np.full(n_factors, np.nan, dtype=np.float64)
    for f in range(n_factors):
        series = values[:, :, f]
        # Cross-sectional average-tie rank weights (scipy rankdata,
        # method="average"), normalized to sum 1 over the finite assets of
        # each row: universe-size invariant and consistent with the canonical
        # turnover definition (see metrics/turnover.py module docstring).
        # Eligible dates use explicit zero weight for unselected/missing
        # signals; insufficient dates remain unknown rather than cash.
        from scipy.stats import rankdata

        ranks = np.full_like(series, np.nan)
        for t in range(series.shape[0]):
            row = series[t]
            finite = np.isfinite(row)
            if finite.sum() < 2:
                continue
            ranks_f = rankdata(row[finite], method="average")
            ranks[t, :] = 0.0
            ranks[t, finite] = ranks_f / np.sum(ranks_f)
        if series.shape[0] < 2:
            continue
        turnover_series = compute_turnover_series(ranks)
        valid = turnover_series[np.isfinite(turnover_series)]
        if valid.size >= min_periods - 1:
            result[f] = float(np.mean(valid))
    return result
```

<a id="metric-turnover_adjusted_ic"></a>
## turnover_adjusted_ic — turnover_adjusted_ic

IC adjusted for turnover-induced transaction costs

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.turnover.compute_turnover_contribution`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.turnover_cost.compute_turnover_cost(returns: 'np.ndarray', min_periods: 'int' = 1) -> 'float | np.ndarray'
```

Return mean realized portfolio cost drag in basis points.

``returns`` is deliberately named for QE's portfolio-panel runtime binder,
but its values must be the positive cost-rate magnitudes from the typed
``cost_drag`` trajectory leg.  It is not portfolio PnL, gross-minus-net,
or factor-rank turnover.  NaN denotes an unobserved period; infinities and
negative observed costs are invalid rather than silently reinterpreted.

### 精确计算公式（实际实现）

```python
def compute_turnover_cost(
    returns: np.ndarray,
    min_periods: int = 1,
) -> float | np.ndarray:
    """Return mean realized portfolio cost drag in basis points.

    ``returns`` is deliberately named for QE's portfolio-panel runtime binder,
    but its values must be the positive cost-rate magnitudes from the typed
    ``cost_drag`` trajectory leg.  It is not portfolio PnL, gross-minus-net,
    or factor-rank turnover.  NaN denotes an unobserved period; infinities and
    negative observed costs are invalid rather than silently reinterpreted.
    """
    if isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer)):
        raise TypeError("min_periods must be a positive integer")
    if min_periods < 1:
        raise ValueError("min_periods must be at least 1")

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim not in (1, 2):
        raise ValueError("returns must have shape (T,) or (T, F)")
    observed = ~np.isnan(values)
    if np.any(np.isinf(values)):
        raise ValueError("cost_drag contains infinite values")
    if np.any(values[observed] < 0.0):
        raise ValueError("cost_drag must contain non-negative cost-rate magnitudes")

    matrix = values[:, None] if values.ndim == 1 else values
    counts = np.sum(~np.isnan(matrix), axis=0)
    totals = np.nansum(matrix, axis=0)
    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    enough = counts >= min_periods
    result[enough] = totals[enough] / counts[enough] * 10_000.0
    return float(result[0]) if values.ndim == 1 else result
```

<a id="metric-turnover_rate"></a>
## turnover_rate — turnover_rate

Average rate of change in factor ranking between periods

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.turnover.compute_turnover`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.shape_evidence.compute_u_shape_score(qr: 'np.ndarray') -> 'np.ndarray'
```

U-shape score per factor, (F,).

Plan §14.3 — NOT ``RankIC ≈ 0``.  Combines three ingredients into
[0, 1]:

1. *Middle-underperforms-tails* (U template fit): fit ``a + b*U(x)``
   where ``U(x) = (x - c)^2`` (c = centre) to the quantile-return
   profile and take positive-bounded R^2.
2. *Curvature*: fraction of the interior second differences that are
   positive (convex).  A U profile is convex; an inverted-U is concave.
3. *U-template-beats-monotone-template*: the U R^2 must EXCEED the
   monotone-linear R^2 (plan §14.3 ingredient "U template fit >
   monotonic template fit").  A factor with strong monotone slope is
   NOT a U even if its U-fit is high.

NaN when the profile has fewer than 4 finite quantile returns.

### 精确计算公式（实际实现）

```python
def compute_u_shape_score(qr: np.ndarray) -> np.ndarray:
    """U-shape score per factor, (F,).

    Plan §14.3 — NOT ``RankIC ≈ 0``.  Combines three ingredients into
    [0, 1]:

    1. *Middle-underperforms-tails* (U template fit): fit ``a + b*U(x)``
       where ``U(x) = (x - c)^2`` (c = centre) to the quantile-return
       profile and take positive-bounded R^2.
    2. *Curvature*: fraction of the interior second differences that are
       positive (convex).  A U profile is convex; an inverted-U is concave.
    3. *U-template-beats-monotone-template*: the U R^2 must EXCEED the
       monotone-linear R^2 (plan §14.3 ingredient "U template fit >
       monotonic template fit").  A factor with strong monotone slope is
       NOT a U even if its U-fit is high.

    NaN when the profile has fewer than 4 finite quantile returns.
    """
    m = _as_matrix(qr)
    nq, F = m.shape
    out = np.full(F, np.nan)
    if nq < 4:
        return out
    x = _template_fit_x(nq)
    ut = (x - 0.5) ** 2
    for f in range(F):
        col = m[:, f]
        finite = np.isfinite(col)
        if np.sum(finite) < 4:
            continue
        y = col[finite]
        xu = ut[finite]
        xm = x[finite]
        if np.ptp(y) == 0.0:
            out[f] = 0.0
            continue
        r2_u = _fit_variance_explained(y, xu)
        r2_m = _fit_variance_explained(y, xm)
        if not (np.isfinite(r2_u) and np.isfinite(r2_m)):
            continue
        # Convex curvature fraction over interior positions.
        interior = finite[1:-1] & finite[:-2] & finite[2:]
        if not np.any(interior):
            continue
        d2 = col[2:][interior] - 2.0 * col[1:-1][interior] + col[:-2][interior]
        curv = float(np.mean(d2 > 0.0))
        # Ingredient 3: U template must beat the monotone template.
        if r2_u <= r2_m:
            out[f] = 0.0
            continue
        # A hill profile is concave; without convex curvature the U label
        # must not win even though the sign-flipped template fits.
        if curv < 0.5:
            out[f] = 0.0
            continue
        u_fit = max(r2_u, 0.0)
        out[f] = float(0.5 * u_fit + 0.3 * curv + 0.2 * (u_fit - r2_m))
    return out
```

<a id="metric-universe_churn"></a>
## universe_churn — Universe Churn

Mean fraction of the tradable universe that changes membership per day, per factor. Churn = fraction of assets tradable on exactly one of two adjacent days (spec §35).

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch, label_bundle`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`2`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.data_quality.compute_universe_churn`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.data_quality.compute_universe_churn(factor_batch: 'FactorBatch', label_bundle: 'LabelBundle') -> 'np.ndarray'
```

Mean fraction of the tradable universe that changes membership per day, (F,).

Churn = fraction of assets that are tradable on exactly one of two
adjacent days (entering or leaving the tradable set).

### 精确计算公式（实际实现）

```python
def compute_universe_churn(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> np.ndarray:
    """Mean fraction of the tradable universe that changes membership per day, (F,).

    Churn = fraction of assets that are tradable on exactly one of two
    adjacent days (entering or leaving the tradable set).
    """
    values = _factor_values(factor_batch)
    labels = _labels(label_bundle, factor_batch.num_assets)
    T, N, F = values.shape
    if T < 2:
        return np.full(F, np.nan)
    label_finite = np.isfinite(labels)
    factor_finite = np.isfinite(values)
    tradable = factor_finite & label_finite[:, :, None]  # (T, N, F)
    prev = tradable[:-1, :, :]
    curr = tradable[1:, :, :]
    churn = np.sum(prev != curr, axis=1)  # (T-1, F)
    return np.mean(churn / N, axis=0)
```

<a id="metric-validation_predictive_dimension"></a>
## validation_predictive_dimension — Validation Predictive Dimension

Validation-side predictive dimension of the authorized train-vs-validation comparison (plan §13.6), per factor.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`correlation`；方向：`neutral`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.validation_predictive_dimension`。

### 计算定义与默认参数

```python
quant_evaluator.registry.metrics.<lambda>(v)
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
```

<a id="metric-validation_retention"></a>
## validation_retention — Validation Retention

Robust validation/train retention per factor (plan §13.6): validation/train where the train denominator is stable, NaN with an explicit reason (train_near_zero / sign_flip_guard / insufficient_data) otherwise - never a blind division by a tiny train value. Grade anchors are versioned policy constants in RetentionPolicy.

- 版本：`2.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`factor_batch`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.generalization_evidence.compute_validation_retention_array`。

### 计算定义与默认参数

```python
quant_evaluator.registry.metrics.<lambda>(v)
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
compute_fn=lambda v: np.asarray(v, dtype=np.float64).reshape(-1),
```

<a id="metric-var_95"></a>
## var_95 — var_95

Value at Risk at 95% confidence level

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`lower_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.risk.var_cvar.compute_var`。

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

### 计算定义与默认参数

```python
quant_evaluator.metrics.exposure_evidence.compute_volatility_exposure(panel: 'FactorLoadingSeries', *, factor_values=None, min_obs=10, weights=None) -> 'float'
```

Signed mean volatility-style exposure of the factor (one typed field).

### 精确计算公式（实际实现）

```python
def compute_volatility_exposure(panel: FactorLoadingSeries, *, factor_values=None, min_obs=10, weights=None) -> float:
    """Signed mean volatility-style exposure of the factor (one typed field)."""
    return _select_style(panel, "volatility",factor_values=factor_values,min_obs=min_obs,weights=weights)
```

<a id="metric-win_rate"></a>
## win_rate — win_rate

Win rate of a return series per factor: fraction of finite returns that are strictly positive, in [0, 1]. NaN when there are no finite returns. (Returns equal to exactly 0 count as neither wins nor losses.)

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.portfolio_stats.compute_win_rate`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.portfolio_stats.compute_win_rate(returns: numpy.ndarray) -> numpy.ndarray
```

Compute win rate (fraction of positive returns).

Args:
    returns: Return series (T,) or (T, F)

Returns:
    Win rate in [0, 1], scalar or shape (F,)

### 精确计算公式（实际实现）

```python
def compute_win_rate(
    returns: np.ndarray,
) -> np.ndarray:
    """
    Compute win rate (fraction of positive returns).

    Args:
        returns: Return series (T,) or (T, F)

    Returns:
        Win rate in [0, 1], scalar or shape (F,)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    valid = np.isfinite(returns)
    n_valid = np.sum(valid, axis=0)

    wins = np.sum((returns > 0) & valid, axis=0)
    win_rate = wins / np.maximum(n_valid, 1)

    # Set to NaN if no valid observations
    win_rate = np.where(n_valid > 0, win_rate, np.nan)

    return win_rate[0] if squeeze else win_rate
```

<a id="metric-worst_12m"></a>
## worst_12m — Worst Rolling 252 Periods (legacy ID)

Worst fixed 252-period (trading) block compounded return of the probe daily PnL series. NaN when the series is shorter than one 12m block.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`252`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_worst_period_return`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_worst_period_return(returns: 'np.ndarray', period: 'str' = 'month', min_periods: 'int' = 10) -> 'float'
```

Calendar worst block return: the minimum compounded return over any
fixed-length trading-period block (``month``=21 / ``quarter``=63 /
``year``=252 periods).

Returns NaN when fewer than ``min_periods`` finite periods exist, or when
the series has fewer than one full block.

Conventions mirror the existing codebase calendar: binary backtest /
report cards fixed period lengths (21 / 63 / 252), no calendar-month
index dependence — a pure mathematical "worst 21-period block" over the
dot-frequency PnL series (kept calendar-window-agnostic because the probe
PnL series carries only a dot index, not calendar dates).

### 精确计算公式（实际实现）

```python
def compute_worst_period_return(
    returns: np.ndarray,
    period: str = "month",
    min_periods: int = 10,
) -> float:
    """Calendar worst block return: the minimum compounded return over any
    fixed-length trading-period block (``month``=21 / ``quarter``=63 /
    ``year``=252 periods).

    Returns NaN when fewer than ``min_periods`` finite periods exist, or when
    the series has fewer than one full block.

    Conventions mirror the existing codebase calendar: binary backtest /
    report cards fixed period lengths (21 / 63 / 252), no calendar-month
    index dependence — a pure mathematical "worst 21-period block" over the
    dot-frequency PnL series (kept calendar-window-agnostic because the probe
    PnL series carries only a dot index, not calendar dates).
    """
    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    if period == "month":
        block = _PERIODS_PER_MONTH
    elif period == "quarter":
        block = _PERIODS_PER_QUARTER
    elif period == "year":
        block = _PERIODS_PER_YEAR
    else:
        raise ValueError(
            f"period must be one of 'month'|'quarter'|'year', got {period!r}"
        )
    if ret.size < block:
        return np.nan
    # Compounded block returns: prod(1+r) over each length-block window.
    blocks = np.full(ret.size - block + 1, np.nan)
    for t in range(ret.size - block + 1):
        blocks[t] = np.prod(1.0 + ret[t : t + block]) - 1.0
    return float(np.min(blocks))
```

<a id="metric-worst_calendar_month"></a>
## worst_calendar_month — worst_calendar_month

Worst calendar-period compounded probe return; explicit DA calendar and partial-period policy

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_calendar_month`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.calendar_returns.compute_worst_calendar_month(returns, time_index, factor_ids, calendar_snapshot, partial_policy='exclude')
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
def compute_worst_calendar_month(returns, time_index, factor_ids, calendar_snapshot, partial_policy="exclude"):
    return _calendar_metric(returns, time_index, factor_ids, calendar_snapshot,
                            frequency="month", partial_policy=partial_policy)
```

<a id="metric-worst_calendar_quarter"></a>
## worst_calendar_quarter — worst_calendar_quarter

Worst calendar-period compounded probe return; explicit DA calendar and partial-period policy

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_calendar_quarter`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.calendar_returns.compute_worst_calendar_quarter(returns, time_index, factor_ids, calendar_snapshot, partial_policy='exclude')
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
def compute_worst_calendar_quarter(returns, time_index, factor_ids, calendar_snapshot, partial_policy="exclude"):
    return _calendar_metric(returns, time_index, factor_ids, calendar_snapshot,
                            frequency="quarter", partial_policy=partial_policy)
```

<a id="metric-worst_calendar_year"></a>
## worst_calendar_year — worst_calendar_year

Worst calendar-period compounded probe return; explicit DA calendar and partial-period policy

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_calendar_year`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.calendar_returns.compute_worst_calendar_year(returns, time_index, factor_ids, calendar_snapshot, partial_policy='exclude')
```

此函数没有独立说明；精确定义见下方源公式。

### 精确计算公式（实际实现）

```python
def compute_worst_calendar_year(returns, time_index, factor_ids, calendar_snapshot, partial_policy="exclude"):
    return _calendar_metric(returns, time_index, factor_ids, calendar_snapshot,
                            frequency="year", partial_policy=partial_policy)
```

<a id="metric-worst_month"></a>
## worst_month — Worst Rolling 21 Periods (legacy ID)

Worst fixed 21-period (trading) block compounded return of the probe daily PnL series. Uses the codebase's fixed trading-period calendar (21/period month); NaN when the series is shorter than one block.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`21`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_worst_period_return`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_worst_period_return(returns: 'np.ndarray', period: 'str' = 'month', min_periods: 'int' = 10) -> 'float'
```

Calendar worst block return: the minimum compounded return over any
fixed-length trading-period block (``month``=21 / ``quarter``=63 /
``year``=252 periods).

Returns NaN when fewer than ``min_periods`` finite periods exist, or when
the series has fewer than one full block.

Conventions mirror the existing codebase calendar: binary backtest /
report cards fixed period lengths (21 / 63 / 252), no calendar-month
index dependence — a pure mathematical "worst 21-period block" over the
dot-frequency PnL series (kept calendar-window-agnostic because the probe
PnL series carries only a dot index, not calendar dates).

### 精确计算公式（实际实现）

```python
def compute_worst_period_return(
    returns: np.ndarray,
    period: str = "month",
    min_periods: int = 10,
) -> float:
    """Calendar worst block return: the minimum compounded return over any
    fixed-length trading-period block (``month``=21 / ``quarter``=63 /
    ``year``=252 periods).

    Returns NaN when fewer than ``min_periods`` finite periods exist, or when
    the series has fewer than one full block.

    Conventions mirror the existing codebase calendar: binary backtest /
    report cards fixed period lengths (21 / 63 / 252), no calendar-month
    index dependence — a pure mathematical "worst 21-period block" over the
    dot-frequency PnL series (kept calendar-window-agnostic because the probe
    PnL series carries only a dot index, not calendar dates).
    """
    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    if period == "month":
        block = _PERIODS_PER_MONTH
    elif period == "quarter":
        block = _PERIODS_PER_QUARTER
    elif period == "year":
        block = _PERIODS_PER_YEAR
    else:
        raise ValueError(
            f"period must be one of 'month'|'quarter'|'year', got {period!r}"
        )
    if ret.size < block:
        return np.nan
    # Compounded block returns: prod(1+r) over each length-block window.
    blocks = np.full(ret.size - block + 1, np.nan)
    for t in range(ret.size - block + 1):
        blocks[t] = np.prod(1.0 + ret[t : t + block]) - 1.0
    return float(np.min(blocks))
```

<a id="metric-worst_quarter"></a>
## worst_quarter — Worst Rolling 63 Periods (legacy ID)

Worst fixed 63-period (trading) block compounded return of the probe daily PnL series. NaN when the series is shorter than one block.

- 版本：`1.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`probe_pnl`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`63`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.underwater.compute_worst_period_return`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.underwater.compute_worst_period_return(returns: 'np.ndarray', period: 'str' = 'month', min_periods: 'int' = 10) -> 'float'
```

Calendar worst block return: the minimum compounded return over any
fixed-length trading-period block (``month``=21 / ``quarter``=63 /
``year``=252 periods).

Returns NaN when fewer than ``min_periods`` finite periods exist, or when
the series has fewer than one full block.

Conventions mirror the existing codebase calendar: binary backtest /
report cards fixed period lengths (21 / 63 / 252), no calendar-month
index dependence — a pure mathematical "worst 21-period block" over the
dot-frequency PnL series (kept calendar-window-agnostic because the probe
PnL series carries only a dot index, not calendar dates).

### 精确计算公式（实际实现）

```python
def compute_worst_period_return(
    returns: np.ndarray,
    period: str = "month",
    min_periods: int = 10,
) -> float:
    """Calendar worst block return: the minimum compounded return over any
    fixed-length trading-period block (``month``=21 / ``quarter``=63 /
    ``year``=252 periods).

    Returns NaN when fewer than ``min_periods`` finite periods exist, or when
    the series has fewer than one full block.

    Conventions mirror the existing codebase calendar: binary backtest /
    report cards fixed period lengths (21 / 63 / 252), no calendar-month
    index dependence — a pure mathematical "worst 21-period block" over the
    dot-frequency PnL series (kept calendar-window-agnostic because the probe
    PnL series carries only a dot index, not calendar dates).
    """
    ret = _as_1d(returns)
    if ret.size < min_periods:
        return np.nan
    if period == "month":
        block = _PERIODS_PER_MONTH
    elif period == "quarter":
        block = _PERIODS_PER_QUARTER
    elif period == "year":
        block = _PERIODS_PER_YEAR
    else:
        raise ValueError(
            f"period must be one of 'month'|'quarter'|'year', got {period!r}"
        )
    if ret.size < block:
        return np.nan
    # Compounded block returns: prod(1+r) over each length-block window.
    blocks = np.full(ret.size - block + 1, np.nan)
    for t in range(ret.size - block + 1):
        blocks[t] = np.prod(1.0 + ret[t : t + block]) - 1.0
    return float(np.min(blocks))
```

<a id="metric-worst_quarter_rank_ic"></a>
## worst_quarter_rank_ic — Worst-Quarter Rank IC

Minimum per-quarter mean rank IC (the factor's worst quarter), per factor (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_worst_quarter_rank_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_worst_quarter_rank_ic(ic_series: 'np.ndarray', min_periods: 'int' = 20, time_index: 'Optional[Sequence]' = None) -> 'np.ndarray'
```

Minimum per-quarter mean rank IC (worst quarter), (F,).

### 精确计算公式（实际实现）

```python
def compute_worst_quarter_rank_ic(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Minimum per-quarter mean rank IC (worst quarter), (F,)."""
    s = _as_series(ic_series)
    out = _worst_period(s, "quarter", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-worst_rolling_21d"></a>
## worst_rolling_21d — Worst 21 trading periods (rolling)

Worst fully matured fixed-length compounded return; never a calendar period

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return(returns: 'Any', *, window: 'int' = 21, min_periods: 'int' = 1) -> 'np.ndarray'
```

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

绑定参数：`args=(), kwargs={'window': 21}`。

### 精确计算公式（实际实现）

```python
def compute_worst_rolling_return(
    returns: Any,
    window: int = 21,
    min_periods: int = 1,
) -> np.ndarray:
    """Per-factor worst compounded return across matured full windows.

    ``min_periods`` is the required count of valid, fully matured windows; it
    never authorizes an expanding prefix. Every row in a candidate window must
    be finite for that factor. Invalid windows are skipped without compressing
    the time grid, and become eligible again only after the unknown row exits.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if (isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer))
            or min_periods < 1):
        raise ValueError("min_periods must be a positive integer")
    values = _validated_returns(returns)
    rolling = np.full((max(0, values.shape[0] - int(window) + 1), values.shape[1]), np.nan)
    for offset, stop in enumerate(range(int(window), values.shape[0] + 1)):
        block = values[stop - int(window):stop]
        valid = np.all(np.isfinite(block), axis=0)
        rolling[offset, valid] = np.prod(1.0 + block[:, valid], axis=0) - 1.0
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for factor in range(values.shape[1]):
        candidates = rolling[np.isfinite(rolling[:, factor]), factor]
        if candidates.size >= int(min_periods):
            result[factor] = np.min(candidates)
    return result
```

<a id="metric-worst_rolling_252d"></a>
## worst_rolling_252d — Worst 252 trading periods (rolling)

Worst fully matured fixed-length compounded return; never a calendar period

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return(returns: 'Any', *, window: 'int' = 252, min_periods: 'int' = 1) -> 'np.ndarray'
```

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

绑定参数：`args=(), kwargs={'window': 252}`。

### 精确计算公式（实际实现）

```python
def compute_worst_rolling_return(
    returns: Any,
    window: int = 21,
    min_periods: int = 1,
) -> np.ndarray:
    """Per-factor worst compounded return across matured full windows.

    ``min_periods`` is the required count of valid, fully matured windows; it
    never authorizes an expanding prefix. Every row in a candidate window must
    be finite for that factor. Invalid windows are skipped without compressing
    the time grid, and become eligible again only after the unknown row exits.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if (isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer))
            or min_periods < 1):
        raise ValueError("min_periods must be a positive integer")
    values = _validated_returns(returns)
    rolling = np.full((max(0, values.shape[0] - int(window) + 1), values.shape[1]), np.nan)
    for offset, stop in enumerate(range(int(window), values.shape[0] + 1)):
        block = values[stop - int(window):stop]
        valid = np.all(np.isfinite(block), axis=0)
        rolling[offset, valid] = np.prod(1.0 + block[:, valid], axis=0) - 1.0
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for factor in range(values.shape[1]):
        candidates = rolling[np.isfinite(rolling[:, factor]), factor]
        if candidates.size >= int(min_periods):
            result[factor] = np.min(candidates)
    return result
```

<a id="metric-worst_rolling_63d"></a>
## worst_rolling_63d — Worst 63 trading periods (rolling)

Worst fully matured fixed-length compounded return; never a calendar period

- 版本：`1.0.0`；状态：`stable`；层级：`extended`。
- 输入依赖：`见函数签名`。
- 输出：`scalar`；单位：`return`；方向：`higher_is_better`。
- 缺失政策：`unknown`；数值政策：`finite`；注册最低期数：`None`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.calendar_returns.compute_worst_rolling_return(returns: 'Any', *, window: 'int' = 63, min_periods: 'int' = 1) -> 'np.ndarray'
```

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

绑定参数：`args=(), kwargs={'window': 63}`。

### 精确计算公式（实际实现）

```python
def compute_worst_rolling_return(
    returns: Any,
    window: int = 21,
    min_periods: int = 1,
) -> np.ndarray:
    """Per-factor worst compounded return across matured full windows.

    ``min_periods`` is the required count of valid, fully matured windows; it
    never authorizes an expanding prefix. Every row in a candidate window must
    be finite for that factor. Invalid windows are skipped without compressing
    the time grid, and become eligible again only after the unknown row exits.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if (isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer))
            or min_periods < 1):
        raise ValueError("min_periods must be a positive integer")
    values = _validated_returns(returns)
    rolling = np.full((max(0, values.shape[0] - int(window) + 1), values.shape[1]), np.nan)
    for offset, stop in enumerate(range(int(window), values.shape[0] + 1)):
        block = values[stop - int(window):stop]
        valid = np.all(np.isfinite(block), axis=0)
        rolling[offset, valid] = np.prod(1.0 + block[:, valid], axis=0) - 1.0
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for factor in range(values.shape[1]):
        candidates = rolling[np.isfinite(rolling[:, factor]), factor]
        if candidates.size >= int(min_periods):
            result[factor] = np.min(candidates)
    return result
```

<a id="metric-worst_year_rank_ic"></a>
## worst_year_rank_ic — Worst-Year Rank IC

Minimum per-year mean rank IC (the factor's worst year), per factor. A robustness check on how bad the factor's worst annual performance is (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_worst_year_rank_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_worst_year_rank_ic(ic_series: 'np.ndarray', min_periods: 'int' = 20, time_index: 'Optional[Sequence]' = None) -> 'np.ndarray'
```

Minimum per-year mean rank IC (worst year), (F,).

### 精确计算公式（实际实现）

```python
def compute_worst_year_rank_ic(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Minimum per-year mean rank IC (worst year), (F,)."""
    s = _as_series(ic_series)
    out = _worst_period(s, "year", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-year_consistency"></a>
## year_consistency — Year Consistency

Fraction of years whose mean IC matches the overall IC sign, per factor. A robustness check on how consistently the factor works across years (spec §30).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`fraction`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.stability_regime.compute_year_consistency`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.stability_regime.compute_year_consistency(ic_series: 'np.ndarray', min_periods: 'int' = 20, time_index: 'Optional[Sequence]' = None) -> 'np.ndarray'
```

Fraction of years whose mean IC matches the overall IC sign, (F,).

### 精确计算公式（实际实现）

```python
def compute_year_consistency(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Fraction of years whose mean IC matches the overall IC sign, (F,)."""
    s = _as_series(ic_series)
    out = _period_consistency(s, "year", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

<a id="metric-yearly_rank_ic"></a>
## yearly_rank_ic — Yearly Rank IC

Mean of the per-year mean rank IC, per factor. A robust annual average that down-weights any single strong year (spec §28).

- 版本：`3.0.0`；状态：`experimental`；层级：`extended`。
- 输入依赖：`ICSeriesArtifact`。
- 输出：`scalar`；单位：`correlation`；方向：`higher_is_better`。
- 缺失政策：`nan`；数值政策：`finite`；注册最低期数：`20`。
- 别名：无。
- 增量模式：`PERIODIC_RECOMPUTE`；注册实现定位：`quant_evaluator.metrics.predictive.compute_yearly_rank_ic`。

### 计算定义与默认参数

```python
quant_evaluator.metrics.predictive.compute_yearly_rank_ic(ic_series: 'np.ndarray', min_periods: 'int' = 20, time_index: 'Optional[Sequence]' = None) -> 'np.ndarray'
```

Mean of the per-year mean rank IC, (F,).

### 精确计算公式（实际实现）

```python
def compute_yearly_rank_ic(
    ic_series: np.ndarray,
    min_periods: int = 20,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Mean of the per-year mean rank IC, (F,)."""
    s = _as_series(ic_series)
    out = _mean_of_period_means(s, "year", time_index)
    return np.where(_valid_counts(s) >= min_periods, out, np.nan)
```

## 共享公式与掩码依赖

以下为上文适配器引用的共享计算函数，避免只展示一层转发却遗漏真实公式。外部NumPy/SciPy标准运算按其参数解释。
输入合同、交易制品与GPU派发仍以相应模块及统一口径为准；本附录不复制数据或生产产物。

### quant_evaluator.metrics.calendar_returns._calendar_metric

```python
def _calendar_metric(
    returns: Any,
    time_index: Sequence[Any],
    factor_ids: Sequence[str],
    calendar_snapshot: Any,
    *,
    frequency: str,
    partial_policy: str,
) -> ScalarMetricArtifact:
    # Lazy import keeps the standalone quant_evaluator wheel importable when
    # the optional data_access package is absent.
    from data_access.r30.calendar_snapshot import CalendarSnapshot
    if not isinstance(calendar_snapshot, CalendarSnapshot):
        raise TypeError("calendar_snapshot must be a Data Access CalendarSnapshot")
    if partial_policy not in {"exclude", "include"}:
        raise ValueError("partial_policy must be 'exclude' or 'include'")
    values = _validated_returns(returns, factor_ids)
    if len(time_index) != values.shape[0]:
        raise ValueError("time_index length must equal returns time dimension")
    local_dates = _local_session_dates(time_index, calendar_snapshot.timezone)
    try:
        expected_dates = tuple(date.fromisoformat(str(day)[:10]) for day in calendar_snapshot.trading_days)
    except ValueError as exc:
        raise ValueError("calendar_snapshot trading_days must contain ISO dates") from exc
    if not expected_dates or any(a >= b for a, b in zip(expected_dates, expected_dates[1:])):
        raise ValueError("calendar_snapshot trading_days must be non-empty and strictly increasing")
    expected_set = set(expected_dates)
    unknown_sessions = [day.isoformat() for day in local_dates if day not in expected_set]
    if unknown_sessions:
        raise ValueError(f"time_index contains sessions absent from calendar snapshot: {unknown_sessions}")

    observed_by_date = {day: row for day, row in zip(local_dates, values)}
    expected_by_period: dict[tuple[int, ...], list[date]] = {}
    for day in expected_dates:
        expected_by_period.setdefault(_period_key(day, frequency), []).append(day)

    period_rows: list[dict[str, Any]] = []
    period_values: list[np.ndarray] = []
    eligibility: list[np.ndarray] = []
    for key, sessions in expected_by_period.items():
        observed = [day for day in sessions if day in observed_by_date]
        bracketed = expected_dates[0] < sessions[0] and expected_dates[-1] > sessions[-1]
        sessions_complete = len(observed) == len(sessions)
        finite_counts = np.zeros(values.shape[1], dtype=np.int64)
        compounded = np.full(values.shape[1], np.nan, dtype=np.float64)
        if observed:
            block = np.stack([observed_by_date[day] for day in observed], axis=0)
            finite = np.all(np.isfinite(block), axis=0)
            finite_counts = np.sum(np.isfinite(block), axis=0)
            compounded[finite] = np.prod(1.0 + block[:, finite], axis=0) - 1.0
        complete_by_factor = sessions_complete & bracketed & (finite_counts == len(sessions))
        eligible = (finite_counts == len(observed)) & (len(observed) > 0)
        if partial_policy == "exclude":
            eligible &= complete_by_factor
        period_values.append(compounded)
        eligibility.append(eligible)
        period_rows.append({
            "period_id": _period_id(key, frequency),
            "expected_session_count": len(sessions),
            "observed_session_count": len(observed),
            "finite_return_counts": tuple(int(x) for x in finite_counts),
            "calendar_coverage_bracketed": bool(bracketed),
            "sessions_complete": bool(sessions_complete),
            "complete_by_factor": tuple(bool(x) for x in complete_by_factor),
            "partial": not bool(sessions_complete and bracketed),
            "included_by_factor": tuple(bool(x) for x in eligible),
            "compounded_returns": tuple(float(x) if np.isfinite(x) else None for x in compounded),
        })

    worst = np.full(values.shape[1], np.nan, dtype=np.float64)
    if period_values:
        matrix = np.stack(period_values)
        mask = np.stack(eligibility)
        for factor in range(values.shape[1]):
            candidates = matrix[mask[:, factor], factor]
            if candidates.size:
                worst[factor] = np.min(candidates)
    metric_id = f"worst_calendar_{frequency}"
    observation_counts = tuple(
        int(sum(bool(mask[factor]) for mask in eligibility))
        for factor in range(values.shape[1])
    )
    return ScalarMetricArtifact(
        metric_id=metric_id,
        domain="risk",
        values=worst,
        factor_axis=FactorAxisRef(factor_ids=tuple(factor_ids)),
        provenance={
            "calendar_snapshot_id": calendar_snapshot.snapshot_id,
            "calendar_market": calendar_snapshot.market,
            "calendar_timezone": calendar_snapshot.timezone,
            "calendar_source_version": calendar_snapshot.source_version,
            "partial_policy": partial_policy,
            "observation_counts": observation_counts,
            "period_rows": tuple(period_rows),
        },
    )
```

### quant_evaluator.metrics.calendar_returns._local_session_dates

```python
def _local_session_dates(time_index: Sequence[Any], timezone_name: str) -> tuple[date, ...]:
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"calendar snapshot has unknown timezone {timezone_name!r}") from exc
    dates: list[date] = []
    for value in time_index:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("time_index entries must be timezone-aware datetime instants")
        dates.append(value.astimezone(timezone).date())
    if len(set(dates)) != len(dates):
        raise ValueError("time_index must contain at most one return row per local trading session")
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("time_index must be strictly increasing in calendar-local session dates")
    return tuple(dates)
```

### quant_evaluator.metrics.calendar_returns._period_id

```python
def _period_id(key: tuple[int, ...], frequency: str) -> str:
    if frequency == "month":
        return f"{key[0]:04d}-{key[1]:02d}"
    if frequency == "quarter":
        return f"{key[0]:04d}-Q{key[1]}"
    return f"{key[0]:04d}"
```

### quant_evaluator.metrics.calendar_returns._period_key

```python
def _period_key(day: date, frequency: str) -> tuple[int, ...]:
    if frequency == "month":
        return (day.year, day.month)
    if frequency == "quarter":
        return (day.year, (day.month - 1) // 3 + 1)
    return (day.year,)
```

### quant_evaluator.metrics.calendar_returns._validated_returns

```python
def _validated_returns(returns: Any, factor_ids: Sequence[str] | None = None) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("returns must have shape (T, F) or (T,)")
    if factor_ids is not None and len(factor_ids) != values.shape[1]:
        raise ValueError("factor_ids length must equal returns factor dimension")
    if np.any(np.isfinite(values) & (values < -1.0)):
        raise ValueError("finite capital returns must be >= -1")
    return values
```

### quant_evaluator.metrics.data_quality._factor_values

```python
def _factor_values(factor_batch: FactorBatch) -> np.ndarray:
    values = np.asarray(factor_batch.values, dtype=np.float64)
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    return values
```

### quant_evaluator.metrics.data_quality._labels

```python
def _labels(label_bundle: LabelBundle, num_assets: int) -> np.ndarray:
    labels, _ = normalize_label_panel(label_bundle, num_assets)
    return labels
```

### quant_evaluator.metrics.exposure.compute_factor_loadings

```python
def compute_factor_loadings(
    factor_values: np.ndarray,
    risk_factors: np.ndarray,
    intercept: bool = True,
    min_obs: int = 10,
    *, weights=None, return_diagnostics: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute factor loadings via cross-sectional OLS regression.

    For each time period, regress factor_values on risk_factors:
    factor[t, i] = alpha[t] + sum_k(beta[t, k] * risk_factor[t, i, k]) + epsilon[t, i]

    Args:
        factor_values: Factor values (T, N)
        risk_factors: Risk factor matrix (T, N, K) where K is number of risk factors
        intercept: Include intercept in regression
        min_obs: Minimum valid observations per period

    Returns:
        (loadings, r_squared, residuals)
        loadings: shape (T, K) or (T, K+1) if intercept=True
        r_squared: shape (T,)
        residuals: shape (T, N)
    """
    factor_values=np.asarray(factor_values,dtype=float)
    risk_factors=np.asarray(risk_factors,dtype=float)
    if factor_values.ndim!=2 or risk_factors.ndim!=3 or risk_factors.shape[:2]!=factor_values.shape:
        raise ValueError("factor/risk inputs must have aligned (T,N)/(T,N,K) axes")
    if isinstance(min_obs,(bool,np.bool_)) or not isinstance(min_obs,(int,np.integer)) or min_obs<2:
        raise ValueError("min_obs must be an integer >=2")
    T, N = factor_values.shape
    K = risk_factors.shape[2]

    num_coefs = K + 1 if intercept else K
    loadings = np.full((T, num_coefs), np.nan, dtype=np.float64)
    r_squared = np.full(T, np.nan, dtype=np.float64)
    residuals = np.full((T, N), np.nan, dtype=np.float64)
    weights=np.ones((T,N)) if weights is None else np.asarray(weights,dtype=float)
    if weights.shape!=(T,N) or np.any(np.isfinite(weights)&(weights<0)):
        raise ValueError("weights must have shape (T,N) and be nonnegative")
    diagnostics=[]

    for t in range(T):
        y = factor_values[t, :]  # (N,)
        X = risk_factors[t, :, :]  # (N, K)

        # Filter finite observations
        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(weights[t]) & (weights[t]>0)
        y_valid = y[valid_mask]
        X_valid = X[valid_mask, :]

        if len(y_valid) < min_obs:
            diagnostics.append({"status":"INSUFFICIENT_OBSERVATIONS","n":len(y_valid),"rank":0,"effective_df":0})
            continue
        _,resid,diag=rank_aware_projection(X_valid,y_valid,add_intercept=intercept,weights=weights[t,valid_mask])
        diagnostics.append(diag)
        if diag["effective_df"]<2:
            continue
        if diag["rank"]==num_coefs:
            loadings[t]=diag["coefficients"]
        if diag["total_variance"]>0 and diag["rank"]>int(intercept):
            r_squared[t]=1.-diag["residual_variance"]/diag["total_variance"]
        residuals[t,valid_mask]=resid
    result=(loadings,r_squared,residuals)
    return (*result,tuple(diagnostics)) if return_diagnostics else result
```

### quant_evaluator.metrics.exposure.rank_aware_projection

```python
def rank_aware_projection(X, y, *, add_intercept=True, rcond=None, weights=None):
    """Centered/scaled weighted least squares; one projection authority.

    Inputs are already joint finite rows. Redundant controls define the same
    subspace; diagnostics distinguish their non-identifiable coefficients from
    well-defined fitted values. No in-sample projection certifies OOS utility.
    """
    X=np.asarray(X,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if X.ndim!=2 or y.ndim!=1 or X.shape[0]!=len(y) or not len(y):
        raise ValueError("projection requires aligned nonempty (N,K) and (N,) arrays")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("projection inputs must be finite joint observations")
    if not isinstance(add_intercept,(bool,np.bool_)):
        raise TypeError("add_intercept must be bool")
    if rcond is not None and (isinstance(rcond,(bool,np.bool_)) or not np.isfinite(rcond) or rcond<0 or rcond>=1):
        raise ValueError("rcond must be None or a finite relative threshold in [0,1)")
    n,k=X.shape
    w=np.ones(n) if weights is None else np.asarray(weights,dtype=np.float64)
    if w.shape!=(n,) or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError("projection weights must be finite strictly positive and aligned")
    w=w/np.max(w); w=w/w.sum()
    if add_intercept:
        x_origin=X[0]+np.sum((X-X[0])*w[:,None],axis=0)
        y_origin=y[0]+np.sum((y-y[0])*w)
        xc=X-x_origin; yc=y-y_origin
    else:
        x_origin=np.zeros(k); y_origin=0.; xc=X; yc=y
    xs=np.sqrt(np.sum(w[:,None]*xc*xc,axis=0)); xs=np.where(xs>0,xs,1.)
    ys=float(np.sqrt(np.sum(w*yc*yc))); ys=ys if ys>0 else 1.
    design=xc/xs
    if add_intercept:
        design=np.column_stack((np.ones(n),design))
    rootw=np.sqrt(w)
    u,singular,vh=np.linalg.svd(design*rootw[:,None],full_matrices=False)
    cutoff=(np.finfo(float).eps*max(design.shape) if rcond is None else rcond)
    rank=int(np.sum(singular>cutoff*singular[0])) if len(singular) else 0
    projected_coordinates=u[:,:rank].T@(yc/ys*rootw)
    beta=vh[:rank].T@(projected_coordinates/singular[:rank])
    rank=int(rank); df=n-rank
    condition=float(singular[0]/singular[rank-1]) if rank else np.inf
    # Project through orthonormal left singular vectors: an ill-conditioned
    # coefficient basis must not amplify error in the fitted subspace.
    predicted_center=(u[:,:rank]@projected_coordinates)*ys/rootw
    residual=yc-predicted_center
    scale=float(np.linalg.norm(yc))
    tolerance=64*np.finfo(np.float64).eps*max(n,k+int(add_intercept))*max(scale,float(np.linalg.norm(predicted_center)),np.finfo(float).tiny)
    status="RANK_DEFICIENT" if rank<design.shape[1] else "OK"
    if df<2:
        status="INSUFFICIENT_DF"
    elif np.linalg.norm(residual)<=tolerance:
        status="NO_RESIDUAL_VARIANCE"
        residual=np.zeros_like(residual)
    slopes=beta[int(add_intercept):]*ys/xs
    coefficients=np.r_[y_origin+beta[0]*ys-x_origin@slopes,slopes] if add_intercept else slopes
    # The centered calculation avoids cancellation in residuals at large means.
    fitted=y-residual
    diagnostics={"coefficients":coefficients,"rank":rank,"effective_df":df,"n":n,
        "condition":condition,"residual_tolerance":tolerance,"status":status,
        "estimation_scope":"SAME_DATE_DESCRIPTIVE","method_version":"centered_wls_svd.v1",
        "total_variance":float(np.sum(w*(y-(y[0]+np.sum(w*(y-y[0]))))**2)),
        "residual_variance":float(np.sum(w*residual**2))}
    return fitted,residual,diagnostics
```

### quant_evaluator.metrics.exposure_evidence._as_factor_loadings

```python
def _as_factor_loadings(panel,factor_values=None,min_obs=10,weights=None):
    if isinstance(panel,FactorLoadingSeries):
        if factor_values is not None or weights is not None:
            raise ValueError("factor loading evidence already binds factor and weights")
        return panel
    if factor_values is None:
        raise TypeError("SecurityExposurePanel is not factor evidence; factor_values are required")
    return build_factor_loading_series(panel,factor_values,min_obs=min_obs,weights=weights)
```

### quant_evaluator.metrics.exposure_evidence._panel_arrays

```python
def _panel_arrays(panel: ExposurePanel) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a panel and return (values, finite-mask) — (T, N, K)."""
    arr = np.asarray(panel.values, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"ExposurePanel.values must be (T, N, K), got {arr.ndim}D"
        )
    valid = np.isfinite(arr)
    if panel.validity is not None:
        valid &= panel.validity
        arr = np.where(valid, arr, np.nan)
    return arr, valid
```

### quant_evaluator.metrics.exposure_evidence._select_style

```python
def _select_style(panel: FactorLoadingSeries, style: str, **kwargs) -> float:
    """Signed mean exposure of one style dimension (NaN when absent)."""
    ev = compute_style_exposure_evidence(panel, absolute=False,**kwargs)
    names = list(ev.style_names)
    if style not in names:
        return float("nan")
    return float(ev.values[names.index(style)])
```

### quant_evaluator.metrics.exposure_evidence.build_factor_loading_series

```python
def build_factor_loading_series(panel, factor_values, *, factor_id="research:single-factor", min_obs=10, weights=None):
    if not isinstance(panel,ExposurePanel):
        raise TypeError("factor regression requires a SecurityExposurePanel")
    values=np.asarray(factor_values,dtype=float)
    arr,valid=_panel_arrays(panel)
    if values.ndim!=2 or values.shape!=arr.shape[:2]:
        raise ValueError("factor_values must match risk time/security axes (T,N)")
    if weights is not None and panel.regression_weights is not None:
        raise ValueError("regression weights already bound by ExposurePanel")
    w=panel.regression_weights if weights is None else np.asarray(weights,dtype=float)
    if w is None: w=np.ones(values.shape)
    if w.shape!=values.shape or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError("weights must be finite nonnegative (T,N)")
    raw,r2,_,diagnostics=compute_factor_loadings(values,arr,min_obs=min_obs,weights=w,return_diagnostics=True)
    standardized=np.full_like(raw[:,1:],np.nan)
    joint=np.isfinite(values)&valid.all(axis=2)&(w>0)
    for t in range(len(values)):
        mask=joint[t]
        if not mask.any() or not np.isfinite(r2[t]): continue
        ww=w[t,mask]/np.max(w[t,mask]); ww=ww/ww.sum(); y=values[t,mask]; z=arr[t,mask]
        yc=y-(y[0]+np.sum(ww*(y-y[0]))); zc=z-(z[0]+np.sum(ww[:,None]*(z-z[0]),axis=0))
        sy=np.sqrt(np.sum(ww*yc*yc)); sz=np.sqrt(np.sum(ww[:,None]*zc*zc,axis=0))
        if sy>0: standardized[t]=np.where(sz>0,raw[t,1:]*sz/sy,np.nan)
    safe_diagnostics=tuple({k:v for k,v in d.items() if k!="coefficients"} for d in diagnostics)
    return FactorLoadingSeries(standardized,raw[:,1:],r2,joint.sum(axis=1),tuple(panel.style_names),
        factor_id,panel.source_ref,panel.provider,panel.date_index or tuple(range(len(values))),
        "support:"+sha256(joint.tobytes()+w.tobytes()).hexdigest(),safe_diagnostics,
        "factor-values:"+sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        panel.weight_ref or ("explicit_weight_array" if weights is not None else "equal_weight"))
```

### quant_evaluator.metrics.exposure_evidence.compute_style_exposure_evidence

```python
def compute_style_exposure_evidence(
    panel: FactorLoadingSeries,
    absolute: bool = False,
    *, factor_values=None, min_obs=10, weights=None,
) -> StyleExposureEvidence:
    """Time-averaged standardized loading from factor-specific WLS evidence.

    ``absolute=False`` returns the signed mean exposure per style (a positive
    value means the factor loads positively on that style dimension across
    the sample); ``absolute=True`` returns the mean of the absolute exposures
    (magnitude — how much of the factor's variance lives on the style).

    A security risk panel requires explicit (T,N) factor_values. Never average
    the security risk panel itself as a proxy for the factor's style exposure.
    """
    panel=_as_factor_loadings(panel,factor_values,min_obs,weights)
    arr=panel.values
    T,K=arr.shape
    out = np.full(K, np.nan)
    counts = np.zeros(K, dtype=np.int64)
    for k in range(K):
        vals = arr[:, k]
        finite = vals[np.isfinite(vals)]
        counts[k] = finite.size
        if finite.size == 0:
            continue
        out[k] = float(np.mean(np.abs(finite))) if absolute else float(np.mean(finite))
    return StyleExposureEvidence(
        values=out,
        style_names=tuple(panel.style_names),
        source_ref=panel.source_ref,
        provider=panel.provider,
        counts=counts,
    )
```

### quant_evaluator.metrics.ic._pairwise_finite_mask

```python
def _pairwise_finite_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise finite mask for two arrays.

    Args:
        x: First array
        y: Second array (must be broadcastable with x)

    Returns:
        Boolean mask where both x and y are finite
    """
    return np.isfinite(x) & np.isfinite(y)
```

### quant_evaluator.metrics.ic._pearson_correlation

```python
def _pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    min_obs: int = 10,
    winsorize: float | None = None,
) -> float:
    """
    Pearson correlation with pairwise finite filtering.

    Args:
        x: Factor values (1D)
        y: Label values (1D)
        min_obs: Minimum observations required
        winsorize: Optional float in (0, 0.5); if set, winsorize both tails of
            x and y to the given fraction before correlating. None (default)
            keeps historical behavior (no winsorization).

    Returns:
        Correlation coefficient, or NaN if insufficient data or constant
    """
    mask = _pairwise_finite_mask(x, y)
    x_valid = x[mask]
    y_valid = y[mask]

    n = len(x_valid)
    if n < min_obs:
        return np.nan

    # Check for constants (robust to all-NaN / inf edges)
    if (np.nanmax(x_valid) - np.nanmin(x_valid) == 0) or (
        np.nanmax(y_valid) - np.nanmin(y_valid) == 0
    ):
        return np.nan

    if winsorize is not None:
        if not 0 < winsorize < 0.5:
            raise ValueError(
                f"winsorize must be in (0, 0.5) or None, got {winsorize!r}"
            )
        lower = winsorize
        upper = 1.0 - winsorize
        for series, index in ((x_valid, 0), (y_valid, 1)):
            q_lo, q_hi = np.nanquantile(series, [lower, upper])
            series = np.clip(series, q_lo, q_hi)
            if index == 0:
                x_win = series
            else:
                y_win = series
        x_valid, y_valid = x_win, y_win

    # Compute Pearson correlation
    corr = np.corrcoef(x_valid, y_valid)[0, 1]

    return corr
```

### quant_evaluator.metrics.ic._reject_boolean_ic_series

```python
def _reject_boolean_ic_series(ic_series: np.ndarray) -> None:
    """
    Reject boolean IC series.

    True/False silently coerces to 1.0/0.0 (e.g. a validity mask), which
    would yield a plausible-looking mean IC that carries no information.

    Raises:
        ValueError: If ic_series has boolean dtype or contains Python bools.
    """
    if ic_series.dtype == bool:
        raise ValueError(
            "ic_series must be numeric, got boolean dtype "
            "(True/False would silently coerce to 1.0/0.0)"
        )
    if ic_series.dtype == object:
        if any(isinstance(v, (bool, np.bool_)) for v in ic_series.ravel()):
            raise ValueError(
                "ic_series must be numeric, got Python bools "
                "(True/False would silently coerce to 1.0/0.0)"
            )
```

### quant_evaluator.metrics.ic._spearman_rank_correlation

```python
def _spearman_rank_correlation(x: np.ndarray, y: np.ndarray, min_obs: int = 10) -> float:
    """
    Spearman rank correlation with pairwise finite filtering and average ties.

    Args:
        x: Factor values (1D)
        y: Label values (1D)
        min_obs: Minimum observations required

    Returns:
        Rank correlation coefficient, or NaN if insufficient data,
        or constant. Statistical evidence strength is assessed separately;
        a binary nonconstant signal has a mathematically defined Spearman IC.
    """
    mask = _pairwise_finite_mask(x, y)
    x_valid = x[mask]
    y_valid = y[mask]

    n = len(x_valid)
    if n < min_obs:
        return np.nan

    # Distinct levels (x_valid/y_valid are already finite, so np.unique is exact)
    x_levels = np.unique(x_valid)
    y_levels = np.unique(y_valid)

    # Constants (all values same)
    if x_levels.size == 1 or y_levels.size == 1:
        return np.nan

    # Use scipy's spearmanr with average tie handling
    corr, _ = stats.spearmanr(x_valid, y_valid)

    return corr
```

### quant_evaluator.metrics.ic.compute_daily_ic

```python
def compute_daily_ic(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    method: str = "pearson",
    min_assets: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute daily IC series for factor batch.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle (T, N) or (T,)
        method: "pearson" or "spearman"
        min_assets: Minimum valid assets per day (default raised to 20 so that
            small cross-sections with N<20 no longer produce spurious IC).

    Returns:
        (ic_series, valid_count_series)
        ic_series: shape (T, F) with IC per day per factor
        valid_count_series: shape (T, F) with count of valid obs

    Raises:
        InvalidContractError: If shapes incompatible
        InsufficientObservations: If no valid periods found
    """
    if factor_batch.num_times != len(label_bundle.values):
        raise InvalidContractError(
            f"Factor time axis ({factor_batch.num_times}) "
            f"does not match label length ({len(label_bundle.values)})"
        )

    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown method: {method}. Must be 'pearson' or 'spearman'")

    corr_fn = _pearson_correlation if method == "pearson" else _spearman_rank_correlation

    values = factor_batch.values  # (T, N, F)
    labels, label_validity = normalize_label_panel(label_bundle, factor_batch.num_assets)

    T, N, F = values.shape
    ic_series = np.full((T, F), np.nan, dtype=np.float64)
    valid_counts = np.zeros((T, F), dtype=np.int32)

    # Compute IC per day per factor
    for t in range(T):
        for f in range(F):
            factor_t = values[t, :, f]  # (N,)
            label_t = labels[t, :]       # (N,)

            # Apply validity masks if present
            if factor_batch.validity is not None:
                factor_valid = factor_batch.validity[t, :, f]
                factor_t = np.where(factor_valid, factor_t, np.nan)

            if label_validity is not None:
                label_t = np.where(label_validity[t], label_t, np.nan)

            # Compute correlation
            ic = corr_fn(factor_t, label_t, min_obs=min_assets)
            ic_series[t, f] = ic

            # Count valid observations
            mask = _pairwise_finite_mask(factor_t, label_t)
            valid_counts[t, f] = int(np.sum(mask))

    return ic_series, valid_counts
```

### quant_evaluator.metrics.ic.compute_mean_ic

```python
def compute_mean_ic(
    ic_series: np.ndarray,
    valid_counts: Optional[np.ndarray] = None,
    min_periods: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean IC and its standard deviation across time.

    Args:
        ic_series: Daily IC series (T, F)
        valid_counts: Valid observation counts (T, F)
        min_periods: Minimum periods required for mean

    Returns:
        (mean_ic, ic_std) arrays of shape (F,)

    Raises:
        ValueError: If ic_series is boolean (or contains Python bools)
    """
    _reject_boolean_ic_series(np.asarray(ic_series))

    # Count non-NaN periods per factor
    valid_periods = np.sum(~np.isnan(ic_series), axis=0)  # (F,)

    # Compute mean and std with fully NaN-suppressed operations (all-NaN or
    # singleton-period columns are expected and produce NaN via the mask below,
    # not RuntimeWarnings).
    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_ic = np.nanmean(ic_series, axis=0)  # (F,)
            ic_std = np.nanstd(ic_series, axis=0, ddof=1)  # (F,)

    # Mask insufficient periods
    insufficient = valid_periods < min_periods
    mean_ic = np.where(insufficient, np.nan, mean_ic)
    ic_std = np.where(insufficient, np.nan, ic_std)

    # Mask non-finite results: inf in the series propagates through nanmean
    # into an infinite (invalid) mean/std — report NaN instead.
    mean_ic = np.where(np.isfinite(mean_ic), mean_ic, np.nan)
    ic_std = np.where(np.isfinite(ic_std), ic_std, np.nan)

    return mean_ic, ic_std
```

### quant_evaluator.metrics.ic_summary.compute_icir

```python
def compute_icir(
    ic_series: np.ndarray,
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Information Coefficient Information Ratio (ICIR).

    ICIR = mean(IC) / std(IC), measures consistency of IC signal.

    Args:
        ic_series: Daily IC series (T, F)
        min_periods: Minimum periods required

    Returns:
        ICIR array of shape (F,), NaN if insufficient periods or zero std
    """
    values = np.asarray(ic_series)
    _reject_boolean_ic_series(values)
    if values.ndim != 2:
        raise ValueError("ic_series must have shape (T, F)")
    if isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer)):
        raise TypeError("min_periods must be an integer")
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2 for sample standard deviation")

    values = values.astype(np.float64, copy=False)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for factor_index in range(values.shape[1]):
        finite = values[np.isfinite(values[:, factor_index]), factor_index]
        if finite.size < min_periods:
            continue
        # Raw ICIR is mean / sample std.  Only exact zero variance is
        # undefined; a small but genuine dispersion must not be thresholded
        # away because that changes the economic statistic by scale.
        if np.all(finite == finite[0]):
            continue
        std_ic = float(np.std(finite, ddof=1))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            value = float(np.mean(finite) / std_ic)
        if np.isfinite(value):
            result[factor_index] = value
    return result
```

### quant_evaluator.metrics.label_panel.normalize_label_panel

```python
def normalize_label_panel(
    label_bundle: LabelBundle,
    num_assets: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return label values and validity with a shared ``(T, N)`` shape."""
    values = np.asarray(label_bundle.values)
    if values.ndim == 1:
        values = np.broadcast_to(values[:, np.newaxis], (values.shape[0], num_assets))
    elif values.ndim == 2:
        if values.shape[1] != num_assets:
            raise InvalidContractError(
                f"Label asset axis ({values.shape[1]}) does not match "
                f"factor asset axis ({num_assets})"
            )
    else:
        raise InvalidContractError("Label values must be 1D or 2D")

    validity = label_bundle.validity
    if validity is None:
        return values, None
    validity = np.asarray(validity, dtype=bool)
    if validity.ndim == 1:
        validity = np.broadcast_to(validity[:, np.newaxis], (values.shape[0], num_assets))
    elif validity.ndim == 2:
        if validity.shape != values.shape:
            raise InvalidContractError(
                f"Label validity shape {validity.shape} does not match "
                f"normalized label shape {values.shape}"
            )
    else:
        raise InvalidContractError("Label validity must be 1D or 2D")
    return values, validity
```

### quant_evaluator.metrics.long_only._matrix

```python
def _matrix(returns):
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim == 1:
        return values[:, None], True
    if values.ndim != 2:
        raise ValueError("returns must be (T,) or (T,F)")
    return values, False
```

### quant_evaluator.metrics.multiple_testing._validate_alpha

```python
def _validate_alpha(alpha: float) -> float:
    """QE-METRIC P0-14: fail-closed validation for the alpha level."""
    try:
        alpha_val = float(alpha)
    except (TypeError, ValueError):
        raise ValueError(f"alpha must be a finite float in (0, 1), got {alpha!r}")
    if not np.isfinite(alpha_val) or not (0.0 < alpha_val < 1.0):
        raise ValueError(
            f"alpha must be strictly inside (0, 1), got {alpha_val!r}"
        )
    return alpha_val
```

### quant_evaluator.metrics.multiple_testing._validate_p_values

```python
def _validate_p_values(p_values: np.ndarray) -> np.ndarray:
    """QE-METRIC P0-14: fail-closed validation for p-value inputs.

    Rejects empty arrays and finite p-values outside [0, 1]. Non-finite
    entries (NaN, +-inf) are allowed and treated as missing downstream.

    Returns the flattened array (a read view) for processing.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        raise ValueError(
            "p_values must be non-empty; got an empty array "
            f"(shape {p.shape})"
        )
    finite = np.isfinite(p)
    out_of_range = finite & ((p < 0.0) | (p > 1.0))
    n_bad = int(np.sum(out_of_range))
    if n_bad > 0:
        first_idx = int(np.nonzero(out_of_range.ravel())[0][0])
        raise ValueError(
            f"p_values contains {n_bad} finite value(s) outside [0, 1] "
            f"(first at flat index {first_idx}: "
            f"{p.ravel()[first_idx]!r}); these are not valid p-values"
        )
    return p
```

### quant_evaluator.metrics.portfolio_stats._validate_missing_return_policy

```python
def _validate_missing_return_policy(policy: str) -> str:
    if policy not in _MISSING_RETURN_POLICIES:
        raise ValueError(
            f"missing_return_policy must be one of "
            f"{_MISSING_RETURN_POLICIES}, got {policy!r}"
        )
    return policy
```

### quant_evaluator.metrics.portfolio_stats.equal_gross_long_short_returns

```python
def equal_gross_long_short_returns(long_members, short_members, forward_returns, *, cost_rate=0.0,
                                  missing_return_policy="drop"):
    """100% gross target-weight portfolio; full-notional turnover including entry.

    A missing selected return invalidates the day under 'drop', not membership.
    'zero_fill' is an explicit flat-mark assumption, never a reweighting rule.
    """
    weights = equal_gross_weights(long_members, short_members)
    returns = np.asarray(forward_returns, float)
    if returns.shape != weights.shape or not np.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("invalid returns shape or commission")
    _validate_missing_return_policy(missing_return_policy)
    missing = ((weights != 0) & ~np.isfinite(returns)).any(axis=1)
    if missing_return_policy == "fail" and missing.any():
        raise ValueError("missing return on a selected position")
    pnl = (weights * np.where(np.isfinite(returns), returns, 0.)).sum(axis=1)
    turnover = np.abs(np.diff(np.vstack([np.zeros((1, weights.shape[1])), weights]), axis=0)).sum(axis=1)
    pnl -= cost_rate * turnover
    if missing_return_policy == "drop":
        pnl[missing] = np.nan
    return pnl
```

### quant_evaluator.metrics.portfolio_stats.equal_gross_weights

```python
def equal_gross_weights(long_members, short_members):
    """Equal absolute weight per selected stock; 100% gross including full short margin.

    Masks must be signal-time decisions. No forward-return filter is used.
    Both legs are required; otherwise the portfolio remains in cash.
    """
    long_members, short_members = np.asarray(long_members, bool), np.asarray(short_members, bool)
    if long_members.shape != short_members.shape or long_members.ndim != 2:
        raise ValueError("membership masks must have matching (time, asset) shapes")
    if np.any(long_members & short_members):
        raise ValueError("an asset cannot be both long and short")
    total = (long_members.sum(axis=1) + short_members.sum(axis=1))[:, None]
    active = (long_members.any(axis=1) & short_members.any(axis=1))[:, None]
    return np.divide(long_members.astype(float)-short_members, total,
                     out=np.zeros(long_members.shape, float), where=active & (total > 0))
```

### quant_evaluator.metrics.predictive._as_series

```python
def _as_series(ic_series: np.ndarray) -> np.ndarray:
    """Coerce to a float64 (T, F) array."""
    s = np.asarray(ic_series, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    return s
```

### quant_evaluator.metrics.predictive._autocorr_lag

```python
def _autocorr_lag(s: np.ndarray, lag: int) -> np.ndarray:
    """Pairwise-finite autocorrelation at ``lag`` on the original axis, (F,)."""
    T, F = s.shape
    if lag >= T:
        return np.full(F, np.nan)
    finite = np.isfinite(s)
    pair = finite[lag:, :] & finite[:-lag, :]
    n = np.sum(pair, axis=0).astype(np.float64)
    x_prev = np.where(pair, s[:-lag, :], 0.0)
    x_curr = np.where(pair, s[lag:, :], 0.0)
    s_prev = np.sum(x_prev, axis=0)
    s_curr = np.sum(x_curr, axis=0)
    s_pp = np.sum(x_prev * x_prev, axis=0)
    s_cc = np.sum(x_curr * x_curr, axis=0)
    s_pc = np.sum(x_prev * x_curr, axis=0)
    num = n * s_pc - s_prev * s_curr
    denom = np.sqrt((n * s_pp - s_prev * s_prev) * (n * s_cc - s_curr * s_curr))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / denom
    corr = np.where((denom <= 0) | (~np.isfinite(denom)) | (n < 2), np.nan, corr)
    return corr
```

### quant_evaluator.metrics.predictive._mean_of_period_means

```python
def _mean_of_period_means(s: np.ndarray, period: str, time_index=None) -> np.ndarray:
    means = _period_means(s, period, time_index)
    with np.errstate(invalid="ignore"):
        return np.nanmean(means, axis=0)
```

### quant_evaluator.metrics.predictive._period_means

```python
def _period_means(
    s: np.ndarray,
    period: str,
    time_index: Optional[Sequence] = None,
) -> np.ndarray:
    """Per-period mean IC, shape (n_periods, F).

    Uses the calendar period when ``time_index`` is provided, otherwise
    contiguous blocks of ``_BLOCK_DAYS[period]`` trading days.
    """
    if time_index is not None and len(time_index) == s.shape[0]:
        try:
            import pandas as pd

            idx = pd.to_datetime(list(time_index))
            df = pd.DataFrame(s, index=idx)
            if period == "year":
                grouped = df.groupby(df.index.year)
            elif period == "month":
                grouped = df.groupby([df.index.year, df.index.month])
            else:  # quarter
                grouped = df.groupby(df.index.to_period("Q"))
            return grouped.mean().to_numpy(dtype=np.float64)  # (n_periods, F)
        except Exception:  # noqa: BLE001 - fall back to block grouping
            pass
    block = _BLOCK_DAYS.get(period, 63)
    T, F = s.shape
    n_blocks = max(1, int(np.ceil(T / block)))
    pad = n_blocks * block - T
    if pad > 0:
        s = np.vstack([s, np.full((pad, F), np.nan)])
    blocks = s.reshape(n_blocks, block, F)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=1)  # (n_blocks, F)
```

### quant_evaluator.metrics.predictive._recent_mean

```python
def _recent_mean(s: np.ndarray, n_days: int) -> np.ndarray:
    tail = s[-n_days:, :]
    with np.errstate(invalid="ignore"):
        return np.nanmean(tail, axis=0)
```

### quant_evaluator.metrics.predictive._rolling_mean_ir

```python
def _rolling_mean_ir(s: np.ndarray, window: int, min_periods: int):
    """Rolling mean and IR (mean/std) over a trailing window, (T, F) each."""
    T, F = s.shape
    finite = np.isfinite(s)
    x = np.where(finite, s, 0.0)
    pref = np.concatenate([np.zeros((1, F)), np.cumsum(x, axis=0)], axis=0)
    fpref = np.concatenate([np.zeros((1, F)), np.cumsum(finite, axis=0)], axis=0)
    pref_sq = np.concatenate([np.zeros((1, F)), np.cumsum(x * x, axis=0)], axis=0)
    ends = np.arange(1, T + 1)
    start = np.maximum(ends - window, 0)
    cnt = fpref[ends] - fpref[start]
    ssum = pref[ends] - pref[start]
    ssq = pref_sq[ends] - pref_sq[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ssum / np.maximum(cnt, 1.0)
        var = ssq - cnt * mean * mean
        std = np.sqrt(np.maximum(var, 0.0) / np.maximum(cnt - 1, 1.0))
        ir = mean / std
    mean = np.where(cnt >= min_periods, mean, np.nan)
    ir = np.where((cnt >= max(2, min_periods)) & (std > 1e-12), ir, np.nan)
    return mean, ir
```

### quant_evaluator.metrics.predictive._valid_counts

```python
def _valid_counts(s: np.ndarray) -> np.ndarray:
    return np.sum(np.isfinite(s), axis=0)
```

### quant_evaluator.metrics.predictive._worst_period

```python
def _worst_period(s: np.ndarray, period: str, time_index=None) -> np.ndarray:
    means = _period_means(s, period, time_index)
    with np.errstate(invalid="ignore"):
        return np.nanmin(means, axis=0)
```

### quant_evaluator.metrics.quality._valid_pair_mask

```python
def _valid_pair_mask(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> np.ndarray:
    """Return the pairwise-valid boolean mask of shape (T, N, F).

    A (t, n, f) cell is valid when the factor value, the normalized label
    value, the factor validity mask (if present), and the label validity
    mask (if present) are all finite/true.
    """
    if factor_batch.num_times != len(label_bundle.values):
        raise InvalidContractError(
            f"Factor time axis ({factor_batch.num_times}) "
            f"does not match label length ({len(label_bundle.values)})"
        )

    values = factor_batch.values  # shape: (T, N, F)
    labels, label_validity = normalize_label_panel(label_bundle, factor_batch.num_assets)

    factor_finite = np.isfinite(values)  # (T, N, F)
    label_finite = np.isfinite(labels)   # (T, N)

    label_finite_expanded = label_finite[:, :, np.newaxis]  # (T, N, 1)
    valid_pairs = factor_finite & label_finite_expanded    # (T, N, F)

    if factor_batch.validity is not None:
        valid_pairs = valid_pairs & factor_batch.validity
    if label_validity is not None:
        valid_pairs = valid_pairs & label_validity[:, :, np.newaxis]

    return valid_pairs
```

### quant_evaluator.metrics.quality.compute_coverage_per_factor

```python
def compute_coverage_per_factor(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compute a per-factor coverage report.

    Unlike the deprecated :func:`compute_coverage`, this never averages
    across factor columns and actually USES ``min_assets``: days on which
    a factor has fewer than ``min_assets`` jointly valid (factor, label)
    observations are counted in ``days_below_min_assets``.

    Args:
        factor_batch: Input factor batch (T, N, F)
        label_bundle: Input label bundle
        min_assets: Minimum valid assets per day for a day to count as
            a "valid day" (enforced, not diagnostic)

    Returns:
        Dict keyed by factor_id with entries:
            {"coverage": float, "num_valid": int, "num_total": int,
             "valid_days": int, "days_below_min_assets": int}

    Raises:
        InvalidContractError: If shapes are incompatible
    """
    valid_pairs = _valid_pair_mask(factor_batch, label_bundle)
    per_day_counts = np.sum(valid_pairs, axis=1)  # (T, F)
    num_times = valid_pairs.shape[0]

    report: Dict[str, Dict[str, float]] = {}
    for index, factor_id in enumerate(factor_batch.factor_ids):
        counts = per_day_counts[:, index]
        num_valid = int(counts.sum())
        num_total = int(valid_pairs[:, :, index].size)
        valid_days = int(np.sum(counts >= min_assets))
        report[factor_id] = {
            "coverage": num_valid / num_total if num_total > 0 else 0.0,
            "num_valid": num_valid,
            "num_total": num_total,
            "valid_days": valid_days,
            "days_below_min_assets": int(num_times) - valid_days,
        }
    return report
```

### quant_evaluator.metrics.quantile._percentile_boundaries

```python
def _percentile_boundaries(
    v_finite: np.ndarray,
    n_quantiles: int,
) -> np.ndarray:
    """Internal boundary computation shared by every assign_quantiles* path.

    QE-Q-P0-002: all quantile binning implementations (NumPy reference, fast,
    Numba, Polars, CuPy) must derive bins from the same percentile boundaries,
    so the boundary values themselves are computed in exactly one place.

    Boundaries are interpolated on the sorted values with the formula
    ``sv[lo] + frac * (sv[lo+1] - sv[lo])`` at position
    ``pos = (b+1)/n_quantiles * (n-1)``.  Positions within 1e-9 of an integer
    snap to the exact sorted value: np.percentile can land one ulp off the
    data value there (its percentile/100 rounding), which would silently flip
    the tie policy for the value sitting exactly on the boundary.  The snap
    keeps every backend bit-identical at the only positions where exact ties
    are possible.
    """
    sv = np.sort(v_finite)
    n = sv.shape[0]
    boundaries = np.empty(n_quantiles - 1, dtype=np.float64)
    for b in range(n_quantiles - 1):
        pos = (b + 1) / n_quantiles * (n - 1)
        lo = int(pos)
        frac = pos - lo
        if lo >= n - 1:
            boundaries[b] = sv[n - 1]
        elif frac < 1e-9:
            boundaries[b] = sv[lo]
        elif frac > 1.0 - 1e-9:
            boundaries[b] = sv[lo + 1]
        else:
            boundaries[b] = sv[lo] + frac * (sv[lo + 1] - sv[lo])
    return boundaries
```

### quant_evaluator.metrics.quantile._searchsorted_bins

```python
def _searchsorted_bins(
    boundaries: np.ndarray,
    v_finite: np.ndarray,
    n_quantiles: int,
    policy: QuantileTiePolicy,
) -> np.ndarray:
    """Internal searchsorted binning honoring the tie policy.

    MIN -> side='left' (value == boundary goes to the LOWER bin)
    MAX -> side='right' (value == boundary goes to the HIGHER bin)
    """
    if policy == QuantileTiePolicy.MIN:
        q_bins = np.searchsorted(boundaries, v_finite, side='left')
    else:  # QuantileTiePolicy.MAX
        q_bins = np.searchsorted(boundaries, v_finite, side='right')
    return np.clip(q_bins, 0, n_quantiles - 1)
```

### quant_evaluator.metrics.quantile._validate_quantile_count

```python
def _validate_quantile_count(n_quantiles):
    if isinstance(n_quantiles, (bool, np.bool_)) or not isinstance(n_quantiles, (int, np.integer)) or n_quantiles < 1:
        raise ValueError("n_quantiles must be a positive integer")
```

### quant_evaluator.metrics.quantile.assign_quantiles_batch

```python
def assign_quantiles_batch(
    values: np.ndarray,
    n_quantiles: int = 5,
    method: str = "max",
) -> np.ndarray:
    """
    Ultra-fast batch quantile assignment with minimal Python loops.

    QE-Q-P0-001: Enforces tie-breaking policy.

    Processes entire time×asset×factor tensor with vectorized operations.
    Best performance for large batches.

    Args:
        values: Input values shape (T, N, F)
        n_quantiles: Number of quantiles
        method: Tie-breaking policy ('min' or 'max'). Default 'max'.

    Returns:
        Quantile assignments shape (T, N, F), -1 for NaN
    """
    policy = validate_tie_policy(method)
    _validate_quantile_count(n_quantiles)
    original_ndim = values.ndim
    if original_ndim not in (2, 3):
        raise ValueError("quantile input must be T x N or T x N x F")
    if values.ndim == 2:
        values = values[:, :, np.newaxis]

    T, N, F = values.shape
    quantiles = np.full((T, N, F), -1, dtype=np.int32)

    # Process all time-factor pairs efficiently
    for t in range(T):
        # Vectorize across all factors at once
        v_t = values[t, :, :]  # (N, F)
        finite_mask_t = np.isfinite(v_t)  # (N, F)

        for f in range(F):
            mask = finite_mask_t[:, f]
            n_finite = np.sum(mask)

            if n_finite < n_quantiles:
                continue

            v_finite = v_t[mask, f]

            # QE-Q-P0-002: same percentile boundaries + tie policy as the
            # reference implementation (shared helpers).
            boundaries = _percentile_boundaries(v_finite, n_quantiles)
            q_bins = _searchsorted_bins(boundaries, v_finite, n_quantiles, policy)

            quantiles[t, mask, f] = q_bins

    return quantiles[:, :, 0] if original_ndim == 2 else quantiles
```

### quant_evaluator.metrics.quantile.compute_quantile_returns

```python
def compute_quantile_returns(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average returns per quantile per period.

    Optimized with vectorized quantile assignment and bincount aggregation.

    Args:
        factor_batch: Factor values (T, N, F)
        label_bundle: Forward returns (T, N)
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile

    Returns:
        (quantile_returns, quantile_counts)
        quantile_returns: shape (T, n_quantiles, F)
        quantile_counts: shape (T, n_quantiles, F)
    """
    values = factor_batch.values  # (T, N, F)
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    labels, label_validity = normalize_label_panel(
        label_bundle, factor_batch.num_assets
    )
    if label_validity is not None:
        labels = np.where(label_validity, labels, np.nan)

    T, N, F = values.shape
    quantile_returns = np.full((T, n_quantiles, F), np.nan, dtype=np.float64)
    quantile_counts = np.zeros((T, n_quantiles, F), dtype=np.int32)

    # Process each factor independently
    for f in range(F):
        # Assign quantiles for this factor across all time periods
        q_assignments_f = assign_quantiles_batch(values[:, :, f], n_quantiles=n_quantiles)  # (T, N)

        # Aggregate per time period
        for t in range(T):
            label_t = labels[t, :]
            valid_labels = np.isfinite(label_t)
            q_t_f = q_assignments_f[t, :]  # (N,)

            # Combined mask: valid quantile assignment AND valid label
            valid_mask = (q_t_f >= 0) & valid_labels

            if not np.any(valid_mask):
                continue

            q_valid = q_t_f[valid_mask]
            label_valid = label_t[valid_mask]

            # Use bincount for fast aggregation - much faster than loop over quantiles
            # bincount sums, so we sum labels and divide by counts
            counts = np.bincount(q_valid, minlength=n_quantiles)
            sums = np.bincount(q_valid, weights=label_valid, minlength=n_quantiles)

            # Apply min_assets filter and compute means
            sufficient_mask = counts >= min_assets
            quantile_counts[t, :, f] = counts
            quantile_returns[t, sufficient_mask, f] = sums[sufficient_mask] / counts[sufficient_mask]

    return quantile_returns, quantile_counts
```

### quant_evaluator.metrics.quantile.compute_quantile_returns_fast

```python
def compute_quantile_returns_fast(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
    use_numba: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute quantile returns with automatic selection of fastest implementation.

    Automatically uses Numba JIT if available (2-5x faster), otherwise falls back
    to optimized numpy implementation.

    Args:
        factor_batch: Factor values (T, N, F)
        label_bundle: Forward returns (T, N)
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile
        use_numba: If True and numba available, use JIT version (default: True)

    Returns:
        (quantile_returns, quantile_counts)
        quantile_returns: shape (T, n_quantiles, F)
        quantile_counts: shape (T, n_quantiles, F)
    """
    if use_numba and _NUMBA_AVAILABLE:
        return compute_quantile_returns_numba(
            factor_batch, label_bundle, n_quantiles, min_assets
        )
    else:
        return compute_quantile_returns(
            factor_batch, label_bundle, n_quantiles, min_assets
        )
```

### quant_evaluator.metrics.quantile_numba.compute_quantile_returns_numba

```python
def compute_quantile_returns_numba(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    n_quantiles: int = 5,
    min_assets: int = 10,
    method: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-performance quantile return computation using Numba JIT.

    QE-Q-P0-001: Enforces tie-breaking policy.
    Achieves 5-10x speedup through JIT compilation and parallel processing.

    Args:
        factor_batch: Factor values (T, N, F)
        label_bundle: Forward returns (T, N)
        n_quantiles: Number of quantiles
        min_assets: Minimum assets per quantile
        method: Tie-breaking policy ('min' or 'max'). Default 'max'.

    Returns:
        (quantile_returns, quantile_counts)
        quantile_returns: shape (T, n_quantiles, F)
        quantile_counts: shape (T, n_quantiles, F)
    """
    # Numba is optional: when absent, the no-op ``jit`` decorator and
    # ``prange``->``range`` alias above make the JIT kernels run as plain
    # Python, so this still returns a correct result (just slower).
    from quant_evaluator.contracts.quantile_policy import validate_tie_policy
    policy = validate_tie_policy(method)

    values = factor_batch.values
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    labels, label_validity = normalize_label_panel(
        label_bundle, factor_batch.num_assets
    )
    if label_validity is not None:
        labels = np.where(label_validity, labels, np.nan)
    labels = np.ascontiguousarray(labels)

    # Assign quantiles with JIT
    quantiles = _assign_quantiles_jit(values, n_quantiles, policy.value)

    # Compute returns with JIT and parallelization
    quantile_returns, quantile_counts = _compute_quantile_returns_jit(
        values, labels, quantiles, n_quantiles, min_assets
    )

    return quantile_returns, quantile_counts
```

### quant_evaluator.metrics.quantile_shape._as_matrix

```python
def _as_matrix(qr: np.ndarray) -> np.ndarray:
    """Coerce to a float64 (n_quantiles, F) matrix."""
    m = np.asarray(qr, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    return m
```

### quant_evaluator.metrics.quantile_shape._finite_columns

```python
def _finite_columns(m: np.ndarray) -> np.ndarray:
    """Boolean (F,) mask of columns with >= 2 finite quantile returns."""
    return np.sum(np.isfinite(m), axis=0) >= 2
```

### quant_evaluator.metrics.risk.drawdown_analysis.compute_drawdown_series

```python
def compute_drawdown_series(
    returns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute drawdown series from returns.

    Args:
        returns: Return series (T,) or (T, F)

    Returns:
        (drawdown_series, cumulative_returns, running_max)
        drawdown_series: Drawdown at each time (negative values), shape (T,) or (T, F)
        cumulative_returns: Cumulative wealth curve, shape (T,) or (T, F)
        running_max: Running maximum wealth, shape (T,) or (T, F)
    """
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    if np.any(returns[np.isfinite(returns)] < -1.0):
        raise ValueError("returns below -100% require an explicit negative-capital contract")
    # Replace NaN with 0 for cumulative computation
    returns_filled = np.where(np.isfinite(returns), returns, 0.0)

    # Cumulative wealth curve
    cum_returns = np.cumprod(1.0 + returns_filled, axis=0)

    # Initial capital is a high-water mark too: a loss on the first
    # observation must count even before an observed wealth peak exists.
    running_max = np.maximum(1.0, np.maximum.accumulate(cum_returns, axis=0))

    # Zero NAV is an absorbing default with an observed 100% loss, not missing
    # evidence. Negative capital is rejected above before compounding.
    drawdown_series = (cum_returns - running_max) / running_max

    if squeeze:
        return drawdown_series[:, 0], cum_returns[:, 0], running_max[:, 0]
    else:
        return drawdown_series, cum_returns, running_max
```

### quant_evaluator.metrics.risk.drawdown_analysis.drawdown_events

```python
def drawdown_events(returns: np.ndarray) -> List[Dict]:
    """Single-pass, observation-aligned events (definition version 0.3).

    A missing return makes subsequent NAV unknown, not a flat day. Such an
    event is censored at the first valuation gap; no recovery is inferred
    across it. Durations are grid intervals, never finite-observation counts.
    """
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("drawdown_events requires 1D returns")
    if np.any(values[np.isfinite(values)] < -1):
        raise ValueError("returns below -100% require an explicit negative-capital contract")
    events = []
    wealth = high = 1.0
    peak = -1
    event = None
    for t, value in enumerate(values):
        if not np.isfinite(value):
            if event is None:
                event = dict(peak_idx=peak, start_idx=t, trough_idx=-1,
                             drawdown=np.nan)
            event.update(recovery_idx=-1, end_idx=len(values) - 1,
                         censored=True, status="INVALID_VALUATION",
                         first_missing_idx=t, duration=len(values) - event["start_idx"],
                         valid_observations=int(np.isfinite(values[event["start_idx"]:]).sum()))
            events.append(event)
            return events
        wealth *= 1 + value
        dd = max(0., 1 - wealth / high)
        # Same exact highwater contract as maximum-drawdown peak indices.
        # A measured underwater loss cannot be relabelled as a new peak.
        if wealth >= high:
            if event is not None:
                event.update(recovery_idx=t, end_idx=t, censored=False,
                             status="RECOVERED", duration=t - event["start_idx"],
                             valid_observations=t - event["start_idx"] + 1)
                events.append(event)
                event = None
            high, peak = max(high, wealth), t
        else:
            if event is None:
                event = dict(peak_idx=peak, start_idx=t, trough_idx=t, drawdown=dd)
            if dd > event["drawdown"]:
                event.update(trough_idx=t, drawdown=dd)
    if event is not None:
        event.update(recovery_idx=-1, end_idx=len(values) - 1, censored=True,
                     status="DEFAULTED" if wealth == 0 else "ACTIVE",
                     duration=len(values) - event["start_idx"],
                     valid_observations=len(values) - event["start_idx"])
        events.append(event)
    return events
```

### quant_evaluator.metrics.risk.var_cvar._validate_confidence_level

```python
def _validate_confidence_level(confidence_level: float) -> None:
    """Fail closed: confidence_level must be strictly inside (0, 1).

    confidence_level == 1.0 would divide by zero in parametric CVaR
    (inf VaR), and <= 0 is meaningless. Raise ValueError otherwise.
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )
```

### quant_evaluator.metrics.risk.var_cvar.compute_cvar

```python
def compute_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "historical",
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR is the expected loss given that loss exceeds VaR threshold.

    Args:
        returns: Return series (T,) or (T, F)
        confidence_level: Confidence level
        method: "historical" or "parametric"
        min_periods: Minimum periods required

    Returns:
        CVaR (positive loss value), scalar or shape (F,)
    """
    _validate_confidence_level(confidence_level)

    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    T, F = returns.shape
    cvar = np.full(F, np.nan)

    if method == "historical":
        # Historical CVaR: mean of returns below VaR threshold
        quantile = 1.0 - confidence_level

        for f in range(F):
            ret_f = returns[:, f]
            valid = np.isfinite(ret_f)
            n_valid = np.sum(valid)

            if n_valid < min_periods:
                continue

            ret_valid = ret_f[valid]

            cvar[f] = max(0.0, empirical_expected_shortfall(ret_valid, confidence_level))

    elif method == "parametric":
        # Parametric CVaR for normal distribution
        z = stats.norm.ppf(1.0 - confidence_level)
        pdf_at_z = stats.norm.pdf(z)

        for f in range(F):
            ret_f = returns[:, f]
            valid = np.isfinite(ret_f)
            n_valid = np.sum(valid)

            if n_valid < min_periods:
                continue

            ret_valid = ret_f[valid]

            mean_ret = np.mean(ret_valid)
            std_ret = np.std(ret_valid, ddof=1)

            if not np.isfinite(std_ret) or std_ret <= 0:
                continue

            # CVaR = mean + std * E[Z | Z < z] = mean - std * pdf(z) / (1 - CL)
            cvar_value = mean_ret - std_ret * pdf_at_z / (1.0 - confidence_level)
            cvar[f] = -cvar_value if cvar_value < 0 else 0.0

    else:
        raise ValueError(f"Unknown CVaR method: {method}")

    return cvar[0] if squeeze else cvar
```

### quant_evaluator.metrics.risk.var_cvar.empirical_expected_shortfall

```python
def empirical_expected_shortfall(returns, confidence_level=0.95):
    """Signed empirical loss ES v2: fixed tail mass, fractional boundary.

    Equal observation weights; finite observations only. This is a descriptive
    same-period estimate, not an inference/adequate-tail-sample certificate.
    compute_cvar is the separately documented positive-loss-clamped display.
    """
    _validate_confidence_level(confidence_level)
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("empirical ES requires a one-dimensional return sample")
    losses = np.sort(-values[np.isfinite(values)])[::-1]
    if not len(losses):
        return float("nan")
    mass = (1.0 - confidence_level) * len(losses)
    full = int(np.floor(mass))
    fraction = mass - full
    return float((losses[:full].sum() + (fraction * losses[full] if full < len(losses) else 0.)) / mass)
```

### quant_evaluator.metrics.robustness._contiguous_sample

```python
def _contiguous_sample(column):
    positions = np.flatnonzero(np.isfinite(column))
    if (np.isinf(column).any() or not len(positions)
            or positions[-1] - positions[0] + 1 != len(positions)):
        return np.empty(0, dtype=float)
    return column[positions[0]:positions[-1] + 1]
```

### quant_evaluator.metrics.robustness._validate_bootstrap_policy

```python
def _validate_bootstrap_policy(t, block_length, num_bootstrap, confidence_level, random_seed):
    for name, value in (("block_length", block_length), ("num_bootstrap", num_bootstrap)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if block_length > t:
        raise ValueError("block_length exceeds time axis")
    if num_bootstrap < 2:
        raise ValueError("num_bootstrap must be at least two")
    if (isinstance(confidence_level, (bool, np.bool_))
            or not isinstance(confidence_level, (int, float, np.integer, np.floating))
            or not np.isfinite(confidence_level) or not 0 < confidence_level < 1):
        raise ValueError("confidence_level must be finite and in (0, 1)")
    if random_seed is not None and (isinstance(random_seed, (bool, np.bool_))
            or not isinstance(random_seed, (int, np.integer)) or random_seed < 0):
        raise ValueError("random_seed must be a nonnegative integer or None")
```

### quant_evaluator.metrics.robustness._validate_hac_policy

```python
def _validate_hac_policy(max_lag, kernel):
    if isinstance(max_lag, (bool, np.bool_)) or not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError("max_lag must be a nonnegative integer")
    if kernel not in ("bartlett", "uniform"):
        raise ValueError("kernel must be bartlett or uniform")
```

### quant_evaluator.metrics.robustness.compute_block_bootstrap_ci

```python
def compute_block_bootstrap_ci(
    ic_series: np.ndarray,
    block_length: int = 10,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence interval via block bootstrap.

    Block bootstrap preserves temporal dependence structure by resampling
    contiguous blocks of observations.
    All factors share original-time-axis draws. A column with missing or
    infinite observations is insufficient (NaN); no calendar compression occurs.

    Args:
        ic_series: Daily IC series (T, F)
        block_length: Length of each bootstrap block
        num_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed

    Returns:
        (ci_lower, ci_upper)
        ci_lower: shape (F,) - lower bound of CI for mean IC
        ci_upper: shape (F,) - upper bound of CI for mean IC
    """
    ic_series = np.asarray(ic_series, dtype=float)
    if ic_series.ndim == 1:
        ic_series = ic_series[:, None]
    if ic_series.ndim != 2:
        raise ValueError("bootstrap requires a time series or T x F matrix")
    T, F = ic_series.shape
    _validate_bootstrap_policy(T, block_length, num_bootstrap, confidence_level, random_seed)

    # Local Generator: never touch the global numpy RNG state.
    rng = np.random.default_rng(random_seed)
    # One original-time-axis draw matrix, independent of factor order/count.
    # This direct multi-candidate producer requires a complete common calendar.
    # Gapped columns remain NaN; callers must not compress them before entry.
    num_blocks = (T + block_length - 1) // block_length
    shared_starts = rng.integers(0, T - block_length + 1,
                                size=(num_bootstrap, num_blocks))

    alpha = 1.0 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    ci_lower = np.full(F, np.nan, dtype=np.float64)
    ci_upper = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ic_f = ic_series[:, f]
        if not np.isfinite(ic_f).all():
            continue
        valid_ic = ic_f

        if len(valid_ic) < block_length * 2:
            continue

        n = len(valid_ic)
        num_blocks = (n + block_length - 1) // block_length

        bootstrap_means = np.full(num_bootstrap, np.nan, dtype=np.float64)

        for b in range(num_bootstrap):
            # Sample blocks with replacement (local Generator)
            block_starts = shared_starts[b]

            # Reconstruct bootstrap sample
            bootstrap_sample = []
            for start in block_starts:
                bootstrap_sample.extend(valid_ic[start:start + block_length])

            bootstrap_sample = np.array(bootstrap_sample[:n])  # Trim to original length
            bootstrap_means[b] = np.mean(bootstrap_sample)

        # Compute percentiles
        ci_lower[f] = np.percentile(bootstrap_means, lower_percentile)
        ci_upper[f] = np.percentile(bootstrap_means, upper_percentile)

    return ci_lower, ci_upper
```

### quant_evaluator.metrics.robustness.compute_hac_tstat

```python
def compute_hac_tstat(
    ic_series: np.ndarray,
    max_lag: int = 5,
    kernel: str = "bartlett",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HAC-robust t-statistic for IC series.

    Tests H0: mean(IC) = 0 using HAC standard errors.
    More conservative than standard t-test when IC is autocorrelated.

    Args:
        ic_series: Daily IC series (T, F)
        max_lag: Maximum lag for HAC estimation
        kernel: Kernel type

    Returns:
        (t_stat_hac, se_hac)
        t_stat_hac: shape (F,) - HAC-robust t-statistics
        se_hac: shape (F,) - HAC standard errors
    """
    _validate_hac_policy(max_lag, kernel)
    ic_series = np.asarray(ic_series, dtype=float)
    if ic_series.ndim == 1:
        ic_series = ic_series[:, None]
    if ic_series.ndim != 2:
        raise ValueError("HAC requires a time series or T x F matrix")
    T, F = ic_series.shape

    t_stat_hac = np.full(F, np.nan, dtype=np.float64)
    se_hac = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ic_f = ic_series[:, f]
        valid_ic = _contiguous_sample(ic_f)

        if len(valid_ic) < max_lag + 10:
            continue

        # Mean IC
        mean_ic = np.mean(valid_ic)
        n = len(valid_ic)

        # HAC variance
        hac_var = compute_hac_variance(
            valid_ic.reshape(-1, 1), max_lag=max_lag, kernel=kernel
        )[0]

        if hac_var <= 0 or np.isnan(hac_var):
            continue

        # HAC standard error
        se = np.sqrt(hac_var)
        se_hac[f] = se

        # HAC t-statistic
        t_stat_hac[f] = mean_ic / se

    return t_stat_hac, se_hac
```

### quant_evaluator.metrics.robustness.compute_hac_variance

```python
def compute_hac_variance(
    series: np.ndarray,
    max_lag: int = 5,
    kernel: str = "bartlett",
) -> np.ndarray:
    """
    Compute Heteroskedasticity and Autocorrelation Consistent (HAC) variance.

    Newey-West HAC variance estimator for time series with autocorrelation.
    Uses weighted sum of autocovariances with kernel weighting.
    Empty endpoints are trimmed; internal missing dates or infinity return NaN.
    All lag autocovariances use denominator n (Newey-West v2). The output
    is variance_of_sample_mean, not long_run_variance. No df correction.

    Args:
        series: Time series (T,) or (T, F)
        max_lag: Maximum lag for HAC estimation
        kernel: Kernel type - "bartlett" (triangular) or "uniform"

    Returns:
        hac_var: shape (F,) - HAC variance estimate per factor
    """
    _validate_hac_policy(max_lag, kernel)
    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    if series.ndim != 2:
        raise ValueError("HAC requires a time series or T x F matrix")

    T, F = series.shape

    if kernel not in ("bartlett", "uniform"):
        raise ValueError(f"Unknown kernel: {kernel}, must be 'bartlett' or 'uniform'")

    hac_var = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ts = series[:, f]
        # Trim empty endpoints only. Internal gaps cannot acquire new lags.
        valid_ts = _contiguous_sample(ts)

        if len(valid_ts) < max_lag + 10:
            continue

        # Demean
        ts_demean = valid_ts - np.mean(valid_ts)
        n = len(ts_demean)

        # Compute lag-0 autocovariance (variance)
        gamma_0 = np.mean(ts_demean ** 2)

        # HAC variance: gamma_0 + 2 * sum(weight(lag) * gamma(lag))
        hac_est = gamma_0

        for lag in range(1, max_lag + 1):
            if lag >= n:
                break

            # Autocovariance at lag
            gamma_lag = np.sum(ts_demean[:-lag] * ts_demean[lag:]) / n

            # Kernel weight
            if kernel == "bartlett":
                weight = 1.0 - lag / (max_lag + 1)
            else:  # uniform
                weight = 1.0

            hac_est += 2.0 * weight * gamma_lag

        hac_var[f] = hac_est / n

    return hac_var
```

### quant_evaluator.metrics.robustness.compute_subsample_ic

```python
def compute_subsample_ic(
    ic_series: np.ndarray,
    num_subsamples: int = 100,
    subsample_fraction: float = 0.8,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IC stability via bootstrap subsampling.

    Randomly samples subsample_fraction of time periods and computes mean IC.
    Repeated num_subsamples times to build distribution.

    Args:
        ic_series: Daily IC series (T, F)
        num_subsamples: Number of bootstrap samples
        subsample_fraction: Fraction of periods to sample (0, 1)
        random_seed: Random seed for reproducibility

    Returns:
        (subsample_means, subsample_stds)
        subsample_means: shape (num_subsamples, F)
        subsample_stds: shape (num_subsamples, F)
    """
    if subsample_fraction <= 0 or subsample_fraction >= 1:
        raise ValueError(f"subsample_fraction must be in (0, 1), got {subsample_fraction}")

    T, F = ic_series.shape
    subsample_size = max(1, int(T * subsample_fraction))

    # Local Generator: never touch the global numpy RNG state.
    rng = np.random.default_rng(random_seed)

    subsample_means = np.full((num_subsamples, F), np.nan, dtype=np.float64)
    subsample_stds = np.full((num_subsamples, F), np.nan, dtype=np.float64)

    for b in range(num_subsamples):
        # Random sample of time indices (local Generator)
        sampled_indices = rng.choice(T, size=subsample_size, replace=False)
        ic_subsample = ic_series[sampled_indices, :]  # (subsample_size, F)

        # Compute mean and std for this subsample
        with np.errstate(invalid='ignore'):
            subsample_means[b, :] = np.nanmean(ic_subsample, axis=0)
            subsample_stds[b, :] = np.nanstd(ic_subsample, axis=0, ddof=1)

    return subsample_means, subsample_stds
```

### quant_evaluator.metrics.robustness.compute_subsample_ic_std

```python
def compute_subsample_ic_std(
    ic_series: np.ndarray,
    num_subsamples: int = 100,
    subsample_fraction: float = 0.8,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Compute standard deviation of mean IC across subsamples.

    Low std indicates robust IC estimate across different time periods.

    Args:
        ic_series: Daily IC series (T, F)
        num_subsamples: Number of bootstrap samples
        subsample_fraction: Fraction of periods to sample
        random_seed: Random seed

    Returns:
        robustness_std: shape (F,) - std of subsample mean ICs
    """
    subsample_means, _ = compute_subsample_ic(
        ic_series, num_subsamples, subsample_fraction, random_seed
    )

    with np.errstate(invalid='ignore'):
        robustness_std = np.nanstd(subsample_means, axis=0, ddof=1)

    return robustness_std
```

### quant_evaluator.metrics.shape_evidence._as_matrix

```python
def _as_matrix(qr: np.ndarray) -> np.ndarray:
    """Coerce to a float64 (n_quantiles, F) matrix."""
    m = np.asarray(qr, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    return m
```

### quant_evaluator.metrics.shape_evidence._bottom_tail_slope_value

```python
def _bottom_tail_slope_value(col: np.ndarray, n_adj: int = 2) -> float:
    """Slope of the first ``n_adj+1`` finite quantile points."""
    seg = col[: n_adj + 1]
    if not np.all(np.isfinite(seg)):
        return np.nan
    diffs = np.diff(seg)
    if diffs.size == 0:
        return np.nan
    return float(np.mean(diffs))
```

### quant_evaluator.metrics.shape_evidence._fit_variance_explained

```python
def _fit_variance_explained(y: np.ndarray, template: np.ndarray) -> float:
    """R^2 of the scalar offset+scale OLS fit ``a + b*template`` against ``y``."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    n = y.shape[0]
    if n < 3:
        return np.nan
    X = np.stack([np.ones(n), t], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 0.0 else np.nan
    return 1.0 - ss_res / ss_tot
```

### quant_evaluator.metrics.shape_evidence._per_window_profile

```python
def _per_window_profile(
    windows: np.ndarray,
) -> np.ndarray:
    """Mean quantile-return across the window axis, (n_quantiles, F)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(windows, axis=0)
```

### quant_evaluator.metrics.shape_evidence._template_fit_x

```python
def _template_fit_x(nq: int) -> np.ndarray:
    """Normalised quantile coordinate array for template regression, (nq,)."""
    if nq <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, nq)
```

### quant_evaluator.metrics.shape_evidence._top_tail_slope_value

```python
def _top_tail_slope_value(col: np.ndarray, n_adj: int = 2) -> float:
    """Slope of the last ``n_adj+1`` finite quantile points, direction-agnostic.

    Returns NaN when the top segment is not fully finite.  The slope is
    ``mean(diff(segment))``.
    """
    seg = col[-n_adj - 1:]
    if not np.all(np.isfinite(seg)):
        return np.nan
    diffs = np.diff(seg)
    if diffs.size == 0:
        return np.nan
    return float(np.mean(diffs))
```

### quant_evaluator.metrics.shape_evidence._windows_or_single

```python
def _windows_or_single(m: np.ndarray):
    """Return (windows, n_quantiles, F) view and whether multi-window."""
    m = np.asarray(m, dtype=np.float64)
    if m.ndim == 2:
        return m.reshape(1, m.shape[0], m.shape[1]), False
    if m.ndim == 3:
        return m, True
    raise ValueError(
        f"shape-stability metrics expect (n_quantiles, F) or (W, n_quantiles, F), "
        f"got shape {m.shape}"
    )
```

### quant_evaluator.metrics.stability_regime._as_series

```python
def _as_series(ic_series: np.ndarray) -> np.ndarray:
    s = np.asarray(ic_series, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    return s
```

### quant_evaluator.metrics.stability_regime._period_consistency

```python
def _period_consistency(s: np.ndarray, period: str, time_index=None) -> np.ndarray:
    """Fraction of period means sharing the sign of the overall mean IC, (F,)."""
    means = _period_means(s, period, time_index)
    with np.errstate(invalid="ignore"):
        overall = np.nanmean(s, axis=0)
    sign = np.sign(overall)
    finite = np.isfinite(means)
    n = np.sum(finite, axis=0)
    same = np.sum((np.sign(means) == sign[None, :]) & finite, axis=0)
    ratio = same / np.maximum(n, 1)
    return np.where(n >= 1, ratio, np.nan)
```

### quant_evaluator.metrics.stability_regime._period_means

```python
def _period_means(s: np.ndarray, period: str, time_index=None) -> np.ndarray:
    if time_index is not None and len(time_index) == s.shape[0]:
        try:
            import pandas as pd

            idx = pd.to_datetime(list(time_index))
            df = pd.DataFrame(s, index=idx)
            if period == "year":
                grouped = df.groupby(df.index.year)
            elif period == "month":
                grouped = df.groupby([df.index.year, df.index.month])
            else:
                grouped = df.groupby(df.index.to_period("Q"))
            return grouped.mean().to_numpy(dtype=np.float64)
        except Exception:  # noqa: BLE001
            pass
    block = _BLOCK_DAYS.get(period, 63)
    T, F = s.shape
    n_blocks = max(1, int(np.ceil(T / block)))
    pad = n_blocks * block - T
    if pad > 0:
        s = np.vstack([s, np.full((pad, F), np.nan)])
    blocks = s.reshape(n_blocks, block, F)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=1)
```

### quant_evaluator.metrics.stability_regime._regime_split

```python
def _regime_split(s: np.ndarray):
    """Split the IC series into early/late halves, returning (early, late)."""
    T, F = s.shape
    mid = T // 2
    return s[:mid, :], s[mid:, :]
```

### quant_evaluator.metrics.stability_regime._valid_counts

```python
def _valid_counts(s: np.ndarray) -> np.ndarray:
    return np.sum(np.isfinite(s), axis=0)
```

### quant_evaluator.metrics.temporal.compute_autocorrelation

```python
def compute_autocorrelation(
    series: np.ndarray,
    max_lag: int = 20,
    min_obs: int = 30,
) -> np.ndarray:
    """
    Compute autocorrelation function (ACF) for time series.

    True-time-axis semantics: for lag k, only pairs (t, t-k) where BOTH
    original positions are finite contribute. Missing observations are never
    compressed out before lagging — calendar gaps destroy real lag alignment
    and this implementation keeps it intact.

    The estimator is the standard correlation between the lag-k pair samples
    (x_t, x_{t-k}), computed with pair-specific means, so it is unaffected by
    where the NaNs sit.

    Args:
        series: Time series (T,) or (T, F) for multiple factors
        max_lag: Maximum lag to compute
        min_obs: Minimum observations required

    Returns:
        acf: shape (max_lag+1,) or (max_lag+1, F)
        acf[0] is always 1.0 (correlation with self)
    """
    for name, value, lower in (("max_lag", max_lag, 0), ("min_obs", min_obs, 2)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    series = np.asarray(series, dtype=float)
    if series.ndim not in (1, 2):
        raise ValueError("ACF requires T or T x F")
    one_dimensional = series.ndim == 1
    if one_dimensional:
        series = series.reshape(-1, 1)

    T, F = series.shape

    if T < min_obs:
        acf = np.full((max_lag + 1, F), np.nan, dtype=np.float64)
        return acf[:, 0] if one_dimensional else acf

    acf = np.full((max_lag + 1, F), np.nan, dtype=np.float64)

    for f in range(F):
        ts = series[:, f]
        finite = np.isfinite(ts)
        n_finite = int(np.sum(finite))

        if n_finite < min_obs:
            continue

        # Lag 0: correlation of the finite values with themselves.
        acf[0, f] = 1.0

        for lag in range(1, min(max_lag + 1, T)):
            # Pairs (t, t-lag) where BOTH original positions are finite.
            pair_mask = finite[lag:] & finite[:-lag]  # length T - lag
            n_pairs = int(np.sum(pair_mask))
            if n_pairs < min_obs:
                continue

            x_prev = ts[:-lag][pair_mask]  # x_{t-lag}
            x_curr = ts[lag:][pair_mask]   # x_t

            # Correlation between the two aligned pair samples.
            x_prev_c = x_prev - np.mean(x_prev)
            x_curr_c = x_curr - np.mean(x_curr)

            denom = np.sqrt(np.sum(x_prev_c ** 2) * np.sum(x_curr_c ** 2))
            if denom <= 0 or not np.isfinite(denom):
                # Constant pair sample(s): correlation undefined.
                continue

            acf[lag, f] = float(np.sum(x_prev_c * x_curr_c) / denom)

    return acf[:, 0] if one_dimensional else acf
```

### quant_evaluator.metrics.temporal.compute_factor_turnover_rate

```python
def compute_factor_turnover_rate(
    factor_values: np.ndarray,
    quantile: float = 0.9,
    *, measure: str = "universe_membership_change",
) -> np.ndarray:
    """
    Compute turnover rate of top/bottom quantile membership.

    Measures how frequently assets enter/exit extreme factor quantiles.
    High turnover indicates unstable factor ordering.

    Args:
        factor_values: Factor values (T, N, F)
        quantile: Quantile threshold (0.9 = top 10%, 0.1 = bottom 10%)

    Returns:
        turnover_rate: shape (T-1, F) - fraction of positions changed
    """
    if measure not in ("universe_membership_change", "top_exit_fraction", "top_entry_fraction", "jaccard_distance"):
        raise ValueError("unknown membership diagnostic; actual turnover requires holdings")
    T, N, F = factor_values.shape

    if quantile <= 0 or quantile >= 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    turnover_rate = np.full((max(T - 1, 0), F), np.nan, dtype=np.float64)

    for f in range(F):
        for t in range(T - 1):
            factor_t = factor_values[t, :, f]
            factor_t1 = factor_values[t + 1, :, f]

            valid_t = np.isfinite(factor_t)
            valid_t1 = np.isfinite(factor_t1)
            if min(np.sum(valid_t), np.sum(valid_t1)) < 10:
                continue
            factor_t_valid = factor_t[valid_t]
            factor_t1_valid = factor_t1[valid_t1]

            # Determine quantile membership
            if quantile > 0.5:
                threshold_t = np.nanquantile(factor_t_valid, quantile)
                threshold_t1 = np.nanquantile(factor_t1_valid, quantile)
                in_quantile_t = valid_t & (factor_t >= threshold_t)
                in_quantile_t1 = valid_t1 & (factor_t1 >= threshold_t1)
            else:
                threshold_t = np.nanquantile(factor_t_valid, quantile)
                threshold_t1 = np.nanquantile(factor_t1_valid, quantile)
                in_quantile_t = valid_t & (factor_t <= threshold_t)
                in_quantile_t1 = valid_t1 & (factor_t1 <= threshold_t1)

            # Count changes
            # A previously selected security with unknown next signal is not
            # proof of a sale. Preserve uncertainty rather than erase it.
            if np.any(in_quantile_t & ~valid_t1):
                continue
            changed = in_quantile_t != in_quantile_t1
            if measure == "universe_membership_change":
                turnover_rate[t, f] = changed.sum() / np.sum(valid_t | valid_t1)
            elif measure == "top_exit_fraction":
                turnover_rate[t, f] = np.sum(in_quantile_t & ~in_quantile_t1) / in_quantile_t.sum()
            elif measure == "top_entry_fraction":
                turnover_rate[t, f] = np.sum(in_quantile_t1 & ~in_quantile_t) / in_quantile_t1.sum()
            else:
                turnover_rate[t, f] = changed.sum() / np.sum(in_quantile_t | in_quantile_t1)

    return turnover_rate
```

### quant_evaluator.metrics.temporal.compute_half_life

```python
def compute_half_life(
    ic_series: np.ndarray,
    min_periods: int = 60,
    *, model: str = "centered_ar1",
) -> np.ndarray:
    """
    Estimate IC half-life using AR(1) model.

    Half-life = -log(2) / log(phi) where phi is AR(1) coefficient.
    Measures IC temporal persistence, NOT predictive-horizon decay.
    The default fits an intercept; zero_mean_ar1 is an explicit research model.

    True-time-axis semantics: phi is estimated on (t, t-1) pairs where BOTH
    original positions are finite. NaNs are never compressed out before
    lagging, so calendar gaps do not fabricate adjacent pairs.

    Args:
        ic_series: Daily IC series (T, F)
        min_periods: Minimum periods for AR estimation

    Returns:
        half_life: shape (F,) - half-life in periods (NaN if phi >= 1 or phi <= 0)
    """
    if model not in ("centered_ar1", "zero_mean_ar1"):
        raise ValueError("unknown AR1 model")
    T, F = ic_series.shape
    half_life = np.full(F, np.nan, dtype=np.float64)

    for f in range(F):
        ic_f = ic_series[:, f]
        finite = np.isfinite(ic_f)
        n_finite = int(np.sum(finite))

        if n_finite < min_periods:
            continue

        # AR(1): IC_t = phi * IC_{t-1} + epsilon, estimated on pairs
        # (t, t-1) where BOTH original positions are finite. Missing
        # observations are never compressed out before lagging — that would
        # fabricate adjacent pairs across calendar gaps.
        pair_mask = finite[1:] & finite[:-1]  # length T - 1
        x = ic_f[:-1][pair_mask]  # IC_{t-1}
        y = ic_f[1:][pair_mask]   # IC_t

        if len(y) < min_periods - 1:
            continue

        if model == "centered_ar1":
            x = x - np.mean(x)
            y = y - np.mean(y)
        # Pair-specific centering is equivalent to an intercept regression.
        denom = np.sum(x * x)
        if not np.isfinite(denom) or denom <= 0:
            continue

        phi = np.sum(x * y) / denom

        # Half-life is only meaningful for 0 < phi < 1
        if 0 < phi < 1:
            half_life[f] = -np.log(2) / np.log(phi)

    return half_life
```

### quant_evaluator.metrics.temporal.compute_ic_autocorrelation

```python
def compute_ic_autocorrelation(
    ic_series: np.ndarray,
    max_lag: int = 20,
    min_obs: int = 30,
) -> np.ndarray:
    """
    Compute autocorrelation of IC series.

    High IC autocorrelation indicates persistent factor performance.

    Args:
        ic_series: Daily IC series (T, F)
        max_lag: Maximum lag
        min_obs: Minimum observations

    Returns:
        ic_acf: shape (max_lag+1, F)
    """
    return compute_autocorrelation(ic_series, max_lag=max_lag, min_obs=min_obs)
```

### quant_evaluator.metrics.temporal.compute_mean_rank_stability

```python
def compute_mean_rank_stability(
    factor_values: np.ndarray,
    lag: int = 1,
    method: str = "spearman",
    min_periods: int = 20,
) -> np.ndarray:
    """
    Compute time-averaged rank stability.

    Args:
        factor_values: Factor values (T, N, F)
        lag: Time lag
        method: "spearman" or "pearson"
        min_periods: Minimum valid periods

    Returns:
        mean_stability: shape (F,)
    """
    stability = compute_rank_stability(factor_values, lag=lag, method=method)

    valid_periods = np.sum(~np.isnan(stability), axis=0)

    with np.errstate(invalid='ignore'):
        mean_stability = np.nanmean(stability, axis=0)

    insufficient = valid_periods < min_periods
    mean_stability = np.where(insufficient, np.nan, mean_stability)

    return mean_stability
```

### quant_evaluator.metrics.temporal.compute_rank_stability

```python
def compute_rank_stability(
    factor_values: np.ndarray,
    lag: int = 1,
    method: str = "spearman",
) -> np.ndarray:
    """
    Compute rank stability across time.

    Measures correlation of factor ranks between t and t+lag.
    High rank stability indicates persistent factor ordering.

    Args:
        factor_values: Factor values (T, N, F)
        lag: Time lag for stability measurement
        method: "spearman" or "pearson"

    Returns:
        stability: shape (T-lag, F) - rank correlation at each time
    """
    T, N, F = factor_values.shape

    if lag >= T:
        raise ValueError(f"lag ({lag}) must be less than T ({T})")

    stability = np.full((T - lag, F), np.nan, dtype=np.float64)

    corr_fn = stats.spearmanr if method == "spearman" else stats.pearsonr

    for f in range(F):
        for t in range(T - lag):
            factor_t = factor_values[t, :, f]
            factor_t_lag = factor_values[t + lag, :, f]

            # Valid observations in both periods
            valid_mask = np.isfinite(factor_t) & np.isfinite(factor_t_lag)

            if np.sum(valid_mask) < 10:
                continue

            factor_t_valid = factor_t[valid_mask]
            factor_t_lag_valid = factor_t_lag[valid_mask]

            # Check for constants
            if len(np.unique(factor_t_valid)) == 1 or len(np.unique(factor_t_lag_valid)) == 1:
                continue

            if method == "spearman":
                corr, _ = stats.spearmanr(factor_t_valid, factor_t_lag_valid)
            else:
                corr, _ = stats.pearsonr(factor_t_valid, factor_t_lag_valid)

            stability[t, f] = corr

    return stability
```

### quant_evaluator.metrics.turnover.compute_turnover_series

```python
def compute_turnover_series(
    weights: np.ndarray,
    method: str = "half_sum_abs",
) -> np.ndarray:
    """
    Compute turnover time series from weight matrix (vectorized).

    Optimized implementation using numba JIT compilation when available,
    falling back to numpy broadcasting for compatibility.

    Args:
        weights: Weight matrix (T, N)
        method: Turnover method

    Returns:
        Turnover series (T,), first observation is NaN
    """
    if method != "half_sum_abs":
        raise ValueError(f"Unknown turnover method: {method}")

    T, N = weights.shape

    if T < 2:
        return np.full(T, np.nan, dtype=np.float64)

    # Use numba-optimized version if available (10-20x faster)
    if HAS_NUMBA:
        return _compute_turnover_series_numba(weights)

    # Fallback: pure numpy vectorized implementation
    turnover_series = np.empty(T, dtype=np.float64)
    turnover_series[0] = np.nan

    # Process differences in-place using slicing
    w_diff = weights[1:] - weights[:-1]  # (T-1, N)

    # Compute finite mask efficiently
    finite_mask = np.isfinite(w_diff)  # (T-1, N)

    # Fast path: if no NaNs, use simple sum
    if np.all(finite_mask):
        turnover_series[1:] = 0.5 * np.sum(np.abs(w_diff), axis=1)
    else:
        # Need to handle NaNs: set them to zero for summation
        abs_diff = np.abs(w_diff)
        abs_diff[~finite_mask] = 0.0

        # Sum and check validity
        turnover_values = 0.5 * np.sum(abs_diff, axis=1)  # (T-1,)
        n_valid = np.sum(finite_mask, axis=1)  # (T-1,)

        # Set to NaN where no valid observations
        turnover_values[n_valid != N] = np.nan
        turnover_series[1:] = turnover_values

    return turnover_series
```

### quant_evaluator.metrics.underwater._as_1d

```python
def _as_1d(returns: object, name: str = "returns") -> np.ndarray:
    """Coerce without deleting positions from the original observation grid."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional series, got ndim={arr.ndim}")
    return arr
```

### quant_evaluator.metrics.underwater._path_events

```python
def _path_events(returns, min_periods):
    from .risk.drawdown_analysis import drawdown_events
    ret = _as_1d(returns)
    if np.isfinite(ret).sum() < min_periods:
        return None
    events = drawdown_events(ret)
    # Scalar APIs cannot express an interval estimate or unknown NAV path.
    # Fail closed, while drawdown_events retains aligned censoring evidence.
    if any(e["status"] == "INVALID_VALUATION" for e in events):
        return None
    return events
```

## 定义完整性指纹

每项注册定义及上列实现源公式均参与本文内容；以下源摘要便于定位函数变化。

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
| `quant_evaluator.metrics.ic._spearman_rank_correlation` | `42c2f94d2c8331977c870459ee5a91096fba3c608ffbfdc05f5ed2a0148e0619` |
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
| `quant_evaluator.metrics.multiple_testing._validate_p_values` | `7ba7985e00eb9fbe0eb83727613948ba5fc710c9b8812f62e0bbe646957f7f89` |
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
| `quant_evaluator.metrics.quantile._percentile_boundaries` | `b69287bce1769a089514326ba8dc46db33c8c8cc7159e902dc0489d6a37a992e` |
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
| `quant_evaluator.metrics.robustness.compute_block_bootstrap_ci` | `e32ba162d8933455a5bb7f2da6948c1c3b4346b73df98857795742a4f0a6a615` |
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
