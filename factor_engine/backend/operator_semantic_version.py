# -*- coding: utf-8
"""算子语义版本：定义变更须 bump version 并在 factor metadata 记录。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSemanticVersion:
    canonical: str
    version: int
    note: str = ""


# 已知语义变更历史（新因子应使用最新 version）
OPERATOR_SEMANTIC_VERSIONS: dict[str, int] = {
    "ts_bds_statistic": 2,  # v9: common-center variance and conditioned correlation integral
    # v9: stable episode-first anchors and local unknown-event coverage.
    "event_historical_response_mean": 2,
    "event_historical_response_sign_balance": 2,
    "event_response_dispersion": 2,
    "event_response_effective_events": 2,
    "event_response_overlap_ratio": 2,
    # v9: sparse pinball fit with original-unit objective and coefficient identity checks.
    "ts_quantile_regression_beta": 2,
    "ts_quantile_regression_slope": 2,
    "ts_quantile_regression_coeff": 2,
    "ts_quantile_regression_coeff_prior": 2,
    "ts_quantile_regression_resid": 2,
    "ts_quantile_beta_spread": 2,
    "ts_quantile_beta_spread_prior": 2,
    "ts_transfer_entropy": 2,  # v9: explicit ordered-category/quantile state policy
    "ts_effective_transfer_entropy": 2,
    "ts_transfer_entropy_peak_strength": 2,
    "ts_transfer_entropy_peak_lag": 2,
    "ts_transfer_entropy_peak_excess": 2,
    "ts_distance_corr": 2,  # v9: relative-unit blockwise biased distance geometry
    "ts_distance_cov": 2,
    "ts_huber_regression_in_sample_resid": 2,  # v9: shared score/restoration-certified Huber fit
    "ts_huber_regression_predictive_resid": 2,
    "ts_vector_state_mahalanobis": 2,  # v9: relative covariance geometry and explicit nullspace failure
    "ts_butterworth_lowpass_causal": 2,  # v9: cutoff is cycles/bar, not Nyquist fraction
    "ts_local_lyapunov_exponent": 2,  # v9: true-bar embeddings and complete successor candidates
    "ts_cross_spectral_coherence": 2,  # v9: unit-invariant Welch energy gate
    "ts_cross_spectral_phase": 2,  # v9: shared valid bins and identifiable direction
    "ts_bicoherence_top_decile_mean": 2,  # v9: unit-invariant power/validity
    "ts_bicoherence_top_decile_excess": 2,
    # v9: explicit Haar scales/window domain and one-sided Parseval weights.
    "ts_wavelet_low_frequency_ratio": 2,
    "ts_wavelet_high_frequency_ratio": 2,
    "ts_wavelet_entropy": 2,
    "ts_wavelet_energy_slope": 2,
    "ts_spectral_low_frequency_ratio": 2,
    "ts_expectile": 2,  # v9: centered score-certified location convergence
    "ts_expectile_beta": 2,  # v9: preconditioned asymmetric regression
    "ts_kernel_granger_score": 2,  # v9: pairwise RBF geometry avoids cancellation
    "ts_hill_tail_index": 2,  # v9: strictly positive classic-Hill domain
    "ts_matrix_profile_motif_frequency": 2,
    "ts_matrix_profile_neighbor_dispersion": 2,
    "ts_residualized_hsic": 2,  # v9: explicit jointly feasible purged folds
    "ts_causal_local_linear_smoother": 2,  # v9: predict current physical endpoint
    # v9: actual convex first/second-difference L1 objectives and bounded solve.
    "ts_total_variation_filter_trailing": 2,
    "ts_l1_trend_filter_trailing": 2,
    "group_peer_beta_deviation": 2,
    # v9: stable prewarped phase-normalized Bessel SOS (gap policy unchanged).
    "ts_bessel_lowpass_causal": 2,
    # v9: correct robust-fit coefficient order and one finite fit/query cohort.
    "cs_huber_resid": 2,
    "cs_lad_resid": 2,
    # v9: complete first-passage horizons and physical-bar breadth history.
    "ts_first_passage_bias": 2,
    "ts_first_passage_hit_probability": 2,
    "ts_first_passage_conditional_time": 2,
    "group_feature_mode_share": 2,
    "group_feature_effective_rank": 2,
    "group_feature_mode_localization": 2,
    "group_feature_spectral_gap": 2,
    "group_feature_second_mode_localization": 2,
    # v9: defined no-reversal zero; valid turnover mass independent of costs.
    "event_response_reversal_strength": 2,
    "ts_turnover_reference_price": 2,
    "ts_turnover_cost_dispersion": 2,
    "ts_turnover_profit_share": 2,
    "ts_turnover_holding_age": 2,
    "ts_turnover_near_cost_mass": 2,
    "ts_turnover_cost_quantile_distance": 2,
    "ts_turnover_cost_entropy": 2,
    "ts_turnover_cost_mode_distance": 2,
    "ts_turnover_cost_skew": 2,
    "ts_turnover_age_dispersion": 2,
    "ts_turnover_old_mass": 2,
    "ts_turnover_cost_entropy_vol_scaled": 2,
    # v9: one-sided full-support FIR and explicit feasible filter domains/history.
    "ts_fir_lowpass_causal": 2,
    "ts_spectral_lowpass_trailing": 2,
    "ts_causal_savgol_endpoint": 2,
    # v8: extendable SampEn cohorts and stable actual native liquidity binding.
    "ts_sample_entropy": 2,
    "ts_pseudocount_sample_entropy": 2,
    "ts_market_liquidity_beta": 2,
    "ts_industry_liquidity_beta": 2,
    # v8: reject infeasible joint domains before I/O; ignored selectors no
    # longer constrain or identify non-coefficient regression outputs.
    "ts_ar_fitted_value": 2,
    "ts_ar_in_sample_resid": 2,
    "ts_ar_forecast": 2,
    "ts_ar_innovation": 2,
    "ts_ar_innovation_z": 2,
    "ts_ar_prior_forecast": 2,
    "ts_ar_prior_innovation": 2,
    "ts_ar_prior_innovation_z": 2,
    "ts_ar_prior_coeff": 2,
    "ts_ar_coeff_stability": 2,
    "ts_multi_regression_coeff": 2,
    "ts_multi_regression_resid": 2,
    "ts_multi_regression_resid_z": 2,
    "ts_multi_regression_r2": 2,
    "ts_huber_regression_coeff": 2,
    "ts_huber_regression_resid_z": 2,
    "ts_multi_regression_coeff_prior": 2,
    "ts_multi_regression_forecast_error": 2,
    "ts_multi_regression_forecast_error_z": 2,
    "ts_multi_regression_r2_prior": 2,
    "ts_multi_regression_adjusted_r2_prior": 2,
    "ts_multi_regression_coeff_stability": 2,
    "ts_huber_regression_coeff_prior": 2,
    "ts_huber_regression_forecast_error": 2,
    "ts_huber_regression_forecast_error_z": 2,
    # v8: stable causal moments and coherent dimensionless state/covariance units.
    "ts_kalman_level": 2,
    "ts_kalman_innovation_z": 2,
    "ts_kalman_trend": 2,
    "ts_kalman_beta": 2,
    "ts_kalman_beta_change": 2,
    "ts_kalman_beta_uncertainty": 2,
    # v8: fixed-model regime updates normalize in log space, including tails.
    "ts_two_state_regime_probability": 2,
    "ts_regime_duration": 2,
    "ts_change_point_probability": 2,
    # v7 stable Ridge kernel; v8 adds configured-domain/ignored-selector rules.
    "ts_ridge_regression_coeff": 3,
    "ts_ridge_regression_resid_z": 3,
    "ts_ridge_regression_coeff_prior": 3,
    "ts_ridge_regression_forecast_error": 3,
    "ts_ridge_regression_forecast_error_z": 3,
    "ts_ridge_regression_in_sample_resid": 2,
    "ts_ridge_regression_predictive_resid": 2,
    # v7: one finite cohort for PCA fit/projection and missing current masks.
    "panel_rolling_pca_loading": 2,
    "panel_rolling_pca_resid": 2,
    "panel_rolling_pca_explained_ratio": 2,
    "panel_rolling_pca_resid_vol": 2,
    "panel_rolling_pca_resid_momentum": 2,
    "industry_rolling_pca_loading": 2,
    "panel_mixture_of_experts_score": 2,  # stable gate over usable experts only
    # v7: GJR shared recursion and prefix-local return-domain validation.
    "ts_garch_next_vol_forecast": 2,
    "ts_garch_vol_surprise": 2,
    "ts_garch_persistence": 2,
    "ts_garch_standardized_shock": 2,
    "ts_gjr_garch_vol_forecast": 2,
    "ts_gjr_leverage": 2,
    # v7: HAR rows preserve original feature/label clock and common cohort.
    "ts_har_rv_next_vol_forecast": 2,
    "ts_har_rv_next_var_forecast": 2,
    "ts_har_rv_forecast_error_z": 2,
    "ts_har_from_return_next_vol": 2,
    "ts_har_from_return_forecast_error_z": 2,
    "group_percentile": 3,  # v3: backend emitters preserve current quantile/null semantics
    "signed_log": 1,  # sign(x)*log(abs(x)+1e-10)
    "compare": 2,  # v2: NULL/NaN propagate
    "maximum": 2,
    "minimum": 2,
    "protected_log": 2,  # NULL preserved
    "protected_div": 2,
    "rank": 1,
    "rank_pct": 1,
    "ts_beta": 3,  # v3: paired finite cohort + explicit ddof semantics
    "ts_zscore": 2,  # v2: all declared window/numeric parameters affect execution
    # m_beta / rolling_beta are runtime aliases of ts_beta and inherit v3.
    "rolling_beta_to_market": 2,  # v2: paired cohort + benchmark alignment
    "downside_beta": 2,  # v2: downside mask after paired finite alignment
    "tail_beta": 2,  # v2: tail mask after paired finite alignment
    "intra_entropy": 2,  # v2: normalized histogram probability entropy
    "intra_limit_first_hit_time": 2,  # v2: limits bind by date+instrument identity
    "intra_limit_duration": 2,  # v2: limits bind by date+instrument identity
    "intra_limit_reopen_count": 2,  # v2: labelled limits + strict transition domain
    "fin_component_score": 2,  # v2: finite FALSE contributes 0; missing remains NaN
    "ts_kurt": 2,  # v2: backend emitter parity for finite/Inf handling
    "cs_quantile": 2,  # v2: backend emitter parity for finite/Inf handling
    "ts_quantile": 2,  # v2: backend emitter parity for finite/Inf handling
    "true_range": 2,  # v2: backend emitter parity for finite/Inf handling
    "ts_ema": 2,  # v2: canonical EMA emitter parity; aliases inherit this version
    "MACD_line": 2,  # v2: backend EMA/finite semantics aligned
    "MACD_signal": 2,  # v2: backend EMA/finite semantics aligned
    "MACD_hist": 2,  # v2: backend EMA/finite semantics aligned
    "ts_max_drawdown": 2,  # v2: generic delegate binds positional scalar parameters
    "overnight_return": 2,  # v2: shared concrete price basis + paired finite positive prices
    "open_close_return": 2,  # v2: shared concrete price basis + paired finite positive prices
    "open_to_vwap_return": 2,  # v2: shared concrete price basis + paired finite positive prices
    "vwap_to_close_return": 2,  # v2: shared concrete price basis + paired finite positive prices
    "ts_corr": 2,
    "ts_cov": 2,
}


def semantic_version(canon: str) -> int:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return OPERATOR_SEMANTIC_VERSIONS.get(name, 1)


def versioned_name(canon: str) -> str:
    return f"{canon}@v{semantic_version(canon)}"


def record_version_in_factor_metadata() -> bool:
    """Factor metadata 应记录 ``operator_semantic_versions`` 映射。"""
    return True
