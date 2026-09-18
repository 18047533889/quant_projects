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
    # R39/R41: correct quadratic coefficient order and finite scored pairs.
    "ts_poly2_resid": 2,
    # R44: feasible information-form H-infinity Riccati recurrence.
    "ts_h_infinity_level_filter": 2,
    # R44: genuine normal-distribution Lilliefors test, not swallowed AttributeError.
    "ts_expanding_lilliefors_pvalue": 2,
    # R54: recursive McGinley state reseeds after a complete finite recovery window.
    "ts_mcginley_dynamic": 2,
    # R56: feasible three-threshold Hill ladder and scale-safe positive tails.
    "ts_evt_threshold_stability": 2,
    # R55: preserve every standard time/identity coordinate in chip kernels.
    "ts_cpt_value": 2,
    # R52: replace holder placeholders with exact multi-panel contracts.
    "holder_shareholder_overlap_ratio": 2,
    "holder_float_concentration_gap": 2,
    "holder_concentration": 2,
    "holder_topk_share_sum": 2,
    "holder_observed_topk_hhi": 2,
    "holder_disclosure_count": 2,
    "holder_disclosure_coverage": 2,
    "holder_entry_share": 2,
    "holder_exit_share": 2,
    "holder_net_entry_share": 2,
    "holder_id_matched_entry_share": 2,
    "holder_id_matched_exit_share": 2,
    "holder_id_matched_churn": 2,
    "holder_weighted_churn": 2,
    "holder_rank_stability": 2,
    "holder_id_overlap_ratio": 2,
    "holder_share_weighted_rank_migration": 2,
    # R50: exact holder contracts and finite nonnegative normalized entropy.
    "holder_class_entropy": 2,
    "holder_nature_entropy": 2,
    "holder_class_js_shift": 2,
    "holder_common_holding_peer_return": 2,
    "holder_concentration_acceleration": 2,
    "holder_concentration_slope": 2,
    "holder_freeze_concentration": 2,
    "holder_pledge_churn": 2,
    "holder_pledge_concentration": 2,
    "holder_pledged_holder_count": 2,
    "holder_shareholder_network_centrality": 2,
    # R50: bounded finite share ratios across actual Polars registrations.
    "holder_freeze_ratio": 2,
    "holder_locked_share_ratio": 2,
    "holder_pledge_ratio": 2,
    # R49: repaired daily path/segment kernels and finite-support jump detection.
    "intra_max_drawdown": 2,
    "intra_max_drawup": 2,
    "intra_segment_realized_vol": 2,
    "intra_positive_jump_variation": 2,
    "intra_negative_jump_variation": 2,
    "intra_signed_jump_ratio": 2,
    "intra_jump_count": 2,
    "intra_jump_concentration": 2,
    "intra_jump_first_time": 2,
    "intra_jump_last_time": 2,
    "intra_jump_clustering": 2,
    "intra_positive_tail_variation": 2,
    "intra_negative_tail_variation": 2,
    "intra_tail_event_count": 2,
    "intra_signed_tail_variation_ratio": 2,
    # R49: first membership observations and unknown boundaries are not transitions.
    "index_reconstitution_churn": 2,
    # R47: correct return bipower and physical-pair autocorrelation estimators.
    "ts_jump_bipower": 2,
    "ts_lag1_autocorr": 2,
    # R45: retain physical session gaps in historical profile windows.
    "intra_return_profile_cosine": 2,
    "intra_volume_profile_cosine": 2,
    "intra_amount_profile_cosine": 2,
    "intra_volume_profile_jsd": 2,
    "intra_amount_profile_jsd": 2,
    "intra_profile_earth_mover_distance": 2,
    # R40/R42: replace effective Polars placeholders with canonical estimators.
    "ts_extremal_index": 2,
    "ts_gpd_shape_pwm": 2,
    "ts_deviation_from_mean": 2,
    # R21: finite-only cross-sections across reference/Polars/SQL paths;
    # normalize also enforces constant-support and singleton/null policies.
    "normalize": 2,
    "cs_rank_01": 2,
    "cs_std": 2,
    # R16: Polars CCI now propagates null/nonfinite active windows like pandas.
    "CCI": 2,
    # R6: final finite/mixed/graph panel contracts; no certification promotion.
    "hump_decay": 2,
    "trade_when": 2,
    "panel_factor_pocket_strength": 2,
    "cs_knn_local_moran": 2,
    "row_sum_skipna": 2,
    "group_signal_attraction_share": 2,
    "relation_diffusion_score": 2,
    # R6: genuine static graph, exact kurtosis and strict mixed/default numerical kernels.
    "cs_rank_gaussian": 2,
    "ts_days_since": 2,
    "panel_peer_graph_aggregate": 2,
    "safe_div_null": 2,
    "signed_power": 2,
    "intra_impulse_event_detector": 2,
    "ts_dominant_cycle_period": 2,
    "ts_sma_cn": 2,
    "lqtp_historical_cvar": 2,
    # R6: exact fiscal/component/event/rolling definitions and mixed literal/panel contracts.
    "ts_product": 2,
    "ts_mad": 2,
    "revision_delta": 2,
    "period_stability": 2,
    "RSI_WILDER": 2,
    "beta_residual_z": 2,
    "beta_divergence_pct": 2,
    "relative_strength_group_pct": 2,
    "ts_vector_state_local_density": 2,
    "cross_event": 2,
    "event_refractory": 2,
    "power": 2,
    "coalesce": 2,
    # R6: strict path-signature contracts and real minute-to-daily slice routing.
    "ts_path_leadlag_area": 2,
    "ts_path_signature_area": 2,
    "ts_path_signature_depth2_norm": 2,
    "intra_slice_mask_reduce": 2,
    "intra_slice_mask_pair_reduce": 2,
    "intra_round_price_clustering_share": 2,
    # R6: scale-safe event/rank estimators, subnormal digits and recovered-peak reset.
    "event_mark_autocorr": 2,
    "event_interval_mark_coupling": 2,
    "ts_score_rank_weighted_mean": 2,
    "report_benford_js_divergence": 2,
    "ts_recovery_fraction": 2,
    "ts_current_drawdown_area": 2,
    # R6: exact crossing/distribution, session availability and finite-scale state contracts.
    "ts_crossing_speed": 2,
    "ts_crossing_acceleration": 2,
    "ts_joint_energy_shift": 2,
    "ts_energy_break_score": 2,
    "ts_copula_central_asymmetry": 2,
    "intra_limit_pre_hit_pressure_profile": 2,
    "intra_eod_reversal_decomposition": 2,
    "session_event_recovery_score": 2,
    "ts_extrema_divergence_strength": 2,
    "ts_extrema_confirmation_rate": 2,
    "ts_state_density": 2,
    "holder_company_ownership_hhi": 3,
    "holder_concentration_change": 2,
    "holder_count_change_rate": 2,
    # R6: strict copula/session contracts and real conditional covariance.
    "cs_rank_copula_mi": 2,
    "cs_rank_copula_entropy": 2,
    "intraday_session_shape_novelty": 2,
    "intraday_profile_pca_residual": 2,
    "intraday_activity_duration_curvature": 2,
    "ts_cov_if": 2,
    # R6: safe log-price spread, explicit full-history smoothing, asym/cycle contracts.
    "ohlc_corwin_schultz_spread": 2,
    "ts_roll_effective_spread": 2,
    "ts_kama": 2,
    "ts_multifractal_asymmetry": 2,
    "ts_threshold_cycle_period": 2,
    "ts_threshold_cycle_asymmetry": 2,
    # R6: fiscal authoritative loading, multifractal scale stability, causal despike.
    "fiscal_acceleration": 2,
    "fiscal_pct_change": 2,
    "fiscal_rolling_std": 2,
    "ts_generalized_hurst_spread_q1_q4": 2,
    "ts_multifractal_spectrum_width": 2,
    "ts_multifractal_curvature": 2,
    "ts_hampel_filter_causal": 2,
    "ts_median3_causal": 2,
    "ts_rolling_median_causal": 2,
    # R6: real group fallbacks, scale-safe jump and complete multiscale contracts.
    "group_decay_linear": 2,
    "group_ts_decay_linear": 2,
    "group_winsorize": 2,
    "intraday_medrv": 2,
    "intraday_minrv": 2,
    "ts_multiscale_trend_consensus": 2,
    "ts_multiscale_trend_dispersion": 2,
    "ts_multiscale_trend_curvature": 2,
    # R6: scalar/Allan/moment contracts and nominal structural coverage/log ratios.
    "scale": 2,
    "cs_regression": 2,
    "ts_hartigan_dip": 2,
    "ts_l_kurtosis": 2,
    "ts_l_skewness": 2,
    "event_allan_factor": 2,
    "event_allan_scaling_slope": 2,
    "event_allan_log_mean": 2,
    "ts_structural_level_density": 2,
    "ts_nearest_structural_level_distance": 2,
    "ts_structural_level_strength": 2,
    # R6: exact AR/binned/survival estimators and real conditional-TE backend.
    "ts_mean_reversion_half_life": 2,
    "ts_mean_reversion_ou_approx_half_life": 2,
    "ts_mean_reversion_ou_approx_half_life_prior": 2,
    "ts_variance_ratio_slope": 2,
    "ts_response_slope_asymmetry": 2,
    "ts_state_age_percentile": 2,
    "ts_state_exit_hazard": 2,
    "ts_state_residual_life": 2,
    "ts_conditional_transfer_entropy": 2,
    # R6: scale-safe dependence and exact sequential state backend contracts.
    "ts_chatterjee_xi": 2,
    "ts_hsic": 2,
    "ts_conditional_mutual_information": 2,
    "ts_distance_correlation_partial_proxy": 2,
    "ts_cusum_pressure": 2,
    "ts_rank_if": 2,
    "state_ewm_if": 2,
    "ts_lag_of_peak_corr": 2,
    # R6: exact two-panel path geometry; scale-safe finite coordinates.
    "ts_vector_path_efficiency": 2,
    "ts_vector_turning_coherence": 2,
    "ts_vector_path_curvature": 2,
    "ts_vector_self_intersection_rate": 2,
    # R6: real robust group estimators and exact leave-one-out peer reductions.
    "cs_trimmed_ols_resid": 2,
    "group_multi_resid": 2,
    "ex_self_mean_gap": 2,
    "ex_self_zscore": 2,
    "ex_self_rank_pct": 2,
    "ex_self_mad_z": 2,
    # R6: explicit recursive state contracts and strict scalar control domains.
    "state_latch": 2,
    "state_hold": 2,
    "state_slew_limit": 2,
    "state_deadband": 2,
    # R6: exact fractional tail membership and scale-safe MI quantile bins.
    "ts_upper_tail_coexceedance_probability": 2,
    "ts_lower_tail_coexceedance_probability": 2,
    "ts_mutual_information": 2,
    "ts_lagged_mutual_information": 2,
    # R6: robust finite-scale kernels and exact native prior/inclusive windows.
    "ts_quantile_range": 2,
    "ts_trimmed_mean": 2,
    "ts_robust_zscore_inclusive": 2,
    "ts_robust_zscore_prior": 2,
    # R6: exact geometry reference backends and scale-stable feature correlation.
    "ts_feature_mode_share": 2,
    "ts_feature_effective_rank": 2,
    "ts_feature_subspace_rotation": 2,
    # R6: bounded interval history and atomic event-domain validation.
    "event_interval_memory": 2,
    "event_local_variation": 2,
    "event_fano_factor": 2,
    "event_fano_excess": 2,
    # R6: canonical gather backends, weighted scales and censored cohort paths.
    "group_topk_mean": 2,
    "ts_value_at_argextreme": 2,
    "cs_weighted_percentile_rank": 2,
    "group_distribution_js_divergence": 2,
    "event_level_survival_share": 2,
    # R6: exact fractional tail mass and finite scaled convolution on real backends.
    "ts_fractional_difference": 2,
    "ts_fractional_difference_discarded_weight_mass": 2,
    # R6: real DC native kernels, finite threshold clocks and stable asymmetry.
    "ts_dc_event_rate": 2,
    "ts_dc_overshoot_ratio": 2,
    "ts_dc_duration_asymmetry": 2,
    "ts_dc_overshoot_asymmetry": 2,
    # R6: validated row filling, tail domains, stable entropy/DFA and exact Higuchi normalization.
    "fillna": 2,
    "ts_quantile_kurtosis": 2,
    "ts_extreme_cluster_ratio": 2,
    "ts_weighted_permutation_entropy": 2,
    "ts_hurst_dfa": 2,
    "ts_higuchi_fractal_dimension": 2,
    # R6: real cross-sectional estimators and exact floor-based stratification.
    "cs_isolation_forest_score": 2,
    "cs_factor_bucket_return": 2,
    "cs_empirical_bayes_shrinkage": 2,
    "cs_shrink_to_group_mean": 2,
    "ts_stratified_mean_spread": 2,
    # R6: canonical structure backends, scale invariance and actual per-bin KM support.
    "ts_bures_corr_shift": 2,
    "ts_kramers_moyal_drift": 2,
    "ts_kramers_moyal_diffusion": 2,
    "group_spd_feature_structure_shift": 2,
    # R6: real model backends and scale-stable dimensionless estimates.
    "ts_variance_ratio_proxy": 2,
    "ts_lo_mackinlay_vr": 2,
    "ts_lo_mackinlay_z": 2,
    "ts_cumulative_deviation_score": 2,
    "ts_level_shift_score": 2,
    "ts_vol_shift_score": 2,
    # R6: actual interval semantics replace proxies; exact prior-window reference.
    "ts_interval_union_coverage": 2,
    "ts_interval_occupancy_entropy": 2,
    "ts_interval_occupancy_mode_distance": 2,
    "ts_interval_nesting_depth": 2,
    "ts_interval_exploration_efficiency": 2,
    "ts_interval_overlap_connected_component_ratio": 2,
    # R6: alpha/span parity, unbiased one-weight undefined state and NaN fill.
    "ts_ewm_std": 2,
    "ts_ewm_var": 2,
    "fillna_const": 2,
    "nonfinite_to_num": 2,
    # R6: bounded finite group imputation and enforced ffill lineage on all backends.
    "group_impute_median": 2,
    "ts_ffill_limited": 2,
    # R6: finite support, stable scale and causal backend-identical risk kernels.
    "ts_positive_ratio": 2,
    "ts_negative_ratio": 2,
    "ts_zero_ratio": 2,
    "ts_abs_concentration": 2,
    "ts_abs_entropy": 2,
    "ts_abs_entropy_normalized": 2,
    "ts_abs_entropy_nats": 2,
    "ts_downside_deviation": 2,
    "ts_upside_deviation": 2,
    "ts_current_drawdown_duration": 2,
    "ts_time_under_water": 2,
    "ts_best_lag_corr_raw": 2,
    "ts_best_lag_corr_excess": 2,
    "ts_price_delay": 2,
    # R6: actual scalar contracts, ordered confirmed pivots, and unknown warmup.
    "pattern_triple_top": 2,
    "pattern_triple_bottom": 2,
    "pattern_123_bull": 2,
    "pattern_123_bear": 2,
    "pattern_rounding_bottom": 2,
    "pattern_rounding_top": 2,
    "pattern_cup": 2,
    "pattern_cup_handle": 2,
    "pattern_bull_pennant": 2,
    "pattern_bear_pennant": 2,
    "pattern_breakout_retest": 2,
    "pattern_breakdown_retest": 2,
    # R6: finite support, exact prior-window bounds and consistent numeric backends.
    "round": 2,
    "truncate": 2,
    "winsorize": 2,
    "winsorize_mean": 2,
    "cs_bucket": 2,
    "cs_bucket_historical": 2,
    "ts_argmax": 2,
    "ts_argmin": 2,
    # R6: canonical/legacy window keywords must reach the same numerical kernel.
    "ts_std": 2,
    "ts_sum": 2,
    "ts_max": 2,
    "ts_min": 2,
    "ts_median": 2,
    "ts_delta": 2,
    "ts_delay": 2,
    # R6: never standardize missing/nonfinite or constant cross-sectional signals.
    "valuation_cashflow_disagreement": 2,
    # R6: envelope backends consume supplied bands and use identical causal formulas.
    "ts_envelope_compression": 2,
    "ts_envelope_pressure": 2,
    "ts_envelope_boundary_dwell": 2,
    # R6: widen hour/minute arithmetic before multiplying; preserve 09:30 as 570.
    "intra_interval_return": 2,
    "intra_interval_volume_share": 2,
    "intra_interval_amount_share": 2,
    # R6: missing/empty/nonfinite memberships never form leave-one-out groups.
    "group_ex_self_mean": 3,
    "group_ex_self_weighted_mean": 3,
    "ts_bds_statistic": 2,  # v9: common-center variance and conditioned correlation integral
    # v9: stable episode-first anchors and local unknown-event coverage.
    "event_historical_response_mean": 3,
    "event_historical_response_sign_balance": 3,
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
    "ts_huber_regression_in_sample_resid": 3,  # v3: explicit scale-degenerate failure
    "ts_huber_regression_predictive_resid": 3,
    "ts_vector_state_mahalanobis": 3,  # v9: relative covariance geometry and explicit nullspace failure
    "ts_butterworth_lowpass_causal": 2,  # v9: cutoff is cycles/bar, not Nyquist fraction
    "ts_local_lyapunov_exponent": 2,  # v9: true-bar embeddings and complete successor candidates
    "ts_cross_spectral_coherence": 2,  # v9: unit-invariant Welch energy gate
    "ts_cross_spectral_phase": 2,  # v9: shared valid bins and identifiable direction
    "ts_bicoherence_top_decile_mean": 2,  # v9: unit-invariant power/validity
    "ts_bicoherence_top_decile_excess": 2,
    # v9: explicit Haar scales/window domain and one-sided Parseval weights.
    "ts_wavelet_low_frequency_ratio": 3,  # R6: scale-normalized finite energy
    "ts_wavelet_high_frequency_ratio": 3,
    "ts_wavelet_entropy": 3,
    "ts_wavelet_energy_slope": 3,
    "ts_spectral_low_frequency_ratio": 3,
    "ts_expectile": 3,  # R6: exact native default/support contracts and strict scalar gate
    "ts_expectile_beta": 3,  # R6: real aligned pair topology and exact native defaults
    "ts_kernel_granger_score": 2,  # v9: pairwise RBF geometry avoids cancellation
    "ts_hill_tail_index": 2,  # v9: strictly positive classic-Hill domain
    "ts_matrix_profile_motif_frequency": 2,
    "ts_matrix_profile_neighbor_dispersion": 2,
    "ts_residualized_hsic": 2,  # v9: explicit jointly feasible purged folds
    "ts_causal_local_linear_smoother": 2,  # v9: predict current physical endpoint
    # v2: canonical Ehlers recurrence averages current and previous input.
    "ts_super_smoother": 3,
    # v2: pairwise midpoints and the final even median avoid overflow and
    # subnormal-loss from premature addition.
    "ts_hodges_lehmann_location": 2,
    # v2: the static feasible domain requires enough rows to populate bins.
    "ts_binned_response_monotonicity": 3,
    "ts_binned_response_curvature": 3,
    # v2: dimensionless max-absolute window scaling; zero variation undefined.
    "intraday_jump_test_stat": 3,
    # v2: normalized weights/Kish moments preserve unit invariance.
    "ts_weighted_standardized_moment": 3,
    # v2: nested threshold + aggregation windows require 2*(window-1) history;
    # prior thresholds reject the impossible min_periods == window boundary.
    "group_tail_centrality": 2,
    "group_tail_lead_score": 2,
    # v2: normalized ACF with Geyer paired-positive prefix; IMS applies cummin.
    "ts_autocorrelation_time": 2,
    "ts_autocorrelation_time_initial_positive_sequence": 2,
    # v2: input kind is explicit/typed; unknown or wrong domains fail closed.
    "ts_return_spectral_entropy": 2,
    "ts_activity_spectral_entropy": 2,
    "ts_spectral_entropy": 2,
    "ts_detrended_level_spectral_entropy": 2,
    # v2: max-scaled probability weights; expected shortfall also requires
    # positive selected-tail mass and retained fractional member support.
    "ts_weighted_semivariance": 3,  # R6: stable weighted scale and physical-gap backend parity
    "ts_weighted_downside_deviation": 3,  # R6: stable weighted scale and physical-gap backend parity
    "ts_weighted_expected_shortfall": 2,
    "ts_weighted_drawdown_area": 3,  # R6: stable weighted scale and physical-gap backend parity
    # v2: shared quantile edges are numerically stable and use the corrected
    # Markov/Kramers-Moyal state discretization contract.
    "ts_markov_persistence": 2,
    "ts_markov_state_entropy": 2,
    "ts_markov_transition_surprisal": 2,
    "ts_markov_entropy_production": 2,
    "ts_kramers_moyal_local_stability": 2,
    "ts_markov_committor": 2,
    "ts_markov_mean_first_passage_time": 3,  # R21: replace incorrect Polars placeholder
    "ts_markov_spectral_gap": 2,
    "ts_markov_stationary_surprisal": 2,
    "ts_km_equilibrium_distance": 3,  # R43: Polars nearest attractor and bracketed zero plateaus
    "ts_km_diffusion_gradient": 3,  # R21: exact reference-backed Polars path
    "ts_km_quasipotential_depth": 3,  # R21: replace Polars volatility placeholder
    "ts_active_information_storage": 3,  # R21: replace all-NaN Polars placeholder
    # v2: stable true beta with a common-scale, dimensionless break ratio.
    "ts_beta_break_score": 2,
    # v2: scaled SVD/common mask and a joint window feasibility guard.
    "ts_price_delay": 2,
    # v2: stable SHA-256 absolute-time seed and canonical semantic identity.
    "ts_best_lag_corr_excess": 2,
    # v2: within-window dimensionless centering preserves the GLR split grid.
    "ts_glr_mean_shift_score": 2,
    "ts_glr_variance_shift_score": 2,
    # v2: Theiler-aware feasible domain and common-scale stable Euclidean KNN.
    "ts_delay_intrinsic_dimension": 2,
    # v2: independent per-scale coverage caps and common-unit scale fitting.
    "ts_vol_pvariation_roughness": 3,
    "ts_vol_scaling_break": 3,
    # v2: squared-mass intraday summaries are stable, zero mass is undefined,
    # and minute cross-session row semantics are explicit. RV scale cohorts
    # remain preserved independently rather than intersected.
    "intraday_volatility_time_centroid": 2,
    "intraday_volatility_concentration": 2,
    "intraday_volatility_entropy": 2,
    "intraday_realized_semivariance_balance": 2,
    "intraday_rv_signature_curvature": 2,
    # v2: corrected MODWT band conditional-dependence semantics.
    "ts_modwt_band_corr": 2,
    # v2: profile optimization requires certified BFGS convergence.
    "ts_fisher_information_shift": 3,  # R23: legacy Polars proxy replaced by canonical delegate.
    "ts_persistence_diagram_shift": 2,  # R23: legacy null placeholder replaced by canonical delegate.
    # v9: actual convex first/second-difference L1 objectives and bounded solve.
    "ts_total_variation_filter_trailing": 2,
    "ts_l1_trend_filter_trailing": 2,
    "group_peer_beta_deviation": 2,
    # v9: stable prewarped phase-normalized Bessel SOS (gap policy unchanged).
    "ts_bessel_lowpass_causal": 2,
    # v9: correct robust-fit coefficient order and one finite fit/query cohort.
    "cs_huber_resid": 4,  # v4: finite-scale fits and complete final backend contracts
    "cs_lad_resid": 3,
    # v9: complete first-passage horizons and physical-bar breadth history.
    "ts_first_passage_bias": 2,
    "ts_first_passage_hit_probability": 3,
    "ts_first_passage_conditional_time": 3,
    "group_feature_mode_share": 2,
    "group_feature_effective_rank": 2,
    "group_feature_mode_localization": 2,
    "group_feature_spectral_gap": 2,
    "group_feature_second_mode_localization": 2,
    # v9: defined no-reversal zero; valid turnover mass independent of costs.
    "event_response_reversal_strength": 2,
    "ts_turnover_reference_price": 3,
    "ts_turnover_cost_dispersion": 3,
    "ts_turnover_profit_share": 3,
    "ts_turnover_holding_age": 3,
    "ts_turnover_near_cost_mass": 3,
    "ts_turnover_cost_quantile_distance": 3,
    "ts_turnover_cost_entropy": 3,
    "ts_turnover_cost_mode_distance": 3,
    "ts_turnover_cost_skew": 3,
    "ts_turnover_age_dispersion": 3,
    "ts_turnover_old_mass": 3,
    "ts_turnover_cost_entropy_vol_scaled": 3,
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
    # R23: finite support, string group identity and declared fallback policies.
    "group_mean": 2,
    "group_sum": 2,
    "group_min": 2,
    "group_max": 2,
    "group_count": 2,
    "group_std": 2,
    "group_zscore": 2,
    "group_rank": 2,
    "group_normalize": 2,
    "group_percentile": 4,  # v3: backend emitters preserve current quantile/null semantics
    "signed_log": 1,  # sign(x)*log(abs(x)+1e-10)
    "compare": 2,  # v2: NULL/NaN propagate
    "maximum": 2,
    "minimum": 2,
    "protected_log": 2,  # NULL preserved
    "protected_div": 3,
    "rank": 2,  # R21: Polars direct rank excludes non-finite support.
    "rank_pct": 1,
    "ts_beta": 3,  # v3: paired finite cohort + explicit ddof semantics
    "ts_zscore": 2,  # v2: all declared window/numeric parameters affect execution
    # m_beta / rolling_beta are runtime aliases of ts_beta and inherit v3.
    "rolling_beta_to_market": 2,  # v2: paired cohort + benchmark alignment
    "downside_beta": 2,  # v2: downside mask after paired finite alignment
    "tail_beta": 2,  # v2: tail mask after paired finite alignment
    "intra_entropy": 2,  # v2: normalized histogram probability entropy
    "intra_limit_first_hit_time": 3,  # v3: labelled timezone-preserving exact Polars delegate
    "intra_limit_duration": 3,  # v3: daily grid and exchange-local daily limit binding
    "intra_limit_reopen_count": 3,  # v3: daily transitions and exchange-local limit binding
    "fin_component_score": 3,  # v2: finite FALSE contributes 0; missing remains NaN
    "ts_kurt": 3,  # v2: backend emitter parity for finite/Inf handling
    "cs_quantile": 3,  # v3: full p contract and finite-scale quantile interpolation
    "ts_quantile": 3,  # v2: backend emitter parity for finite/Inf handling
    "true_range": 2,  # v2: backend emitter parity for finite/Inf handling
    "ts_ema": 2,  # v2: canonical EMA emitter parity; aliases inherit this version
    "MACD_line": 3,  # v2: backend EMA/finite semantics aligned
    "MACD_signal": 3,  # v2: backend EMA/finite semantics aligned
    "MACD_hist": 3,  # v2: backend EMA/finite semantics aligned
    "ts_max_drawdown": 2,  # v2: generic delegate binds positional scalar parameters
    "overnight_return": 3,  # R6: supplied price panels, no extra shift/proxy, native basis/axis parity
    "open_close_return": 3,  # R6: supplied price panels, no extra shift/proxy, native basis/axis parity
    "open_to_vwap_return": 3,  # R6: supplied price panels, no extra shift/proxy, native basis/axis parity
    "vwap_to_close_return": 3,  # R6: supplied price panels, no extra shift/proxy, native basis/axis parity
    # R22: canonical Polars delegates replace broken legacy implementations.
    "ts_causal_local_linear_smoother": 2,
    "ts_causal_savgol_endpoint": 2,
    "ts_betti_1_max_persistence": 2,
    "ts_corr": 3,  # R23: avoid overflow/underflow in norm products.
    "ts_regression_r2": 2,
    "ts_regression_slope": 2,  # R23: includes retval=r2.
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
