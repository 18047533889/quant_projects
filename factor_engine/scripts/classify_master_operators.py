#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classify 245 master list operators against current HEAD implementation.

This script performs semantic diff and disposition classification for all
operators from FactorEngine_全部新增算子_Master清单_20260812.md.
"""
import sys
sys.path.insert(0, '/home/shw/quant_projects/factor_engine')

from cleaned_operators.operator_surface import (
    DAILY_CANONICALS,
    extended_only_canonicals,
    RESEARCH_ONLY_CANONICALS,
)

# Master list operators (245 total)
# R41-R47 historical candidates (156)
R41_R47_CANDIDATES = [
    # R41 (31)
    'fiscal_acceleration', 'fiscal_delta', 'fiscal_lag', 'fiscal_pct_change',
    'intra_market_profile_corr_ex_self', 'report_asof', 'same_calendar_month_return',
    'same_clock_lag', 'cn_sma', 'cs_isolation_forest_score', 'event_window_return_asof',
    'financial_snapshot_lag', 'fiscal_capital_stock', 'fiscal_rolling_regression',
    'fiscal_rolling_std', 'intraday_value_at_extreme_state', 'panel_peer_graph_aggregate',
    'price_delay_score', 'winsorized_ratio', 'HMA', 'KAMA', 'QQE', 'RSX', 'WMA',
    'cs_factor_bucket_return', 'pastor_stambaugh_beta', 'same_calendar_day_mean',
    'ALMA', 'CoppockCurve', 'ElderRay', 'FisherTransform',

    # R42 (25)
    'intra_same_slot_zscore', 'intra_neighbor_event_class', 'intra_state_count',
    'intra_state_sum', 'intra_state_vwap', 'intra_state_interval_moment',
    'intra_state_follow_ratio', 'intra_state_follow_beta', 'intra_state_follow_corr',
    'intra_state_pair_same_slot_corr', 'intra_range_gap_flag',
    'intra_volume_peak_ridge_valley_state', 'intra_price_peak_ridge_valley_state',
    'intra_smart_money_vwap_ratio', 'intra_smart_money_fcm_score',
    'panel_apm_residual_tstat', 'panel_day_night_beta_gap', 'panel_async_beta_ex_self',
    'intra_jump_wavelet_morphology', 'intra_cojump_breadth_ex_self',
    'cs_topological_anomaly_score', 'panel_predictability_mosaic_score',
    'panel_factor_pocket_strength', 'intra_price_shape_cosine_match',
    'intra_distribution_moment',

    # R43 (11)
    'intra_slice_mask_reduce', 'intra_slice_mask_pair_reduce',
    'intra_multiresolution_resample_reduce', 'panel_intraday_ordered_reduce',
    'ts_ewm_std', 'panel_ewm_beta_ex_self', 'panel_cmra_ex_self',
    'intra_idiosyncratic_semivariance_balance_ex_self', 'panel_similarity_crowding_score',
    'panel_cluster_risk_score', 'intra_functional_beta_profile_ex_self',

    # R44 (9)
    'fiscal_cost_stickiness_score', 'fundamental_cash_flow_duration',
    'intra_round_price_clustering_share', 'intra_round_price_barrier_response',
    'intra_session_segment_reduce', 'intra_session_boundary_jump',
    'trading_calendar_mask', 'suspension_restart_response',
    'intra_limit_pre_hit_pressure_profile',

    # R45 (53)
    'intra_impulse_event_detector', 'intra_post_impulse_response',
    'intra_probe_outcome_score', 'intra_supply_absorption_score',
    'intra_consolidation_quality', 'intra_response_curve_features',
    'intra_volume_at_price_profile', 'intra_volume_profile_peak_geometry',
    'intra_volume_profile_supply_structure', 'turnover_chip_distribution',
    'turnover_chip_distribution_transport', 'turnover_chip_age_cost_surface',
    'intra_log_signature_features', 'intra_functional_pca_shape',
    'intra_functional_motif_score', 'intra_shapelet_match', 'intra_rqa_features',
    'intra_persistent_homology_features', 'intra_topological_anomaly_score',
    'intra_optimal_transport_profile_shift', 'intra_dmd_koopman_features',
    'intra_kalman_latent_price', 'intra_state_space_volume_components',
    'intra_hmm_state_features', 'intra_hsmm_duration_features',
    'intra_change_point_sequence_features', 'intra_hawkes_event_features',
    'intra_multifractal_spectrum', 'intra_wavelet_scattering_features',
    'intra_emd_hilbert_huang_features', 'intra_visibility_graph_features',
    'intra_information_flow_features', 'intra_covariance_manifold_shift',
    'intra_diffusion_map_state', 'intra_common_trading_intensity',
    'intra_price_efficiency_state_space', 'intra_neural_cde_embedding',
    'intra_contrastive_path_embedding', 'intra_matrix_profile_session_features',
    'intra_dtw_archetype_features', 'intra_local_conditional_entropy',
    'intra_business_time_deformation', 'intra_quantile_dependence_features',
    'intra_kramers_moyal_dynamics', 'intra_extreme_event_interval_memory',
    'intra_signature_lead_lag_network', 'intra_validated_lead_lag_network',
    'intra_dynamic_stock_graph_features', 'intra_tensor_common_mode',
    'intra_topological_peer_anomaly', 'intra_critical_transition_score',
    'intra_realized_measure_state_vector', 'intra_symbolic_dynamics_features',

    # R46 (20)
    'intra_eod_reversal_decomposition', 'intra_volume_shock_state',
    'intra_comovement_curve_ex_self', 'intra_price_volume_cross_wavelet',
    'intra_bicoherence_features', 'intra_frequency_granger_price_volume',
    'intra_range_competition_profile', 'intra_absorption_curve_area',
    'intra_volume_profile_value_area', 'turnover_chip_overhang_surface',
    'fundamental_latent_balance_sheet_factor', 'fundamental_working_capital_financing_state',
    'fundamental_cost_stickiness_panel', 'intra_multiscale_state_residence',
    'intra_visibility_motif_transition', 'intra_recurrence_network_features',
    'intra_transfer_operator_metastability', 'intra_local_drift_diffusion_surface',
    'intra_session_ot_map', 'panel_intraday_low_rank_residual_ex_self',

    # R47 (7)
    'fin_schema_gate', 'ts_lagged_predictability_score', 'cs_predictability_mosaic_score',
    'intra_functional_autoencoder_score', 'intra_hmm_posterior_entropy',
    'intra_liquidity_resilience_curve_fit', 'intra_function_on_function_anomaly_response',
]

# Additional primitives (8)
ADDITIONAL_PRIMITIVES = [
    'intra_event_window_reduce', 'intra_event_pre_post_contrast',
    'ts_event_decay_kernel', 'intra_state_transition_entropy',
    'intra_state_dwell_stats', 'cs_robust_mahalanobis_score',
    'intra_piecewise_linear_path_features', 'ts_online_change_point_score',
]

# Filter/State-space candidates (28)
FILTER_STATE_CANDIDATES = [
    'state_change_point_adaptive_ema', 'state_filter_reset_on_break',
    'state_gain_scheduler', 'state_elastic_turnover_prox',
    'cs_shrink_to_market_mean', 'cs_shrink_to_group_mean',
    'cs_empirical_bayes_shrinkage', 'ts_l1_trend_filter_trailing',
    'ts_total_variation_filter_trailing', 'ts_robust_kalman_level',
    'ts_one_euro_filter', 'ts_alpha_beta_filter',
    'ts_bessel_lowpass_causal', 'ts_fir_lowpass_causal',
    'ts_causal_savgol_endpoint', 'ts_vidya', 'ts_mcginley_dynamic',
    'ts_nlms_filter', 'ts_rls_filter', 'ts_student_t_kalman_filter',
    'ts_adaptive_noise_kalman', 'ts_wavelet_shrinkage_trailing',
    'ts_ssa_denoise_trailing', 'cs_peer_graph_smooth',
    'ts_h_infinity_level_filter', 'ts_particle_filter_level',
    'ts_modwt_denoise_trailing', 'ts_spectral_lowpass_trailing',
]

# Fiscal/Relation backlog (31)
FISCAL_RELATION_BACKLOG = [
    'row_sum_skipna', 'industry_size_neutralize', 'fiscal_perpetual_inventory',
    'fiscal_standardized_surprise', 'fiscal_direction_consistency',
    'fiscal_pair_direction_agreement', 'fiscal_autocorr', 'fiscal_ar_resid_std',
    'fiscal_rolling_regression_resid', 'fiscal_asymmetric_elasticity',
    'industry_fiscal_resid', 'fiscal_accrual_quality',
    'accounting_comparability_score', 'fiscal_asymmetric_timeliness',
    'fiscal_reversal_ratio', 'fiscal_logit_score', 'cash_flow_lifecycle_stage',
    'date_diff_days', 'fin_seasonal_zscore', 'fin_seasonal_percentile',
    'fiscal_true_streak', 'group_cs_resid', 'holder_concentration_change',
    'laborforce_efficiency', 'relation_entropy', 'relation_jaccard',
    'relation_period_change', 'revision_delta', 'years_since_date',
    'fundamental_staleness_days', 'fiscal_rolling_slope',
]

# PROD_RECERTIFY (22) - existing operators needing certification
PROD_RECERTIFY = [
    'ts_hampel_filter_causal', 'ts_median3_causal', 'ts_rolling_median_causal',
    'ts_robust_ema', 'ts_super_smoother', 'ts_kama',
    'ts_butterworth_lowpass_causal', 'ts_causal_local_linear_smoother',
    'state_adaptive_deadband', 'state_rank_deadband', 'state_quantile_hysteresis',
    'state_adaptive_slew_limit', 'state_l1_turnover_prox', 'state_l2_partial_adjustment',
    'state_cost_aware_deadband', 'state_cost_aware_slew',
    'state_confidence_weighted_ema', 'state_uncertainty_deadband',
    'ts_quantile_range', 'ts_trimmed_mean', 'ts_robust_zscore_inclusive',
    'ts_robust_zscore_prior',
]

def classify_operator(name: str, all_existing: set, daily: set, extended: set, research: set):
    """Classify a single operator."""

    # Check if PROD_RECERTIFY
    if name in PROD_RECERTIFY:
        if name in all_existing:
            surface = 'daily' if name in daily else ('extended' if name in extended else 'research')
            return {
                'disposition': 'PROD_RECERTIFY',
                'reason': f'Exists in {surface} surface, needs backend/evidence certification',
                'current_exact': name,
                'current_surface': surface,
            }

    # EXISTING_EXACT
    if name in all_existing:
        surface = 'daily' if name in daily else ('extended' if name in extended else 'research')
        return {
            'disposition': 'EXISTING_EXACT',
            'reason': f'Found in {surface} surface',
            'current_exact': name,
            'current_surface': surface,
        }

    # Check aliases
    alias_map = {
        'KAMA': 'ts_kama',
        'cn_sma': 'ts_sma_cn',
        'price_delay_score': 'ts_price_delay',
        'financial_snapshot_lag': 'fin_lag',
        'fiscal_lag': 'fin_lag',
        'fiscal_delta': 'fin_diff',
        'fiscal_pct_change': 'fin_pct_change',
    }

    if name in alias_map:
        canonical = alias_map[name]
        if canonical in all_existing:
            surface = 'daily' if canonical in daily else ('extended' if canonical in extended else 'research')
            return {
                'disposition': 'EXISTING_ALIAS',
                'reason': f'Alias for {canonical} ({surface})',
                'current_alias': canonical,
                'current_surface': surface,
            }

    # Check semantic equivalents
    semantic_map = {
        'fiscal_acceleration': 'fin_growth_acceleration',
        'report_asof': 'ARCHITECTURE_SUPERSEDED',
        'row_sum_skipna': 'row_sum_skipna',  # Need to check
        'industry_size_neutralize': 'industry_size_neutralize',  # Need to check
        'cash_flow_lifecycle_stage': 'cash_flow_lifecycle_stage',  # Need to check
        'date_diff_days': 'date_diff_days',  # Need to check
        'fiscal_perpetual_inventory': 'fiscal_perpetual_inventory',  # Need to check
        'holder_concentration': 'holder_concentration',  # Need to check
        'relation_entropy': 'relation_entropy',  # Need to check
        'relation_hhi': 'relation_hhi',  # Need to check
    }

    if name in semantic_map:
        equiv = semantic_map[name]
        if equiv == 'ARCHITECTURE_SUPERSEDED':
            return {
                'disposition': 'ARCHITECTURE_SUPERSEDED',
                'reason': 'PIT join should be handled by DataAccess layer',
            }
        if equiv in all_existing:
            surface = 'daily' if equiv in daily else ('extended' if equiv in extended else 'research')
            return {
                'disposition': 'EXISTING_EQUIVALENT',
                'reason': f'Semantic equivalent: {equiv} ({surface})',
                'semantic_equivalent': equiv,
                'current_surface': surface,
            }

    # Blocked operators
    blocked_ops = {
        'fin_schema_gate': 'Missing schema validation infrastructure',
        'laborforce_efficiency': 'Missing employee_count field in A-share data',
        'revision_delta': 'Requires real vintage/revision identity',
    }

    if name in blocked_ops:
        return {
            'disposition': 'BLOCKED_DATA_CONTRACT',
            'reason': blocked_ops[name],
        }

    # Research-first indicators (complex topological/ML)
    research_indicators = [
        'cs_isolation_forest_score', 'intra_smart_money_fcm_score',
        'cs_topological_anomaly_score', 'panel_predictability_mosaic_score',
        'intra_persistent_homology_features', 'intra_topological_anomaly_score',
        'intra_dmd_koopman_features', 'intra_hmm_state_features',
        'intra_hsmm_duration_features', 'intra_hawkes_event_features',
        'intra_wavelet_scattering_features', 'intra_emd_hilbert_huang_features',
        'intra_visibility_graph_features', 'intra_information_flow_features',
        'intra_covariance_manifold_shift', 'intra_diffusion_map_state',
        'intra_neural_cde_embedding', 'intra_contrastive_path_embedding',
        'intra_dynamic_stock_graph_features', 'intra_topological_peer_anomaly',
        'ts_online_change_point_score', 'ts_nlms_filter', 'ts_rls_filter',
        'ts_student_t_kalman_filter', 'ts_adaptive_noise_kalman',
        'cs_peer_graph_smooth', 'ts_h_infinity_level_filter',
        'ts_particle_filter_level', 'ts_modwt_denoise_trailing',
        'ts_spectral_lowpass_trailing', 'cs_predictability_mosaic_score',
        'intra_functional_autoencoder_score', 'intra_hmm_posterior_entropy',
        'intra_function_on_function_anomaly_response',
    ]

    if name in research_indicators:
        return {
            'disposition': 'RESEARCH_ONLY',
            'reason': 'Complex ML/topological method, research surface first',
        }

    # Default: TRUE_GAP
    return {
        'disposition': 'TRUE_GAP_IMPLEMENT',
        'reason': 'Not found in current implementation, needs development',
    }

def main():
    # Load all existing operators
    daily_set = set(DAILY_CANONICALS)
    extended_set = extended_only_canonicals()
    research_set = set(RESEARCH_ONLY_CANONICALS)
    all_existing = daily_set | extended_set | research_set

    print(f"Current implementation: {len(all_existing)} operators")
    print(f"  Daily: {len(daily_set)}, Extended: {len(extended_set)}, Research: {len(research_set)}")
    print()

    # Combine all master list operators
    all_candidates = (
        R41_R47_CANDIDATES +
        ADDITIONAL_PRIMITIVES +
        FILTER_STATE_CANDIDATES +
        FISCAL_RELATION_BACKLOG +
        PROD_RECERTIFY
    )

    # Deduplicate
    unique_candidates = sorted(set(all_candidates))
    print(f"Master list: {len(unique_candidates)} unique candidates")
    print()

    # Classify each
    results = []
    for name in unique_candidates:
        classification = classify_operator(name, all_existing, daily_set, extended_set, research_set)
        results.append((name, classification))

    # Summary
    disposition_counts = {}
    for name, cls in results:
        disp = cls['disposition']
        disposition_counts[disp] = disposition_counts.get(disp, 0) + 1

    print("Classification Summary:")
    print("=" * 60)
    for disp, count in sorted(disposition_counts.items()):
        print(f"  {disp:30s} {count:4d}")
    print("=" * 60)
    print(f"  TOTAL{' ' * 24} {len(results):4d}")
    print()

    # Write CSV
    import csv
    output_path = '/home/shw/quant_projects/factor_engine/evidence/operator_master_gap_preflight_854bdc22.csv'

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Summary as comments
        writer.writerow(['# Operator Master List Gap Analysis - HEAD: 854bdc2278678e3db03c894c059cd5fdbb7dac1b'])
        writer.writerow(['# Generated: 2026-08-12'])
        writer.writerow([f'# Total candidates: {len(unique_candidates)}'])
        for disp, count in sorted(disposition_counts.items()):
            writer.writerow([f'# {disp}: {count}'])
        writer.writerow([])

        # Header
        writer.writerow([
            'candidate', 'source_round', 'current_exact', 'current_alias',
            'semantic_equivalent', 'composable', 'replacement', 'current_surface',
            'pandas_backend', 'polars_backend', 'duckdb_backend', 'field_legal',
            'pit_legal', 'stateful', 'checkpointable', 'parameter_evidence',
            'disposition', 'reason'
        ])

        # Data
        for name, cls in results:
            # Determine source round
            if name in R41_R47_CANDIDATES[:31]:
                source = 'R41'
            elif name in R41_R47_CANDIDATES[31:56]:
                source = 'R42'
            elif name in R41_R47_CANDIDATES[56:67]:
                source = 'R43'
            elif name in R41_R47_CANDIDATES[67:76]:
                source = 'R44'
            elif name in R41_R47_CANDIDATES[76:129]:
                source = 'R45'
            elif name in R41_R47_CANDIDATES[129:149]:
                source = 'R46'
            elif name in R41_R47_CANDIDATES[149:]:
                source = 'R47'
            elif name in ADDITIONAL_PRIMITIVES:
                source = 'PRIMITIVES'
            elif name in FILTER_STATE_CANDIDATES:
                source = 'FILTER_STATE'
            elif name in FISCAL_RELATION_BACKLOG:
                source = 'FISCAL_RELATION'
            elif name in PROD_RECERTIFY:
                source = 'PROD_RECERTIFY'
            else:
                source = 'UNKNOWN'

            writer.writerow([
                name,
                source,
                cls.get('current_exact', ''),
                cls.get('current_alias', ''),
                cls.get('semantic_equivalent', ''),
                '',  # composable
                cls.get('replacement', ''),
                cls.get('current_surface', ''),
                '',  # pandas_backend
                '',  # polars_backend
                '',  # duckdb_backend
                '',  # field_legal
                '',  # pit_legal
                '',  # stateful
                '',  # checkpointable
                '',  # parameter_evidence
                cls['disposition'],
                cls['reason'],
            ])

    print(f"Written: {output_path}")

    # Print detailed breakdown
    print("\nDetailed Breakdown:")
    print("=" * 60)
    for disp in sorted(disposition_counts.keys()):
        print(f"\n{disp} ({disposition_counts[disp]}):")
        for name, cls in results:
            if cls['disposition'] == disp:
                print(f"  - {name}")

if __name__ == '__main__':
    main()
