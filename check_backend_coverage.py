#!/usr/bin/env python3
"""Check backend coverage for new operators."""
import sys
sys.path.insert(0, '.')

from cleaned_operators.registry import OperatorRegistry

# R47 operators
r47_ops = [
    'ALMA', 'CoppockCurve', 'ElderRay', 'FisherTransform', 'HMA', 'QQE', 'RSX',
    'cs_predictability_mosaic_score', 'intra_consolidation_quality',
    'intra_eod_reversal_decomposition', 'intra_event_pre_post_contrast',
    'intra_event_window_reduce', 'intra_impulse_event_detector',
    'intra_limit_pre_hit_pressure_profile', 'intra_liquidity_resilience_curve_fit',
    'intra_multiresolution_resample_reduce', 'intra_neighbor_event_class',
    'intra_post_impulse_response', 'intra_probe_outcome_score',
    'intra_range_gap_flag', 'intra_response_curve_features',
    'intra_round_price_barrier_response', 'intra_round_price_clustering_share',
    'intra_same_slot_zscore', 'intra_session_boundary_jump',
    'intra_slice_mask_pair_reduce', 'intra_slice_mask_reduce',
    'intra_state_count', 'intra_state_dwell_stats', 'intra_state_follow_beta',
    'intra_state_follow_corr', 'intra_state_follow_ratio',
    'intra_state_interval_moment', 'intra_state_pair_same_slot_corr',
    'intra_state_sum', 'intra_state_transition_entropy', 'intra_state_vwap',
    'intra_supply_absorption_score', 'intra_volume_at_price_profile',
    'intra_volume_profile_peak_geometry', 'intra_volume_profile_supply_structure',
    'intra_volume_profile_value_area', 'panel_async_beta_ex_self',
    'panel_factor_pocket_strength', 'panel_predictability_mosaic_score',
    'turnover_chip_age_cost_surface', 'turnover_chip_overhang_surface'
]

# TRUE_GAP operators (from 2026-08-13)
true_gap_ops = [
    'fiscal_acceleration', 'fiscal_pct_change', 'fiscal_rolling_std',
    'fiscal_accrual_quality', 'fiscal_direction_consistency',
    'ts_bessel_lowpass_causal', 'ts_fir_lowpass_causal',
    'ts_spectral_lowpass_trailing', 'ts_causal_savgol_endpoint',
    'panel_day_night_beta_gap', 'pastor_stambaugh_beta', 'price_delay_score',
    'report_asof', 'event_window_return_asof', 'cs_isolation_forest_score',
    'cs_factor_bucket_return', 'cs_empirical_bayes_shrinkage',
    'cs_shrink_to_group_mean', 'panel_peer_graph_aggregate',
    'ts_alpha_beta_filter', 'ts_h_infinity_level_filter',
    'ts_adaptive_noise_kalman', 'ts_student_t_kalman_filter',
    'ts_mcginley_dynamic', 'ts_vidya', 'ts_one_euro_filter',
    'ts_nlms_filter', 'ts_rls_filter', 'ts_ssa_denoise_trailing',
    'ts_wavelet_shrinkage_trailing', 'ts_total_variation_filter_trailing',
    'ts_l1_trend_filter_trailing', 'financial_snapshot_lag',
    'same_calendar_day_mean', 'same_calendar_month_return',
    'fiscal_capital_stock', 'fiscal_perpetual_inventory',
    'cash_flow_lifecycle_stage', 'laborforce_efficiency', 'years_since_date',
    'cs_normalize', 'cs_clip', 'cs_median', 'cs_iqr', 'cs_trim_mean',
    'cs_outlier_score', 'cs_robust_scale', 'ts_returns', 'ts_sma',
    'ts_cumsum', 'ts_cumprod'
]

all_new_ops = sorted(set(r47_ops + true_gap_ops))

# Load all operators by calling load_all()
from cleaned_operators import load_all
load_all()

print('Backend Coverage for New Operators:')
print('=' * 100)
print(f'{"Operator":<55} | {"pandas":<8} | {"polars":<8} | {"sql":<8}')
print('=' * 100)

pandas_only = []
has_polars = []
has_sql = []
not_found = []

for op_name in all_new_ops:
    if op_name in OperatorRegistry._operators:
        backends = OperatorRegistry._operators[op_name]

        has_pd = 'pandas_numpy' in backends
        has_pl = 'polars' in backends
        has_ddb = 'sql' in backends

        pd_str = 'YES' if has_pd else 'NO'
        pl_str = 'YES' if has_pl else 'NO'
        sql_str = 'YES' if has_ddb else 'NO'

        print(f'{op_name:<55} | {pd_str:<8} | {pl_str:<8} | {sql_str:<8}')

        if has_pd and not has_pl and not has_ddb:
            pandas_only.append(op_name)
        elif has_pl:
            has_polars.append(op_name)
        if has_ddb:
            has_sql.append(op_name)
    else:
        print(f'{op_name:<55} | {"NOT FOUND":<27}')
        not_found.append(op_name)

print('=' * 100)
print(f'\nSummary:')
print(f'  Total new operators: {len(all_new_ops)}')
print(f'  Pandas-only (need Polars): {len(pandas_only)}')
print(f'  Have Polars: {len(has_polars)}')
print(f'  Have SQL: {len(has_sql)}')
print(f'  Not found in catalog: {len(not_found)}')

if pandas_only:
    print(f'\n\nOperators needing Polars backend ({len(pandas_only)}):')
    for op in pandas_only:
        print(f'  - {op}')

if not_found:
    print(f'\n\nOperators not found in catalog ({len(not_found)}):')
    for op in not_found:
        print(f'  - {op}')
