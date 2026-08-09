#!/usr/bin/env python3
"""Quick runtime check: execute the ops that failed the previous audit once."""
from __future__ import annotations

import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
for p in (str(FE_ROOT.parent), str(FE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cleaned_operators import load_all
from cleaned_operators.production_hardening import factor_production_targets
from cleaned_operators.registry import OperatorRegistry
from scripts.audit_all_factor_production import _build_call, _panels, _to_frame, _minute_source

TARGETS = [
    'cs_impute_mean',
    'cs_impute_median',
    'cs_tail_breadth',
    'event_allan_factor',
    'event_interval_memory',
    'event_local_variation',
    'fin_accrual_ratio',
    'fin_cagr',
    'fin_capex_growth',
    'fin_cash_conversion',
    'fin_cashflow_persistence',
    'fin_cash_sales_divergence',
    'fin_component_score',
    'fin_contract_asset_growth',
    'fin_contract_liability_growth',
    'fin_delta_noa',
    'fin_earnings_cash_gap_volatility',
    'fin_earnings_persistence',
    'fin_earnings_smoothness',
    'fin_equity_capital_growth',
    'fin_expense_sales_divergence',
    'fin_financing_gap',
    'fin_goodwill_risk_score',
    'fin_growth',
    'fin_growth_acceleration',
    'fin_growth_change',
    'fin_growth_persistence',
    'fin_growth_stability',
    'fin_growth_volatility',
    'fin_inventory_sales_divergence',
    'fin_log_change',
    'fin_margin_persistence',
    'fin_net_borrowing_cashflow',
    'fin_pct_change',
    'fin_qoq',
    'fin_receivable_sales_divergence',
    'fin_roe_cash_gap',
    'fin_working_capital_accruals',
    'fin_yoy',
    'group_current_members_tail_coexceedance',
    'intraday_barrier_approach_acceleration',
    'intraday_profile_pca_residual',
    'intraday_profile_surprise_energy',
    'intraday_session_shape_novelty',
    'intraday_subsampled_rv_dispersion',
    'intraday_volatility_signature_slope',
    'intraday_wasserstein_pair_distance',
    'intra_lunch_gap_return',
    'intra_segment_amount_share',
    'intra_segment_realized_vol',
    'intra_segment_return',
    'intra_segment_volume_share',
    'intra_segment_vwap_deviation',
    'relation_distribution_excess_kurtosis',
    'relation_distribution_pearson_kurtosis',
    'session_event_recovery_score',
    'state_episode_efficiency',
    'state_episode_excursion_balance',
    'state_episode_mae',
    'state_episode_mfe',
    'state_episode_retrace_ratio',
    'state_hold',
    'state_latch',
    'ts_bicoherence_top_decile_excess',
    'ts_bicoherence_top_decile_mean',
    'ts_binned_response_curvature',
    'ts_binned_response_monotonicity',
    'ts_conditional_mutual_information',
    'ts_cross_spectral_coherence',
    'ts_cross_spectral_phase',
    'ts_distance_corr',
    'ts_distance_cov',
    'ts_dmd_dominant_frequency',
    'ts_dmd_dominant_growth_rate',
    'ts_dmd_mode_concentration',
    'ts_evt_threshold_stability',
    'ts_forbidden_ordinal_pattern_excess',
    'ts_forbidden_ordinal_pattern_ratio',
    'ts_forbidden_ordinal_pattern_signed_excess',
    'ts_generalized_hurst_exponent',
    'ts_glr_mean_shift_score',
    'ts_glr_variance_shift_score',
    'ts_hvg_assortativity',
    'ts_hvg_clustering_coefficient',
    'ts_hvg_degree_entropy',
    'ts_hvg_forward_backward_asymmetry',
    'ts_hvg_motif_entropy',
    'ts_hysteresis_age',
    'ts_hysteresis_state',
    'ts_kramers_moyal_diffusion',
    'ts_kramers_moyal_drift',
    'ts_lagged_mutual_information',
    'ts_lempel_ziv_complexity',
    'ts_mutual_information',
    'ts_ordinal_irreversibility',
    'ts_pettitt_change_score',
    'ts_recurrence_determinism',
    'ts_recurrence_laminarity',
    'ts_recurrence_longest_vertical_length',
    'ts_recurrence_mean_diagonal_length',
    'ts_residualized_hsic',
    'ts_return_spectral_entropy',
    'ts_rolling_sr_gaussian_mean_shift_score',
    'ts_spectral_entropy',
    'ts_state_entry_strength',
    'ts_state_integral',
    'ts_transfer_entropy_peak_excess',
    'ts_variance_ratio_proxy',
    'ts_fractional_difference_discarded_weight_mass',
    'ts_mean_abs_deviation',
    'ts_median_abs_deviation',
    'ts_nearest_structural_level_distance',
    'ts_rqa_determinism_fixed_rr',
    'ts_rqa_laminarity_fixed_rr',
    'ts_structural_level_density',
    'ts_structural_level_strength',
    'intraday_volume_clock_path_efficiency',
    'intraday_volume_clock_roughness',
    'ts_edge_effective_spread',
'ts_dmd_level_dominant_frequency',
'ts_dmd_level_dominant_growth_rate',
'ts_dmd_level_mode_concentration',
'ts_dmd_return_dominant_frequency',
'ts_dmd_return_dominant_growth_rate',
'ts_dmd_return_mode_concentration',
]


def main() -> int:
    load_all()
    panels = _panels()
    template = panels["x"]
    failures: list[str] = []
    for canonical in TARGETS:
        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        if operator is None:
            failures.append(f"{canonical}: no reference")
            continue
        try:
            arguments, kw = _build_call(canonical, operator, panels)
            out = _to_frame(operator.calculate(*arguments, **kw), template)
            assert out.shape == template.shape, f"shape {out.shape}"
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{canonical}: {type(exc).__name__}: {exc}")
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures:
            print("  " + f)
        return 1
    print("ALL %d previously-failing ops now execute cleanly" % len(TARGETS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
