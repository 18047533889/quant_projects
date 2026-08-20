# -*- coding: utf-8 -*-
"""R47 new-operator explicit policy pack.

The R47 taskbook added ~48 new operators (intraday state / event / slice /
profile / chip / limit / indicator / panel families).  Every one of them is a
causal, trailing-only operator (intraday minute→daily EOD operators use only the
completed session + past days; daily indicators use trailing windows).  They
must therefore carry an explicit ``pit_safe=True`` policy — the default
fail-closed rule gives EXTENDED operators without an explicit policy
``pit_safe=False``, which the production audit rejects as "PIT policy is not
causal".

This module is loaded from ``cleaned_operators.__init__`` AFTER the operator
modules; it merges the new canonicals into the shared ``_EXPLICIT_POLICIES``
dict in place, so the concurrent-session edits in ``operator_policy.py`` are
left untouched.  The merge is purely additive and idempotent.
"""
from __future__ import annotations

# Importing operator_policy executes its module body (which builds and filters
# ``_EXPLICIT_POLICIES``); the live dict is then extended below.
from cleaned_operators.operator_policy import _EXPLICIT_POLICIES  # noqa: E402

_R47_POLICIES: dict[str, dict[str, object]] = {
    # ---- intraday state / event / slice / profile (minute -> daily EOD) ----
    "intra_kalman_latent_price": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_space_volume_components": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_functional_motif_score": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_visibility_graph_features": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_count": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_sum": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_vwap": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_interval_moment": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_follow_ratio": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_follow_beta": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_follow_corr": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_pair_same_slot_corr": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_dwell_stats": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_state_transition_entropy": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_neighbor_event_class": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_range_gap_flag": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_event_window_reduce": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_event_pre_post_contrast": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_impulse_event_detector": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_post_impulse_response": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_probe_outcome_score": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_supply_absorption_score": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_consolidation_quality": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_response_curve_features": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_liquidity_resilience_curve_fit": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_slice_mask_reduce": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_slice_mask_pair_reduce": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_multiresolution_resample_reduce": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_same_slot_zscore": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_session_boundary_jump": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_volume_at_price_profile": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_volume_profile_peak_geometry": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_volume_profile_supply_structure": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_volume_profile_value_area": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_round_price_clustering_share": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_round_price_barrier_response": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_limit_pre_hit_pressure_profile": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_eod_reversal_decomposition": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    # ---- R47 pattern recognition operators ----
    "intra_smart_money_fcm_score": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_price_peak_ridge_valley_state": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_volume_peak_ridge_valley_state": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intraday_value_at_extreme_state": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    # ---- daily technical indicators / chip (trailing only) ----
    "HMA": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "QQE": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "RSX": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ALMA": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "CoppockCurve": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ElderRay": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "FisherTransform": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "turnover_chip_age_cost_surface": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "turnover_chip_overhang_surface": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # ---- daily cross-sectional / panel ----
    "panel_async_beta_ex_self": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "panel_factor_pocket_strength": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "cs_predictability_mosaic_score": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    "panel_predictability_mosaic_score": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    # ---- fiscal_batch1 (TRUE_GAP operators 2026-08-13) ----
    "fiscal_acceleration": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 3},
    "fiscal_pct_change": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 2},
    "fiscal_rolling_std": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 3},
    "fiscal_accrual_quality": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 4},
    "fiscal_direction_consistency": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 3},
    # ---- frequency_filter_batch (TRUE_GAP operators 2026-08-13) ----
    "ts_bessel_lowpass_causal": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_fir_lowpass_causal": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_spectral_lowpass_trailing": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_causal_savgol_endpoint": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # ---- panel_batch1 (TRUE_GAP operators 2026-08-13) ----
    "panel_day_night_beta_gap": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "pastor_stambaugh_beta": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "price_delay_score": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "report_asof": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "event_window_return_asof": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # ---- cs_batch1: cross-sectional TRUE_GAP operators (2026-08-13) ----
    "cs_isolation_forest_score": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    "cs_factor_bucket_return": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    "cs_empirical_bayes_shrinkage": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    "cs_shrink_to_group_mean": {"scope": "group", "pit_safe": True, "min_periods": 1},
    "panel_peer_graph_aggregate": {"scope": "cs", "pit_safe": True, "min_periods": 1},
    # ---- frequency-domain / causal filters (trailing only) ----
    "ts_bessel_lowpass_causal": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_fir_lowpass_causal": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_spectral_lowpass_trailing": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_causal_savgol_endpoint": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # ---- Kalman filter variants (2026-08-13 TRUE_GAP batch) ----
    "ts_alpha_beta_filter": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_h_infinity_level_filter": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_adaptive_noise_kalman": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_student_t_kalman_filter": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # ---- Adaptive filters (2026-08-13 TRUE_GAP batch) ----
    "ts_mcginley_dynamic": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_vidya": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_one_euro_filter": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_nlms_filter": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_rls_filter": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # ---- denoise / regularization filters (2026-08-13 TRUE_GAP batch) ----
    "ts_ssa_denoise_trailing": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "ts_wavelet_shrinkage_trailing": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_total_variation_filter_trailing": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "ts_l1_trend_filter_trailing": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    # ---- Intraday topology/manifold operators (2026-08-13 TRUE_GAP batch) ----
    "intra_matrix_profile_session_features": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_dmd_koopman_features": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_covariance_manifold_shift": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    "intra_critical_transition_score": {"scope": "session_intraday", "pit_safe": True, "min_periods": 1, "session_aware": True, "reset_at_session_boundary": True},
    # ---- Smart money / graph intraday (2026-08-13) ----
    "intra_dynamic_stock_graph_features": {"scope": "session_intraday", "pit_safe": True, "min_periods": 5, "session_aware": True, "reset_at_session_boundary": True},
    "intra_common_trading_intensity": {"scope": "session_intraday", "pit_safe": True, "min_periods": 3, "session_aware": True, "reset_at_session_boundary": True},
    "intra_local_conditional_entropy": {"scope": "session_intraday", "pit_safe": True, "min_periods": 10, "session_aware": True, "reset_at_session_boundary": True},
    "intra_smart_money_vwap_ratio": {"scope": "session_intraday", "pit_safe": True, "min_periods": 4, "session_aware": True, "reset_at_session_boundary": True},
    # ---- TRUE_GAP time-semantic operators (2026-08-13) ----
    "financial_snapshot_lag": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "same_calendar_day_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "same_calendar_month_return": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    # ---- fiscal_batch3: capital stock / lifecycle / efficiency (2026-08-13) ----
    "fiscal_capital_stock": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 8},
    "fiscal_perpetual_inventory": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 8},
    "cash_flow_lifecycle_stage": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 3},
    "laborforce_efficiency": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 3},
    "years_since_date": {"scope": "fundamental_period", "pit_safe": True, "min_periods": 1},
}

_EXPLICIT_POLICIES.update(_R47_POLICIES)
