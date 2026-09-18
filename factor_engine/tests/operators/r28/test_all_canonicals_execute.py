# -*- coding: utf-8 -*-
"""R28 §七十二..七十七 / P0-004: every retained canonical has a REAL execution
test.  Each canonical is invoked on a valid synthetic fixture via the shared
audit fixture machinery plus the R28 domain-override layer, and must return a
same-shaped DataFrame with a finite non-NaN value in the observable region, no
exception and no inf.

Gate: R28_EVERY_RETAINED_CANONICAL_HAS_REAL_TEST.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

# The shared production-audit fixture builder lives in scripts/; load it by file
# path so collection works under every pytest import mode.
import importlib.util

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_audit_spec = importlib.util.spec_from_file_location(
    "r28_audit_helpers", _REPO_ROOT / "scripts" / "audit_all_factor_production.py"
)
audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit)


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()
    return True


# ---------------------------------------------------------------------------
# R28 fixture-domain override layer (layered on top of the shared audit fixtures)
# ---------------------------------------------------------------------------
#: Return-family canonical prefixes: their generic ``x``/``y`` panel must be the
#: *returns* series (not the price level) so GARCH/HAR/AR/transfer-entropy/DMD
#: see a stationary input and the price-vs-return DQ gate is not tripped.
_RETURN_FAMILY_PREFIXES = (
    "ts_garch_", "ts_gjr_", "ts_har_", "ts_ar_", "ts_lo_mackinlay_",
    "ts_transfer_entropy", "ts_effective_transfer_entropy",
    "ts_conditional_transfer_entropy", "ts_best_lag_corr", "ts_dmd_",
    "ts_hankel_", "ts_ssa_", "ts_path_", "ts_dfa_hurst", "ts_regression_",
    "ts_quantile_regression", "ts_multi_regression", "ts_ridge_regression",
    "ts_poly2_", "ts_lz_complexity", "ts_multiscale_entropy",
    "ts_spectral_", "ts_return_spectral_entropy", "cs_knn_", "cs_multi_",
    "cs_neutralize", "cs_regression", "cs_huber", "cs_lad", "cs_wls",
    "cs_isotonic", "cs_spline", "cs_trimmed", "cs_ridge", "cs_rank_copula",
    "cs_shrinkage", "cs_robust", "cs_local", "cs_mahalanobis", "cs_isolation",
    "intra_market_model_", "fiscal_regression_", "fiscal_ar_", "ts_kalman_",
    "ts_signature_mahalanobis", "ts_recurrence_", "ts_rqa_",
)
#: Domain-trig canonicals needing inputs in [-1, 1].
_TRIG_DOMAIN = {
    "acos", "acos_bounded", "asin", "asin_bounded",
}


#: Canonicals whose output shape is legitimately NOT the input-panel shape
#: (long-format / multi-row-per-instrument outputs).  The shape assertion is
#: skipped for these; only finite-value presence is checked.
_LONG_FORMAT_CANONICALS: set[str] = set()


def _apply_overrides():
    # return-family: map the generic panel param names to the returns panel
    for canon in sorted(OperatorRegistry.list_canonical()):
        if not canon.startswith(_RETURN_FAMILY_PREFIXES):
            continue
        for key in ("x", "y", "a", "b", "left", "right", "numerator", "denominator"):
            audit._PANEL_SPECIAL[(canon, key)] = "returns"
    # trig domain: map to a bounded (-1,1) panel
    for canon in _TRIG_DOMAIN:
        for key in ("x", "a", "left", "numerator"):
            audit._PANEL_SPECIAL[(canon, key)] = "unit"
    # per-canonical scalar/panel disambiguation (params whose generic name is
    # ALSO a panel column but are scalars here — or vice-versa)
    audit._SPECIAL_SCALARS.update(
        {
            ("power", "y"): 2.0,          # pow(x, 2) exponent is a scalar
            ("pow", "y"): 2.0,
            ("ts_ar_fitted_value", "order"): 1,
            ("ts_ar_in_sample_resid", "order"): 1,
            ("ts_dfa_hurst", "window"): 220,     # window must exceed max_scale*4
            ("ts_dfa_hurst", "max_scale"): 32,
            ("ts_dfa_hurst", "min_scale"): 4,
            ("panel_mixture_of_experts_score", "n_experts"): 2,
            ("panel_mixture_of_experts_score", "window"): 120,
            ("panel_regime_conditioned_forecast", "n_regimes"): 3,
            ("panel_regime_conditioned_forecast", "window"): 120,
            ("erd_sign_consistent_decay", "direction"): 1,
            ("fillna", "max_ffill_gap"): 5,
            ("fillna", "method"): "zero",
            ("hfl_hurst_ratio", "max_agg"): 10,
            ("hfl_hurst_ratio", "window"): 120,
            ("hfl_variance_ratio", "k"): 5,
            ("hfl_variance_ratio", "window"): 120,
            ("ex_self_mad_z", "min_peers"): 3,
            ("ex_self_mean_gap", "min_peers"): 3,
            ("ex_self_rank_pct", "min_peers"): 3,
            ("ex_self_zscore", "min_peers"): 3,
        }
    )
    audit._SCALAR_VALUES["min_breadth"] = 4
    audit._SCALAR_VALUES["max_lags"] = 3
    audit._SCALAR_VALUES["min_exceed"] = 5
    audit._SCALAR_VALUES["min_warmup"] = 5
    audit._SCALAR_VALUES["min_bars"] = 12
    audit._SPECIAL_SCALARS[("event_interval_mark_coupling", "max_boundary_extension")] = 5
    audit._SPECIAL_SCALARS[("intra_segment_amount_share", "market")] = "ashare"
    audit._SPECIAL_SCALARS[("intra_segment_volume_share", "market")] = "ashare"
    audit._SPECIAL_SCALARS[("intraday_value_at_extreme_state", "quantile")] = 0.90
    audit._SPECIAL_SCALARS[("intra_amihud", "scale")] = 1.0e8
    audit._SPECIAL_SCALARS[("ts_conditional_transfer_entropy", "window")] = 120
    audit._SPECIAL_SCALARS[("ts_conditional_transfer_entropy", "bins")] = 2
    audit._SPECIAL_SCALARS[("ts_effective_transfer_entropy", "window")] = 120
    audit._SPECIAL_SCALARS[("ts_effective_transfer_entropy", "bins")] = 3
    audit._SPECIAL_SCALARS[("ts_extrema_confirmation_rate", "tolerance")] = 2
    audit._SPECIAL_POSITIONAL["ts_extreme_cluster_ratio"] = ("x", "window")
    audit._SPECIAL_KWARGS["ts_extreme_cluster_ratio"] = {
        "threshold": 0.0,
        "side": "absolute",
        "min_periods": 20,
    }
    audit._SPECIAL_SCALARS[("ts_multifractal_asymmetry", "min_periods")] = 40
    audit._SPECIAL_SCALARS[("ts_multiscale_permutation_entropy_slope", "window")] = 200
    audit._SPECIAL_SCALARS[("ts_multiscale_permutation_entropy_slope", "order")] = 2
    audit._SPECIAL_SCALARS[("ts_signature_mahalanobis_anomaly", "path_window")] = 12
    audit._SPECIAL_SCALARS[("ts_signature_mahalanobis_anomaly", "history_window")] = 120
    audit._SPECIAL_SCALARS[("valuation_growth_mismatch", "scale")] = 1.0
    audit._SPECIAL_SCALARS[("ts_matrix_profile_discord_score", "m")] = 3
    audit._SPECIAL_SCALARS[("ts_motif_recurrence_count", "m")] = 3
    audit._SPECIAL_SCALARS[("open_close_return", "price_basis")] = "RAW"
    audit._SPECIAL_SCALARS[("open_to_vwap_return", "price_basis")] = "RAW"
    audit._SPECIAL_SCALARS[("overnight_return", "price_basis")] = "RAW"
    audit._SPECIAL_SCALARS[("vwap_to_close_return", "price_basis")] = "RAW"
    audit._SPECIAL_SCALARS[("panel_day_night_beta_gap", "day_start")] = 9
    audit._SPECIAL_SCALARS[("panel_day_night_beta_gap", "day_end")] = 15
    audit._SPECIAL_SCALARS[("panel_day_night_beta_gap", "night_start")] = 16
    audit._SPECIAL_SCALARS[("panel_day_night_beta_gap", "night_end")] = 23
    audit._SPECIAL_SCALARS[("panel_factor_pocket_strength", "threshold")] = 0.5
    audit._SPECIAL_SCALARS[("panel_peer_graph_aggregate", "method")] = "mean"
    audit._SPECIAL_SCALARS[("ts_activity_spectral_entropy", "input_kind")] = "NonNegativeActivity"
    audit._SPECIAL_POSITIONAL["ts_activity_spectral_entropy"] = ("x", "window")
    audit._SPECIAL_KWARGS["ts_activity_spectral_entropy"] = {"input_kind": "NonNegativeActivity"}
    audit._SPECIAL_POSITIONAL["ts_detrended_level_spectral_entropy"] = ("x", "window")
    audit._SPECIAL_SCALARS[("ts_kalman_beta", "scale_mode")] = "absolute"
    for canon in ("ts_multiscale_trend_consensus", "ts_multiscale_trend_curvature", "ts_multiscale_trend_dispersion"):
        audit._SPECIAL_SCALARS[(canon, "window")] = 60
        audit._PANEL_SPECIAL[(canon, "x")] = "stat_clean"
        audit._SPECIAL_SCALARS[(canon, "scales")] = (5, 10, 20, 40)
    audit._SPECIAL_SCALARS[("ts_mutual_information_nats", "min_periods")] = 10
    audit._SPECIAL_SCALARS[("ts_normalized_mutual_information", "min_periods")] = 10
    audit._SPECIAL_SCALARS[("ts_nlms_filter", "order")] = 3
    audit._SPECIAL_SCALARS[("ts_kama", "min_periods")] = 11
    audit._SPECIAL_SCALARS[("ts_quantile_kurtosis", "min_periods")] = 8
    audit._SPECIAL_POSITIONAL["ts_return_spectral_entropy"] = ("x", "window")
    audit._SPECIAL_KWARGS["ts_return_spectral_entropy"] = {"input_kind": "ReturnDecimal"}
    audit._SPECIAL_SCALARS[("ts_ridge_regression_coeff", "window")] = 60
    audit._SPECIAL_SCALARS[("ts_ridge_regression_coeff_prior", "window")] = 60
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("ts_ridge_regression"):
            audit._SPECIAL_SCALARS[(canon, "window")] = 60
    audit._SPECIAL_SCALARS[("ts_rls_filter", "order")] = 3
    audit._SPECIAL_POSITIONAL["ts_spectral_entropy"] = ("x", "window")
    audit._SPECIAL_KWARGS["ts_spectral_entropy"] = {"input_kind": "ReturnDecimal"}
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("ts_wavelet_"):
            audit._SPECIAL_SCALARS[(canon, "window")] = 64
    audit._SPECIAL_SCALARS[("ts_kalman_beta_change", "scale_mode")] = "absolute"
    audit._SPECIAL_KWARGS["ts_detrended_level_spectral_entropy"] = {"input_kind": "PriceContinuous"}
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("ts_kalman_"):
            audit._SPECIAL_SCALARS[(canon, "scale_mode")] = "absolute"
    audit._PANEL_SPECIAL.update(
        {
            ("ts_har_rv_next_vol_forecast", "rv"): "returns",
            ("ts_har_rv_next_var_forecast", "rv"): "returns",
            ("ts_har_rv_forecast", "rv"): "returns",
            ("ts_har_rv_forecast_error_z", "rv"): "returns",
            ("ts_har_rv_innovation_z", "rv"): "returns",
            ("accounting_comparability_score", "scaled_earnings"): "wide_scaled_earnings",
            ("accounting_comparability_score", "report_return"): "wide_returns",
            ("accounting_comparability_score", "industry"): "industry",
            ("accounting_comparability_score", "period_id"): "period_id",
            ("fiscal_asymmetric_timeliness", "scaled_earnings"): "scaled_earnings",
            ("fiscal_asymmetric_timeliness", "report_return"): "returns",
            ("fiscal_asymmetric_timeliness", "period_id"): "period_id",
            ("cs_empirical_bayes_shrinkage", "estimate"): "wide_returns",
            ("cs_empirical_bayes_shrinkage", "std_err"): "wide_std_err",
            ("cs_shrinkage_mahalanobis", "f1"): "wide_returns",
            ("cs_shrinkage_mahalanobis", "f2"): "wide_volume",
            ("cs_shrinkage_mahalanobis", "f3"): "wide_amount",
            ("cs_shrinkage_mahalanobis", "f4"): "wide_scaled_earnings",
            ("cs_shrinkage_mahalanobis", "f5"): "wide_std_err",
            ("cs_neutralize", "target"): "wide_returns",
            ("cs_neutralize", "exposure"): "wide_amount",
            ("cs_neutralize", "control"): "wide_volume",
            ("hfl_hurst_ratio", "ret"): "returns",
            ("hfl_variance_ratio", "ret"): "returns",
            ("index_reconstitution_churn", "member"): "condition",
            ("index_event_decay", "entry_event"): "signed_event",
            ("laborforce_efficiency", "employees"): "volume",
            ("lf1_liquidity_decay_base", "shock_mark"): "returns",
            ("multi_index_entry_intensity", "entry_index_a"): "condition",
            ("multi_index_entry_intensity", "entry_index_b"): "condition",
            ("multi_index_entry_intensity", "entry_index_c"): "condition",
            ("panel_day_night_beta_gap", "benchmark_ret"): "market",
            ("panel_peer_graph_aggregate", "similarity"): "similarity",
            ("pastor_stambaugh_beta", "liquidity_proxy"): "returns",
            ("ts_activity_spectral_entropy", "x"): "volume",
            ("ts_detrended_level_spectral_entropy", "x"): "close",
            ("ts_deviation_from_mean", "feature"): "returns",
            ("ts_expanding_chi_square_pvalue", "x"): "stat_clean",
            ("ts_expanding_chi_square_pvalue", "y"): "stat_clean",
            ("ts_expanding_durbin_watson_statistic", "residuals"): "stat_clean",
            ("ts_return_spectral_entropy", "x"): "returns",
            ("ts_expanding_lilliefors_pvalue", "x"): "stat_clean",
            ("ts_jump_bipower", "feature"): "returns",
            ("ts_lag1_autocorr", "feature"): "returns",
            ("ts_multifractal_asymmetry", "x"): "multifractal_level",
            ("fiscal_logit_score", "x"): "fiscal_unit",
            ("ts_roll_effective_spread", "price"): "roll_price",
            ("ts_spectral_entropy", "x"): "returns",
        }
    )
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith(("ofi_", "sv_")):
            audit._PANEL_SPECIAL[(canon, "signed_volume")] = "returns"
    audit._PANEL_SPECIAL[("vr1_range_everage", "close_return")] = "returns"
    audit._PANEL_SPECIAL[("years_since_date", "date_panel")] = "period_id"
    audit._PANEL_SPECIAL[("tod_overnight_activity_ratio", "overnight_mark")] = "condition"
    audit._PANEL_SPECIAL[("years_since_date", "reference_date")] = "period_id"
    audit._PANEL_SPECIAL[("tod_overnight_activity_ratio", "intraday_mark")] = "condition"
    audit._PANEL_SPECIAL[("sv_signed_beta_market", "market_signed_volume")] = "market"
    # HAR window must be >= 30 (OLS design); the generic fallback of 20 is
    # infeasible by the operator's own feasibility contract.
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("ts_har_"):
            audit._SPECIAL_SCALARS[(canon, "window")] = 60
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("val1_"):
            for key in ("value_metric", "own_metric", "market_metric", "reference_metric", "fcf_yield", "own_pe", "market_pe", "roa", "earnings_stability", "earnings_yield"):
                audit._PANEL_SPECIAL[(canon, key)] = "returns"
    for canon in ("vax_liquidity_adjusted_return", "vax_liquidity_penalty_exposure", "vax_ret_per_liquidity_unit"):
        audit._PANEL_SPECIAL[(canon, "amihud")] = "volume"
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("ts_multi_regression"):
            audit._SPECIAL_SCALARS[(canon, "window")] = 60
    # cs_knn family requires k >= 20 nearest neighbours -> needs a WIDE universe;
    # f1/f2/f3 must be DISTINCT features or the neighbour cloud is rank-1.
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("cs_knn_"):
            audit._SPECIAL_SCALARS[(canon, "k")] = 20
            audit._PANEL_SPECIAL[(canon, "f1")] = "wide_returns"
            audit._PANEL_SPECIAL[(canon, "f2")] = "wide_volume"
            audit._PANEL_SPECIAL[(canon, "f3")] = "wide_amount"
            audit._PANEL_SPECIAL[(canon, "x")] = "wide_returns"
            audit._PANEL_SPECIAL[(canon, "y")] = "wide_returns"
    # multi-feature models (x1..x4, market_state): feed DISTINCT panels so the
    # design is not singular (all same panel -> collinear -> correct fail-closed NaN).
    for canon in sorted(OperatorRegistry.list_canonical()):
        if (
            canon.startswith("panel_")
            or canon.startswith("ts_multi_regression")
            or canon.startswith("intra_market_model")
        ):
            audit._PANEL_SPECIAL[(canon, "x1")] = "returns"
            audit._PANEL_SPECIAL[(canon, "x2")] = "volume"
            audit._PANEL_SPECIAL[(canon, "x3")] = "amount"
            audit._PANEL_SPECIAL[(canon, "x4")] = "condition"
            audit._PANEL_SPECIAL[(canon, "market_state")] = "market"
    # A-share trading-state flags are typed booleans ({0,1,NaN}); map them to
    # the boolean ``condition`` panel so TradableBool/EventBool contracts hold.
    for canon in sorted(OperatorRegistry.list_canonical()):
        if canon.startswith("ashare_") or canon.startswith("a_share_"):
            for key in ("valid_trade", "is_suspend", "tradable_state", "limit_up", "limit_down"):
                audit._PANEL_SPECIAL[(canon, key)] = "condition"
    # long-format outputs (multi-row per instrument) are a legitimate different
    # shape; record them as opaque-return without a shape assertion.
    _LONG_FORMAT_CANONICALS.add("baseline_scaled_wasserstein_distance")
    # tail_beta needs an explicit benchmark series (not a self-inclusion beta).
    audit._PANEL_SPECIAL[("tail_beta", "benchmark_ret")] = "market"
    audit._PANEL_SPECIAL[("tail_beta", "y")] = "returns"
    audit._PANEL_SPECIAL[("tail_beta", "x")] = "returns"
    # per-canonical ``q`` disambiguation — the generic fallback has no ``q``;
    # each family declares the quantile/lag it means.
    audit._SPECIAL_SCALARS[("ts_lo_mackinlay_vr", "q")] = 2  # variance-ratio lag
    audit._SPECIAL_SCALARS[("ts_lo_mackinlay_z", "q")] = 2
    audit._SPECIAL_SCALARS[("ts_quantile", "q")] = 0.5
    audit._SPECIAL_SCALARS[("tail_beta", "q")] = 0.05  # tail quantile in (0, 1)
    audit._SPECIAL_SCALARS[("tail_beta", "window")] = 120  # enough tail observations
    audit._SPECIAL_SCALARS[("ts_expected_shortfall", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_expected_shortfall_asymmetry", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_tail_mean", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_extremal_index", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_extreme_cluster_ratio", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_weighted_expected_shortfall", "q")] = 0.05
    audit._SPECIAL_SCALARS[("ts_expected_shortfall", "q")] = 0.10
    audit._SPECIAL_SCALARS[("ts_expected_shortfall", "window")] = 200
    audit._SPECIAL_SCALARS[("ts_expected_shortfall", "min_tail_count")] = 5
    audit._SPECIAL_POSITIONAL["cs_neutralize"] = ("target", "exposure")
    audit._SPECIAL_KWARGS["cs_neutralize"] = {"add_intercept": True, "min_obs": 3}
    audit._SPECIAL_SCALARS[("ts_dc_event_rate", "threshold")] = 0.01
    for _c in (
        "pattern_head_shoulders", "pattern_inverse_head_shoulders",
        "pattern_triple_bottom", "pattern_triple_top",
    ):
        audit._SPECIAL_SCALARS[(_c, "left_window")] = 2
        audit._SPECIAL_SCALARS[(_c, "right_window")] = 2
        audit._SPECIAL_SCALARS[(_c, "history_window")] = 80
    for _c in ("pattern_head_shoulders", "pattern_inverse_head_shoulders"):
        audit._SPECIAL_SCALARS[(_c, "shoulder_tolerance")] = 0.50
        audit._SPECIAL_SCALARS[(_c, "head_min_prominence")] = 0.0
        audit._SPECIAL_SCALARS[(_c, "max_neckline_slope")] = 10.0
    for _c in ("pattern_triple_bottom", "pattern_triple_top"):
        audit._SPECIAL_SCALARS[(_c, "tolerance")] = 0.50
        audit._SPECIAL_SCALARS[(_c, "min_depth")] = 0.0
        audit._SPECIAL_SCALARS[(_c, "min_spacing")] = 1
        audit._SPECIAL_SCALARS[(_c, "max_spacing")] = 80
    audit._SPECIAL_SCALARS[("relation_jaccard", "periods")] = 1
    audit._SPECIAL_SCALARS[("relation_weighted_std_ex_self", "min_peers")] = 2
    for _c in ("ts_dc_duration_asymmetry", "ts_dc_overshoot_asymmetry", "ts_dc_overshoot_ratio"):
        audit._SPECIAL_SCALARS[(_c, "threshold")] = 1.0
        audit._SPECIAL_SCALARS[(_c, "window")] = 80
    for _c in sorted(OperatorRegistry.list_canonical()):
        if _c.startswith("ts_turnover_"):
            audit._SPECIAL_SCALARS[(_c, "window")] = 20
        if _c.startswith("ts_huber_regression"):
            audit._SPECIAL_SCALARS[(_c, "window")] = 60
            audit._PANEL_SPECIAL[(_c, "x")] = "returns"
            audit._PANEL_SPECIAL[(_c, "y")] = "amount"
    for _c in (
        "ts_nth_pivot_high", "ts_nth_pivot_high_age", "ts_nth_pivot_low",
        "ts_nth_pivot_low_age", "ts_resistance_fit_r2", "ts_support_fit_r2",
    ):
        audit._SPECIAL_SCALARS[(_c, "left_window")] = 2
        audit._SPECIAL_SCALARS[(_c, "right_window")] = 2
        audit._SPECIAL_SCALARS[(_c, "history_window")] = 80
        audit._SPECIAL_SCALARS[(_c, "n")] = 2
        audit._SPECIAL_SCALARS[(_c, "points")] = 3
    for _c in ("report_filing_delay_surprise", "report_revision_magnitude"):
        audit._SPECIAL_SCALARS[(_c, "window")] = 4
        audit._SPECIAL_SCALARS[(_c, "min_periods")] = 3
    audit._SPECIAL_SCALARS[("revision_delta", "mode")] = "absolute"
    audit._SPECIAL_SCALARS[("ts_evt_threshold_stability", "k_min")] = 2
    audit._SPECIAL_SCALARS[("ts_evt_threshold_stability", "k_max")] = 4
    audit._SPECIAL_SCALARS[("ts_evt_threshold_stability", "window")] = 120
    audit._SPECIAL_SCALARS[("ts_variance_ratio_slope", "max_q")] = 10
    audit._SPECIAL_SCALARS[("ts_variance_ratio_slope", "min_lag")] = 2
    # expectile family: q in (0,1)
    for _c in ("ts_expectile", "ts_expectile_beta", "ts_expectile_regression_coeff",
               "ts_expectile_regression_coeff_prior", "ts_expectile_regression_forecast_error",
               "ts_expectile_regression_resid"):
        audit._SPECIAL_SCALARS[(_c, "q")] = 0.5
    # directional-change family needs a real threshold
    for _c in ("directional_change_extent", "directional_change_state",
               "ts_dc_duration_asymmetry", "ts_dc_overshoot_asymmetry",
               "ts_dc_overshoot_ratio", "ts_dc_overshoot_count"):
        audit._SPECIAL_SCALARS[(_c, "threshold")] = 0.01
    # tail / expectile / EVT / envelope estimators need a large-enough window for
    # the extreme tail to be observable (window=20 -> all-NaN).
    for _c in sorted(OperatorRegistry.list_canonical()):
        if any(k in _c for k in (
            "ts_expectile", "ts_expected_shortfall", "ts_tail_", "ts_extremal",
            "ts_extreme_", "ts_evt_", "ts_envelope", "ts_pickands",
            "ts_upper_tail", "ts_weighted_expected_shortfall", "ts_quantile_",
            "ts_conditional_mutual_information", "ts_active_information_storage",
            "ts_first_passage", "ts_delay_intrinsic_dimension",
            "Supertrend", "cs_rank_copula",
        )):
            audit._SPECIAL_SCALARS[(_c, "window")] = 120
    # fractal / tail / multifractal / pivot / roll-spread estimators need a long
    # window for the structure to be observable.
    for _c in sorted(OperatorRegistry.list_canonical()):
        if any(k in _c for k in (
            "ts_gpd", "ts_hill_tail", "ts_mean_excess", "ts_lower_tail",
            "ts_upper_tail", "ts_higuchi", "ts_hurst", "ts_multifractal",
            "ts_nth_pivot", "ts_price_delay", "ts_resistance_fit",
            "ts_roll_effective_spread", "ts_huber_regression", "ts_generalized_hurst",
        )):
            audit._SPECIAL_SCALARS[(_c, "window")] = 200
    # EVT tail estimators: k_min/k_max define the tail-observation band.
    for _c in ("ts_evt_threshold_stability", "ts_pickands_tail_index",
               "ts_upper_tail_coexceedance_probability"):
        audit._SPECIAL_SCALARS[(_c, "k_min")] = 10
        audit._SPECIAL_SCALARS[(_c, "k_max")] = 60
        audit._SPECIAL_SCALARS[(_c, "window")] = 200
    # scalar params the shared audit fixture does not know yet
    audit._SCALAR_VALUES.update(
        {
            "coverage_threshold": 0.5,
            "transition_prob": 0.5,
            "label_horizon": 1,
            "min_conditioning_events": 2,
            "min_events": 3,
            "min_scale": 8,
            "max_scale": 64,
            "scale": "mad",
            "center": "median",
            "top_k": 2,
            "k": 8,
            "n_neighbors": 8,
            "min_cluster": 3,
            "threshold": 0.0,
        }
    )
    # Final precedence: the broad tail loop above intentionally uses production-
    # sized defaults, but this canonical's legal compact oracle is an upper-tail
    # Hill band with at least three estimates (k=2..4).
    audit._SPECIAL_SCALARS[("ts_evt_threshold_stability", "side")] = "upper"
    audit._SPECIAL_SCALARS[("ts_evt_threshold_stability", "k_min")] = 2
    audit._SPECIAL_SCALARS[("ts_evt_threshold_stability", "k_max")] = 4
    audit._SPECIAL_SCALARS[("ts_evt_threshold_stability", "window")] = 10
    audit._SPECIAL_SCALARS[("ts_state_age_percentile", "min_completed_runs")] = 5
    for _c in (
        "ts_huber_regression_coeff", "ts_huber_regression_coeff_prior",
        "ts_huber_regression_forecast_error", "ts_huber_regression_forecast_error_z",
        "ts_huber_regression_resid_z",
    ):
        audit._SPECIAL_SCALARS[(_c, "window")] = 60
        audit._SPECIAL_SCALARS[(_c, "coefficient_index")] = 1
        audit._PANEL_SPECIAL[(_c, "y")] = "returns"
        audit._PANEL_SPECIAL[(_c, "x1")] = "volume"
        audit._PANEL_SPECIAL[(_c, "x2")] = "amount"
        audit._PANEL_SPECIAL[(_c, "x3")] = "condition"
        audit._PANEL_SPECIAL[(_c, "x4")] = "market"

    # === R28 fixture repairs (this agent's batch) =================================
    # These are LEGAL-sample / support-threshold fixes only: no source operator is
    # touched and no formal statistical gate is relaxed.  Each canonical below was
    # emitting all-NaN on the handoff fixture because its synthetic inputs were
    # degenerate for that particular estimator, not because the operator is broken.
    # ts_expected_shortfall_asymmetry: at q=0.05 the per-window tail holds only
    # ~6 obs at window=120 (< the min_tail_count default 10) -> all-NaN.  Lengthen
    # the window so the tail band is observable and use a feasible tail floor.
    audit._SPECIAL_SCALARS[("ts_expected_shortfall_asymmetry", "window")] = 240
    audit._SPECIAL_SCALARS[("ts_expected_shortfall_asymmetry", "min_tail_count")] = 5
    # ts_state_exit_hazard / ts_state_residual_life: the shared survival kernel
    # needs >= min_completed_runs COMPLETED episodes (each separated by an inactive
    # row) inside history_window.  The default 20 cannot fit in 220 rows switching
    # every 7; lower to a feasible support count (the fixture is re-shaped in
    # _panels_for to switch every 5 rows so >=5 completed runs land in-window).
    audit._SPECIAL_SCALARS[("ts_state_exit_hazard", "min_completed_runs")] = 5
    audit._SPECIAL_SCALARS[("ts_state_residual_life", "min_completed_runs")] = 5
    # ts_extrema_divergence_strength: a near-smooth close/open pair yields no
    # confirmed extrema.  Two clearly-oscillating, phase-aligned price panels are
    # supplied in _panels_for; keep prominence small so the pivot ledger fires and
    # match_lag modest so same-side y extrema are found.
    audit._SPECIAL_SCALARS[("ts_extrema_divergence_strength", "prominence")] = 0.01
    audit._SPECIAL_SCALARS[("ts_extrema_divergence_strength", "match_lag")] = 2
    audit._SPECIAL_SCALARS[("ts_extrema_divergence_strength", "confirmation")] = 3
    # ts_first_passage_conditional_time: barrier*scale (~price level) is never
    # touched -> all-NaN.  _panels_for supplies a random-walk x and a small
    # constant volatility so barrier=1.0 is actually crossed within horizon.
    # ts_threshold_cycle_*: price (~50) never crosses the +/-2 band -> no cycles.
    # _panels_for supplies an oscillating series in [-3, 3] that crosses both
    # thresholds; the default lower/upper (-2/+2) and window (20) are kept.
    # ts_transition_intensity: normalize=True over a near-smooth price level gives
    # a degenerate MAD (all |dx| equal) -> NaN.  Feed the returns panel so the
    # increments vary and the MAD is non-degenerate.
    audit._PANEL_SPECIAL[("ts_transition_intensity", "x")] = "returns"

load_all()  # registry must be fully loaded before the override loop sees canonicals
_apply_overrides()


def _unit_panel(rows=220, columns=6):
    return audit._panels()["returns"].clip(-0.99, 0.99)


def _wide_panels():
    """A wide-universe (30-asset) panel set for cross-sectional models with a
    hard neighbour-count floor (cs_knn k>=20).  Features are kept DISTINCT so the
    neighbour feature cloud is not rank-1 (which would fail the tangent-space
    rank gate and emit all-NaN)."""
    base = audit._panels(rows=220, columns=30)
    returns = base["ret"] if "ret" in base else base["returns"]
    out = dict(base)
    out["wide_returns"] = returns
    out["wide_volume"] = base["volume"] / 1e6  # scale to return-like magnitude
    out["wide_amount"] = base["amount"] / base["amount"].to_numpy().max()
    out["wide_scaled_earnings"] = 0.2 * returns + 0.03 * out["wide_amount"]
    out["wide_std_err"] = 0.02 + 0.01 * out["wide_volume"].abs()
    report_number = np.arange(len(returns)) // 10
    period_labels = np.array(
        [f"{2020 + int(n) // 4}Q{int(n) % 4 + 1}" for n in report_number],
        dtype=object,
    )
    out["period_id"] = pd.DataFrame(
        np.repeat(period_labels[:, None], returns.shape[1], axis=1),
        index=returns.index, columns=returns.columns,
    )
    return out


#: audit._panels() is deterministic and costs ~3 s per build — build ONCE per
#: module and shallow-copy per test (the operator kernels read, never mutate).
_MODULE_PANELS = audit._panels()
_MODULE_UNIT = _MODULE_PANELS["returns"].clip(-0.99, 0.99)
_MODULE_WIDE = _wide_panels()
_MODULE_INTRADAY = audit._panels(rows=48, columns=6)


def _panels_for(canonical: str) -> dict:
    # cross-sectional statistics (cs_knn AND the wider cs_* residual/lof/dip
    # family) need enough instruments for the per-row cross-section to be defined.
    if canonical.startswith("cs_") or canonical in {"accounting_comparability_score", "laborforce_efficiency"}:
        return {k: v.copy() for k, v in _MODULE_WIDE.items()}
    if canonical.startswith(("intra_", "intraday_", "minute_", "micro_", "session_")):
        panels = {k: v.copy() for k, v in _MODULE_INTRADAY.items()}
        panels["unit"] = panels["returns"].clip(-0.99, 0.99)
        return panels
    panels = {k: v.copy() for k, v in _MODULE_PANELS.items()}
    panels["unit"] = _MODULE_UNIT.copy()
    if canonical in {"ts_expanding_chi_square_pvalue", "ts_expanding_durbin_watson_statistic", "ts_expanding_lilliefors_pvalue", "ts_multiscale_trend_consensus", "ts_multiscale_trend_curvature", "ts_multiscale_trend_dispersion"}:
        rows, cols = panels["returns"].shape
        grid = np.arange(rows, dtype=float)[:, None] + np.arange(cols, dtype=float)[None, :]
        panels["stat_clean"] = pd.DataFrame(2.0 + np.sin(grid / 7.0) ** 2, index=panels["returns"].index, columns=panels["returns"].columns)
    panels["scaled_earnings"] = 0.2 * panels["returns"] + 0.001 * panels["volume"] / 1e6
    if canonical.startswith(("fiscal_", "period_", "report_", "revision_", "ttm_", "yoy_")) or canonical in {
        "piotroski_f_score", "piotroski_f_score_tolerant"
    }:
        report_number = np.arange(len(panels["returns"])) // 5
        fiscal_labels = np.array(
            [f"{2010 + int(n) // 4}Q{int(n) % 4 + 1}" for n in report_number],
            dtype=object,
        )
        panels["period_id"] = pd.DataFrame(
            np.repeat(fiscal_labels[:, None], panels["returns"].shape[1], axis=1),
            index=panels["returns"].index,
            columns=panels["returns"].columns,
        )
    if canonical == "ts_multifractal_asymmetry":
        rows, cols = panels["returns"].shape
        rng = np.random.default_rng(20260918)
        increments = rng.lognormal(mean=0.0, sigma=0.4, size=(rows, cols))
        panels["multifractal_level"] = pd.DataFrame(
            np.cumsum(increments, axis=0),
            index=panels["returns"].index,
            columns=panels["returns"].columns,
        )
    if canonical == "fiscal_logit_score":
        rows, cols = panels["returns"].shape
        report = (np.arange(rows) // 5)[:, None]
        asset = np.arange(cols)[None, :]
        bounded = 0.15 + 0.7 / (1.0 + np.exp(-(report - 12.0 - asset) / 5.0))
        panels["fiscal_unit"] = pd.DataFrame(
            bounded, index=panels["returns"].index, columns=panels["returns"].columns
        )
    if canonical in {
        "pattern_head_shoulders", "pattern_inverse_head_shoulders",
        "pattern_triple_bottom", "pattern_triple_top",
        "ts_nth_pivot_high", "ts_nth_pivot_high_age", "ts_nth_pivot_low",
        "ts_nth_pivot_low_age", "ts_resistance_fit_r2", "ts_support_fit_r2",
    }:
        rows, cols = panels["close"].shape
        step = np.arange(rows, dtype=float)[:, None]
        asset = np.arange(cols, dtype=float)[None, :]
        centre = 100.0 + 0.01 * step + 0.02 * asset
        wave = (4.0 + 0.8 * np.sin(step / 29.0)) * np.sin(2.0 * np.pi * step / 12.0)
        panels["pattern_high"] = pd.DataFrame(
            centre + wave + 0.5, index=panels["close"].index, columns=panels["close"].columns
        )
        panels["pattern_low"] = pd.DataFrame(
            centre + wave - 0.5, index=panels["close"].index, columns=panels["close"].columns
        )
        audit._PANEL_SPECIAL[(canonical, "high")] = "pattern_high"
        audit._PANEL_SPECIAL[(canonical, "low")] = "pattern_low"
    if canonical.startswith("recipe_micro_"):
        panels = {k: v.copy() for k, v in _MODULE_INTRADAY.items()}
        panels["unit"] = panels["returns"].clip(-0.99, 0.99)
        minute_index = pd.DatetimeIndex(np.concatenate([
            pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=60, freq="min").to_numpy()
            for day in pd.date_range("2024-01-02", periods=4, freq="B")
        ]))
        rows, cols = len(minute_index), panels["close"].shape[1]
        columns = panels["close"].columns
        step = np.arange(rows, dtype=float)[:, None]
        asset = np.arange(cols, dtype=float)[None, :]
        panels["micro_price"] = pd.DataFrame(
            100.0 + 0.002 * step + 0.20 * np.sin(step / 7.0 + asset),
            index=minute_index, columns=columns,
        )
        panels["micro_volume"] = pd.DataFrame(
            1000.0 + 10.0 * asset + 100.0 * (1.0 + np.sin(step / 11.0)),
            index=minute_index, columns=columns,
        )
        audit._PANEL_SPECIAL[(canonical, "close")] = "micro_price"
        audit._PANEL_SPECIAL[(canonical, "volume")] = "micro_volume"
    if canonical == "relation_jaccard":
        rows, cols = panels["close"].shape
        snapshot_no = np.arange(rows) // 4
        snapshots = np.array(
            [f"{2010 + int(n) // 4}Q{int(n) % 4 + 1}" for n in snapshot_no], dtype=object
        )
        entity = np.empty((rows, cols), dtype=object)
        for row in range(rows):
            for col in range(cols):
                entity[row, col] = f"E{(row + col) % 5}"
        panels["relation_entity"] = pd.DataFrame(entity, index=panels["close"].index, columns=panels["close"].columns)
        panels["relation_snapshot"] = pd.DataFrame(
            np.repeat(snapshots[:, None], cols, axis=1),
            index=panels["close"].index, columns=panels["close"].columns,
        )
        audit._PANEL_SPECIAL[(canonical, "entity_id")] = "relation_entity"
        audit._PANEL_SPECIAL[(canonical, "snapshot_id")] = "relation_snapshot"
    if canonical == "relation_weighted_std_ex_self":
        panels["group"].iloc[:, :] = "ALL"
        panels["positive_weight"] = 1.0 + panels["volume"] / panels["volume"].to_numpy().max()
        audit._PANEL_SPECIAL[(canonical, "weight")] = "positive_weight"
    if canonical == "session_event_recovery_score":
        close = panels["close"]
        recovery = pd.DataFrame(100.0, index=close.index, columns=close.columns)
        event = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        positions = pd.Series(np.arange(len(close)), index=close.index)
        for _, day_positions in positions.groupby(close.index.normalize()):
            locs = day_positions.to_numpy()
            for offset, shock in ((30, 2.0), (90, -2.5), (150, 3.0)):
                if offset < len(locs):
                    recovery.iloc[locs[offset], :] += shock
                    event.iloc[locs[offset], :] = 1.0
        panels["session_recovery_price"] = recovery
        panels["session_recovery_event"] = event
        audit._PANEL_SPECIAL[(canonical, "x")] = "session_recovery_price"
        audit._PANEL_SPECIAL[(canonical, "event")] = "session_recovery_event"
    if canonical == "state_hold" or canonical.startswith("state_since_"):
        rows, cols = panels["close"].shape
        reset = np.zeros((rows, cols), dtype=float)
        reset[::47, :] = 1.0
        update = np.zeros((rows, cols), dtype=float)
        update[3::17, :] = 1.0
        panels["episode_reset"] = pd.DataFrame(reset, index=panels["close"].index, columns=panels["close"].columns)
        panels["episode_update"] = pd.DataFrame(update, index=panels["close"].index, columns=panels["close"].columns)
        audit._PANEL_SPECIAL[(canonical, "reset_condition")] = "episode_reset"
        audit._PANEL_SPECIAL[(canonical, "update_condition")] = "episode_update"
    if canonical.startswith("ts_dc_"):
        rows, cols = panels["close"].shape
        step = np.arange(rows, dtype=float)[:, None]
        asset = np.arange(cols, dtype=float)[None, :]
        price = 100.0 + 4.0 * np.sin(2.0 * np.pi * step / 17.0 + asset / 5.0)
        panels["dc_price"] = pd.DataFrame(price, index=panels["close"].index, columns=panels["close"].columns)
        panels["dc_scale"] = pd.DataFrame(1.0, index=panels["close"].index, columns=panels["close"].columns)
        audit._PANEL_SPECIAL[(canonical, "x")] = "dc_price"
        audit._PANEL_SPECIAL[(canonical, "scale")] = "dc_scale"
    if canonical in {"ts_envelope_boundary_dwell", "ts_envelope_compression", "ts_envelope_pressure"}:
        panels["envelope_upper"] = panels["close"] + 2.0
        panels["envelope_lower"] = panels["close"] - 2.0
        audit._PANEL_SPECIAL[(canonical, "x")] = "close"
        audit._PANEL_SPECIAL[(canonical, "upper")] = "envelope_upper"
        audit._PANEL_SPECIAL[(canonical, "lower")] = "envelope_lower"
    if canonical in {"ts_sign_cluster_index", "ts_sign_persistence"}:
        audit._PANEL_SPECIAL[(canonical, "x")] = "returns"
    if canonical.startswith("ts_state_"):
        rows, cols = panels["close"].shape
        state = ((np.arange(rows)[:, None] // 7 + np.arange(cols)[None, :]) % 2).astype(float)
        panels["state_episode"] = pd.DataFrame(state, index=panels["close"].index, columns=panels["close"].columns)
        audit._PANEL_SPECIAL[(canonical, "state")] = "state_episode"
        audit._PANEL_SPECIAL[(canonical, "x")] = "state_episode"
    if canonical == "ts_evt_threshold_stability":
        rows, cols = panels["close"].shape
        rng = np.random.default_rng(20260919)
        tail = np.exp(rng.normal(0.0, 1.0, size=(rows, cols)))
        panels["positive_tail"] = pd.DataFrame(tail, index=panels["close"].index, columns=panels["close"].columns)
        audit._PANEL_SPECIAL[(canonical, "x")] = "positive_tail"
    if canonical.startswith("ts_huber_regression"):
        panels["huber_y"] = 0.4 * panels["returns"] + 0.003 * np.sin(np.arange(len(panels["returns"]))[:, None] / 5.0)
        audit._PANEL_SPECIAL[(canonical, "x")] = "returns"
        audit._PANEL_SPECIAL[(canonical, "y")] = "huber_y"
    if canonical.startswith("ts_turnover_"):
        rows, cols = panels["close"].shape
        step = np.arange(rows, dtype=float)[:, None]
        asset = np.arange(cols, dtype=float)[None, :]
        panels["turnover_price"] = pd.DataFrame(
            100.0 + 0.4 * step + np.sin(step / 3.0 + asset / 7.0),
            index=panels["close"].index, columns=panels["close"].columns,
        )
        panels["turnover_decimal"] = pd.DataFrame(
            0.20, index=panels["close"].index, columns=panels["close"].columns,
        )
        for key in ("price", "close", "x"):
            audit._PANEL_SPECIAL[(canonical, key)] = "turnover_price"
        for key in ("turnover", "turnover_rate"):
            audit._PANEL_SPECIAL[(canonical, key)] = "turnover_decimal"
    if canonical in {"report_filing_delay_surprise", "report_revision_magnitude", "revision_delta"}:
        rows, cols = panels["close"].shape
        columns, index = panels["close"].columns, panels["close"].index
        if canonical == "report_filing_delay_surprise":
            delay_values = np.resize(np.array([10.0, 12.0, 9.0, 15.0, 11.0, 18.0]), rows)
            panels["report_delay"] = pd.DataFrame(
                np.repeat(delay_values[:, None], cols, axis=1), index=index, columns=columns,
            )
            panels["report_event"] = pd.DataFrame(1.0, index=index, columns=columns)
            audit._PANEL_SPECIAL[(canonical, "delay")] = "report_delay"
            audit._PANEL_SPECIAL[(canonical, "filing_event")] = "report_event"
        elif canonical == "report_revision_magnitude":
            current = np.resize(np.array([110.0, 125.0, 87.0, 160.0, 121.0, 210.0]), rows)
            previous = np.resize(np.array([100.0, 120.0, 90.0, 150.0, 110.0, 200.0]), rows)
            report_no = np.arange(rows)
            periods = np.array([f"{2000 + int(n) // 4}Q{int(n) % 4 + 1}" for n in report_no], dtype=object)
            panels["revision_current"] = pd.DataFrame(np.repeat(current[:, None], cols, axis=1), index=index, columns=columns)
            panels["revision_previous"] = pd.DataFrame(np.repeat(previous[:, None], cols, axis=1), index=index, columns=columns)
            panels["revision_period"] = pd.DataFrame(np.repeat(periods[:, None], cols, axis=1), index=index, columns=columns)
            panels["revision_event"] = pd.DataFrame(1.0, index=index, columns=columns)
            audit._PANEL_SPECIAL[(canonical, "x")] = "revision_current"
            audit._PANEL_SPECIAL[(canonical, "prev_x")] = "revision_previous"
            audit._PANEL_SPECIAL[(canonical, "current_period_id")] = "revision_period"
            audit._PANEL_SPECIAL[(canonical, "prev_period_id")] = "revision_period"
            audit._PANEL_SPECIAL[(canonical, "revision_event")] = "revision_event"
        else:
            base = 100.0 + 10.0 * (np.arange(rows) // 2)
            values = base + np.where(np.arange(rows) % 2 == 0, 0.0, np.resize(np.array([5.0, -10.0, 15.0]), rows))
            report_no = np.arange(rows) // 2
            periods = np.array([f"{2000 + int(n) // 4}Q{int(n) % 4 + 1}" for n in report_no], dtype=object)
            revision_ids = np.arange(rows) % 2
            panels["revision_value"] = pd.DataFrame(np.repeat(values[:, None], cols, axis=1), index=index, columns=columns)
            panels["revision_period"] = pd.DataFrame(np.repeat(periods[:, None], cols, axis=1), index=index, columns=columns)
            panels["revision_id"] = pd.DataFrame(np.repeat(revision_ids[:, None], cols, axis=1), index=index, columns=columns)
            audit._PANEL_SPECIAL[(canonical, "x")] = "revision_value"
            audit._PANEL_SPECIAL[(canonical, "period_id")] = "revision_period"
            audit._PANEL_SPECIAL[(canonical, "revision_id")] = "revision_id"
    signed = np.where(
            np.arange(len(panels["returns"]))[:, None] % 11 == 0,
            1.0,
            np.where(np.arange(len(panels["returns"]))[:, None] % 17 == 0, -1.0, 0.0),
        )
    panels["signed_event"] = pd.DataFrame(
        np.repeat(signed, panels["returns"].shape[1], axis=1),
        index=panels["returns"].index,
        columns=panels["returns"].columns,
    )
    panels.setdefault("returns", panels.get("ret"))
    if canonical == "panel_day_night_beta_gap":
        hourly = pd.date_range("2024-01-01", periods=len(panels["returns"]), freq="h")
        for value in panels.values():
            if isinstance(value, pd.DataFrame) and len(value) == len(hourly):
                value.index = hourly
    if canonical == "same_calendar_month_return":
        monthly = pd.date_range("2005-01-01", periods=len(panels["returns"]), freq="MS")
        for value in panels.values():
            if isinstance(value, pd.DataFrame) and len(value) == len(monthly):
                value.index = monthly
    if canonical == "ts_activity_spectral_entropy":
        rows, cols = panels["returns"].shape
        phase = np.arange(rows, dtype=float)[:, None]
        scale = np.arange(1, cols + 1, dtype=float)[None, :]
        activity = 2.0 + 0.5 * np.sin(phase / 5.0) + 0.001 * phase * scale
        panels["volume"] = pd.DataFrame(activity, index=panels["returns"].index, columns=panels["returns"].columns)
    if canonical == "ts_roll_effective_spread":
        rows, cols = panels["close"].shape
        step = np.arange(rows, dtype=float)[:, None]
        asset = np.arange(cols, dtype=float)[None, :]
        # Roll's estimator needs negatively autocorrelated price changes.
        bounce = np.where((step.astype(int) + asset.astype(int)) % 2 == 0, 0.02, -0.02)
        price = 100.0 + 0.001 * step + bounce
        panels["roll_price"] = pd.DataFrame(
            price, index=panels["close"].index, columns=panels["close"].columns
        )
    if canonical == "panel_peer_graph_aggregate":
        symbols = list(panels["returns"].columns)
        weights = np.ones((len(symbols), len(symbols)), dtype=float)
        np.fill_diagonal(weights, 0.0)
        panels["similarity"] = pd.DataFrame(weights, index=symbols, columns=symbols)
    if canonical.startswith("ex_self_"):
        panels["group"].iloc[:, :] = "ALL"
    # === R28 fixture repairs (this agent's batch) =================================
    # Dedicated synthetic panels for canonicals whose all-NaN output was a
    # fixture-input problem (no source operator is modified).
    if canonical == "ts_extrema_divergence_strength":
        rows, cols = panels["close"].shape
        t = np.arange(rows, dtype=float)[:, None]
        ast = np.arange(cols, dtype=float)[None, :]
        osc = 100.0 + 6.0 * np.sin(2.0 * np.pi * t / 13.0 + ast / 3.0) \
              + 2.0 * np.cos(2.0 * np.pi * t / 5.0)
        panels["extrema_x"] = pd.DataFrame(
            osc, index=panels["close"].index, columns=panels["close"].columns
        )
        # y is a phase-aligned, slightly-amplified copy of x so same-side extrema
        # coincide and the divergence is the (non-degenerate) relative move.
        panels["extrema_y"] = pd.DataFrame(
            1.1 * osc, index=panels["close"].index, columns=panels["close"].columns
        )
        audit._PANEL_SPECIAL[("ts_extrema_divergence_strength", "x")] = "extrema_x"
        audit._PANEL_SPECIAL[("ts_extrema_divergence_strength", "y")] = "extrema_y"
    if canonical == "ts_first_passage_conditional_time":
        rows, cols = panels["close"].shape
        rng = np.random.default_rng(20260919)
        fp = np.cumsum(rng.normal(0.0, 0.02, size=(rows, cols)), axis=0)
        panels["fp_x"] = pd.DataFrame(
            fp, index=panels["close"].index, columns=panels["close"].columns
        )
        # Small constant volatility: barrier=1.0 -> a ~0.05 band that the random
        # walk actually crosses within horizon (price-level barrier never was).
        panels["fp_scale"] = pd.DataFrame(
            0.05 + np.zeros((rows, cols)),
            index=panels["close"].index, columns=panels["close"].columns,
        )
        audit._PANEL_SPECIAL[("ts_first_passage_conditional_time", "x")] = "fp_x"
        audit._PANEL_SPECIAL[("ts_first_passage_conditional_time", "scale")] = "fp_scale"
    if canonical in ("ts_threshold_cycle_asymmetry", "ts_threshold_cycle_period"):
        rows, cols = panels["close"].shape
        t = np.arange(rows, dtype=float)[:, None]
        ast = np.arange(cols, dtype=float)[None, :]
        # Oscillates in [-3, 3] so it crosses the default +/-2 band and forms
        # hysteresis cycles inside the 20-bar window.
        cyc = 3.0 * np.sin(2.0 * np.pi * t / 11.0 + ast / 4.0)
        panels["cycle_x"] = pd.DataFrame(
            cyc, index=panels["close"].index, columns=panels["close"].columns
        )
        audit._PANEL_SPECIAL[(canonical, "x")] = "cycle_x"
    if canonical in ("ts_state_exit_hazard", "ts_state_residual_life"):
        rows, cols = panels["close"].shape
        # Switch every 5 rows (instead of 7) so >=5 completed episodes land
        # inside history_window and the survival kernel clears min_completed_runs.
        st = ((np.arange(rows)[:, None] // 5 + np.arange(cols)[None, :]) % 2).astype(float)
        panels["state_episode"] = pd.DataFrame(
            st, index=panels["close"].index, columns=panels["close"].columns
        )
        audit._PANEL_SPECIAL[(canonical, "state")] = "state_episode"
    if canonical == "ts_transition_intensity":
        rows, cols = panels["close"].shape
        # A real binary state (toggling every 5 rows) so transitions are sparse
        # and well-defined; combined with x=returns (mapped in _apply_overrides)
        # the normalized intensity has a non-degenerate MAD.
        st = ((np.arange(rows)[:, None] // 5 + np.arange(cols)[None, :]) % 2).astype(float)
        panels["state_episode"] = pd.DataFrame(
            st, index=panels["close"].index, columns=panels["close"].columns
        )
        audit._PANEL_SPECIAL[("ts_transition_intensity", "state")] = "state_episode"
    return panels


def _collect_canonicals():
    load_all()
    canonicals = sorted(OperatorRegistry.list_canonical())
    # R28: support a comma-separated subset so the 1436-test file can be run in
    # memory-bounded batches on the small server (scripts/run_r28_execute_batches.py).
    subset = os.environ.get("R28_EXECUTE_SUBSET", "")
    if subset:
        allowed = {s.strip() for s in subset.split(",") if s.strip()}
        canonicals = [c for c in canonicals if c in allowed]
    return canonicals


CURRENT_CANONICALS = _collect_canonicals()


# Exact-only allowlist. Entries must name the expected exception class and a
# stable message fragment; expected generic-fixture rejections are reported as
# skips and are never counted as finite execution coverage.
_CONTRACT_REJECTED: dict[str, tuple[type[Exception], str]] = {}


@pytest.mark.parametrize("canonical", CURRENT_CANONICALS, ids=CURRENT_CANONICALS)
def test_canonical_executes(canonical):
    reg = OperatorRegistry
    backends = reg._operators.get(canonical, {}) or {}
    if "pandas_numpy" in backends:
        backend_name, op = "pandas_numpy", backends["pandas_numpy"]
    else:
        backend_name, op = next(iter(backends.items())) if backends else (None, None)
    if op is None:
        pytest.fail(f"{canonical}: registered canonical without runtime implementation")

    panels = _panels_for(canonical)
    try:
        args, kwargs = audit._build_call(canonical, op, panels)
    except Exception as exc:
        pytest.fail(f"{canonical}: fixture builder failed: {type(exc).__name__}: {exc}")
    # The DatetimeIndex stays external for native Polars calls and is restored
    # by row position after strict height/column checks.  This is value-path
    # coverage, not timestamp-identity coverage.
    # Minute inputs may emit either minute or daily panels.  Select the expected
    # topology from the canonical's declared output grain; names/prefixes are
    # not evidence of output frequency.
    output_grain = reg._catalog.get(canonical, {}).get("output_grain")
    if output_grain == "daily":
        template = panels["close"]
    elif canonical == "session_event_recovery_score":
        # Authored EOD kernel: one exact row per normalized trade date.  The
        # generated catalog currently omits output_grain for this canonical,
        # so assert the documented/runtime daily topology explicitly.
        daily_index = panels["close"].index.normalize().unique()
        template = pd.DataFrame(index=daily_index, columns=panels["close"].columns, dtype=float)
    else:
        template = next((a for a in args if isinstance(a, pd.DataFrame)), panels["close"])
    if backend_name == "polars":
        def _to_polars(value):
            if not isinstance(value, pd.DataFrame):
                return value
            frame = value.copy()
            return pl.from_pandas(frame.reset_index(drop=True))

        args = [_to_polars(value) for value in args]
        kwargs = {key: _to_polars(value) for key, value in kwargs.items()}
    try:
        out = op.calculate(*args, **kwargs)
    except Exception as exc:
        expected = _CONTRACT_REJECTED.get(canonical)
        if expected is not None:
            exc_type, message = expected
            assert isinstance(exc, exc_type), (
                f"{canonical}: expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
            )
            assert message in str(exc), (
                f"{canonical}: expected rejection containing {message!r}, got {exc!s}"
            )
            pytest.skip(
                f"{canonical}: expected generic-fixture rejection: "
                f"{exc_type.__name__}: {message}"
            )
        pytest.fail(f"{canonical}: runtime raised {type(exc).__name__}: {exc}")

    if isinstance(out, (int, float, np.number)):
        assert np.isfinite(float(out)), f"{canonical}: scalar non-finite"
        return
    if isinstance(out, pl.DataFrame):
        assert out.height == len(template.index), f"{canonical}: Polars height mismatch"
        assert out.columns == list(template.columns), f"{canonical}: Polars columns/order mismatch"
        out = out.to_pandas()
        out.index = template.index
    if isinstance(out, pd.DataFrame):
        if canonical not in _LONG_FORMAT_CANONICALS:
            assert out.shape == template.shape, f"{canonical}: shape {out.shape} != {template.shape}"
            assert out.index.equals(template.index), f"{canonical}: index mismatch"
            assert out.columns.equals(template.columns), f"{canonical}: columns mismatch"
        vals = out.to_numpy(dtype=float)
    else:
        try:
            vals = np.asarray(out, dtype=float)
        except Exception as exc:
            pytest.fail(
                f"{canonical}: unexpected nonnumeric output "
                f"{type(out).__name__}: {type(exc).__name__}: {exc}"
            )
        if vals.size == 0:
            pytest.fail(f"{canonical}: empty output")
        if vals.ndim == 0:
            assert np.isfinite(float(vals)), f"{canonical}: scalar array non-finite"
            return

    assert not np.isinf(vals).any(), f"{canonical}: produced inf"
    if np.isfinite(vals).any():
        return
    pytest.fail(f"{canonical}: all-NaN output on a valid fixture")
