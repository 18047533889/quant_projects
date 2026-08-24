# -*- coding: utf-8 -*-
"""R47 新增算子 typed signatures.

Typed signatures for the 2026-08-11 new-operator taskbook families: intraday
state / event-window / impulse-response / slice-mask / volume-profile /
round-price / limit-EOD primitives, daily technical indicators, turnover chip
surfaces and cross-sectional predictability / async-beta panel operators.

Merged into ``OPERATOR_SIGNATURES`` from ``operator_types.py``.
"""
from __future__ import annotations

from factor_engine.backend.operator_types import ArgSpec, OperatorSignature, TypeKind

_F = TypeKind.SERIES_FLOAT
_B = TypeKind.SERIES_BOOL
_W = TypeKind.WINDOW
_INT = TypeKind.SCALAR_INT
_FLT = TypeKind.SCALAR_FLOAT
_ANY = TypeKind.ANY
_S = TypeKind.SERIES_STRING


def _sig(canonical: str, *args: ArgSpec, output: TypeKind = _F) -> OperatorSignature:
    return OperatorSignature(canonical, tuple(args), output=output)


def phase3_operator_signatures() -> dict[str, OperatorSignature]:
    """R47 new-operator signatures (pandas reference; minute panels are SERIES_FLOAT)."""
    signatures: dict[str, OperatorSignature] = {}

    # ---- intraday state family (state_ops) ---------------------------------
    signatures["intra_state_count"] = _sig(
        "intra_state_count",
        ArgSpec("state", _F), ArgSpec("target", _ANY), ArgSpec("window_days", _INT),
    )
    signatures["intra_state_sum"] = _sig(
        "intra_state_sum",
        ArgSpec("x", _F), ArgSpec("state", _F), ArgSpec("target", _ANY),
        ArgSpec("window_days", _INT),
    )
    signatures["intra_state_vwap"] = _sig(
        "intra_state_vwap",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("state", _F),
        ArgSpec("target", _ANY), ArgSpec("window_days", _INT),
    )
    signatures["intra_state_interval_moment"] = _sig(
        "intra_state_interval_moment",
        ArgSpec("state", _F), ArgSpec("target", _ANY), ArgSpec("moment", _S),
        ArgSpec("window_days", _INT), ArgSpec("min_events", _INT),
    )
    signatures["intra_state_follow_ratio"] = _sig(
        "intra_state_follow_ratio",
        ArgSpec("x", _F), ArgSpec("state", _F), ArgSpec("target", _ANY),
        ArgSpec("lead_bars", _INT), ArgSpec("window_days", _INT),
    )
    signatures["intra_state_follow_beta"] = _sig(
        "intra_state_follow_beta",
        ArgSpec("x", _F), ArgSpec("state", _F), ArgSpec("target", _ANY),
        ArgSpec("lead_bars", _INT), ArgSpec("window_days", _INT), ArgSpec("min_events", _INT),
    )
    signatures["intra_state_follow_corr"] = _sig(
        "intra_state_follow_corr",
        ArgSpec("x", _F), ArgSpec("state", _F), ArgSpec("target", _ANY),
        ArgSpec("lead_bars", _INT), ArgSpec("window_days", _INT), ArgSpec("min_events", _INT),
    )
    signatures["intra_state_pair_same_slot_corr"] = _sig(
        "intra_state_pair_same_slot_corr",
        ArgSpec("state_a", _F), ArgSpec("target_a", _ANY), ArgSpec("state_b", _F),
        ArgSpec("target_b", _ANY), ArgSpec("window_days", _INT), ArgSpec("min_slots", _INT),
    )
    signatures["intra_state_dwell_stats"] = _sig(
        "intra_state_dwell_stats",
        ArgSpec("state", _F), ArgSpec("target_state", _ANY), ArgSpec("output", _S),
        ArgSpec("min_slots", _INT),
    )
    signatures["intra_state_transition_entropy"] = _sig(
        "intra_state_transition_entropy",
        ArgSpec("state", _F), ArgSpec("min_slots", _INT), ArgSpec("normalize", _ANY),
        ArgSpec("include_self", _ANY),
    )
    signatures["intra_neighbor_event_class"] = _sig(
        "intra_neighbor_event_class",
        ArgSpec("event_mask", _F), ArgSpec("radius", _INT), ArgSpec("isolated_code", _INT),
        ArgSpec("clustered_code", _INT),
    )
    signatures["intra_range_gap_flag"] = _sig(
        "intra_range_gap_flag",
        ArgSpec("high", _F), ArgSpec("low", _F), ArgSpec("event_mask", _F),
        ArgSpec("neighbor_bars", _INT),
    )

    # ---- intraday event-window / impulse-response (event_response) ---------
    signatures["intra_event_window_reduce"] = _sig(
        "intra_event_window_reduce",
        ArgSpec("x", _F), ArgSpec("event_mask", _F), ArgSpec("pre", _INT),
        ArgSpec("post", _INT), ArgSpec("reducer", _S), ArgSpec("event_select", _S),
        ArgSpec("min_obs", _INT),
    )
    signatures["intra_event_pre_post_contrast"] = _sig(
        "intra_event_pre_post_contrast",
        ArgSpec("x", _F), ArgSpec("event_mask", _F), ArgSpec("pre", _INT),
        ArgSpec("post", _INT), ArgSpec("metric", _S), ArgSpec("event_select", _S),
        ArgSpec("min_obs", _INT),
    )
    signatures["intra_impulse_event_detector"] = _sig(
        "intra_impulse_event_detector",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("event", _S),
        ArgSpec("threshold", _S), ArgSpec("z", _FLT), ArgSpec("min_bars", _INT),
        ArgSpec("merge_gap", _INT), ArgSpec("output", _S),
    )
    signatures["intra_post_impulse_response"] = _sig(
        "intra_post_impulse_response",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("amount", _F),
        ArgSpec("direction", _S), ArgSpec("threshold", _S), ArgSpec("z", _FLT),
        ArgSpec("horizon", _INT), ArgSpec("output", _S),
    )
    signatures["intra_probe_outcome_score"] = _sig(
        "intra_probe_outcome_score",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("amount", _F),
        ArgSpec("direction", _S), ArgSpec("z", _FLT), ArgSpec("probe_horizon", _INT),
        ArgSpec("response_horizon", _INT), ArgSpec("output", _S),
    )
    signatures["intra_supply_absorption_score"] = _sig(
        "intra_supply_absorption_score",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("amount", _F),
        ArgSpec("event", _S), ArgSpec("horizon", _INT), ArgSpec("output", _S),
    )
    signatures["intra_consolidation_quality"] = _sig(
        "intra_consolidation_quality",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("amount", _F),
        ArgSpec("trigger", _S), ArgSpec("trigger_z", _FLT), ArgSpec("horizon", _INT),
        ArgSpec("output", _S),
    )
    signatures["intra_response_curve_features"] = _sig(
        "intra_response_curve_features",
        ArgSpec("price", _F), ArgSpec("activity", _F), ArgSpec("trigger", _S),
        ArgSpec("horizon", _INT), ArgSpec("curve", _S), ArgSpec("output", _S),
    )
    signatures["intra_liquidity_resilience_curve_fit"] = _sig(
        "intra_liquidity_resilience_curve_fit",
        ArgSpec("price", _F), ArgSpec("activity", _F), ArgSpec("shock_threshold", _FLT),
        ArgSpec("horizon", _INT), ArgSpec("output", _S),
    )

    # ---- intraday slice / profile / round-price (slice_profile) ------------
    signatures["intra_slice_mask_reduce"] = _sig(
        "intra_slice_mask_reduce",
        ArgSpec("x", _F), ArgSpec("mask_field", _F), ArgSpec("window", _ANY),
        ArgSpec("slice", _ANY), ArgSpec("mask_side", _S), ArgSpec("mask_q", _FLT),
        ArgSpec("reducer", _S), ArgSpec("min_bars", _INT),
    )
    signatures["intra_slice_mask_pair_reduce"] = _sig(
        "intra_slice_mask_pair_reduce",
        ArgSpec("x", _F), ArgSpec("y", _F), ArgSpec("mask_field", _F),
        ArgSpec("window", _ANY), ArgSpec("slice", _ANY), ArgSpec("mask_side", _S),
        ArgSpec("mask_q", _FLT), ArgSpec("y_lag", _INT), ArgSpec("reducer", _S),
        ArgSpec("min_pairs", _INT),
    )
    signatures["intra_multiresolution_resample_reduce"] = _sig(
        "intra_multiresolution_resample_reduce",
        ArgSpec("x", _F), ArgSpec("bar_minutes", _INT), ArgSpec("lookback_days", _INT),
        ArgSpec("reducer", _S), ArgSpec("session_split", _ANY), ArgSpec("min_coverage", _FLT),
    )
    signatures["intra_same_slot_zscore"] = _sig(
        "intra_same_slot_zscore",
        ArgSpec("x", _F), ArgSpec("history_days", _INT), ArgSpec("ddof", _INT),
        ArgSpec("min_history", _INT),
    )
    signatures["intra_session_boundary_jump"] = _sig(
        "intra_session_boundary_jump",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("pre_close", _F),
        ArgSpec("boundary", _S), ArgSpec("pre_bars", _INT), ArgSpec("post_bars", _INT),
        ArgSpec("output", _S),
    )
    signatures["intra_volume_at_price_profile"] = _sig(
        "intra_volume_at_price_profile",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("bins", _INT),
        ArgSpec("weighting", _S), ArgSpec("price_basis", _S), ArgSpec("normalize", _ANY),
        ArgSpec("output", _S),
    )
    signatures["intra_volume_profile_peak_geometry"] = _sig(
        "intra_volume_profile_peak_geometry",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("bins", _INT),
        ArgSpec("smooth", _INT), ArgSpec("min_prominence", _FLT), ArgSpec("output", _S),
    )
    signatures["intra_volume_profile_supply_structure"] = _sig(
        "intra_volume_profile_supply_structure",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("bins", _INT),
        ArgSpec("decay", _FLT), ArgSpec("output", _S),
    )
    signatures["intra_volume_profile_value_area"] = _sig(
        "intra_volume_profile_value_area",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("bins", _INT),
        ArgSpec("target_mass", _FLT), ArgSpec("output", _S),
    )
    signatures["intra_round_price_clustering_share"] = _sig(
        "intra_round_price_clustering_share",
        ArgSpec("price", _F), ArgSpec("lattice", _FLT), ArgSpec("tolerance_ticks", _FLT),
        ArgSpec("window", _ANY), ArgSpec("output", _S), ArgSpec("min_bars", _INT),
    )
    signatures["intra_round_price_barrier_response"] = _sig(
        "intra_round_price_barrier_response",
        ArgSpec("price", _F), ArgSpec("lattice", _FLT), ArgSpec("lookback_days", _INT),
        ArgSpec("tolerance_ticks", _FLT), ArgSpec("output", _S), ArgSpec("min_events", _INT),
    )

    # ---- intraday limit / EOD (limit_eod) ---------------------------------
    signatures["intra_limit_pre_hit_pressure_profile"] = _sig(
        "intra_limit_pre_hit_pressure_profile",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("high_limit", _F),
        ArgSpec("low_limit", _F), ArgSpec("side", _S), ArgSpec("pre_window", _INT),
        ArgSpec("output", _S), ArgSpec("require_hit", _ANY),
    )
    signatures["intra_eod_reversal_decomposition"] = _sig(
        "intra_eod_reversal_decomposition",
        ArgSpec("price", _F), ArgSpec("volume", _F), ArgSpec("amount", _F),
        ArgSpec("window_minutes", _INT), ArgSpec("baseline", _S), ArgSpec("output", _S),
    )

    # ---- daily technical indicators (new_indicators) -----------------------
    signatures["HMA"] = _sig(
        "HMA", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("rounding", _S)
    )
    signatures["QQE"] = _sig(
        "QQE", ArgSpec("x", _F), ArgSpec("length", _INT), ArgSpec("smooth", _INT),
        ArgSpec("factor", _FLT), ArgSpec("output", _S),
    )
    signatures["RSX"] = _sig("RSX", ArgSpec("x", _F), ArgSpec("length", _INT))
    signatures["ALMA"] = _sig(
        "ALMA", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("offset", _FLT),
        ArgSpec("sigma", _FLT),
    )
    signatures["CoppockCurve"] = _sig(
        "CoppockCurve", ArgSpec("close", _F), ArgSpec("roc1", _INT), ArgSpec("roc2", _INT),
        ArgSpec("wma_window", _INT), ArgSpec("roc_mode", _S),
    )
    signatures["ElderRay"] = _sig(
        "ElderRay", ArgSpec("high", _F), ArgSpec("low", _F), ArgSpec("close", _F),
        ArgSpec("ema", _INT), ArgSpec("output", _S),
    )
    signatures["FisherTransform"] = _sig(
        "FisherTransform", ArgSpec("high", _F), ArgSpec("low", _F), ArgSpec("window", _W),
        ArgSpec("smooth", _FLT), ArgSpec("signal_smooth", _FLT), ArgSpec("output", _S),
    )

    # ---- turnover chip surfaces (chip_ops) ---------------------------------
    signatures["turnover_chip_age_cost_surface"] = _sig(
        "turnover_chip_age_cost_surface",
        ArgSpec("close", _F), ArgSpec("turnover", _F), ArgSpec("window", _W),
        ArgSpec("price_bins", _INT), ArgSpec("age_bins", _INT), ArgSpec("output", _S),
    )
    signatures["turnover_chip_overhang_surface"] = _sig(
        "turnover_chip_overhang_surface",
        ArgSpec("close", _F), ArgSpec("turnover", _F), ArgSpec("window", _W),
        ArgSpec("bins", _INT), ArgSpec("output", _S),
    )

    # ---- cross-sectional / panel (panel_gap) -------------------------------
    signatures["panel_async_beta_ex_self"] = _sig(
        "panel_async_beta_ex_self",
        ArgSpec("ret", _F), ArgSpec("weight", _F), ArgSpec("window", _W),
        ArgSpec("min_periods", _INT), ArgSpec("refresh_freq", _INT),
    )
    signatures["panel_factor_pocket_strength"] = _sig(
        "panel_factor_pocket_strength",
        ArgSpec("ret", _F), ArgSpec("factor", _F), ArgSpec("window", _W),
        ArgSpec("min_periods", _INT), ArgSpec("threshold", _FLT),
    )
    signatures["cs_predictability_mosaic_score"] = _sig(
        "cs_predictability_mosaic_score",
        ArgSpec("base_signal", _F), ArgSpec("realized_return", _F),
        ArgSpec("state_feature", _F), ArgSpec("window", _W), ArgSpec("min_history", _INT),
        ArgSpec("clusters", _INT), ArgSpec("lag", _INT),
    )
    signatures["panel_predictability_mosaic_score"] = _sig(
        "panel_predictability_mosaic_score",
        ArgSpec("base_signal", _F), ArgSpec("realized_return", _F),
        ArgSpec("state_feature", _F), ArgSpec("window", _W), ArgSpec("min_history", _INT),
        ArgSpec("clusters", _INT), ArgSpec("lag", _INT),
    )

    return signatures
