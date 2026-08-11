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
}

_EXPLICIT_POLICIES.update(_R47_POLICIES)
