# -*- coding: utf-8
"""Phase-2 operator expansion typed signatures.

Covers the 2026-08 operator expansion: robust statistics, conditional rolling,
state/event, downside risk, group ex-self, return decomposition and A-share
limit behavior.  Merged into ``OPERATOR_SIGNATURES`` from ``operator_types.py``.
"""
from __future__ import annotations

from factor_engine.backend.operator_types import ArgSpec, OperatorSignature, TypeKind

_F = TypeKind.SERIES_FLOAT
_B = TypeKind.SERIES_BOOL
_W = TypeKind.WINDOW
_INT = TypeKind.SCALAR_INT
_FLT = TypeKind.SCALAR_FLOAT
_ANY = TypeKind.ANY
_G = TypeKind.GROUP_KEY


def _sig(canonical: str, *args: ArgSpec, output: TypeKind = _F) -> OperatorSignature:
    return OperatorSignature(canonical, tuple(args), output=output)


def phase2_operator_signatures() -> dict[str, OperatorSignature]:
    signatures: dict[str, OperatorSignature] = {}

    # ---- robust statistics -------------------------------------------------
    signatures["ts_quantile_range"] = _sig(
        "ts_quantile_range",
        ArgSpec("x", _F), ArgSpec("window", _W),
        ArgSpec("q_low", _FLT), ArgSpec("q_high", _FLT), ArgSpec("min_periods", _INT),
    )
    signatures["ts_trimmed_mean"] = _sig(
        "ts_trimmed_mean",
        ArgSpec("x", _F), ArgSpec("window", _W),
        ArgSpec("trim_ratio", _FLT), ArgSpec("min_periods", _INT),
    )
    signatures["ts_robust_zscore"] = _sig(
        "ts_robust_zscore",
        ArgSpec("x", _F), ArgSpec("window", _W),
        ArgSpec("center", _ANY), ArgSpec("scale", _ANY), ArgSpec("clip", _ANY),
    )

    # ---- direction / concentration ----------------------------------------
    for name, extra in (
        ("ts_positive_ratio", ("threshold",)),
        ("ts_negative_ratio", ("threshold",)),
        ("ts_zero_ratio", ("tolerance",)),
    ):
        signatures[name] = _sig(
            name,
            ArgSpec("x", _F), ArgSpec("window", _W),
            ArgSpec(extra[0], _FLT), ArgSpec("min_periods", _INT),
        )
    signatures["ts_abs_concentration"] = _sig(
        "ts_abs_concentration", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_periods", _INT)
    )
    signatures["ts_abs_entropy"] = _sig(
        "ts_abs_entropy",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("normalize", _ANY), ArgSpec("min_periods", _INT),
    )

    # ---- conditional rolling ----------------------------------------------
    signatures["ts_min_if"] = _sig(
        "ts_min_if", ArgSpec("x", _F), ArgSpec("condition", _B), ArgSpec("window", _W), ArgSpec("min_periods", _INT)
    )
    signatures["ts_max_if"] = _sig(
        "ts_max_if", ArgSpec("x", _F), ArgSpec("condition", _B), ArgSpec("window", _W), ArgSpec("min_periods", _INT)
    )
    signatures["ts_quantile_if"] = _sig(
        "ts_quantile_if",
        ArgSpec("x", _F), ArgSpec("condition", _B), ArgSpec("window", _W),
        ArgSpec("q", _FLT), ArgSpec("min_periods", _INT),
    )
    signatures["ts_corr_if"] = _sig(
        "ts_corr_if",
        ArgSpec("x", _F), ArgSpec("y", _F), ArgSpec("condition", _B), ArgSpec("window", _W), ArgSpec("min_periods", _INT),
    )
    signatures["ts_beta_if"] = _sig(
        "ts_beta_if",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("condition", _B), ArgSpec("window", _W), ArgSpec("min_periods", _INT),
    )
    signatures["ts_regression_resid_if"] = _sig(
        "ts_regression_resid_if",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("condition", _B), ArgSpec("window", _W), ArgSpec("min_periods", _INT),
    )

    # ---- state / event -----------------------------------------------------
    signatures["ts_transition_count"] = _sig(
        "ts_transition_count", ArgSpec("condition", _B), ArgSpec("window", _W), ArgSpec("missing_policy", _ANY)
    )
    signatures["ts_time_since_change"] = _sig(
        "ts_time_since_change", ArgSpec("condition", _B), ArgSpec("max_lookback", _INT), ArgSpec("missing_policy", _ANY)
    )
    signatures["ts_event_spacing_mean"] = _sig(
        "ts_event_spacing_mean", ArgSpec("condition", _B), ArgSpec("window", _W), ArgSpec("min_events", _INT)
    )
    signatures["ts_event_spacing_cv"] = _sig(
        "ts_event_spacing_cv", ArgSpec("condition", _B), ArgSpec("window", _W), ArgSpec("min_events", _INT)
    )

    # ---- downside risk -----------------------------------------------------
    signatures["ts_downside_deviation"] = _sig(
        "ts_downside_deviation",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("target", _FLT), ArgSpec("min_periods", _INT),
    )
    signatures["ts_upside_deviation"] = _sig(
        "ts_upside_deviation",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("target", _FLT), ArgSpec("min_periods", _INT),
    )
    signatures["ts_current_drawdown_duration"] = _sig(
        "ts_current_drawdown_duration", ArgSpec("x", _F), ArgSpec("window", _W)
    )
    signatures["ts_time_under_water"] = _sig(
        "ts_time_under_water", ArgSpec("x", _F), ArgSpec("window", _W)
    )
    signatures["ts_best_lag_corr"] = _sig(
        "ts_best_lag_corr",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("max_lag", _INT),
    )
    signatures["ts_price_delay"] = _sig(
        "ts_price_delay", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("max_lag", _INT)
    )

    # ---- group ex-self / robust residual ----------------------------------
    signatures["group_ex_self_mean"] = _sig(
        "group_ex_self_mean", ArgSpec("x", _F), ArgSpec("group", _G)
    )
    signatures["group_ex_self_weighted_mean"] = _sig(
        "group_ex_self_weighted_mean", ArgSpec("x", _F), ArgSpec("weight", _F), ArgSpec("group", _G)
    )
    signatures["hierarchical_group_neutralize"] = _sig(
        "hierarchical_group_neutralize", ArgSpec("x", _F), ArgSpec("group", _G), ArgSpec("subgroup", _G)
    )
    signatures["cs_robust_resid"] = _sig(
        "cs_robust_resid",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("trim_ratio", _FLT), ArgSpec("add_intercept", _ANY),
    )
    signatures["cs_trimmed_ols_resid"] = _sig(
        "cs_trimmed_ols_resid",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("trim_ratio", _FLT), ArgSpec("add_intercept", _ANY),
    )
    signatures["cs_huber_resid"] = _sig(
        "cs_huber_resid",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("add_intercept", _ANY),
    )
    signatures["cs_lad_resid"] = _sig(
        "cs_lad_resid",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("add_intercept", _ANY),
    )

    # ---- return decomposition ---------------------------------------------
    signatures["overnight_return"] = _sig(
        "overnight_return", ArgSpec("open", _F), ArgSpec("pre_close", _F)
    )
    signatures["open_close_return"] = _sig(
        "open_close_return", ArgSpec("open", _F), ArgSpec("close", _F)
    )
    signatures["open_to_vwap_return"] = _sig(
        "open_to_vwap_return", ArgSpec("open", _F), ArgSpec("vwap", _F)
    )
    signatures["vwap_to_close_return"] = _sig(
        "vwap_to_close_return", ArgSpec("vwap", _F), ArgSpec("close", _F)
    )

    # ---- A-share limit behavior -------------------------------------------
    signatures["ashare_limit_distance"] = _sig(
        "ashare_limit_distance", ArgSpec("close", _F), ArgSpec("upper_limit", _F)
    )
    signatures["ashare_limit_up_touch"] = _sig(
        "ashare_limit_up_touch", ArgSpec("high", _F), ArgSpec("upper_limit", _F), ArgSpec("tick_tolerance", _FLT)
    )
    signatures["ashare_limit_down_touch"] = _sig(
        "ashare_limit_down_touch", ArgSpec("low", _F), ArgSpec("lower_limit", _F), ArgSpec("tick_tolerance", _FLT)
    )
    signatures["ashare_limit_one_price"] = _sig(
        "ashare_limit_one_price",
        ArgSpec("open", _F), ArgSpec("high", _F), ArgSpec("low", _F), ArgSpec("close", _F),
        ArgSpec("upper_limit", _F), ArgSpec("lower_limit", _F),
        ArgSpec("side", _ANY), ArgSpec("tick_tolerance", _FLT),
    )
    signatures["ashare_limit_failed"] = _sig(
        "ashare_limit_failed",
        ArgSpec("high", _F), ArgSpec("close", _F), ArgSpec("upper_limit", _F), ArgSpec("tick_tolerance", _FLT),
    )
    signatures["ashare_open_at_upper_limit"] = _sig(
        "ashare_open_at_upper_limit", ArgSpec("open", _F), ArgSpec("upper_limit", _F), ArgSpec("tick_tolerance", _FLT)
    )
    signatures["ashare_limit_open_failed"] = _sig(
        "ashare_limit_open_failed", ArgSpec("open", _F), ArgSpec("low", _F), ArgSpec("upper_limit", _F), ArgSpec("tick_tolerance", _FLT)
    )

    # ---- Filter Layer (2026-08-12) ----------------------------------------
    signatures["ts_hampel_filter_causal"] = _sig(
        "ts_hampel_filter_causal",
        ArgSpec("x", _F), ArgSpec("window", _W),
        ArgSpec("n_sigma", _FLT), ArgSpec("replacement", _ANY), ArgSpec("scale_floor", _FLT),
    )
    signatures["ts_median3_causal"] = _sig(
        "ts_median3_causal",
        ArgSpec("x", _F),
    )
    signatures["ts_rolling_median_causal"] = _sig(
        "ts_rolling_median_causal",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_periods", _INT),
    )
    signatures["state_adaptive_slew_limit"] = _sig(
        "state_adaptive_slew_limit",
        ArgSpec("x", _F), ArgSpec("slew_mult", _FLT),
        ArgSpec("scale_window", _INT), ArgSpec("scale_method", _ANY),
    )
    signatures["ts_kama"] = _sig(
        "ts_kama",
        ArgSpec("x", _F), ArgSpec("er_window", _INT),
        ArgSpec("fast_period", _INT), ArgSpec("slow_period", _INT), ArgSpec("min_periods", _INT),
    )

    # ---- relation aggregation (variadic ranked panels) --------------------
    for name in ("relation_hhi", "relation_entropy", "relation_topk_sum", "relation_rank_weighted_sum"):
        signatures[name] = OperatorSignature(
            name,
            (
                ArgSpec("s1", _F),
                ArgSpec("s2", _F, required=False),
                ArgSpec("s3", _F, required=False),
                ArgSpec("s4", _F, required=False),
                ArgSpec("s5", _F, required=False),
                ArgSpec("s6", _F, required=False),
                ArgSpec("s7", _F, required=False),
                ArgSpec("s8", _F, required=False),
                ArgSpec("s9", _F, required=False),
                ArgSpec("s10", _F, required=False),
            ),
        )
    signatures["relation_category_share"] = _sig(
        "relation_category_share", ArgSpec("value", _F), ArgSpec("category", _G)
    )
    signatures["relation_peer_weighted_mean_ex_self"] = _sig(
        "relation_peer_weighted_mean_ex_self", ArgSpec("value", _F), ArgSpec("weight", _F), ArgSpec("group", _G)
    )
    signatures["relation_entry_count"] = _sig(
        "relation_entry_count", ArgSpec("member", _B), ArgSpec("window", _W)
    )
    signatures["relation_exit_count"] = _sig(
        "relation_exit_count", ArgSpec("member", _B), ArgSpec("window", _W)
    )
    signatures["relation_weighted_change"] = _sig(
        "relation_weighted_change", ArgSpec("value", _F), ArgSpec("weight", _F)
    )

    # ---- index / event ----------------------------------------------------
    signatures["index_member"] = _sig(
        "index_member", ArgSpec("member", _B)
    )
    signatures["index_weight_change"] = _sig(
        "index_weight_change", ArgSpec("weight", _F), ArgSpec("window", _W)
    )
    signatures["index_entry_exit_event"] = _sig(
        "index_entry_exit_event", ArgSpec("member", _B)
    )
    signatures["index_membership_age"] = _sig(
        "index_membership_age", ArgSpec("member", _B), ArgSpec("max_lookback", _INT)
    )
    signatures["event_cumulative_return_past"] = _sig(
        "event_cumulative_return_past",
        ArgSpec("ret", _F), ArgSpec("event", _B), ArgSpec("window", _W), ArgSpec("event_effective_lag", _INT),
    )
    signatures["event_abnormal_return_past"] = _sig(
        "event_abnormal_return_past",
        ArgSpec("ret", _F), ArgSpec("benchmark_ret", _F), ArgSpec("event", _B), ArgSpec("window", _W), ArgSpec("event_effective_lag", _INT),
    )
    signatures["fin_applicability_mask"] = _sig(
        "fin_applicability_mask", ArgSpec("value", _F), ArgSpec("threshold", _FLT)
    )
    signatures["calendar_day_diff"] = _sig(
        "calendar_day_diff", ArgSpec("date1", _F), ArgSpec("date2", _F)
    )
    signatures["trading_day_diff"] = _sig(
        "trading_day_diff", ArgSpec("date1", _F), ArgSpec("date2", _F)
    )
    signatures["relation_distinct_count"] = _sig(
        "relation_distinct_count", ArgSpec("entity_ids", _F)
    )
    signatures["relation_overlap_ratio"] = _sig(
        "relation_overlap_ratio",
        ArgSpec("current_ids", _F), ArgSpec("previous_ids", _F), ArgSpec("method", _ANY)
    )

    # ---- model-type rolling regression (2026-08 P2) ------------------------
    signatures["ts_huber_regression_resid"] = _sig(
        "ts_huber_regression_resid",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_periods", _INT),
    )
    signatures["ts_ridge_regression_resid"] = _sig(
        "ts_ridge_regression_resid",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("alpha", _FLT), ArgSpec("min_periods", _INT),
    )
    signatures["ts_quantile_regression_slope"] = _sig(
        "ts_quantile_regression_slope",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("q", _FLT), ArgSpec("min_periods", _INT),
    )
    signatures["ts_ar_coefficient"] = _sig(
        "ts_ar_coefficient",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("lag", _INT), ArgSpec("min_periods", _INT),
    )
    signatures["ts_ar_fitted_value"] = _sig(
        "ts_ar_fitted_value",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("order", _INT),
    )
    signatures["ts_ar_in_sample_resid"] = _sig(
        "ts_ar_in_sample_resid",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("order", _INT),
    )
    signatures["ts_variance_ratio"] = _sig(
        "ts_variance_ratio",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("q", _INT), ArgSpec("min_periods", _INT),
    )
    signatures["ts_cusum_break_score"] = _sig(
        "ts_cusum_break_score", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_periods", _INT)
    )
    signatures["ts_level_shift_score"] = _sig(
        "ts_level_shift_score", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_periods", _INT)
    )
    signatures["ts_vol_shift_score"] = _sig(
        "ts_vol_shift_score", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_periods", _INT)
    )
    signatures["fin_announcement_lag"] = _sig(
        "fin_announcement_lag", ArgSpec("period_end_date", _F), ArgSpec("pub_date", _F)
    )

    # ---- advanced information theory (2026-08 Gemini round) ----------------
    for _name in ("ts_transfer_entropy", "ts_effective_transfer_entropy"):
        signatures[_name] = _sig(
            _name,
            ArgSpec("target", _F), ArgSpec("source", _F), ArgSpec("window", _W),
            ArgSpec("bins", _INT), ArgSpec("lag", _INT),
        )
    signatures["ts_score_rank_weighted_mean"] = _sig(
        "ts_score_rank_weighted_mean",
        ArgSpec("target", _F), ArgSpec("score", _F), ArgSpec("window", _W),
        ArgSpec("decay", _FLT),
    )
    signatures["report_benford_js_divergence"] = _sig(
        "report_benford_js_divergence", ArgSpec("amount", _F), ArgSpec("window", _W)
    )

    # ---- advanced dependence structure (2026-08 Gemini round) --------------
    signatures["ts_bures_corr_shift"] = _sig(
        "ts_bures_corr_shift",
        ArgSpec("x", _F), ArgSpec("y", _F),
        ArgSpec("recent_window", _W), ArgSpec("prior_window", _W),
    )
    for _name in ("ts_kramers_moyal_drift", "ts_kramers_moyal_diffusion"):
        signatures[_name] = _sig(
            _name, ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("bins", _INT)
        )
    signatures["cs_sliced_wasserstein_copula_shift"] = _sig(
        "cs_sliced_wasserstein_copula_shift",
        ArgSpec("f1", _F), ArgSpec("f2", _F), ArgSpec("f3", _F),
        ArgSpec("window", _W), ArgSpec("directions", _INT),
    )
    signatures["group_spd_feature_structure_shift"] = _sig(
        "group_spd_feature_structure_shift",
        ArgSpec("f1", _F), ArgSpec("f2", _F), ArgSpec("f3", _F),
        ArgSpec("group", _G), ArgSpec("reference_window", _W),
    )
    signatures["holder_class_js_shift"] = _sig(
        "holder_class_js_shift",
        ArgSpec("s1", _F), ArgSpec("s2", _F), ArgSpec("s3", _F), ArgSpec("s4", _F), ArgSpec("s5", _F),
        ArgSpec("ps1", _F), ArgSpec("ps2", _F), ArgSpec("ps3", _F), ArgSpec("ps4", _F), ArgSpec("ps5", _F),
    )

    # ---- advanced intraday (2026-08 Gemini round) --------------------------
    signatures["intraday_wasserstein_pair_distance"] = _sig(
        "intraday_wasserstein_pair_distance", ArgSpec("x", _F), ArgSpec("y", _F)
    )
    signatures["intraday_barrier_approach_acceleration"] = _sig(
        "intraday_barrier_approach_acceleration",
        ArgSpec("close", _F), ArgSpec("high_limit", _F), ArgSpec("low_limit", _F),
        ArgSpec("lookback", _INT),
    )
    for _name in ("intraday_quantile_curve_pca_score", "intraday_quantile_curve_pca_residual"):
        signatures[_name] = _sig(
            _name, ArgSpec("returns", _F), ArgSpec("window", _W), ArgSpec("k", _INT)
        )

    # ---- topological / information geometry (2026-08 Gemini round) ---------
    for _name in ("ts_betti_1_max_persistence", "ts_persistence_diagram_shift"):
        signatures[_name] = _sig(
            _name,
            ArgSpec("x", _F), ArgSpec("window", _W),
            ArgSpec("tau", _INT), ArgSpec("embedding_dim", _INT),
        )
    signatures["ts_fisher_information_shift"] = _sig(
        "ts_fisher_information_shift",
        ArgSpec("x", _F), ArgSpec("recent_window", _W), ArgSpec("prior_window", _W),
    )

    # ---- market-state description language (2026-08-08, §18) ---------------
    # A. Quantile-hit dynamics.
    signatures["ts_quantilogram"] = _sig(
        "ts_quantilogram",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("quantile", _FLT),
        ArgSpec("lag", _INT), ArgSpec("side", _ANY),
    )
    signatures["ts_cross_quantilogram"] = _sig(
        "ts_cross_quantilogram",
        ArgSpec("target", _F), ArgSpec("source", _F), ArgSpec("window", _W),
        ArgSpec("target_q", _FLT), ArgSpec("source_q", _FLT), ArgSpec("lag", _INT),
        ArgSpec("target_side", _ANY), ArgSpec("source_side", _ANY),
    )
    signatures["ts_quantile_crossing_spectral_concentration"] = _sig(
        "ts_quantile_crossing_spectral_concentration",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("quantile", _FLT), ArgSpec("side", _ANY),
    )
    # C. Extreme dependence.
    signatures["ts_extremogram"] = _sig(
        "ts_extremogram",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("quantile", _FLT),
        ArgSpec("lag", _INT), ArgSpec("side", _ANY),
    )
    signatures["ts_cross_extremogram"] = _sig(
        "ts_cross_extremogram",
        ArgSpec("target", _F), ArgSpec("source", _F), ArgSpec("window", _W),
        ArgSpec("target_q", _FLT), ArgSpec("source_q", _FLT), ArgSpec("lag", _INT),
        ArgSpec("target_side", _ANY), ArgSpec("source_side", _ANY),
    )
    signatures["ts_extremal_dependence_decay"] = _sig(
        "ts_extremal_dependence_decay",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("quantile", _FLT),
        ArgSpec("side", _ANY), ArgSpec("max_lag", _INT),
    )
    # B. Expectile.
    signatures["ts_expectile"] = _sig(
        "ts_expectile", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("tau", _FLT)
    )
    signatures["ts_expectile_beta"] = _sig(
        "ts_expectile_beta",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("tau", _FLT),
    )
    # D. Directional-change intrinsic time.
    for _name in (
        "ts_dc_overshoot_ratio", "ts_dc_event_rate",
        "ts_dc_duration_asymmetry", "ts_dc_overshoot_asymmetry",
    ):
        signatures[_name] = _sig(
            _name,
            ArgSpec("x", _F), ArgSpec("scale", _F), ArgSpec("threshold", _FLT), ArgSpec("window", _W),
        )
    # E. Multi-field covariance geometry.
    for _name in ("ts_feature_mode_share", "ts_feature_effective_rank"):
        signatures[_name] = _sig(
            _name, ArgSpec("f1", _F), ArgSpec("f2", _F), ArgSpec("f3", _F), ArgSpec("window", _W)
        )
    for _name in ("ts_feature_subspace_rotation",):
        signatures[_name] = _sig(
            _name,
            ArgSpec("f1", _F), ArgSpec("f2", _F), ArgSpec("f3", _F),
            ArgSpec("window", _W), ArgSpec("recent_window", _W), ArgSpec("prior_window", _W),
        )
    signatures["ts_beta_break_score"] = OperatorSignature(
        "ts_beta_break_score",
        (ArgSpec("y", _F), ArgSpec("x", _F),
         ArgSpec("recent_window", _W), ArgSpec("prior_window", _W)),
        output=_F, output_unit="dimensionless",
    )
    # F. Conditional / band dependence.
    signatures["ts_conditional_transfer_entropy"] = _sig(
        "ts_conditional_transfer_entropy",
        ArgSpec("target", _F), ArgSpec("source", _F), ArgSpec("condition", _F),
        ArgSpec("window", _W), ArgSpec("bins", _INT), ArgSpec("lag", _INT),
        ArgSpec("min_transitions", _INT),
    )
    signatures["ts_modwt_band_corr"] = _sig(
        "ts_modwt_band_corr",
        ArgSpec("x", _F), ArgSpec("y", _F), ArgSpec("window", _W),
        ArgSpec("level", _INT), ArgSpec("band", _INT),
    )
    # G. Sampling-scale / noise diagnostics and profile surprise (intraday).
    signatures["intraday_subsampled_rv_dispersion"] = _sig(
        "intraday_subsampled_rv_dispersion",
        ArgSpec("returns", _F), ArgSpec("sampling", _INT),
    )
    signatures["intraday_volatility_signature_slope"] = _sig(
        "intraday_volatility_signature_slope",
        ArgSpec("returns", _F), ArgSpec("max_interval", _INT),
    )
    signatures["intraday_realized_power_variation"] = _sig(
        "intraday_realized_power_variation",
        ArgSpec("returns", _F), ArgSpec("order", _FLT), ArgSpec("sampling", _INT),
    )
    signatures["intraday_profile_surprise_energy"] = _sig(
        "intraday_profile_surprise_energy",
        ArgSpec("x", _F), ArgSpec("history_days", _INT), ArgSpec("n_slots", _INT), ArgSpec("cap", _FLT),
    )
    signatures["intraday_profile_phase_shift"] = _sig(
        "intraday_profile_phase_shift",
        ArgSpec("x", _F), ArgSpec("history_days", _INT), ArgSpec("max_shift", _INT), ArgSpec("n_slots", _INT),
    )
    # H. No-L2 spread estimators.
    signatures["ohlc_corwin_schultz_spread"] = _sig(
        "ohlc_corwin_schultz_spread",
        ArgSpec("high", _F), ArgSpec("low", _F), ArgSpec("smooth_window", _W),
    )
    signatures["ts_roll_effective_spread"] = _sig(
        "ts_roll_effective_spread", ArgSpec("price", _F), ArgSpec("window", _W)
    )
    # J. Local non-linear cross-section.
    signatures["cs_knn_local_linear_residual"] = _sig(
        "cs_knn_local_linear_residual",
        ArgSpec("target", _F), ArgSpec("f1", _F), ArgSpec("f2", _F), ArgSpec("f3", _F),
        ArgSpec("k", _INT), ArgSpec("ridge", _FLT),
    )
    signatures["cs_knn_tangent_residual"] = _sig(
        "cs_knn_tangent_residual",
        ArgSpec("target", _F), ArgSpec("f1", _F), ArgSpec("f2", _F), ArgSpec("f3", _F),
        ArgSpec("k", _INT),
    )
    signatures["cs_knn_local_gradient_norm"] = _sig(
        "cs_knn_local_gradient_norm",
        ArgSpec("target", _F), ArgSpec("f1", _F), ArgSpec("f2", _F), ArgSpec("f3", _F),
        ArgSpec("k", _INT), ArgSpec("ridge", _FLT),
    )
    for _name in ("cs_rank_copula_mi", "cs_rank_copula_entropy"):
        signatures[_name] = _sig(
            _name, ArgSpec("a", _F), ArgSpec("b", _F), ArgSpec("grid", _INT)
        )
    # K + L. Systemic tail / relation diffusion.
    signatures["group_tail_centrality"] = _sig(
        "group_tail_centrality",
        ArgSpec("x", _F), ArgSpec("group_id", _G), ArgSpec("window", _W),
        ArgSpec("quantile", _FLT), ArgSpec("side", _ANY),
    )
    signatures["group_tail_lead_score"] = _sig(
        "group_tail_lead_score",
        ArgSpec("x", _F), ArgSpec("group_id", _G), ArgSpec("window", _W),
        ArgSpec("quantile", _FLT), ArgSpec("side", _ANY), ArgSpec("lag", _INT),
    )
    signatures["relation_diffusion_score"] = _sig(
        "relation_diffusion_score",
        ArgSpec("x", _F), ArgSpec("group", _G), ArgSpec("alpha", _FLT), ArgSpec("steps", _INT),
    )
    # M. Marked event.
    signatures["event_mark_autocorr"] = _sig(
        "event_mark_autocorr",
        ArgSpec("event", _B), ArgSpec("mark", _F), ArgSpec("history_window", _W), ArgSpec("event_lag", _INT),
    )
    signatures["event_interval_mark_coupling"] = _sig(
        "event_interval_mark_coupling",
        ArgSpec("event", _B), ArgSpec("mark", _F), ArgSpec("window", _W),
    )
    # N. Update clock.
    for _name in (
        "update_path_efficiency", "update_acceleration",
        "update_surprise", "update_direction_persistence",
    ):
        signatures[_name] = _sig(
            _name, ArgSpec("x", _F), ArgSpec("update_event", _B), ArgSpec("n_updates", _INT)
        )

    # O. 2026-08-08 Gemini-recommended primitives.
    signatures["group_topk_mean"] = _sig(
        "group_topk_mean",
        ArgSpec("target", _F), ArgSpec("score", _F), ArgSpec("group", _G),
        ArgSpec("k", _INT), ArgSpec("exclude_self", _ANY),
    )
    signatures["ts_value_at_argextreme"] = _sig(
        "ts_value_at_argextreme",
        ArgSpec("value", _F), ArgSpec("score", _F), ArgSpec("window", _W),
        ArgSpec("mode", _ANY), ArgSpec("include_current", _ANY),
    )
    signatures["ts_weighted_standardized_moment"] = _sig(
        "ts_weighted_standardized_moment",
        ArgSpec("x", _F), ArgSpec("weight", _F), ArgSpec("window", _W), ArgSpec("order", _INT),
    )
    signatures["ts_cov_if"] = _sig(
        "ts_cov_if",
        ArgSpec("x", _F), ArgSpec("y", _F), ArgSpec("condition", _B),
        ArgSpec("window", _W), ArgSpec("min_periods", _INT),
    )
    for _name in (
        "ts_return_spectral_entropy",
        "ts_spectral_entropy",
        "ts_detrended_level_spectral_entropy",
    ):
        signatures[_name] = _sig(
            _name,
            ArgSpec("x", _F),
            ArgSpec("window", _W),
            ArgSpec("input_kind", _ANY, required=False),
        )
    signatures["ts_dominant_cycle_period"] = _sig(
        "ts_dominant_cycle_period",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_peak_share", _FLT),
    )
    signatures["ts_activity_clock_lagged_value"] = _sig(
        "ts_activity_clock_lagged_value",
        ArgSpec("x", _F), ArgSpec("activity", _F), ArgSpec("budget", _FLT),
        ArgSpec("scale_window", _W), ArgSpec("max_lookback", _INT),
    )
    signatures["ts_activity_clock_age"] = _sig(
        "ts_activity_clock_age",
        ArgSpec("activity", _F), ArgSpec("budget", _FLT),
        ArgSpec("scale_window", _W), ArgSpec("max_lookback", _INT),
    )
    signatures["cs_weighted_percentile_rank"] = _sig(
        "cs_weighted_percentile_rank", ArgSpec("x", _F), ArgSpec("weight", _F)
    )
    signatures["group_distribution_js_divergence"] = _sig(
        "group_distribution_js_divergence",
        ArgSpec("x", _F), ArgSpec("group", _G),
        ArgSpec("bins", _INT), ArgSpec("min_group_size", _INT),
    )
    signatures["event_level_survival_share"] = _sig(
        "event_level_survival_share",
        ArgSpec("event", _B), ArgSpec("level", _F), ArgSpec("x", _F),
        ArgSpec("history_window", _W), ArgSpec("direction", _ANY),
    )
    signatures["ts_max_drawdown_activity_cost"] = _sig(
        "ts_max_drawdown_activity_cost",
        ArgSpec("x", _F), ArgSpec("activity", _F), ArgSpec("window", _W),
    )
    signatures["cs_multi_robust_resid"] = _sig(
        "cs_multi_robust_resid",
        ArgSpec("y", _F), ArgSpec("x1", _F),
        ArgSpec("x2", _F, required=False), ArgSpec("x3", _F, required=False),
        ArgSpec("add_intercept", _ANY),
    )
    signatures["relation_pagerank_centrality"] = _sig(
        "relation_pagerank_centrality",
        ArgSpec("x", _F), ArgSpec("group", _G), ArgSpec("damping", _FLT),
    )
    signatures["intraday_activity_duration_curvature"] = _sig(
        "intraday_activity_duration_curvature",
        ArgSpec("activity", _F), ArgSpec("buckets", _INT),
    )
    # Research-surface transforms.
    signatures["ts_wavelet_lowpass_reconstruct"] = _sig(
        "ts_wavelet_lowpass_reconstruct",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("level", _INT),
    )
    signatures["ts_signature_mahalanobis_anomaly"] = _sig(
        "ts_signature_mahalanobis_anomaly",
        ArgSpec("f1", _F), ArgSpec("f2", _F), ArgSpec("f3", _F),
        ArgSpec("path_window", _W), ArgSpec("history_window", _W),
    )
    signatures["ts_betti_crocker_bifurcation_score"] = _sig(
        "ts_betti_crocker_bifurcation_score",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("tau", _INT), ArgSpec("dim", _INT),
    )
    # ---- P. 2026-08-08 Gemini V2 round primitives -------------------------
    # HVG network.
    for _hvg in (
        "ts_hvg_degree_entropy",
        "ts_hvg_forward_backward_asymmetry",
        "ts_hvg_clustering_coefficient",
        "ts_hvg_assortativity",
        "ts_hvg_motif_entropy",
    ):
        signatures[_hvg] = _sig(
            _hvg, ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_periods", _INT)
        )
    # RQA line structure.
    for _rqa in (
        "ts_recurrence_determinism",
        "ts_recurrence_laminarity",
        "ts_recurrence_mean_diagonal_length",
        "ts_recurrence_longest_vertical_length",
    ):
        signatures[_rqa] = _sig(
            _rqa,
            ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("dim", _INT),
            ArgSpec("delay", _INT), ArgSpec("eps_fraction", _FLT),
            ArgSpec("min_line", _INT), ArgSpec("min_periods", _INT),
        )
    # change-point scores.
    for _cp in (
        "ts_glr_mean_shift_score",
        "ts_glr_variance_shift_score",
        "ts_pettitt_change_score",
    ):
        signatures[_cp] = _sig(
            _cp, ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_segment", _INT)
        )
    # OHLC microstructure spreads.
    signatures["ts_edge_effective_spread"] = _sig(
        "ts_edge_effective_spread",
        ArgSpec("open", _F), ArgSpec("high", _F), ArgSpec("low", _F),
        ArgSpec("close", _F), ArgSpec("window", _W),
    )
    signatures["ts_abdi_ranaldo_spread"] = _sig(
        "ts_abdi_ranaldo_spread",
        ArgSpec("close", _F), ArgSpec("high", _F), ArgSpec("low", _F),
        ArgSpec("window", _W),
    )
    signatures["ts_pastor_stambaugh_liquidity_gamma"] = _sig(
        "ts_pastor_stambaugh_liquidity_gamma",
        ArgSpec("ret", _F), ArgSpec("benchmark_ret", _F), ArgSpec("amount", _F),
        ArgSpec("window", _W), ArgSpec("min_periods", _INT),
    )
    # robust scale / location.
    signatures["ts_qn_scale"] = _sig(
        "ts_qn_scale", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_periods", _INT)
    )
    signatures["ts_hodges_lehmann_location"] = _sig(
        "ts_hodges_lehmann_location", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_periods", _INT)
    )
    # extreme value / event counting.
    signatures["ts_pickands_tail_index"] = _sig(
        "ts_pickands_tail_index",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("k", _INT), ArgSpec("side", _ANY),
    )
    signatures["ts_evt_threshold_stability"] = _sig(
        "ts_evt_threshold_stability",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("k_min", _INT),
        ArgSpec("k_max", _INT), ArgSpec("side", _ANY),
    )
    signatures["event_allan_factor"] = _sig(
        "event_allan_factor",
        ArgSpec("event", _B), ArgSpec("window", _W), ArgSpec("max_scale", _INT),
    )
    # compositional data.
    signatures["composition_clr_component"] = _sig(
        "composition_clr_component",
        ArgSpec("target", _F), ArgSpec("x1", _F, required=False),
        ArgSpec("x2", _F, required=False), ArgSpec("x3", _F, required=False),
        ArgSpec("x4", _F, required=False), ArgSpec("x5", _F, required=False),
        ArgSpec("x6", _F, required=False), ArgSpec("x7", _F, required=False),
    )
    signatures["composition_entropy"] = _sig(
        "composition_entropy",
        ArgSpec("x1", _F), ArgSpec("x2", _F), ArgSpec("x3", _F),
        ArgSpec("x4", _F, required=False), ArgSpec("x5", _F, required=False),
        ArgSpec("x6", _F, required=False), ArgSpec("x7", _F, required=False),
        ArgSpec("x8", _F, required=False),
    )
    for _cod in ("composition_aitchison_distance", "composition_ilr_balance", "composition_js_divergence"):
        signatures[_cod] = _sig(
            _cod,
            ArgSpec("x1", _F), ArgSpec("x2", _F), ArgSpec("x3", _F),
            ArgSpec("y1", _F), ArgSpec("y2", _F), ArgSpec("y3", _F),
            ArgSpec("x4", _F, required=False), ArgSpec("x5", _F, required=False),
            ArgSpec("x6", _F, required=False), ArgSpec("y4", _F, required=False),
            ArgSpec("y5", _F, required=False), ArgSpec("y6", _F, required=False),
        )
    # global-state / transport.
    signatures["cs_hartigan_dip"] = _sig(
        "cs_hartigan_dip", ArgSpec("x", _F), ArgSpec("min_cross", _INT)
    )
    signatures["group_wasserstein_barycenter_distance"] = _sig(
        "group_wasserstein_barycenter_distance",
        ArgSpec("x", _F), ArgSpec("group", _G), ArgSpec("window", _W),
        ArgSpec("min_group_size", _INT),
    )
    # cross-spectral.
    for _xs in ("ts_cross_spectral_coherence", "ts_cross_spectral_phase"):
        signatures[_xs] = _sig(
            _xs, ArgSpec("x", _F), ArgSpec("y", _F), ArgSpec("window", _W)
        )
    # intraday impact decay (minute-source).
    signatures["intraday_impact_decay_rate"] = _sig(
        "intraday_impact_decay_rate",
        ArgSpec("ret", _F), ArgSpec("amount", _F),
        ArgSpec("horizon", _INT), ArgSpec("shock_quantile", _FLT),
    )
    # multifractal asymmetry.
    signatures["ts_multifractal_asymmetry"] = _sig(
        "ts_multifractal_asymmetry", ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("min_periods", _INT)
    )
    # research-surface (DMD / bicoherence / kernel / BDS / SR).
    for _dmd in ("ts_dmd_dominant_growth_rate", "ts_dmd_dominant_frequency"):
        signatures[_dmd] = _sig(
            _dmd,
            ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("rank", _INT),
            ArgSpec("dim", _INT), ArgSpec("delay", _INT),
        )
    signatures["ts_dmd_mode_concentration"] = _sig(
        "ts_dmd_mode_concentration",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("rank", _INT),
        ArgSpec("dim", _INT), ArgSpec("delay", _INT), ArgSpec("top_k", _INT),
    )
    signatures["ts_bicoherence_top_decile_mean"] = _sig(
        "ts_bicoherence_top_decile_mean",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("n_segments", _INT),
    )
    # R9-OP-028: deprecated alias signature retained so legacy recipes load.
    signatures["ts_bicoherence_max"] = _sig(
        "ts_bicoherence_max",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("n_segments", _INT),
    )
    signatures["ts_kernel_granger_score"] = _sig(
        "ts_kernel_granger_score",
        ArgSpec("y", _F), ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("lag", _INT),
    )
    signatures["ts_residualized_hsic"] = _sig(
        "ts_residualized_hsic",
        ArgSpec("x", _F), ArgSpec("y", _F), ArgSpec("z", _F), ArgSpec("window", _W),
    )
    signatures["ts_bds_statistic"] = _sig(
        "ts_bds_statistic",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("embedding_dim", _INT),
        ArgSpec("distance_multiplier", _FLT),
    )
    signatures["ts_rolling_sr_gaussian_mean_shift_score"] = _sig(
        "ts_rolling_sr_gaussian_mean_shift_score",
        ArgSpec("x", _F), ArgSpec("window", _W), ArgSpec("shift_sigma", _FLT),
        ArgSpec("baseline_window", _W),
    )

    # Filter Layer (2026-08-12): robust EMA with innovation clipping.
    signatures["ts_robust_ema"] = _sig(
        "ts_robust_ema",
        ArgSpec("x", _F), ArgSpec("span", _INT), ArgSpec("clip_sigma", _FLT),
        ArgSpec("warmup_window", _INT), ArgSpec("scale_floor", _FLT),
    )
    signatures["ts_super_smoother"] = _sig(
        "ts_super_smoother",
        ArgSpec("x", _F), ArgSpec("period", _INT),
    )
    signatures["ts_butterworth_lowpass_causal"] = _sig(
        "ts_butterworth_lowpass_causal",
        ArgSpec("x", _F), ArgSpec("cutoff_period", _INT), ArgSpec("order", _INT),
    )
    signatures["ts_causal_local_linear_smoother"] = _sig(
        "ts_causal_local_linear_smoother",
        ArgSpec("x", _F), ArgSpec("window", _INT), ArgSpec("min_periods", _INT),
    )

    # Filter Layer: hysteresis and turnover control (2026-08-12 P0).
    signatures["state_adaptive_deadband"] = _sig(
        "state_adaptive_deadband",
        ArgSpec("x", _F), ArgSpec("band_mult", _FLT),
        ArgSpec("scale_window", _W), ArgSpec("scale_method", TypeKind.SCALAR_STR),
    )
    signatures["state_rank_deadband"] = _sig(
        "state_rank_deadband",
        ArgSpec("x", _F), ArgSpec("band_pct", _FLT), ArgSpec("group", _G),
    )
    signatures["state_quantile_hysteresis"] = _sig(
        "state_quantile_hysteresis",
        ArgSpec("x", _F), ArgSpec("enter_quantile", _FLT),
        ArgSpec("exit_quantile", _FLT), ArgSpec("group", _G),
    )
    signatures["state_adaptive_slew_limit"] = _sig(
        "state_adaptive_slew_limit",
        ArgSpec("x", _F), ArgSpec("slew_mult", _FLT),
        ArgSpec("scale_window", _W), ArgSpec("scale_method", TypeKind.SCALAR_STR),
    )
    signatures["state_l1_turnover_prox"] = _sig(
        "state_l1_turnover_prox",
        ArgSpec("x", _F), ArgSpec("lambda_turnover", _FLT),
    )
    signatures["state_l2_partial_adjustment"] = _sig(
        "state_l2_partial_adjustment",
        ArgSpec("x", _F), ArgSpec("lambda_smooth", _FLT),
    )
    signatures["state_confidence_weighted_ema"] = _sig(
        "state_confidence_weighted_ema",
        ArgSpec("x", _F), ArgSpec("confidence", _F),
        ArgSpec("alpha_min", _FLT), ArgSpec("alpha_max", _FLT),
    )
    signatures["state_uncertainty_deadband"] = _sig(
        "state_uncertainty_deadband",
        ArgSpec("x", _F), ArgSpec("uncertainty", _F), ArgSpec("k_sigma", _FLT),
    )
    signatures["state_cost_aware_deadband"] = _sig(
        "state_cost_aware_deadband",
        ArgSpec("x", _F), ArgSpec("cost_proxy", _F), ArgSpec("cost_mult", _FLT),
    )
    signatures["state_cost_aware_slew"] = _sig(
        "state_cost_aware_slew",
        ArgSpec("x", _F), ArgSpec("cost_proxy", _F), ArgSpec("slew_mult", _FLT),
    )

    return signatures
