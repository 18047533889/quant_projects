# -*- coding: utf-8
"""Phase-2 operator expansion typed signatures.

Covers the 2026-08 operator expansion: robust statistics, conditional rolling,
state/event, downside risk, group ex-self, return decomposition and A-share
limit behavior.  Merged into ``OPERATOR_SIGNATURES`` from ``operator_types.py``.
"""
from __future__ import annotations

from backend.operator_types import ArgSpec, OperatorSignature, TypeKind

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

    return signatures
