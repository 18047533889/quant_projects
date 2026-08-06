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
        "ts_transition_count", ArgSpec("condition", _B), ArgSpec("window", _W)
    )
    signatures["ts_time_since_change"] = _sig(
        "ts_time_since_change", ArgSpec("condition", _B), ArgSpec("max_lookback", _INT)
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
    signatures["ashare_limit_touch"] = _sig(
        "ashare_limit_touch", ArgSpec("close", _F), ArgSpec("upper_limit", _F), ArgSpec("tick_tolerance", _FLT)
    )
    signatures["ashare_limit_one_price"] = _sig(
        "ashare_limit_one_price", ArgSpec("close", _F), ArgSpec("upper_limit", _F), ArgSpec("lower_limit", _F)
    )
    signatures["ashare_limit_failed"] = _sig(
        "ashare_limit_failed", ArgSpec("close", _F), ArgSpec("upper_limit", _F), ArgSpec("window", _W)
    )
    signatures["ashare_limit_open_break"] = _sig(
        "ashare_limit_open_break", ArgSpec("open", _F), ArgSpec("upper_limit", _F)
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
        "event_cumulative_return_past", ArgSpec("ret", _F), ArgSpec("event", _B), ArgSpec("window", _W)
    )
    signatures["event_abnormal_return_past"] = _sig(
        "event_abnormal_return_past",
        ArgSpec("ret", _F), ArgSpec("benchmark_ret", _F), ArgSpec("event", _B), ArgSpec("window", _W),
    )
    signatures["fin_applicability_mask"] = _sig(
        "fin_applicability_mask", ArgSpec("value", _F), ArgSpec("threshold", _FLT)
    )
    signatures["trading_day_diff"] = _sig(
        "trading_day_diff", ArgSpec("date1", _F), ArgSpec("date2", _F)
    )
    signatures["fin_announcement_lag"] = _sig(
        "fin_announcement_lag", ArgSpec("period_end_date", _F), ArgSpec("pub_date", _F)
    )

    return signatures
