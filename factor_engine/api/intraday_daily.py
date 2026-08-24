# -*- coding: utf-8 -*-
"""Source-backed intraday→daily factor authoring functions.

These functions return opaque ``SourceRef`` columns.  Aggregation happens at the
logical source boundary before the value enters a daily factor DAG.
"""
from __future__ import annotations

from typing import Any, Callable

from factor_engine.api.source_ref import source_col, transform_source_col

_TIMESTAMP_CONVENTIONS = frozenset({"bar_end", "bar_start"})


def _validated_params(params: dict[str, Any]) -> dict[str, Any]:
    output = dict(params)
    bar_minutes = int(output.get("bar_minutes", 5))
    min_bars = int(output.get("min_bars", 2))
    min_coverage = float(output.get("min_coverage", 0.8))
    history_days = int(output.get("history_days", 0))
    convention = str(output.get("timestamp_convention", "bar_end")).lower()
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    if min_bars < 2:
        raise ValueError("min_bars must be >= 2")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0,1]")
    if history_days < 0:
        raise ValueError("history_days must be non-negative")
    if convention not in _TIMESTAMP_CONVENTIONS:
        raise ValueError("timestamp_convention must be bar_end or bar_start")
    if "minutes" in output and int(output["minutes"]) <= 0:
        raise ValueError("minutes must be positive")
    if "q" in output and not 0 < float(output["q"]) < 1:
        raise ValueError("q must be in (0,1)")
    output.update(
        {
            "bar_minutes": bar_minutes,
            "min_bars": min_bars,
            "min_coverage": min_coverage,
            "history_days": history_days,
            "timestamp_convention": convention,
        }
    )
    return output


def _feature(name: str, **params: Any):
    base = source_col("StockMinuteBar", "Close")
    return transform_source_col(
        base,
        "intraday_feature",
        feature=str(name),
        **_validated_params(params),
    )


def _factory(
    name: str,
    defaults: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    defaults = dict(defaults or {})

    def function(**kwargs: Any):
        return _feature(name, **{**defaults, **kwargs})

    function.__name__ = name
    function.__doc__ = (
        f"Build SourceRef-backed intraday daily feature {name!r}."
    )
    return function


_COMMON = {
    "bar_minutes": 5,
    "cutoff_time": "session_close",
    "min_coverage": 0.80,
    "min_bars": 2,
    "timestamp_convention": "bar_end",
}

intraday_realized_variance = _factory("realized_variance", _COMMON)
intraday_realized_vol = _factory("realized_vol", _COMMON)
intraday_realized_skew = _factory("realized_skew", _COMMON)
intraday_realized_kurtosis = _factory("realized_kurtosis", _COMMON)
intraday_realized_quarticity = _factory("realized_quarticity", _COMMON)
intraday_upside_semivariance = _factory("upside_semivariance", _COMMON)
intraday_downside_semivariance = _factory("downside_semivariance", _COMMON)
intraday_bipower_variation = _factory("bipower_variation", _COMMON)
intraday_jump_variation = _factory("jump_variation", _COMMON)
intraday_jump_ratio = _factory("jump_ratio", _COMMON)
intraday_signed_jump = _factory("signed_jump", {**_COMMON, "threshold": 0.0})
intraday_max_abs_return = _factory("max_abs_return", _COMMON)
intraday_tail_return_sum = _factory("tail_return_sum", {**_COMMON, "q": 0.95})
intraday_jump_count = _factory("jump_count", {**_COMMON, "threshold": 0.01})
intraday_return = _factory("return", _COMMON)
intraday_open_to_close_return = intraday_return
intraday_first_nmin_return = _factory(
    "first_nmin_return", {**_COMMON, "minutes": 30}
)
intraday_last_nmin_return = _factory(
    "last_nmin_return", {**_COMMON, "minutes": 30}
)
intraday_morning_return = _factory(
    "morning_return", {**_COMMON, "split_time": "12:00"}
)
intraday_afternoon_return = _factory(
    "afternoon_return", {**_COMMON, "split_time": "12:00"}
)
intraday_morning_afternoon_reversal = _factory(
    "morning_afternoon_reversal", {**_COMMON, "split_time": "12:00"}
)
intraday_trend_slope = _factory("trend_slope", _COMMON)
intraday_trend_r2 = _factory("trend_r2", _COMMON)
intraday_path_length = _factory("path_length", _COMMON)
intraday_path_efficiency = _factory("path_efficiency", _COMMON)
intraday_reversal_count = _factory("reversal_count", _COMMON)
intraday_return_autocorr = _factory("return_autocorr", {**_COMMON, "lag": 1})
intraday_max_drawdown = _factory("max_drawdown", _COMMON)
intraday_max_runup = _factory("max_runup", _COMMON)
intraday_time_of_high = _factory("time_of_high", _COMMON)
intraday_time_of_low = _factory("time_of_low", _COMMON)
intraday_close_location = _factory("close_location", _COMMON)
intraday_opening_range = _factory("opening_range", {**_COMMON, "minutes": 30})
intraday_opening_range_position = _factory(
    "opening_range_position", {**_COMMON, "minutes": 30}
)
intraday_opening_drive = _factory("opening_drive", {**_COMMON, "minutes": 30})
intraday_gap_continuation = _factory("gap_continuation", _COMMON)
intraday_gap_fill_ratio = _factory("gap_fill_ratio", _COMMON)
intraday_limit_up_touch_fraction = _factory("limit_up_touch_fraction", _COMMON)
intraday_limit_down_touch_fraction = _factory("limit_down_touch_fraction", _COMMON)
intraday_limit_up_close = _factory("limit_up_close", _COMMON)
intraday_limit_down_close = _factory("limit_down_close", _COMMON)
intraday_closing_return = _factory("closing_return", {**_COMMON, "minutes": 30})
intraday_closing_ramp = _factory("closing_ramp", {**_COMMON, "minutes": 30})
intraday_vwap = _factory("vwap", _COMMON)
intraday_close_to_vwap = _factory("close_to_vwap", _COMMON)
intraday_high_to_vwap = _factory("high_to_vwap", _COMMON)
intraday_low_to_vwap = _factory("low_to_vwap", _COMMON)
intraday_vwap_slope = _factory("vwap_slope", _COMMON)
intraday_vwap_deviation_mean = _factory("vwap_deviation_mean", _COMMON)
intraday_vwap_deviation_std = _factory("vwap_deviation_std", _COMMON)
intraday_vwap_cross_count = _factory("vwap_cross_count", _COMMON)
intraday_volume_first_share = _factory(
    "volume_first_share", {**_COMMON, "minutes": 30}
)
intraday_volume_last_share = _factory(
    "volume_last_share", {**_COMMON, "minutes": 30}
)
intraday_volume_peak_time = _factory("volume_peak_time", _COMMON)
intraday_volume_hhi = _factory("volume_hhi", _COMMON)
intraday_volume_entropy = _factory("volume_entropy", _COMMON)
intraday_volume_profile_skew = _factory("volume_profile_skew", _COMMON)
intraday_volume_profile_slope = _factory("volume_profile_slope", _COMMON)
intraday_turnover_hhi = _factory("turnover_hhi", _COMMON)
intraday_turnover_entropy = _factory("turnover_entropy", _COMMON)
intraday_return_volume_corr = _factory("return_volume_corr", _COMMON)
intraday_abs_return_volume_corr = _factory("abs_return_volume_corr", _COMMON)
intraday_signed_volume_imbalance = _factory("signed_volume_imbalance", _COMMON)
intraday_average_trade_price = _factory("average_trade_price", _COMMON)
intraday_active_volume_share = _factory("active_volume_share", _COMMON)
intraday_volume_weighted_return = _factory("volume_weighted_return", _COMMON)
intraday_price_impact = _factory("price_impact", _COMMON)
intraday_amihud = _factory("amihud", _COMMON)
intraday_turnover_per_volatility = _factory("turnover_per_volatility", _COMMON)
intraday_profile_zscore = _factory(
    "profile_zscore", {**_COMMON, "history_days": 20}
)
intraday_profile_deviation = _factory(
    "profile_deviation", {**_COMMON, "history_days": 20}
)
intraday_abnormal_volume_profile = _factory(
    "abnormal_volume_profile", {**_COMMON, "history_days": 20}
)
intraday_abnormal_return_profile = _factory(
    "abnormal_return_profile", {**_COMMON, "history_days": 20}
)
intraday_abnormal_vol_profile = _factory(
    "abnormal_vol_profile", {**_COMMON, "history_days": 20}
)

# ---- 2026-08 operator expansion: genuinely missing intraday features --------
intraday_lunch_gap_return = _factory("lunch_gap_return", _COMMON)
intraday_return_activity_corr = _factory("return_activity_corr", {**_COMMON, "activity": "volume"})
intraday_vwap_above_ratio = _factory("vwap_above_ratio", _COMMON)
intraday_kyle_lambda_proxy = _factory("kyle_lambda_proxy", _COMMON)
intraday_extreme_bar_return = _factory("extreme_bar_return", {**_COMMON, "side": "max"})
intraday_segment_return = _factory("segment_return", {**_COMMON, "segment": "morning"})
intraday_segment_volume_share = _factory("segment_volume_share", {**_COMMON, "segment": "morning"})
intraday_segment_amount_share = _factory("segment_amount_share", {**_COMMON, "segment": "morning"})
intraday_segment_vwap_deviation = _factory("segment_vwap_deviation", {**_COMMON, "segment": "morning"})
intraday_segment_realized_vol = _factory("segment_realized_vol", {**_COMMON, "segment": "morning"})
intraday_limit_first_hit_time = _factory("limit_first_hit_time", _COMMON)
intraday_limit_duration = _factory("limit_duration", _COMMON)
intraday_limit_reopen_count = _factory("limit_reopen_count", _COMMON)

# The document-spec ``intra_*`` names are registered as native operators in
# ``cleaned_operators/microstructure/intraday_agg.py`` (they take explicit
# minute-bar panel inputs).  The SourceRef-backed ``intraday_*`` factories
# remain for the read-minute-implicitly path.

# Build function registry with both intraday_* and intra_* aliases
_all_intraday = {
    name: value
    for name, value in globals().copy().items()
    if name.startswith("intraday_") and callable(value)
}

# Add intra_* aliases for all intraday_* functions (Phase 4 requirement)
INTRADAY_DAILY_DSL_FUNCTIONS = {}
for name, func in _all_intraday.items():
    INTRADAY_DAILY_DSL_FUNCTIONS[name] = func
    # Create intra_* alias (e.g. intraday_realized_variance -> intra_realized_variance)
    alias = name.replace("intraday_", "intra_", 1)
    INTRADAY_DAILY_DSL_FUNCTIONS[alias] = func
