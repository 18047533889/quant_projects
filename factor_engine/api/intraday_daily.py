# -*- coding: utf-8 -*-
"""Source-backed intraday→daily factor operators.

These helpers deliberately return opaque ``SourceRef`` columns instead of normal
shape-preserving cleaned operators: minute observations are aggregated to one
value per date×instrument before they enter a daily factor DAG.
"""
from __future__ import annotations
from typing import Any, Callable
from api.source_ref import source_col


def _feature(name: str, **params: Any):
    return source_col("IntradayFeature", str(name), **params)


def _factory(name: str, defaults: dict[str, Any] | None = None) -> Callable[..., Any]:
    defaults=dict(defaults or {})
    def fn(**kwargs: Any):
        params={**defaults,**kwargs}
        return _feature(name,**params)
    fn.__name__=name
    return fn

_COMMON={"bar_minutes":5,"cutoff_time":"session_close","min_coverage":0.80}

# Realized risk / jumps
intraday_realized_variance=_factory("realized_variance",_COMMON)
intraday_realized_vol=_factory("realized_vol",_COMMON)
intraday_realized_skew=_factory("realized_skew",_COMMON)
intraday_realized_kurtosis=_factory("realized_kurtosis",_COMMON)
intraday_realized_quarticity=_factory("realized_quarticity",_COMMON)
intraday_upside_semivariance=_factory("upside_semivariance",_COMMON)
intraday_downside_semivariance=_factory("downside_semivariance",_COMMON)
intraday_bipower_variation=_factory("bipower_variation",_COMMON)
intraday_jump_variation=_factory("jump_variation",_COMMON)
intraday_jump_ratio=_factory("jump_ratio",_COMMON)
intraday_signed_jump=_factory("signed_jump",{**_COMMON,"threshold":0.0})
intraday_max_abs_return=_factory("max_abs_return",_COMMON)
intraday_tail_return_sum=_factory("tail_return_sum",{**_COMMON,"q":0.95})
intraday_jump_count=_factory("jump_count",{**_COMMON,"threshold":0.01})

# Path shape / session behavior
intraday_return=_factory("return",_COMMON)
intraday_open_to_close_return=intraday_return
intraday_first_nmin_return=_factory("first_nmin_return",{**_COMMON,"minutes":30})
intraday_last_nmin_return=_factory("last_nmin_return",{**_COMMON,"minutes":30})
intraday_morning_return=_factory("morning_return",{**_COMMON,"split_time":"12:00"})
intraday_afternoon_return=_factory("afternoon_return",{**_COMMON,"split_time":"12:00"})
intraday_morning_afternoon_reversal=_factory("morning_afternoon_reversal",{**_COMMON,"split_time":"12:00"})
intraday_trend_slope=_factory("trend_slope",_COMMON)
intraday_trend_r2=_factory("trend_r2",_COMMON)
intraday_path_efficiency=_factory("path_efficiency",_COMMON)
intraday_return_autocorr=_factory("return_autocorr",{**_COMMON,"lag":1})
intraday_max_drawdown=_factory("max_drawdown",_COMMON)
intraday_max_runup=_factory("max_runup",_COMMON)
intraday_time_of_high=_factory("time_of_high",_COMMON)
intraday_time_of_low=_factory("time_of_low",_COMMON)
intraday_close_location=_factory("close_location",_COMMON)

# Open / close effects
intraday_opening_range=_factory("opening_range",{**_COMMON,"minutes":30})
intraday_opening_range_position=_factory("opening_range_position",{**_COMMON,"minutes":30})
intraday_opening_drive=_factory("opening_drive",{**_COMMON,"minutes":30})
intraday_gap_continuation=_factory("gap_continuation",_COMMON)
intraday_gap_fill_ratio=_factory("gap_fill_ratio",_COMMON)
intraday_closing_return=_factory("closing_return",{**_COMMON,"minutes":30})
intraday_closing_ramp=_factory("closing_ramp",{**_COMMON,"minutes":30})

# VWAP family
intraday_vwap=_factory("vwap",_COMMON)
intraday_close_to_vwap=_factory("close_to_vwap",_COMMON)
intraday_high_to_vwap=_factory("high_to_vwap",_COMMON)
intraday_low_to_vwap=_factory("low_to_vwap",_COMMON)
intraday_vwap_slope=_factory("vwap_slope",_COMMON)
intraday_vwap_deviation_mean=_factory("vwap_deviation_mean",_COMMON)
intraday_vwap_deviation_std=_factory("vwap_deviation_std",_COMMON)
intraday_vwap_cross_count=_factory("vwap_cross_count",_COMMON)

# Volume / turnover profile
intraday_volume_first_share=_factory("volume_first_share",{**_COMMON,"minutes":30})
intraday_volume_last_share=_factory("volume_last_share",{**_COMMON,"minutes":30})
intraday_volume_peak_time=_factory("volume_peak_time",_COMMON)
intraday_volume_hhi=_factory("volume_hhi",_COMMON)
intraday_volume_entropy=_factory("volume_entropy",_COMMON)
intraday_volume_profile_skew=_factory("volume_profile_skew",_COMMON)
intraday_volume_profile_slope=_factory("volume_profile_slope",_COMMON)
intraday_turnover_hhi=_factory("turnover_hhi",_COMMON)
intraday_turnover_entropy=_factory("turnover_entropy",_COMMON)

# Price×volume microstructure proxies
intraday_return_volume_corr=_factory("return_volume_corr",_COMMON)
intraday_abs_return_volume_corr=_factory("abs_return_volume_corr",_COMMON)
intraday_signed_volume_imbalance=_factory("signed_volume_imbalance",_COMMON)
intraday_volume_weighted_return=_factory("volume_weighted_return",_COMMON)
intraday_price_impact=_factory("price_impact",_COMMON)
intraday_amihud=_factory("amihud",_COMMON)
intraday_turnover_per_volatility=_factory("turnover_per_volatility",_COMMON)

# Time-of-day abnormality. History is explicitly bounded and excludes current day.
intraday_profile_zscore=_factory("profile_zscore",{**_COMMON,"history_days":20})
intraday_profile_deviation=_factory("profile_deviation",{**_COMMON,"history_days":20})
intraday_abnormal_volume_profile=_factory("abnormal_volume_profile",{**_COMMON,"history_days":20})
intraday_abnormal_return_profile=_factory("abnormal_return_profile",{**_COMMON,"history_days":20})
intraday_abnormal_vol_profile=_factory("abnormal_vol_profile",{**_COMMON,"history_days":20})

INTRADAY_DAILY_DSL_FUNCTIONS={
    name:value for name,value in globals().copy().items()
    if name.startswith("intraday_") and callable(value) and name not in {"intraday_open_to_close_return"}
}
INTRADAY_DAILY_DSL_FUNCTIONS["intraday_open_to_close_return"]=intraday_return
