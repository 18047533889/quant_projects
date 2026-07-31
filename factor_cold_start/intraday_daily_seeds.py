"""Source-backed minute→daily cold-start seeds.

These are deliberately tagged ``minute`` so OHLCV-only campaigns do not request
minute data implicitly.
"""
from __future__ import annotations
from .model import ColdStartFactor

_FEATURES={
"intraday_realized_vol":"intraday_realized_vol(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_realized_skew":"intraday_realized_skew(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_realized_kurtosis":"intraday_realized_kurtosis(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_downside_semivariance":"intraday_downside_semivariance(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_jump_ratio":"intraday_jump_ratio(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_first_nmin_return":"intraday_first_nmin_return(bar_minutes=5, minutes=30, cutoff_time='session_close', min_coverage=0.9)",
"intraday_last_nmin_return":"intraday_last_nmin_return(bar_minutes=5, minutes=30, cutoff_time='session_close', min_coverage=0.9)",
"intraday_trend_slope":"intraday_trend_slope(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_path_efficiency":"intraday_path_efficiency(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_max_drawdown":"intraday_max_drawdown(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_time_of_high":"intraday_time_of_high(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_opening_range":"intraday_opening_range(bar_minutes=5, minutes=30, cutoff_time='session_close', min_coverage=0.9)",
"intraday_gap_fill_ratio":"intraday_gap_fill_ratio(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_closing_ramp":"intraday_closing_ramp(bar_minutes=5, minutes=30, cutoff_time='session_close', min_coverage=0.9)",
"intraday_close_to_vwap":"intraday_close_to_vwap(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_vwap_slope":"intraday_vwap_slope(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_volume_first_share":"intraday_volume_first_share(bar_minutes=5, minutes=30, cutoff_time='session_close', min_coverage=0.9)",
"intraday_volume_last_share":"intraday_volume_last_share(bar_minutes=5, minutes=30, cutoff_time='session_close', min_coverage=0.9)",
"intraday_volume_hhi":"intraday_volume_hhi(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_volume_entropy":"intraday_volume_entropy(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_signed_volume_imbalance":"intraday_signed_volume_imbalance(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_price_impact":"intraday_price_impact(bar_minutes=5, cutoff_time='session_close', min_coverage=0.9)",
"intraday_profile_zscore":"intraday_profile_zscore(bar_minutes=5, history_days=20, cutoff_time='session_close', min_coverage=0.9)",
"intraday_abnormal_volume_profile":"intraday_abnormal_volume_profile(bar_minutes=5, history_days=20, cutoff_time='session_close', min_coverage=0.9)",
}

def intraday_daily_seeds(market:str):
    return tuple(ColdStartFactor(
        factor_id=f"{market}_intraday_daily_{name}",market=market,surface="extended",formula=formula,
        family="intraday_to_daily",subfamily="minute_aggregation",horizon=5,complexity="moderate",
        availability_tier="minute",rationale=f"Minute-derived daily feature {name}.",direction_hint="unknown",
        metadata={"source":"intraday_daily_seeds","requires_minute_data":True,"operator":name},
    ) for name,formula in _FEATURES.items())
