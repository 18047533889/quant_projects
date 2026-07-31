"""Market-neutral cold-start seeds for reviewed technical extensions.

The seed layer intentionally uses only core OHLCV fields so the same operator
coverage is available for A-share and US daily research. Operators accepting a
generic activity/turnover panel use ``volume`` here; production strategies may
replace it with a true turnover-rate field when available.
"""
from __future__ import annotations

from .model import ColdStartFactor

# formula, subfamily, horizon, complexity
_SPECS: tuple[tuple[str, str, int | None, str], ...] = (
    ("ts_prev_high(close,20)","prior_high",20,"basic"),
    ("ts_prev_low(close,20)","prior_low",20,"basic"),
    ("ts_distance_to_high(close,20)","distance_prior_high",20,"basic"),
    ("ts_distance_to_low(close,20)","distance_prior_low",20,"basic"),
    ("ts_breakout_high(close,20)","breakout_high",20,"basic"),
    ("ts_breakdown_low(close,20)","breakdown_low",20,"basic"),
    ("ts_new_high(close,60)","new_high",60,"basic"),
    ("ts_new_low(close,60)","new_low",60,"basic"),
    ("ts_channel_position(close,20)","price_channel_position",20,"basic"),
    ("ts_days_since_high(close,60)","days_since_high",60,"moderate"),
    ("ts_days_since_low(close,60)","days_since_low",60,"moderate"),
    ("ts_range_expansion(high,low,20)","range_expansion",20,"basic"),
    ("ts_confirmed_pivot_high(high,3,3)","confirmed_pivot_high",7,"moderate"),
    ("ts_confirmed_pivot_low(low,3,3)","confirmed_pivot_low",7,"moderate"),
    ("ts_last_pivot_high(high,3,3,60)","last_pivot_high",66,"moderate"),
    ("ts_last_pivot_low(low,3,3,60)","last_pivot_low",66,"moderate"),
    ("ts_pivot_high_age(high,3,3,60)","pivot_high_age",66,"moderate"),
    ("ts_pivot_low_age(low,3,3,60)","pivot_low_age",66,"moderate"),
    ("ts_resistance_level(high,3,3,60,3)","resistance_level",66,"moderate"),
    ("ts_support_level(low,3,3,60,3)","support_level",66,"moderate"),
    ("ts_resistance_slope(high,3,3,60,3)","resistance_slope",66,"moderate"),
    ("ts_support_slope(low,3,3,60,3)","support_slope",66,"moderate"),
    ("ts_distance_to_resistance(close,high,3,3,60,3)","distance_resistance",66,"moderate"),
    ("ts_distance_to_support(close,low,3,3,60,3)","distance_support",66,"moderate"),
    ("ts_resistance_break(close,high,3,3,60,3)","resistance_break",66,"moderate"),
    ("ts_support_break(close,low,3,3,60,3)","support_break",66,"moderate"),
    ("rolling_vwap(close,volume,20)","rolling_vwap",20,"basic"),
    ("vwap_deviation(close,volume,20)","vwap_deviation",20,"basic"),
    ("relative_volume(volume,20)","relative_volume",20,"basic"),
    ("volume_zscore(volume,20)","volume_zscore",20,"basic"),
    ("dollar_volume(close,volume)","dollar_volume",1,"basic"),
    ("dollar_volume_zscore(close,volume,20)","dollar_volume_zscore",20,"basic"),
    ("volume_momentum(volume,20)","volume_momentum",20,"basic"),
    ("turnover_momentum(volume,20)","activity_momentum",20,"basic"),
    ("turnover_zscore(volume,20)","activity_zscore",20,"basic"),
    ("return_volume_corr(ts_pct(close,1),volume,20)","return_volume_corr",20,"composite"),
    ("abs_return_volume_corr(ts_pct(close,1),volume,20)","abs_return_volume_corr",20,"composite"),
    ("signed_volume(ts_pct(close,1),volume)","signed_volume",1,"composite"),
    ("signed_dollar_volume(ts_pct(close,1),close,volume)","signed_dollar_volume",1,"composite"),
    ("rolling_obv(close,volume,20)","rolling_obv",20,"basic"),
    ("rolling_pvt(close,volume,20)","rolling_pvt",20,"basic"),
    ("CMF(high,low,close,volume,20)","chaikin_money_flow",20,"basic"),
    ("MFI(high,low,close,volume,14)","money_flow_index",14,"basic"),
    ("donchian_upper(high,20)","donchian_upper",20,"basic"),
    ("donchian_lower(low,20)","donchian_lower",20,"basic"),
    ("donchian_mid(high,low,20)","donchian_mid",20,"basic"),
    ("donchian_position(close,high,low,20)","donchian_position",20,"basic"),
    ("bollinger_pct_b(close,20,2.0)","bollinger_pct_b",20,"basic"),
    ("bollinger_width(close,20,2.0)","bollinger_width",20,"basic"),
    ("AROON_up(high,25)","aroon_up",25,"moderate"),
    ("AROON_down(low,25)","aroon_down",25,"moderate"),
    ("AROON(high,low,25)","aroon_oscillator",25,"moderate"),
    ("CCI(high,low,close,20)","cci",20,"moderate"),
    ("StochasticK(high,low,close,14)","stochastic_k",14,"basic"),
    ("StochasticD(high,low,close,14)","stochastic_d",16,"moderate"),
    ("WilliamsR(high,low,close,14)","williams_r",14,"basic"),
    ("efficiency_ratio(close,20)","efficiency_ratio",20,"basic"),
    ("choppiness_index(high,low,close,14)","choppiness_index",14,"moderate"),
    ("parkinson_vol(high,low,20)","parkinson_vol",20,"basic"),
    ("garman_klass_vol(open,high,low,close,20)","garman_klass_vol",20,"basic"),
    ("rogers_satchell_vol(open,high,low,close,20)","rogers_satchell_vol",20,"basic"),
    ("yang_zhang_vol(open,high,low,close,20)","yang_zhang_vol",20,"moderate"),
    ("overnight_volatility(open,close,20)","overnight_volatility",20,"basic"),
    ("intraday_volatility(open,close,20)","intraday_volatility",20,"basic"),
    ("range_volatility(high,low,close,20)","range_volatility",20,"basic"),
    ("ulcer_index(close,20)","ulcer_index",39,"moderate"),
    ("candle_body(open,close)","candle_body",1,"basic"),
    ("candle_abs_body(open,close)","candle_abs_body",1,"basic"),
    ("candle_range(high,low)","candle_range",1,"basic"),
    ("candle_body_ratio(open,high,low,close)","candle_body_ratio",1,"basic"),
    ("candle_upper_shadow(open,high,close)","upper_shadow",1,"basic"),
    ("candle_lower_shadow(open,low,close)","lower_shadow",1,"basic"),
    ("candle_upper_shadow_ratio(open,high,low,close)","upper_shadow_ratio",1,"basic"),
    ("candle_lower_shadow_ratio(open,high,low,close)","lower_shadow_ratio",1,"basic"),
    ("candle_close_location(high,low,close)","close_location",1,"basic"),
    ("candle_gap(open,close)","candle_gap",2,"basic"),
    ("candle_gap_pct(open,close)","candle_gap_pct",2,"basic"),
    ("candle_direction(open,close)","candle_direction",1,"basic"),
    ("candle_range_atr(open,high,low,close,14)","candle_range_atr",14,"moderate"),
    ("cdl_doji(open,high,low,close)","doji",1,"basic"),
    ("cdl_hammer(open,high,low,close)","hammer",1,"basic"),
    ("cdl_inverted_hammer(open,high,low,close)","inverted_hammer",1,"basic"),
    ("cdl_shooting_star(open,high,low,close)","shooting_star",1,"basic"),
    ("cdl_marubozu(open,high,low,close)","marubozu",1,"basic"),
    ("cdl_spinning_top(open,high,low,close)","spinning_top",1,"basic"),
    ("cdl_engulfing(open,high,low,close)","engulfing",2,"basic"),
    ("cdl_inside_bar(open,high,low,close)","inside_bar",2,"basic"),
    ("cdl_outside_bar(open,high,low,close)","outside_bar",2,"basic"),
)


def technical_extension_seeds(market: str) -> tuple[ColdStartFactor, ...]:
    if market not in {"ashare", "us"}:
        raise ValueError("market must be ashare or us")
    prefix = "cn" if market == "ashare" else "us"
    rows: list[ColdStartFactor] = []
    for index, (formula, subfamily, horizon, complexity) in enumerate(_SPECS, start=1):
        rows.append(
            ColdStartFactor(
                factor_id=f"{prefix}_ta_ext_{index:04d}",
                market=market,
                surface="extended",
                formula=formula,
                family="technical_extension",
                subfamily=subfamily,
                horizon=horizon,
                complexity=complexity,
                availability_tier="core",
                rationale="Reviewed causal technical/price-volume extension seed.",
                direction_hint="unknown",
                metadata={"causal": True, "frequency": "1d", "production_seed": True},
            )
        )
    return tuple(rows)
