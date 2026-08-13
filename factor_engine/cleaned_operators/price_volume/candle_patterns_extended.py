# -*- coding: utf-8 -*-
"""Extended causal candlestick-pattern factors.

These are deterministic geometry classifiers, not trading recommendations.
Most patterns keep trend/location/volume context separate so factor mining can
combine raw shape with independent features; the trend-gated reversal patterns
(``cdl_hammer`` / ``cdl_hanging_man``) embed their classical prior-downtrend /
prior-uptrend context because the name itself implies it (audit item 5).
Outputs are -1/0/+1 where a bullish/bearish interpretation is intrinsic to the
geometry and 0/1 otherwise. No operator reads bars after the output timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.price_volume.candle_geometry_v2 import _validate_ohlc

_EPS = 1e-12


def _parts(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame):
    body_signed = close - open_
    body = body_signed.abs()
    rng = (high - low).clip(lower=0.0)
    upper_body = pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()), index=open_.index, columns=open_.columns)
    lower_body = pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()), index=open_.index, columns=open_.columns)
    upper = high - upper_body
    lower = lower_body - low
    return body_signed, body, rng, upper, lower, upper_body, lower_body


def _safe_ratio(num, den):
    return num / den.replace(0, np.nan)


def _min_tick(close):
    """Zero-amplitude gate (audit item 4, A-share critical).

    A 一字板 bar (O=H=L=C) has body=range=upper=lower=0, so every doji subtype
    trivially satisfies its own thresholds and both dragonfly and gravestone
    fire.  A bar whose range is at or below ~1e-6 of price carries no pattern
    information and must not be classified as *any* candle pattern.
    """
    return 1e-6 * close.abs()


def _prior_trend(close, *, up, window=5):
    """Causal prior-trend context measured on the bar *before* the current
    candle: whether its close sits above (``up``) or below (``down``) the mean of
    the preceding ``window`` closes.  NaN where the baseline is not yet available
    so callers can emit "cannot judge" instead of a confident 0 (audit item 5).
    """
    prev = close.shift(1)
    base = close.shift(1).rolling(window, min_periods=window).mean()
    cmp = (prev > base) if up else (prev < base)
    return cmp.where(base.notna())


def _tick_size(price):
    """A-share tick convention: 0.01 for price < 10, 0.05 for < 100, else 0.1."""
    return pd.DataFrame(
        np.where(
            price.abs() < 10.0,
            0.01,
            np.where(price.abs() < 100.0, 0.05, 0.1),
        ),
        index=price.index,
        columns=price.columns,
    )


def _long_bodies(open_, high, low, close, min_body_frac=0.5):
    _, body, rng, _, _, _, _ = _parts(open_, high, low, close)
    return (body >= min_body_frac * rng) & (rng > _min_tick(close))


def _short_shadows(open_, high, low, close, max_body_frac=0.4):
    _, body, _, upper, lower, _, _ = _parts(open_, high, low, close)
    return (upper <= max_body_frac * body) & (lower <= max_body_frac * body)


def _register(name, fn, description, *, bars: int, replace_source="", override_reason=""):
    metadata = OperatorMetadata(
        name=name,
        category="candle_pattern",
        description=description,
        param_names=["open", "high", "low", "close"],
        return_type="series",
        tags=["pit_safe", "causal", "candle", f"bars_{bars}", "production_extension"],
    )
    def _calculate_series(self, open, high, low, close, **kwargs):
        return fn(open, high, low, close)
    cls = type(
        f"Production_{name}", (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="candle_pattern",
        business_category="technical_extension",
        canonical=name,
        source="production_candle_extensions",
        backend="pandas_numpy",
        status="production",
        replace=bool(replace_source),
        replacement_reason=override_reason,
        expected_old_source=replace_source,
    )(cls)


def _doji_mask(open_, high, low, close):
    _, body, rng, _, _, _, _ = _parts(open_, high, low, close)
    return (body <= 0.10 * rng) & (rng > _min_tick(close))


def dragonfly_doji(open_, high, low, close):
    _, body, rng, upper, lower, _, _ = _parts(open_, high, low, close)
    out = ((body <= 0.10*rng) & (lower >= 0.60*rng) & (upper <= 0.10*rng) & (rng > _min_tick(close))).astype(float)
    return out.where(_validate_ohlc(open_, high, low, close))


def gravestone_doji(open_, high, low, close):
    _, body, rng, upper, lower, _, _ = _parts(open_, high, low, close)
    out = ((body <= 0.10*rng) & (upper >= 0.60*rng) & (lower <= 0.10*rng) & (rng > _min_tick(close))).astype(float)
    return out.where(_validate_ohlc(open_, high, low, close))


def hammer(open_, high, low, close):
    # Classical hammer = hammer geometry in a prior *downtrend* (audit item 5).
    _, body, rng, upper, lower, _, _ = _parts(open_, high, low, close)
    flag = (body <= 0.35*rng) & (lower >= 2.0*body) & (upper <= 0.35*body.clip(lower=_EPS)) & (rng > _min_tick(close))
    downtrend = _prior_trend(close, up=False)
    pattern = flag & downtrend.fillna(False)
    out = pattern.astype(float)
    return out.where(downtrend.notna()).where(_validate_ohlc(open_, high, low, close))


def hanging_man(open_, high, low, close):
    # Hanging-man = hammer geometry in a prior *uptrend* (audit item 5).  The old
    # sign-flipped geometry fired without any trend context, fabricating a bearish
    # call on a bare hammer shape.
    _, body, rng, upper, lower, _, _ = _parts(open_, high, low, close)
    flag = (body <= 0.35*rng) & (lower >= 2.0*body) & (upper <= 0.35*body.clip(lower=_EPS)) & (rng > _min_tick(close))
    uptrend = _prior_trend(close, up=True)
    pattern = flag & uptrend.fillna(False)
    out = -pattern.astype(float)
    return out.where(uptrend.notna()).where(_validate_ohlc(open_, high, low, close))


def harami(open_, high, low, close):
    prev_open, prev_close = open_.shift(1), close.shift(1)
    prev_hi = pd.DataFrame(np.maximum(prev_open.to_numpy(), prev_close.to_numpy()), index=open_.index, columns=open_.columns)
    prev_lo = pd.DataFrame(np.minimum(prev_open.to_numpy(), prev_close.to_numpy()), index=open_.index, columns=open_.columns)
    cur_hi = pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()), index=open_.index, columns=open_.columns)
    cur_lo = pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()), index=open_.index, columns=open_.columns)
    inside = (cur_hi < prev_hi) & (cur_lo > prev_lo)
    bullish = inside & (prev_close < prev_open) & (close > open_)
    bearish = inside & (prev_close > prev_open) & (close < open_)
    out = bullish.astype(float) - bearish.astype(float)
    valid = _validate_ohlc(open_, high, low, close) & _validate_ohlc(open_, high, low, close).shift(1)
    return out.where(valid)


def harami_cross(open_, high, low, close):
    base = harami(open_, high, low, close)
    return base * _doji_mask(open_, high, low, close).astype(float)


def piercing(open_, high, low, close):
    prev_open, prev_close = open_.shift(1), close.shift(1)
    midpoint = (prev_open + prev_close) / 2.0
    flag = (
        (prev_close < prev_open) & (close > open_) &
        (open_ <= prev_close) & (close > midpoint) & (close < prev_open)
    )
    valid = _validate_ohlc(open_, high, low, close) & _validate_ohlc(open_, high, low, close).shift(1)
    return flag.astype(float).where(valid)


def dark_cloud(open_, high, low, close):
    prev_open, prev_close = open_.shift(1), close.shift(1)
    midpoint = (prev_open + prev_close) / 2.0
    flag = (
        (prev_close > prev_open) & (close < open_) &
        (open_ >= prev_close) & (close < midpoint) & (close > prev_open)
    )
    valid = _validate_ohlc(open_, high, low, close) & _validate_ohlc(open_, high, low, close).shift(1)
    return (-flag.astype(float)).where(valid)


def morning_star(open_, high, low, close):
    o2, c2 = open_.shift(2), close.shift(2)
    o1, c1 = open_.shift(1), close.shift(1)
    body2 = (c2-o2).abs(); body1 = (c1-o1).abs()
    range2 = (high.shift(2)-low.shift(2)).replace(0, np.nan)
    range1 = (high.shift(1)-low.shift(1)).replace(0, np.nan)
    midpoint2 = (o2+c2)/2.0
    flag = (
        (c2 < o2) & (body2 >= 0.50*range2) &
        (body1 <= 0.35*range1) &
        (close > open_) & (close > midpoint2)
    )
    valid = (
        _validate_ohlc(open_, high, low, close)
        & _validate_ohlc(open_, high, low, close).shift(1)
        & _validate_ohlc(open_, high, low, close).shift(2)
    )
    return flag.astype(float).where(valid)


def evening_star(open_, high, low, close):
    o2, c2 = open_.shift(2), close.shift(2)
    o1, c1 = open_.shift(1), close.shift(1)
    body2 = (c2-o2).abs(); body1 = (c1-o1).abs()
    range2 = (high.shift(2)-low.shift(2)).replace(0, np.nan)
    range1 = (high.shift(1)-low.shift(1)).replace(0, np.nan)
    midpoint2 = (o2+c2)/2.0
    flag = (
        (c2 > o2) & (body2 >= 0.50*range2) &
        (body1 <= 0.35*range1) &
        (close < open_) & (close < midpoint2)
    )
    valid = (
        _validate_ohlc(open_, high, low, close)
        & _validate_ohlc(open_, high, low, close).shift(1)
        & _validate_ohlc(open_, high, low, close).shift(2)
    )
    return (-flag.astype(float)).where(valid)


def three_white_soldiers(open_, high, low, close):
    # Fuller classical semantics (audit item 7): three rising bullish candles with
    # long bodies, short shadows and each opening inside the prior body.
    bull0 = close > open_; bull1 = close.shift(1) > open_.shift(1); bull2 = close.shift(2) > open_.shift(2)
    rising = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
    opens_inside = (
        (open_ >= open_.shift(1)) & (open_ <= close.shift(1)) &
        (open_.shift(1) >= open_.shift(2)) & (open_.shift(1) <= close.shift(2))
    )
    long_b = (
        _long_bodies(open_, high, low, close)
        & _long_bodies(open_.shift(1), high.shift(1), low.shift(1), close.shift(1))
        & _long_bodies(open_.shift(2), high.shift(2), low.shift(2), close.shift(2))
    )
    short_sh = (
        _short_shadows(open_, high, low, close)
        & _short_shadows(open_.shift(1), high.shift(1), low.shift(1), close.shift(1))
        & _short_shadows(open_.shift(2), high.shift(2), low.shift(2), close.shift(2))
    )
    out = (bull0 & bull1 & bull2 & rising & opens_inside & long_b & short_sh).astype(float)
    valid = (
        _validate_ohlc(open_, high, low, close)
        & _validate_ohlc(open_, high, low, close).shift(1)
        & _validate_ohlc(open_, high, low, close).shift(2)
    )
    return out.where(valid)


def three_black_crows(open_, high, low, close):
    # Fuller classical semantics (audit item 7): three falling bearish candles with
    # long bodies, short shadows and each opening inside the prior body.
    bear0 = close < open_; bear1 = close.shift(1) < open_.shift(1); bear2 = close.shift(2) < open_.shift(2)
    falling = (close < close.shift(1)) & (close.shift(1) < close.shift(2))
    opens_inside = (
        (open_ <= open_.shift(1)) & (open_ >= close.shift(1)) &
        (open_.shift(1) <= open_.shift(2)) & (open_.shift(1) >= close.shift(2))
    )
    long_b = (
        _long_bodies(open_, high, low, close)
        & _long_bodies(open_.shift(1), high.shift(1), low.shift(1), close.shift(1))
        & _long_bodies(open_.shift(2), high.shift(2), low.shift(2), close.shift(2))
    )
    short_sh = (
        _short_shadows(open_, high, low, close)
        & _short_shadows(open_.shift(1), high.shift(1), low.shift(1), close.shift(1))
        & _short_shadows(open_.shift(2), high.shift(2), low.shift(2), close.shift(2))
    )
    out = -(bear0 & bear1 & bear2 & falling & opens_inside & long_b & short_sh).astype(float)
    valid = (
        _validate_ohlc(open_, high, low, close)
        & _validate_ohlc(open_, high, low, close).shift(1)
        & _validate_ohlc(open_, high, low, close).shift(2)
    )
    return out.where(valid)


_TICK_K = 2  # tweezer "equal high/low" tolerance in ticks (audit item 6)


def tweezer_top(open_, high, low, close):
    # Tick-aware tolerance (audit item 6): the old 1e-4*price relative band gave
    # low-price stocks fewer ticks of slack than high-price stocks.  Now two highs
    # are "equal" when within ``_TICK_K`` A-share ticks of each other.
    same_high = (high - high.shift(1)).abs() <= _tick_size(high) * _TICK_K
    reversal = (close.shift(1) > open_.shift(1)) & (close < open_)
    out = -(same_high & reversal).astype(float)
    valid = _validate_ohlc(open_, high, low, close) & _validate_ohlc(open_, high, low, close).shift(1)
    return out.where(valid)


def tweezer_bottom(open_, high, low, close):
    same_low = (low - low.shift(1)).abs() <= _tick_size(low) * _TICK_K
    reversal = (close.shift(1) < open_.shift(1)) & (close > open_)
    out = (same_low & reversal).astype(float)
    valid = _validate_ohlc(open_, high, low, close) & _validate_ohlc(open_, high, low, close).shift(1)
    return out.where(valid)


for _name, _fn, _desc, _bars, _replace_source, _reason in [
    ("cdl_dragonfly_doji", dragonfly_doji, "Dragonfly-doji geometry with a zero-amplitude gate.", 1, "", ""),
    ("cdl_gravestone_doji", gravestone_doji, "Gravestone-doji geometry with a zero-amplitude gate.", 1, "", ""),
    ("cdl_hammer", hammer, "Hammer geometry in a prior downtrend (classical context).", 1, "production_technical_extensions", "audit item 5: embed prior-downtrend context inside cdl_hammer"),
    ("cdl_hanging_man", hanging_man, "Hanging-man geometry in a prior uptrend (classical context).", 1, "", ""),
    ("cdl_harami", harami, "Bullish +1 / bearish -1 harami body pattern.", 2, "", ""),
    ("cdl_harami_cross", harami_cross, "Harami whose current candle is a doji.", 2, "", ""),
    ("cdl_piercing", piercing, "Two-bar bullish piercing geometry.", 2, "", ""),
    ("cdl_dark_cloud_cover", dark_cloud, "Two-bar bearish dark-cloud-cover geometry.", 2, "", ""),
    ("cdl_morning_star", morning_star, "Three-bar bullish morning-star geometry.", 3, "", ""),
    ("cdl_evening_star", evening_star, "Three-bar bearish evening-star geometry.", 3, "", ""),
    ("cdl_three_white_soldiers", three_white_soldiers, "Three rising bullish candles: long bodies, short shadows, opens inside prior body.", 3, "", ""),
    ("cdl_three_black_crows", three_black_crows, "Three falling bearish candles: long bodies, short shadows, opens inside prior body.", 3, "", ""),
    ("cdl_tweezer_top", tweezer_top, "Two-bar equal-high bearish reversal geometry (tick-aware).", 2, "", ""),
    ("cdl_tweezer_bottom", tweezer_bottom, "Two-bar equal-low bullish reversal geometry (tick-aware).", 2, "", ""),
]:
    _register(_name, _fn, _desc, bars=_bars, replace_source=_replace_source, override_reason=_reason)
