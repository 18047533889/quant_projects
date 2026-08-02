# -*- coding: utf-8 -*-
"""Extended causal candlestick-pattern factors.

These are deterministic geometry classifiers, not trading recommendations. Trend
context is intentionally separate so factor mining can combine raw pattern shape
with independent trend/location/volume features. Outputs are -1/0/+1 where a
bullish/bearish interpretation is intrinsic to the geometry and 0/1 otherwise.
No operator reads bars after the output timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

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


def _register(name, fn, description, *, bars: int):
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
    )(cls)


def _doji_mask(open_, high, low, close):
    _, body, rng, _, _, _, _ = _parts(open_, high, low, close)
    return body <= 0.10 * rng


def dragonfly_doji(open_, high, low, close):
    _, body, rng, upper, lower, _, _ = _parts(open_, high, low, close)
    return ((body <= 0.10*rng) & (lower >= 0.60*rng) & (upper <= 0.10*rng)).astype(float)


def gravestone_doji(open_, high, low, close):
    _, body, rng, upper, lower, _, _ = _parts(open_, high, low, close)
    return ((body <= 0.10*rng) & (upper >= 0.60*rng) & (lower <= 0.10*rng)).astype(float)


def hanging_man(open_, high, low, close):
    # Raw hanging-man geometry is hammer-like; encode bearish convention only.
    _, body, rng, upper, lower, _, _ = _parts(open_, high, low, close)
    flag = (body <= 0.35*rng) & (lower >= 2.0*body) & (upper <= 0.35*body.clip(lower=_EPS))
    return -flag.astype(float)


def harami(open_, high, low, close):
    prev_open, prev_close = open_.shift(1), close.shift(1)
    prev_hi = pd.DataFrame(np.maximum(prev_open.to_numpy(), prev_close.to_numpy()), index=open_.index, columns=open_.columns)
    prev_lo = pd.DataFrame(np.minimum(prev_open.to_numpy(), prev_close.to_numpy()), index=open_.index, columns=open_.columns)
    cur_hi = pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()), index=open_.index, columns=open_.columns)
    cur_lo = pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()), index=open_.index, columns=open_.columns)
    inside = (cur_hi < prev_hi) & (cur_lo > prev_lo)
    bullish = inside & (prev_close < prev_open) & (close > open_)
    bearish = inside & (prev_close > prev_open) & (close < open_)
    return bullish.astype(float) - bearish.astype(float)


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
    return flag.astype(float)


def dark_cloud(open_, high, low, close):
    prev_open, prev_close = open_.shift(1), close.shift(1)
    midpoint = (prev_open + prev_close) / 2.0
    flag = (
        (prev_close > prev_open) & (close < open_) &
        (open_ >= prev_close) & (close < midpoint) & (close > prev_open)
    )
    return -flag.astype(float)


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
    return flag.astype(float)


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
    return -flag.astype(float)


def three_white_soldiers(open_, high, low, close):
    bull0 = close > open_; bull1 = close.shift(1) > open_.shift(1); bull2 = close.shift(2) > open_.shift(2)
    rising = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
    opens_inside = (
        (open_ >= open_.shift(1)) & (open_ <= close.shift(1)) &
        (open_.shift(1) >= open_.shift(2)) & (open_.shift(1) <= close.shift(2))
    )
    return (bull0 & bull1 & bull2 & rising & opens_inside).astype(float)


def three_black_crows(open_, high, low, close):
    bear0 = close < open_; bear1 = close.shift(1) < open_.shift(1); bear2 = close.shift(2) < open_.shift(2)
    falling = (close < close.shift(1)) & (close.shift(1) < close.shift(2))
    opens_inside = (
        (open_ <= open_.shift(1)) & (open_ >= close.shift(1)) &
        (open_.shift(1) <= open_.shift(2)) & (open_.shift(1) >= close.shift(2))
    )
    return -(bear0 & bear1 & bear2 & falling & opens_inside).astype(float)


def tweezer_top(open_, high, low, close):
    scale = pd.concat([high.abs(), high.shift(1).abs()], axis=0, keys=["a","b"]).groupby(level=1).max()
    same_high = (high-high.shift(1)).abs() <= 1e-4*scale.clip(lower=1.0)
    reversal = (close.shift(1) > open_.shift(1)) & (close < open_)
    return -(same_high & reversal).astype(float)


def tweezer_bottom(open_, high, low, close):
    scale = pd.concat([low.abs(), low.shift(1).abs()], axis=0, keys=["a","b"]).groupby(level=1).max()
    same_low = (low-low.shift(1)).abs() <= 1e-4*scale.clip(lower=1.0)
    reversal = (close.shift(1) < open_.shift(1)) & (close > open_)
    return (same_low & reversal).astype(float)


for _name, _fn, _desc, _bars in [
    ("cdl_dragonfly_doji", dragonfly_doji, "Dragonfly-doji raw geometry.", 1),
    ("cdl_gravestone_doji", gravestone_doji, "Gravestone-doji raw geometry.", 1),
    ("cdl_hanging_man", hanging_man, "Raw hanging-man geometry; trend context is separate.", 1),
    ("cdl_harami", harami, "Bullish +1 / bearish -1 harami body pattern.", 2),
    ("cdl_harami_cross", harami_cross, "Harami whose current candle is a doji.", 2),
    ("cdl_piercing", piercing, "Two-bar bullish piercing geometry.", 2),
    ("cdl_dark_cloud_cover", dark_cloud, "Two-bar bearish dark-cloud-cover geometry.", 2),
    ("cdl_morning_star", morning_star, "Three-bar bullish morning-star geometry.", 3),
    ("cdl_evening_star", evening_star, "Three-bar bearish evening-star geometry.", 3),
    ("cdl_three_white_soldiers", three_white_soldiers, "Three consecutive rising bullish candles.", 3),
    ("cdl_three_black_crows", three_black_crows, "Three consecutive falling bearish candles.", 3),
    ("cdl_tweezer_top", tweezer_top, "Two-bar equal-high bearish reversal geometry.", 2),
    ("cdl_tweezer_bottom", tweezer_bottom, "Two-bar equal-low bullish reversal geometry.", 2),
]:
    _register(_name, _fn, _desc, bars=_bars)
