# -*- coding: utf-8 -*-
"""Unified adaptive Japanese-candlestick pattern engine.

This is a FactorEngine semantic definition, not a byte-for-byte TA-Lib clone.
Thresholds are relative to *prior* rolling body/range statistics and therefore
remain causal and tunable. Common native ``cdl_*`` operators may coexist as
backward-compatible recipes over this engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _pi(v, name, minimum=1):
    if isinstance(v, bool):
        raise ValueError(f"{name} must be integer")
    v = int(v)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return v


def _near(a, b, tol):
    return (a - b).abs() <= tol


def _shift(x, n):
    return x.shift(int(n))


def _engine(o, h, l, c, pattern, body_window, shadow_window, penetration):
    bw = _pi(body_window, "body_window", 2)
    sw = _pi(shadow_window, "shadow_window", 2)
    pen = float(penetration)
    if not 0 <= pen <= 1:
        raise ValueError("penetration must be in [0,1]")
    p = str(pattern).strip().lower().removeprefix("cdl_")
    body = (c - o).abs()
    rng = (h - l).abs()
    upper = h - pd.DataFrame(
        np.maximum(o.to_numpy(float), c.to_numpy(float)),
        index=o.index,
        columns=o.columns,
    )
    lower = pd.DataFrame(
        np.minimum(o.to_numpy(float), c.to_numpy(float)),
        index=o.index,
        columns=o.columns,
    ) - l
    bavg = body.shift(1).rolling(bw, min_periods=bw).mean()
    ravg = rng.shift(1).rolling(sw, min_periods=sw).mean()
    tol = 0.1 * ravg
    bull = c > o
    bear = c < o
    doji = body <= 0.1 * bavg
    long_body = body >= 1.3 * bavg
    short_body = body <= 0.6 * bavg
    long_upper = upper >= 0.8 * ravg
    long_lower = lower >= 0.8 * ravg
    po, pc, ph, pl = _shift(o, 1), _shift(c, 1), _shift(h, 1), _shift(l, 1)
    pbull = pc > po
    pbear = pc < po
    pbody = (pc - po).abs()
    sign = pd.DataFrame(0.0, index=o.index, columns=o.columns)

    def emit(mask, direction):
        if np.isscalar(direction):
            return sign.mask(mask, float(direction))
        if isinstance(direction, pd.DataFrame):
            values = direction.reindex(index=sign.index, columns=sign.columns)
        else:
            values = pd.DataFrame(direction, index=sign.index, columns=sign.columns)
        return sign.where(~mask, values.astype(float))

    if p == "long_line":
        return emit(long_body, np.where(bull, 1, -1))
    if p == "short_line":
        return emit(short_body, np.where(bull, 1, -1))
    if p == "high_wave":
        return emit(short_body & long_upper & long_lower, np.where(bull, 1, -1))
    if p == "long_legged_doji":
        return emit(doji & long_upper & long_lower, 1)
    if p == "rickshaw_man":
        return emit(
            doji
            & long_upper
            & long_lower
            & (((o + c) / 2 - (h + l) / 2).abs() <= 0.15 * rng),
            1,
        )
    if p == "takuri":
        return emit(doji & (lower >= 2.5 * bavg) & (upper <= 0.3 * bavg), 1)
    if p == "belt_hold":
        return emit(
            long_body & ((bull & (o - l <= 0.1 * rng)) | (bear & (h - o <= 0.1 * rng))),
            np.where(bull, 1, -1),
        )
    if p == "closing_marubozu":
        return emit(
            long_body & ((bull & (h - c <= 0.05 * rng)) | (bear & (c - l <= 0.05 * rng))),
            np.where(bull, 1, -1),
        )
    if p == "homing_pigeon":
        return emit(pbear & bear & (o < po) & (c > pc), 1)
    if p == "matching_low":
        return emit(pbear & bear & _near(c, pc, tol), 1)
    if p == "counterattack":
        return emit(((pbear & bull) | (pbull & bear)) & _near(c, pc, tol), np.where(bull, 1, -1))
    if p == "separating_lines":
        return emit(
            ((pbear & bull) | (pbull & bear)) & _near(o, po, tol) & long_body,
            np.where(bull, 1, -1),
        )
    if p in {"on_neck", "in_neck", "thrusting"}:
        prev_mid = (po + pc) / 2
        bull2 = pbear & bull & (o < pc)
        if p == "on_neck":
            mask = bull2 & _near(c, pc, tol)
        elif p == "in_neck":
            mask = bull2 & (c > pc) & (c < pc + 0.25 * pbody)
        else:
            mask = bull2 & (c > pc + 0.25 * pbody) & (c < prev_mid)
        return emit(mask, -1)
    if p == "doji_star":
        return emit(doji & ((pbear & (h < pc)) | (pbull & (l > pc))), np.where(pbear, 1, -1))

    o2, c2, h2, l2 = _shift(o, 2), _shift(c, 2), _shift(h, 2), _shift(l, 2)
    bull2 = c2 > o2
    bear2 = c2 < o2
    body2 = (c2 - o2).abs()
    doji1 = _shift(doji, 1)
    doji2 = _shift(doji, 2)
    engulf = (
        bull2 & pbear & bull & (o <= pc) & (c >= po)
        | bear2 & pbull & bear & (o >= pc) & (c <= po)
    )
    current_max = pd.DataFrame(
        np.maximum(o.to_numpy(float), c.to_numpy(float)), index=o.index, columns=o.columns
    )
    previous_max = pd.DataFrame(
        np.maximum(o2.to_numpy(float), c2.to_numpy(float)), index=o.index, columns=o.columns
    )
    current_min = pd.DataFrame(
        np.minimum(o.to_numpy(float), c.to_numpy(float)), index=o.index, columns=o.columns
    )
    previous_min = pd.DataFrame(
        np.minimum(o2.to_numpy(float), c2.to_numpy(float)), index=o.index, columns=o.columns
    )
    harami = (bull2 | bear2) & (current_max <= previous_max) & (current_min >= previous_min)
    if p == "3_inside":
        return emit(_shift(harami, 1) & ((bear2 & (c > o2)) | (bull2 & (c < o2))), np.where(c > o2, 1, -1))
    if p == "3_outside":
        return emit(_shift(engulf, 1) & ((bear2 & (c > pc)) | (bull2 & (c < pc))), np.where(c > pc, 1, -1))
    if p == "tristar":
        return emit(doji & doji1 & doji2, np.where(c > _shift(c, 2), 1, -1))
    if p == "abandoned_baby":
        bullmask = bear2 & doji1 & (_shift(h, 1) < l2) & bull & (l > _shift(h, 1)) & (c > o2 - body2 * pen)
        bearmask = bull2 & doji1 & (_shift(l, 1) > h2) & bear & (h < _shift(l, 1)) & (c < o2 + body2 * pen)
        return emit(bullmask | bearmask, np.where(bullmask, 1, -1))
    if p == "stick_sandwich":
        return emit(bear2 & pbull & bear & _near(c, c2, tol), 1)
    if p == "identical_3_crows":
        return emit(bear & pbear & bear2 & _near(o, pc, tol) & _near(po, c2, tol), -1)
    if p == "advance_block":
        return emit(bull & pbull & bull2 & (body < _shift(body, 1)) & (_shift(body, 1) < _shift(body, 2)), -1)
    if p == "stalled_pattern":
        return emit(bull & pbull & bull2 & short_body & (_shift(body, 1) >= _shift(body, 2)), -1)
    if p == "3_stars_south":
        return emit(
            bear & pbear & bear2 & (l > pl) & (pl > l2) & (body < _shift(body, 1)) & (_shift(body, 1) < _shift(body, 2)),
            1,
        )
    if p == "unique_3_river":
        return emit(bear2 & pbear & bull & (l2 > pl) & (l > pl) & short_body, 1)
    if p == "upside_gap_2_crows":
        return emit(bull2 & pbear & bear & (pl > h2) & (o > po) & (c < c2), -1)
    if p == "tasuki_gap":
        up = bull2 & pbull & bear & (_shift(l, 1) > h2) & (o > pc) & (c < _shift(o, 1)) & (c > h2)
        dn = bear2 & pbear & bull & (_shift(h, 1) < l2) & (o < pc) & (c > _shift(o, 1)) & (c < l2)
        return emit(up | dn, np.where(up, 1, -1))
    if p == "xside_gap_3_methods":
        up = bull2 & pbull & bear & (_shift(l, 1) > h2) & (o > c2) & (c < c2)
        dn = bear2 & pbear & bull & (_shift(h, 1) < l2) & (o < c2) & (c > c2)
        return emit(up | dn, np.where(up, 1, -1))

    o3, c3 = _shift(o, 3), _shift(c, 3)
    bull3 = c3 > o3
    bear3 = c3 < o3
    if p == "3_line_strike":
        up = bull3 & _shift(bull, 2) & pbull & bear & (c < o3) & (o > pc)
        dn = bear3 & _shift(bear, 2) & pbear & bull & (c > o3) & (o < pc)
        return emit(up | dn, np.where(up, -1, 1))
    if p == "rise_fall_3_methods":
        o4, c4 = _shift(o, 4), _shift(c, 4)
        bull4 = c4 > o4
        bear4 = c4 < o4
        inside3 = (
            (_shift(h, 3) < _shift(h, 4))
            & (_shift(l, 3) > _shift(l, 4))
            & (_shift(h, 2) < _shift(h, 4))
            & (_shift(l, 2) > _shift(l, 4))
            & (_shift(h, 1) < _shift(h, 4))
            & (_shift(l, 1) > _shift(l, 4))
        )
        up = bull4 & inside3 & bull & (c > c4)
        dn = bear4 & inside3 & bear & (c < c4)
        return emit(up | dn, np.where(up, 1, -1))
    if p == "ladder_bottom":
        o4, c4 = _shift(o, 4), _shift(c, 4)
        return emit((c4 < o4) & bear3 & bear2 & pbear & bull & (c > po), 1)
    if p == "breakaway":
        o4, c4 = _shift(o, 4), _shift(c, 4)
        up = (c4 < o4) & (_shift(h, 3) < _shift(l, 4)) & bull & (c > _shift(c, 2))
        dn = (c4 > o4) & (_shift(l, 3) > _shift(h, 4)) & bear & (c < _shift(c, 2))
        return emit(up | dn, np.where(up, 1, -1))
    if p == "mat_hold":
        o4, c4 = _shift(o, 4), _shift(c, 4)
        return emit((c4 > o4) & (_shift(l, 3) > c4) & bull & (c > c4), 1)
    raise ValueError(f"unsupported candlestick pattern: {pattern!r}")


class CandlestickPatternEngine(SeriesOperator):
    metadata = OperatorMetadata(
        name="candlestick_pattern",
        category="candle_pattern",
        description="Adaptive bounded Japanese-candlestick pattern engine.",
        param_names=[
            "open",
            "high",
            "low",
            "close",
            "pattern",
            "body_window",
            "shadow_window",
            "penetration",
        ],
        return_type="series",
        tags=["pit_safe", "causal", "bounded_history", "production_extension"],
    )

    def _calculate_series(
        self,
        open,
        high,
        low,
        close,
        pattern,
        body_window=10,
        shadow_window=10,
        penetration=0.3,
        **kwargs,
    ):
        return _engine(open, high, low, close, pattern, body_window, shadow_window, penetration)


register_operator(
    name="candlestick_pattern",
    category="candle_pattern",
    business_category="technical_extension",
    canonical="candlestick_pattern",
    source="candle_pattern_engine_v2",
    backend="pandas_numpy",
    status="production",
)(CandlestickPatternEngine)

import cleaned_operators.operator_surface as _surface

_surface.EXTENDED_ONLY_CANONICALS = frozenset(
    set(_surface.EXTENDED_ONLY_CANONICALS) | {"candlestick_pattern"}
)
