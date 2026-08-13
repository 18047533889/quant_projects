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

from cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.price_volume.candle_geometry_v2 import _validate_ohlc


def _pi(v, name, minimum=1):
    if isinstance(v, bool):
        raise ValueError(f"{name} must be integer")
    v = float(v)
    if not np.isfinite(v) or v != float(int(v)):
        raise ValueError(f"{name} must be an integer")
    v = int(v)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return v


def _near(a, b, tol):
    return (a - b).abs() <= tol


def _shift(x, n):
    return x.shift(int(n))


def _sign_direction(c, o):
    """Direction of close vs open; 0 for a doji (close==open), review R4-12."""
    return np.sign(c.to_numpy(float) - o.to_numpy(float))


# Pattern -> params that actually affect the output.  Varying an inactive
# parameter (e.g. ``penetration`` for a pattern that never reads it) produces
# identical outputs and pollutes the search space (review P0-10).  ``pattern``
# itself is always active.
_CANDLE_PARAM_DEPS: dict[str, frozenset[str]] = {
    "long_line": frozenset({"body_window"}),
    "short_line": frozenset({"body_window"}),
    "high_wave": frozenset({"body_window", "shadow_window"}),
    "long_legged_doji": frozenset({"body_window", "shadow_window"}),
    "rickshaw_man": frozenset({"body_window", "shadow_window"}),
    "takuri": frozenset({"body_window"}),
    "belt_hold": frozenset({"body_window", "shadow_window"}),
    "closing_marubozu": frozenset({"body_window", "shadow_window"}),
    "homing_pigeon": frozenset(),
    "matching_low": frozenset({"shadow_window"}),
    "counterattack": frozenset({"shadow_window"}),
    "separating_lines": frozenset({"body_window", "shadow_window"}),
    "on_neck": frozenset(),
    "in_neck": frozenset(),
    "thrusting": frozenset(),
    "doji_star": frozenset({"body_window"}),
    "3_inside": frozenset(),
    "3_outside": frozenset(),
    "tristar": frozenset({"body_window"}),
    "abandoned_baby": frozenset({"body_window", "penetration"}),
    "stick_sandwich": frozenset({"shadow_window"}),
    "identical_3_crows": frozenset({"shadow_window"}),
    "advance_block": frozenset({"body_window"}),
    "stalled_pattern": frozenset({"body_window"}),
    "3_stars_south": frozenset({"body_window"}),
    "unique_3_river": frozenset({"body_window"}),
    "upside_gap_2_crows": frozenset(),
    "tasuki_gap": frozenset(),
    "xside_gap_3_methods": frozenset(),
    "3_line_strike": frozenset(),
    "rise_fall_3_methods": frozenset(),
    "ladder_bottom": frozenset(),
    "breakaway": frozenset(),
    "mat_hold": frozenset(),
    "2_crows": frozenset(),
    "evening_doji_star": frozenset({"penetration"}),
}


def candlestick_active_params(pattern: str) -> frozenset[str]:
    """Params (beyond ``pattern``) that actually affect output for a pattern.

    Search / grammar generators should only vary the returned params for the
    selected ``pattern``; every other candidate is a dead knob (review P0-10).
    """
    p = str(pattern).strip().lower().removeprefix("cdl_")
    deps = _CANDLE_PARAM_DEPS.get(p)
    if deps is None:
        raise ValueError(f"unsupported candlestick pattern: {pattern!r}")
    return frozenset({"pattern"}) | deps


def _candle_active_param_patterns(param: str) -> tuple[str, ...]:
    """Pattern names (with and without the ``cdl_`` prefix) for which ``param``
    actually affects the output (ParamSpec.active_when, review R4-91)."""
    out = {p for p, deps in _CANDLE_PARAM_DEPS.items() if param in deps}
    return tuple(sorted(out | {"cdl_" + p for p in out}))


_CANDLE_PATTERNS_ALL = tuple(sorted(set(_CANDLE_PARAM_DEPS) | {"cdl_" + p for p in _CANDLE_PARAM_DEPS}))
_CANDLE_BODY_ACTIVE = _candle_active_param_patterns("body_window")
_CANDLE_SHADOW_ACTIVE = _candle_active_param_patterns("shadow_window")
_CANDLE_PENETRATION_ACTIVE = _candle_active_param_patterns("penetration")


def _engine(o, h, l, c, pattern, body_window, shadow_window, penetration):
    bw = _pi(body_window, "body_window", 2)
    sw = _pi(shadow_window, "shadow_window", 2)
    pen = float(penetration)
    if not 0 <= pen <= 1:
        raise ValueError("penetration must be in [0,1]")
    p = str(pattern).strip().lower().removeprefix("cdl_")
    if p not in _CANDLE_PARAM_DEPS:
        raise ValueError(f"unsupported candlestick pattern: {pattern!r}")
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
    # P0-08 warmup/suspension validity: a pattern must emit NaN — not 0, which
    # would be read as "confirmed no pattern" — until every OHLC history /
    # body baseline / range baseline it references is available.
    cur = o.notna() & h.notna() & l.notna() & c.notna()
    # Audit item 4 (A-share critical): a zero-amplitude 一字板 bar (O=H=L=C) has
    # body=range=upper=lower=0, so every doji subtype would trivially fire.
    # Gate it out of every pattern (emit NaN "cannot judge") on a tiny relative
    # tick so a zero-range bar is never classified as any pattern.
    cur = cur & (rng > 1e-6 * c.abs())
    # round-3 audit item 15: unified OHLC structural invariant — a bar whose
    # High < max(O,C) / Low > min(O,C) / High < Low, or with a non-positive
    # price, cannot be classified as ANY pattern (fail closed to NaN).  Applied
    # to the current bar and every history bar each pattern reads.
    valid = _validate_ohlc(o, h, l, c)
    cur = cur & valid
    prev1 = po.notna() & pc.notna() & ph.notna() & pl.notna() & valid.shift(1)
    o2, c2, h2, l2 = _shift(o, 2), _shift(c, 2), _shift(h, 2), _shift(l, 2)
    o3, c3, h3, l3 = _shift(o, 3), _shift(c, 3), _shift(h, 3), _shift(l, 3)
    o4, c4, h4, l4 = _shift(o, 4), _shift(c, 4), _shift(h, 4), _shift(l, 4)
    prev2 = prev1 & o2.notna() & c2.notna() & h2.notna() & l2.notna() & valid.shift(2)
    prev3 = prev2 & o3.notna() & c3.notna() & h3.notna() & l3.notna() & valid.shift(3)
    prev4 = prev3 & o4.notna() & c4.notna() & h4.notna() & l4.notna() & valid.shift(4)
    has_b = bavg.notna()
    has_r = ravg.notna()
    has_b1 = bavg.shift(1).notna()
    has_b2 = bavg.shift(2).notna()
    sign = pd.DataFrame(0.0, index=o.index, columns=o.columns)

    def emit(mask, direction, valid):
        base = sign.where(valid, np.nan)
        if np.isscalar(direction):
            return base.mask(mask, float(direction))
        if isinstance(direction, pd.DataFrame):
            values = direction.reindex(index=sign.index, columns=sign.columns)
        else:
            values = pd.DataFrame(direction, index=sign.index, columns=sign.columns)
        return base.where(~mask, values.astype(float))

    if p == "long_line":
        return emit(long_body, _sign_direction(c, o), cur & has_b)
    if p == "short_line":
        return emit(short_body, _sign_direction(c, o), cur & has_b)
    if p == "high_wave":
        return emit(short_body & long_upper & long_lower, _sign_direction(c, o), cur & has_b & has_r)
    if p == "long_legged_doji":
        # Directionless pattern (review R4-12): no bull/bear connotation.  The
        # output is the EventBool (1 = present, 0 = not, NaN = cannot judge);
        # the direction component is 0/neutral and must not be read as bullish.
        return emit(doji & long_upper & long_lower, 1.0, cur & has_b & has_r)
    if p == "rickshaw_man":
        return emit(
            doji
            & long_upper
            & long_lower
            & (((o + c) / 2 - (h + l) / 2).abs() <= 0.15 * rng),
            1.0,
            cur & has_b & has_r,
        )
    if p == "takuri":
        return emit(doji & (lower >= 2.5 * bavg) & (upper <= 0.3 * bavg), 1, cur & has_b)
    if p == "belt_hold":
        return emit(
            long_body & ((bull & (o - l <= 0.1 * rng)) | (bear & (h - o <= 0.1 * rng))),
            _sign_direction(c, o),
            cur & has_b,
        )
    if p == "closing_marubozu":
        return emit(
            long_body & ((bull & (h - c <= 0.05 * rng)) | (bear & (c - l <= 0.05 * rng))),
            _sign_direction(c, o),
            cur & has_b,
        )
    if p == "homing_pigeon":
        return emit(pbear & bear & (o < po) & (c > pc), 1, cur & prev1)
    if p == "matching_low":
        return emit(pbear & bear & _near(c, pc, tol), 1, cur & prev1 & has_r)
    if p == "counterattack":
        return emit(((pbear & bull) | (pbull & bear)) & _near(c, pc, tol), _sign_direction(c, o), cur & prev1 & has_r)
    if p == "separating_lines":
        return emit(
            ((pbear & bull) | (pbull & bear)) & _near(o, po, tol) & long_body,
            _sign_direction(c, o),
            cur & prev1 & has_r & has_b,
        )
    if p in {"on_neck", "in_neck", "thrusting"}:
        prev_mid = np.where(2 != 0, (po + pc) / 2, np.nan)
        bull2 = pbear & bull & (o < pc)
        if p == "on_neck":
            mask = bull2 & _near(c, pc, tol)
        elif p == "in_neck":
            mask = bull2 & (c > pc) & (c < pc + 0.25 * pbody)
        else:
            mask = bull2 & (c > pc + 0.25 * pbody) & (c < prev_mid)
        return emit(mask, -1, cur & prev1)
    if p == "doji_star":
        return emit(doji & ((pbear & (h < pc)) | (pbull & (l > pc))), np.sign(pc.to_numpy(float) - po.to_numpy(float)), cur & prev1 & has_b)

    bull2 = c2 > o2
    bear2 = c2 < o2
    body2 = (c2 - o2).abs()
    doji1 = _shift(doji, 1)
    doji2 = _shift(doji, 2)
    # P0-09 rewrite — clean three-bar indexing (bar1=t-2, bar2=t-1,
    # confirmation=t).  The old code defined ``harami``/``engulf`` on a t-2 vs t
    # span and then shifted the whole mask by one, so ``3_inside`` effectively
    # compared t-1 vs t-3 and ``3_outside``'s engulf carried a spurious t-3
    # ``bull2`` term — neither matched the standard three-bar structure.
    b1_hi = pd.DataFrame(np.maximum(o2.to_numpy(float), c2.to_numpy(float)), index=o.index, columns=o.columns)
    b1_lo = pd.DataFrame(np.minimum(o2.to_numpy(float), c2.to_numpy(float)), index=o.index, columns=o.columns)
    b2_hi = pd.DataFrame(np.maximum(po.to_numpy(float), pc.to_numpy(float)), index=o.index, columns=o.columns)
    b2_lo = pd.DataFrame(np.minimum(po.to_numpy(float), pc.to_numpy(float)), index=o.index, columns=o.columns)
    # bar[t-1] strictly inside body[t-2]
    harami1 = (bull2 | bear2) & (b2_hi <= b1_hi) & (b2_lo >= b1_lo)
    # bar[t-1] engulfs body[t-2] (only bars t-2 and t-1 are read)
    engulf1 = (
        (bear2 & pbull & (po <= c2) & (pc >= o2))   # t-2 bear, t-1 bull engulfs
        | (bull2 & pbear & (po >= c2) & (pc <= o2))  # t-2 bull, t-1 bear engulfs
    )
    if p == "3_inside":
        return emit(
            harami1 & ((bear2 & (c > o2)) | (bull2 & (c < o2))),
            np.where(c > o2, 1, -1),
            cur & prev1 & prev2,
        )
    if p == "3_outside":
        return emit(
            engulf1 & ((bear2 & (c > pc)) | (bull2 & (c < pc))),
            np.where(c > pc, 1, -1),
            cur & prev1 & prev2,
        )
    if p == "tristar":
        return emit(doji & doji1 & doji2, np.where(c > _shift(c, 2), 1, -1), cur & prev2 & has_b & has_b1 & has_b2)
    if p == "abandoned_baby":
        bullmask = bear2 & doji1 & (_shift(h, 1) < l2) & bull & (l > _shift(h, 1)) & (c > o2 - body2 * pen)
        bearmask = bull2 & doji1 & (_shift(l, 1) > h2) & bear & (h < _shift(l, 1)) & (c < o2 + body2 * pen)
        return emit(bullmask | bearmask, np.where(bullmask, 1, -1), cur & prev1 & prev2 & has_b1)
    if p == "stick_sandwich":
        return emit(bear2 & pbull & bear & _near(c, c2, tol), 1, cur & prev2 & has_r)
    if p == "identical_3_crows":
        return emit(bear & pbear & bear2 & _near(o, pc, tol) & _near(po, c2, tol), -1, cur & prev1 & prev2 & has_r)
    if p == "advance_block":
        return emit(bull & pbull & bull2 & (body < _shift(body, 1)) & (_shift(body, 1) < _shift(body, 2)), -1, cur & prev2 & has_b & has_b1 & has_b2)
    if p == "stalled_pattern":
        return emit(bull & pbull & bull2 & short_body & (_shift(body, 1) >= _shift(body, 2)), -1, cur & prev2 & has_b & has_b1 & has_b2)
    if p == "3_stars_south":
        return emit(
            bear & pbear & bear2 & (l > pl) & (pl > l2) & (body < _shift(body, 1)) & (_shift(body, 1) < _shift(body, 2)),
            1,
            cur & prev1 & prev2 & has_b & has_b1 & has_b2,
        )
    if p == "unique_3_river":
        return emit(bear2 & pbear & bull & (l2 > pl) & (l > pl) & short_body, 1, cur & prev1 & prev2 & has_b)
    if p == "upside_gap_2_crows":
        return emit(bull2 & pbear & bear & (pl > h2) & (o > po) & (c < c2), -1, cur & prev1 & prev2)
    if p == "tasuki_gap":
        up = bull2 & pbull & bear & (_shift(l, 1) > h2) & (o > pc) & (c < _shift(o, 1)) & (c > h2)
        dn = bear2 & pbear & bull & (_shift(h, 1) < l2) & (o < pc) & (c > _shift(o, 1)) & (c < l2)
        return emit(up | dn, np.where(up, 1, -1), cur & prev1 & prev2)
    if p == "xside_gap_3_methods":
        up = bull2 & pbull & bear & (_shift(l, 1) > h2) & (o > c2) & (c < c2)
        dn = bear2 & pbear & bull & (_shift(h, 1) < l2) & (o < c2) & (c > c2)
        return emit(up | dn, np.where(up, 1, -1), cur & prev1 & prev2)

    bull3 = c3 > o3
    bear3 = c3 < o3
    if p == "3_line_strike":
        up = bull3 & _shift(bull, 2) & pbull & bear & (c < o3) & (o > pc)
        dn = bear3 & _shift(bear, 2) & pbear & bull & (c > o3) & (o < pc)
        return emit(up | dn, np.where(up, -1, 1), cur & prev1 & prev2 & prev3)
    if p == "rise_fall_3_methods":
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
        return emit(up | dn, np.where(up, 1, -1), cur & prev1 & prev2 & prev3 & prev4)
    if p == "ladder_bottom":
        return emit((c4 < o4) & bear3 & bear2 & pbear & bull & (c > po), 1, cur & prev1 & prev2 & prev3 & prev4)
    if p == "breakaway":
        up = (c4 < o4) & (_shift(h, 3) < _shift(l, 4)) & bull & (c > _shift(c, 2))
        dn = (c4 > o4) & (_shift(l, 3) > _shift(h, 4)) & bear & (c < _shift(c, 2))
        return emit(up | dn, np.where(up, 1, -1), cur & prev2 & prev3 & prev4)
    if p == "mat_hold":
        return emit((c4 > o4) & (_shift(l, 3) > c4) & bull & (c > c4), 1, cur & prev3 & prev4)
    raise ValueError(f"unsupported candlestick pattern: {pattern!r}")


class CandlestickPatternEngine(SeriesOperator):
    """Adaptive bounded Japanese-candlestick pattern engine.

    Output contract (review R4-12): directional patterns emit ``+1`` (bullish) /
    ``-1`` (bearish); directionless doji-family patterns emit ``+1`` as the
    *EventBool* with a neutral direction component (0) — consume them as
    event/magnitude, not as a bullish call.  Warmup / missing-history rows are
    NaN, never 0.  Custom adaptive patterns (``tristar``/``breakaway``/
    ``advance_block`` …) are *adaptive* re-readings of the classic names, not
    byte-for-byte TA-Lib definitions (semantic_family=adaptive_custom, R4-92).
    """

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
        tags=[
            "pit_safe", "causal", "bounded_history", "production_extension",
            "semantic_family:adaptive_custom",  # R4-92: not byte-for-byte TA-Lib
        ],
        param_specs={
            "pattern": ParamSpec(dtype=str, searchable=True),
            # R4-91: each window/penetration knob only enters the search/GP
            # grammar for the patterns that actually read it (dead knobs are
            # excluded via active_when).
            # round-3 cross-cutting: body/shadow windows are lookback-horizon
            # scale dimensions (engine convention: window -> HORIZON, full search).
            "body_window": ParamSpec(dtype=int, min=2, active_when=("pattern", _CANDLE_BODY_ACTIVE), param_role=ParamRole.HORIZON),
            "shadow_window": ParamSpec(dtype=int, min=2, active_when=("pattern", _CANDLE_SHADOW_ACTIVE), param_role=ParamRole.HORIZON),
            "penetration": ParamSpec(dtype=float, min=0.0, max=1.0, active_when=("pattern", _CANDLE_PENETRATION_ACTIVE)),
        },
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

_surface.extend_extended_only({"candlestick_pattern"})
