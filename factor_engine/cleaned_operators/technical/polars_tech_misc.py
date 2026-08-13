# -*- coding: utf-8 -*-
"""Native Polars backends for miscellaneous technical indicators.

Covers Keltner channels, Donchian channels, Bollinger percent-B/width,
efficiency ratio, choppiness index, and the recursive Supertrend / PSAR
state machines (implemented with NumPy kernels over polars column arrays).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})
_EPS = 1e-12


def _pi(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _pf(value, name: str, minimum: float | None = None) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _cols(*frames: pl.DataFrame) -> list[str]:
    out = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        out = [c for c in out if c in frame.columns]
    return out


def _make(base: pl.DataFrame, cols: list[str], values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({c: values[:, i] for i, c in enumerate(cols)})


def _one(frame: pl.DataFrame, column: str, expr: pl.Expr) -> pl.Series:
    return frame.select(expr.alias(column)).to_series()


def _result(base: pl.DataFrame, values: dict[str, pl.Series]) -> pl.DataFrame:
    return base.with_columns([series.alias(name) for name, series in values.items()])


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den != 0).then(num / den).otherwise(None)


def _ema(expr: pl.Expr, span: int) -> pl.Expr:
    return expr.fill_nan(None).ewm_mean(span=span, adjust=False, min_samples=span).fill_null(strategy="forward")


def _wilder(expr: pl.Expr, window: int) -> pl.Expr:
    return expr.fill_nan(None).ewm_mean(alpha=1.0 / window, adjust=False, min_samples=window).fill_null(strategy="forward")


def _tr_propagate(h: pl.Expr, l: pl.Expr, c: pl.Expr) -> pl.Expr:
    # pandas np.maximum.reduce propagates NaN on any missing component.
    previous = c.shift(1)
    hl = h - l
    hc = (h - previous).abs()
    lc = (l - previous).abs()
    max1 = (hl + hc + (hl - hc).abs()) / 2.0
    return (max1 + lc + (max1 - lc).abs()) / 2.0


def _tr_ignore(h: pl.Expr, l: pl.Expr, c: pl.Expr) -> pl.Expr:
    # technical_extensions _true_range: concat(...).groupby(level=1).max()
    # ignores NaN row-wise (first bar uses high-low when prev close is missing).
    previous = c.shift(1)
    return pl.max_horizontal(
        (h - l).fill_nan(None),
        (h - previous).abs().fill_nan(None),
        (l - previous).abs().fill_nan(None),
    )


def _rolling_max_1d(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        ok = np.isfinite(seg)
        if int(ok.sum() < w:
            continue
        out[t] = float(np.max(seg[ok]))
    return out


def _rolling_min_1d(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        ok = np.isfinite(seg)
        if int(ok.sum() < w:
            continue
        out[t] = float(np.min(seg[ok]))
    return out


def _rolling_sum_1d(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        ok = np.isfinite(seg)
        if int(ok.sum() < w:
            continue
        out[t] = float(np.sum(seg[ok]))
    return out


# ---------------------------------------------------------------------------
# Keltner channels (indicators_v2 reference)
# ---------------------------------------------------------------------------


def KeltnerMid(close, ema_window):
    w = _pi(ema_window, "ema_window")
    values = {}
    for c in _cols(close):
        values[c] = _one(close, c, _ema(pl.col(c), w))
    return _result(close, values)


def KeltnerUpper(high, low, close, ema_window, atr_window, multiplier):
    ew = _pi(ema_window, "ema_window")
    aw = _pi(atr_window, "atr_window")
    mult = _pf(multiplier, "multiplier", 0)
    values = {}
    for c in _cols(high, low, close):
        frame = pl.DataFrame({"high": high[c], "low": low[c], "close": close[c]})
        mid = _ema(pl.col("close"), ew)
        atr = _wilder(_tr_propagate(pl.col("high"), pl.col("low"), pl.col("close")), aw)
        values[c] = _one(frame, c, mid + mult * atr)
    return _result(close, values)


def KeltnerLower(high, low, close, ema_window, atr_window, multiplier):
    ew = _pi(ema_window, "ema_window")
    aw = _pi(atr_window, "atr_window")
    mult = _pf(multiplier, "multiplier", 0)
    values = {}
    for c in _cols(high, low, close):
        frame = pl.DataFrame({"high": high[c], "low": low[c], "close": close[c]})
        mid = _ema(pl.col("close"), ew)
        atr = _wilder(_tr_propagate(pl.col("high"), pl.col("low"), pl.col("close")), aw)
        values[c] = _one(frame, c, mid - mult * atr)
    return _result(close, values)


def KeltnerPosition(high, low, close, ema_window, atr_window, multiplier):
    upper = KeltnerUpper(high, low, close, ema_window, atr_window, multiplier)
    lower = KeltnerLower(high, low, close, ema_window, atr_window, multiplier)
    values = {}
    for c in _cols(close, upper, lower):
        frame = pl.DataFrame({"close": close[c], "upper": upper[c], "lower": lower[c]})
        values[c] = _one(frame, c, _safe_div(pl.col("close") - pl.col("lower"), pl.col("upper") - pl.col("lower")))
    return _result(close, values)


# ---------------------------------------------------------------------------
# Donchian channels (technical_extensions reference)
# ---------------------------------------------------------------------------


def donchian_upper(high, window):
    w = _pi(window, "window")
    cols = _cols(high)
    return _result(high, {c: _one(high, c, pl.col(c).shift(1).rolling_max(w, min_samples=w)) for c in cols})


def donchian_lower(low, window):
    w = _pi(window, "window")
    cols = _cols(low)
    return _result(low, {c: _one(low, c, pl.col(c).shift(1).rolling_min(w, min_samples=w)) for c in cols})


def donchian_mid(high, low, window):
    upper = donchian_upper(high, window)
    lower = donchian_lower(low, window)
    cols = _cols(upper, lower)
    return _result(high, {c: (upper[c] + lower[c]) / 2.0 for c in cols})


def donchian_position(close, high, low, window):
    upper = donchian_upper(high, window)
    lower = donchian_lower(low, window)
    values = {}
    for c in _cols(close, upper, lower):
        frame = pl.DataFrame({"close": close[c], "upper": upper[c], "lower": lower[c]})
        values[c] = _one(frame, c, _safe_div(pl.col("close") - pl.col("lower"), pl.col("upper") - pl.col("lower")))
    return _result(close, values)


# ---------------------------------------------------------------------------
# Bollinger percent-B / width
# ---------------------------------------------------------------------------


def bollinger_pct_b(x, window, std_dev):
    w = _pi(window, "window", 2)
    k = _pf(std_dev, "std_dev", 0)
    values = {}
    for c in _cols(x):
        col = pl.col(c)
        mean = col.rolling_mean(w, min_samples=w)
        std = col.rolling_std(w, min_samples=w)
        lower, upper = mean - k * std, mean + k * std
        values[c] = _one(x, c, _safe_div(col - lower, upper - lower))
    return _result(x, values)


def bollinger_width(x, window, std_dev):
    w = _pi(window, "window", 2)
    k = _pf(std_dev, "std_dev", 0)
    values = {}
    for c in _cols(x):
        col = pl.col(c)
        mean = col.rolling_mean(w, min_samples=w)
        std = col.rolling_std(w, min_samples=w)
        values[c] = _one(x, c, _safe_div(2.0 * k * std, mean.abs()))
    return _result(x, values)


# ---------------------------------------------------------------------------
# efficiency ratio / choppiness index
# ---------------------------------------------------------------------------


def efficiency_ratio(close, window):
    w = _pi(window, "window")
    values = {}
    for c in _cols(close):
        col = pl.col(c)
        change = (col - col.shift(w)).abs()
        path = col.diff().abs().rolling_sum(w, min_samples=w)
        values[c] = _one(close, c, _safe_div(change, path))
    return _result(close, values)


def choppiness_index(high, low, close, window):
    w = _pi(window, "window", 2)
    values = {}
    for c in _cols(high, low, close):
        frame = pl.DataFrame({"high": high[c], "low": low[c], "close": close[c]})
        tr = _tr_ignore(pl.col("high"), pl.col("low"), pl.col("close"))
        num = tr.rolling_sum(w, min_samples=w)
        den = pl.col("high").rolling_max(w, min_samples=w) - pl.col("low").rolling_min(w, min_samples=w)
        ratio = _safe_div(num, den).clip(lower_bound=_EPS)
        values[c] = _one(frame, c, 100.0 * ratio.log10() / np.log10(float(w)))
    return _result(close, values)


# ---------------------------------------------------------------------------
# Supertrend / PSAR recursive state machines
# ---------------------------------------------------------------------------


def _supertrend_1d(high, low, close, atr, mult):
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    basic_u = (high + low) / 2.0 + mult * atr
    basic_l = (high + low) / 2.0 - mult * atr
    final_u = basic_u.copy()
    final_l = basic_l.copy()
    trend = np.ones(n, dtype=int)
    for t in range(1, n):
        # break + rewarm: a bar is state-valid only when high/low/close AND the
        # derived ATR-based bands are ALL finite — otherwise the recursion breaks
        # and re-warms over the next contiguous valid segment (round-11 P0).
        if not (np.isfinite(high[t]) and np.isfinite(low[t]) and np.isfinite(close[t])
                and np.isfinite(basic_u[t]) and np.isfinite(basic_l[t])):
            trend[t] = 0  # sentinel: state broken, next valid bar re-seeds
            continue
        if trend[t - 1] == 0:
            # Re-seed after a gap from the current bar's own valid bands.
            trend[t] = 1
            final_u[t] = basic_u[t]
            final_l[t] = basic_l[t]
            out[t] = final_l[t]
            continue
        if np.isfinite(final_u[t - 1]) and (basic_u[t] >= final_u[t - 1] and close[t - 1] <= final_u[t - 1]):
            final_u[t] = final_u[t - 1]
        if np.isfinite(final_l[t - 1]) and (basic_l[t] <= final_l[t - 1] and close[t - 1] >= final_l[t - 1]):
            final_l[t] = final_l[t - 1]
        if trend[t - 1] > 0 and close[t] < final_l[t]:
            trend[t] = -1
        elif trend[t - 1] < 0 and close[t] > final_u[t]:
            trend[t] = 1
        else:
            trend[t] = trend[t - 1]
        out[t] = final_l[t] if trend[t] > 0 else final_u[t]
    return out


def Supertrend(high, low, close, atr_window, multiplier):
    w = _pi(atr_window, "atr_window", 2)
    mult = _pf(multiplier, "multiplier", 0)
    if mult <= 0:
        raise ValueError("multiplier must be > 0 (Supertrend multiplier=0 collapses upper/lower to the midpoint)")
    cols = _cols(high, low, close)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        h, l, cl = high[c].to_numpy(), low[c].to_numpy(), close[c].to_numpy()
        frame = pl.DataFrame({"high": high[c], "low": low[c], "close": close[c]})
        # Wilder ATR via the polars ewm path (proven to match pandas ewm NaN semantics)
        atr = frame.select(_wilder(_tr_propagate(pl.col("high"), pl.col("low"), pl.col("close")), w).alias("atr"))["atr"].to_numpy()
        out[:, i] = _supertrend_1d(h, l, cl, atr, mult)
    return _make(close, cols, out)


def SupertrendDirection(high, low, close, atr_window, multiplier):
    st = Supertrend(high, low, close, atr_window, multiplier)
    cols = _cols(st)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cv = close[c].to_numpy()
        sv = st[c].to_numpy()
        direction = np.where(cv >= sv, 1.0, -1.0)
        direction[~np.isfinite(sv)] = np.nan
        out[:, i] = direction
    return _make(close, cols, out)


def _psar_1d(high, low, af0, afmax):
    n = len(high)
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return out
    bull = True
    sar = low[0]
    ep = high[0]
    af = af0
    live = True
    for t in range(1, n):
        if not (np.isfinite(high[t]) and np.isfinite(low[t])):
            # break + rewarm: a missing bar invalidates the state; the next
            # jointly-valid bar re-seeds from its own range (round-11 P0).
            live = False
            continue
        if not live:
            # Re-seed after a gap from the current bar's own range.
            bull = True
            sar = low[t]
            ep = high[t]
            af = af0
            live = True
            out[t] = sar
            continue
        sar = sar + af * (ep - sar)
        if bull:
            if t >= 2:
                sar = min(sar, low[t - 1], low[t - 2])
            else:
                sar = min(sar, low[t - 1])
            if low[t] < sar:
                bull = False
                sar = ep
                ep = low[t]
                af = af0
            elif high[t] > ep:
                ep = high[t]
                af = min(af + af0, afmax)
        else:
            if t >= 2:
                sar = max(sar, high[t - 1], high[t - 2])
            else:
                sar = max(sar, high[t - 1])
            if high[t] > sar:
                bull = True
                sar = ep
                ep = high[t]
                af = af0
            elif low[t] < ep:
                ep = low[t]
                af = min(af + af0, afmax)
        out[t] = sar
    return out


def PSAR(high, low, acceleration, maximum):
    af0 = _pf(acceleration, "acceleration", 0)
    afmax = _pf(maximum, "maximum", 0)
    if af0 <= 0 or afmax < af0:
        raise ValueError("require 0 < acceleration <= maximum")
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _psar_1d(high[c].to_numpy(), low[c].to_numpy(), af0, afmax)
    return _make(high, cols, out)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("KeltnerMid", ("close", "ema_window"), KeltnerMid, "Keltner EMA midline."),
    ("KeltnerUpper", ("high", "low", "close", "ema_window", "atr_window", "multiplier"), KeltnerUpper, "Keltner upper envelope."),
    ("KeltnerLower", ("high", "low", "close", "ema_window", "atr_window", "multiplier"), KeltnerLower, "Keltner lower envelope."),
    ("KeltnerPosition", ("high", "low", "close", "ema_window", "atr_window", "multiplier"), KeltnerPosition, "Close position inside Keltner channel."),
    ("donchian_upper", ("high", "window"), donchian_upper, "Prior-window Donchian upper channel."),
    ("donchian_lower", ("low", "window"), donchian_lower, "Prior-window Donchian lower channel."),
    ("donchian_mid", ("high", "low", "window"), donchian_mid, "Donchian channel midpoint."),
    ("donchian_position", ("close", "high", "low", "window"), donchian_position, "Close position inside prior Donchian channel."),
    ("bollinger_pct_b", ("x", "window", "std_dev"), bollinger_pct_b, "Bollinger percent-B."),
    ("bollinger_width", ("x", "window", "std_dev"), bollinger_width, "Normalized Bollinger bandwidth."),
    ("efficiency_ratio", ("close", "window"), efficiency_ratio, "Kaufman efficiency ratio."),
    ("choppiness_index", ("high", "low", "close", "window"), choppiness_index, "Choppiness index based on true-range concentration."),
    ("Supertrend", ("high", "low", "close", "atr_window", "multiplier"), Supertrend, "Recursive Supertrend level."),
    ("SupertrendDirection", ("high", "low", "close", "atr_window", "multiplier"), SupertrendDirection, "Recursive Supertrend direction (+1/-1)."),
    ("PSAR", ("high", "low", "acceleration", "maximum"), PSAR, "Parabolic SAR; recursive/stateful."),
)


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    tags = ["pit_safe", "causal", "polars", "native"]
    # Keltner family recurses through ``_ema``/``_wilder`` (ewm adjust=False),
    # same full-replay contract as the pandas reference (review P0-05).
    if name in {"Supertrend", "SupertrendDirection", "PSAR", "KeltnerMid", "KeltnerUpper", "KeltnerLower", "KeltnerPosition"}:
        tags += ["stateful", "full_replay"]
    metadata = OperatorMetadata(
        name=name,
        category="technical_signal",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=tags,
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsTechMisc_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="technical_signal",
        business_category="technical",
        canonical=name,
        source="polars_tech_misc",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
