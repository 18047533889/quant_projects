# -*- coding: utf-8 -*-
"""Native Polars backend for chart-structure / pattern operators.

These operators are inherently sequential, per-column state machines
(confirmed-pivot detection, bounded history, OLS line fitting, quadratic
rounding scores, breakout-retest loops).  Polars expressions cannot express
them, so each column is evaluated with an independent NumPy kernel operating
directly on the Polars column arrays — the same algorithm as the pandas
reference, wrapped into a ``pl.DataFrame`` result.  No pandas DataFrame is
created on the fast path.
"""
from __future__ import annotations

from collections import deque
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


def _sdiv(num, den) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=float)
    if 0 < k < len(x):
        out[k:] = x[: len(x) - k]
    elif k < 0 and -k < len(x):
        out[: k] = x[-k:]
    return out


def _closeness(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    denom = (np.abs(a) + np.abs(b)) / 2.0
    rel = _sdiv(np.abs(a - b), denom)
    return np.clip(1.0 - rel / tol, 0.0, 1.0)


def _between(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return ((v >= lo) & (v <= hi)).astype(float)


# ---------------------------------------------------------------------------
# confirmed-pivot kernels (single column)
# ---------------------------------------------------------------------------


def _events_1d(series: np.ndarray, left: int, right: int, high: bool):
    n = len(series)
    prices = np.full(n, np.nan, dtype=float)
    positions = np.full(n, np.nan, dtype=float)
    for pivot in range(left, n - right):
        window = series[pivot - left : pivot + right + 1]
        if not np.isfinite(window).all():
            continue
        center = float(series[pivot])
        others = np.delete(window, left)
        ok = center > float(np.max(others)) if high else center < float(np.min(others))
        if ok:
            prices[pivot + right] = center
            positions[pivot + right] = float(pivot)
    return prices, positions


def _recent_events_1d(series, left, right, history, high):
    n = len(series)
    prices, positions = _events_1d(series, left, right, high)
    out: list[list[tuple[int, float, float]]] = [[] for _ in range(n)]
    active: list[tuple[int, float, float]] = []
    for t in range(n):
        cutoff = t - history + 1
        active = [e for e in active if e[0] >= cutoff]
        if np.isfinite(prices[t]) and np.isfinite(positions[t]):
            active.append((t, float(positions[t]), float(prices[t])))
        out[t] = list(active)
    return out


def _event_stat_1d(series, left, right, history, n, high, which) -> np.ndarray:
    events = _recent_events_1d(series, left, right, history, high)
    arr = np.full(len(series), np.nan, dtype=float)
    for t, ev in enumerate(events):
        if len(ev) < n:
            continue
        confirm, pivot, price = ev[-n]
        if which == "price":
            arr[t] = price
        elif which == "pivot_age":
            arr[t] = float(t - pivot)
        elif which == "confirmation_age":
            arr[t] = float(t - confirm)
    return arr


def _event_count_1d(series, left, right, history, high) -> np.ndarray:
    events = _recent_events_1d(series, left, right, history, high)
    return np.asarray([float(len(ev)) for ev in events], dtype=float)


def _spacing_1d(series, left, right, history, high) -> np.ndarray:
    events = _recent_events_1d(series, left, right, history, high)
    arr = np.full(len(series), np.nan, dtype=float)
    for t, ev in enumerate(events):
        if len(ev) >= 2:
            arr[t] = ev[-1][1] - ev[-2][1]
    return arr


def _bounded_last_1d(series, left, right, history, high, output) -> np.ndarray:
    n = len(series)
    prices, positions = _events_1d(series, left, right, high)
    out = np.full(n, np.nan, dtype=float)
    active: deque[tuple[int, float, float]] = deque()
    for t in range(n):
        cutoff = t - history + 1
        while active and active[0][0] < cutoff:
            active.popleft()
        if np.isfinite(prices[t]):
            active.append((t, positions[t], prices[t]))
        if not active:
            continue
        confirm, pivot, price = active[-1]
        if output == "price":
            out[t] = price
        elif output == "confirmation_age":
            out[t] = float(t - confirm)
        elif output == "pivot_age":
            out[t] = float(t - pivot)
    return out


def _bounded_line_1d(series, left, right, history, points, high, output, log=False) -> np.ndarray:
    n = len(series)
    prices, positions = _events_1d(series, left, right, high)
    out = np.full(n, np.nan, dtype=float)
    active: deque[tuple[int, float, float]] = deque()
    for t in range(n):
        cutoff = t - history + 1
        while active and active[0][0] < cutoff:
            active.popleft()
        if np.isfinite(prices[t]) and np.isfinite(positions[t]):
            active.append((t, float(positions[t]), float(prices[t])))
        if len(active) < 2:
            continue
        selected = list(active)[-points:]
        xs = np.asarray([e[1] for e in selected], dtype=float)
        ys = np.asarray([e[2] for e in selected], dtype=float)
        if log:
            ys = np.log(ys)
        xbar, ybar = xs.mean(), ys.mean()
        denom = float(np.dot(xs - xbar, xs - xbar))
        if denom <= 0:
            continue
        slope = float(np.dot(xs - xbar, ys - ybar) / denom)
        out[t] = slope if output == "slope" else ybar + slope * (float(t) - xbar)
    return out


def _fit_r2_1d(series, left, right, history, points, high) -> np.ndarray:
    events = _recent_events_1d(series, left, right, history, high)
    n = len(series)
    arr = np.full(n, np.nan, dtype=float)
    for t, ev in enumerate(events):
        if len(ev) < points:
            continue
        selected = ev[-points:]
        x = np.asarray([z[1] for z in selected], dtype=float)
        y = np.asarray([z[2] for z in selected], dtype=float)
        xb, yb = x.mean(), y.mean()
        den = np.sum((x - xb) ** 2)
        if den <= _EPS:
            continue
        slope = np.sum((x - xb) * (y - yb)) / den
        fit = yb + slope * (x - xb)
        ss_tot = np.sum((y - yb) ** 2)
        ss_res = np.sum((y - fit) ** 2)
        arr[t] = 1.0 - ss_res / ss_tot if ss_tot > _EPS else 1.0
    return arr


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


def _slope_win_1d(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    idx = np.arange(w, dtype=float)
    xb = idx.mean()
    den = np.sum((idx - xb) ** 2)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        if len(seg) < w or not np.isfinite(seg).all():
            continue
        out[t] = float(np.sum((idx - xb) * (seg - seg.mean())) / den)
    return out


def _rolling_mean_1d(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        ok = np.isfinite(seg)
        if int(ok.sum() < w:
            continue
        out[t] = float(np.mean(seg[ok]))
    return out


def _true_range_1d(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = len(h)
    prev = _shift(c, 1)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        hl = h[t] - l[t]
        hc = np.abs(h[t] - prev[t])
        lc = np.abs(l[t] - prev[t])
        if np.isfinite(prev[t]):
            out[t] = max(hl, hc, lc)
        else:
            out[t] = hl if np.isfinite(hl) else np.nan
    return out


def _tr_propagate_1d(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = len(h)
    prev = _shift(c, 1)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        hl = h[t] - l[t]
        hc = np.abs(h[t] - prev[t])
        lc = np.abs(l[t] - prev[t])
        values = [v for v in (hl, hc, lc) if np.isfinite(v)]
        out[t] = max(values) if values else np.nan
    return out


# ---------------------------------------------------------------------------
# registered operators
# ---------------------------------------------------------------------------


def ts_nth_pivot_high(high, left_window, right_window, history_window, n):
    lw, rw, hw, nn = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(n, "n")
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, nn, True, "price")
    return _make(high, cols, out)


def ts_nth_pivot_low(low, left_window, right_window, history_window, n):
    lw, rw, hw, nn = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(n, "n")
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, nn, False, "price")
    return _make(low, cols, out)


def ts_nth_pivot_high_age(high, left_window, right_window, history_window, n):
    lw, rw, hw, nn = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(n, "n")
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, nn, True, "pivot_age")
    return _make(high, cols, out)


def ts_nth_pivot_low_age(low, left_window, right_window, history_window, n):
    lw, rw, hw, nn = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(n, "n")
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, nn, False, "pivot_age")
    return _make(low, cols, out)


def ts_pivot_high_count(high, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _event_count_1d(high[c].to_numpy(), lw, rw, hw, True)
    return _make(high, cols, out)


def ts_pivot_low_count(low, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _event_count_1d(low[c].to_numpy(), lw, rw, hw, False)
    return _make(low, cols, out)


def ts_pivot_high_spacing(high, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _spacing_1d(high[c].to_numpy(), lw, rw, hw, True)
    return _make(high, cols, out)


def ts_pivot_low_spacing(low, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _spacing_1d(low[c].to_numpy(), lw, rw, hw, False)
    return _make(low, cols, out)


def ts_last_pivot_high(high, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bounded_last_1d(high[c].to_numpy(), lw, rw, hw, True, "price")
    return _make(high, cols, out)


def ts_last_pivot_low(low, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bounded_last_1d(low[c].to_numpy(), lw, rw, hw, False, "price")
    return _make(low, cols, out)


def ts_pivot_high_age(high, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bounded_last_1d(high[c].to_numpy(), lw, rw, hw, True, "pivot_age")
    return _make(high, cols, out)


def ts_pivot_low_age(low, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bounded_last_1d(low[c].to_numpy(), lw, rw, hw, False, "pivot_age")
    return _make(low, cols, out)


def ts_resistance_level(high, left_window, right_window, history_window, points):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bounded_line_1d(high[c].to_numpy(), lw, rw, hw, k, True, "level")
    return _make(high, cols, out)


def ts_support_level(low, left_window, right_window, history_window, points):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bounded_line_1d(low[c].to_numpy(), lw, rw, hw, k, False, "level")
    return _make(low, cols, out)


def ts_resistance_slope(high, left_window, right_window, history_window, points):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bounded_line_1d(high[c].to_numpy(), lw, rw, hw, k, True, "slope")
    return _make(high, cols, out)


def ts_support_slope(low, left_window, right_window, history_window, points):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bounded_line_1d(low[c].to_numpy(), lw, rw, hw, k, False, "slope")
    return _make(low, cols, out)


def ts_resistance_log_slope(high, left_window, right_window, history_window, points):
    # Log-price slope (audit 13): comparable across price levels.
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bounded_line_1d(high[c].to_numpy(), lw, rw, hw, k, True, "slope", log=True)
    return _make(high, cols, out)


def ts_support_log_slope(low, left_window, right_window, history_window, points):
    # Log-price slope (audit 13): comparable across price levels.
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bounded_line_1d(low[c].to_numpy(), lw, rw, hw, k, False, "slope", log=True)
    return _make(low, cols, out)


def _dist_line(close, line, up: bool) -> pl.DataFrame:
    cols = _cols(close)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        lvl = line[c].to_numpy()
        out[:, i] = _sdiv(close[c].to_numpy(), lvl) - 1.0
    return _make(close, cols, out)


def ts_distance_to_resistance(close, high, left_window, right_window, history_window, points):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(close, high)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        lvl = _bounded_line_1d(high[c].to_numpy(), lw, rw, hw, k, True, "level")
        out[:, i] = _sdiv(close[c].to_numpy(), lvl) - 1.0
    return _make(close, cols, out)


def ts_distance_to_support(close, low, left_window, right_window, history_window, points):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(close, low)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        lvl = _bounded_line_1d(low[c].to_numpy(), lw, rw, hw, k, False, "level")
        out[:, i] = _sdiv(close[c].to_numpy(), lvl) - 1.0
    return _make(close, cols, out)


def ts_resistance_break(close, high, left_window, right_window, history_window, points):
    dist = ts_distance_to_resistance(close, high, left_window, right_window, history_window, points)
    cols = _cols(dist)
    out = np.full((dist.height, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = np.clip(dist[c].to_numpy(), 0.0, None)
    return _make(dist, cols, out)


def ts_support_break(close, low, left_window, right_window, history_window, points):
    dist = ts_distance_to_support(close, low, left_window, right_window, history_window, points)
    cols = _cols(dist)
    out = np.full((dist.height, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = np.clip(-dist[c].to_numpy(), 0.0, None)
    return _make(dist, cols, out)


def ts_resistance_fit_r2(high, left_window, right_window, history_window, points):
    # R^2 on a 2-point fit is meaningless (always 1.0); require >= 3 points (audit 24).
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 3)
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _fit_r2_1d(high[c].to_numpy(), lw, rw, hw, k, True)
    return _make(high, cols, out)


def ts_support_fit_r2(low, left_window, right_window, history_window, points):
    # R^2 on a 2-point fit is meaningless (always 1.0); require >= 3 points (audit 24).
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 3)
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _fit_r2_1d(low[c].to_numpy(), lw, rw, hw, k, False)
    return _make(low, cols, out)


def _swing_segment_1d(high, low, left, right, history):
    """Single-column swing kernel: most recent adjacent opposite-sign pivot pair.

    Returns ``(amplitude, duration, velocity)`` arrays (audits 22/23).  The
    merged chronological pivot stream is scanned from the newest end for the last
    high->low or low->high adjacent pair, so amplitude/duration/velocity can never
    disagree about the underlying swing event.
    """
    h_events = _recent_events_1d(high, left, right, history, True)
    l_events = _recent_events_1d(low, left, right, history, False)
    n = len(high)
    amp = np.full(n, np.nan, dtype=float)
    dur = np.full(n, np.nan, dtype=float)
    for t in range(n):
        merged = [
            (int(pos), True, float(px)) for _, pos, px in h_events[t]
        ] + [
            (int(pos), False, float(px)) for _, pos, px in l_events[t]
        ]
        if len(merged) > 1:
            merged.sort(key=lambda e: e[0])
        for i in range(len(merged) - 2, -1, -1):
            p0, s0, px0 = merged[i]
            p1, s1, px1 = merged[i + 1]
            if s0 != s1:
                if s0:  # high then low
                    h_px, l_px = px0, px1
                else:   # low then high
                    h_px, l_px = px1, px0
                amp[t] = abs(float(h_px) - float(l_px))
                dur[t] = abs(int(p0) - int(p1))
                break
    vel = _sdiv(amp, np.where(dur == 0, np.nan, dur))
    return amp, dur, vel


def _swing_amplitude_1d(high, low, left, right, history):
    amp, _, _ = _swing_segment_1d(high, low, left, right, history)
    return amp


def ts_swing_amplitude(high, low, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _swing_amplitude_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw)
    return _make(high, cols, out)


def ts_swing_amplitude_pct(high, low, close, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(high, low, close)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        amp = _swing_amplitude_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw)
        out[:, i] = _sdiv(amp, np.abs(close[c].to_numpy()))
    return _make(close, cols, out)


def ts_swing_duration(high, low, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        _, dur, _ = _swing_segment_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw)
        out[:, i] = dur
    return _make(high, cols, out)


def ts_swing_velocity(high, low, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        _, _, vel = _swing_segment_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw)
        out[:, i] = vel
    return _make(high, cols, out)


def ts_swing_amplitude_atr(high, low, close, left_window, right_window, history_window, atr_window):
    lw, rw, hw, aw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(atr_window, "atr_window")
    cols = _cols(high, low, close)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        amp = _swing_amplitude_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw)
        tr = _true_range_1d(high[c].to_numpy(), low[c].to_numpy(), close[c].to_numpy())
        atr = _rolling_mean_1d(tr, aw)
        out[:, i] = _sdiv(amp, atr)
    return _make(high, cols, out)


def _channel_width_1d(high, low, left, right, history, points):
    res = _bounded_line_1d(high, left, right, history, points, True, "level")
    sup = _bounded_line_1d(low, left, right, history, points, False, "level")
    return res - sup


def ts_channel_width(high, low, left_window, right_window, history_window, points):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _channel_width_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw, k)
    return _make(high, cols, out)


def ts_channel_width_pct(close, high, low, left_window, right_window, history_window, points):
    width = ts_channel_width(high, low, left_window, right_window, history_window, points)
    cols = _cols(close)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _sdiv(width[c].to_numpy(), np.abs(close[c].to_numpy()))
    return _make(close, cols, out)


def ts_channel_width_atr(high, low, close, left_window, right_window, history_window, points, atr_window):
    lw, rw, hw, k, aw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2), _pi(atr_window, "atr_window")
    cols = _cols(high, low, close)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        width = _channel_width_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw, k)
        tr = _true_range_1d(high[c].to_numpy(), low[c].to_numpy(), close[c].to_numpy())
        atr = _rolling_mean_1d(tr, aw)
        out[:, i] = _sdiv(width, atr)
    return _make(high, cols, out)


def ts_channel_width_slope(high, low, left_window, right_window, history_window, points, window):
    lw, rw, hw, k, w = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2), _pi(window, "window", 2)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        width = _channel_width_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw, k)
        out[:, i] = _slope_win_1d(width, w)
    return _make(high, cols, out)


def _slopes_1d(high, low, left, right, history, points):
    res = _bounded_line_1d(high, left, right, history, points, True, "slope")
    sup = _bounded_line_1d(low, left, right, history, points, False, "slope")
    return res, sup


def ts_line_convergence(high, low, left_window, right_window, history_window, points):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        rs, ss = _slopes_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw, k)
        out[:, i] = ss - rs
    return _make(high, cols, out)


def ts_line_parallelism(high, low, left_window, right_window, history_window, points):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        # Normalized log-price slopes so parallelism is price-level independent (audit 25).
        rs = _bounded_line_1d(high[c].to_numpy(), lw, rw, hw, k, True, "slope", log=True)
        ss = _bounded_line_1d(low[c].to_numpy(), lw, rw, hw, k, False, "slope", log=True)
        out[:, i] = -np.abs(rs - ss)
    return _make(high, cols, out)


def ts_pattern_symmetry(high, low, left_window, right_window, history_window):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        h1 = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 1, True, "price")
        h2 = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 2, True, "price")
        l1 = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 1, False, "price")
        l2 = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 2, False, "price")
        hs = _sdiv(np.abs(h1 - h2), (np.abs(h1) + np.abs(h2)) / 2.0)
        ls = _sdiv(np.abs(l1 - l2), (np.abs(l1) + np.abs(l2)) / 2.0)
        out[:, i] = 1.0 / (1.0 + hs + ls)
    return _make(high, cols, out)


def ts_consolidation_width(high, low, window):
    w = _pi(window, "window", 2)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_max_1d(high[c].to_numpy(), w) - _rolling_min_1d(low[c].to_numpy(), w)
    return _make(high, cols, out)


def ts_consolidation_slope(close, window):
    w = _pi(window, "window", 2)
    cols = _cols(close)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _slope_win_1d(close[c].to_numpy(), w)
    return _make(close, cols, out)


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------


def pattern_double_top(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    tol = _pf(tolerance, "tolerance", 0)
    md = _pf(min_depth, "min_depth", 0)
    mn, mx = _pi(min_spacing, "min_spacing"), _pi(max_spacing, "max_spacing")
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        h1 = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 1, True, "price")
        h2 = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 2, True, "price")
        l1 = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 1, False, "price")
        depth = np.clip(_sdiv(h1, l1) - 1.0, 0.0, None)
        spacing = _spacing_1d(high[c].to_numpy(), lw, rw, hw, True)
        out[:, i] = _closeness(h1, h2, tol) * (depth >= md).astype(float) * _between(spacing, mn, mx)
    return _make(high, cols, out)


def pattern_double_bottom(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    tol = _pf(tolerance, "tolerance", 0)
    md = _pf(min_depth, "min_depth", 0)
    mn, mx = _pi(min_spacing, "min_spacing"), _pi(max_spacing, "max_spacing")
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        l1 = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 1, False, "price")
        l2 = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 2, False, "price")
        h1 = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 1, True, "price")
        depth = np.clip(_sdiv(h1, l1) - 1.0, 0.0, None)
        spacing = _spacing_1d(low[c].to_numpy(), lw, rw, hw, False)
        out[:, i] = _closeness(l1, l2, tol) * (depth >= md).astype(float) * _between(spacing, mn, mx)
    return _make(high, cols, out)


def pattern_head_shoulders(high, low, left_window, right_window, history_window, shoulder_tolerance, head_min_prominence, max_neckline_slope):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    st = _pf(shoulder_tolerance, "shoulder_tolerance", 0)
    hp = _pf(head_min_prominence, "head_min_prominence", 0)
    ms = _pf(max_neckline_slope, "max_neckline_slope", 0)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        r = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 1, True, "price")
        h = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 2, True, "price")
        l = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 3, True, "price")
        shoulders = _closeness(l, r, st)
        base = (np.abs(l) + np.abs(r)) / 2.0
        prom = _sdiv(h, base) - 1.0
        neckline = np.abs(_bounded_line_1d(low[c].to_numpy(), lw, rw, hw, 2, False, "slope"))
        out[:, i] = shoulders * (prom >= hp).astype(float) * (neckline <= ms).astype(float)
    return _make(high, cols, out)


def pattern_inverse_head_shoulders(high, low, left_window, right_window, history_window, shoulder_tolerance, head_min_prominence, max_neckline_slope):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    st = _pf(shoulder_tolerance, "shoulder_tolerance", 0)
    hp = _pf(head_min_prominence, "head_min_prominence", 0)
    ms = _pf(max_neckline_slope, "max_neckline_slope", 0)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        r = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 1, False, "price")
        h = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 2, False, "price")
        l = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 3, False, "price")
        shoulders = _closeness(l, r, st)
        base = (np.abs(l) + np.abs(r)) / 2.0
        prom = _sdiv(base, np.abs(h)) - 1.0
        neckline = np.abs(_bounded_line_1d(high[c].to_numpy(), lw, rw, hw, 2, True, "slope"))
        out[:, i] = shoulders * (prom >= hp).astype(float) * (neckline <= ms).astype(float)
    return _make(high, cols, out)


def _slope_pattern(high, low, left_window, right_window, history_window, points, slope_threshold, kind):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    th = _pf(slope_threshold, "slope_threshold", 0)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        rs, ss = _slopes_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw, k)
        if kind == "sym_triangle":
            flag = (rs < -th) & (ss > th)
        elif kind == "ascending_triangle":
            flag = (np.abs(rs) <= th) & (ss > th)
        elif kind == "descending_triangle":
            flag = (rs < -th) & (np.abs(ss) <= th)
        elif kind == "rising_wedge":
            flag = (rs > th) & (ss > th) & (ss > rs)
        elif kind == "falling_wedge":
            flag = (rs < -th) & (ss < -th) & (rs < ss)
        elif kind == "rectangle":
            flag = (np.abs(rs) <= th) & (np.abs(ss) <= th)
        elif kind == "broadening":
            flag = (rs > th) & (ss < -th)
        else:
            raise ValueError(kind)
        out[:, i] = flag.astype(float)
    return _make(high, cols, out)


def pattern_sym_triangle(high, low, left_window, right_window, history_window, points, slope_threshold):
    return _slope_pattern(high, low, left_window, right_window, history_window, points, slope_threshold, "sym_triangle")


def pattern_ascending_triangle(high, low, left_window, right_window, history_window, points, slope_threshold):
    return _slope_pattern(high, low, left_window, right_window, history_window, points, slope_threshold, "ascending_triangle")


def pattern_descending_triangle(high, low, left_window, right_window, history_window, points, slope_threshold):
    return _slope_pattern(high, low, left_window, right_window, history_window, points, slope_threshold, "descending_triangle")


def pattern_rising_wedge(high, low, left_window, right_window, history_window, points, slope_threshold):
    return _slope_pattern(high, low, left_window, right_window, history_window, points, slope_threshold, "rising_wedge")


def pattern_falling_wedge(high, low, left_window, right_window, history_window, points, slope_threshold):
    return _slope_pattern(high, low, left_window, right_window, history_window, points, slope_threshold, "falling_wedge")


def pattern_rectangle(high, low, left_window, right_window, history_window, points, slope_threshold):
    return _slope_pattern(high, low, left_window, right_window, history_window, points, slope_threshold, "rectangle")


def pattern_rising_channel(high, low, left_window, right_window, history_window, points, slope_threshold, parallel_tolerance):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    th = _pf(slope_threshold, "slope_threshold", 0)
    pt = _pf(parallel_tolerance, "parallel_tolerance", 0)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        rs, ss = _slopes_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw, k)
        flag = (rs > th) & (ss > th) & (np.abs(rs - ss) <= pt)
        out[:, i] = flag.astype(float)
    return _make(high, cols, out)


def pattern_falling_channel(high, low, left_window, right_window, history_window, points, slope_threshold, parallel_tolerance):
    lw, rw, hw, k = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window"), _pi(points, "points", 2)
    th = _pf(slope_threshold, "slope_threshold", 0)
    pt = _pf(parallel_tolerance, "parallel_tolerance", 0)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        rs, ss = _slopes_1d(high[c].to_numpy(), low[c].to_numpy(), lw, rw, hw, k)
        flag = (rs < -th) & (ss < -th) & (np.abs(rs - ss) <= pt)
        out[:, i] = flag.astype(float)
    return _make(high, cols, out)


def pattern_broadening(high, low, left_window, right_window, history_window, points, slope_threshold):
    return _slope_pattern(high, low, left_window, right_window, history_window, points, slope_threshold, "broadening")


def _flag_pattern(close, high, low, volume, impulse_window, flag_window, min_impulse, max_retracement, max_width, volume_decay_threshold, bull):
    iw, fw = _pi(impulse_window, "impulse_window", 2), _pi(flag_window, "flag_window", 2)
    mi = _pf(min_impulse, "min_impulse", 0)
    mr = _pf(max_retracement, "max_retracement", 0)
    mw = _pf(max_width, "max_width", 0)
    vd = _pf(volume_decay_threshold, "volume_decay_threshold")
    cols = _cols(close, high, low, volume)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        c_arr, h_arr, l_arr, v_arr = close[c].to_numpy(), high[c].to_numpy(), low[c].to_numpy(), volume[c].to_numpy()
        impulse = _sdiv(_shift(c_arr, fw), _shift(c_arr, fw + iw)) - 1.0
        if bull:
            peak = _shift(c_arr, fw)
            trough = _rolling_min_1d(l_arr, fw)
            retr = _sdiv(peak - trough, np.abs(peak))
            direction = impulse >= mi
        else:
            base = _shift(c_arr, fw)
            rebound = _rolling_max_1d(h_arr, fw)
            retr = _sdiv(rebound - base, np.abs(base))
            direction = impulse <= -mi
        width = _sdiv(_rolling_max_1d(h_arr, fw) - _rolling_min_1d(l_arr, fw), np.abs(c_arr))
        vdec = -_slope_win_1d(v_arr, fw)
        out[:, i] = (direction & (retr <= mr) & (width <= mw) & (vdec >= vd)).astype(float)
    return _make(close, cols, out)


def pattern_bull_flag(close, high, low, volume, impulse_window, flag_window, min_impulse, max_retracement, max_width, volume_decay_threshold):
    return _flag_pattern(close, high, low, volume, impulse_window, flag_window, min_impulse, max_retracement, max_width, volume_decay_threshold, True)


def pattern_bear_flag(close, high, low, volume, impulse_window, flag_window, min_impulse, max_retracement, max_width, volume_decay_threshold):
    return _flag_pattern(close, high, low, volume, impulse_window, flag_window, min_impulse, max_retracement, max_width, volume_decay_threshold, False)


def _similar_1d(stack: np.ndarray, tol: float) -> np.ndarray:
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    mx = np.max(np.where(valid, stack, -np.inf), axis=0)
    mn = np.min(np.where(valid, stack, np.inf), axis=0)
    abs_sum = np.where(valid, np.abs(stack), 0.0).sum(axis=0)
    mid = np.divide(abs_sum, count, out=np.full(count.shape, np.nan, dtype=float), where=count > 0)
    complete = count == stack.shape[0]
    denom = mid * float(tol)
    ratio = np.full(count.shape, np.nan, dtype=float)
    np.divide(mx - mn, denom, out=ratio, where=complete & (denom > 0))
    return np.where(complete, np.clip(1.0 - ratio, 0.0, 1.0), np.nan)


def _triple(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing, *, top):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    tol = _pf(tolerance, "tolerance", 1e-12)
    md = _pf(min_depth, "min_depth", 0)
    mn, mx = _pi(min_spacing, "min_spacing"), _pi(max_spacing, "max_spacing")
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        hc, lc = high[c].to_numpy(), low[c].to_numpy()
        if top:
            hs = np.stack([_event_stat_1d(hc, lw, rw, hw, k, True, "price") for k in (1, 2, 3)])
            l1 = _event_stat_1d(lc, lw, rw, hw, 1, False, "price")
            mean_high = hs.sum(axis=0) / 3.0
            depth = (mean_high / l1 - 1.0)
            spacing = _spacing_1d(hc, lw, rw, hw, True)
            similar = _similar_1d(hs, tol)
        else:
            ls = np.stack([_event_stat_1d(lc, lw, rw, hw, k, False, "price") for k in (1, 2, 3)])
            h1 = _event_stat_1d(hc, lw, rw, hw, 1, True, "price")
            mean_low = ls.sum(axis=0) / 3.0
            depth = _sdiv(h1, mean_low) - 1.0
            spacing = _spacing_1d(lc, lw, rw, hw, False)
            similar = _similar_1d(ls, tol)
        out[:, i] = similar * (depth >= md).astype(float) * _between(spacing, mn, mx)
    return _make(high, cols, out)


def pattern_triple_top(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing):
    return _triple(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing, top=True)


def pattern_triple_bottom(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing):
    return _triple(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing, top=False)


def pattern_123_bull(high, low, left_window, right_window, history_window, min_swing):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    ms = _pf(min_swing, "min_swing", 0)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        h1 = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 1, True, "price")
        l1 = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 1, False, "price")
        l2 = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 2, False, "price")
        flag = (l1 > l2) & (_sdiv(h1, l1) - 1.0 >= ms)
        out[:, i] = flag.astype(float)
    return _make(high, cols, out)


def pattern_123_bear(high, low, left_window, right_window, history_window, min_swing):
    lw, rw, hw = _pi(left_window, "left_window"), _pi(right_window, "right_window"), _pi(history_window, "history_window")
    ms = _pf(min_swing, "min_swing", 0)
    cols = _cols(high, low)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        l1 = _event_stat_1d(low[c].to_numpy(), lw, rw, hw, 1, False, "price")
        h1 = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 1, True, "price")
        h2 = _event_stat_1d(high[c].to_numpy(), lw, rw, hw, 2, True, "price")
        flag = (h1 < h2) & (_sdiv(h1, l1) - 1.0 >= ms)
        out[:, i] = flag.astype(float)
    return _make(high, cols, out)


def _quadratic_1d(arr: np.ndarray, window: int, up: bool) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    x = np.linspace(-1.0, 1.0, window)
    for t in range(window - 1, n):
        y = arr[t - window + 1 : t + 1]
        if not np.isfinite(y).all():
            continue
        mean_abs = float(np.nanmean(np.abs(y)))
        if mean_abs == 0:
            continue
        yn = y / mean_abs
        coef = np.polyfit(x, yn, 2)
        fit = np.polyval(coef, x)
        tot = np.sum((yn - yn.mean()) ** 2)
        r2 = 1.0 - np.sum((yn - fit) ** 2) / tot if tot > 1e-12 else 0.0
        curvature = coef[0] if up else -coef[0]
        out[t] = max(0.0, float(curvature)) * max(0.0, float(r2))
    return out


def pattern_rounding_bottom(close, window, min_fit):
    w = _pi(window, "window", 5)
    mf = _pf(min_fit, "min_fit", 0)
    cols = _cols(close)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        score = _quadratic_1d(close[c].to_numpy(), w, True)
        out[:, i] = np.where(score >= mf, score, 0.0)
    return _make(close, cols, out)


def pattern_rounding_top(close, window, min_fit):
    w = _pi(window, "window", 5)
    mf = _pf(min_fit, "min_fit", 0)
    cols = _cols(close)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        score = _quadratic_1d(close[c].to_numpy(), w, False)
        out[:, i] = np.where(score >= mf, score, 0.0)
    return _make(close, cols, out)


def pattern_cup(close, window, min_depth, max_edge_diff, min_fit):
    w = _pi(window, "window", 5)
    md = _pf(min_depth, "min_depth", 0)
    me = _pf(max_edge_diff, "max_edge_diff", 0)
    mf = _pf(min_fit, "min_fit", 0)
    cols = _cols(close)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        arr = close[c].to_numpy()
        score = _quadratic_1d(arr, w, True)
        left = _shift(arr, w - 1)
        edge = _sdiv(np.abs(arr - left), (np.abs(arr) + np.abs(left)) / 2.0)
        center = _shift(arr, w // 2)
        depth = _sdiv((left + arr) / 2.0, center) - 1.0
        out[:, i] = score * (edge <= me).astype(float) * (depth >= md).astype(float) * (score >= mf).astype(float)
    return _make(close, cols, out)


def pattern_cup_handle(close, high, low, cup_window, handle_window, min_depth, max_edge_diff, min_fit, max_handle_retracement):
    cw = _pi(cup_window, "cup_window", 5)
    hw = _pi(handle_window, "handle_window", 2)
    md = _pf(min_depth, "min_depth", 0)
    me = _pf(max_edge_diff, "max_edge_diff", 0)
    mf = _pf(min_fit, "min_fit", 0)
    mhr = _pf(max_handle_retracement, "max_handle_retracement", 0)
    cols = _cols(close, high, low)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        arr = close[c].to_numpy()
        h_arr, l_arr = high[c].to_numpy(), low[c].to_numpy()
        shifted = _shift(arr, hw)
        score = _quadratic_1d(shifted, cw, True)
        left = _shift(shifted, cw - 1)
        edge = _sdiv(np.abs(shifted - left), (np.abs(shifted) + np.abs(left)) / 2.0)
        center = _shift(shifted, cw // 2)
        depth = _sdiv((left + shifted) / 2.0, center) - 1.0
        cup = score * (edge <= me).astype(float) * (depth >= md).astype(float) * (score >= mf).astype(float)
        base = _shift(arr, hw)
        trough = _rolling_min_1d(l_arr, hw)
        retr = _sdiv(base - trough, np.abs(base))
        out[:, i] = cup * (retr <= mhr).astype(float)
    return _make(close, cols, out)


def _pennant(close, high, low, volume, impulse_window, pennant_window, min_impulse, max_width, volume_decay_threshold, bull):
    iw = _pi(impulse_window, "impulse_window", 2)
    pw = _pi(pennant_window, "pennant_window", 3)
    mi = _pf(min_impulse, "min_impulse", 0)
    mw = _pf(max_width, "max_width", 0)
    vd = _pf(volume_decay_threshold, "volume_decay_threshold")
    cols = _cols(close, high, low, volume)
    rows = close.height
    half = max(2, pw // 2)
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        c_arr, h_arr, l_arr, v_arr = close[c].to_numpy(), high[c].to_numpy(), low[c].to_numpy(), volume[c].to_numpy()
        imp = _sdiv(_shift(c_arr, pw), _shift(c_arr, pw + iw)) - 1.0
        width = _sdiv(_rolling_max_1d(h_arr, pw) - _rolling_min_1d(l_arr, pw), np.abs(c_arr))
        v = -_slope_win_1d(v_arr, pw)
        rng_now = _rolling_mean_1d(h_arr - l_arr, half)
        rng_old = _rolling_mean_1d(_shift(h_arr - l_arr, half), half)
        compress = rng_now < rng_old
        direction = imp >= mi if bull else imp <= -mi
        out[:, i] = (direction & (width <= mw) & (v >= vd) & compress).astype(float)
    return _make(close, cols, out)


def pattern_bull_pennant(close, high, low, volume, impulse_window, pennant_window, min_impulse, max_width, volume_decay_threshold):
    return _pennant(close, high, low, volume, impulse_window, pennant_window, min_impulse, max_width, volume_decay_threshold, True)


def pattern_bear_pennant(close, high, low, volume, impulse_window, pennant_window, min_impulse, max_width, volume_decay_threshold):
    return _pennant(close, high, low, volume, impulse_window, pennant_window, min_impulse, max_width, volume_decay_threshold, False)


def _retest_1d(close, window, max_wait, tolerance, break_up):
    n = len(close)
    out = np.zeros(n, dtype=float)
    level = np.full(n, np.nan, dtype=float)
    for t in range(n):
        seg = close[max(0, t - window) : t]
        ok = np.isfinite(seg)
        if int(ok.sum() >= window:
            level[t] = float(np.max(seg[ok]) if break_up else np.min(seg[ok]))
    event = close > level if break_up else close < level
    last_level = np.nan
    age = max_wait + 1
    for t in range(n):
        if bool(event[t]) and np.isfinite(level[t]):
            last_level = level[t]
            age = 0
            continue
        age += 1
        if age <= max_wait and np.isfinite(last_level):
            px = close[t]
            rel = px / last_level - 1.0 if last_level else np.nan
            ok = (-tolerance <= rel <= tolerance) and (
                (px >= last_level * (1 - tolerance)) if break_up else (px <= last_level * (1 + tolerance))
            )
            out[t] = 1.0 if ok else 0.0
    return out


def pattern_breakout_retest(close, window, max_wait, tolerance):
    w = _pi(window, "window", 2)
    wait = _pi(max_wait, "max_wait")
    tol = _pf(tolerance, "tolerance", 0)
    cols = _cols(close)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _retest_1d(close[c].to_numpy(), w, wait, tol, True)
    return _make(close, cols, out)


def pattern_breakdown_retest(close, window, max_wait, tolerance):
    w = _pi(window, "window", 2)
    wait = _pi(max_wait, "max_wait")
    tol = _pf(tolerance, "tolerance", 0)
    cols = _cols(close)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _retest_1d(close[c].to_numpy(), w, wait, tol, False)
    return _make(close, cols, out)


# ---------------------------------------------------------------------------
# confirmed pivots (technical_extensions)
# ---------------------------------------------------------------------------


def ts_confirmed_pivot_high(high, left_window, right_window):
    lw = _pi(left_window, "left_window")
    rw = _pi(right_window, "right_window")
    cols = _cols(high)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        prices, _ = _events_1d(high[c].to_numpy(), lw, rw, True)
        out[:, i] = prices
    return _make(high, cols, out)


def ts_confirmed_pivot_low(low, left_window, right_window):
    lw = _pi(left_window, "left_window")
    rw = _pi(right_window, "right_window")
    cols = _cols(low)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        prices, _ = _events_1d(low[c].to_numpy(), lw, rw, False)
        out[:, i] = prices
    return _make(low, cols, out)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("ts_nth_pivot_high", ("high", "left_window", "right_window", "history_window", "n"), ts_nth_pivot_high, "Nth most recent confirmed pivot high."),
    ("ts_nth_pivot_low", ("low", "left_window", "right_window", "history_window", "n"), ts_nth_pivot_low, "Nth most recent confirmed pivot low."),
    ("ts_nth_pivot_high_age", ("high", "left_window", "right_window", "history_window", "n"), ts_nth_pivot_high_age, "Age of nth confirmed pivot high."),
    ("ts_nth_pivot_low_age", ("low", "left_window", "right_window", "history_window", "n"), ts_nth_pivot_low_age, "Age of nth confirmed pivot low."),
    ("ts_pivot_high_count", ("high", "left_window", "right_window", "history_window"), ts_pivot_high_count, "Count of confirmed pivot highs in bounded history."),
    ("ts_pivot_low_count", ("low", "left_window", "right_window", "history_window"), ts_pivot_low_count, "Count of confirmed pivot lows in bounded history."),
    ("ts_pivot_high_spacing", ("high", "left_window", "right_window", "history_window"), ts_pivot_high_spacing, "Spacing between latest two confirmed pivot highs."),
    ("ts_pivot_low_spacing", ("low", "left_window", "right_window", "history_window"), ts_pivot_low_spacing, "Spacing between latest two confirmed pivot lows."),
    ("ts_last_pivot_high", ("high", "left_window", "right_window", "history_window"), ts_last_pivot_high, "Latest confirmed pivot high inside bounded history."),
    ("ts_last_pivot_low", ("low", "left_window", "right_window", "history_window"), ts_last_pivot_low, "Latest confirmed pivot low inside bounded history."),
    ("ts_pivot_high_age", ("high", "left_window", "right_window", "history_window"), ts_pivot_high_age, "Bars since original pivot-high bar after confirmation."),
    ("ts_pivot_low_age", ("low", "left_window", "right_window", "history_window"), ts_pivot_low_age, "Bars since original pivot-low bar after confirmation."),
    ("ts_resistance_level", ("high", "left_window", "right_window", "history_window", "points"), ts_resistance_level, "Projected resistance line from recent confirmed pivot highs."),
    ("ts_support_level", ("low", "left_window", "right_window", "history_window", "points"), ts_support_level, "Projected support line from recent confirmed pivot lows."),
    ("ts_resistance_slope", ("high", "left_window", "right_window", "history_window", "points"), ts_resistance_slope, "Slope of bounded resistance line."),
    ("ts_support_slope", ("low", "left_window", "right_window", "history_window", "points"), ts_support_slope, "Slope of bounded support line."),
    ("ts_resistance_log_slope", ("high", "left_window", "right_window", "history_window", "points"), ts_resistance_log_slope, "Log-price slope of bounded resistance line."),
    ("ts_support_log_slope", ("low", "left_window", "right_window", "history_window", "points"), ts_support_log_slope, "Log-price slope of bounded support line."),
    ("ts_distance_to_resistance", ("close", "high", "left_window", "right_window", "history_window", "points"), ts_distance_to_resistance, "Signed close-to-resistance distance."),
    ("ts_distance_to_support", ("close", "low", "left_window", "right_window", "history_window", "points"), ts_distance_to_support, "Signed close-to-support distance."),
    ("ts_resistance_break", ("close", "high", "left_window", "right_window", "history_window", "points"), ts_resistance_break, "Positive breakout magnitude above resistance."),
    ("ts_support_break", ("close", "low", "left_window", "right_window", "history_window", "points"), ts_support_break, "Positive breakdown magnitude below support."),
    ("ts_resistance_fit_r2", ("high", "left_window", "right_window", "history_window", "points"), ts_resistance_fit_r2, "R-squared of recent confirmed resistance pivots."),
    ("ts_support_fit_r2", ("low", "left_window", "right_window", "history_window", "points"), ts_support_fit_r2, "R-squared of recent confirmed support pivots."),
    ("ts_swing_amplitude", ("high", "low", "left_window", "right_window", "history_window"), ts_swing_amplitude, "Absolute amplitude between latest confirmed pivots."),
    ("ts_swing_amplitude_pct", ("high", "low", "close", "left_window", "right_window", "history_window"), ts_swing_amplitude_pct, "Swing amplitude normalized by close."),
    ("ts_swing_duration", ("high", "low", "left_window", "right_window", "history_window"), ts_swing_duration, "Bar distance between latest opposite confirmed pivots."),
    ("ts_swing_velocity", ("high", "low", "left_window", "right_window", "history_window"), ts_swing_velocity, "Swing amplitude per bar."),
    ("ts_swing_amplitude_atr", ("high", "low", "close", "left_window", "right_window", "history_window", "atr_window"), ts_swing_amplitude_atr, "Swing amplitude normalized by ATR."),
    ("ts_channel_width", ("high", "low", "left_window", "right_window", "history_window", "points"), ts_channel_width, "Projected resistance minus support width."),
    ("ts_channel_width_pct", ("close", "high", "low", "left_window", "right_window", "history_window", "points"), ts_channel_width_pct, "Channel width normalized by close."),
    ("ts_channel_width_atr", ("high", "low", "close", "left_window", "right_window", "history_window", "points", "atr_window"), ts_channel_width_atr, "Channel width normalized by ATR."),
    ("ts_channel_width_slope", ("high", "low", "left_window", "right_window", "history_window", "points", "window"), ts_channel_width_slope, "Rolling slope of projected channel width."),
    ("ts_line_convergence", ("high", "low", "left_window", "right_window", "history_window", "points"), ts_line_convergence, "Support slope minus resistance slope."),
    ("ts_line_parallelism", ("high", "low", "left_window", "right_window", "history_window", "points"), ts_line_parallelism, "Negative absolute slope difference."),
    ("ts_pattern_symmetry", ("high", "low", "left_window", "right_window", "history_window"), ts_pattern_symmetry, "Continuous symmetry score of recent pivots."),
    ("ts_consolidation_width", ("high", "low", "window"), ts_consolidation_width, "High-low width of a consolidation window."),
    ("ts_consolidation_slope", ("close", "window"), ts_consolidation_slope, "OLS slope inside a consolidation window."),
    ("pattern_double_top", ("high", "low", "left_window", "right_window", "history_window", "tolerance", "min_depth", "min_spacing", "max_spacing"), pattern_double_top, "Continuous confirmed double-top structure score."),
    ("pattern_double_bottom", ("high", "low", "left_window", "right_window", "history_window", "tolerance", "min_depth", "min_spacing", "max_spacing"), pattern_double_bottom, "Continuous confirmed double-bottom structure score."),
    ("pattern_head_shoulders", ("high", "low", "left_window", "right_window", "history_window", "shoulder_tolerance", "head_min_prominence", "max_neckline_slope"), pattern_head_shoulders, "Confirmed head-and-shoulders structure score."),
    ("pattern_inverse_head_shoulders", ("high", "low", "left_window", "right_window", "history_window", "shoulder_tolerance", "head_min_prominence", "max_neckline_slope"), pattern_inverse_head_shoulders, "Confirmed inverse head-and-shoulders score."),
    ("pattern_sym_triangle", ("high", "low", "left_window", "right_window", "history_window", "points", "slope_threshold"), pattern_sym_triangle, "Symmetrical triangle structure."),
    ("pattern_ascending_triangle", ("high", "low", "left_window", "right_window", "history_window", "points", "slope_threshold"), pattern_ascending_triangle, "Ascending triangle structure."),
    ("pattern_descending_triangle", ("high", "low", "left_window", "right_window", "history_window", "points", "slope_threshold"), pattern_descending_triangle, "Descending triangle structure."),
    ("pattern_rising_wedge", ("high", "low", "left_window", "right_window", "history_window", "points", "slope_threshold"), pattern_rising_wedge, "Rising wedge structure."),
    ("pattern_falling_wedge", ("high", "low", "left_window", "right_window", "history_window", "points", "slope_threshold"), pattern_falling_wedge, "Falling wedge structure."),
    ("pattern_rectangle", ("high", "low", "left_window", "right_window", "history_window", "points", "slope_threshold"), pattern_rectangle, "Flat support/resistance rectangle structure."),
    ("pattern_rising_channel", ("high", "low", "left_window", "right_window", "history_window", "points", "slope_threshold", "parallel_tolerance"), pattern_rising_channel, "Rising parallel channel structure."),
    ("pattern_falling_channel", ("high", "low", "left_window", "right_window", "history_window", "points", "slope_threshold", "parallel_tolerance"), pattern_falling_channel, "Falling parallel channel structure."),
    ("pattern_broadening", ("high", "low", "left_window", "right_window", "history_window", "points", "slope_threshold"), pattern_broadening, "Broadening/megaphone structure."),
    ("pattern_bull_flag", ("close", "high", "low", "volume", "impulse_window", "flag_window", "min_impulse", "max_retracement", "max_width", "volume_decay_threshold"), pattern_bull_flag, "Bull flag structure score."),
    ("pattern_bear_flag", ("close", "high", "low", "volume", "impulse_window", "flag_window", "min_impulse", "max_retracement", "max_width", "volume_decay_threshold"), pattern_bear_flag, "Bear flag structure score."),
    ("pattern_triple_top", ("high", "low", "left_window", "right_window", "history_window", "tolerance", "min_depth", "min_spacing", "max_spacing"), pattern_triple_top, "Confirmed triple-top structure score."),
    ("pattern_triple_bottom", ("high", "low", "left_window", "right_window", "history_window", "tolerance", "min_depth", "min_spacing", "max_spacing"), pattern_triple_bottom, "Confirmed triple-bottom structure score."),
    ("pattern_123_bull", ("high", "low", "left_window", "right_window", "history_window", "min_swing"), pattern_123_bull, "123 bullish reversal structure."),
    ("pattern_123_bear", ("high", "low", "left_window", "right_window", "history_window", "min_swing"), pattern_123_bear, "123 bearish reversal structure."),
    ("pattern_rounding_bottom", ("close", "window", "min_fit"), pattern_rounding_bottom, "Quadratic rounding-bottom structure score."),
    ("pattern_rounding_top", ("close", "window", "min_fit"), pattern_rounding_top, "Quadratic rounding-top structure score."),
    ("pattern_cup", ("close", "window", "min_depth", "max_edge_diff", "min_fit"), pattern_cup, "Cup-without-handle structure score."),
    ("pattern_cup_handle", ("close", "high", "low", "cup_window", "handle_window", "min_depth", "max_edge_diff", "min_fit", "max_handle_retracement"), pattern_cup_handle, "Cup-and-handle structure score."),
    ("pattern_bull_pennant", ("close", "high", "low", "volume", "impulse_window", "pennant_window", "min_impulse", "max_width", "volume_decay_threshold"), pattern_bull_pennant, "Bull pennant structure score."),
    ("pattern_bear_pennant", ("close", "high", "low", "volume", "impulse_window", "pennant_window", "min_impulse", "max_width", "volume_decay_threshold"), pattern_bear_pennant, "Bear pennant structure score."),
    ("pattern_breakout_retest", ("close", "window", "max_wait", "tolerance"), pattern_breakout_retest, "Breakout retest confirmation."),
    ("pattern_breakdown_retest", ("close", "window", "max_wait", "tolerance"), pattern_breakdown_retest, "Breakdown retest confirmation."),
    ("ts_confirmed_pivot_high", ("high", "left_window", "right_window"), ts_confirmed_pivot_high, "Confirmed pivot high at confirmation timestamp."),
    ("ts_confirmed_pivot_low", ("low", "left_window", "right_window"), ts_confirmed_pivot_low, "Confirmed pivot low at confirmation timestamp."),
)


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    category = "chart_pattern" if name.startswith("pattern_") else "price_structure"
    metadata = OperatorMetadata(
        name=name,
        category=category,
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "bounded_history", "polars", "native"],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsStructure_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category=category,
        business_category="technical_structure",
        canonical=name,
        source="polars_structure",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
