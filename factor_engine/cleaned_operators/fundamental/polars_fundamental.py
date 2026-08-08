# -*- coding: utf-8 -*-
"""Native Polars backend for PIT-safe fundamental period operators.

The pandas reference for these operators is itself a per-column fiscal-ordinal
state machine (``_walk_periods``): values advance only when a new report period
becomes visible, revisions update from the revision timestamp onward, and
calculations never look past the current report period.  Polars expressions
cannot express this, so each column is evaluated with an independent NumPy
kernel that reuses the exact reference semantics; results are wrapped into a
``pl.DataFrame`` without creating pandas DataFrames on the fast path.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fiscal_strict import (
    period_ordinal,
    pl_quarter_from_cumulative,
    pl_ttm_from_quarterly,
)
from cleaned_operators.fundamental.transforms_v2 import _period_key

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


def _safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= _EPS:
        return np.nan
    return float(a / b)


def _period_insert(order: list, key) -> None:
    """Insert ``key`` into ``order`` keeping fiscal-ordinal sorted order.

    Mirrors the pandas ``_period_insert``: report periods advance by fiscal
    ordinal, so a late-disclosed revision or back-filled older period must not
    reorder the lag / TTM / growth sequence (audit §4.1).
    """
    target = period_ordinal(key)
    if target is None:
        order.append(key)
        return
    for position, existing in enumerate(order):
        existing_ord = period_ordinal(existing)
        if existing_ord is not None and existing_ord > target:
            order.insert(position, key)
            return
    order.append(key)


def _values(order: list, visible: OrderedDict, current, count: int | None = None):
    try:
        pos = order.index(current)
    except ValueError:
        return []
    keys = order[: pos + 1]
    if count is not None:
        keys = keys[-int(count):]
    return [float(visible[k]) for k in keys if k in visible and np.isfinite(visible[k])]


def _lag_value(order, visible, current, periods: int):
    """Exact ordinal lag: the value whose fiscal ordinal is ``current - periods``.

    A skipped report period yields NaN instead of a non-adjacent quarter's value
    (audit §4.2) — mirrors the pandas ``_lag_value``.
    """
    target = period_ordinal(current)
    if target is None:
        return np.nan
    target -= periods
    for key in reversed(order):
        if period_ordinal(key) == target:
            return float(visible.get(key, np.nan))
    return np.nan


def _walk_1d(xv: np.ndarray, pv: list, fn: Callable) -> np.ndarray:
    n = len(xv)
    order: list = []
    visible: OrderedDict = OrderedDict()
    arr = np.full(n, np.nan, dtype=float)
    for i, (value, raw_period) in enumerate(zip(xv, pv)):
        key = _period_key(raw_period)
        if key is not None and np.isfinite(value):
            if key not in visible:
                _period_insert(order, key)
            visible[key] = float(value)
        if key is None or key not in visible:
            continue
        try:
            arr[i] = fn(order, visible, key)
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            arr[i] = np.nan
    return arr


def _sdiv_num_den(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    out[~np.isfinite(out)] = np.nan
    return out


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=float)
    if 0 < k < len(x):
        out[k:] = x[: len(x) - k]
    return out


def _rolling_sum_1d(x: np.ndarray, w: int, min_periods: int = 1) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        ok = np.isfinite(seg)
        if int(ok.sum()) < min_periods:
            continue
        out[t] = float(np.sum(seg[ok]))
    return out


def _rolling_mean_1d(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        ok = np.isfinite(seg)
        if int(ok.sum()) < w:
            continue
        out[t] = float(np.mean(seg[ok]))
    return out


def _rolling_std_1d(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        ok = np.isfinite(seg)
        if int(ok.sum()) < w:
            continue
        out[t] = float(np.std(seg[ok], ddof=1))
    return out


def _pv_of(frame: pl.DataFrame, column: str) -> list:
    return frame[column].to_list()


def _xv_of(frame: pl.DataFrame, column: str) -> np.ndarray:
    return np.asarray(frame[column].to_numpy(), dtype=float)


# ---------------------------------------------------------------------------
# transforms_v2 family
# ---------------------------------------------------------------------------


def _pct_1d(xv, pv, periods):
    p = _pi(periods, "periods")
    return _walk_1d(xv, pv, lambda o, v, c: _safe_div(float(v[c]), _lag_value(o, v, c, p)) - 1.0)


def _lag_walk_1d(xv, pv, periods):
    p = _pi(periods, "periods")
    return _walk_1d(xv, pv, lambda o, v, c: _lag_value(o, v, c, p))


def _rolling_stat_1d(xv, pv, periods, reducer):
    n = _pi(periods, "periods", 2)
    return _walk_1d(
        xv, pv,
        lambda o, v, c: float(reducer(np.asarray(_values(o, v, c, n), dtype=float)))
        if len(_values(o, v, c, n)) == n else np.nan,
    )


def fin_lag(x, period_id, periods=1):
    p = _pi(periods, "periods")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _lag_walk_1d(_xv_of(x, c), _pv_of(period_id, c), p)
    return _make(x, cols, out)


def fin_diff(x, period_id, periods=1):
    p = _pi(periods, "periods")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        out[:, i] = _walk_1d(xv, pv, lambda o, v, cc: float(v[cc]) - _lag_value(o, v, cc, p))
    return _make(x, cols, out)


def fin_pct_change(x, period_id, periods=1):
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _pct_1d(_xv_of(x, c), _pv_of(period_id, c), periods)
    return _make(x, cols, out)


def fin_log_change(x, period_id, periods=1):
    p = _pi(periods, "periods")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        def calc(o, v, cc):
            old = _lag_value(o, v, cc, p)
            cur = float(v[cc])
            return float(np.log(cur / old)) if cur > 0 and old > 0 else np.nan
        out[:, i] = _walk_1d(xv, pv, calc)
    return _make(x, cols, out)


def fin_ttm(x, period_id, periods_per_year=4):
    n = _pi(periods_per_year, "periods_per_year")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        out[:, i] = _walk_1d(
            xv, pv,
            lambda o, v, cc: float(np.sum(_values(o, v, cc, n))) if len(_values(o, v, cc, n)) == n else np.nan,
        )
    return _make(x, cols, out)


def fin_average_balance(x, period_id, periods=2):
    n = _pi(periods, "periods")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        out[:, i] = _walk_1d(
            xv, pv,
            lambda o, v, cc: float(np.mean(_values(o, v, cc, n))) if len(_values(o, v, cc, n)) == n else np.nan,
        )
    return _make(x, cols, out)


def fin_cagr(x, period_id, periods=4, periods_per_year=4):
    p = _pi(periods, "periods")
    ppy = _pi(periods_per_year, "periods_per_year")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        def calc(o, v, cc):
            old = _lag_value(o, v, cc, p)
            cur = float(v[cc])
            if cur <= 0 or old <= 0:
                return np.nan
            return float((cur / old) ** (ppy / p) - 1.0)
        out[:, i] = _walk_1d(xv, pv, calc)
    return _make(x, cols, out)


def fin_growth_acceleration(x, period_id, short_periods=1, long_periods=4):
    s = _pi(short_periods, "short_periods")
    l = _pi(long_periods, "long_periods")
    if s >= l:
        raise ValueError("short_periods must be < long_periods")
    short = fin_pct_change(x, period_id, s)
    long = fin_pct_change(x, period_id, l)
    cols = _cols(short, long)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = short[c].to_numpy() - long[c].to_numpy()
    return _make(x, cols, out)


def fin_growth_change(x, period_id, growth_periods=4, compare_periods=1):
    g = fin_pct_change(x, period_id, _pi(growth_periods, "growth_periods"))
    cp = _pi(compare_periods, "compare_periods")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        g_arr = g[c].to_numpy()
        out[:, i] = g_arr - _lag_walk_1d(g_arr, _pv_of(period_id, c), cp)
    return _make(x, cols, out)


def fin_std(x, period_id, periods=8):
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_stat_1d(_xv_of(x, c), _pv_of(period_id, c), periods, lambda a: np.std(a, ddof=1))
    return _make(x, cols, out)


def fin_mad(x, period_id, periods=8):
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_stat_1d(_xv_of(x, c), _pv_of(period_id, c), periods, lambda a: np.mean(np.abs(a - np.mean(a))))
    return _make(x, cols, out)


def fin_cv(x, period_id, periods=8):
    def reducer(a):
        mean = float(np.mean(a))
        sd = float(np.std(a, ddof=1))
        return sd / abs(mean) if abs(mean) > _EPS else np.nan
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_stat_1d(_xv_of(x, c), _pv_of(period_id, c), periods, reducer)
    return _make(x, cols, out)


def fin_stability(x, period_id, periods=8):
    cv = fin_cv(x, period_id, periods)
    cols = _cols(cv)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = 1.0 / (1.0 + np.abs(cv[c].to_numpy()))
    return _make(x, cols, out)


def fin_range(x, period_id, periods=8):
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_stat_1d(_xv_of(x, c), _pv_of(period_id, c), periods, lambda a: np.max(a) - np.min(a))
    return _make(x, cols, out)


def fin_zscore_history(x, period_id, periods=8):
    n = _pi(periods, "periods", 2)
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        def calc(o, v, cc):
            vals = np.asarray(_values(o, v, cc, n), dtype=float)
            if len(vals) != n:
                return np.nan
            sd = float(np.std(vals, ddof=1))
            return (float(vals[-1]) - float(np.mean(vals))) / sd if sd > _EPS else 0.0
        out[:, i] = _walk_1d(xv, pv, calc)
    return _make(x, cols, out)


def fin_percentile_history(x, period_id, periods=8):
    n = _pi(periods, "periods", 2)
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        def calc(o, v, cc):
            vals = np.asarray(_values(o, v, cc, n), dtype=float)
            if len(vals) != n:
                return np.nan
            return float((np.sum(vals < vals[-1]) + 0.5 * np.sum(vals == vals[-1])) / len(vals))
        out[:, i] = _walk_1d(xv, pv, calc)
    return _make(x, cols, out)


def _trend_stat_1d(xv, pv, periods, which):
    n = _pi(periods, "periods", 3)
    def calc(o, v, cc):
        y = np.asarray(_values(o, v, cc, n), dtype=float)
        if len(y) != n:
            return np.nan
        xx = np.arange(n, dtype=float)
        xb, yb = float(xx.mean()), float(y.mean())
        den = float(np.sum((xx - xb) ** 2))
        if den <= _EPS:
            return np.nan
        slope = float(np.sum((xx - xb) * (y - yb)) / den)
        if which == "slope":
            return slope
        fitted = yb + slope * (xx - xb)
        resid = y - fitted
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - yb) ** 2))
        if which == "r2":
            return 1.0 - ss_res / ss_tot if ss_tot > _EPS else 1.0
        if n <= 2:
            return np.nan
        mse = ss_res / (n - 2)
        se = float(np.sqrt(mse / den)) if mse >= 0 else np.nan
        return slope / se if np.isfinite(se) and se > _EPS else np.nan
    return _walk_1d(xv, pv, calc)


def fin_trend_slope(x, period_id, periods=8):
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _trend_stat_1d(_xv_of(x, c), _pv_of(period_id, c), periods, "slope")
    return _make(x, cols, out)


def fin_trend_r2(x, period_id, periods=8):
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _trend_stat_1d(_xv_of(x, c), _pv_of(period_id, c), periods, "r2")
    return _make(x, cols, out)


def fin_trend_tstat(x, period_id, periods=8):
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _trend_stat_1d(_xv_of(x, c), _pv_of(period_id, c), periods, "tstat")
    return _make(x, cols, out)


def fin_trend_acceleration(x, period_id, short_periods=4, long_periods=8):
    s = _pi(short_periods, "short_periods", 3)
    l = _pi(long_periods, "long_periods", 3)
    if s >= l:
        raise ValueError("short_periods must be < long_periods")
    short = fin_trend_slope(x, period_id, s)
    long = fin_trend_slope(x, period_id, l)
    cols = _cols(short, long)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = short[c].to_numpy() - long[c].to_numpy()
    return _make(x, cols, out)


def fin_monotonicity(x, period_id, periods=8):
    n = _pi(periods, "periods", 2)
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        def calc(o, v, cc):
            vals = np.asarray(_values(o, v, cc, n), dtype=float)
            if len(vals) != n:
                return np.nan
            d = np.diff(vals)
            return float((np.sum(d > 0) - np.sum(d < 0)) / max(1, len(d)))
        out[:, i] = _walk_1d(xv, pv, calc)
    return _make(x, cols, out)


def _streak_1d(xv, pv, positive):
    def calc(o, v, cc):
        vals = np.asarray(_values(o, v, cc, None), dtype=float)
        if len(vals) < 2:
            return 0.0
        d = np.diff(vals)
        count = 0
        for z in d[::-1]:
            if (z > 0) if positive else (z < 0):
                count += 1
            else:
                break
        return float(count)
    return _walk_1d(xv, pv, calc)


def fin_positive_streak(x, period_id, max_periods=8):
    n = _pi(max_periods, "max_periods", 2)
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        def calc(o, v, cc):
            vals = np.asarray(_values(o, v, cc, n), dtype=float)
            if len(vals) < 2:
                return 0.0
            count = 0
            for z in np.diff(vals)[::-1]:
                if z > 0:
                    count += 1
                else:
                    break
            return float(count)
        out[:, i] = _walk_1d(xv, pv, calc)
    return _make(x, cols, out)


def fin_negative_streak(x, period_id, max_periods=8):
    n = _pi(max_periods, "max_periods", 2)
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        def calc(o, v, cc):
            vals = np.asarray(_values(o, v, cc, n), dtype=float)
            if len(vals) < 2:
                return 0.0
            count = 0
            for z in np.diff(vals)[::-1]:
                if z < 0:
                    count += 1
                else:
                    break
            return float(count)
        out[:, i] = _walk_1d(xv, pv, calc)
    return _make(x, cols, out)


def fin_sign_change_count(x, period_id, periods=8):
    n = _pi(periods, "periods", 3)
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, pv = _xv_of(x, c), _pv_of(period_id, c)
        def calc(o, v, cc):
            vals = np.asarray(_values(o, v, cc, n), dtype=float)
            if len(vals) != n:
                return np.nan
            signs = np.sign(np.diff(vals))
            signs = signs[signs != 0]
            return float(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0.0
        out[:, i] = _walk_1d(xv, pv, calc)
    return _make(x, cols, out)


def fin_growth_volatility(x, period_id, growth_periods=1, window_periods=8):
    g = fin_pct_change(x, period_id, _pi(growth_periods, "growth_periods"))
    return fin_std(g, period_id, _pi(window_periods, "window_periods", 2))


def fin_growth_stability(x, period_id, growth_periods=1, window_periods=8):
    vol = fin_growth_volatility(x, period_id, growth_periods, window_periods)
    cols = _cols(vol)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = 1.0 / (1.0 + np.abs(vol[c].to_numpy()))
    return _make(x, cols, out)


def fin_growth_persistence(x, period_id, growth_periods=1, window_periods=8):
    g = fin_pct_change(x, period_id, _pi(growth_periods, "growth_periods"))
    n = _pi(window_periods, "window_periods", 2)
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _walk_1d(
            g[c].to_numpy(), _pv_of(period_id, c),
            lambda o, v, cc: float(np.mean(np.asarray(_values(o, v, cc, n), dtype=float) > 0))
            if len(_values(o, v, cc, n)) == n else np.nan,
        )
    return _make(x, cols, out)


def fin_ratio(numerator, denominator):
    cols = _cols(numerator, denominator)
    rows = numerator.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _sdiv_num_den(numerator[c].to_numpy(), denominator[c].to_numpy())
    return _make(numerator, cols, out)


def fin_turnover(flow, balance, period_id, average_periods=2):
    avg = fin_average_balance(balance, period_id, _pi(average_periods, "average_periods"))
    cols = _cols(flow, balance)
    rows = flow.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _sdiv_num_den(flow[c].to_numpy(), avg[c].to_numpy())
    return _make(flow, cols, out)


def fin_divergence(x, y, period_id, periods=4):
    px = fin_pct_change(x, period_id, _pi(periods, "periods"))
    py = fin_pct_change(y, period_id, _pi(periods, "periods"))
    cols = _cols(px, py)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = px[c].to_numpy() - py[c].to_numpy()
    return _make(x, cols, out)


def fin_cash_earnings_gap(earnings, cashflow, scale_base):
    cols = _cols(earnings, cashflow, scale_base)
    rows = earnings.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _sdiv_num_den(
            earnings[c].to_numpy() - cashflow[c].to_numpy(),
            np.abs(scale_base[c].to_numpy()),
        )
    return _make(earnings, cols, out)


def fin_accrual_ratio(earnings, cashflow, assets):
    cols = _cols(earnings, cashflow, assets)
    rows = earnings.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _sdiv_num_den(
            earnings[c].to_numpy() - cashflow[c].to_numpy(),
            np.abs(assets[c].to_numpy()),
        )
    return _make(earnings, cols, out)


def fin_working_capital_change(working_capital, period_id, periods=1):
    return fin_diff(working_capital, period_id, _pi(periods, "periods"))


# ---------------------------------------------------------------------------
# expectation_v2 family
# ---------------------------------------------------------------------------


def fin_surprise(actual, expected, scale_base):
    cols = _cols(actual, expected, scale_base)
    rows = actual.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _sdiv_num_den(
            actual[c].to_numpy() - expected[c].to_numpy(),
            np.abs(scale_base[c].to_numpy()),
        )
    return _make(actual, cols, out)


def fin_surprise_zscore(actual, expected, scale_base, window_days=252):
    window = _pi(window_days, "window_days", 2)
    surprise = fin_surprise(actual, expected, scale_base)
    cols = _cols(surprise)
    rows = actual.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        arr = surprise[c].to_numpy()
        shifted = _shift(arr, 1)
        mean = _rolling_mean_1d(shifted, window)
        std = _rolling_std_1d(shifted, window)
        out[:, i] = _sdiv_num_den(arr - mean, std)
    return _make(surprise, cols, out)


def fin_surprise_event_zscore(actual, expected, scale_base, period_id, periods=8):
    count = _pi(periods, "periods", 3)
    surprise = fin_surprise(actual, expected, scale_base)
    cols = _cols(surprise, period_id)
    rows = actual.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        def calc(o, v, cc):
            values = np.asarray(_values(o, v, cc, count), dtype=float)
            if len(values) != count:
                return np.nan
            history = values[:-1]
            std = float(np.std(history, ddof=1))
            if not np.isfinite(std) or std <= _EPS:
                return np.nan
            return float((values[-1] - np.mean(history)) / std)
        out[:, i] = _walk_1d(surprise[c].to_numpy(), _pv_of(period_id, c), calc)
    return _make(surprise, cols, out)


def fin_surprise_event_percentile(actual, expected, scale_base, period_id, periods=8):
    count = _pi(periods, "periods", 2)
    surprise = fin_surprise(actual, expected, scale_base)
    cols = _cols(surprise, period_id)
    rows = actual.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        def calc(o, v, cc):
            values = np.asarray(_values(o, v, cc, count), dtype=float)
            if len(values) != count:
                return np.nan
            current_value = values[-1]
            return float((np.sum(values < current_value) + 0.5 * np.sum(values == current_value)) / len(values))
        out[:, i] = _walk_1d(surprise[c].to_numpy(), _pv_of(period_id, c), calc)
    return _make(surprise, cols, out)


def _revision_masks_1d(xv, pv):
    """Polars mirror of the pandas complete-data revision masks (R4-26).

    A revision can only be asserted when current value, prior value, current
    period id and prior period id are ALL present.  A missing input (or a prior
    observation missing because of a data gap) makes the row undetermined -> the
    caller must emit NaN, never a guessed 0.  The first row of the panel is a
    valid baseline (no prior disclosure to revise -> complete, "no revision").
    """
    n = len(xv)
    prev_x = _shift(xv, 1)
    pid_keys = [_period_key(raw) for raw in pv]
    prev_pid = [None] + pid_keys[:-1]
    first_row = np.zeros(n, dtype=bool)
    first_row[0] = True
    prev_ok = np.isfinite(prev_x) & np.array(
        [key is not None for key in prev_pid], dtype=bool
    )
    complete = np.isfinite(xv) & np.array(
        [key is not None for key in pid_keys], dtype=bool
    ) & (prev_ok | first_row)
    same = np.array(
        [pid_keys[t] == prev_pid[t] for t in range(n)], dtype=bool
    ) & prev_ok
    delta = xv - prev_x
    changed = (np.abs(delta) > 0) & prev_ok
    return complete, same, changed, delta


def _revision_event_1d(xv, pv) -> np.ndarray:
    complete, same, changed, _ = _revision_masks_1d(xv, pv)
    return complete & same & changed


def _revision_complete_1d(xv, pv) -> np.ndarray:
    complete, _, _, _ = _revision_masks_1d(xv, pv)
    return complete


def fin_expectation_revision(expected, target_period_id):
    cols = _cols(expected, target_period_id)
    rows = expected.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        arr = expected[c].to_numpy()
        pv = _pv_of(target_period_id, c)
        complete, same, changed, delta = _revision_masks_1d(arr, pv)
        revision = complete & same & changed
        # 0 only when the data is complete and a same-target revision is
        # confirmed absent; missing inputs -> NaN (review R4-26).
        out[:, i] = np.where(revision, delta, 0.0)
        out[:, i] = np.where(complete, out[:, i], np.nan)
    return _make(expected, cols, out)


def fin_expectation_revision_pct(expected, target_period_id):
    cols = _cols(expected, target_period_id)
    rows = expected.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        arr = expected[c].to_numpy()
        pv = _pv_of(target_period_id, c)
        complete, same, changed, _ = _revision_masks_1d(arr, pv)
        revision = complete & same & changed
        prev_x = _shift(arr, 1)
        denom = np.where(prev_x != 0, prev_x, np.nan)
        pct = _sdiv_num_den(arr, denom) - 1.0
        pct[~np.isfinite(pct)] = np.nan
        out[:, i] = np.where(revision, pct, 0.0)
        out[:, i] = np.where(complete, out[:, i], np.nan)
    return _make(expected, cols, out)


def _rolling_sum_nanmin_1d(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling_sum_1d(x, w, 1)


def fin_expectation_revision_speed(expected, target_period_id, window_days=60):
    window = _pi(window_days, "window_days", 2)
    revision = fin_expectation_revision_pct(expected, target_period_id)
    cols = _cols(revision)
    rows = expected.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        complete = _revision_complete_1d(
            expected[c].to_numpy(), _pv_of(target_period_id, c)
        )
        speed = _rolling_sum_nanmin_1d(revision[c].to_numpy(), window)
        out[:, i] = np.where(complete, speed, np.nan)
    return _make(revision, cols, out)


def fin_expectation_revision_count(expected, target_period_id, window_days=60):
    window = _pi(window_days, "window_days", 2)
    cols = _cols(expected, target_period_id)
    rows = expected.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv = expected[c].to_numpy()
        pv = _pv_of(target_period_id, c)
        complete = _revision_complete_1d(xv, pv)
        event = _revision_event_1d(xv, pv)
        count = _rolling_sum_1d(event.astype(float), window, 1)
        out[:, i] = np.where(complete, count, np.nan)
    return _make(expected, cols, out)


def fin_expectation_revision_magnitude(expected, target_period_id, window_days=60):
    window = _pi(window_days, "window_days", 2)
    revision = fin_expectation_revision_pct(expected, target_period_id)
    cols = _cols(revision)
    rows = expected.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        complete = _revision_complete_1d(
            expected[c].to_numpy(), _pv_of(target_period_id, c)
        )
        magnitude = _rolling_sum_nanmin_1d(np.abs(revision[c].to_numpy()), window)
        out[:, i] = np.where(complete, magnitude, np.nan)
    return _make(revision, cols, out)


def fin_days_since_expectation_revision(expected, target_period_id, max_days=252):
    cap = _pi(max_days, "max_days")
    cols = _cols(expected, target_period_id)
    rows = expected.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv = expected[c].to_numpy()
        pv = _pv_of(target_period_id, c)
        event = _revision_event_1d(xv, pv)
        # Observed clock (R4-27 / P1-06): never age blindly across an
        # unobservable gap.  A gap row -> NaN and the age reference is reset; a
        # complete non-event after a gap stays censored (NaN, NOT 0 — a 0 would
        # read as "revision happened today") and only resumes at a NEW revision
        # event (the ``event`` branch emits 0 and restarts the clock).
        # ``age`` starts at ``cap`` (matching the pandas reference) so the first
        # complete non-event row reads as ``cap``, not 0.
        age = cap
        arr = np.full(rows, np.nan, dtype=float)
        for t in range(rows):
            complete = bool(np.isfinite(xv[t]) and _period_key(pv[t]) is not None)
            if not complete:
                age = None
                continue
            if bool(event[t]):
                arr[t] = 0.0
                age = 0
            elif age is not None:
                age = min(cap, age + 1)
                arr[t] = float(age)
            else:
                # P1-06: after a gap the revision age is censored — output NaN,
                # NOT 0.  Only a new revision event resumes the clock.
                arr[t] = np.nan
                age = None
        out[:, i] = arr
    return _make(expected, cols, out)


def fin_expectation_dispersion(expected_std, expected_mean):
    cols = _cols(expected_std, expected_mean)
    rows = expected_std.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _sdiv_num_den(
            np.abs(expected_std[c].to_numpy()),
            np.abs(expected_mean[c].to_numpy()),
        )
    return _make(expected_std, cols, out)


def _beat_miss_streak_1d(diff_arr, pv, count, beat):
    def calc(o, v, cc):
        values = _values(o, v, cc, count)
        streak = 0
        for value in values[::-1]:
            condition = value > 0 if beat else value < 0
            if not condition:
                break
            streak += 1
        return float(streak)
    return _walk_1d(diff_arr, pv, calc)


def fin_beat_streak(actual, expected, period_id, max_periods=8):
    count = _pi(max_periods, "max_periods", 2)
    cols = _cols(actual, expected, period_id)
    rows = actual.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        diff = actual[c].to_numpy() - expected[c].to_numpy()
        out[:, i] = _beat_miss_streak_1d(diff, _pv_of(period_id, c), count, True)
    return _make(actual, cols, out)


def fin_miss_streak(actual, expected, period_id, max_periods=8):
    count = _pi(max_periods, "max_periods", 2)
    cols = _cols(actual, expected, period_id)
    rows = actual.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        diff = actual[c].to_numpy() - expected[c].to_numpy()
        out[:, i] = _beat_miss_streak_1d(diff, _pv_of(period_id, c), count, False)
    return _make(actual, cols, out)


# ---------------------------------------------------------------------------
# flow_semantics family
# ---------------------------------------------------------------------------


def fin_quarter_from_cumulative(x, period_id, fiscal_quarter):
    """Convert fiscal YTD cumulative values to one-quarter flows (strict kernel)."""
    return pl_quarter_from_cumulative(
        x,
        period_id,
        fiscal_quarter=fiscal_quarter,
        revision_policy="latest_available",
    )


def fin_ttm_quarterly(x, period_id, periods_per_year=4):
    """TTM over the latest consecutive single-period flow values (strict kernel)."""
    return pl_ttm_from_quarterly(
        x,
        period_id,
        periods=_pi(periods_per_year, "periods_per_year"),
        require_consecutive=True,
        revision_policy="latest_available",
    )


def fin_ttm_cumulative(x, period_id, fiscal_quarter, periods_per_year=4):
    """Convert fiscal YTD cumulative values to quarters, then calculate TTM."""
    quarterly = pl_quarter_from_cumulative(
        x,
        period_id,
        fiscal_quarter=fiscal_quarter,
        revision_policy="latest_available",
    )
    return pl_ttm_from_quarterly(
        quarterly,
        period_id,
        periods=_pi(periods_per_year, "periods_per_year"),
        require_consecutive=True,
        revision_policy="latest_available",
    )


# ---------------------------------------------------------------------------
# seasonal family (fiscal ordinal)
# ---------------------------------------------------------------------------


def _quarter_number(value):
    if value is None:
        return None
    try:
        v = float(value)
        if np.isfinite(v) and int(v) == v and 1 <= int(v) <= 4:
            return int(v)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        s = value.strip().upper()
        if s in {"Q1", "1Q"}: return 1
        if s in {"Q2", "2Q"}: return 2
        if s in {"Q3", "3Q"}: return 3
        if s in {"Q4", "4Q"}: return 4
    return None


def _seasonal_history_1d(xv, pv, qv, years, min_history):
    n = len(xv)
    order: list = []
    visible: OrderedDict = OrderedDict()
    quarters: OrderedDict = OrderedDict()
    arr = np.full(n, np.nan, dtype=float)
    for t, (value, raw_period, raw_quarter) in enumerate(zip(xv, pv, qv)):
        key = _period_key(raw_period)
        quarter = _quarter_number(raw_quarter)
        if key is not None and np.isfinite(value) and quarter is not None:
            if key not in visible:
                order.append(key)
            visible[key] = float(value)
            quarters[key] = quarter
        if key is None or key not in visible or key not in quarters:
            continue
        cur_quarter = quarters[key]
        position = order.index(key)
        prior_keys = order[:position]
        prior = []
        for pk in prior_keys:
            if quarters.get(pk) == cur_quarter and np.isfinite(visible.get(pk, np.nan)):
                prior.append(float(visible[pk]))
        prior = prior[-(years * 4):] if years * 4 > 0 else prior
        if len(prior) >= min_history and np.isfinite(float(visible[key])):
            arr[t] = (float(visible[key]), list(prior), len(prior))
        else:
            arr[t] = None
    return arr, visible, quarters


def fin_seasonal_zscore(x, period_id, fiscal_quarter, years=5, min_history=2, revision_policy="latest_available"):
    years = _pi(years, "years")
    min_history = _pi(min_history, "min_history")
    cols = _cols(x, period_id, fiscal_quarter)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv = _xv_of(x, c)
        pv = _pv_of(period_id, c)
        qv = _pv_of(fiscal_quarter, c)
        order: list = []
        visible: OrderedDict = OrderedDict()
        quarters: OrderedDict = OrderedDict()
        for t, (value, raw_period, raw_quarter) in enumerate(zip(xv, pv, qv)):
            key = _period_key(raw_period)
            quarter = _quarter_number(raw_quarter)
            if key is not None and np.isfinite(value) and quarter is not None:
                if key not in visible:
                    order.append(key)
                visible[key] = float(value)
                quarters[key] = quarter
            if key is None or key not in visible or key not in quarters:
                continue
            cur_quarter = quarters[key]
            position = order.index(key)
            prior = [
                float(visible[pk]) for pk in order[:position]
                if quarters.get(pk) == cur_quarter and np.isfinite(visible.get(pk, np.nan))
            ]
            prior = prior[-(years * 4):]
            if len(prior) >= min_history and np.isfinite(float(visible[key])):
                sd = float(np.std(prior, ddof=1)) if len(prior) > 1 else np.nan
                if np.isfinite(sd) and sd > _EPS:
                    out[t, i] = (float(visible[key]) - float(np.mean(prior))) / sd
    return _make(x, cols, out)


def fin_seasonal_percentile(x, period_id, fiscal_quarter, years=5, min_history=2, revision_policy="latest_available"):
    years = _pi(years, "years")
    min_history = _pi(min_history, "min_history")
    cols = _cols(x, period_id, fiscal_quarter)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv = _xv_of(x, c)
        pv = _pv_of(period_id, c)
        qv = _pv_of(fiscal_quarter, c)
        order: list = []
        visible: OrderedDict = OrderedDict()
        quarters: OrderedDict = OrderedDict()
        for t, (value, raw_period, raw_quarter) in enumerate(zip(xv, pv, qv)):
            key = _period_key(raw_period)
            quarter = _quarter_number(raw_quarter)
            if key is not None and np.isfinite(value) and quarter is not None:
                if key not in visible:
                    order.append(key)
                visible[key] = float(value)
                quarters[key] = quarter
            if key is None or key not in visible or key not in quarters:
                continue
            cur_quarter = quarters[key]
            position = order.index(key)
            prior = [
                float(visible[pk]) for pk in order[:position]
                if quarters.get(pk) == cur_quarter and np.isfinite(visible.get(pk, np.nan))
            ]
            prior = prior[-(years * 4):]
            if len(prior) >= min_history and np.isfinite(float(visible[key])):
                current = float(visible[key])
                out[t, i] = (sum(value < current for value in prior) + 0.5 * sum(value == current for value in prior)) / len(prior)
    return _make(x, cols, out)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("fin_lag", ("x", "period_id", "periods"), fin_lag, "Lag by visible reporting periods, never by trading days."),
    ("fin_diff", ("x", "period_id", "periods"), fin_diff, "Difference versus a prior visible report period."),
    ("fin_pct_change", ("x", "period_id", "periods"), fin_pct_change, "Percent change versus a prior visible report period."),
    ("fin_log_change", ("x", "period_id", "periods"), fin_log_change, "Log change versus a prior visible report period."),
    ("fin_qoq", ("x", "period_id"), lambda x, period_id: fin_pct_change(x, period_id, 1), "Quarter-over-quarter percent change."),
    ("fin_yoy", ("x", "period_id", "periods_per_year"), lambda x, period_id, periods_per_year=4: fin_pct_change(x, period_id, _pi(periods_per_year, "periods_per_year")), "Year-over-year percent change."),
    ("fin_ttm", ("x", "period_id", "periods_per_year"), fin_ttm, "Trailing-twelve-months sum."),
    ("fin_average_balance", ("x", "period_id", "periods"), fin_average_balance, "Average balance over visible periods."),
    ("fin_growth", ("x", "period_id", "periods"), lambda x, period_id, periods=1: fin_pct_change(x, period_id, periods), "Period-over-period growth."),
    ("fin_cagr", ("x", "period_id", "periods", "periods_per_year"), fin_cagr, "Compound annual growth rate."),
    ("fin_growth_acceleration", ("x", "period_id", "short_periods", "long_periods"), fin_growth_acceleration, "Short minus long growth."),
    ("fin_growth_change", ("x", "period_id", "growth_periods", "compare_periods"), fin_growth_change, "Growth change versus prior report period."),
    ("fin_std", ("x", "period_id", "periods"), fin_std, "Sample std of recent visible report values."),
    ("fin_mad", ("x", "period_id", "periods"), fin_mad, "Mean absolute deviation of recent visible report values."),
    ("fin_cv", ("x", "period_id", "periods"), fin_cv, "Coefficient of variation."),
    ("fin_stability", ("x", "period_id", "periods"), fin_stability, "Inverse-coefficient stability."),
    ("fin_range", ("x", "period_id", "periods"), fin_range, "Range of recent visible report values."),
    ("fin_zscore_history", ("x", "period_id", "periods"), fin_zscore_history, "Latest value z-score against recent report values."),
    ("fin_percentile_history", ("x", "period_id", "periods"), fin_percentile_history, "Latest value percentile among recent report values."),
    ("fin_trend_slope", ("x", "period_id", "periods"), fin_trend_slope, "OLS slope of recent visible report values."),
    ("fin_trend_r2", ("x", "period_id", "periods"), fin_trend_r2, "R-squared of the recent report-value trend."),
    ("fin_trend_tstat", ("x", "period_id", "periods"), fin_trend_tstat, "t-statistic of the recent report-value trend."),
    ("fin_trend_acceleration", ("x", "period_id", "short_periods", "long_periods"), fin_trend_acceleration, "Short minus long trend slope."),
    ("fin_monotonicity", ("x", "period_id", "periods"), fin_monotonicity, "Signed monotonicity of recent report values."),
    ("fin_positive_streak", ("x", "period_id", "max_periods"), fin_positive_streak, "Bounded streak of positive report-to-report changes."),
    ("fin_negative_streak", ("x", "period_id", "max_periods"), fin_negative_streak, "Bounded streak of negative report-to-report changes."),
    ("fin_sign_change_count", ("x", "period_id", "periods"), fin_sign_change_count, "Count of sign changes among recent report deltas."),
    ("fin_growth_volatility", ("x", "period_id", "growth_periods", "window_periods"), fin_growth_volatility, "Std of growth over report-period window."),
    ("fin_growth_stability", ("x", "period_id", "growth_periods", "window_periods"), fin_growth_stability, "Inverse volatility of growth."),
    ("fin_growth_persistence", ("x", "period_id", "growth_periods", "window_periods"), fin_growth_persistence, "Fraction of recent periods with positive growth."),
    ("fin_ratio", ("numerator", "denominator"), fin_ratio, "Ratio with zero/infinity handled as missing."),
    ("fin_common_size", ("x", "base"), lambda x, base: fin_ratio(x, base), "Common-size ratio."),
    ("fin_turnover", ("flow", "balance", "period_id", "average_periods"), fin_turnover, "Flow divided by average balance."),
    ("fin_divergence", ("x", "y", "period_id", "periods"), fin_divergence, "Difference of same-period percent changes."),
    ("fin_cash_earnings_gap", ("earnings", "cashflow", "scale_base"), fin_cash_earnings_gap, "Scaled earnings-minus-cashflow gap."),
    ("fin_accrual_ratio", ("earnings", "cashflow", "assets"), fin_accrual_ratio, "Accrual ratio scaled by absolute assets."),
    ("fin_cash_conversion", ("cashflow", "earnings"), lambda cashflow, earnings: fin_ratio(cashflow, earnings), "Cash conversion ratio."),
    ("fin_working_capital_change", ("working_capital", "period_id", "periods"), fin_working_capital_change, "Working-capital period change."),
    ("fin_surprise", ("actual", "expected", "scale_base"), fin_surprise, "Scaled actual-minus-expectation surprise."),
    ("fin_surprise_zscore", ("actual", "expected", "scale_base", "window_days"), fin_surprise_zscore, "Daily-observation prior-window z-score of realized surprise."),
    ("fin_surprise_event_zscore", ("actual", "expected", "scale_base", "period_id", "periods"), fin_surprise_event_zscore, "Report-event surprise z-score over distinct visible periods."),
    ("fin_surprise_event_percentile", ("actual", "expected", "scale_base", "period_id", "periods"), fin_surprise_event_percentile, "Report-event surprise percentile over distinct visible periods."),
    ("fin_expectation_revision", ("expected", "target_period_id"), fin_expectation_revision, "Same-target-period change in analyst expectation."),
    ("fin_expectation_revision_pct", ("expected", "target_period_id"), fin_expectation_revision_pct, "Same-target-period percent expectation revision."),
    ("fin_expectation_revision_speed", ("expected", "target_period_id", "window_days"), fin_expectation_revision_speed, "Bounded cumulative expectation revision speed."),
    ("fin_expectation_revision_count", ("expected", "target_period_id", "window_days"), fin_expectation_revision_count, "Count of same-target estimate changes in a bounded daily window."),
    ("fin_expectation_revision_magnitude", ("expected", "target_period_id", "window_days"), fin_expectation_revision_magnitude, "Absolute expectation revision accumulated in a bounded daily window."),
    ("fin_days_since_expectation_revision", ("expected", "target_period_id", "max_days"), fin_days_since_expectation_revision, "Bounded trading days since the latest same-target estimate revision."),
    ("fin_expectation_dispersion", ("expected_std", "expected_mean"), fin_expectation_dispersion, "Consensus dispersion scaled by absolute consensus mean."),
    ("fin_actual_expectation_divergence", ("actual", "expected", "scale_base"), lambda actual, expected, scale_base: fin_surprise(actual, expected, scale_base), "Generic actual/expectation divergence."),
    ("fin_beat_streak", ("actual", "expected", "period_id", "max_periods"), fin_beat_streak, "Bounded consecutive positive surprise streak."),
    ("fin_miss_streak", ("actual", "expected", "period_id", "max_periods"), fin_miss_streak, "Bounded consecutive negative surprise streak."),
    ("fin_revision_delta", ("x", "period_id"), lambda x, period_id: _revision_compose(x, period_id, "delta"), "Value change while the visible report period is unchanged."),
    ("fin_revision_pct", ("x", "period_id"), lambda x, period_id: _revision_compose(x, period_id, "pct"), "Percent revision while the visible report period is unchanged."),
    ("fin_revision_direction", ("x", "period_id"), lambda x, period_id: _revision_compose(x, period_id, "direction"), "Sign of the latest same-period revision."),
    ("fin_revision_count", ("x", "period_id", "window_days"), lambda x, period_id, window_days=252: _revision_compose(x, period_id, "count", window_days=window_days), "Count of visible revisions in a bounded trading-day window."),
    ("fin_revision_magnitude", ("x", "period_id", "window_days"), lambda x, period_id, window_days=252: _revision_compose(x, period_id, "magnitude", window_days=window_days), "Absolute revision magnitude accumulated over a bounded window."),
    ("fin_restated_flag", ("x", "period_id", "window_days"), lambda x, period_id, window_days=252: _revision_compose(x, period_id, "restated", window_days=window_days), "Whether a same-period revision occurred in the bounded window."),
    ("fin_days_since_update", ("x", "period_id", "max_days"), lambda x, period_id, max_days=504: _revision_compose(x, period_id, "days_since_update", max_days=max_days), "Bounded trading days since report-period or value update."),
    ("fin_staleness", ("x", "period_id", "max_days"), lambda x, period_id, max_days=504: _revision_compose(x, period_id, "days_since_update", max_days=max_days), "Bounded accounting-data staleness in trading days."),
    ("fin_ttm_quarterly", ("x", "period_id", "periods_per_year"), fin_ttm_quarterly, "Sum the latest complete set of single-period flow observations."),
    ("fin_quarter_from_cumulative", ("x", "period_id", "fiscal_quarter"), fin_quarter_from_cumulative, "Convert fiscal YTD cumulative values to one-quarter flows."),
    ("fin_ttm_cumulative", ("x", "period_id", "fiscal_quarter", "periods_per_year"), fin_ttm_cumulative, "Fiscal YTD cumulative values converted to TTM."),
    ("fin_seasonal_zscore", ("x", "period_id", "fiscal_quarter", "years", "min_history", "revision_policy"), fin_seasonal_zscore, "Causal same-quarter fiscal z-score."),
    ("fin_seasonal_percentile", ("x", "period_id", "fiscal_quarter", "years", "min_history", "revision_policy"), fin_seasonal_percentile, "Causal same-quarter fiscal percentile."),
)


def _revision_compose(x, period_id, which, window_days=252, max_days=504):
    if which == "delta":
        cols = _cols(x, period_id)
        rows = x.height
        out = np.full((rows, len(cols)), np.nan, dtype=float)
        for i, c in enumerate(cols):
            xv = _xv_of(x, c)
            pv = _pv_of(period_id, c)
            complete, same, changed, delta = _revision_masks_1d(xv, pv)
            revision = complete & same & changed
            # 0 only when data is complete and a same-period revision is
            # confirmed absent; missing inputs -> NaN (review R4-26).
            values = np.where(revision, delta, 0.0)
            out[:, i] = np.where(complete, values, np.nan)
        return _make(x, cols, out)
    if which == "pct":
        cols = _cols(x, period_id)
        rows = x.height
        out = np.full((rows, len(cols)), np.nan, dtype=float)
        for i, c in enumerate(cols):
            xv = _xv_of(x, c)
            pv = _pv_of(period_id, c)
            complete, same, changed, _ = _revision_masks_1d(xv, pv)
            revision = complete & same & changed
            prev_x = _shift(xv, 1)
            denom = np.where(prev_x != 0, prev_x, np.nan)
            pct = _sdiv_num_den(xv, denom) - 1.0
            pct[~np.isfinite(pct)] = np.nan
            values = np.where(revision, pct, 0.0)
            out[:, i] = np.where(complete, values, np.nan)
        return _make(x, cols, out)
    if which == "direction":
        delta = _revision_compose(x, period_id, "delta")
        cols = _cols(delta)
        rows = x.height
        out = np.full((rows, len(cols)), np.nan, dtype=float)
        for i, c in enumerate(cols):
            out[:, i] = np.sign(delta[c].to_numpy())
        return _make(delta, cols, out)
    window = _pi(window_days, "window_days")
    if which == "count":
        cols = _cols(x, period_id)
        rows = x.height
        out = np.full((rows, len(cols)), np.nan, dtype=float)
        for i, c in enumerate(cols):
            xv = _xv_of(x, c)
            pv = _pv_of(period_id, c)
            complete = _revision_complete_1d(xv, pv)
            event = _revision_event_1d(xv, pv)
            count = _rolling_sum_1d(event.astype(float), window, 1)
            out[:, i] = np.where(complete, count, np.nan)
        return _make(x, cols, out)
    if which == "magnitude":
        pct = _revision_compose(x, period_id, "pct")
        cols = _cols(pct)
        rows = x.height
        out = np.full((rows, len(cols)), np.nan, dtype=float)
        for i, c in enumerate(cols):
            xv = _xv_of(x, c)
            pv = _pv_of(period_id, c)
            complete = _revision_complete_1d(xv, pv)
            magnitude = _rolling_sum_nanmin_1d(np.abs(pct[c].to_numpy()), window)
            out[:, i] = np.where(complete, magnitude, np.nan)
        return _make(pct, cols, out)
    if which == "restated":
        count = _revision_compose(x, period_id, "count", window_days=window_days)
        cols = _cols(count)
        rows = x.height
        out = np.full((rows, len(cols)), np.nan, dtype=float)
        for i, c in enumerate(cols):
            arr = count[c].to_numpy()
            flag = np.where(arr > 0, 1.0, 0.0)
            out[:, i] = np.where(np.isfinite(arr), flag, np.nan)
        return _make(count, cols, out)
    # days_since_update / staleness — observed clock (R4-27): never age blindly
    # across an unobservable gap.  A missing current row emits NaN and resets the
    # age reference; a resumed observation after a gap is treated as a fresh
    # update boundary (age 0) because the age cannot be confirmed across it.
    cap = _pi(max_days, "max_days")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv = _xv_of(x, c)
        pv = _pv_of(period_id, c)
        age = None
        last_x = None
        last_pid = None
        arr = np.full(rows, np.nan, dtype=float)
        for t in range(rows):
            complete = bool(np.isfinite(xv[t]) and _period_key(pv[t]) is not None)
            if not complete:
                # Cannot observe an update event today -> age is unknown.
                age = None
                continue
            if last_x is None:
                # First complete observation: the value just became visible.
                arr[t] = 0.0
                age = 0
                last_x, last_pid = xv[t], pv[t]
                continue
            update_event = bool(last_pid != pv[t] or last_x != xv[t])
            if update_event:
                arr[t] = 0.0
                age = 0
            elif age is not None:
                age = min(cap, age + 1)
                arr[t] = float(age)
            else:
                # Observed-clock resume after a gap: fresh reference (age 0).
                arr[t] = 0.0
                age = 0
            last_x, last_pid = xv[t], pv[t]
        out[:, i] = arr
    return _make(x, cols, out)


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="fundamental_period",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["fundamental", "period_aware", "pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsFundamental_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="fundamental_period",
        business_category="fundamental",
        canonical=name,
        source="polars_fundamental",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
