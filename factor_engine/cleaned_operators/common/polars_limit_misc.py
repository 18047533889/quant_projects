# -*- coding: utf-8 -*-
"""Native Polars backends for limit-state, benchmark, cash-flow-stage, date-diff
and event-decay helpers.

Elementwise / simple per-column NumPy kernels over polars column arrays; results
wrapped into ``pl.DataFrame`` without constructing pandas DataFrames.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


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


def _safe_ratio_1d(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    out[~np.isfinite(out)] = np.nan
    return out


def _touch_1d(close, upper, tolerance):
    valid = np.isfinite(close) & np.isfinite(upper)
    return np.where(valid, (close >= upper - tolerance).astype(float), np.nan)


def _rolling_max_1d(x: np.ndarray, w: int, min_periods: int = 1) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        ok = np.isfinite(seg)
        if int(ok.sum()) < min_periods:
            continue
        out[t] = float(np.max(seg[ok]))
    return out


# ---------------------------------------------------------------------------
# limit-state
# ---------------------------------------------------------------------------


def ashare_limit_distance(close, upper_limit):
    cols = _cols(close, upper_limit)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _safe_ratio_1d(close[c].to_numpy(), upper_limit[c].to_numpy()) - 1.0
    return _make(close, cols, out)


def ashare_limit_touch(close, upper_limit, tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(close, upper_limit)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _touch_1d(close[c].to_numpy(), upper_limit[c].to_numpy(), tolerance)
    return _make(close, cols, out)


def ashare_limit_open_break(open_px, upper_limit):
    cols = _cols(open_px, upper_limit)
    rows = open_px.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        op, ul = open_px[c].to_numpy(), upper_limit[c].to_numpy()
        valid = np.isfinite(op) & np.isfinite(ul)
        out[:, i] = np.where(valid, (op >= ul).astype(float), np.nan)
    return _make(open_px, cols, out)


def ashare_limit_one_price(close, upper_limit, lower_limit):
    cols = _cols(close, upper_limit, lower_limit)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cl, ul, ll = (f[c].to_numpy() for f in (close, upper_limit, lower_limit))
        valid = np.isfinite(cl) & np.isfinite(ul) & np.isfinite(ll)
        out[:, i] = np.where(valid, ((cl >= ul) & (cl <= ll)).astype(float), np.nan)
    return _make(close, cols, out)


def ashare_limit_failed(close, upper_limit, window=20):
    w = _pi(window, "window")
    cols = _cols(close, upper_limit)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cl, ul = close[c].to_numpy(), upper_limit[c].to_numpy()
        touch = (cl >= ul).astype(float)
        ever_touched = _rolling_max_1d(touch, w, 1)
        valid = np.isfinite(cl) & np.isfinite(ul) & np.isfinite(ever_touched)
        out[:, i] = np.where(valid, ((ever_touched > 0) & (cl < ul)).astype(float), np.nan)
    return _make(close, cols, out)


# ---------------------------------------------------------------------------
# benchmark / masks / cash-flow stage
# ---------------------------------------------------------------------------


def benchmark_excess_return(ret, benchmark_ret):
    cols = _cols(ret, benchmark_ret)
    rows = ret.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = ret[c].to_numpy() - benchmark_ret[c].to_numpy()
    return _make(ret, cols, out)


def benchmark_relative_price(numerator, denominator):
    cols = _cols(numerator, denominator)
    rows = numerator.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _safe_ratio_1d(numerator[c].to_numpy(), denominator[c].to_numpy())
    return _make(numerator, cols, out)


def fin_applicability_mask(value, threshold=0.0):
    thr = _pf(threshold, "threshold", None)
    cols = _cols(value)
    rows = value.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        values = value[c].to_numpy()
        out[:, i] = np.where(np.isfinite(values) & (values > thr), 1.0, 0.0)
    return _make(value, cols, out)


def cash_flow_lifecycle_stage(operating, investing, financing):
    cols = _cols(operating, investing, financing)
    rows = operating.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        o = operating[c].to_numpy()
        iv = investing[c].to_numpy()
        f = financing[c].to_numpy()
        valid = np.isfinite(o) & np.isfinite(iv) & np.isfinite(f)
        out[:, i] = np.where(valid & (o > 0) & (iv < 0) & (f < 0), 1.0,
            np.where(valid & (o > 0) & (iv < 0) & (f >= 0), 2.0,
            np.where(valid & (o <= 0) & (iv < 0), 3.0,
            np.where(valid & (o <= 0) & (iv >= 0), 4.0,
            np.where(valid & (o > 0) & (iv >= 0) & (f < 0), 5.0, np.nan)))))
    return _make(operating, cols, out)


# ---------------------------------------------------------------------------
# date diffs / announcement lag
# ---------------------------------------------------------------------------


def _date_diff_1d(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(d1, dtype="datetime64[ns]")
    b_arr = np.asarray(d2, dtype="datetime64[ns]")
    n = len(a_arr)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        if np.isnat(a_arr[t]) or np.isnat(b_arr[t]):
            continue
        days = (a_arr[t] - b_arr[t]) / np.timedelta64(1, "D")
        out[t] = float(days)
    return out


def date_diff_days(left, right):
    cols = _cols(left, right)
    rows = left.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _date_diff_1d(left[c].to_list(), right[c].to_list())
    return _make(left, cols, out)


def trading_day_diff(date1, date2):
    return date_diff_days(date1, date2)


def fin_announcement_lag(period_end_date, pub_date):
    return date_diff_days(period_end_date, pub_date)


# ---------------------------------------------------------------------------
# event decay
# ---------------------------------------------------------------------------


def event_decay_asof(event, half_life=20.0):
    hl = max(float(half_life), 1.0)
    weight = 0.5 ** (1.0 / hl)
    cols = _cols(event)
    rows = event.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        ev = event[c].to_numpy()
        truth = np.isfinite(ev) & (ev != 0)
        series = np.where(truth, 1.0, 0.0)
        acc = 0.0
        for t in range(rows):
            acc = acc * weight + series[t]
            out[t, i] = acc
    return _make(event, cols, out)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("ashare_limit_distance", ("close", "upper_limit"), ashare_limit_distance, "Distance to upper limit."),
    ("ashare_limit_touch", ("close", "upper_limit", "tick_tolerance"), ashare_limit_touch, "Close touched upper limit within tolerance."),
    ("ashare_limit_open_break", ("open", "upper_limit"), ashare_limit_open_break, "Open broke the upper limit."),
    ("ashare_limit_one_price", ("close", "upper_limit", "lower_limit"), ashare_limit_one_price, "One-price limit lock."),
    ("ashare_limit_failed", ("close", "upper_limit", "window"), ashare_limit_failed, "Failed limit-up after touching."),
    ("benchmark_excess_return", ("ret", "benchmark_ret"), benchmark_excess_return, "Return minus benchmark return."),
    ("benchmark_relative_price", ("numerator", "denominator"), benchmark_relative_price, "Numerator over denominator."),
    ("fin_applicability_mask", ("value", "threshold"), fin_applicability_mask, "1 where value exceeds threshold."),
    ("cash_flow_lifecycle_stage", ("operating", "investing", "financing"), cash_flow_lifecycle_stage, "Cash-flow lifecycle stage 1-5."),
    ("event_decay_asof", ("event", "half_life"), event_decay_asof, "Exponential decay of past events."),
)


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="math",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsLimitMisc_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="math",
        business_category="elementwise_math",
        canonical=name,
        source="polars_limit_misc",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
