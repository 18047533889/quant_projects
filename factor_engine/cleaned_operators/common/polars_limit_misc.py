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


def ashare_limit_up_touch(high, upper_limit, tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(high, upper_limit)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _touch_1d(high[c].to_numpy(), upper_limit[c].to_numpy(), tolerance)
    return _make(high, cols, out)


def ashare_limit_down_touch(low, lower_limit, tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(low, lower_limit)
    rows = low.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        lo, ll = low[c].to_numpy(), lower_limit[c].to_numpy()
        valid = np.isfinite(lo) & np.isfinite(ll)
        out[:, i] = np.where(valid, (lo <= ll + tolerance).astype(float), np.nan)
    return _make(low, cols, out)


def ashare_open_at_upper_limit(open_px, upper_limit, tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(open_px, upper_limit)
    rows = open_px.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        op, ul = open_px[c].to_numpy(), upper_limit[c].to_numpy()
        valid = np.isfinite(op) & np.isfinite(ul)
        out[:, i] = np.where(valid, (op >= ul - tolerance).astype(float), np.nan)
    return _make(open_px, cols, out)


def ashare_limit_open_failed(open_px, low, upper_limit, tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(open_px, low, upper_limit)
    rows = open_px.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        op, lo, ul = open_px[c].to_numpy(), low[c].to_numpy(), upper_limit[c].to_numpy()
        valid = np.isfinite(op) & np.isfinite(lo) & np.isfinite(ul)
        opened = op >= ul - tolerance
        broke = lo < ul - tolerance
        out[:, i] = np.where(valid, (opened & broke).astype(float), np.nan)
    return _make(open_px, cols, out)


def ashare_limit_one_price(open_px, high, low, close, upper_limit, lower_limit, side="up", tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(open_px, high, low, close, upper_limit, lower_limit)
    rows = open_px.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    direction = str(side).lower()
    for i, c in enumerate(cols):
        op, hi, lo, cl = (f[c].to_numpy() for f in (open_px, high, low, close))
        ul, ll = upper_limit[c].to_numpy(), lower_limit[c].to_numpy()
        limit = ul if direction == "up" else ll
        valid = np.isfinite(op) & np.isfinite(hi) & np.isfinite(lo) & np.isfinite(cl) & np.isfinite(limit)
        at = (
            (op >= limit - tolerance) & (op <= limit + tolerance)
            & (hi >= limit - tolerance) & (hi <= limit + tolerance)
            & (lo >= limit - tolerance) & (lo <= limit + tolerance)
            & (cl >= limit - tolerance) & (cl <= limit + tolerance)
        )
        if direction == "up":
            out[:, i] = np.where(valid, (at & (cl >= ul - tolerance)).astype(float), np.nan)
        elif direction == "down":
            out[:, i] = np.where(valid, (at & (cl <= ll + tolerance)).astype(float), np.nan)
        else:
            raise ValueError("side must be 'up' or 'down'")
    return _make(open_px, cols, out)


def ashare_limit_failed(high, close, upper_limit, tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(high, close, upper_limit)
    rows = high.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        hi, cl, ul = high[c].to_numpy(), close[c].to_numpy(), upper_limit[c].to_numpy()
        valid = np.isfinite(hi) & np.isfinite(cl) & np.isfinite(ul)
        touched = hi >= ul - tolerance
        held = cl >= ul - tolerance
        out[:, i] = np.where(valid, (touched & ~held).astype(float), np.nan)
    return _make(high, cols, out)


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
        finite = np.isfinite(values)
        # Unknown (NaN) stays NaN; 1 when finite and > threshold, 0 when finite
        # and <= threshold (audit §4.6) — mirrors the pandas backend.
        out[:, i] = np.where(
            finite,
            np.where(values > thr, 1.0, 0.0),
            np.nan,
        )
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


def event_decay_asof(event, half_life=20.0, missing_policy="carry"):
    hl = _pf(half_life, "half_life", 1.0)
    if missing_policy not in {"carry", "break"}:
        raise ValueError("missing_policy must be 'carry' or 'break'")
    weight = 0.5 ** (1.0 / hl)
    cols = _cols(event)
    rows = event.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        ev = event[c].to_numpy()
        acc = 0.0
        seen = False
        for t in range(rows):
            finite = bool(np.isfinite(ev[t]))
            if finite:
                seen = True
                acc = acc * weight + float(ev[t])
            elif not seen:
                out[t, i] = np.nan
                continue
            elif missing_policy == "break":
                out[t, i] = np.nan
                seen = False
                acc = 0.0
                continue
            else:  # "carry"（默认）
                acc = acc * weight
            out[t, i] = acc
    return _make(event, cols, out)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("ashare_limit_distance", ("close", "upper_limit"), ashare_limit_distance, "Distance to upper limit."),
    ("ashare_limit_up_touch", ("high", "upper_limit", "tick_tolerance"), ashare_limit_up_touch, "High touched upper limit within tolerance."),
    ("ashare_limit_down_touch", ("low", "lower_limit", "tick_tolerance"), ashare_limit_down_touch, "Low touched lower limit within tolerance."),
    ("ashare_open_at_upper_limit", ("open", "upper_limit", "tick_tolerance"), ashare_open_at_upper_limit, "Opened at the upper limit."),
    ("ashare_limit_open_failed", ("open", "low", "upper_limit", "tick_tolerance"), ashare_limit_open_failed, "Opened at the limit then broke intraday."),
    ("ashare_limit_one_price", ("open", "high", "low", "close", "upper_limit", "lower_limit", "side", "tick_tolerance"), ashare_limit_one_price, "One-price limit lock (OHLC at the same limit)."),
    ("ashare_limit_failed", ("high", "close", "upper_limit", "tick_tolerance"), ashare_limit_failed, "Touched the limit but failed to hold at close."),
    ("benchmark_excess_return", ("ret", "benchmark_ret"), benchmark_excess_return, "Return minus benchmark return."),
    ("benchmark_relative_price", ("numerator", "denominator"), benchmark_relative_price, "Numerator over denominator."),
    ("fin_applicability_mask", ("value", "threshold"), fin_applicability_mask, "1 where value exceeds threshold."),
    ("cash_flow_lifecycle_stage", ("operating", "investing", "financing"), cash_flow_lifecycle_stage, "Cash-flow lifecycle stage 1-5."),
    ("event_decay_asof", ("event", "half_life", "missing_policy"), event_decay_asof, "Exponential decay of past events (preserving sign/magnitude)."),
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

# Classify limit-state helpers on the extended surface so the static operator
# surface covers the final registry exactly.
import cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.extend_extended_only({
        "ashare_limit_up_touch",
        "ashare_limit_down_touch",
        "ashare_open_at_upper_limit",
        "ashare_limit_open_failed",
        "ashare_limit_one_price",
        "ashare_limit_failed",
    })
