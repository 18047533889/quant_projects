# -*- coding: utf-8 -*-
"""COS-aware PIT primitives and final semantic overrides."""
from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.overhaul.base import (
    EPS, Spec, aligned_pd, frame_pd, pl, pl_base_with, pl_cols,
    positive_int, register_specs,
)
from factor_engine.cleaned_operators.fiscal_strict import (
    period_ordinal, pd_ttm_from_quarterly, pd_yoy_by_period,
)


def pd_true_range(high, low, close, **_):
    high, low, close = aligned_pd(high, low, close)
    h, l, c = (f.to_numpy(dtype=float) for f in (high, low, close))
    prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    out = h - l
    has_prev = np.isfinite(prev)
    out[has_prev] = np.maximum.reduce([out[has_prev], np.abs(h[has_prev] - prev[has_prev]), np.abs(l[has_prev] - prev[has_prev])])
    out[~(np.isfinite(h) & np.isfinite(l))] = np.nan
    return frame_pd(close, out)


def pl_true_range(high, low, close, **_):
    cols = [c for c in pl_cols(close) if c in high.columns and c in low.columns]
    return close.with_columns([
        # PARITY-B: pandas reference (pd_true_range) keeps the row finite whenever
        # high & low are finite, using the previous close ONLY where it is finite
        # (the max is over the terms that exist; the seed high-low stays).  A NaN
        # previous close therefore does NOT blank the row (the three-term max
        # skips it).  The old ``max_horizontal`` propagated the NaN previous close
        # and blanked rows after every gap — matching the oracle now.
        pl.when((high[c].cast(pl.Float64, strict=False).is_null() | high[c].cast(pl.Float64, strict=False).is_nan()) | (low[c].cast(pl.Float64, strict=False).is_null() | low[c].cast(pl.Float64, strict=False).is_nan())).then(None)
        .otherwise(
            pl.max_horizontal(
                high[c].cast(pl.Float64, strict=False) - low[c].cast(pl.Float64, strict=False),
                (high[c].cast(pl.Float64, strict=False) - close[c].cast(pl.Float64, strict=False).shift(1)).abs().fill_null(float("-inf")).fill_nan(float("-inf")),
                (low[c].cast(pl.Float64, strict=False) - close[c].cast(pl.Float64, strict=False).shift(1)).abs().fill_null(float("-inf")).fill_nan(float("-inf")),
            )
        ).alias(c)
        for c in cols
    ])


def _datetime_days(values):
    parsed = pd.to_datetime(values.reshape(-1), errors="coerce", utc=True)
    raw = np.asarray(parsed.view("int64"), dtype=float).reshape(values.shape)
    raw[raw == float(np.iinfo(np.int64).min)] = np.nan
    return raw / 86_400_000_000_000.0


def pd_staleness(available_at, decision_time, **_):
    available_at, decision_time = aligned_pd(available_at, decision_time)
    out = _datetime_days(decision_time.to_numpy(dtype=object)) - _datetime_days(available_at.to_numpy(dtype=object))
    out[(~np.isfinite(out)) | (out < 0)] = np.nan
    return frame_pd(available_at, out)


def pl_staleness(available_at, decision_time, **_):
    cols = [c for c in pl_cols(available_at) if c in decision_time.columns]
    return available_at.with_columns([(((decision_time[c].cast(pl.Datetime, strict=False) - available_at[c].cast(pl.Datetime, strict=False)).dt.total_seconds() / 86400.0).cast(pl.Float64).alias(c)) for c in cols])


def _revision_array(values, periods, revisions, mode):
    out = np.full(values.shape, np.nan)
    for col in range(values.shape[1]):
        last: dict[int, tuple[Any, float]] = {}
        for row in range(values.shape[0]):
            ordinal = period_ordinal(periods[row, col])
            value, revision = values[row, col], revisions[row, col]
            if ordinal is None or not np.isfinite(value) or pd.isna(revision):
                continue
            previous = last.get(ordinal)
            if previous is not None and revision != previous[0]:
                if mode == "absolute":
                    out[row, col] = value - previous[1]
                elif abs(previous[1]) > EPS:
                    out[row, col] = value / previous[1] - 1.0
            last[ordinal] = (revision, float(value))
    return out


def pd_revision_delta(x, period_id, revision_id, mode="absolute", **_):
    x, period_id, revision_id = aligned_pd(x, period_id, revision_id)
    mode = str(mode).lower()
    if mode not in {"absolute", "ratio"}:
        raise ValueError("mode must be absolute or ratio")
    return frame_pd(x, _revision_array(x.to_numpy(dtype=float), period_id.to_numpy(dtype=object), revision_id.to_numpy(dtype=object), mode))


def pl_revision_delta(x, period_id, revision_id, mode="absolute", **_):
    mode = str(mode).lower()
    if mode not in {"absolute", "ratio"}:
        raise ValueError("mode must be absolute or ratio")
    cols = [c for c in pl_cols(x) if c in period_id.columns and c in revision_id.columns]
    values = np.column_stack([x[c].cast(pl.Float64, strict=False).to_numpy() for c in cols])
    periods = np.column_stack([np.asarray(period_id[c].to_list(), dtype=object) for c in cols])
    revisions = np.column_stack([np.asarray(revision_id[c].to_list(), dtype=object) for c in cols])
    out = _revision_array(values, periods, revisions, mode)
    return pl_base_with(x, {c: pl.Series(c, out[:, i]) for i, c in enumerate(cols)})


def _stability_column(values, periods, count, method, consecutive):
    out = np.full(values.shape, np.nan)
    known: dict[int, float] = {}
    for row, (value, raw_period) in enumerate(zip(values, periods)):
        ordinal = period_ordinal(raw_period)
        if ordinal is None:
            continue
        if np.isfinite(value):
            known[ordinal] = float(value)
        sample = [known[key] for key in [ordinal - lag for lag in range(count)] if key in known]
        if (consecutive and len(sample) != count) or len(sample) < 2:
            continue
        array = np.asarray(sample)
        if method == "std":
            out[row] = np.std(array, ddof=1)
        elif method == "mad":
            out[row] = np.median(np.abs(array - np.median(array)))
        else:
            mean = np.mean(array)
            if abs(mean) > EPS:
                out[row] = np.std(array, ddof=1) / abs(mean)
    return out


def pd_period_stability(x, period_id, periods=8, method="mad", require_consecutive=True, **_):
    x, period_id = aligned_pd(x, period_id)
    count, method = positive_int(periods, "periods"), str(method).lower()
    if method not in {"std", "mad", "cv"}:
        raise ValueError("method must be std, mad, or cv")
    out = np.full(x.shape, np.nan)
    for col in range(x.shape[1]):
        out[:, col] = _stability_column(x.iloc[:, col].to_numpy(dtype=float), period_id.iloc[:, col].to_numpy(dtype=object), count, method, bool(require_consecutive))
    return frame_pd(x, out)


def pl_period_stability(x, period_id, periods=8, method="mad", require_consecutive=True, **_):
    count, method = positive_int(periods, "periods"), str(method).lower()
    if method not in {"std", "mad", "cv"}:
        raise ValueError("method must be std, mad, or cv")
    cols = [c for c in pl_cols(x) if c in period_id.columns]
    return pl_base_with(x, {c: pl.Series(c, _stability_column(x[c].cast(pl.Float64, strict=False).to_numpy(), np.asarray(period_id[c].to_list(), dtype=object), count, method, bool(require_consecutive))) for c in cols})


def pd_safe_div(x, y, epsilon=EPS, **_):
    if not isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        x = pd.DataFrame(x, index=y.index, columns=y.columns, dtype=float)
    elif not isinstance(y, pd.DataFrame) and isinstance(x, pd.DataFrame):
        y = pd.DataFrame(y, index=x.index, columns=x.columns, dtype=float)
    x, y = aligned_pd(x, y)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    valid = np.isfinite(xv) & np.isfinite(yv) & (np.abs(yv) > epsilon)
    out = np.full(x.shape, np.nan)
    out[valid] = xv[valid] / yv[valid]
    return frame_pd(x, out)


def pl_safe_div(x, y, epsilon=EPS, **_):
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    cols = [c for c in pl_cols(x) if c in y.columns]
    return x.with_columns([pl.when(x[c].cast(pl.Float64, strict=False).is_finite() & y[c].cast(pl.Float64, strict=False).is_finite() & (y[c].cast(pl.Float64, strict=False).abs() > epsilon)).then(x[c].cast(pl.Float64, strict=False) / y[c].cast(pl.Float64, strict=False)).otherwise(None).alias(c) for c in cols])


def register() -> None:
    register_specs({
        "true_range": Spec("price_volume", ["high", "low", "close"], "true range with explicit first-row high-low fallback", pd_true_range, pl_true_range),
        "fundamental_staleness": Spec("fundamental_period", ["available_at", "decision_time"], "calendar-day age of visible fundamental data", pd_staleness, pl_staleness),
        "revision_delta": Spec("fundamental_period", ["x", "period_id", "revision_id", "mode"], "change between visible revisions of the same fiscal period", pd_revision_delta, pl_revision_delta),
        "period_stability": Spec("fundamental_period", ["x", "period_id", "periods", "method", "require_consecutive"], "stability over exact fiscal periods", pd_period_stability, pl_period_stability),
        "safe_div_null": Spec("elementwise", ["x", "y", "epsilon"], "finite safe division returning null for near-zero denominator", pd_safe_div, pl_safe_div),
        "ttm_from_quarterly": Spec("fundamental_period", ["x", "period_id", "periods", "require_consecutive", "revision_policy"], "strict consecutive-period TTM", pd_ttm_from_quarterly),
        "yoy_by_period": Spec("fundamental_period", ["x", "period_id", "periods", "denominator", "require_consecutive", "revision_policy"], "strict fiscal-period growth", pd_yoy_by_period),
    })


register()
