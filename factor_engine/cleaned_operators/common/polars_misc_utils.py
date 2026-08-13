# -*- coding: utf-8 -*-
"""Native Polars backends for a small set of simple panel operators.

Covers elementwise utilities (sqrt_abs, book_to_price, earnings_yield, share
ratios), the recursive SMA (ts_sma_cn), the lag ratio (ts_ratio) and rolling
sign-ratio series (ts_positive/negative/zero_ratio).  Each is evaluated
per-column with Polars expressions or a small NumPy kernel; results are wrapped
into ``pl.DataFrame`` without constructing pandas DataFrames.
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


def _result(base: pl.DataFrame, values: dict[str, pl.Series]) -> pl.DataFrame:
    return base.with_columns([series.alias(name) for name, series in values.items()])


def _one(frame: pl.DataFrame, column: str, expr: pl.Expr) -> pl.Series:
    return frame.select(expr.alias(column)).to_series()


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den != 0).then(num / den).otherwise(None)


def sqrt_abs(x):
    values = {}
    for c in _cols(x):
        values[c] = _one(x, c, pl.col(c).abs().sqrt())
    return _result(x, values)


def book_to_price(pb):
    values = {}
    for c in _cols(pb):
        values[c] = _one(pb, c, pl.when(pl.col(c) > 0).then(1.0 / pl.col(c)).otherwise(None))
    return _result(pb, values)


def earnings_yield(pe):
    values = {}
    for c in _cols(pe):
        values[c] = _one(pe, c, pl.when(pl.col(c) > 0).then(1.0 / pl.col(c)).otherwise(None))
    return _result(pe, values)


def float_share_ratio(numerator, denominator):
    values = {}
    for c in _cols(numerator, denominator):
        frame = pl.DataFrame({"num": numerator[c], "den": denominator[c]})
        values[c] = _one(frame, c, _safe_div(pl.col("num"), pl.col("den")))
    return _result(numerator, values)


def ts_ratio(x):
    values = {}
    for c in _cols(x):
        prev = pl.col(c).shift(1)
        values[c] = _one(x, c, _safe_div(pl.col(c), prev))
    return _result(x, values)


def ts_sma_cn(x, n, m):
    n_i = _pi(n, "n")
    m_i = _pi(m, "m")
    if m_i > n_i:
        raise ValueError("sma m must satisfy 1 <= m <= n")
    alpha = np.where(float(n_i) != 0, float(m_i) / float(n_i), np.nan)
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        arr = x[c].to_numpy()
        state = np.nan
        for t in range(rows):
            value = arr[t]
            if not np.isfinite(value):
                continue
            state = float(value) if not np.isfinite(state) else alpha * float(value) + (1.0 - alpha) * state
            out[t, i] = state
    return _result(x, {c: pl.Series(name=c, values=out[:, i]) for i, c in enumerate(cols)})


def _rolling_sign_ratio_1d(x: np.ndarray, w: int, sign: int, threshold: float, min_periods: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        seg = x[lo : t + 1]
        finite = seg[np.isfinite(seg)]
        if finite.size < min_periods:
            continue
        if sign > 0:
            hits = float(np.sum(finite > threshold))
        elif sign < 0:
            hits = float(np.sum(finite < threshold))
        else:
            hits = float(np.sum(np.abs(finite) <= threshold))
        out[t] = np.where(float(finite.size) != 0, hits / float(finite.size), np.nan)
    return out


def _sign_ratio_op(x, window, threshold, sign, name):
    w = _pi(window, "window")
    thr = _pf(threshold, "threshold" if sign else "tolerance", None)
    mp = max(1, int(_pi(window if False else 1, "min_periods")) if False else 1)
    # min_periods default 1 per reference
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_sign_ratio_1d(x[c].to_numpy(), w, sign, thr, 1)
    return _result(x, {c: pl.Series(name=c, values=out[:, i]) for i, c in enumerate(cols)})


def ts_positive_ratio(x, window, threshold=0.0, min_periods=1):
    w = _pi(window, "window")
    thr = _pf(threshold, "threshold", None)
    mp = max(1, int(min_periods))
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_sign_ratio_1d(x[c].to_numpy(), w, 1, thr, mp)
    return _result(x, {c: pl.Series(name=c, values=out[:, i]) for i, c in enumerate(cols)})


def ts_negative_ratio(x, window, threshold=0.0, min_periods=1):
    w = _pi(window, "window")
    thr = _pf(threshold, "threshold", None)
    mp = max(1, int(min_periods))
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_sign_ratio_1d(x[c].to_numpy(), w, -1, thr, mp)
    return _result(x, {c: pl.Series(name=c, values=out[:, i]) for i, c in enumerate(cols)})


def ts_zero_ratio(x, window, tolerance=0.0, min_periods=1):
    w = _pi(window, "window")
    tol = _pf(tolerance, "tolerance", None)
    mp = max(1, int(min_periods))
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_sign_ratio_1d(x[c].to_numpy(), w, 0, tol, mp)
    return _result(x, {c: pl.Series(name=c, values=out[:, i]) for i, c in enumerate(cols)})


_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("sqrt_abs", ("x",), sqrt_abs, "sqrt(abs(x)), inf -> nan."),
    ("book_to_price", ("pb",), book_to_price, "1/pb for positive pb."),
    ("earnings_yield", ("pe",), earnings_yield, "1/pe for positive pe."),
    ("float_share_ratio", ("numerator", "denominator"), float_share_ratio, "numerator / denominator with zero handled as missing."),
    ("free_float_share_ratio", ("numerator", "denominator"), float_share_ratio, "numerator / denominator with zero handled as missing."),
    ("ts_ratio", ("x",), ts_ratio, "x / previous observation, zero denominator -> missing."),
    ("ts_sma_cn", ("x", "n", "m"), ts_sma_cn, "Recursive SMA with alpha = m/n."),
    ("ts_positive_ratio", ("x", "window", "threshold", "min_periods"), ts_positive_ratio, "Rolling share of values above threshold."),
    ("ts_negative_ratio", ("x", "window", "threshold", "min_periods"), ts_negative_ratio, "Rolling share of values below threshold."),
    ("ts_zero_ratio", ("x", "window", "tolerance", "min_periods"), ts_zero_ratio, "Rolling share of near-zero values."),
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
        f"PolarsMiscUtils_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="math",
        business_category="elementwise_math",
        canonical=name,
        source="polars_misc_utils",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
