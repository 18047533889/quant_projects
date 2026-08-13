# -*- coding: utf-8 -*-
"""Native Polars backends for limit-state helpers, cross-sectional normalizers
and rolling regression statics.

Per-column NumPy kernels (rolling regression) and panel-level kernels
(cross-sectional unitize / winsorize) over polars column arrays; results are
wrapped into ``pl.DataFrame`` without constructing pandas DataFrames.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


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


def _panel(*frames: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return np.stack([f[c].to_numpy() for f in frames for c in cols], axis=1) if len(frames) == 1 else None


# ---------------------------------------------------------------------------
# limit-state helpers (elementwise)
# ---------------------------------------------------------------------------


def limit_up_close(close, upper_limit, tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(close, upper_limit)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cl, ul = close[c].to_numpy(), upper_limit[c].to_numpy()
        valid = np.isfinite(cl) & np.isfinite(ul)
        out[:, i] = np.where(valid, (cl >= ul - tolerance).astype(float), np.nan)
    return _make(close, cols, out)


def limit_down_close(close, lower_limit, tick_tolerance=0.005):
    tolerance = _pf(tick_tolerance, "tick_tolerance", 0)
    cols = _cols(close, lower_limit)
    rows = close.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cl, ll = close[c].to_numpy(), lower_limit[c].to_numpy()
        valid = np.isfinite(cl) & np.isfinite(ll)
        out[:, i] = np.where(valid, (cl <= ll + tolerance).astype(float), np.nan)
    return _make(close, cols, out)


def tradable_state(listed, suspended, limit_up, limit_down):
    cols = _cols(listed, suspended, limit_up, limit_down)
    rows = listed.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        li, su, lu, ld = (f[c].to_numpy() for f in (listed, suspended, limit_up, limit_down))
        valid = np.isfinite(li) & np.isfinite(su) & np.isfinite(lu) & np.isfinite(ld)
        flag = (li > 0) & (su <= 0) & (lu <= 0) & (ld <= 0)
        out[:, i] = np.where(valid, flag.astype(float), np.nan)
    return _make(listed, cols, out)


def holder_concentration(top_holder_shares, total_shares):
    cols = _cols(top_holder_shares, total_shares)
    rows = top_holder_shares.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        num, den = top_holder_shares[c].to_numpy(), total_shares[c].to_numpy()
        out[:, i] = np.divide(num, den, out=np.full(rows, np.nan), where=den != 0)
    return _make(top_holder_shares, cols, out)


# ---------------------------------------------------------------------------
# cross-sectional normalizers (axis=1)
# ---------------------------------------------------------------------------


def unitize(x):
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    arr = np.stack([x[c].to_numpy() for c in cols], axis=1)
    abs_arr = np.abs(arr)
    finite = np.isfinite(abs_arr)
    row_max = np.full(rows, np.nan, dtype=float)
    for t in range(rows):
        ok = finite[t]
        if ok.any():
            row_max[t] = float(np.max(abs_arr[t][ok]))
    denom = np.where(row_max == 0, np.nan, row_max)
    for t in range(rows):
        if np.isfinite(denom[t]):
            out[t] = np.clip(arr[t] / denom[t], -1.0, 1.0)
        else:
            out[t] = np.full(len(cols), np.nan)
    return _make(x, cols, out)


def winsorize_mean(x, trim_pct=0.1):
    trim = max(0.0, min(float(trim_pct), 0.49))
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    arr = np.stack([x[c].to_numpy() for c in cols], axis=1)
    for t in range(rows):
        valid = arr[t][np.isfinite(arr[t])]
        if valid.size == 0:
            continue
        lower = float(np.quantile(valid, trim))
        upper = float(np.quantile(valid, 1.0 - trim))
        clipped = np.clip(arr[t], lower, upper)
        out[t] = np.full(len(cols), float(np.mean(clipped[np.isfinite(arr[t])])))
    return _make(x, cols, out)


# ---------------------------------------------------------------------------
# rolling regression statics
# ---------------------------------------------------------------------------


def _fit_1d_np(y: np.ndarray, x: np.ndarray, add_intercept: bool):
    mask = np.isfinite(y) & np.isfinite(x)
    yv, xv = y[mask], x[mask]
    p = 2 if add_intercept else 1
    if yv.size <= p or np.var(xv) <= 0:
        return None
    design = xv[:, None]
    if add_intercept:
        design = np.column_stack((np.ones(xv.size), xv))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    fitted = design @ beta
    resid = yv - fitted
    sse = float(resid @ resid)
    centered = yv - yv.mean()
    sst = float(centered @ centered)
    r2 = np.nan if sst <= 0 else 1.0 - sse / sst
    intercept = float(beta[0]) if add_intercept else 0.0
    slope = float(beta[-1])
    current_resid = (
        float(y[-1] - intercept - slope * x[-1])
        if np.isfinite(y[-1]) and np.isfinite(x[-1])
        else np.nan
    )
    return slope, intercept, current_resid, float(r2)


def _reg_stat(y, x, window, min_periods, add_intercept, idx):
    cols = _cols(y, x)
    rows = y.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    w = int(window)
    mp = 3 if min_periods is None else max(1, int(min_periods))
    ai = bool(add_intercept)
    for i, c in enumerate(cols):
        yv, xv = y[c].to_numpy(), x[c].to_numpy()
        for t in range(rows):
            start = max(0, t - w + 1)
            yy, xx = yv[start : t + 1], xv[start : t + 1]
            if int(np.sum(np.isfinite(yy) & np.isfinite(xx)) < mp:
                continue
            fit = _fit_1d_np(yy, xx, ai)
            if fit is not None:
                out[t, i] = fit[idx]
    return _make(y, cols, out)


def ts_regression_intercept(y, x, window, min_periods=None, add_intercept=True):
    return _reg_stat(y, x, window, min_periods, add_intercept, 1)


def ts_regression_r2(y, x, window, min_periods=None, add_intercept=True):
    return _reg_stat(y, x, window, min_periods, add_intercept, 3)


def ts_regression_resid(y, x, window, min_periods=None, add_intercept=True):
    return _reg_stat(y, x, window, min_periods, add_intercept, 2)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("limit_up_close", ("close", "upper_limit", "tick_tolerance"), limit_up_close, "Close at upper limit within tick tolerance."),
    ("limit_down_close", ("close", "lower_limit", "tick_tolerance"), limit_down_close, "Close at lower limit within tick tolerance."),
    ("tradable_state", ("listed", "suspended", "limit_up", "limit_down"), tradable_state, "Tradable-state indicator."),
    ("holder_concentration", ("top_holder_shares", "total_shares"), holder_concentration, "Top-holder concentration ratio."),
    ("unitize", ("x",), unitize, "Cross-sectional normalize to [-1, 1]."),
    ("winsorize_mean", ("x", "trim_pct"), winsorize_mean, "Cross-sectional trimmed mean."),
    ("ts_regression_intercept", ("y", "x", "window", "min_periods", "add_intercept"), ts_regression_intercept, "Rolling regression intercept."),
    ("ts_regression_r2", ("y", "x", "window", "min_periods", "add_intercept"), ts_regression_r2, "Rolling regression R-squared."),
    ("ts_regression_resid", ("y", "x", "window", "min_periods", "add_intercept"), ts_regression_resid, "Rolling regression residual at window end."),
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
        f"PolarsCsMisc_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="math",
        business_category="elementwise_math",
        canonical=name,
        source="polars_cs_misc",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
