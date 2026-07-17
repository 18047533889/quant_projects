# -*- coding: utf-8 -*-
"""Shared rolling-OLS computation for all user-facing regression outputs."""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock
import weakref

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import PandasFunctionOperator, aligned_pd, frame_pd, window_params
from cleaned_operators.registry import OperatorRegistry

_APPLIED = False
_CACHE_LIMIT = 16
_CACHE: "OrderedDict[tuple, tuple[weakref.ReferenceType, weakref.ReferenceType, dict[str, pd.DataFrame]]]" = OrderedDict()
_LOCK = RLock()
_STATS_COMPUTATIONS = 0


def _fit(y: np.ndarray, x: np.ndarray, add_intercept: bool):
    mask = np.isfinite(y) & np.isfinite(x)
    yv, xv = y[mask], x[mask]
    parameters = 2 if add_intercept else 1
    if yv.size <= parameters or np.var(xv) <= 0:
        return None
    design = xv[:, None]
    if add_intercept:
        design = np.column_stack((np.ones(xv.size), xv))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    fitted = design @ beta
    residuals = yv - fitted
    sse = float(residuals @ residuals)
    centered = yv - yv.mean()
    sst = float(centered @ centered)
    r2 = np.nan if sst <= 0 else 1.0 - sse / sst
    dof = yv.size - design.shape[1]
    tstat = np.nan
    if dof > 0:
        sigma2 = sse / dof
        slope_variance = sigma2 * np.linalg.pinv(design.T @ design)[-1, -1]
        if np.isfinite(slope_variance) and slope_variance > 0:
            tstat = float(beta[-1] / np.sqrt(slope_variance))
    intercept = float(beta[0]) if add_intercept else 0.0
    slope = float(beta[-1])
    current_residual = (
        float(y[-1] - intercept - slope * x[-1])
        if np.isfinite(y[-1]) and np.isfinite(x[-1]) else np.nan
    )
    return slope, intercept, current_residual, float(r2), float(tstat)


def _fingerprint(frame: pd.DataFrame) -> tuple:
    return (
        id(frame), frame.shape, id(frame.index), id(frame.columns),
        frame.index[0] if len(frame.index) else None,
        frame.index[-1] if len(frame.index) else None,
        tuple(frame.columns),
    )


def _compute_all(y, x, window, min_periods=None, add_intercept=True):
    global _STATS_COMPUTATIONS
    if not isinstance(y, pd.DataFrame) or not isinstance(x, pd.DataFrame):
        raise TypeError("rolling OLS fusion requires pandas DataFrame inputs")
    source_y, source_x = y, x
    w, mp = window_params(window, min_periods, default_mp=3)
    key = (_fingerprint(source_y), _fingerprint(source_x), w, mp, bool(add_intercept))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0]() is source_y and cached[1]() is source_x:
            _CACHE.move_to_end(key)
            return cached[2]

    y, x = aligned_pd(source_y, source_x)
    yv, xv = y.to_numpy(dtype=float), x.to_numpy(dtype=float)
    raw = np.full((5, *y.shape), np.nan, dtype=float)
    for column in range(y.shape[1]):
        for row in range(y.shape[0]):
            start = max(0, row - w + 1)
            yy = yv[start: row + 1, column]
            xx = xv[start: row + 1, column]
            if int(np.sum(np.isfinite(yy) & np.isfinite(xx))) < mp:
                continue
            fitted = _fit(yy, xx, bool(add_intercept))
            if fitted is not None:
                raw[:, row, column] = fitted

    outputs = {
        name: frame_pd(y, raw[index])
        for index, name in enumerate(("slope", "intercept", "resid", "r2", "tstat"))
    }
    with _LOCK:
        _STATS_COMPUTATIONS += 1
        _CACHE[key] = (weakref.ref(source_y), weakref.ref(source_x), outputs)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return outputs


def _output(name):
    def calculate(y, x, window, min_periods=None, add_intercept=True, **_):
        return _compute_all(y, x, window, min_periods, add_intercept)[name]
    return calculate


def _slope_compat(y, x, window, *legacy_args, min_periods=None, add_intercept=True, lag=None, retval=None, **_):
    if len(legacy_args) > 2:
        raise TypeError("ts_regression accepts at most legacy lag and retval arguments")
    if legacy_args:
        lag = int(legacy_args[0])
    if len(legacy_args) == 2:
        retval = str(legacy_args[1])
    lag_i = 0 if lag is None else int(lag)
    if lag_i < 0:
        return pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
    if lag_i:
        x = x.shift(lag_i)
    requested = str(retval or "slope").lower()
    output = {
        "slope": "slope", "beta": "slope", "intercept": "intercept",
        "residual": "resid", "resid": "resid", "r_squared": "r2",
        "r2": "r2", "tstat": "tstat", "t_stat": "tstat",
    }.get(requested)
    if output is None:
        raise ValueError(f"unsupported regression retval: {retval!r}")
    return _compute_all(y, x, window, min_periods, add_intercept)[output]


def clear_rolling_ols_cache() -> None:
    global _STATS_COMPUTATIONS
    with _LOCK:
        _CACHE.clear()
        _STATS_COMPUTATIONS = 0


def rolling_ols_cache_info() -> dict[str, int]:
    with _LOCK:
        return {"entries": len(_CACHE), "stats_computations": _STATS_COMPUTATIONS}


def install_rolling_ols_fusion() -> None:
    global _APPLIED
    if _APPLIED:
        return
    for canonical, output in (
        ("ts_regression_slope", "slope"),
        ("ts_regression_intercept", "intercept"),
        ("ts_regression_resid", "resid"),
        ("ts_regression_r2", "r2"),
        ("ts_regression_tstat", "tstat"),
    ):
        function = _slope_compat if canonical == "ts_regression_slope" else _output(output)
        params = (
            ["y", "x", "window", "lag", "retval", "min_periods", "add_intercept"]
            if canonical == "ts_regression_slope"
            else ["y", "x", "window", "min_periods", "add_intercept"]
        )
        OperatorRegistry.register(
            PandasFunctionOperator(
                canonical, "time_series_regression", params,
                "rolling OLS output projected from a shared five-stat computation",
                function,
            ),
            canonical=canonical,
            backend="pandas_numpy",
            source="rolling_ols_fused_cache",
            status="production",
            backend_explicit=True,
            replace="pandas_numpy" in OperatorRegistry.backends_for(canonical),
            replacement_reason="share one rolling OLS fit while retaining historical ts_regression signature",
            semantic_version="3.0",
        )
    _APPLIED = True


__all__ = [
    "install_rolling_ols_fusion",
    "clear_rolling_ols_cache",
    "rolling_ols_cache_info",
]
