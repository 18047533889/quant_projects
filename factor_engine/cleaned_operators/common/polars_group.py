# -*- coding: utf-8 -*-
"""Native Polars backends for cross-sectional group ex-self and robust
regression-residual operators.

These operate per row (a cross-section across panel columns) using the same
NumPy kernels as the pandas references; results are wrapped into
``pl.DataFrame`` without constructing pandas DataFrames.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})

# R11 #137: cross-section minimum breadth (mirrors group_ext._MIN_BREADTH /
# _BREADTH_PARAM_RATIO so the pandas and polars twins share the same gate).
_MIN_BREADTH = 10
_BREADTH_PARAM_RATIO = 5.0


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


def _row_arrays(*frames: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return np.stack([f[c].to_numpy() for f in frames for c in cols], axis=1) if len(frames) == 1 else None


def group_ex_self_mean(x, group):
    cols = _cols(x, group)
    rows = x.height
    xv = np.stack([x[c].to_numpy() for c in cols], axis=1)
    gv = np.stack([group[c].to_numpy() for c in cols], axis=1)
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for t in range(rows):
        finite = np.isfinite(xv[t])
        labels = np.unique(gv[t])
        for label in labels:
            idx = (gv[t] == label) & finite
            count = int(np.sum(idx))
            if count <= 1:
                continue
            total = float(np.sum(xv[t][idx]))
            for j in np.flatnonzero(gv[t] == label):
                if not finite[j]:
                    continue
                out[t, j] = (total - xv[t][j]) / (count - 1.0)
    return _make(x, cols, out)


def group_ex_self_weighted_mean(x, weight, group):
    cols = _cols(x, weight, group)
    rows = x.height
    xv = np.stack([x[c].to_numpy() for c in cols], axis=1)
    wv = np.stack([weight[c].to_numpy() for c in cols], axis=1)
    gv = np.stack([group[c].to_numpy() for c in cols], axis=1)
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for t in range(rows):
        g_row = gv[t]
        for label in np.unique(g_row):
            idx = g_row == label
            valid_w = np.isfinite(wv[t]) & idx
            total_w = float(np.sum(wv[t][valid_w]))
            if not np.isfinite(total_w) or total_w <= 0.0:
                continue
            weighted = float(np.sum(wv[t][valid_w] * xv[t][valid_w]))
            for j in np.flatnonzero(idx):
                own_w = wv[t][j]
                if not np.isfinite(own_w) or not np.isfinite(xv[t][j]):
                    continue
                denom = total_w - own_w
                if denom <= 0.0:
                    continue
                out[t, j] = (weighted - own_w * xv[t][j]) / denom
    return _make(x, cols, out)


def _demean_composite_row(values: np.ndarray, group_labels: np.ndarray, subgroup_labels: np.ndarray) -> np.ndarray:
    """Demean by the composite ``(group, subgroup)`` key (R11 #134).

    Mirrors ``group_ext.HierarchicalGroupNeutralize._demean_composite_row``: a
    bare subgroup label that repeats across parent groups must NOT be merged;
    the composite key keeps subgroup means within their own parent group.
    """
    out = values.copy()
    n = len(values)
    keys = [None] * n
    for i in range(n):
        keys[i] = (group_labels[i], subgroup_labels[i])
    seen: list[tuple] = []
    for key in keys:
        if key not in seen:
            seen.append(key)
    for key in seen:
        idx = np.flatnonzero(np.fromiter((k == key for k in keys), dtype=bool, count=n))
        group_values = values[idx]
        finite = group_values[np.isfinite(group_values)]
        if finite.size == 0:
            continue
        mean = float(np.mean(finite))
        out[idx] = values[idx] - mean
    return out


def hierarchical_group_neutralize(x, group, subgroup):
    cols = _cols(x, group, subgroup)
    rows = x.height
    xv = np.stack([x[c].to_numpy() for c in cols], axis=1)
    gv = np.stack([group[c].to_numpy() for c in cols], axis=1)
    sv = np.stack([subgroup[c].to_numpy() for c in cols], axis=1)
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for t in range(rows):
        out[t] = _demean_composite_row(xv[t].copy(), gv[t], sv[t])
    return _make(x, cols, out)


def cs_robust_resid(y, x, trim_ratio=0.1, add_intercept=True):
    """Trimmed cross-sectional OLS residual.

    Note (R5 P1-37c): this trims the most extreme ``x`` observations and then
    fits an ordinary least-squares line.  It is *trimmed OLS*, not a true robust
    regression (no Huber / LAD weighting).  The name and signature are kept for
    compatibility with the pandas twin (``group_ext.CsRobustResid``).
    R11 #137: a cross-section minimum-breadth gate matches the pandas twin —
    3 stocks must not drive the regression.
    """
    trim = _pf(trim_ratio, "trim_ratio")
    if not (0.0 <= trim < 0.5):
        raise ValueError("cs_robust_resid requires 0 <= trim_ratio < 0.5")
    cols = _cols(y, x)
    rows = y.height
    yv = np.stack([y[c].to_numpy() for c in cols], axis=1)
    xv = np.stack([x[c].to_numpy() for c in cols], axis=1)
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    # R11 #137: cross-section minimum breadth (mirror of group_ext).
    min_breadth = max(_MIN_BREADTH, int(_BREADTH_PARAM_RATIO * (2 if add_intercept else 1)))
    for t in range(rows):
        valid = np.isfinite(xv[t]) & np.isfinite(yv[t])
        if valid.sum() < min_breadth:
            continue
        xs = xv[t][valid]
        ys = yv[t][valid]
        cut = int(np.floor(trim * xs.size))
        if cut > 0:
            keep = np.ones(xs.size, dtype=bool)
            keep[np.argsort(xs)[:cut]] = False
            keep[np.argsort(xs)[-cut:]] = False
            xs = xs[keep]
            ys = ys[keep]
        if xs.size < (2 if add_intercept else 1) + 1 or np.std(xs) == 0:
            continue
        if add_intercept:
            coeffs = np.polyfit(xs, ys, 1)
            fitted = np.polyval(coeffs, xv[t])
        else:
            slope = float(np.sum(xs * ys) / np.sum(xs * xs))
            fitted = slope * xv[t]
        out[t] = yv[t] - fitted
    return _make(y, cols, out)


_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("group_ex_self_mean", ("x", "group"), group_ex_self_mean, "Group mean excluding self."),
    ("group_ex_self_weighted_mean", ("x", "weight", "group"), group_ex_self_weighted_mean, "Group weighted mean excluding self."),
    ("hierarchical_group_neutralize", ("x", "group", "subgroup"), hierarchical_group_neutralize, "Subgroup then group demean."),
    ("cs_robust_resid", ("y", "x", "trim_ratio", "add_intercept"), cs_robust_resid, "Trimmed-OLS cross-sectional residual (not a true robust regression)."),
)


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="group_neutralization",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsGroup_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="group_neutralization",
        business_category="group_neutralization",
        canonical=name,
        source="polars_group",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
