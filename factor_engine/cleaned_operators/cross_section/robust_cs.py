# -*- coding: utf-8 -*-
"""Robust cross-sectional residual and density operators (P1/P2).

For each trading row the operator fits a model across instruments and returns a
per-instrument scalar (residual / distance / density anomaly).  Kernels are
causal (only the current day's cross-section is used).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12
_CANONICALS: list[str] = []


def _meta(name: str, description: str, params: list[str], *, unit: str = "level") -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="cross_sectional",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "cross_sectional", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:cs",
            f"unit:{unit}", "cost:4",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _mk(name: str, description: str, params: list[str], fn, *, unit: str = "level"):
    metadata = _meta(name, description, params, unit=unit)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"RobustCs_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="cross_sectional",
        business_category="cross_sectional",
        canonical=name,
        source="cross_section.robust_cs",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {name}
    )
    return cls


def _design(features: list[np.ndarray], add_intercept: bool = True) -> np.ndarray:
    cols = [np.ones(len(features[0]))] if add_intercept else []
    cols.extend(features)
    return np.column_stack(cols)


def _row_wise(y: pd.DataFrame, features: list[pd.DataFrame], fn) -> np.ndarray:
    yv = y.to_numpy(dtype=float)
    fv = [f.to_numpy(dtype=float) for f in features]
    rows = yv.shape[0]
    out = np.full(yv.shape, np.nan, dtype=float)
    for row in range(rows):
        out[row] = fn(yv[row], [f[row] for f in fv])
    return out


def _cs_ridge_resid(y, *features, alpha=0.1, add_intercept=True):
    a = float(alpha)

    def _fn(yrow, frows):
        valid = np.isfinite(yrow)
        for f in frows:
            valid &= np.isfinite(f)
        if valid.sum() < 5:
            return np.full(len(yrow), np.nan)
        X = _design([f[valid] for f in frows], add_intercept)
        yy = yrow[valid]
        pen = a * np.eye(X.shape[1])
        pen[0, 0] = 0.0
        beta, *_ = np.linalg.lstsq(X.T @ X + pen, X.T @ yy, rcond=None)
        out = np.full(len(yrow), np.nan)
        out[valid] = yrow[valid] - X @ beta
        return out

    return _frame_like(y, _row_wise(y, list(features), _fn))


_mk(
    "cs_ridge_resid",
    "每日横截面岭回归残差。",
    ["y", "x1", "x2", "x3", "x4", "alpha", "add_intercept"],
    lambda y, x1=None, x2=None, x3=None, x4=None, alpha=0.1, add_intercept=True:
    _cs_ridge_resid(y, *[v for v in (x1, x2, x3, x4) if v is not None], alpha=alpha, add_intercept=add_intercept),
)


def _cs_quantile_resid(y, x, q=0.5):
    quantile = float(q)
    if not (0.0 < quantile < 1.0):
        raise ValueError("q must be in (0,1)")

    def _fn(yrow, frows):
        xr = frows[0]
        valid = np.isfinite(yrow) & np.isfinite(xr)
        if valid.sum() < 5 or np.std(xr[valid]) <= _EPS:
            return np.full(len(yrow), np.nan)
        X = np.column_stack([np.ones(valid.sum()), xr[valid]])
        yy = yrow[valid]
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        for _ in range(8):
            resid = yy - X @ beta
            weight = np.clip(np.where(resid > 0, quantile, 1.0 - quantile), 1e-6, None)
            beta, *_ = np.linalg.lstsq(X * weight[:, None], yy * weight, rcond=None)
        out = np.full(len(yrow), np.nan)
        out[valid] = yrow[valid] - X @ beta
        return out

    return _frame_like(y, _row_wise(y, [x], _fn))


_mk("cs_quantile_resid", "每日横截面分位数回归残差。", ["y", "x", "q"], _cs_quantile_resid)


def _cs_spline_resid(y, x, knots=4):
    k = max(2, int(knots))

    def _fn(yrow, frows):
        xr = frows[0]
        valid = np.isfinite(yrow) & np.isfinite(xr)
        if valid.sum() < max(6, k * 2):
            return np.full(len(yrow), np.nan)
        xs, ys = xr[valid], yrow[valid]
        qs = np.quantile(xs, np.linspace(0, 1, k + 1))
        qs = np.unique(qs)
        if len(qs) < 2:
            return np.full(len(yrow), np.nan)
        # piecewise-linear basis (truncated linear splines) -> OLS
        bases = [np.ones(len(xs))]
        bases.extend([np.maximum(xs - knot, 0.0) for knot in qs[1:]])
        X = np.column_stack(bases)
        beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        out = np.full(len(yrow), np.nan)
        pred = np.ones(len(yrow))
        for c, knot in enumerate(qs[1:]):
            pred = pred + beta[c + 1] * np.maximum(xr - knot, 0.0)
        pred = beta[0] + pred - 1.0
        out[valid] = yrow[valid] - (beta[0] + sum(beta[c + 1] * np.maximum(xr[valid] - knot, 0.0) for c, knot in enumerate(qs[1:])))
        return out

    return _frame_like(y, _row_wise(y, [x], _fn))


_mk("cs_spline_resid", "每日横截面线性样条回归残差。", ["y", "x", "knots"], _cs_spline_resid)


def _cs_mahalanobis(*features):
    fv = [f.to_numpy(dtype=float) for f in features]
    n, n_cols = fv[0].shape
    p = len(fv)
    out = np.full((n, n_cols), np.nan, dtype=float)
    for row in range(n):
        X = np.column_stack([f[row] for f in fv])
        valid = np.all(np.isfinite(X), axis=1)
        if valid.sum() < p + 3:
            continue
        Xv = X[valid]
        mu = Xv.mean(axis=0)
        cov = np.cov(Xv.T)
        try:
            icov = np.linalg.inv(cov + _EPS * np.eye(p))
        except np.linalg.LinAlgError:
            continue
        d = np.sqrt(np.einsum("ij,jk,ik->i", Xv - mu, icov, Xv - mu))
        out[row, valid] = d
    return _frame_like(features[0], out)


_mk(
    "cs_mahalanobis_distance",
    "多特征空间距横截面中心的马氏距离。",
    ["f1", "f2", "f3", "f4"],
    lambda f1, f2=None, f3=None, f4=None: _cs_mahalanobis(*[v for v in (f1, f2, f3, f4) if v is not None]),
    unit="distance",
)


def _cs_knn_distance(*features, k=5):
    kk = max(2, int(k))
    fv = [f.to_numpy(dtype=float) for f in features]
    n, n_cols = fv[0].shape
    p = len(fv)
    out = np.full((n, n_cols), np.nan, dtype=float)
    for row in range(n):
        X = np.column_stack([f[row] for f in fv])
        valid = np.all(np.isfinite(X), axis=1)
        if valid.sum() < kk + 1:
            continue
        Xv = X[valid]
        sd = np.std(Xv, axis=0)
        Xn = Xv / np.where(sd > _EPS, sd, 1.0)
        # pairwise euclidean
        d2 = np.sum((Xn[:, None, :] - Xn[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        idx = np.argsort(d2, axis=1)[:, :kk]
        dist = np.sqrt(np.take_along_axis(d2, idx, axis=1)).mean(axis=1)
        out[row, valid] = dist
    return _frame_like(features[0], out)


_mk(
    "cs_knn_distance",
    "多特征空间到 K 近邻的平均距离。",
    ["f1", "f2", "f3", "f4", "k"],
    lambda f1, f2=None, f3=None, f4=None, k=5: _cs_knn_distance(*[v for v in (f1, f2, f3, f4) if v is not None], k=int(k)),
    unit="distance",
)


def _cs_local_density_score(*features, k=5, radius=1.0):
    knn = _cs_knn_distance(*features, k=int(k)).to_numpy(dtype=float)
    return -np.log(knn + _EPS)


_mk(
    "cs_local_density_score",
    "多特征空间局部密度异常度（负对数 KNN 距离）。",
    ["f1", "f2", "f3", "f4", "k"],
    lambda f1, f2=None, f3=None, f4=None, k=5: _cs_local_density_score(*[v for v in (f1, f2, f3, f4) if v is not None], k=int(k)),
)
