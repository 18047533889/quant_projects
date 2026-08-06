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


def _knn_blockwise(Xn: np.ndarray, k: int, block: int = 500) -> np.ndarray:
    """k-NN mean distances over a standardised matrix, computed in row blocks.

    A single ``N x N`` distance matrix for a 5000-instrument cross-section is
    ~200 MB of float64; blockwise rows bound peak memory while keeping the
    result identical.
    """
    n = Xn.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        d2 = np.sum((Xn[lo:hi, None, :] - Xn[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        kk = min(k, n - 1)
        if kk < 1:
            continue
        part = np.partition(d2, kth=kk - 1, axis=1)[:, :kk]
        out[lo:hi] = np.sqrt(part).mean(axis=1)
    return out


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
        out[row, valid] = _knn_blockwise(Xn, kk)
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


def _cs_shrinkage_mahalanobis(*features, shrinkage=0.1):
    s = float(shrinkage)
    if not (0.0 <= s < 1.0):
        raise ValueError("shrinkage must be in [0, 1)")
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
        cov = np.atleast_2d(np.cov(Xv.T))  # single-feature cross-sections -> 1x1
        shrunk = (1.0 - s) * cov + s * np.diag(np.diag(cov))
        try:
            icov = np.linalg.inv(shrunk + _EPS * np.eye(p))
        except np.linalg.LinAlgError:
            continue
        d = np.sqrt(np.einsum("ij,jk,ik->i", Xv - mu, icov, Xv - mu))
        out[row, valid] = d
    return _frame_like(features[0], out)


_mk(
    "cs_shrinkage_mahalanobis",
    "收缩协方差的马氏距离（向对角收缩，缓解病态协方差）。",
    ["f1", "f2", "f3", "f4", "shrinkage"],
    lambda f1, f2=None, f3=None, f4=None, shrinkage=0.1: _cs_shrinkage_mahalanobis(
        *[v for v in (f1, f2, f3, f4) if v is not None], shrinkage=float(shrinkage)),
    unit="distance",
)


def _cs_robust_mahalanobis_mad(*features):
    """MAD-based robust Mahalanobis: median centre + per-dimension MAD scale."""
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
        center = np.median(Xv, axis=0)
        mad = 1.4826 * np.median(np.abs(Xv - center), axis=0)
        mad = np.where(mad > _EPS, mad, 1.0)
        z = (Xv - center) / mad
        d = np.sqrt(np.sum(z * z, axis=1))
        out[row, valid] = d
    return _frame_like(features[0], out)


_mk(
    "cs_robust_mahalanobis_mad",
    "MAD 稳健马氏距离（中位数中心 + MAD 尺度）。",
    ["f1", "f2", "f3", "f4"],
    lambda f1, f2=None, f3=None, f4=None: _cs_robust_mahalanobis_mad(
        *[v for v in (f1, f2, f3, f4) if v is not None]),
    unit="distance",
)


def _knn_full(Xn: np.ndarray, k: int, block: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """k nearest neighbours (indices + distances) over a standardised matrix,
    computed in row blocks."""
    n = Xn.shape[0]
    kk = max(1, min(k, n - 1))
    idx = np.zeros((n, kk), dtype=int)
    dist = np.zeros((n, kk), dtype=float)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        d2 = np.sum((Xn[lo:hi, None, :] - Xn[None, :, :]) ** 2, axis=2)
        for i in range(lo, hi):
            d2[i - lo, i] = np.inf
        part = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        pd2 = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(pd2, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        pd2 = np.take_along_axis(pd2, order, axis=1)
        idx[lo:hi] = part
        dist[lo:hi] = np.sqrt(pd2)
    return idx, dist


def _lof_row(Xn: np.ndarray, k: int) -> np.ndarray:
    kk = max(1, min(k, Xn.shape[0] - 1))
    n = Xn.shape[0]
    if n < kk + 1:
        return np.full(n, np.nan)
    idx, dist = _knn_full(Xn, kk)
    k_dist = dist[:, -1]  # distance to kth neighbour
    # local reachability density = 1 / mean reachability distance
    reach = np.maximum(k_dist[idx], dist)
    lrd = 1.0 / np.maximum(reach.mean(axis=1), _EPS)
    # LOF = mean(neighbour lrd) / lrd_i
    lof = lrd[idx].mean(axis=1) / np.maximum(lrd, _EPS)
    return lof


def _cs_actual_lof(*features, k=20):
    kk = max(2, int(k))
    fv = [f.to_numpy(dtype=float) for f in features]
    n, n_cols = fv[0].shape
    p = len(fv)
    out = np.full((n, n_cols), np.nan, dtype=float)
    for row in range(n):
        X = np.column_stack([f[row] for f in fv])
        valid = np.all(np.isfinite(X), axis=1)
        nv = int(valid.sum())
        eff_k = max(1, min(kk, nv - 1))
        if nv < eff_k + 2:
            continue
        Xv = X[valid]
        sd = np.std(Xv, axis=0)
        Xn = Xv / np.where(sd > _EPS, sd, 1.0)
        out[row, valid] = _lof_row(Xn, eff_k)
    return _frame_like(features[0], out)


_mk(
    "cs_actual_lof_score",
    "局部离群因子 LOF（分块计算，避免全量距离矩阵）。",
    ["f1", "f2", "f3", "f4", "k"],
    lambda f1, f2=None, f3=None, f4=None, k=20: _cs_actual_lof(
        *[v for v in (f1, f2, f3, f4) if v is not None], k=int(k)),
    unit="level",
)


def _relative_density_row(Xn: np.ndarray, k: int) -> np.ndarray:
    kk = max(1, min(k, Xn.shape[0] - 1))
    n = Xn.shape[0]
    if n < kk + 1:
        return np.full(n, np.nan)
    idx, dist = _knn_full(Xn, kk)
    dens = 1.0 / np.maximum(dist.mean(axis=1), _EPS)  # own density proxy
    neigh_dens = dens[idx].mean(axis=1)               # mean neighbour density
    return dens / np.maximum(neigh_dens, _EPS)


def _cs_relative_density(*features, k=20):
    kk = max(2, int(k))
    fv = [f.to_numpy(dtype=float) for f in features]
    n, n_cols = fv[0].shape
    p = len(fv)
    out = np.full((n, n_cols), np.nan, dtype=float)
    for row in range(n):
        X = np.column_stack([f[row] for f in fv])
        valid = np.all(np.isfinite(X), axis=1)
        nv = int(valid.sum())
        eff_k = max(1, min(kk, nv - 1))
        if nv < eff_k + 2:
            continue
        Xv = X[valid]
        sd = np.std(Xv, axis=0)
        Xn = Xv / np.where(sd > _EPS, sd, 1.0)
        out[row, valid] = _relative_density_row(Xn, eff_k)
    return _frame_like(features[0], out)


_mk(
    "cs_relative_density_ratio",
    "局部密度相对其 k 近邻平均密度的比值。",
    ["f1", "f2", "f3", "f4", "k"],
    lambda f1, f2=None, f3=None, f4=None, k=20: _cs_relative_density(
        *[v for v in (f1, f2, f3, f4) if v is not None], k=int(k)),
)


def _cs_residual_percentile(resid: pd.DataFrame) -> pd.DataFrame:
    rv = resid.to_numpy(dtype=float)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        a = rv[row]
        valid = np.isfinite(a)
        if valid.sum() < 2:
            continue
        order = np.argsort(np.argsort(a[valid]))
        out[row, valid] = order / (valid.sum() - 1.0)
    return _frame_like(resid, out)


_mk(
    "cs_residual_percentile",
    "横截面残差分位（0~1，横截面 rank 归一化）。",
    ["resid"],
    _cs_residual_percentile,
)
