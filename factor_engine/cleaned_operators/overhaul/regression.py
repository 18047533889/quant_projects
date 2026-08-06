# -*- coding: utf-8 -*-
"""Audited rolling and cross-sectional regression operators."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    Spec,
    aligned_pd,
    frame_pd,
    pl_unary_rolling_map,
    register_specs,
    window_params,
)


def _fit_1d(
    y: np.ndarray,
    x: np.ndarray,
    add_intercept: bool,
) -> tuple[float, float, float, float, float] | None:
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
    dof = yv.size - design.shape[1]
    tstat = np.nan
    if dof > 0:
        sigma2 = sse / dof
        slope_var = sigma2 * np.linalg.pinv(design.T @ design)[-1, -1]
        if np.isfinite(slope_var) and slope_var > 0:
            tstat = float(beta[-1] / np.sqrt(slope_var))
    intercept = float(beta[0]) if add_intercept else 0.0
    slope = float(beta[-1])
    current_resid = (
        float(y[-1] - intercept - slope * x[-1])
        if np.isfinite(y[-1]) and np.isfinite(x[-1])
        else np.nan
    )
    return slope, intercept, current_resid, float(r2), float(tstat)


def _rolling_regression(
    y: pd.DataFrame,
    x: pd.DataFrame,
    window: int,
    min_periods: int | None,
    add_intercept: bool,
    output: str,
) -> pd.DataFrame:
    y, x = aligned_pd(y, x)
    w, mp = window_params(window, min_periods, default_mp=3)
    yv, xv = y.to_numpy(dtype=float), x.to_numpy(dtype=float)
    out = np.full(y.shape, np.nan, dtype=float)
    output_idx = {"slope": 0, "intercept": 1, "resid": 2, "r2": 3, "tstat": 4}[output]
    for col in range(y.shape[1]):
        for row in range(y.shape[0]):
            start = max(0, row - w + 1)
            yy, xx = yv[start : row + 1, col], xv[start : row + 1, col]
            if int(np.sum(np.isfinite(yy) & np.isfinite(xx))) < mp:
                continue
            fit = _fit_1d(yy, xx, bool(add_intercept))
            if fit is not None:
                out[row, col] = fit[output_idx]
    return frame_pd(y, out)


def pd_reg_slope(y, x, window, min_periods=None, add_intercept=True, **_):
    return _rolling_regression(y, x, window, min_periods, add_intercept, "slope")


def pd_reg_intercept(y, x, window, min_periods=None, add_intercept=True, **_):
    return _rolling_regression(y, x, window, min_periods, add_intercept, "intercept")


def pd_reg_resid(y, x, window, min_periods=None, add_intercept=True, **_):
    return _rolling_regression(y, x, window, min_periods, add_intercept, "resid")


def pd_reg_in_sample_resid(y, x, window, min_periods=None, add_intercept=True, **_):
    """样本内当前残差：训练窗含当前行（与 ``ts_regression_resid`` 一致）。"""
    return _rolling_regression(y, x, window, min_periods, add_intercept, "resid")


def _rolling_forecast_error(
    y: pd.DataFrame,
    x: pd.DataFrame,
    window: int,
    min_periods: int | None,
    add_intercept: bool,
    zscore: bool,
) -> pd.DataFrame:
    """out-of-sample 预测误差：训练窗截止 t-1，预测 t，误差 = y_t - ŷ_t。

    ``zscore=True`` 时输出误差除以样本内残差标准差（ddof=1）。
    """
    y, x = aligned_pd(y, x)
    w, mp = window_params(window, min_periods, default_mp=3)
    yv, xv = y.to_numpy(dtype=float), x.to_numpy(dtype=float)
    out = np.full(y.shape, np.nan, dtype=float)
    for col in range(y.shape[1]):
        for row in range(y.shape[0]):
            start = max(0, row - w + 1)
            yy, xx = yv[start:row, col], xv[start:row, col]  # 训练数据：t-window 至 t-1
            if int(np.sum(np.isfinite(yy) & np.isfinite(xx))) < mp:
                continue
            fit = _fit_1d(yy, xx, bool(add_intercept))
            if fit is None:
                continue
            slope, intercept, _, _, _ = fit
            if not (np.isfinite(yv[row, col]) and np.isfinite(xv[row, col])):
                continue
            err = yv[row, col] - (intercept + slope * xv[row, col])
            if not zscore:
                out[row, col] = float(err)
                continue
            pair = np.isfinite(yy) & np.isfinite(xx)
            r = yy[pair] - (intercept + slope * xx[pair])
            sd = float(np.std(r, ddof=1)) if r.size > 2 else np.nan
            out[row, col] = float(err / sd) if (sd and sd > 0) else np.nan
    return frame_pd(y, out)


def pd_reg_forecast_error(y, x, window, min_periods=None, add_intercept=True, **_):
    return _rolling_forecast_error(y, x, window, min_periods, bool(add_intercept), zscore=False)


def pd_reg_forecast_error_z(y, x, window, min_periods=None, add_intercept=True, **_):
    return _rolling_forecast_error(y, x, window, min_periods, bool(add_intercept), zscore=True)


def pd_reg_resid_mean(y, x, window, min_periods=None, add_intercept=True, **_):
    resid = _rolling_regression(y, x, window, min_periods, add_intercept, "resid")
    w, mp = window_params(window, min_periods, default_mp=3)
    return resid.rolling(w, min_periods=max(1, int(mp))).mean()


def pd_reg_r2(y, x, window, min_periods=None, add_intercept=True, **_):
    return _rolling_regression(y, x, window, min_periods, add_intercept, "r2")


def pd_reg_tstat(y, x, window, min_periods=None, add_intercept=True, **_):
    return _rolling_regression(y, x, window, min_periods, add_intercept, "tstat")


def pd_time_slope(x, window=20, min_periods=None, **_):
    w, mp = window_params(window, min_periods, default_mp=2)
    arr, out = x.to_numpy(dtype=float), np.full(x.shape, np.nan)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            values = arr[max(0, row - w + 1) : row + 1, col]
            mask = np.isfinite(values)
            if mask.sum() < mp:
                continue
            t = np.arange(values.size, dtype=float)[mask]
            y = values[mask]
            t -= t.mean()
            denom = float(t @ t)
            if denom > 0:
                out[row, col] = float(t @ (y - y.mean()) / denom)
    return frame_pd(x, out)


def pl_time_slope(x, window=20, min_periods=None, **_):
    w, mp = window_params(window, min_periods, default_mp=2)

    def fn(values):
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < mp:
            return np.nan
        t = np.arange(values.size, dtype=float)[mask]
        y = values[mask]
        t -= t.mean()
        denom = float(t @ t)
        return np.nan if denom <= 0 else float(t @ (y - y.mean()) / denom)

    return pl_unary_rolling_map(x, w, 1, fn)


def pd_trend_tstat(x, window, min_periods=None, **_):
    w, mp = window_params(window, min_periods, default_mp=3)
    arr, out = x.to_numpy(dtype=float), np.full(x.shape, np.nan)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            values = arr[max(0, row - w + 1) : row + 1, col]
            if np.isfinite(values).sum() < mp:
                continue
            fit = _fit_1d(values, np.arange(values.size, dtype=float), True)
            if fit is not None:
                out[row, col] = fit[4]
    return frame_pd(x, out)


def pd_max_drawdown(x, window, min_periods=2, **_):
    w, mp = window_params(window, min_periods, default_mp=2)
    mp = max(mp, 2)
    arr, out = x.to_numpy(dtype=float), np.full(x.shape, np.nan)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            values = arr[max(0, row - w + 1) : row + 1, col]
            valid = values[np.isfinite(values)]
            if valid.size < mp or np.any(valid <= 0):
                continue
            peaks = np.maximum.accumulate(valid)
            out[row, col] = float(np.min(valid / peaks - 1.0))
    return frame_pd(x, out)


def pd_partial_corr(x, y, z, window, min_periods=None, **_):
    x, y, z = aligned_pd(x, y, z)
    w, mp = window_params(window, min_periods, default_mp=3)
    xa, ya, za = (f.to_numpy(dtype=float) for f in (x, y, z))
    out = np.full(x.shape, np.nan)
    for col in range(x.shape[1]):
        for row in range(x.shape[0]):
            start = max(0, row - w + 1)
            xv, yv, zv = xa[start : row + 1, col], ya[start : row + 1, col], za[start : row + 1, col]
            mask = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(zv)
            if mask.sum() < mp or np.var(zv[mask]) <= 0:
                continue
            design = np.column_stack((np.ones(mask.sum()), zv[mask]))
            rx = xv[mask] - design @ np.linalg.lstsq(design, xv[mask], rcond=None)[0]
            ry = yv[mask] - design @ np.linalg.lstsq(design, yv[mask], rcond=None)[0]
            if np.std(rx) > 0 and np.std(ry) > 0:
                out[row, col] = float(np.clip(np.corrcoef(rx, ry)[0, 1], -1, 1))
    return frame_pd(x, out)


def pd_nth_value(x, window, n=1, order="largest", min_periods=None, **_):
    w, mp = window_params(window, min_periods, default_mp=n)
    n = int(n)
    if n < 1 or n > w:
        raise ValueError("n must satisfy 1 <= n <= window")
    if order not in {"largest", "smallest"}:
        raise ValueError("order must be 'largest' or 'smallest'")
    arr, out = x.to_numpy(dtype=float), np.full(x.shape, np.nan)
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            values = arr[max(0, row - w + 1) : row + 1, col]
            valid = np.sort(values[np.isfinite(values)])
            if valid.size >= max(mp, n):
                out[row, col] = valid[-n] if order == "largest" else valid[n - 1]
    return frame_pd(x, out)


def _ols_residual_1d(
    y: np.ndarray,
    features: Sequence[np.ndarray],
    *,
    weights: np.ndarray | None,
    add_intercept: bool,
    min_obs: int | None,
) -> np.ndarray:
    out = np.full(y.shape, np.nan)
    if not features:
        return out
    mask = np.isfinite(y)
    for feature in features:
        mask &= np.isfinite(feature)
    if weights is not None:
        mask &= np.isfinite(weights) & (weights > 0)
    coefficients = len(features) + (1 if add_intercept else 0)
    required = max(coefficients + 1, 3) if min_obs is None else int(min_obs)
    if required <= coefficients:
        raise ValueError("min_obs must exceed fitted coefficient count")
    if mask.sum() < required:
        return out
    design = np.column_stack([feature[mask] for feature in features])
    if add_intercept:
        design = np.column_stack((np.ones(design.shape[0]), design))
    if (
        design.shape[0] <= design.shape[1]
        or np.linalg.matrix_rank(design) < design.shape[1]
        or np.linalg.cond(design) > 1e12
    ):
        return out
    target = y[mask]
    fit_design, fit_target = design, target
    if weights is not None:
        root = np.sqrt(weights[mask])
        fit_design, fit_target = design * root[:, None], target * root
    beta, *_ = np.linalg.lstsq(fit_design, fit_target, rcond=None)
    out[mask] = target - design @ beta
    return out


def pd_cs_multi_resid(y, *features, add_intercept=True, min_obs=None, **_):
    if not features:
        raise ValueError("cs_multi_resid requires at least one exposure")
    aligned = aligned_pd(y, *features)
    target, xs = aligned[0], aligned[1:]
    out = np.full(target.shape, np.nan)
    for row in range(target.shape[0]):
        out[row] = _ols_residual_1d(
            target.iloc[row].to_numpy(dtype=float),
            [x.iloc[row].to_numpy(dtype=float) for x in xs],
            weights=None,
            add_intercept=bool(add_intercept),
            min_obs=min_obs,
        )
    return frame_pd(target, out)


def pd_cs_wls_resid(y, x, weight, add_intercept=True, min_obs=5, **_):
    y, x, weight = aligned_pd(y, x, weight)
    out = np.full(y.shape, np.nan)
    for row in range(y.shape[0]):
        out[row] = _ols_residual_1d(
            y.iloc[row].to_numpy(dtype=float),
            [x.iloc[row].to_numpy(dtype=float)],
            weights=weight.iloc[row].to_numpy(dtype=float),
            add_intercept=bool(add_intercept),
            min_obs=min_obs,
        )
    return frame_pd(y, out)


def pd_cs_neutralize(
    y,
    *exposures,
    group=None,
    weight=None,
    add_intercept=True,
    min_obs=None,
    **_,
):
    if not exposures and group is None:
        raise ValueError("cs_neutralize requires exposures and/or group")
    frames = [y, *exposures]
    if group is not None:
        frames.append(group)
    if weight is not None:
        frames.append(weight)
    aligned = aligned_pd(*frames)
    target = aligned[0]
    n_exposures = len(exposures)
    xs = list(aligned[1 : 1 + n_exposures])
    pos = 1 + n_exposures
    groups = aligned[pos] if group is not None else None
    pos += 1 if group is not None else 0
    weights = aligned[pos] if weight is not None else None
    out = np.full(target.shape, np.nan)
    for row in range(target.shape[0]):
        features = [x.iloc[row].to_numpy(dtype=float) for x in xs]
        if groups is not None:
            dummies = pd.get_dummies(groups.iloc[row], dummy_na=False, dtype=float)
            if add_intercept and dummies.shape[1] > 0:
                dummies = dummies.iloc[:, 1:]
            features.extend(dummies.iloc[:, j].to_numpy(dtype=float) for j in range(dummies.shape[1]))
        out[row] = _ols_residual_1d(
            target.iloc[row].to_numpy(dtype=float),
            features,
            weights=None if weights is None else weights.iloc[row].to_numpy(dtype=float),
            add_intercept=bool(add_intercept),
            min_obs=min_obs,
        )
    return frame_pd(target, out)


def register() -> None:
    regression_params = ["y", "x", "window", "min_periods", "add_intercept"]
    register_specs({
        "ts_regression_slope": Spec("time_series_regression", regression_params, "滚动 OLS 斜率", pd_reg_slope),
        "ts_regression_intercept": Spec("time_series_regression", regression_params, "滚动 OLS 截距", pd_reg_intercept),
        "ts_regression_resid": Spec("time_series_regression", regression_params, "滚动 OLS 当前残差", pd_reg_resid),
        "ts_regression_in_sample_resid": Spec("time_series_regression", regression_params, "样本内 OLS 当前残差（fit 含当前行）", pd_reg_in_sample_resid),
        "ts_regression_forecast_error": Spec("time_series_regression", regression_params, "out-of-sample OLS 预测误差（fit 截止 t-1）", pd_reg_forecast_error),
        "ts_regression_forecast_error_z": Spec("time_series_regression", regression_params, "OLS 预测误差 / 样本内残差 std", pd_reg_forecast_error_z),
        "ts_regression_resid_mean": Spec("time_series_regression", regression_params, "滚动 OLS 当前残差的窗口均值", pd_reg_resid_mean),
        "ts_regression_r2": Spec("time_series_regression", regression_params, "滚动 OLS 决定系数", pd_reg_r2),
        "ts_regression_tstat": Spec("time_series_regression", regression_params, "滚动 OLS 斜率 t 值", pd_reg_tstat),
        "ts_time_slope": Spec("time_series_regression", ["x", "window", "min_periods"], "缺失感知滚动时间斜率", pd_time_slope, pl_time_slope),
        "ts_trend_tstat": Spec("time_series_regression", ["x", "window", "min_periods"], "滚动时间趋势 t 值", pd_trend_tstat),
        "ts_max_drawdown": Spec("time_series_risk", ["x", "window", "min_periods"], "严格窗口最大回撤", pd_max_drawdown),
        "ts_partial_corr": Spec("time_series_regression", ["x", "y", "z", "window", "min_periods"], "共同样本滚动偏相关", pd_partial_corr),
        "ts_nth_value": Spec("time_series_order", ["x", "window", "n", "order", "min_periods"], "滚动第 N 大或第 N 小值", pd_nth_value),
        "cs_multi_resid": Spec("cross_sectional_regression", ["y", "x1", "x2", "...", "add_intercept", "min_obs"], "多变量横截面 OLS 残差", pd_cs_multi_resid),
        "cs_wls_resid": Spec("cross_sectional_regression", ["y", "x", "weight", "add_intercept", "min_obs"], "加权横截面回归残差", pd_cs_wls_resid),
        "cs_neutralize": Spec("cross_sectional_regression", ["y", "exposures", "group", "weight", "add_intercept", "min_obs"], "连续和分类风险暴露联合中性化", pd_cs_neutralize),
    })
