# -*- coding: utf-8 -*-
"""Rolling multi-variable / robust / quantile regression operators (P0).

Daily panels in, daily panels out.  Each kernel is a causal rolling regression
per (instrument, date): the current row's statistic uses only the window ending
at that row.  All fits degrade to NaN rather than fabricate values.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.ts_model._rolling_core import (
    aligned,
    build_design,
    frame_like,
    huber_fit,
    metadata,
    ols_fit,
    quantile_fit,
    ridge_fit,
    rolling_fit,
)

# The historic IRLS ``quantile_fit`` is an expectile fit; the ``expectile_*``
# operators registered below share the exact same kernel with an honest name.
from cleaned_operators.ts_model._rolling_core import expectile_fit  # noqa: E402

_CANONICALS: list[str] = []


def _gather_features(args: tuple[Any, ...], n: int) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    for i in range(n):
        x = args[i]
        if x is not None:
            out.append(x)
    return out


def _build_fit(fit_fn: Callable[..., Any], extra: Any, add_intercept: bool) -> Callable[..., Any]:
    """Bind the per-window fit callable.

    Ridge needs to know whether the design starts with an intercept so it only
    exempts the intercept column from the L2 penalty.  ``extra`` carries the
    ridge ``alpha`` (or any other scalar hyper-parameter of ``fit_fn``).
    """
    if fit_fn is ridge_fit:
        return lambda d, v: ridge_fit(d, v, extra, has_intercept=bool(add_intercept))
    if extra is not None:
        return lambda d, v: fit_fn(d, v, extra)
    return fit_fn


def _multi_regression(
    y: pd.DataFrame,
    features: list[pd.DataFrame],
    window: int,
    min_periods: int,
    add_intercept: bool,
    fit_fn: Callable[..., Any],
    extra: Any,
    stat: str,
    coeff_index: int,
    *,
    fit_lag: int = 0,
    stability_k: int = 0,
) -> pd.DataFrame:
    """Rolling multi-variable regression over ``(row, col)`` panels.

    ``fit_lag`` is the number of rows between the end of the training window
    and the row whose statistic is reported.  ``fit_lag=0`` (legacy) trains on
    the window *including* the current row and reports the in-sample residual.
    ``fit_lag>=1`` trains on rows strictly before the current one and reports
    the out-of-sample forecast error at the current row.  ``stability_k>0``
    additionally reports the standard deviation of the fitted coefficient over
    the last ``stability_k`` consecutive fits (each ending ``fit_lag`` rows
    before its row).
    """
    if not features:
        raise ValueError("at least one feature panel is required")
    frames = [y] + features
    aligned_frames = aligned(*frames)
    y = aligned_frames[0]
    feats = aligned_frames[1:]
    yv = y.to_numpy(dtype=float)
    rows, cols = yv.shape
    xs = [f.to_numpy(dtype=float) for f in feats]
    n_coeffs = len(feats) + (1 if add_intercept else 0)
    if coeff_index < 0 or coeff_index >= n_coeffs:
        raise ValueError(f"coefficient_index {coeff_index} out of range [0, {n_coeffs})")
    if stability_k > 0 and stat != "coeff":
        raise ValueError("stability_k>0 requires stat='coeff'")
    fit = _build_fit(fit_fn, extra, add_intercept)
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    mp = max(int(min_periods), n_coeffs + 1)
    lag = max(0, int(fit_lag))
    k = max(0, int(stability_k))
    for col in range(cols):
        ycol = yv[:, col]
        xcols = [x[:, col] for x in xs]
        for row in range(rows):
            fit_end = row - lag
            if fit_end < 0:
                continue
            start = max(0, fit_end - w + 1)
            seg_y = ycol[start : fit_end + 1]
            seg_xs = [x[start : fit_end + 1] for x in xcols]
            valid = np.isfinite(seg_y)
            for x in seg_xs:
                valid &= np.isfinite(x)
            if valid.sum() < mp:
                continue
            vy = seg_y[valid]
            vxs = [x[valid] for x in seg_xs]
            if any(np.std(vx) <= 0.0 for vx in vxs):
                continue
            design = build_design(vxs, add_intercept)
            b = fit(design, vy)
            if b is None:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                pred = design @ b
                e = vy - pred
            if stat == "coeff":
                if k > 0:
                    coeffs = []
                    for j in range(k):
                        fe = row - lag - j
                        if fe < 0:
                            break
                        s2 = max(0, fe - w + 1)
                        sy = ycol[s2 : fe + 1]
                        sx = [x[s2 : fe + 1] for x in xcols]
                        v2 = np.isfinite(sy)
                        for x in sx:
                            v2 &= np.isfinite(x)
                        if v2.sum() < mp:
                            break
                        vy2 = sy[v2]
                        vx2 = [x[v2] for x in sx]
                        if any(np.std(vx2) <= 0.0 for vx2 in vx2):
                            break
                        d2 = build_design(vx2, add_intercept)
                        b2 = fit(d2, vy2)
                        if b2 is None:
                            break
                        coeffs.append(float(b2[coeff_index]))
                    if len(coeffs) >= 2:
                        out[row, col] = float(np.std(coeffs))
                else:
                    out[row, col] = float(b[coeff_index])
            elif stat == "resid":
                if np.isfinite(ycol[row]):
                    cur_xs = [x[row] for x in xcols]
                    terms = ([1.0] if add_intercept else []) + cur_xs
                    out[row, col] = float(ycol[row] - float(np.dot(terms, b)))
            elif stat == "resid_z":
                if np.isfinite(ycol[row]):
                    ddof = max(design.shape[1], 1)
                    sd = float(np.sqrt(np.sum(e * e) / max(len(e) - ddof, 1))) if len(e) > ddof else np.nan
                    if sd is not None and np.isfinite(sd) and sd > 0.0:
                        cur_xs = [x[row] for x in xcols]
                        terms = ([1.0] if add_intercept else []) + cur_xs
                        resid = float(ycol[row] - float(np.dot(terms, b)))
                        out[row, col] = resid / sd
            elif stat in ("r2", "r2_adj"):
                ss_res = float(np.sum(e * e))
                ss_tot = float(np.sum((vy - np.mean(vy)) ** 2))
                if ss_tot > 0.0:
                    r2 = float(max(0.0, 1.0 - ss_res / ss_tot))
                    if stat == "r2_adj":
                        n = len(vy)
                        p = design.shape[1]
                        out[row, col] = float(1.0 - (1.0 - r2) * (n - 1.0) / max(n - p - 1.0, 1.0))
                    else:
                        out[row, col] = r2
    return frame_like(y, out)


def _register_multi(name: str, description: str, fit_fn: Callable[..., Any], extra: Any, stat: str, unit: str,
                    *, fit_lag: int = 0, stability_k: int = 0, cost: int = 5):
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.dynamic_regression",
        backend="pandas_numpy",
        status="experimental",
    )
    class _MultiOp(SeriesOperator):
        metadata = metadata(
            name, description,
            ["y", "x1", "x2", "x3", "x4", "window", "coefficient_index", "min_periods", "add_intercept"],
            unit=unit, cost=cost,
        )

        def _calculate_series(self, y, x1=None, x2=None, x3=None, x4=None,
                              window=60, coefficient_index=1, min_periods=10, add_intercept=True, **_):
            feats = _gather_features((x1, x2, x3, x4), 4)
            return _multi_regression(
                y, feats, int(window), int(min_periods), bool(add_intercept),
                fit_fn, extra, stat, int(coefficient_index),
                fit_lag=int(fit_lag), stability_k=int(stability_k),
            )

    return _MultiOp


_register_multi(
    "ts_multi_regression_coeff", "多变量滚动回归指定系数。",
    ols_fit, None, "coeff", "level",
)
_register_multi(
    "ts_multi_regression_resid", "多变量滚动回归当前残差。",
    ols_fit, None, "resid", "level",
)
_register_multi(
    "ts_multi_regression_resid_z", "多变量滚动回归标准化残差。",
    ols_fit, None, "resid_z", "level",
)
_register_multi(
    "ts_multi_regression_r2", "多变量滚动回归 R²。",
    ols_fit, None, "r2", "r2",
)
_register_multi(
    "ts_huber_regression_coeff", "Huber 稳健回归斜率。",
    huber_fit, None, "coeff", "level",
)
_register_multi(
    "ts_huber_regression_resid_z", "Huber 稳健回归标准化残差。",
    huber_fit, None, "resid_z", "level",
)
_register_multi(
    "ts_ridge_regression_coeff", "岭回归系数。",
    ridge_fit, 0.1, "coeff", "level",
)
_register_multi(
    "ts_ridge_regression_resid_z", "岭回归标准化残差。",
    ridge_fit, 0.1, "resid_z", "level",
)

# ---------------------------------------------------------------------------
# Prior-window (out-of-sample) variants.  ``fit_lag=1`` trains each window on
# rows strictly before the current one and reports the current row's coefficient
# / forecast error against that historical fit, so the output is a true
# out-of-sample anomaly rather than an in-sample residual.
# ---------------------------------------------------------------------------
_register_multi(
    "ts_multi_regression_coeff_prior", "多变量回归系数（截至 t-1 训练）。",
    ols_fit, None, "coeff", "level", fit_lag=1,
)
_register_multi(
    "ts_multi_regression_forecast_error", "多变量回归当前样本相对历史拟合的预测误差。",
    ols_fit, None, "resid", "level", fit_lag=1,
)
_register_multi(
    "ts_multi_regression_forecast_error_z", "多变量回归预测误差 / 历史训练窗口残差标准差。",
    ols_fit, None, "resid_z", "level", fit_lag=1,
)
_register_multi(
    "ts_multi_regression_r2_prior", "多变量回归训练窗口 R²（截至 t-1）。",
    ols_fit, None, "r2", "r2", fit_lag=1,
)
_register_multi(
    "ts_multi_regression_adjusted_r2_prior", "多变量回归训练窗口调整 R²（截至 t-1）。",
    ols_fit, None, "r2_adj", "r2", fit_lag=1,
)
_register_multi(
    "ts_multi_regression_coeff_stability", "多变量回归系数在最近 K 个滚动窗口的标准差。",
    ols_fit, None, "coeff", "level", fit_lag=1, stability_k=5, cost=7,
)
_register_multi(
    "ts_huber_regression_coeff_prior", "Huber 稳健回归斜率（截至 t-1 训练）。",
    huber_fit, None, "coeff", "level", fit_lag=1,
)
_register_multi(
    "ts_huber_regression_forecast_error", "Huber 回归预测误差（截至 t-1 训练）。",
    huber_fit, None, "resid", "level", fit_lag=1,
)
_register_multi(
    "ts_huber_regression_forecast_error_z", "Huber 回归预测误差 / 历史残差标准差。",
    huber_fit, None, "resid_z", "level", fit_lag=1,
)
_register_multi(
    "ts_ridge_regression_coeff_prior", "岭回归系数（截至 t-1 训练）。",
    ridge_fit, 0.1, "coeff", "level", fit_lag=1,
)
_register_multi(
    "ts_ridge_regression_forecast_error", "岭回归预测误差（截至 t-1 训练）。",
    ridge_fit, 0.1, "resid", "level", fit_lag=1,
)
_register_multi(
    "ts_ridge_regression_forecast_error_z", "岭回归预测误差 / 历史残差标准差。",
    ridge_fit, 0.1, "resid_z", "level", fit_lag=1,
)


def _single_regression(
    y: pd.DataFrame,
    x: pd.DataFrame,
    window: int,
    min_periods: int,
    add_intercept: bool,
    fit_fn: Callable[..., Any],
    extra: Any,
    stat: str,
    q: float,
    coeff_index: int,
    *,
    fit_lag: int = 0,
) -> pd.DataFrame:
    y, x = aligned(y, x)
    return _multi_regression(
        y, [x], int(window), int(min_periods), bool(add_intercept),
        fit_fn, extra, stat, int(coeff_index), fit_lag=int(fit_lag),
    )


def _expectile_op(name: str, description: str, stat: str, *, fit_lag: int = 0):
    """Register an explicitly-named expectile regression operator.

    The underlying kernel is the IRLS asymmetric-weighted least squares that is
    historically called ``quantile_*`` in this codebase; the ``expectile_*``
    names carry the honest mathematical label (expectile, not pinball quantile).
    """
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.dynamic_regression",
        backend="pandas_numpy",
        status="experimental",
    )
    class _ExpectileOp(SeriesOperator):
        metadata = metadata(
            name, description, ["y", "x", "window", "q", "min_periods"], unit="level",
        )

        def _calculate_series(self, y, x, window=60, q=0.5, min_periods=10, **_):
            qv = float(q)
            if not (0.0 < qv < 1.0):
                raise ValueError("q must be in (0, 1)")
            return _single_regression(y, x, int(window), int(min_periods), True,
                                      expectile_fit, qv, stat, qv, 1, fit_lag=int(fit_lag))

    _CANONICALS.append(name)
    return _ExpectileOp


_expectile_op("ts_expectile_regression_coeff", "expectile 回归斜率（IRLS 非对称加权最小二乘）。", "coeff")
_expectile_op("ts_expectile_regression_resid", "expectile 回归当前残差。", "resid")
_expectile_op("ts_expectile_regression_coeff_prior", "expectile 回归斜率（截至 t-1 训练）。", "coeff", fit_lag=1)
_expectile_op("ts_expectile_regression_forecast_error", "expectile 回归预测误差（截至 t-1 训练）。", "resid", fit_lag=1)


@register_operator(
    name="ts_expectile_beta_spread",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_expectile_beta_spread",
    source="ts_model.dynamic_regression",
    backend="pandas_numpy",
    status="experimental",
)
class TsExpectileBetaSpread(SeriesOperator):
    """高 expectile Beta 减 低 expectile Beta：上涨/下跌尾部非对称响应。"""

    metadata = metadata(
        "ts_expectile_beta_spread", "expectile(q_high) - expectile(q_low)。",
        ["y", "x", "window", "q_high", "q_low", "min_periods"], unit="level",
    )

    def _calculate_series(self, y, x, window=60, q_high=0.9, q_low=0.1, min_periods=10, **_):
        qh, ql = float(q_high), float(q_low)
        if not (0.0 < ql < qh < 1.0):
            raise ValueError("q_low < q_high must hold in (0, 1)")
        high = _single_regression(y, x, int(window), int(min_periods), True,
                                  expectile_fit, qh, "coeff", qh, 1)
        low = _single_regression(y, x, int(window), int(min_periods), True,
                                 expectile_fit, ql, "coeff", ql, 1)
        return high - low


@register_operator(
    name="ts_quantile_regression_coeff",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_coeff",
    source="ts_model.dynamic_regression",
    backend="pandas_numpy",
    status="experimental",
)
class TsQuantileRegressionCoeff(SeriesOperator):
    """条件分位数回归斜率（IRLS 近似）。"""

    metadata = metadata(
        "ts_quantile_regression_coeff", "分位数回归斜率。",
        ["y", "x", "window", "q", "min_periods"], unit="level",
    )

    def _calculate_series(self, y, x, window=60, q=0.5, min_periods=10, **_):
        quantile = float(q)
        if not (0.0 < quantile < 1.0):
            raise ValueError("q must be in (0, 1)")
        return _single_regression(y, x, int(window), int(min_periods), True,
                                  quantile_fit, quantile, "coeff", quantile, 1)


@register_operator(
    name="ts_quantile_regression_resid",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_resid",
    source="ts_model.dynamic_regression",
    backend="pandas_numpy",
    status="experimental",
)
class TsQuantileRegressionResid(SeriesOperator):
    """当前值相对条件分位数预测的偏差。"""

    metadata = metadata(
        "ts_quantile_regression_resid", "分位数回归残差。",
        ["y", "x", "window", "q", "min_periods"], unit="level",
    )

    def _calculate_series(self, y, x, window=60, q=0.5, min_periods=10, **_):
        quantile = float(q)
        if not (0.0 < quantile < 1.0):
            raise ValueError("q must be in (0, 1)")
        return _single_regression(y, x, int(window), int(min_periods), True,
                                  quantile_fit, quantile, "resid", quantile, 1)


@register_operator(
    name="ts_quantile_beta_spread",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_quantile_beta_spread",
    source="ts_model.dynamic_regression",
    backend="pandas_numpy",
    status="experimental",
)
class TsQuantileBetaSpread(SeriesOperator):
    """高分位 Beta 减 低分位 Beta：上涨/下跌尾部非对称响应。"""

    metadata = metadata(
        "ts_quantile_beta_spread", "beta(q_high) - beta(q_low)。",
        ["y", "x", "window", "q_high", "q_low", "min_periods"], unit="level",
    )

    def _calculate_series(self, y, x, window=60, q_high=0.9, q_low=0.1, min_periods=10, **_):
        qh, ql = float(q_high), float(q_low)
        if not (0.0 < ql < qh < 1.0):
            raise ValueError("q_low < q_high must hold in (0, 1)")
        high = _single_regression(y, x, int(window), int(min_periods), True,
                                  quantile_fit, qh, "coeff", qh, 1)
        low = _single_regression(y, x, int(window), int(min_periods), True,
                                 quantile_fit, ql, "coeff", ql, 1)
        return high - low


_CANONICALS.extend(
    [
        "ts_multi_regression_coeff",
        "ts_multi_regression_resid",
        "ts_multi_regression_resid_z",
        "ts_multi_regression_r2",
        "ts_huber_regression_coeff",
        "ts_huber_regression_resid_z",
        "ts_ridge_regression_coeff",
        "ts_ridge_regression_resid_z",
        "ts_quantile_regression_coeff",
        "ts_quantile_regression_resid",
        "ts_quantile_beta_spread",
        "ts_multi_regression_coeff_prior",
        "ts_multi_regression_forecast_error",
        "ts_multi_regression_forecast_error_z",
        "ts_multi_regression_r2_prior",
        "ts_multi_regression_adjusted_r2_prior",
        "ts_multi_regression_coeff_stability",
        "ts_huber_regression_coeff_prior",
        "ts_huber_regression_forecast_error",
        "ts_huber_regression_forecast_error_z",
        "ts_ridge_regression_coeff_prior",
        "ts_ridge_regression_forecast_error",
        "ts_ridge_regression_forecast_error_z",
        "ts_expectile_regression_coeff",
        "ts_expectile_regression_resid",
        "ts_expectile_regression_coeff_prior",
        "ts_expectile_regression_forecast_error",
        "ts_expectile_beta_spread",
    ]
)

import cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.EXTENDED_ONLY_CANONICALS = frozenset(
    set(_surface.EXTENDED_ONLY_CANONICALS) | set(_CANONICALS)
)
