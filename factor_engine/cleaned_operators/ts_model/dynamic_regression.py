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

_CANONICALS: list[str] = []


def _gather_features(args: tuple[Any, ...], n: int) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    for i in range(n):
        x = args[i]
        if x is not None:
            out.append(x)
    return out


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
) -> pd.DataFrame:
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
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    mp = max(int(min_periods), n_coeffs + 1)
    for col in range(cols):
        ycol = yv[:, col]
        xcols = [x[:, col] for x in xs]
        for row in range(rows):
            start = max(0, row - w + 1)
            seg_y = ycol[start : row + 1]
            seg_xs = [x[start : row + 1] for x in xcols]
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
            b = fit_fn(design, vy, extra) if extra is not None else fit_fn(design, vy)
            if b is None:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                pred = design @ b
                e = vy - pred
            if stat == "coeff":
                out[row, col] = float(b[coeff_index])
            elif stat == "resid":
                if np.isfinite(seg_y[-1]):
                    cur_xs = [x[-1] for x in seg_xs]
                    terms = ([1.0] if add_intercept else []) + cur_xs
                    out[row, col] = float(seg_y[-1] - float(np.dot(terms, b)))
            elif stat == "resid_z":
                if np.isfinite(seg_y[-1]):
                    ddof = max(design.shape[1], 1)
                    sd = float(np.sqrt(np.sum(e * e) / max(len(e) - ddof, 1))) if len(e) > ddof else np.nan
                    if sd is not None and np.isfinite(sd) and sd > 0.0:
                        cur_xs = [x[-1] for x in seg_xs]
                        terms = ([1.0] if add_intercept else []) + cur_xs
                        resid = float(seg_y[-1] - float(np.dot(terms, b)))
                        out[row, col] = resid / sd
            elif stat == "r2":
                ss_res = float(np.sum(e * e))
                ss_tot = float(np.sum((vy - np.mean(vy)) ** 2))
                if ss_tot > 0.0:
                    out[row, col] = float(max(0.0, 1.0 - ss_res / ss_tot))
    return frame_like(y, out)


def _register_multi(name: str, description: str, fit_fn: Callable[..., Any], extra: Any, stat: str, unit: str):
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
            unit=unit, cost=5,
        )

        def _calculate_series(self, y, x1=None, x2=None, x3=None, x4=None,
                              window=60, coefficient_index=1, min_periods=10, add_intercept=True, **_):
            feats = _gather_features((x1, x2, x3, x4), 4)
            return _multi_regression(
                y, feats, int(window), int(min_periods), bool(add_intercept),
                fit_fn, extra, stat, int(coefficient_index),
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
) -> pd.DataFrame:
    y, x = aligned(y, x)
    return _multi_regression(
        y, [x], int(window), int(min_periods), bool(add_intercept),
        fit_fn, extra, stat, int(coeff_index),
    )


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
    ]
)

import cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.EXTENDED_ONLY_CANONICALS = frozenset(
    set(_surface.EXTENDED_ONLY_CANONICALS) | set(_CANONICALS)
)
