# -*- coding: utf-8 -*-
"""GTJA-191 所需的严格 pandas_numpy 算子实现。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.registry import OperatorRegistry


def _rolling_pairwise_ols(
    y: pd.DataFrame,
    x: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
    lag: int,
    retval: str,
) -> pd.DataFrame:
    w = max(2, int(window))
    mp = max(2, min(int(min_periods), w))
    k = int(lag)
    if k < 0:
        return pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
    if k:
        x = x.shift(k)
    y, x = y.align(x, join="inner", axis=0)
    y, x = y.align(x, join="inner", axis=1)
    valid = pd.DataFrame(
        np.isfinite(y.to_numpy(dtype=float, copy=False))
        & np.isfinite(x.to_numpy(dtype=float, copy=False)),
        index=y.index,
        columns=y.columns,
    )
    ym = y.where(valid)
    xm = x.where(valid)
    count = valid.rolling(window=w, min_periods=1).sum()
    cov = ym.rolling(window=w, min_periods=mp).cov(xm)
    var_x = xm.rolling(window=w, min_periods=mp).var(ddof=1)
    slope = (cov / var_x.where(var_x.abs() > 1e-15)).where(count >= mp)
    mode = str(retval or "slope").lower()
    if mode in {"slope", "beta", "0"}:
        return slope
    mean_y = ym.rolling(window=w, min_periods=mp).mean()
    mean_x = xm.rolling(window=w, min_periods=mp).mean()
    intercept = (mean_y - slope * mean_x).where(count >= mp)
    if mode in {"intercept", "alpha", "1"}:
        return intercept
    if mode in {"r_squared", "r2", "2"}:
        corr = ym.rolling(window=w, min_periods=mp).corr(xm)
        return (corr * corr).where(count >= mp)
    if mode in {"residual", "resid", "3"}:
        return (y - (slope * x + intercept)).where(valid & (count >= mp))
    if mode in {"fit", "fitted", "4"}:
        return (slope * x + intercept).where(valid & (count >= mp))
    raise ValueError(f"unsupported ts_regression retval: {retval!r}")


def _rolling_time_slope_1d(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    values = np.asarray(arr, dtype=float)
    out = np.full(values.size, np.nan, dtype=float)
    for i in range(values.size):
        start = max(0, i - window + 1)
        seg = values[start : i + 1]
        valid = np.isfinite(seg)
        if int(valid.sum()) < min_periods:
            continue
        t = np.arange(seg.size, dtype=float)[valid]
        v = seg[valid]
        tc = t - t.mean()
        denom = float(np.dot(tc, tc))
        if denom > 0.0:
            out[i] = float(np.dot(tc, v - v.mean()) / denom)
    return out


def _rolling_days_since_extreme_1d(arr: np.ndarray, window: int, *, maximum: bool) -> np.ndarray:
    values = np.asarray(arr, dtype=float)
    out = np.full(values.size, np.nan, dtype=float)
    for i in range(values.size):
        start = max(0, i - window + 1)
        seg = values[start : i + 1]
        valid = np.isfinite(seg)
        if not valid.any():
            continue
        extreme = np.nanmax(seg) if maximum else np.nanmin(seg)
        positions = np.flatnonzero(valid & (seg == extreme))
        out[i] = float(seg.size - 1 - positions[-1])
    return out


def _rolling_product(x: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    w = max(1, int(window))
    mp = max(1, min(int(min_periods), w))
    finite = pd.DataFrame(
        np.isfinite(x.to_numpy(dtype=float, copy=False)), index=x.index, columns=x.columns
    )
    clean = x.where(finite)
    count = finite.rolling(w, min_periods=1).sum()
    zero_count = clean.eq(0).rolling(w, min_periods=1).sum()
    negative_count = clean.lt(0).rolling(w, min_periods=1).sum()
    log_abs = np.log(clean.abs().where(clean.ne(0)))
    magnitude = np.exp(log_abs.rolling(w, min_periods=1).sum())
    sign = pd.DataFrame(
        np.where((negative_count % 2).eq(0), 1.0, -1.0),
        index=x.index,
        columns=x.columns,
    )
    result = (magnitude * sign).mask(zero_count.gt(0), 0.0)
    return result.where(count >= mp)


def _broadcast_pair(left, right) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将 Series/标量广播为同形 DataFrame，供元素级安全算子使用。"""
    template = left if isinstance(left, pd.DataFrame) else right if isinstance(right, pd.DataFrame) else None
    if template is None:
        raise TypeError("at least one protected-div input must be a DataFrame")

    def as_frame(value) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.reindex(index=template.index, columns=template.columns)
        if isinstance(value, pd.Series):
            return pd.DataFrame(
                np.broadcast_to(value.to_numpy()[:, None], template.shape),
                index=template.index,
                columns=template.columns,
            )
        return pd.DataFrame(float(value), index=template.index, columns=template.columns)

    return as_frame(left), as_frame(right)


def _safe_divide(left, right, *, epsilon: float, default: float, missing_default: bool) -> pd.DataFrame:
    x, y = _broadcast_pair(left, right)
    xarr = x.to_numpy(dtype=float, copy=False)
    yarr = y.to_numpy(dtype=float, copy=False)
    missing = ~np.isfinite(xarr) | ~np.isfinite(yarr)
    small = np.abs(yarr) <= float(epsilon)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = xarr / yarr
    raw[small] = float(default)
    raw[missing] = float(default) if missing_default else np.nan
    raw[~np.isfinite(raw) & ~missing] = float(default)
    return pd.DataFrame(raw, index=x.index, columns=x.columns)


@register_operator(name="ts_argmax", category="time_series", business_category="time_series", canonical="ts_argmax", source="gtja_compat", backend="pandas_numpy")
class GTJATSArgmax(SeriesOperator):
    metadata = OperatorMetadata(name="ts_argmax", category="time_series", description="窗口最大值距当前 bar 的距离（0=当前，tie 取最近）", examples=["ts_argmax(high, 20)"], param_names=["x", "window"], return_type="series", tags=["time_series", "gtja", "pit_safe"])

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        w = max(1, int(kwargs.get("d", window)))
        return x.apply(lambda s: _rolling_days_since_extreme_1d(s.to_numpy(), w, maximum=True))


@register_operator(name="ts_argmin", category="time_series", business_category="time_series", canonical="ts_argmin", source="gtja_compat", backend="pandas_numpy")
class GTJATSArgmin(SeriesOperator):
    metadata = OperatorMetadata(name="ts_argmin", category="time_series", description="窗口最小值距当前 bar 的距离（0=当前，tie 取最近）", examples=["ts_argmin(low, 20)"], param_names=["x", "window"], return_type="series", tags=["time_series", "gtja", "pit_safe"])

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        w = max(1, int(kwargs.get("d", window)))
        return x.apply(lambda s: _rolling_days_since_extreme_1d(s.to_numpy(), w, maximum=False))


@register_operator(name="ts_regression", category="time_series", business_category="time_series", canonical="ts_regression", source="gtja_compat", backend="pandas_numpy")
class GTJATSRegression(SeriesOperator):
    metadata = OperatorMetadata(name="ts_regression", category="time_series", description="pairwise-finite 滚动 OLS", examples=["ts_regression(y, x, 20, 0, 'slope')"], param_names=["y", "x", "window", "lag", "retval"], return_type="series", tags=["time_series", "regression", "gtja", "pit_safe"])

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, lag: int = 0, retval: str = "slope", **kwargs) -> pd.DataFrame:
        return _rolling_pairwise_ols(y, x, window=int(window), min_periods=int(kwargs.get("min_periods", min(3, int(window)))), lag=int(lag), retval=str(retval))


@register_operator(name="ts_time_slope", category="time_series", business_category="time_series", canonical="ts_time_slope", source="gtja_compat", backend="pandas_numpy")
class GTJATimeSlope(SeriesOperator):
    metadata = OperatorMetadata(name="ts_time_slope", category="time_series", description="窗口内序列对时间位置的 OLS 斜率", examples=["ts_time_slope(close, 20)"], param_names=["x", "window"], return_type="series", tags=["time_series", "regression", "gtja", "pit_safe"])

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        w = max(2, int(window))
        mp = max(2, min(int(kwargs.get("min_periods", w)), w))
        return x.apply(lambda s: _rolling_time_slope_1d(s.to_numpy(), w, mp))


@register_operator(name="ts_product", category="time_series", business_category="time_series", canonical="ts_product", source="gtja_compat", backend="pandas_numpy")
class GTJATSProduct(SeriesOperator):
    metadata = OperatorMetadata(name="ts_product", category="time_series", description="保留零与负号的滚动乘积", examples=["ts_product(x, 5)"], param_names=["x", "window"], return_type="series", tags=["time_series", "gtja", "pit_safe"])

    def _calculate_series(self, x: pd.DataFrame, window: int = 5, **kwargs) -> pd.DataFrame:
        w = max(1, int(window))
        mp = max(1, min(int(kwargs.get("min_periods", 1)), w))
        return _rolling_product(x, w, mp)


@register_operator(name="protected_div", category="data_cleaning", business_category="data_cleaning", canonical="protected_div", source="gtja_compat", backend="pandas_numpy")
class GTJAProtectedDiv(SeriesOperator):
    metadata = OperatorMetadata(name="protected_div", category="data_cleaning", description="支持 Series/标量双向广播的安全除法", examples=["x / 20", "1 / x"], param_names=["x", "y", "epsilon", "default"], return_type="series", tags=["data_cleaning", "gtja", "scalar_broadcast"])

    def _calculate_series(self, x, y, epsilon: float = 1e-8, default: float = 0.0, **kwargs) -> pd.DataFrame:
        return _safe_divide(x, y, epsilon=float(epsilon), default=float(default), missing_default=False)


@register_operator(name="div_or_default", category="data_cleaning", business_category="data_cleaning", canonical="div_or_default", source="gtja_compat", backend="pandas_numpy")
class GTJADivOrDefault(SeriesOperator):
    metadata = OperatorMetadata(name="div_or_default", category="data_cleaning", description="支持 Series/标量双向广播的填充安全除法", examples=["div_or_default(x, y)"], param_names=["x", "y", "epsilon", "default"], return_type="series", tags=["data_cleaning", "gtja", "scalar_broadcast"])

    def _calculate_series(self, x, y, epsilon: float = 1e-8, default: float = 0.0, **kwargs) -> pd.DataFrame:
        return _safe_divide(x, y, epsilon=float(epsilon), default=float(default), missing_default=True)


for _alias in ("Slope", "slope", "TS_TIME_SLOPE"):
    OperatorRegistry.register_alias(_alias, "ts_time_slope")
