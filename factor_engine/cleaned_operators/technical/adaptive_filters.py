# -*- coding: utf-8 -*-
"""Adaptive filtering operators (2026-08).

Five TRUE_GAP adaptive filters following R47 conventions:
  1. ts_mcginley_dynamic    -- McGinley Dynamic adaptive MA
  2. ts_vidya               -- Variable Index Dynamic Average
  3. ts_one_euro_filter     -- 1€ Filter with adaptive cutoff frequency
  4. ts_nlms_filter         -- Normalized Least Mean Squares adaptive filter
  5. ts_rls_filter          -- Recursive Least Squares adaptive filter

All operators are causal, trailing-only, scope="ts", pit_safe=True.
Dual backend: pandas_numpy + polars.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.base_polars import (
    OperatorMetadata as PolarsMeta,
    SeriesOperator as PolarsSeriesOp,
    register_operator as register_polars,
)

_EPS = 1e-12
_SKIP = frozenset({"date", "stock_code"})


# ---------------------------------------------------------------------------
# shared helpers (pandas)
# ---------------------------------------------------------------------------
def _meta(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    param_specs: dict[str, ParamSpec] | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="technical_signal",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "technical_signal", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:technical_signal",
            f"unit:{unit}", "cost:1", "adaptive_filter",
        ],
        param_specs=param_specs or {},
    )


def _pi(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _pf(value, name: str, minimum: float | None = None, maximum: float | None = None) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


# ---------------------------------------------------------------------------
# polars helpers
# ---------------------------------------------------------------------------
def _cols(*frames: pl.DataFrame) -> list[str]:
    out = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        out = [c for c in out if c in frame.columns]
    return out


def _result(base: pl.DataFrame, values: dict[str, pl.Series]) -> pl.DataFrame:
    return base.with_columns([series.alias(name) for name, series in values.items()])


def _one(frame: pl.DataFrame, column: str, expr: pl.Expr) -> pl.Series:
    return frame.select(expr.alias(column)).to_series()


# ---------------------------------------------------------------------------
# 1. ts_mcginley_dynamic
# ---------------------------------------------------------------------------
def _mcginley_dynamic_series(x: pd.Series, window: int, power: float) -> pd.Series:
    """McGinley Dynamic: MD_t = MD_{t-1} + (x_t - MD_{t-1}) / (c * (x_t/MD_{t-1})^power).

    Starts from SMA(window) then adapts the smoothing factor based on price momentum.
    """
    n = len(x)
    out = np.full(n, np.nan)
    w = max(1, int(window))
    pw = float(power)

    # warmup: SMA
    for i in range(w - 1, n):
        seg = x.iloc[i - w + 1 : i + 1]
        if seg.isna().any():
            continue
        out[i] = seg.mean()
        break

    if not np.isfinite(out).any():
        return pd.Series(out, index=x.index)

    start = int(np.where(np.isfinite(out))[0][0])
    for t in range(start + 1, n):
        curr = x.iloc[t]
        prev_md = out[t - 1]
        if not np.isfinite(curr) or not np.isfinite(prev_md):
            continue
        if abs(prev_md) < _EPS:
            out[t] = prev_md
            continue
        ratio = abs(curr / prev_md)
        if ratio < _EPS:
            out[t] = prev_md
            continue
        denom = w * (ratio ** pw)
        if denom < _EPS:
            out[t] = prev_md
            continue
        out[t] = prev_md + (curr - prev_md) / denom

    return pd.Series(out, index=x.index)


@register_operator(
    name="ts_mcginley_dynamic",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_mcginley_dynamic",
    source="technical.adaptive_filters",
    backend="pandas_numpy",
)
class McGinleyDynamic(SeriesOperator):
    """McGinley Dynamic adaptive moving average.

    Adapts smoothing based on price momentum: faster when price accelerates,
    slower in stable regimes. Starts from SMA(window), then applies adaptive update.
    """

    metadata = _meta(
        "ts_mcginley_dynamic",
        "McGinley Dynamic 自适应均线：根据价格动量调整平滑系数。",
        ["x", "window", "power"],
        unit="price",
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "power": ParamSpec(dtype=float, min=0.1, max=10.0),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 20, power: float = 4.0, **_: Any
    ) -> pd.DataFrame:
        w = _pi(window, "window", 2)
        pw = _pf(power, "power", 0.1, 10.0)
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            out[col] = _mcginley_dynamic_series(x[col], w, pw)
        return out


# ---------------------------------------------------------------------------
# 2. ts_vidya
# ---------------------------------------------------------------------------
def _vidya_series(x: pd.Series, window: int, smooth: int) -> pd.Series:
    """Variable Index Dynamic Average: EMA with alpha modulated by CMO volatility."""
    n = len(x)
    out = np.full(n, np.nan)
    w = max(2, int(window))
    sm = max(1, int(smooth))

    # CMO (Chande Momentum Oscillator) for volatility
    delta = x.diff()
    up = delta.clip(lower=0).rolling(window=w, min_periods=w).sum()
    down = (-delta.clip(upper=0)).rolling(window=w, min_periods=w).sum()
    total = up + down
    cmo = ((up - down) / total.replace(0, np.nan)).abs()

    # base alpha
    alpha_base = 2.0 / (sm + 1.0)

    # first valid: simple mean
    for i in range(w - 1, n):
        seg = x.iloc[i - w + 1 : i + 1]
        if seg.isna().any():
            continue
        out[i] = seg.mean()
        break

    if not np.isfinite(out).any():
        return pd.Series(out, index=x.index)

    start = int(np.where(np.isfinite(out))[0][0])
    for t in range(start + 1, n):
        curr = x.iloc[t]
        prev = out[t - 1]
        c = cmo.iloc[t]
        if not np.isfinite(curr) or not np.isfinite(prev):
            continue
        if not np.isfinite(c):
            # fallback to base alpha
            alpha = alpha_base
        else:
            alpha = alpha_base * c
        out[t] = alpha * curr + (1.0 - alpha) * prev

    return pd.Series(out, index=x.index)


@register_operator(
    name="ts_vidya",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_vidya",
    source="technical.adaptive_filters",
    backend="pandas_numpy",
)
class VIDYA(SeriesOperator):
    """Variable Index Dynamic Average.

    EMA with adaptive alpha modulated by CMO (Chande Momentum Oscillator) volatility.
    Higher volatility increases alpha for faster adaptation.
    """

    metadata = _meta(
        "ts_vidya",
        "可变指数动态平均：EMA alpha 由 CMO 波动率调节。",
        ["x", "window", "smooth"],
        unit="price",
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "smooth": ParamSpec(dtype=int, min=1),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 20, smooth: int = 9, **_: Any
    ) -> pd.DataFrame:
        w = _pi(window, "window", 2)
        sm = _pi(smooth, "smooth", 1)
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            out[col] = _vidya_series(x[col], w, sm)
        return out


# ---------------------------------------------------------------------------
# 3. ts_one_euro_filter
# ---------------------------------------------------------------------------
def _one_euro_series(x: pd.Series, min_cutoff: float, beta: float) -> pd.Series:
    """1€ Filter: exponential smoothing with adaptive cutoff frequency.

    Cutoff frequency increases with velocity for lag reduction.
    """
    n = len(x)
    out = np.full(n, np.nan)
    mc = float(min_cutoff)
    b = float(beta)

    # find first valid
    first_valid = None
    for i in range(n):
        if np.isfinite(x.iloc[i]):
            out[i] = x.iloc[i]
            first_valid = i
            break

    if first_valid is None:
        return pd.Series(out, index=x.index)

    prev_filtered = out[first_valid]
    prev_dx = 0.0

    for t in range(first_valid + 1, n):
        curr = x.iloc[t]
        if not np.isfinite(curr):
            continue

        # velocity estimate (exponentially smoothed derivative)
        dx = curr - prev_filtered
        alpha_d = 1.0 / (1.0 + 1.0 / mc)  # derivative smoothing
        dx_filtered = alpha_d * dx + (1.0 - alpha_d) * prev_dx

        # adaptive cutoff
        cutoff = mc + b * abs(dx_filtered)
        # alpha for main signal
        alpha = 1.0 / (1.0 + 1.0 / cutoff)

        # filter
        filtered = alpha * curr + (1.0 - alpha) * prev_filtered
        out[t] = filtered
        prev_filtered = filtered
        prev_dx = dx_filtered

    return pd.Series(out, index=x.index)


@register_operator(
    name="ts_one_euro_filter",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_one_euro_filter",
    source="technical.adaptive_filters",
    backend="pandas_numpy",
)
class OneEuroFilter(SeriesOperator):
    """1€ Filter with adaptive cutoff frequency.

    Exponential filter that increases cutoff (reduces lag) when velocity is high.
    Originally designed for noisy interactive systems.
    """

    metadata = _meta(
        "ts_one_euro_filter",
        "1€ 滤波器：根据速度自适应调整截止频率。",
        ["x", "min_cutoff", "beta"],
        unit="price",
        param_specs={
            "min_cutoff": ParamSpec(dtype=float, min=0.001, max=10.0),
            "beta": ParamSpec(dtype=float, min=0.0, max=10.0),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, min_cutoff: float = 1.0, beta: float = 0.007, **_: Any
    ) -> pd.DataFrame:
        mc = _pf(min_cutoff, "min_cutoff", 0.001, 10.0)
        b = _pf(beta, "beta", 0.0, 10.0)
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            out[col] = _one_euro_series(x[col], mc, b)
        return out


# ---------------------------------------------------------------------------
# 4. ts_nlms_filter
# ---------------------------------------------------------------------------
def _nlms_series(x: pd.Series, order: int, mu: float, eps: float) -> pd.Series:
    """Normalized LMS adaptive filter."""
    n = len(x)
    out = np.full(n, np.nan)
    p = max(1, int(order))

    # initialize weights
    w = np.zeros(p, dtype=float)

    for t in range(p, n):
        # input buffer
        buf = np.array([x.iloc[t - i] for i in range(p)], dtype=float)
        if not np.all(np.isfinite(buf)):
            continue

        # prediction
        y_pred = np.dot(w, buf)

        # error
        curr = x.iloc[t]
        if not np.isfinite(curr):
            continue
        err = curr - y_pred

        # normalized update
        norm = np.dot(buf, buf) + eps
        w += (mu / norm) * err * buf

        out[t] = y_pred

    return pd.Series(out, index=x.index)


@register_operator(
    name="ts_nlms_filter",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_nlms_filter",
    source="technical.adaptive_filters",
    backend="pandas_numpy",
)
class NLMSFilter(SeriesOperator):
    """Normalized Least Mean Squares adaptive filter.

    Linear predictor with weights updated via normalized gradient descent.
    Predicts current value from trailing buffer of length `order`.
    """

    metadata = _meta(
        "ts_nlms_filter",
        "归一化最小均方自适应滤波器：权重归一化梯度更新。",
        ["x", "order", "mu", "eps"],
        unit="price",
        param_specs={
            "order": ParamSpec(dtype=int, min=1, max=100),
            "mu": ParamSpec(dtype=float, min=0.0, max=2.0),
            "eps": ParamSpec(dtype=float, min=1e-12, max=1.0),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, order: int = 5, mu: float = 0.1, eps: float = 1e-6, **_: Any
    ) -> pd.DataFrame:
        p = _pi(order, "order", 1)
        if p > 100:
            raise ValueError("order must be <= 100")
        m = _pf(mu, "mu", 0.0, 2.0)
        e = _pf(eps, "eps", 1e-12, 1.0)
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            out[col] = _nlms_series(x[col], p, m, e)
        return out


# ---------------------------------------------------------------------------
# 5. ts_rls_filter
# ---------------------------------------------------------------------------
def _rls_series(x: pd.Series, order: int, lam: float, delta: float) -> pd.Series:
    """Recursive Least Squares adaptive filter."""
    n = len(x)
    out = np.full(n, np.nan)
    p = max(1, int(order))

    # initialize
    w = np.zeros(p, dtype=float)
    P = np.eye(p, dtype=float) / delta  # inverse correlation matrix

    for t in range(p, n):
        # input buffer
        buf = np.array([x.iloc[t - i] for i in range(p)], dtype=float)
        if not np.all(np.isfinite(buf)):
            continue

        # prediction
        y_pred = np.dot(w, buf)

        # desired
        curr = x.iloc[t]
        if not np.isfinite(curr):
            continue

        # error
        err = curr - y_pred

        # gain vector
        P_buf = P @ buf
        denom = lam + buf @ P_buf
        if abs(denom) < _EPS:
            out[t] = y_pred
            continue
        k = P_buf / denom

        # update weights
        w += k * err

        # update inverse correlation
        P = (P - np.outer(k, P_buf)) / lam

        out[t] = y_pred

    return pd.Series(out, index=x.index)


@register_operator(
    name="ts_rls_filter",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_rls_filter",
    source="technical.adaptive_filters",
    backend="pandas_numpy",
)
class RLSFilter(SeriesOperator):
    """Recursive Least Squares adaptive filter.

    Exponentially weighted least squares with forgetting factor `lambda`.
    Faster convergence than LMS but higher computational cost.
    """

    metadata = _meta(
        "ts_rls_filter",
        "递归最小二乘自适应滤波器：指数加权最小二乘。",
        ["x", "order", "lambda_", "delta"],
        unit="price",
        param_specs={
            "order": ParamSpec(dtype=int, min=1, max=50),
            "lambda_": ParamSpec(dtype=float, min=0.9, max=1.0),
            "delta": ParamSpec(dtype=float, min=0.01, max=100.0),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, order: int = 5, lambda_: float = 0.99, delta: float = 1.0, **_: Any
    ) -> pd.DataFrame:
        p = _pi(order, "order", 1)
        if p > 50:
            raise ValueError("order must be <= 50")
        lam = _pf(lambda_, "lambda_", 0.9, 1.0)
        d = _pf(delta, "delta", 0.01, 100.0)
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            out[col] = _rls_series(x[col], p, lam, d)
        return out


# ---------------------------------------------------------------------------
# Polars backend implementations
# ---------------------------------------------------------------------------


def _mcginley_polars(x: pl.DataFrame, window: int, power: float):
    w = _pi(window, "window", 2)
    pw = _pf(power, "power", 0.1, 10.0)
    values = {}
    for c in _cols(x):
        arr = x[c].to_numpy()
        n = len(arr)
        out = np.full(n, np.nan)

        # warmup: SMA
        for i in range(w - 1, n):
            seg = arr[i - w + 1 : i + 1]
            if np.all(np.isfinite(seg)):
                out[i] = np.mean(seg)
                break

        if not np.isfinite(out).any():
            values[c] = pl.Series(c, out)
            continue

        start = int(np.where(np.isfinite(out))[0][0])
        for t in range(start + 1, n):
            curr = arr[t]
            prev_md = out[t - 1]
            if not np.isfinite(curr) or not np.isfinite(prev_md):
                continue
            if abs(prev_md) < _EPS:
                out[t] = prev_md
                continue
            ratio = abs(curr / prev_md)
            if ratio < _EPS:
                out[t] = prev_md
                continue
            denom = w * (ratio ** pw)
            if denom < _EPS:
                out[t] = prev_md
                continue
            out[t] = prev_md + (curr - prev_md) / denom

        values[c] = pl.Series(c, out)
    return _result(x, values)


def _vidya_polars(x: pl.DataFrame, window: int, smooth: int):
    w = _pi(window, "window", 2)
    sm = _pi(smooth, "smooth", 1)
    alpha_base = 2.0 / (sm + 1.0)
    values = {}

    for c in _cols(x):
        arr = x[c].to_numpy()
        n = len(arr)
        out = np.full(n, np.nan)

        # CMO
        delta = np.diff(arr, prepend=np.nan)
        up = np.clip(delta, 0, None)
        down = -np.clip(delta, None, 0)

        cmo = np.full(n, np.nan)
        for i in range(w - 1, n):
            up_sum = np.nansum(up[i - w + 1 : i + 1])
            down_sum = np.nansum(down[i - w + 1 : i + 1])
            total = up_sum + down_sum
            if total > _EPS:
                cmo[i] = abs((up_sum - down_sum) / total)

        # first valid
        for i in range(w - 1, n):
            seg = arr[i - w + 1 : i + 1]
            if np.all(np.isfinite(seg)):
                out[i] = np.mean(seg)
                break

        if not np.isfinite(out).any():
            values[c] = pl.Series(c, out)
            continue

        start = int(np.where(np.isfinite(out))[0][0])
        for t in range(start + 1, n):
            curr = arr[t]
            prev = out[t - 1]
            c_val = cmo[t]
            if not np.isfinite(curr) or not np.isfinite(prev):
                continue
            if np.isfinite(c_val):
                alpha = alpha_base * c_val
            else:
                alpha = alpha_base
            out[t] = alpha * curr + (1.0 - alpha) * prev

        values[c] = pl.Series(c, out)
    return _result(x, values)


def _one_euro_polars(x: pl.DataFrame, min_cutoff: float, beta: float):
    mc = _pf(min_cutoff, "min_cutoff", 0.001, 10.0)
    b = _pf(beta, "beta", 0.0, 10.0)
    values = {}

    for c in _cols(x):
        arr = x[c].to_numpy()
        n = len(arr)
        out = np.full(n, np.nan)

        first_valid = None
        for i in range(n):
            if np.isfinite(arr[i]):
                out[i] = arr[i]
                first_valid = i
                break

        if first_valid is None:
            values[c] = pl.Series(c, out)
            continue

        prev_filtered = out[first_valid]
        prev_dx = 0.0

        for t in range(first_valid + 1, n):
            curr = arr[t]
            if not np.isfinite(curr):
                continue

            dx = curr - prev_filtered
            alpha_d = 1.0 / (1.0 + 1.0 / mc)
            dx_filtered = alpha_d * dx + (1.0 - alpha_d) * prev_dx

            cutoff = mc + b * abs(dx_filtered)
            alpha = 1.0 / (1.0 + 1.0 / cutoff)

            filtered = alpha * curr + (1.0 - alpha) * prev_filtered
            out[t] = filtered
            prev_filtered = filtered
            prev_dx = dx_filtered

        values[c] = pl.Series(c, out)
    return _result(x, values)


def _nlms_polars(x: pl.DataFrame, order: int, mu: float, eps: float):
    p = _pi(order, "order", 1)
    if p > 100:
        raise ValueError("order must be <= 100")
    m = _pf(mu, "mu", 0.0, 2.0)
    e = _pf(eps, "eps", 1e-12, 1.0)
    values = {}

    for c in _cols(x):
        arr = x[c].to_numpy()
        n = len(arr)
        out = np.full(n, np.nan)
        w = np.zeros(p, dtype=float)

        for t in range(p, n):
            buf = np.array([arr[t - i] for i in range(p)], dtype=float)
            if not np.all(np.isfinite(buf)):
                continue

            y_pred = np.dot(w, buf)
            curr = arr[t]
            if not np.isfinite(curr):
                continue
            err = curr - y_pred

            norm = np.dot(buf, buf) + e
            w += (m / norm) * err * buf
            out[t] = y_pred

        values[c] = pl.Series(c, out)
    return _result(x, values)


def _rls_polars(x: pl.DataFrame, order: int, lam: float, delta: float):
    p = _pi(order, "order", 1)
    if p > 50:
        raise ValueError("order must be <= 50")
    l = _pf(lam, "lambda_", 0.9, 1.0)
    d = _pf(delta, "delta", 0.01, 100.0)
    values = {}

    for c in _cols(x):
        arr = x[c].to_numpy()
        n = len(arr)
        out = np.full(n, np.nan)
        w = np.zeros(p, dtype=float)
        P = np.eye(p, dtype=float) / d

        for t in range(p, n):
            buf = np.array([arr[t - i] for i in range(p)], dtype=float)
            if not np.all(np.isfinite(buf)):
                continue

            y_pred = np.dot(w, buf)
            curr = arr[t]
            if not np.isfinite(curr):
                continue

            err = curr - y_pred
            P_buf = P @ buf
            denom = l + buf @ P_buf
            if abs(denom) < _EPS:
                out[t] = y_pred
                continue
            k = P_buf / denom
            w += k * err
            P = (P - np.outer(k, P_buf)) / l
            out[t] = y_pred

        values[c] = pl.Series(c, out)
    return _result(x, values)


# ---------------------------------------------------------------------------
# Polars operator registration
# ---------------------------------------------------------------------------


@register_polars(
    name="ts_mcginley_dynamic",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_mcginley_dynamic",
    source="technical.adaptive_filters",
    backend="polars",
    status="production",
)
class McGinleyDynamicPolars(PolarsSeriesOp):
    metadata = PolarsMeta(
        name="ts_mcginley_dynamic",
        category="technical_signal",
        description="McGinley Dynamic adaptive moving average (Polars).",
        param_names=["x", "window", "power"],
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, x, window=20, power=4.0, **_):
        return _mcginley_polars(x, window, power)


@register_polars(
    name="ts_vidya",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_vidya",
    source="technical.adaptive_filters",
    backend="polars",
    status="production",
)
class VIDYAPolars(PolarsSeriesOp):
    metadata = PolarsMeta(
        name="ts_vidya",
        category="technical_signal",
        description="Variable Index Dynamic Average (Polars).",
        param_names=["x", "window", "smooth"],
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, x, window=20, smooth=9, **_):
        return _vidya_polars(x, window, smooth)


@register_polars(
    name="ts_one_euro_filter",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_one_euro_filter",
    source="technical.adaptive_filters",
    backend="polars",
    status="production",
)
class OneEuroFilterPolars(PolarsSeriesOp):
    metadata = PolarsMeta(
        name="ts_one_euro_filter",
        category="technical_signal",
        description="1€ Filter with adaptive cutoff (Polars).",
        param_names=["x", "min_cutoff", "beta"],
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, x, min_cutoff=1.0, beta=0.007, **_):
        return _one_euro_polars(x, min_cutoff, beta)


@register_polars(
    name="ts_nlms_filter",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_nlms_filter",
    source="technical.adaptive_filters",
    backend="polars",
    status="production",
)
class NLMSFilterPolars(PolarsSeriesOp):
    metadata = PolarsMeta(
        name="ts_nlms_filter",
        category="technical_signal",
        description="Normalized LMS adaptive filter (Polars).",
        param_names=["x", "order", "mu", "eps"],
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, x, order=5, mu=0.1, eps=1e-6, **_):
        return _nlms_polars(x, order, mu, eps)


@register_polars(
    name="ts_rls_filter",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_rls_filter",
    source="technical.adaptive_filters",
    backend="polars",
    status="production",
)
class RLSFilterPolars(PolarsSeriesOp):
    metadata = PolarsMeta(
        name="ts_rls_filter",
        category="technical_signal",
        description="Recursive Least Squares adaptive filter (Polars).",
        param_names=["x", "order", "lambda_", "delta"],
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, x, order=5, lambda_=0.99, delta=1.0, **_):
        return _rls_polars(x, order, lambda_, delta)


# ---------------------------------------------------------------------------
# Register to EXTENDED_ONLY_CANONICALS
# ---------------------------------------------------------------------------
_CANONICALS = [
    "ts_mcginley_dynamic",
    "ts_vidya",
    "ts_one_euro_filter",
    "ts_nlms_filter",
    "ts_rls_filter",
]

try:
    from factor_engine.cleaned_operators import operator_surface as _surface
    _surface.extend_extended_only(set(_CANONICALS))
except ImportError:
    pass
