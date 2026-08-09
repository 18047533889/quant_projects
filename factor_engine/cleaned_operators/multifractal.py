# -*- coding: utf-8 -*-
"""Multifractal spectrum operators (2026-08 geometry/math expansion).

The generalised Hurst exponent family characterises how the scaling of price
increments depends on the moment order ``q``.  For a trailing window ``X`` the
structure function at lag ``tau`` is the ``q``-th absolute moment of the
lag-``tau`` increments computed over *aligned finite pairs*,

    S_q(tau) = mean |X_{t+tau} - X_t|^q ,

and the scaling law ``log S_q(tau) = a + b * log tau`` is fitted by ordinary
least squares over the dyadic lags ``tau in {1, 2, 4, 8}``.  The generalised
Hurst exponent is ``H(q) = b / q``.

* ``ts_generalized_hurst_exponent``  — ``H(q)`` for a single moment order.
* ``ts_multifractal_width``          — ``H(1) - H(4)`` (spectrum width; large
  width = multifractal / heterogeneous scaling).
* ``ts_multifractal_curvature``      — the quadratic coefficient ``c`` of the
  fit ``H(q) = a + b*q + c*q^2`` over ``q in {1, 2, 3, 4}``.

All operators are trailing-window, prefix-causal and deterministic.  NaN
inputs are dropped from the window via aligned finite pairs; a window that
lacks enough aligned pairs for the largest lag emits NaN (never Inf, never a
fabricated zero).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_LAGS = (1, 2, 4, 8)
_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="multifractal",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "multifractal", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:scaling",
            f"unit:{unit}", f"cost:{cost}",
        ],
        # R4-95: the trailing ``window`` is a real horizon — it must be fully
        # accumulated before an estimate is emitted (R4-94).  Values inside the
        # window are used as aligned finite pairs, so at most ``window`` rows
        # are consumed (gaps reduce the pair count, never the row requirement).
        window_semantics="max_rows",
    )


def _check_window(window: int) -> int:
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    return w


_MIN_PAIRS_PER_LAG = 8
_MIN_LAGS_FOR_FIT = 3
_MIN_SCALING_R2 = 0.9


def _structure_function(vals: np.ndarray, lag: int, q: float) -> float:
    """``mean |X_{t+tau} - X_t|^q`` over aligned finite pairs in the window."""
    n = int(vals.shape[0])
    if n < lag + 2:
        return np.nan
    x = vals[: n - lag]
    y = vals[lag:]
    m = np.isfinite(x) & np.isfinite(y)
    # Audit P1-D: every lag needs a minimum number of valid aligned pairs —
    # two points would let a single outlier dominate the moment.
    if int(m.sum()) < _MIN_PAIRS_PER_LAG:
        return np.nan
    d = np.abs(x[m] - y[m])
    s = float(np.mean(d ** q))
    if not np.isfinite(s) or s <= 0.0:
        return np.nan
    return s


def _hurst_generalized(vals: np.ndarray, q: float) -> float:
    """Generalised Hurst exponent ``H(q)`` from the dyadic-lag OLS fit.

    Audit P1-D: the log-log scaling fit needs at least three valid lags and a
    minimum R² — a two-point line would produce a spuriously precise Hurst.
    """
    log_t: list[float] = []
    log_s: list[float] = []
    for lag in _LAGS:
        s = _structure_function(vals, lag, q)
        if not np.isfinite(s):
            continue
        log_t.append(np.log(float(lag)))
        log_s.append(np.log(s))
    if len(log_t) < _MIN_LAGS_FOR_FIT:
        return np.nan
    slope, intercept = np.polyfit(log_t, log_s, 1)
    if not np.isfinite(slope):
        return np.nan
    # R² of the log-log fit: fail closed when the scaling law is not clean.
    fitted = slope * np.asarray(log_t) + intercept
    ss_res = float(np.sum((np.asarray(log_s) - fitted) ** 2))
    ss_tot = float(np.sum((np.asarray(log_s) - np.mean(log_s)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    if not np.isfinite(r2) or r2 < _MIN_SCALING_R2:
        return np.nan
    return float(slope) / q


def _hurst_series(x2d: np.ndarray, window: int, q: float) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            # R4-94: ``window`` is a real horizon — the trailing window must be
            # fully accumulated before an H estimate is emitted.  Previously the
            # ``max(0, r - w + 1)`` truncation let a partial window start
            # estimating H as soon as the aligned-pair floor was met (e.g.
            # ~10 rows into a window=120 series), which is not the same scaling
            # law as the full window and produced unstable early values.
            if r + 1 < w:
                continue
            i0 = r - w + 1
            out[r, c] = _hurst_generalized(col[i0 : r + 1], q)
    return out


def _width_series(x2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            if r + 1 < w:
                continue
            i0 = r - w + 1
            chunk = col[i0 : r + 1]
            h1 = _hurst_generalized(chunk, 1.0)
            h4 = _hurst_generalized(chunk, 4.0)
            if np.isfinite(h1) and np.isfinite(h4):
                out[r, c] = h1 - h4
    return out


def _curvature_series(x2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    qs = np.array([1.0, 2.0, 3.0, 4.0])
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            if r + 1 < w:
                continue
            i0 = r - w + 1
            chunk = col[i0 : r + 1]
            hs = np.asarray([_hurst_generalized(chunk, q_) for q_ in qs], dtype=float)
            ok = np.isfinite(hs)
            if int(ok.sum()) < 3:
                continue
            coeffs = np.polyfit(qs[ok], hs[ok], 2)
            if np.isfinite(coeffs[0]):
                out[r, c] = float(coeffs[0])
    return out


@register_operator(
    name="ts_generalized_hurst_exponent",
    category="multifractal",
    business_category="multifractal",
    canonical="ts_generalized_hurst_exponent",
    source="multifractal",
)
class TsGeneralizedHurstExponent(SeriesOperator):
    """广义 Hurst 指数 H(q)：log S_q(tau) 对 log tau 的 OLS 斜率 / q。

    q=2 接近经典 Hurst 指数；q 大时强调大增量（间歇性/尾部），q 小时强调
    典型尺度。常数窗口 / 最大 lag 有效配对不足 -> NaN。P1。
    """

    metadata = _metadata(
        "ts_generalized_hurst_exponent",
        "广义 Hurst 指数 H(q)（dyadic-lag 结构函数 OLS 斜率 / q）。",
        ["x", "window", "q"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, q: float = 2.0, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        qv = float(q)
        if not (np.isfinite(qv) and qv > 0.0):
            raise ValueError("q must be a finite float > 0")
        return frame_like(x, _hurst_series(x.to_numpy(dtype=float), w, qv))


@register_operator(
    name="ts_multifractal_width",
    category="multifractal",
    business_category="multifractal",
    canonical="ts_multifractal_width",
    source="multifractal",
)
class TsMultifractalWidth(SeriesOperator):
    """多重分形谱宽 H(1)-H(4)。

    大 -> 不同 moment order 的标度行为差异大（多重分形 / 异质波动）；
    近 0 -> 单分形（Hurst 与 moment order 无关）。P1。
    """

    metadata = _metadata(
        "ts_multifractal_width",
        "多重分形谱宽 H(1)-H(4)。",
        ["x", "window"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        return frame_like(x, _width_series(x.to_numpy(dtype=float), w))


@register_operator(
    name="ts_multifractal_curvature",
    category="multifractal",
    business_category="multifractal",
    canonical="ts_multifractal_curvature",
    source="multifractal",
)
class TsMultifractalCurvature(SeriesOperator):
    """H(q) 对 q 的二次拟合曲率系数 c。

    H(q) = a + b*q + c*q^2 在 q∈{1,2,3,4} 拟合；c<0 表示谱的弧形下弯
    （强间歇性），c≈0 表示近似线性谱。P2。
    """

    metadata = _metadata(
        "ts_multifractal_curvature",
        "H(q) 对 q 二次拟合的曲率系数 c。",
        ["x", "window"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        return frame_like(x, _curvature_series(x.to_numpy(dtype=float), w))


_NEW_CANONICALS = (
    "ts_generalized_hurst_exponent",
    "ts_multifractal_width",
    "ts_multifractal_curvature",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
