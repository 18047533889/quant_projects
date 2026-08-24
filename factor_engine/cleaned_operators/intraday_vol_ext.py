# -*- coding: utf-8 -*-
"""Intraday realized-volatility *shape* operators (2026-08 geometry/math
expansion).

Beyond the level of realized variance these describe *where inside the trading
day* volatility is concentrated and how the realized-volatility estimator
responds to sampling frequency:

* ``intraday_volatility_time_centroid`` — early/late timing of the squared-return
  mass: ``TC = 2*sum(tau_m * r_m^2)/sum(r_m^2) - 1``, ``tau_m = m/(N-1)``.
* ``intraday_volatility_concentration`` — Herfindahl concentration of the
  normalized squared-return distribution ``sum p_i^2``, ``p = r^2/RV``.
* ``intraday_volatility_entropy`` — base-e Shannon entropy of the same
  distribution normalized by ``log N``.
* ``intraday_realized_semivariance_balance`` — upside vs downside realized
  semivariance balance ``(RSV+ - RSV-)/(RSV+ + RSV- + eps)``.
* ``intraday_rv_signature_curvature`` — curvature ``c`` of the volatility
  signature ``log RV(Delta) = a + b*log Delta + c*(log Delta)^2``.

All operators are trailing-window, prefix-causal and deterministic.
Session-slot aware (review R4-21): the real minute-slot axis is preserved —
missing slots keep their coordinate and their return contributes no RV mass
(shape operators), and the RV-signature operator fails closed on an interior
gap because block aggregation across a gap is meaningless.  A leading NaN
prefix (session warm-up) is dropped to the trailing contiguous finite run.
Windows with fewer than 5 finite returns emit NaN, and degenerate windows
(zero variance, insufficient scales) emit NaN.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import check_window, frame_like, register_polars_bridge

_EPS = 1e-12
_MIN_FINITE = 5
_RV_SCALES = (1, 2, 5, 10)


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday_microstructure", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:intraday_volatility_shape",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _rolling_returns(returns: np.ndarray, window: int, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    """Trailing-window per-column reduction over a minute-return panel.

    Session-slot aware (review R4-21): the raw window slice (missing minutes
    kept as NaN at their true slot positions) is passed to ``fn``.  Shape
    kernels mask missing slots (their return contributes no RV mass) while
    ``_rv_curvature`` fails closed on an interior gap.  Never drop-and-reconnect.
    """
    rows, cols = returns.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = returns[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            v = col[i0 : r + 1]
            if np.isfinite(v).sum() < _MIN_FINITE:
                continue
            out[r, c] = fn(v)
    return out


def _time_centroid(v: np.ndarray) -> float:
    n = int(v.size)
    if n < 2:
        return np.nan
    finite = np.isfinite(v)
    w = np.where(finite, v * v, 0.0)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= _EPS:
        return np.nan
    # Real minute-slot positions within the session window (0..n-1).  A missing
    # slot keeps its coordinate and contributes zero RV mass — the centroid is
    # NOT re-derived on a compressed/drop-reconnected axis.
    tau = np.arange(n, dtype=float) / (n - 1.0)
    return float(2.0 * float(np.sum(tau * w)) / total - 1.0)


def _concentration(v: np.ndarray) -> float:
    finite = np.isfinite(v)
    w = np.where(finite, v * v, 0.0)
    rv = float(np.sum(w))
    if not np.isfinite(rv) or rv <= _EPS:
        return np.nan
    p = w / rv
    return float(np.sum(p * p))


def _entropy(v: np.ndarray) -> float:
    finite = np.isfinite(v)
    n_finite = int(finite.sum())
    if n_finite < 2:
        return np.nan
    w = np.where(finite, v * v, 0.0)
    rv = float(np.sum(w))
    if not np.isfinite(rv) or rv <= _EPS:
        return np.nan
    p = w / rv
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = p * np.log(p)
    contrib[~np.isfinite(contrib)] = 0.0  # 0*log(0) -> 0
    return float(-float(np.sum(contrib)) / np.log(n_finite))


def _semi_balance(v: np.ndarray) -> float:
    finite = np.isfinite(v)
    pos = float(np.sum(v[finite & (v > 0.0)] ** 2))
    neg = float(np.sum(v[finite & (v < 0.0)] ** 2))
    return float((pos - neg) / (pos + neg + _EPS))


def _rv_curvature(v: np.ndarray) -> float:
    finite = np.isfinite(v)
    if not np.any(finite):
        return np.nan
    first = int(np.flatnonzero(finite)[0])
    # Block aggregation across an interior gap would span a real-time hole and
    # is meaningless (review R4-21): a NaN after the first finite value fails
    # the window.  A leading warm-up prefix is dropped to the contiguous suffix.
    if np.any(~finite[first:]):
        return np.nan
    v = v[first:]
    n = int(v.size)
    pts: list[tuple[float, float]] = []
    for s in _RV_SCALES:
        if n < s:
            continue
        nblocks = n // s
        if nblocks < 1:
            continue
        agg = v[: nblocks * s].reshape(nblocks, s).sum(axis=1)
        rv = float(np.sum(agg * agg))
        if np.isfinite(rv) and rv > _EPS:
            pts.append((float(np.log(s)), float(np.log(rv))))
    if len(pts) < 3:
        return np.nan
    xs = np.asarray([a for a, _ in pts], dtype=float)
    ys = np.asarray([b for _, b in pts], dtype=float)
    coeffs = np.polyfit(xs, ys, 2)
    return float(coeffs[0])


@register_operator(
    name="intraday_volatility_time_centroid",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_volatility_time_centroid",
    source="intraday_vol_ext",
)
class IntradayVolatilityTimeCentroid(SeriesOperator):
    """日内波动时间质心：日内 r^2 质量的重心早/晚。

    ``tau_m=m/(N-1)``，``TC=2*sum(tau*r^2)/sum(r^2)-1``，范围 [-1,1]：
    +1 = 波动集中在日内尾盘，-1 = 集中在开盘。单位 ratio。P1。
    """

    metadata = _metadata(
        "intraday_volatility_time_centroid",
        "日内已实现方差的时间质心（早/晚偏置）。",
        ["returns", "window"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self, returns: pd.DataFrame, window: int = 240, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(returns, _rolling_returns(returns.to_numpy(dtype=float), w, _time_centroid))


@register_operator(
    name="intraday_volatility_concentration",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_volatility_concentration",
    source="intraday_vol_ext",
)
class IntradayVolatilityConcentration(SeriesOperator):
    """日内波动集中度：归一化 r^2 分布的 Herfindahl 指数。

    ``p=r^2/RV``，``HHI=sum p_i^2``，范围 (0,1]；1 = 波动全部来自单根分钟，
    接近 1/N = 完全均匀。单位 ratio。P1。
    """

    metadata = _metadata(
        "intraday_volatility_concentration",
        "日内已实现方差集中度（HHI）。",
        ["returns", "window"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self, returns: pd.DataFrame, window: int = 240, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(returns, _rolling_returns(returns.to_numpy(dtype=float), w, _concentration))


@register_operator(
    name="intraday_volatility_entropy",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_volatility_entropy",
    source="intraday_vol_ext",
)
class IntradayVolatilityEntropy(SeriesOperator):
    """日内波动熵：归一化 r^2 分布的 Shannon 熵。

    ``H = -sum p log p / log N``（base-e，N 为有限收益数），范围 [0,1]；
    低 = 方差集中在少数分钟，高 = 均匀铺满全天。单位 entropy。P1。
    """

    metadata = _metadata(
        "intraday_volatility_entropy",
        "日内已实现方差分布归一化熵。",
        ["returns", "window"],
        unit="entropy",
        cost=2,
    )

    def _calculate_series(
        self, returns: pd.DataFrame, window: int = 240, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(returns, _rolling_returns(returns.to_numpy(dtype=float), w, _entropy))


@register_operator(
    name="intraday_realized_semivariance_balance",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_realized_semivariance_balance",
    source="intraday_vol_ext",
)
class IntradayRealizedSemivarianceBalance(SeriesOperator):
    """日内已实现半方差平衡：上行/下行方差失衡。

    ``RSV+=sum r^2*1[r>0]``、``RSV-=sum r^2*1[r<0]``；
    ``balance=(RSV+-RSV-)/(RSV++RSV-+eps)``，范围约 [-1,1]。单位 ratio。P1。
    """

    metadata = _metadata(
        "intraday_realized_semivariance_balance",
        "上行 vs 下行已实现半方差失衡。",
        ["returns", "window"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self, returns: pd.DataFrame, window: int = 240, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(returns, _rolling_returns(returns.to_numpy(dtype=float), w, _semi_balance))


@register_operator(
    name="intraday_rv_signature_curvature",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_rv_signature_curvature",
    source="intraday_vol_ext",
)
class IntradayRvSignatureCurvature(SeriesOperator):
    """日内 volatility signature 曲率（log RV vs log Δ 二次项系数）。

    ``RV(Δ)=Σ (Δ 分钟块收益和)^2``（非重叠块，Δ∈{1,2,5,10}），拟合
    ``log RV = a + b·log Δ + c·(log Δ)^2``。负曲率 = 低频 RV 相对高频衰减变缓
    （噪声主导在短间隔），正曲率 = 长间隔 RV 相对加速上升。不足 3 个有效采样点
    → NaN。单位 ratio。P2。
    """

    metadata = _metadata(
        "intraday_rv_signature_curvature",
        "日内 RV signature 二次曲率。",
        ["returns", "window"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self, returns: pd.DataFrame, window: int = 240, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(returns, _rolling_returns(returns.to_numpy(dtype=float), w, _rv_curvature))


_NEW_CANONICALS = (
    "intraday_volatility_time_centroid",
    "intraday_volatility_concentration",
    "intraday_volatility_entropy",
    "intraday_realized_semivariance_balance",
    "intraday_rv_signature_curvature",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
