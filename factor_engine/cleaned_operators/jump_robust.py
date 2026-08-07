# -*- coding: utf-8 -*-
"""Jump-robust intraday realized-variation operators (2026-08 geometry/math
expansion).

Minute-return panels in, rolling (trailing-window) jump-robust realized
variance estimates out.  MedRV (median realized variance) and MinRV (minimum
realized variance) are the two canonical jump-robust integrated-variance
estimators; the jump test statistic compares realized variance (RV) to bipower
variation (BV) scaled by realized quarticity (TQ) to produce a *signed* jump
z-statistic.

* ``intraday_medrv``       — MedRV, median-based jump-robust integrated variance.
* ``intraday_minrv``       — MinRV, minimum-based jump-robust integrated variance.
* ``intraday_jump_test_stat`` — ``(RV-BV)/sqrt((pi/2)^2 * TQ + eps)`` signed z.

All operators are trailing-window, prefix-causal and deterministic.  NaN
returns are dropped from the window; a window with fewer than 5 finite returns
emits NaN, and a degenerate window (too few observations for the estimator)
also emits NaN.  Values are never Inf and never fabricated zeros.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import check_window, frame_like, register_polars_bridge

_EPS = 1e-12
_MEDRV_CONST = np.pi / (6.0 - 4.0 * np.sqrt(2.0) + np.pi)  # ~0.9015
_MINRV_CONST = np.pi / (np.pi - 2.0)                        # ~2.7516
_PI_OVER_2_SQ = (np.pi / 2.0) ** 2
_MIN_FINITE = 5


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
            f"signature:{','.join(params)}->series", "domain:intraday_variation",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _rolling_returns(returns: np.ndarray, window: int, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    """Trailing-window per-column reduction over a minute-return panel."""
    rows, cols = returns.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = returns[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            v = col[i0 : r + 1]
            v = v[np.isfinite(v)]
            if v.size < _MIN_FINITE:
                continue
            out[r, c] = fn(v)
    return out


def _medrv(v: np.ndarray) -> float:
    n = int(v.size)
    if n < 3:
        return np.nan
    a = np.abs(v)
    med = np.empty(n - 2, dtype=float)
    for i in range(2, n):
        med[i - 2] = float(np.median(a[i - 2 : i + 1]))
    return float(_MEDRV_CONST * (n / (n - 2)) * float(np.sum(med * med)))


def _minrv(v: np.ndarray) -> float:
    n = int(v.size)
    if n < 2:
        return np.nan
    a = np.abs(v)
    return float(_MINRV_CONST * (n / (n - 1)) * float(np.sum(np.minimum(a[1:], a[:-1]) ** 2)))


def _jump_z(v: np.ndarray) -> float:
    n = int(v.size)
    if n < 2:
        return np.nan
    rv = float(np.sum(v * v))
    bv = float((np.pi / 2.0) * np.sum(np.abs(v[1:]) * np.abs(v[:-1])))
    tq = float((n / 3.0) * np.sum(v ** 4))
    return float((rv - bv) / np.sqrt(_PI_OVER_2_SQ * tq + _EPS))


@register_operator(
    name="intraday_medrv",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_medrv",
    source="jump_robust",
)
class IntradayMedRV(SeriesOperator):
    """日内 MedRV 已实现方差（中位数跳跃稳健估计）。

    ``C * N/(N-2) * sum_i med(|r_i|,|r_{i-1}|,|r_{i-2}|)^2``，``C=pi/(6-4sqrt2+pi)``。
    相比 RV 对单根分钟跳跃不敏感；缺失分钟先剔除，随后前 2 个观测无三元组 → 仅
    贡献窗口计数。单位 variance，窗口内 <5 个有限值 → NaN。P1。
    """

    metadata = _metadata(
        "intraday_medrv",
        "日内 MedRV 中位数跳跃稳健已实现方差。",
        ["returns", "window"],
        unit="variance",
        cost=3,
    )

    def _calculate_series(
        self, returns: pd.DataFrame, window: int = 240, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(returns, _rolling_returns(returns.to_numpy(dtype=float), w, _medrv))


@register_operator(
    name="intraday_minrv",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_minrv",
    source="jump_robust",
)
class IntradayMinRV(SeriesOperator):
    """日内 MinRV 已实现方差（最小值跳跃稳健估计）。

    ``C * N/(N-1) * sum_i min(|r_i|,|r_{i-1}|)^2``，``C=pi/(pi-2)``。
    对单根跳跃同样稳健，但估计量水平略低于 MedRV。单位 variance。P1。
    """

    metadata = _metadata(
        "intraday_minrv",
        "日内 MinRV 最小值跳跃稳健已实现方差。",
        ["returns", "window"],
        unit="variance",
        cost=3,
    )

    def _calculate_series(
        self, returns: pd.DataFrame, window: int = 240, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(returns, _rolling_returns(returns.to_numpy(dtype=float), w, _minrv))


@register_operator(
    name="intraday_jump_test_stat",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_jump_test_stat",
    source="jump_robust",
)
class IntradayJumpTestStat(SeriesOperator):
    """日内跳跃检验 z 统计量（有符号）。

    ``RV=sum r^2``、``BV=(pi/2)*sum |r_i||r_{i-1}|``、``TQ=N/3*sum r^4``；
    ``Z=(RV-BV)/sqrt(((pi/2)^2)*TQ+eps)``。正 = 存在向上跳跃驱动方差；负 = 连续
    路径主导（或极端负跳跃）。单位 zscore。P2。
    """

    metadata = _metadata(
        "intraday_jump_test_stat",
        "日内跳跃检验有符号 z 统计量。",
        ["returns", "window"],
        unit="zscore",
        cost=4,
    )

    def _calculate_series(
        self, returns: pd.DataFrame, window: int = 240, **_: Any
    ) -> pd.DataFrame:
        w = check_window(window)
        return frame_like(returns, _rolling_returns(returns.to_numpy(dtype=float), w, _jump_z))


_NEW_CANONICALS = (
    "intraday_medrv",
    "intraday_minrv",
    "intraday_jump_test_stat",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
