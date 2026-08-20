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
* ``intraday_jump_test_stat`` — standard Barndorff-Nielsen–Shephard (2006)
  linear jump-test z-statistic ``(RV-BV)/sqrt((θ-2)/3·Σr⁴)``, θ-2=(π/2)²+π-5.

All operators are trailing-window, prefix-causal and deterministic.  The
minute-slot axis is never compressed (review R4-20): a missing minute inside
the window makes the window invalid (fail-closed -> NaN) — the surrounding
returns are NEVER re-connected and treated as adjacent.  A leading NaN prefix
(session warm-up: the first minute has no prior price) is the only NaN allowed
and the estimator runs on the trailing contiguous finite run.  A window with
fewer than 5 minutes emits NaN, and a degenerate window (too few observations
for the estimator) also emits NaN.  Values are never Inf and never fabricated
zeros.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import check_window, frame_like, register_polars_bridge

_EPS = 1e-12
# Standard MedRV constant (Andersen–Dobrev–Schaumburg 2012): pi/(6-4√3+pi).
# The sqrt(2) variant was a transcription error and changed the estimator's
# level by ~35% (1.4195 vs 0.9016) — 3rd-round audit P0-01.
_MEDRV_CONST = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)  # ~1.4195
_MINRV_CONST = np.pi / (np.pi - 2.0)                        # ~2.7516
# Barndorff-Nielsen–Shephard (2006) linear jump-test variance factor:
# Var(sqrt(N)(RV-BV)) -> (θ-2)·IQ with θ = mu1^-4 + 2 mu1^-2 - 5,
# mu1 = E|Z| = sqrt(2/pi)  =>  θ-2 = (π/2)² + π - 5 ≈ 0.6090.  With the
# realized-quarticity estimator RQ = (N/3)·Σr⁴ → IQ, the z-statistic is
#   Z = sqrt(N)(RV-BV)/sqrt((θ-2)·RQ) = (RV-BV)/sqrt((θ-2)/3·Σr⁴).
# 3rd-round audit P0-02: the previous (π/2)²·TQ denominator was ~2√N too large
# (a ~30x deflation at N=240) and was not a standard BNS statistic.
_THETA_MINUS_2 = (np.pi / 2.0) ** 2 + np.pi - 5.0  # ~0.6090
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
    """Trailing-window per-column reduction over a minute-return panel.

    Session-slot semantics (review R4-20): the minute axis is never compressed.
    A NaN AFTER the first finite value is a genuine missing minute (10:01 gone
    between 10:00 and 10:02) — the estimator must NOT treat 10:00 and 10:02 as
    adjacent, so the whole window fails closed (NaN).  A leading NaN prefix
    (session warm-up, first minute has no prior price) is dropped to the
    trailing contiguous finite run.
    """
    rows, cols = returns.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = returns[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            v = col[i0 : r + 1]
            finite = np.isfinite(v)
            n_fin = int(finite.sum())
            if n_fin < _MIN_FINITE:
                continue
            first = int(np.flatnonzero(finite)[0])
            if np.any(~finite[first:]):
                continue  # interior missing minute -> window invalid
            out[r, c] = fn(v[first:])
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
    rq4 = float(np.sum(v ** 4))  # raw quarticity sum Σr⁴
    denom2 = max(_THETA_MINUS_2 / 3.0 * rq4, _EPS)
    return float((rv - bv) / np.sqrt(denom2))


@register_operator(
    name="intraday_medrv",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_medrv",
    source="jump_robust",
)
class IntradayMedRV(SeriesOperator):
    """日内 MedRV 已实现方差（中位数跳跃稳健估计）。

    ``C * N/(N-2) * sum_i med(|r_i|,|r_{i-1}|,|r_{i-2}|)^2``，``C=pi/(6-4sqrt3+pi)``。
    相比 RV 对单根分钟跳跃不敏感。分钟 slot 轴不压缩：窗口内缺失分钟 → 窗口
    invalid（NaN），绝不剔除后把相邻分钟重连；仅允许前导 NaN（开盘无前价）并
    在尾部连续有限段上计算。单位 variance，窗口内 <5 个有限值 → NaN。P1。
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
    """日内跳跃检验 BNS 线性 z 统计量（Barndorff-Nielsen–Shephard 2006）。

    ``RV=sum r^2``、``BV=(pi/2)*sum |r_i||r_{i-1}|``、
    ``Z=(RV-BV)/sqrt((theta-2)/3*sum r^4)``，``theta-2=(pi/2)^2+pi-5≈0.6090``。

    这是**幅度统计量**：正 = 跳跃主导波动（向上或向下跳跃都会使 RV-BV>0）；
    负 = 连续样本路径主导。它不携带方向信息——跳跃方向请使用有符号跳跃算子，
    不要把本统计量的正负解释为向上/向下跳跃。单位 zscore。P2。
    """

    metadata = _metadata(
        "intraday_jump_test_stat",
        "日内跳跃检验 BNS 线性 z 统计量（幅度，无方向）。",
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

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
