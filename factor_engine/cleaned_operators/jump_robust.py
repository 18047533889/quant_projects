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

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator, strict_int_param
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_bridge

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
        output_unit=unit,
        panel_params=("returns",),
        panel_arity=1,
        scalar_params=("window",),
        param_specs={
            "window": ParamSpec(
                dtype=int, min=_MIN_FINITE, default=240,
                history_semantics="max_rows", param_role=ParamRole.HORIZON,
            ),
        },
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
    w = strict_int_param(window, "window", lower=_MIN_FINITE)
    for c in range(cols):
        col = returns[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            v = col[i0 : r + 1]
            if np.isinf(v).any():
                continue
            finite = ~np.isnan(v)
            n_fin = int(finite.sum())
            if n_fin < _MIN_FINITE:
                continue
            first = int(np.flatnonzero(finite)[0])
            if np.any(~finite[first:]):
                continue  # interior missing minute -> window invalid
            value = fn(v[first:])
            if np.isfinite(value):
                out[r, c] = value
    return out


def _rolling_medrv_vec(returns: np.ndarray, window: int) -> np.ndarray:
    """Vectorized trailing-window MedRV — matrix form of
    ``_rolling_returns(returns, window, _medrv)`` with identical semantics.

    Per (row, column): any inf anywhere in the trailing window [max(0,r-w+1), r]
    fails closed (NaN, including the leading segment); the non-finite slots of a
    valid window must form a leading prefix (session warm-up) — the estimator
    runs on [first finite slot in window, r] and any interior NaN fails closed;
    fewer than 5 finite values -> NaN; zero window magnitude -> NaN.  Triple
    medians are computed on raw |x| (median commutes with division by the
    positive window magnitude), accumulated via an inclusive longdouble prefix
    sum — all terms non-negative, so the window difference has no catastrophic
    cancellation and 1e-300-scale squares do not underflow to zero.  The
    degree-two statistic is restored exactly like ``_variance_from_normalized``
    (longdouble, NaN on overflow or underflow-to-zero), elementwise.
    ``window`` must already be validated by ``strict_int_param``.
    """
    rows, cols = returns.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    if rows < _MIN_FINITE:
        return out
    X = np.asarray(returns, dtype=float)
    ld = np.longdouble
    row_idx = np.arange(rows)[:, None]
    rc_idx = np.broadcast_to(row_idx, (rows, cols))
    i0 = np.maximum(row_idx - (window - 1), 0)  # (rows, 1) window start
    isfin = np.isfinite(X)
    # first finite slot at or after the window start, per column (rows = none)
    next_fin = np.minimum.accumulate(np.where(isfin, rc_idx, rows)[::-1], axis=0)[::-1]
    seg_start = next_fin[i0[:, 0]]  # (rows, cols)
    n_seg = rc_idx - seg_start + 1
    # last non-finite slot <= r per column (-1 if none); a valid window has it
    # strictly before the segment start (non-finites form a leading prefix)
    last_bad = np.maximum.accumulate(np.where(isfin, -1, rc_idx), axis=0)
    # any inf inside the trailing window [i0, r] fails closed (leading too)
    last_inf = np.maximum.accumulate(np.where(np.isinf(X), rc_idx, -1), axis=0)
    # window magnitude: max |x| over finite slots in [i0, r]; slots before the
    # run start are non-finite -> -inf, so this equals the segment max
    mag_m = np.where(isfin, np.abs(X), -np.inf)
    cum_max = np.maximum.accumulate(mag_m, axis=0)
    if rows >= window:
        sw_max = np.lib.stride_tricks.sliding_window_view(mag_m, window, axis=0)
        mag = np.vstack([cum_max[: window - 1], sw_max.max(axis=-1)])
    else:
        mag = cum_max
    # median of each consecutive |x| triple (triple starting at j ends at j+2);
    # a triple touching any non-finite slot contributes exactly 0
    tri = np.sort(
        np.lib.stride_tricks.sliding_window_view(np.abs(X), 3, axis=0), axis=-1
    )[..., 1]
    tri_ok = isfin[2:] & isfin[1:-1] & isfin[:-2]
    s3 = np.where(tri_ok, tri.astype(ld) ** 2, ld(0.0))
    p_inc = np.cumsum(s3, axis=0)  # inclusive prefix of squared medians
    zero_row = np.zeros((1, cols), dtype=ld)
    p_pad = np.vstack([zero_row, p_inc])  # p_pad[j] = sum(s3[:j])
    # p_end[r] = sum(s3[:r-1]) = sum of triples starting at <= r-2
    p_end = np.vstack([zero_row, zero_row, p_inc])
    sum3 = p_end - np.take_along_axis(p_pad, np.clip(seg_start, 0, p_pad.shape[0] - 1), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_ratio = n_seg / (n_seg - 2)
    restored = ld(_MEDRV_CONST) * ld(n_ratio) * sum3
    tiny = ld(np.nextafter(0.0, 1.0))
    fmax = np.finfo(float).max
    ok = (
        (last_bad < seg_start)
        & (last_inf < i0)
        & (n_seg >= _MIN_FINITE)
        & (mag > 0.0)
        & np.isfinite(restored)
        & (restored <= fmax)
        & ~((restored > 0.0) & (restored < tiny))
    )
    return np.where(ok, restored, np.nan).astype(float)


def _variance_from_normalized(value: float, magnitude: float) -> float:
    """Restore a degree-two statistic without emitting overflow/underflow zeros."""
    restored = np.longdouble(value) * np.longdouble(magnitude) ** 2
    tiny = np.longdouble(np.nextafter(0.0, 1.0))
    if not np.isfinite(restored) or restored > np.finfo(float).max:
        return np.nan
    if restored > 0.0 and restored < tiny:
        return np.nan
    return float(restored)


def _medrv(v: np.ndarray) -> float:
    n = int(v.size)
    if n < 3 or not np.all(np.isfinite(v)):
        return np.nan
    magnitude = float(np.max(np.abs(v)))
    if magnitude == 0.0:
        return np.nan
    a = np.abs(v / magnitude)
    med = np.empty(n - 2, dtype=float)
    for i in range(2, n):
        med[i - 2] = float(np.median(a[i - 2 : i + 1]))
    normalized = float(_MEDRV_CONST * (n / (n - 2)) * float(np.sum(med * med)))
    return _variance_from_normalized(normalized, magnitude)


def _minrv(v: np.ndarray) -> float:
    n = int(v.size)
    if n < 2 or not np.all(np.isfinite(v)):
        return np.nan
    magnitude = float(np.max(np.abs(v)))
    if magnitude == 0.0:
        return np.nan
    a = np.abs(v / magnitude)
    normalized = float(_MINRV_CONST * (n / (n - 1)) * float(np.sum(np.minimum(a[1:], a[:-1]) ** 2)))
    return _variance_from_normalized(normalized, magnitude)


def _jump_z(v: np.ndarray) -> float:
    n = int(v.size)
    if n < 2:
        return np.nan
    magnitude = float(np.max(np.abs(v)))
    if not np.all(np.isfinite(v)) or magnitude == 0.0:
        return np.nan
    # The declared BNS ratio is homogeneous of degree zero. Normalize within
    # this window before squaring/fourth powers, never add a units-based floor.
    v = v / magnitude
    rv = float(np.sum(v * v))
    bv = float((np.pi / 2.0) * np.sum(np.abs(v[1:]) * np.abs(v[:-1])))
    rq4 = float(np.sum(v ** 4))  # raw quarticity sum Σr⁴
    denom2 = _THETA_MINUS_2 / 3.0 * rq4
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
        w = strict_int_param(window, "window", lower=_MIN_FINITE)
        return frame_like(returns, _rolling_medrv_vec(returns.to_numpy(dtype=float), w))


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
        w = strict_int_param(window, "window", lower=_MIN_FINITE)
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
        w = strict_int_param(window, "window", lower=_MIN_FINITE)
        return frame_like(returns, _rolling_returns(returns.to_numpy(dtype=float), w, _jump_z))


_NEW_CANONICALS = (
    "intraday_medrv",
    "intraday_minrv",
    "intraday_jump_test_stat",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
