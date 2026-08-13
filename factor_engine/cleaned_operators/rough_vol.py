# -*- coding: utf-8 -*-
"""Volatility roughness operators (2026-08 geometry/math expansion).

Rough-volatility models describe how the *increment variation* of a price path
scales with the sampling lag.  For the normalized structure function
``S_p(Delta) = mean |Delta_Delta x|^p`` taken over the finite endpoint pairs, a
pure fractional process obeys ``log S_p(Delta) ~ a + p*H*log Delta``, so the
slope of ``log S_p`` vs ``log Delta`` divided by ``p`` estimates the
Hurst/roughness parameter ``H`` (``H`` close to 0.5 = Brownian; ``H < 0.5`` =
rougher).

The *mean*-normalized form (never ``sum``) is used throughout the H estimator:
different lags admit different numbers of valid increments (delta=1 has N-1,
delta=8 has N-8, delta=16 has N-16), so a ``sum`` would pollute
``log S_p(delta)`` with ``log N_delta`` and systematically bias the slope.

* ``ts_vol_pvariation_roughness`` — Hurst ``H`` from a single slope fit over the
  user-supplied scale grid (``scales=(1,2,4)``).
* ``ts_vol_scaling_break`` — short-lag Hurst minus long-lag Hurst
  (``H(1,2) - H(8,16)``); a large positive value means the process is rough at
  short lags but smooth at long lags (volatility-regime/scale break).

Both are trailing-window, prefix-causal and deterministic.  The original time
axis is preserved (review R4-59): lag increments are taken at their true lag
positions and an increment whose endpoint is NaN contributes nothing to
``S_p`` — values are never dropped and re-connected, which would change the
real time meaning of the lags (1/2/4/8/16).  Missing data are guarded per-scale
by a data-quality gate: a scale needs at least ``min_pairs`` finite endpoint
pairs AND a finite-pair ``coverage >= min_pair_fraction`` (fraction of the
possible pairs), AND the used scales' coverages must not be severely imbalanced
(``max/min coverage <= _MAX_COVERAGE_IMBALANCE``); a window that cannot
estimate all required scales on comparable data emits NaN rather than a biased
estimate.  Windows with too few finite observations emit NaN, never Inf.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    ParamRole,
    SeriesOperator,
    register_operator,
)
from cleaned_operators.rolling_pack import check_window, frame_like, register_polars_bridge

_EPS = 1e-12
_MIN_FINITE = 5
_LONG_LAG = 16  # ts_vol_scaling_break needs N >= LONG_LAG+1 increments
# Missing-data guard: per-scale coverage (fraction of possible pairs that are
# finite) may not be severely imbalanced across the *used* scales — a missing
# pattern that only lets the short lags see data must not manufacture fake
# scaling.  With coverage in [min_pair_fraction, 1] the default
# ``min_pair_fraction=0.5`` keeps max/min <= 2, so this only binds when a caller
# loosens ``min_pair_fraction`` well below 0.5.
_MAX_COVERAGE_IMBALANCE = 4.0


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    param_specs: dict[str, ParamSpec] | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_volatility",
        description=description,
        param_names=params,
        return_type="series",
        param_specs=param_specs or {},
        tags=[
            "time_series_volatility", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:rough_volatility",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _rolling_values(values: np.ndarray, window: int, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    """Trailing-window per-column reduction over a price/level panel.

    Review R4-59: the raw window (missing values kept at their true positions)
    is passed to ``fn``; lag increments are computed at their true lag offsets
    and NaN-endpoint increments are skipped — never drop-and-reconnect.
    """
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = values[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            v = col[i0 : r + 1]
            if np.isfinite(v).sum() < _MIN_FINITE:
                continue
            out[r, c] = fn(v)
    return out


def _validate_p(p: float) -> float:
    pp = float(p)
    if not np.isfinite(pp) or pp <= 0.0:
        raise ValueError("p must be a positive finite number")
    return pp


def _validate_min_pairs(min_pairs: int) -> int:
    mp = int(min_pairs)
    if mp < 1:
        raise ValueError("min_pairs must be a positive integer")
    return mp


def _validate_min_pair_fraction(min_pair_fraction: float) -> float:
    mpf = float(min_pair_fraction)
    if not np.isfinite(mpf) or not (0.0 < mpf <= 1.0):
        raise ValueError("min_pair_fraction must be in (0, 1]")
    return mpf


def _validate_scales(scales: Any) -> tuple[int, ...]:
    """Strict positive-integer scale grid (review R4-60).

    A float that is not an exact integer (e.g. ``1.9``) is rejected instead of
    being truncated to ``1`` and silently merged with a duplicate grid.
    """
    if isinstance(scales, (int, float, np.integer, np.floating)) and not isinstance(scales, bool):
        scales = (scales,)
    if not isinstance(scales, (tuple, list)) or len(scales) < 2:
        raise ValueError("scales must be a sequence of at least two positive integers")
    out: list[int] = []
    for s in scales:
        if isinstance(s, (bool, np.bool_)):
            raise ValueError("scales must contain only positive integers")
        if isinstance(s, (int, np.integer)):
            si = int(s)
        elif isinstance(s, (float, np.floating)):
            if not np.isfinite(float(s)) or float(s) != float(int(s)):
                raise ValueError(f"scale {s!r} must be a positive integer")
            si = int(s)
        else:
            try:
                si = int(s)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"scale {s!r} must be a positive integer") from exc
            if float(s) != float(si):
                raise ValueError(f"scale {s!r} must be a positive integer")
        if si < 1:
            raise ValueError("scales must contain only positive integers")
        out.append(si)
    cleaned = tuple(sorted(set(out)))
    if len(cleaned) < 2:
        raise ValueError("scales must contain at least two distinct values")
    return cleaned


def _structure_function(v: np.ndarray, delta: int, p: float) -> tuple[float, int, float] | None:
    """Normalized structure function ``S_p(delta) = mean |x_{t+delta}-x_t|^p``.

    Computed on the ORIGINAL time axis (review R4-59): the lag-delta increment
    at position ``t`` is ``v[t+delta]-v[t]`` regardless of gaps elsewhere, and an
    increment whose either endpoint is NaN is skipped — values are never dropped
    and re-connected.  Returns ``(S_p, n_pairs, coverage)`` where ``n_pairs`` is
    the number of finite endpoint pairs and ``coverage = n_pairs / (N - delta)``
    is the fraction of the delta-lag pairs that are finite.  ``None`` when there
    are no finite pairs.
    """
    d = int(delta)
    n = int(v.size)
    if n <= d:
        return None
    inc = v[d:] - v[:-d]
    finite = np.isfinite(inc)
    n_pairs = int(finite.sum())
    if n_pairs == 0:
        return None
    coverage = np.where((n - d) != 0, n_pairs / (n - d), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        sp = float(np.mean(np.abs(inc[finite]) ** p))
    return sp, n_pairs, coverage


def _scale_passes_gate(
    res: tuple[float, int, float] | None, min_pairs: int, min_pair_fraction: float
) -> bool:
    """Per-scale data-quality gate (BUG 2): enough finite pairs AND coverage.

    A scale whose missingness leaves too few increments — or covers only a small
    fraction of its possible pairs — is not comparable to the other scales and
    must not enter the scaling fit.
    """
    if res is None:
        return False
    sp, n_pairs, coverage = res
    if n_pairs < min_pairs:
        return False
    if coverage < min_pair_fraction:
        return False
    if not (np.isfinite(sp) and sp > _EPS):
        return False
    return True


def _coverages_balanced(covs: list[float], imbalance_bound: float) -> bool:
    """Per-scale coverages must not be severely imbalanced across used scales."""
    return np.where(min(covs) <= imbalance_bound != 0, max(covs) / min(covs) <= imbalance_bound, np.nan)


def _pv_roughness(
    v: np.ndarray,
    p: float,
    scales: tuple[int, ...],
    min_pairs: int,
    min_pair_fraction: float,
    imbalance_bound: float,
) -> float:
    pts: list[tuple[float, float]] = []
    covs: list[float] = []
    for d in scales:
        res = _structure_function(v, d, p)
        if not _scale_passes_gate(res, min_pairs, min_pair_fraction):
            continue
        sp, n_pairs, coverage = res
        pts.append((float(np.log(d)), float(np.log(sp))))
        covs.append(coverage)
    if len(pts) < 2:
        return np.nan
    if not _coverages_balanced(covs, imbalance_bound):
        return np.nan
    xs = np.asarray([a for a, _ in pts], dtype=float)
    ys = np.asarray([b for _, b in pts], dtype=float)
    slope = float(np.polyfit(xs, ys, 1)[0])
    return np.where(p) != 0, float(slope / p), np.nan)


def _scaling_break(
    v: np.ndarray,
    p: float,
    min_pairs: int,
    min_pair_fraction: float,
    imbalance_bound: float,
) -> float:
    n = int(v.size)
    if n < _LONG_LAG + 1:
        return np.nan
    lags = (1, 2, 8, _LONG_LAG)
    sps: list[float] = []
    covs: list[float] = []
    for d in lags:
        res = _structure_function(v, d, p)
        if not _scale_passes_gate(res, min_pairs, min_pair_fraction):
            return np.nan
        sp, n_pairs, coverage = res
        sps.append(sp)
        covs.append(coverage)
    if not _coverages_balanced(covs, imbalance_bound):
        return np.nan
    v1, v2, v8, v16 = sps
    # slope_short = (log S(2) - log S(1)) / (log 2 - log 1); log 1 = 0.
    h_short = np.where(np.log(2.0) / p) != 0, float((np.log(v2) - np.log(v1)) / np.log(2.0) / p), np.nan)
    # slope_long = (log S(16) - log S(8)) / (log 16 - log 8) = ... / log 2.
    h_long = np.where(np.log(2.0) / p) != 0, float((np.log(v16) - np.log(v8)) / np.log(2.0) / p), np.nan)
    return float(h_short - h_long)


@register_operator(
    name="ts_vol_pvariation_roughness",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_vol_pvariation_roughness",
    source="rough_vol",
)
class TsVolPvariationRoughness(SeriesOperator):
    """p-变差尺度指数：``log S_p(Δ)`` 对 ``log Δ`` 的斜率 / p → Hurst/roughness ``H``。

    这是**通用 p-变差 scaling 指数**（generic p-variation scaling exponent）：
    输入任意 level 序列 ``x``（价格、波动率、对数波动率……），估计的是该序列
    增量变差的尺度行为，并非特指「波动率过程的 H」。``S_p(Δ)=mean|Δ_Δ x|^p``
    （滞后 Δ 增量、有限端点对的均值，保留原时间轴，R4-59；用均值而非求和，
    消除不同 Δ 有效增量数不同带来的 ``log N_Δ`` 偏置）。在 ``scales`` 网格上
    最小二乘拟合 ``log S_p(Δ)=a+b·log Δ``，``H=b/p``。``H≈0.5`` = 布朗运动，
    ``H<0.5`` = 粗糙。每个 scale 需通过数据质量门：``n_pairs>=min_pairs`` 且
    ``coverage>=min_pair_fraction``，且所用 scale 的 coverage 最大/最小比
    <= ``_MAX_COVERAGE_IMBALANCE``，否则该行输出 NaN（不做有偏估计）。
    不足 2 个有效 scale → NaN。单位 ratio。P1。精确泛用名
    ``ts_pvariation_scaling_exponent`` 为本算子的别名。
    """

    metadata = _metadata(
        "ts_vol_pvariation_roughness",
        "p-变差尺度指数（Hurst/roughness H；通用 scaling exponent）。",
        ["x", "window", "p", "scales", "min_pairs", "min_pair_fraction"],
        unit="ratio",
        cost=6,
        param_specs={
            "min_pairs": ParamSpec(dtype=int, min=1, default=5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "min_pair_fraction": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        p: float = 2.0,
        scales: tuple[int, ...] = (1, 2, 4),
        min_pairs: int = 5,
        min_pair_fraction: float = 0.5,
        **_: Any,
    ) -> pd.DataFrame:
        w = check_window(window)
        pp = _validate_p(p)
        sc = _validate_scales(scales)
        mp = _validate_min_pairs(min_pairs)
        mpf = _validate_min_pair_fraction(min_pair_fraction)
        fn = lambda v: _pv_roughness(v, pp, sc, mp, mpf, _MAX_COVERAGE_IMBALANCE)  # noqa: E731
        return frame_like(x, _rolling_values(x.to_numpy(dtype=float), w, fn))


@register_operator(
    name="ts_vol_scaling_break",
    category="time_series_volatility",
    business_category="time_series_volatility",
    canonical="ts_vol_scaling_break",
    source="rough_vol",
)
class TsVolScalingBreak(SeriesOperator):
    """尺度断裂：短滞后 Hurst 与长滞后 Hurst 之差 ``H(1,2) - H(8,16)``。

    分别用滞后 (1,2) 与 (8,16) 的 ``log S_p`` 斜率 / p 估 H（``S_p`` 为有限端点对
    的均值结构函数，消除 scale-count 偏置）。正值 = 短程粗糙、长程平滑（波动尺度
    断裂/regime 结构）；≈0 = 单一尺度行为。四个滞后每个都需通过数据质量门
    （``min_pairs`` / ``min_pair_fraction`` / coverage 失衡上限），否则 NaN。
    有效值不足（滞后 16 需要 ≥17 个有限观测）→ NaN。单位 ratio。P2。
    """

    metadata = _metadata(
        "ts_vol_scaling_break",
        "短程 vs 长程 Hurst 指数之差。",
        ["x", "window", "p", "min_pairs", "min_pair_fraction"],
        unit="ratio",
        cost=6,
        param_specs={
            "min_pairs": ParamSpec(dtype=int, min=1, default=5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "min_pair_fraction": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        p: float = 2.0,
        min_pairs: int = 5,
        min_pair_fraction: float = 0.5,
        **_: Any,
    ) -> pd.DataFrame:
        w = check_window(window)
        pp = _validate_p(p)
        mp = _validate_min_pairs(min_pairs)
        mpf = _validate_min_pair_fraction(min_pair_fraction)
        fn = lambda v: _scaling_break(v, pp, mp, mpf, _MAX_COVERAGE_IMBALANCE)  # noqa: E731
        return frame_like(x, _rolling_values(x.to_numpy(dtype=float), w, fn))


_NEW_CANONICALS = (
    "ts_vol_pvariation_roughness",
    "ts_vol_scaling_break",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


def _register_precise_alias() -> None:
    """R4-61: expose the precise generic name as an alias.

    The canonical stays ``ts_vol_pvariation_roughness`` (polars_geometry_math
    hard-codes that name in its polars-backend list, so a canonical rename is
    not loadable without editing that module).  The generic spelling
    ``ts_pvariation_scaling_exponent`` is registered as an alias so new DSL
    expressions can use the precise name; old expressions keep working.
    """
    from cleaned_operators.registry import OperatorRegistry

    OperatorRegistry.register_alias(
        "ts_pvariation_scaling_exponent",
        "ts_vol_pvariation_roughness",
    )


_register_surface()
_register_precise_alias()
