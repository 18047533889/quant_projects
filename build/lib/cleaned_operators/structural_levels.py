# -*- coding: utf-8 -*-
"""Confirmed structural price levels (2026-08 geometry/math expansion).

A *structural level* is a price where the market demonstrably turned.  This
module mines those turning points from a price series and characterises the
current price's relationship to the recent structural skeleton:

* ``ts_structural_level_density``           — is the current price sitting inside a
  dense historical turning-point zone (mean Gaussian kernel mass on log-distance)?
* ``ts_nearest_structural_level_distance``  — how far (in log-return volatility
  units) is the nearest level, optionally restricted to resistance/support?
* ``ts_structural_level_strength``          — recency-weighted reaction to nearby
  levels (sum of nearby log-distance weighted by exp(-decay*age)).

Shared kernel — *confirmed pivots*, mined by the streaming
``StreamingConfirmedPivotLedger`` in ``cleaned_operators/common/_pivot_ledger.py``
(R11 P0: no retrospective history rewrite).  Bar ``k`` is a confirmed peak when
``x_k`` is a strict max over the full ``[k-confirmation, k+confirmation]``
neighbourhood (every element finite — a NaN inside the window blocks
confirmation) **and** the rally from the most recent confirmed trough exceeds
``prominence*x_k`` (troughs symmetric).  The first pivot of the chain is accepted
directly (it establishes the reference; without it the alternating chain can
never start).  A pivot becomes usable only once its confirmation window has
passed: pivot at bar ``k`` is used from row ``k+confirmation`` onward, with
``age = row - (k+confirmation)``.  ``max_pivot_age`` caps that usable age: once
a pivot is older than the cap it stops contributing, and if no active pivot
remains within the cap the level reading is NaN (R11 round-3 #14 — a stale
pivot from months ago must not keep being forward-filled).  A future more-extreme same-side candidate
*supplements* the ledger with a ``PivotSuperseded`` record — the already
published pivot is never deleted from history, and the supersession only affects
rows from the new pivot's confirmation row onward.  The *output* at row ``t``
only ever consumes data up to row ``t`` — prefix-causal and PIT-safe.

All operators are trailing-window per-column, deterministic, NaN-safe
(all-NaN / degenerate window -> NaN), and reject invalid parameters.  R14 P2:
each operator also applies a scale-coverage gate — a trailing window with fewer
than ``min_periods`` finite observations or covering less than
``min_coverage_fraction`` of the nominal window yields NaN (the level structure
is not comparable).  ``ts_structural_level_strength`` exposes the proximity
bandwidth as an explicit ``cutoff`` parameter (R14 P2 hidden-constant fix).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from cleaned_operators.common._pivot_ledger import confirmed_pivot_events
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12

# R11 round-3 #14: a confirmed pivot must not be forward-filled indefinitely.
# A pivot is usable from its confirmation row (``pivot_at + confirmation``) with
# ``age = t - (pivot_at + confirmation)``; once ``age > max_pivot_age`` it stops
# contributing and, when no active pivot remains within the age cap, the level
# reading becomes NaN (fail-closed) instead of broadcasting a stale six-month-old
# level.  ``None`` means "no explicit cap" — the trailing ``window`` bound alone
# applies (backward compatible).  This is a governance staleness knob, not a
# search dimension.
_MAX_PIVOT_AGE_SPEC = ParamSpec(
    dtype=int,
    min=1,
    default=None,
    param_role=ParamRole.POLICY,
    searchable=False,
)


def _active_pivots_within_age(
    active: tuple, t: int, conf: int, max_pivot_age: int | None
) -> tuple:
    """Filter confirmed pivots to those still within the staleness cap.

    ``max_pivot_age is None`` keeps the trailing-window behaviour unchanged
    (the ``window`` bound alone applies).  Otherwise a pivot whose usable age
    ``t - (pivot_at + confirmation)`` exceeds ``max_pivot_age`` is dropped —
    an old pivot must not keep being broadcast when no newer pivot exists.
    """
    if max_pivot_age is None:
        return active
    return tuple(ev for ev in active if (t - ev.pivot_at - conf) <= max_pivot_age)


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
        category="structural_levels",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "structural_levels", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_geometry",
            f"unit:{unit}", f"cost:{cost}",
        ],
        param_specs=param_specs or {},
    )


def _check_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    fv = float(value)
    if not np.isfinite(fv) or fv != float(int(fv)):
        raise ValueError(f"{name} must be an integer")
    iv = int(fv)
    if iv < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return iv


def _check_positive_float(value: Any, name: str) -> float:
    fv = float(value)
    if not np.isfinite(fv) or fv <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return fv


def _check_coverage_fraction(value: Any, name: str = "min_coverage_fraction") -> float:
    fv = float(value)
    if not np.isfinite(fv) or not (0.0 <= fv <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")
    return fv


def _window_covered(
    x: np.ndarray,
    t: int,
    window: int,
    min_periods: int,
    min_coverage_fraction: float,
) -> bool:
    """R14 P2 scale-coverage gate for structural levels.

    A structural-level reading is only comparable when the trailing price window
    that defines the neighbourhood contains enough finite observations
    (``min_periods``) covering a minimum fraction of the nominal window
    (``min_coverage_fraction``).  With few finite observations the level
    structure is not a fair reading of the window — the output must be NaN.
    """
    lo = max(0, t - window + 1)
    chunk = x[lo : t + 1]
    eff = int(np.isfinite(chunk).sum())
    if eff < int(min_periods):
        return False
    return eff / chunk.shape[0] >= float(min_coverage_fraction)


def _density_series(
    price2d: np.ndarray,
    window: int,
    prominence: float,
    confirmation: int,
    bandwidth: float,
    min_periods: int,
    min_coverage_fraction: float,
    max_pivot_age: int | None = None,
) -> np.ndarray:
    rows, cols = price2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    conf = int(confirmation)
    bw = float(bandwidth)
    w = int(window)
    mp = int(min_periods)
    mcf = float(min_coverage_fraction)
    for c in range(cols):
        x = price2d[:, c]
        ledger = confirmed_pivot_events(x, conf, float(prominence), positive_only=True)
        for t in range(rows):
            xt = x[t]
            if not np.isfinite(xt) or xt <= 0.0:
                continue
            if not _window_covered(x, t, w, mp, mcf):
                continue
            pt = float(xt)
            active = _active_pivots_within_age(
                ledger.active_at(t, window=w), t, conf, max_pivot_age
            )
            ds = [math.log(pt / ev.value) for ev in active if ev.value > 0.0]
            if not ds:
                continue
            d = np.asarray(ds, dtype=float)
            # R4-90: absolute level density (kernel mass per time bar), not a
            # nearby fraction — a NEW distant pivot must add mass, never pull the
            # mean down.  (1/window)·Σ K(d_i) is time-normalised.
            out[t, c] = float(np.sum(np.exp(-0.5 * (d / bw) ** 2)) / w)
    return out


def _nearest_distance_series(
    price2d: np.ndarray,
    window: int,
    prominence: float,
    confirmation: int,
    direction: str,
    min_periods: int,
    min_coverage_fraction: float,
    max_pivot_age: int | None = None,
) -> np.ndarray:
    rows, cols = price2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    conf = int(confirmation)
    w = int(window)
    mp = int(min_periods)
    mcf = float(min_coverage_fraction)
    for c in range(cols):
        x = price2d[:, c]
        ledger = confirmed_pivot_events(x, conf, float(prominence), positive_only=True)
        # rolling std of log-returns over the trailing window
        lr = np.full(rows, np.nan, dtype=float)
        for t in range(1, rows):
            if np.isfinite(x[t - 1]) and np.isfinite(x[t]) and x[t - 1] > 0.0 and x[t] > 0.0:
                lr[t] = math.log(x[t] / x[t - 1])
        scale = np.full(rows, np.nan, dtype=float)
        for t in range(rows):
            i0 = max(0, t - w + 1)
            v = lr[i0 : t + 1]
            v = v[np.isfinite(v)]
            if v.size >= 2:
                scale[t] = float(np.std(v))
        for t in range(rows):
            xt = x[t]
            if not np.isfinite(xt) or xt <= 0.0 or not np.isfinite(scale[t]):
                continue
            if not _window_covered(x, t, w, mp, mcf):
                continue
            pt = float(xt)
            best = np.inf
            for ev in _active_pivots_within_age(
                ledger.active_at(t, window=w), t, conf, max_pivot_age
            ):
                p = ev.value
                if p <= 0.0:
                    continue
                if direction == "any":
                    adi = abs(math.log(pt / p))
                elif direction == "above":
                    if not (p > pt):
                        continue
                    adi = abs(math.log(pt / p))
                else:  # below
                    if not (p < pt):
                        continue
                    adi = abs(math.log(pt / p))
                if adi < best:
                    best = adi
            if np.isfinite(best):
                out[t, c] = best / (scale[t] + _EPS)
    return out


def _strength_series(
    price2d: np.ndarray,
    window: int,
    prominence: float,
    confirmation: int,
    decay: float,
    cutoff: float,
    min_periods: int,
    min_coverage_fraction: float,
    max_pivot_age: int | None = None,
) -> np.ndarray:
    rows, cols = price2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    conf = int(confirmation)
    w = int(window)
    dec = float(decay)
    cut = float(cutoff)
    mp = int(min_periods)
    mcf = float(min_coverage_fraction)
    for c in range(cols):
        x = price2d[:, c]
        ledger = confirmed_pivot_events(x, conf, float(prominence), positive_only=True)
        for t in range(rows):
            xt = x[t]
            if not np.isfinite(xt) or xt <= 0.0:
                continue
            if not _window_covered(x, t, w, mp, mcf):
                continue
            pt = float(xt)
            active = _active_pivots_within_age(
                ledger.active_at(t, window=w), t, conf, max_pivot_age
            )
            if not active:
                continue
            total = 0.0
            for ev in active:
                p = ev.value
                if p <= 0.0:
                    continue
                di = math.log(pt / p)
                adi = abs(di)
                if adi > cut:
                    continue
                age = float(t - ev.pivot_at - conf)
                # R4-19: proximity weight must PEAK on the level (distance 0)
                # and decay with distance, not the reverse.  Old code used
                # |log(P/L)| so being exactly on a level contributed 0.
                total += math.exp(-dec * age) * (1.0 - adi / cut)
            out[t, c] = total
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_structural_level_density",
    category="structural_levels",
    business_category="structural_levels",
    canonical="ts_structural_level_density",
    source="structural_levels",
)
class TsStructuralLevelDensity(SeriesOperator):
    """当前价格位于历史结构密集区的程度（对数距离高斯核时间归一和 /window）。

    高 → 现价被大量历史确认转折价位包围（密集转折带）；低 → 价格处于历史
    稀疏区。时间归一后新出现的远处 pivot 只会增加质量，不会拉低均值（R4-90）。P1。
    """

    metadata = _metadata(
        "ts_structural_level_density",
        "结构价位密度（exp(-0.5*(log-dist/bandwidth)^2) 时间归一和 /window，绝对密度）。",
        ["price", "window", "prominence", "confirmation", "bandwidth",
         "min_periods", "min_coverage_fraction", "max_pivot_age"],
        unit="ratio",
        cost=5,
        param_specs={"max_pivot_age": _MAX_PIVOT_AGE_SPEC},
    )

    def _calculate_series(
        self,
        price: pd.DataFrame,
        window: int = 120,
        prominence: float = 0.02,
        confirmation: int = 3,
        bandwidth: float = 0.03,
        min_periods: int = 20,
        min_coverage_fraction: float = 0.5,
        max_pivot_age: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        prom = _check_positive_float(prominence, "prominence")
        conf = _check_int(confirmation, "confirmation", 1)
        bw = _check_positive_float(bandwidth, "bandwidth")
        mp = _check_int(min_periods, "min_periods", 1)
        mcf = _check_coverage_fraction(min_coverage_fraction)
        m_age = w if max_pivot_age is None else _check_int(max_pivot_age, "max_pivot_age", 1)
        return frame_like(
            price,
            _density_series(price.to_numpy(dtype=float), w, prom, conf, bw, mp, mcf, m_age),
        )


@register_operator(
    name="ts_nearest_structural_level_distance",
    category="structural_levels",
    business_category="structural_levels",
    canonical="ts_nearest_structural_level_distance",
    source="structural_levels",
)
class TsNearestStructuralLevelDistance(SeriesOperator):
    """到最近确认结构价位的对数距离 / 对数收益滚动标准差（阻力/支撑可选）。

    direction="above" 只看上方阻力位；"below" 只看下方支撑位；"any" 全部。
    该方向无价位 → NaN。P1。
    """

    metadata = _metadata(
        "ts_nearest_structural_level_distance",
        "最近结构价位距离（按对数收益波动归一），支持上方/下方方向过滤。",
        ["price", "window", "prominence", "confirmation", "direction",
         "min_periods", "min_coverage_fraction", "max_pivot_age"],
        unit="ratio",
        cost=5,
        param_specs={"max_pivot_age": _MAX_PIVOT_AGE_SPEC},
    )

    def _calculate_series(
        self,
        price: pd.DataFrame,
        window: int = 120,
        prominence: float = 0.02,
        confirmation: int = 3,
        direction: str = "any",
        min_periods: int = 20,
        min_coverage_fraction: float = 0.5,
        max_pivot_age: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        prom = _check_positive_float(prominence, "prominence")
        conf = _check_int(confirmation, "confirmation", 1)
        if direction not in ("any", "above", "below"):
            raise ValueError("direction must be 'any', 'above' or 'below'")
        mp = _check_int(min_periods, "min_periods", 1)
        mcf = _check_coverage_fraction(min_coverage_fraction)
        m_age = w if max_pivot_age is None else _check_int(max_pivot_age, "max_pivot_age", 1)
        return frame_like(
            price,
            _nearest_distance_series(
                price.to_numpy(dtype=float), w, prom, conf, direction, mp, mcf, m_age
            ),
        )


@register_operator(
    name="ts_structural_level_strength",
    category="structural_levels",
    business_category="structural_levels",
    canonical="ts_structural_level_strength",
    source="structural_levels",
)
class TsStructuralLevelStrength(SeriesOperator):
    """近旁结构价位的按年龄衰减反应强度（距离核 (1-|log-dist|/cutoff)+ 的加权和）。

    权重在价位上（距离 0）最大，随距离线性衰减到 cutoff（R4-19）。cutoff 是
    "什么算近旁价位"的经济阈值（对数距离带宽），已作为显式参数暴露在契约中。
    高 → 现价贴近多个较新确认价位（潜在反应区）。P1。
    """

    metadata = _metadata(
        "ts_structural_level_strength",
        "近旁结构价位加权反应：Σ exp(-decay*age)*(1-|log-dist|/cutoff)，距离越近权重越大。",
        ["price", "window", "prominence", "confirmation", "decay", "cutoff",
         "min_periods", "min_coverage_fraction", "max_pivot_age"],
        unit="strength",
        cost=5,
        # ``cutoff`` is a float proximity bandwidth (log-distance).  Declaring
        # the ParamSpec overrides the legacy name whitelist that would otherwise
        # force ``cutoff`` to an integer (R14 P2 hidden-constant fix).
        param_specs={
            "cutoff": ParamSpec(dtype=float, min=0.0),
            "max_pivot_age": _MAX_PIVOT_AGE_SPEC,
        },
    )

    def _calculate_series(
        self,
        price: pd.DataFrame,
        window: int = 120,
        prominence: float = 0.02,
        confirmation: int = 3,
        decay: float = 0.05,
        cutoff: float = 0.05,
        min_periods: int = 20,
        min_coverage_fraction: float = 0.5,
        max_pivot_age: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        prom = _check_positive_float(prominence, "prominence")
        conf = _check_int(confirmation, "confirmation", 1)
        dec = float(decay)
        if not np.isfinite(dec) or dec < 0.0:
            raise ValueError("decay must be a finite non-negative number")
        cut = _check_positive_float(cutoff, "cutoff")
        mp = _check_int(min_periods, "min_periods", 1)
        mcf = _check_coverage_fraction(min_coverage_fraction)
        m_age = w if max_pivot_age is None else _check_int(max_pivot_age, "max_pivot_age", 1)
        return frame_like(
            price,
            _strength_series(price.to_numpy(dtype=float), w, prom, conf, dec, cut, mp, mcf, m_age),
        )


_NEW_CANONICALS = (
    "ts_structural_level_density",
    "ts_nearest_structural_level_distance",
    "ts_structural_level_strength",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
