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

Shared kernel — *confirmed pivots*.  Bar ``k`` is a confirmed peak when ``x_k`` is
a strict max over ``[k-confirmation, k+confirmation]`` **and** the rally from the
most recent confirmed trough exceeds ``prominence*x_k`` (troughs symmetric).  The
first pivot of the chain is accepted directly (it establishes the reference;
without it the alternating chain can never start).  A pivot becomes usable only
once its confirmation window has passed: pivot at bar ``k`` is used from row
``k+confirmation`` onward, with ``age = row - (k+confirmation)``.  So although
pivot *detection* looks ``confirmation`` bars ahead, the *output* at row ``t``
only ever consumes data up to row ``t`` — prefix-causal and PIT-safe.

All operators are trailing-window per-column, deterministic, NaN-safe
(all-NaN / degenerate window -> NaN), and reject invalid parameters.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
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


def _scan_confirmed_pivots(x: np.ndarray, confirmation: int, prominence: float) -> list[tuple[int, float, bool]]:
    """Return ``(bar_idx, price, is_peak)`` for all confirmed pivots of a column.

    ``x`` is the 1D per-column price series.  NaN bars are skipped (they neither
    form pivots nor break the chain).  A strict peak/trough alternation state
    machine (review R4-18): after a peak the next accepted pivot is a trough
    (prominence-filtered against the peak), and vice-versa; a same-side more
    extreme candidate *replaces* the previous pivot instead of appending, so the
    chain can never contain two consecutive peaks or two consecutive troughs.
    The first non-flat pivot seeds the chain.
    """
    n = len(x)
    conf = int(confirmation)
    prom = float(prominence)
    pivots: list[tuple[int, float, bool]] = []
    last_type: bool | None = None  # True=peak, False=trough
    last_val: float | None = None
    for k in range(n):
        xk = x[k]
        if not np.isfinite(xk) or xk <= 0.0:
            continue
        lo = max(0, k - conf)
        hi = min(n - 1, k + conf)
        is_peak_cand = True
        is_trough_cand = True
        for j in range(lo, hi + 1):
            if j == k:
                continue
            xj = x[j]
            if not np.isfinite(xj):
                continue
            if xk <= xj:
                is_peak_cand = False
            if xk >= xj:
                is_trough_cand = False
            if not is_peak_cand and not is_trough_cand:
                break
        if is_peak_cand and is_trough_cand:
            continue  # flat neighbourhood (all equal): not a pivot
        pxk = float(xk)
        if last_type is None:  # seed the chain
            is_peak = is_peak_cand
            pivots.append((k, pxk, is_peak))
            last_type = is_peak
            last_val = pxk
        elif last_type:  # last was a peak -> next legal pivot is a trough
            if is_trough_cand and last_val - pxk > prom * abs(pxk):
                pivots.append((k, pxk, False))
                last_type = False
                last_val = pxk
            elif is_peak_cand and pxk > last_val:
                # same-side more extreme peak: replace, do not append (R4-18)
                pivots[-1] = (k, pxk, True)
                last_val = pxk
        else:  # last was a trough -> next legal pivot is a peak
            if is_peak_cand and pxk - last_val > prom * abs(pxk):
                pivots.append((k, pxk, True))
                last_type = True
                last_val = pxk
            elif is_trough_cand and pxk < last_val:
                # same-side more extreme trough: replace, do not append (R4-18)
                pivots[-1] = (k, pxk, False)
                last_val = pxk
    return pivots


def _usable_pivots(
    pivots: list[tuple[int, float, bool]], t: int, window: int, confirmation: int
) -> list[tuple[int, float, bool]]:
    """Pivots whose confirmation has passed and whose bar lies in the trailing window."""
    conf = int(confirmation)
    i0 = max(0, t - int(window) + 1)
    return [(k, p, is_peak) for (k, p, is_peak) in pivots if k + conf <= t and k >= i0]


def _density_series(price2d: np.ndarray, window: int, prominence: float, confirmation: int, bandwidth: float) -> np.ndarray:
    rows, cols = price2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    conf = int(confirmation)
    bw = float(bandwidth)
    w = int(window)
    for c in range(cols):
        x = price2d[:, c]
        pivots = _scan_confirmed_pivots(x, conf, float(prominence))
        for t in range(rows):
            xt = x[t]
            if not np.isfinite(xt) or xt <= 0.0:
                continue
            pt = float(xt)
            ds = [math.log(pt / p) for (k, p, _is_peak) in _usable_pivots(pivots, t, w, conf) if p > 0.0]
            if not ds:
                continue
            d = np.asarray(ds, dtype=float)
            # R4-90: absolute level density (kernel mass per time bar), not a
            # nearby fraction — a NEW distant pivot must add mass, never pull the
            # mean down.  (1/window)·Σ K(d_i) is time-normalised.
            out[t, c] = float(np.sum(np.exp(-0.5 * (d / bw) ** 2)) / w)
    return out


def _nearest_distance_series(
    price2d: np.ndarray, window: int, prominence: float, confirmation: int, direction: str
) -> np.ndarray:
    rows, cols = price2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    conf = int(confirmation)
    w = int(window)
    for c in range(cols):
        x = price2d[:, c]
        pivots = _scan_confirmed_pivots(x, conf, float(prominence))
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
            pt = float(xt)
            best = np.inf
            for (k, p, _is_peak) in _usable_pivots(pivots, t, w, conf):
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
    price2d: np.ndarray, window: int, prominence: float, confirmation: int, decay: float
) -> np.ndarray:
    rows, cols = price2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    conf = int(confirmation)
    w = int(window)
    dec = float(decay)
    for c in range(cols):
        x = price2d[:, c]
        pivots = _scan_confirmed_pivots(x, conf, float(prominence))
        for t in range(rows):
            xt = x[t]
            if not np.isfinite(xt) or xt <= 0.0:
                continue
            pt = float(xt)
            usable = _usable_pivots(pivots, t, w, conf)
            if not usable:
                continue
            total = 0.0
            cutoff = 0.05
            for (k, p, _is_peak) in usable:
                if p <= 0.0:
                    continue
                di = math.log(pt / p)
                adi = abs(di)
                if adi > cutoff:
                    continue
                age = float(t - k - conf)
                # R4-19: proximity weight must PEAK on the level (distance 0)
                # and decay with distance, not the reverse.  Old code used
                # |log(P/L)| so being exactly on a level contributed 0.
                total += math.exp(-dec * age) * (1.0 - adi / cutoff)
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
        ["price", "window", "prominence", "confirmation", "bandwidth"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        price: pd.DataFrame,
        window: int = 120,
        prominence: float = 0.02,
        confirmation: int = 3,
        bandwidth: float = 0.03,
        **_: Any,
    ) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        prom = _check_positive_float(prominence, "prominence")
        conf = _check_int(confirmation, "confirmation", 1)
        bw = _check_positive_float(bandwidth, "bandwidth")
        return frame_like(
            price,
            _density_series(price.to_numpy(dtype=float), w, prom, conf, bw),
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
        ["price", "window", "prominence", "confirmation", "direction"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        price: pd.DataFrame,
        window: int = 120,
        prominence: float = 0.02,
        confirmation: int = 3,
        direction: str = "any",
        **_: Any,
    ) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        prom = _check_positive_float(prominence, "prominence")
        conf = _check_int(confirmation, "confirmation", 1)
        if direction not in ("any", "above", "below"):
            raise ValueError("direction must be 'any', 'above' or 'below'")
        return frame_like(
            price,
            _nearest_distance_series(price.to_numpy(dtype=float), w, prom, conf, direction),
        )


@register_operator(
    name="ts_structural_level_strength",
    category="structural_levels",
    business_category="structural_levels",
    canonical="ts_structural_level_strength",
    source="structural_levels",
)
class TsStructuralLevelStrength(SeriesOperator):
    """近旁结构价位的按年龄衰减反应强度（距离核 (1-|log-dist|/0.05)+ 的加权和）。

    权重在价位上（距离 0）最大，随距离线性衰减到 cutoff（R4-19）。高 → 现价贴近
    多个较新确认价位（潜在反应区）。P1。
    """

    metadata = _metadata(
        "ts_structural_level_strength",
        "近旁结构价位加权反应：Σ exp(-decay*age)*(1-|log-dist|/0.05)，距离越近权重越大。",
        ["price", "window", "prominence", "confirmation", "decay"],
        unit="strength",
        cost=5,
    )

    def _calculate_series(
        self,
        price: pd.DataFrame,
        window: int = 120,
        prominence: float = 0.02,
        confirmation: int = 3,
        decay: float = 0.05,
        **_: Any,
    ) -> pd.DataFrame:
        w = _check_int(window, "window", 2)
        prom = _check_positive_float(prominence, "prominence")
        conf = _check_int(confirmation, "confirmation", 1)
        dec = float(decay)
        if not np.isfinite(dec) or dec < 0.0:
            raise ValueError("decay must be a finite non-negative number")
        return frame_like(
            price,
            _strength_series(price.to_numpy(dtype=float), w, prom, conf, dec),
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
