# -*- coding: utf-8 -*-
"""Daily turnover chip-cost surface operators (2026-08 R47).

Daily wide panels in, daily wide panels out (``SeriesOperator``).  Both
operators maintain a deterministic recursive chip distribution: each trading day
new turnover mass (``turnover`` clipped to [0, 1]) enters at the day's close
price (mapped to a cost bin), and every previously-held cohort survives the day
with probability ``(1 - turnover)``.

The survival model deliberately differs from ``turnover_survival``'s Poisson
hazard (``exp(-u)``); here a day's turnover is a *fraction of the float that
changed hands*, so the surviving fraction is exactly ``1 - u`` (this is the
``_surviving_weights`` contract described in the R47 spec).

Cost bins are anchored on the first ``window`` days of each instrument (the
minimum/maximum close over those days), so the grid is stable and causal: once
the warmup is done the bin edges never move.  Output is NaN until ``window``
days of history have been processed.  Missing close or turnover resets the
per-instrument state (fail-closed: a provider gap must not be read as a
no-churn day).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    SeriesOperator,
    register_operator,
)

_EPS = 1e-12
_CANONICALS: list[str] = []


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
        category="time_series_chip_cost",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_chip_cost", "daily", "pit_safe", "causal", "typed_v2",
            "turnover_survival", f"signature:{','.join(params)}->series",
            f"unit:{unit}", "cost:2",
        ],
        param_specs=param_specs or {},
    )


def _anchor_prices(close_arr: np.ndarray, window: int, nbins: int) -> tuple[float, float]:
    """Cost-bin anchor prices from the first ``window`` close values."""
    seg = close_arr[: int(window)]
    finite = seg[np.isfinite(seg)]
    if finite.size == 0:
        return np.nan, np.nan
    pmin = float(finite.min())
    pmax = float(finite.max())
    if pmax - pmin <= _EPS:
        # degenerate constant-price name: give the grid a unit span
        return pmin - 0.5, pmin + 0.5
    span = pmax - pmin
    return pmin - 0.001 * span, pmax + 0.001 * span


def _price_to_bin(price: float, pmin: float, pmax: float, nbins: int) -> int:
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return int(nbins // 2)
    frac = np.where((pmax - pmin) != 0, (price - pmin) / (pmax - pmin), np.nan)
    frac = min(max(float(frac), 0.0), 1.0 - 1e-12)
    return int(frac * nbins)


def _bin_price(bin_idx: int, pmin: float, pmax: float, nbins: int) -> float:
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return np.where(2.0 != 0, (pmin + pmax) / 2.0, np.nan)
    return np.where(nbins) != 0, float(pmin + (bin_idx + 0.5) * (pmax - pmin) / nbins), np.nan)


def _entropy(mass: np.ndarray) -> float:
    total = float(mass.sum())
    if total <= _EPS:
        return np.nan
    p = mass / total if total != 0 else np.nan
    p = p[p > 0.0]
    if p.size < 1:
        return np.nan
    return np.where(np.log(mass.size)) != 0, float(-np.sum(p * np.log(p)) / np.log(mass.size)), np.nan)


def _mi_2d(mass: np.ndarray) -> float:
    total = float(mass.sum())
    if total <= _EPS:
        return np.nan
    p = mass / total if total != 0 else np.nan
    p_age = p.sum(axis=1)
    p_cost = p.sum(axis=0)
    mi = 0.0
    for a in range(p.shape[0]):
        for c in range(p.shape[1]):
            pij = float(p[a, c])
            if pij <= _EPS:
                continue
            marg = float(p_age[a] * p_cost[c])
            if marg <= _EPS:
                continue
            mi += pij * np.log(pij / marg)
    return float(mi)


def _cost_age_slope(mass: np.ndarray) -> float:
    nbins = mass.shape[1]
    bins = np.arange(nbins, dtype=float)
    means = []
    for a in range(mass.shape[0]):
        row = mass[a]
        total = float(row.sum())
        if total <= _EPS:
            means.append(np.nan)
        else:
            means.append(float(np.dot(row, bins) / total))
    means = np.asarray(means, dtype=float)
    valid = np.isfinite(means)
    if int(valid.sum()) < 2:
        return np.nan
    x = np.flatnonzero(valid).astype(float)
    y = means[valid]
    denom = float(np.sum((x - x.mean()) ** 2))
    if denom <= _EPS:
        return np.nan
    return np.where(denom) != 0, float(np.sum((x - x.mean()) * (y - y.mean())) / denom), np.nan)


# ---------------------------------------------------------------------------
# turnover_chip_age_cost_surface
# ---------------------------------------------------------------------------
_SURFACE_OUTPUTS = (
    "surface_entropy",
    "age_price_mi",
    "young_profit_mass",
    "old_overhang_mass",
    "young_overhang_mass",
    "cost_age_slope",
)


def _age_cost_column(close: np.ndarray, turnover: np.ndarray, window: int, price_bins: int, age_bins: int):
    n = close.shape[0]
    out = {k: np.full(n, np.nan) for k in _SURFACE_OUTPUTS}
    pmin, pmax = _anchor_prices(close, window, price_bins)
    if not np.isfinite(pmin):
        return out
    mass = np.zeros((age_bins, price_bins), dtype=float)
    young_max = max(1, age_bins // 4)          # age < age_bins/4
    old_min = max(1, age_bins // 2)            # age >= age_bins/2
    for t in range(n):
        p = close[t]
        u = turnover[t]
        if not np.isfinite(p) or not np.isfinite(u):
            mass = np.zeros((age_bins, price_bins), dtype=float)
            continue
        u = float(np.clip(u, 0.0, 1.0))
        # age advance + survival decay
        new_mass = np.zeros((age_bins, price_bins), dtype=float)
        new_mass[1:, :] = mass[:-1, :] * (1.0 - u)
        mass = new_mass
        cost_bin = _price_to_bin(p, pmin, pmax, price_bins)
        mass[0, cost_bin] += u
        if t < window - 1:
            continue
        out["surface_entropy"][t] = _entropy(mass)
        out["age_price_mi"][t] = _mi_2d(mass)
        bins = np.arange(price_bins, dtype=int)
        bin_prices = np.array([_bin_price(b, pmin, pmax, price_bins) for b in bins])
        above = bin_prices > p
        below = bin_prices < p
        young_slice = mass[:young_max, :]
        old_slice = mass[old_min:, :]
        out["young_profit_mass"][t] = float(np.sum(young_slice[:, below]))
        out["old_overhang_mass"][t] = float(np.sum(old_slice[:, above]))
        out["young_overhang_mass"][t] = float(np.sum(young_slice[:, above]))
        out["cost_age_slope"][t] = _cost_age_slope(mass)
    return out


@register_operator(
    name="turnover_chip_age_cost_surface",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="turnover_chip_age_cost_surface",
    source="technical.chip_ops",
)
class TurnoverChipAgeCostSurface(SeriesOperator):
    """2D age x cost chip surface statistics.

    Maintains an ``age_bins x price_bins`` grid of turnover-surviving chip mass
    (age advances one trading day, new turnover mass enters age 0, existing mass
    decays by ``(1-turnover)``).  Cost bins are anchored on the first ``window``
    closes.  Outputs:

    * ``surface_entropy``     -- Shannon entropy of the normalised 2D mass.
    * ``age_price_mi``        -- mutual information between age and cost marginals.
    * ``young_profit_mass``   -- mass with age < age_bins/4 and cost below the close.
    * ``old_overhang_mass``   -- mass with age >= age_bins/2 and cost above the close.
    * ``young_overhang_mass`` -- young mass above the close.
    * ``cost_age_slope``      -- linear slope of mean cost bin vs age.

    NaN until ``window`` days of history.
    """

    metadata = _meta(
        "turnover_chip_age_cost_surface",
        "2D 年龄×成本筹码面统计（熵/MI/获利/套牢/成本-年龄斜率）。",
        ["close", "turnover", "window", "price_bins", "age_bins", "output"],
        unit="ratio",
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "price_bins": ParamSpec(dtype=int, min=2),
            "age_bins": ParamSpec(dtype=int, min=2),
            "output": ParamSpec(dtype=str, choices=_SURFACE_OUTPUTS),
        },
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        turnover: pd.DataFrame,
        window: int = 120,
        price_bins: int = 64,
        age_bins: int = 16,
        output: str = "surface_entropy",
        **_: Any,
    ) -> pd.DataFrame:
        window = max(2, int(window))
        price_bins = max(2, int(price_bins))
        age_bins = max(2, int(age_bins))
        result = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
        cv = close.to_numpy(dtype=float)
        tv = turnover.to_numpy(dtype=float)
        for c in range(close.shape[1]):
            col_out = _age_cost_column(cv[:, c], tv[:, c], window, price_bins, age_bins)
            result.iloc[:, c] = col_out[str(output)]
        return result


# ---------------------------------------------------------------------------
# turnover_chip_overhang_surface
# ---------------------------------------------------------------------------
_OVERHANG_OUTPUTS = (
    "overhang_mass",
    "near_overhang_mass",
    "under_price_mass",
    "supply_vacuum",
    "distance_weighted_overhang",
    "nearest_upper_peak",
    "nearest_lower_peak",
    "overhead_supply_ratio",
)


def _overhang_column(
    close: np.ndarray,
    turnover: np.ndarray,
    window: int,
    bins: int,
    decay: float,
):
    n = close.shape[0]
    out = {k: np.full(n, np.nan) for k in _OVERHANG_OUTPUTS}
    pmin, pmax = _anchor_prices(close, window, bins)
    if not np.isfinite(pmin):
        return out
    dist = np.zeros(bins, dtype=float)
    for t in range(n):
        p = close[t]
        u = turnover[t]
        if not np.isfinite(p) or not np.isfinite(u):
            dist = np.zeros(bins, dtype=float)
            continue
        u = float(np.clip(u, 0.0, 1.0))
        dist = dist * (1.0 - u)
        cost_bin = _price_to_bin(p, pmin, pmax, bins)
        dist[cost_bin] += u
        if t < window - 1:
            continue
        bin_prices = np.array([_bin_price(b, pmin, pmax, bins) for b in range(bins)])
        above_mask = bin_prices > p
        below_mask = bin_prices < p
        overhang = float(dist[above_mask].sum())
        under = float(dist[below_mask].sum())
        out["overhang_mass"][t] = overhang
        out["under_price_mass"][t] = under
        near = dist[above_mask & (bin_prices <= p * 1.05)]
        out["near_overhang_mass"][t] = float(near.sum())
        max_mass = float(dist.max())
        peak_thresh = 0.02 * max_mass if max_mass > _EPS else _EPS
        cur_bin = _price_to_bin(p, pmin, pmax, bins)
        # nearest upper / lower peaks
        upper_peaks = [j for j in range(cur_bin + 1, bins) if dist[j] >= peak_thresh]
        lower_peaks = [j for j in range(cur_bin - 1, -1, -1) if dist[j] >= peak_thresh]
        if upper_peaks:
            j = upper_peaks[0]
            out["nearest_upper_peak"][t] = (bin_prices[j] - p) / p
        if lower_peaks:
            j = lower_peaks[0]
            out["nearest_lower_peak"][t] = (p - bin_prices[j]) / p
        # supply vacuum: next upper peak with no mass >= threshold in between
        for j in range(cur_bin + 1, bins):
            if dist[j] >= peak_thresh:
                intervening = dist[cur_bin + 1 : j]
                if intervening.size == 0 or float(np.max(intervening)) < peak_thresh:
                    out["supply_vacuum"][t] = (bin_prices[j] - p) / p
                break
        dwo = 0.0
        for j in range(cur_bin + 1, bins):
            rel = (bin_prices[j] - p) / p if p != 0 else np.nan
            dwo += float(dist[j]) * np.exp(-float(decay) * rel)
        out["distance_weighted_overhang"][t] = dwo
        if under > _EPS:
            out["overhead_supply_ratio"][t] = overhang / under
    return out


@register_operator(
    name="turnover_chip_overhang_surface",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="turnover_chip_overhang_surface",
    source="technical.chip_ops",
)
class TurnoverChipOverhangSurface(SeriesOperator):
    """1D turnover-surviving cost distribution overhang statistics.

    Builds the cost distribution via turnover survival (new turnover mass at
    today's cost bin, prior mass survives by ``(1-turnover)``).  Outputs:

    * ``overhang_mass``             -- mass above the current close.
    * ``near_overhang_mass``        -- mass within 5% above the close.
    * ``under_price_mass``          -- mass below the close.
    * ``supply_vacuum``             -- relative distance to the next upper peak
      (mass >= 2% of max) with no significant mass in between.
    * ``distance_weighted_overhang``-- sum over upper bins of mass*exp(-decay*rel).
    * ``nearest_upper_peak``        -- normalised distance to the nearest upper peak.
    * ``nearest_lower_peak``        -- normalised distance to the nearest lower peak.
    * ``overhead_supply_ratio``     -- overhang / under-price mass.

    NaN until ``window`` days of history.
    """

    metadata = _meta(
        "turnover_chip_overhang_surface",
        "换手存活筹码成本分布上方套牢量/近端供给/供给真空等统计。",
        ["close", "turnover", "window", "bins", "output"],
        unit="ratio",
        param_specs={
            "window": ParamSpec(dtype=int, min=2),
            "bins": ParamSpec(dtype=int, min=2),
            "output": ParamSpec(dtype=str, choices=_OVERHANG_OUTPUTS),
        },
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        turnover: pd.DataFrame,
        window: int = 120,
        bins: int = 128,
        output: str = "overhang_mass",
        **_: Any,
    ) -> pd.DataFrame:
        window = max(2, int(window))
        bins = max(2, int(bins))
        result = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
        cv = close.to_numpy(dtype=float)
        tv = turnover.to_numpy(dtype=float)
        for c in range(close.shape[1]):
            col_out = _overhang_column(cv[:, c], tv[:, c], window, bins, 20.0)
            result.iloc[:, c] = col_out[str(output)]
        return result


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_CANONICALS))


_CANONICALS.extend(["turnover_chip_age_cost_surface", "turnover_chip_overhang_surface"])
_register_surface()
