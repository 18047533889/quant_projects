# -*- coding: utf-8 -*-
"""Turnover-survival / chip-cost family (2026-08, derived from Gemini CGO ideas).

The family models the *survival* of past traded chips.  Chips bought on day
``t-n`` are still held today with approximate probability equal to the product
of the daily no-turnover probabilities since then:

    w_{t-n} = u_{t-n} * prod_{j=1}^{n-1} (1 - u_{t-j})          (raw survival weight)
    w_n     = w_{t-n} / sum_k w_{t-k}                            (normalised)

where ``u`` is the decimal free-float turnover rate.  The reference (average
acquisition) price is the weight-normalised mean of past prices:

    RP_t = sum_n w_n * P_{t-n}

Only strictly-past rows ``[t-W, t-1]`` enter the cost distribution; the current
``P_t`` never contributes to its own reference price.  All six operators share
one per-column kernel so survival weights, cost moments and cost quantiles are
computed once.

Missing-value policy (production contract)
    * NaN turnover is treated as ``u = 0`` (suspension / no-trade day: held
      chips do not turn over, and no new chips are created).
    * NaN price at a lag contributes zero weight (you cannot acquire at a
      missing price).
    * Output is NaN while fewer than ``min_periods`` valid prices exist in the
      trailing history, or when the total survival weight collapses to ~0, or
      when the reference price is not positive.
    * Turnover is clipped to ``[0, 1-eps]`` so the survival product never
      divides by zero when turnover approaches 100%.

All operators are prefix-causal, trailing-window, and return the same panel
axes as their inputs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _default_min_periods(window: int) -> int:
    """Minimum valid-price lags required; grows with the window."""
    return max(5, window // 10)


def _column_stats(
    price: np.ndarray,
    turnover: np.ndarray,
    window: int,
    min_periods: int,
    band_pct: float,
    q_high: float,
    q_low: float,
) -> dict[str, np.ndarray]:
    """Compute all chip-cost statistics for one column, prefix-causal.

    ``price`` / ``turnover`` are 1-D arrays of equal length.  Returns a dict of
    arrays (reference price, cost dispersion, profit share, holding age,
    near-cost mass, cost-quantile distance).
    """
    rows = price.shape[0]
    ref = np.full(rows, np.nan)
    disp = np.full(rows, np.nan)
    profit = np.full(rows, np.nan)
    age = np.full(rows, np.nan)
    near = np.full(rows, np.nan)
    qdist = np.full(rows, np.nan)

    for t in range(rows):
        lo = max(0, t - window)
        prices = price[lo:t]           # strictly past [t-W, t-1]
        turns = turnover[lo:t]
        length = prices.shape[0]
        if length < 1:
            continue
        price_ok = np.isfinite(prices)
        n_valid = int(price_ok.sum())
        if n_valid < min_periods:
            continue
        current = price[t]
        if not np.isfinite(current) or current <= 0.0:
            continue

        # NaN turnover -> 0 (suspension); clip to [0, 1-eps].
        u = np.where(np.isfinite(turns), np.clip(turns, 0.0, 1.0 - _EPS), 0.0)
        one_minus = 1.0 - u

        # suffix survival: suf[k] = prod_{m=k}^{L-1}(1-u[m]); suf[L] = 1.
        r = np.cumprod(one_minus[::-1])
        suf = np.empty(length + 1)
        suf[length] = 1.0
        suf[:length] = r[::-1]
        # raw weight at lag k: u[k] * suf[k+1]
        w = u * suf[1 : length + 1]
        w = np.where(price_ok, w, 0.0)

        total = float(w.sum())
        if not np.isfinite(total) or total <= _EPS:
            continue
        wn = w / total

        rp = float(np.sum(wn * prices))
        if not np.isfinite(rp) or rp <= 0.0:
            continue
        ref[t] = rp

        # cost dispersion: sqrt( sum wn * log(price/rp)^2 )
        log_dist = np.log(np.where(price_ok, prices, rp) / rp)
        disp[t] = float(np.sqrt(np.sum(wn * log_dist * log_dist)))

        # profit share: weight below current price (strictly cheaper).
        profit[t] = float(np.sum(wn[price_ok & (prices < current)]))

        # holding age: lag n = L - k (days held).
        lags = length - np.arange(length, dtype=float)
        age[t] = float(np.sum(wn * lags))

        # near-cost mass: weight within +/- band_pct of the current price.
        rel = np.abs(prices - current) / current
        near[t] = float(np.sum(wn[price_ok & (rel <= band_pct)]))

        # weighted cost quantiles.
        order = np.argsort(prices)
        cs = np.sort(prices)
        cw = wn[order]
        cdf = np.cumsum(cw)
        q = _weighted_quantile(cs, cdf, (q_low, q_high))
        if np.all(np.isfinite(q)) and (q[1] - q[0]) > 0.0:
            qdist[t] = float((q[1] - q[0]) / rp)

    return {
        "ref": ref,
        "disp": disp,
        "profit": profit,
        "age": age,
        "near": near,
        "qdist": qdist,
    }


def _weighted_quantile(
    values: np.ndarray, cdf: np.ndarray, quantiles: tuple[float, ...]
) -> np.ndarray:
    """Linear-interpolated weighted quantiles from a sorted (value, cdf) pair.

    ``cdf`` is the cumulative weight (already normalised to end at ~1).  A
    quantile ``q`` maps to the value where the ECDF crosses ``q``; endpoints
    clamp to ``values[0]`` / ``values[-1]``.
    """
    out = np.empty(len(quantiles))
    for i, qq in enumerate(quantiles):
        qq = float(qq)
        if qq <= 0.0:
            out[i] = values[0]
        elif qq >= 1.0:
            out[i] = values[-1]
        else:
            idx = int(np.searchsorted(cdf, qq, side="left"))
            idx = min(max(idx, 1), values.shape[0] - 1)
            hi_cdf = float(cdf[idx])
            lo_cdf = float(cdf[idx - 1])
            span = hi_cdf - lo_cdf
            if span <= _EPS:
                out[i] = float(values[idx])
            else:
                frac = (qq - lo_cdf) / span
                out[i] = float(values[idx - 1] + frac * (values[idx] - values[idx - 1]))
    return out


def _run_all(
    price: pd.DataFrame,
    turnover: pd.DataFrame,
    window: int,
    band_pct: float,
    q_high: float,
    q_low: float,
) -> dict[str, pd.DataFrame]:
    """Run the shared kernel over every column; return one panel per statistic."""
    w = max(2, int(window))
    band = max(0.0, float(band_pct))
    qh = float(q_high)
    ql = float(q_low)
    if not 0.0 < ql < qh < 1.0:
        raise ValueError("need 0 < q_low < q_high < 1")
    mp = _default_min_periods(w)

    pv = price.to_numpy(dtype=float)
    tv = turnover.to_numpy(dtype=float)
    rows, cols = pv.shape
    out: dict[str, list[np.ndarray]] = {
        "ref": [], "disp": [], "profit": [], "age": [], "near": [], "qdist": []
    }
    for c in range(cols):
        stats = _column_stats(pv[:, c], tv[:, c], w, mp, band, qh, ql)
        for key in out:
            out[key].append(stats[key])
    return {
        key: frame_like(price, np.column_stack(arrays))
        for key, arrays in out.items()
    }


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> Any:
    from cleaned_operators.base import OperatorMetadata

    return OperatorMetadata(
        name=name,
        category="time_series_chip_cost",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_chip_cost", "daily", "pit_safe", "causal", "typed_v2",
            "turnover_survival", f"signature:{','.join(params)}->series",
            f"unit:{unit}", "cost:1",
        ],
    )


@register_operator(
    name="ts_turnover_reference_price",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_reference_price",
    source="turnover_survival",
)
class TsTurnoverReferencePrice(SeriesOperator):
    """Weighted average acquisition price of surviving chips.

    ``RP_t = sum_n w_n * P_{t-n}`` with normalised turnover-survival weights
    over the strictly-past window ``[t-W, t-1]`` (the current price never enters
    its own reference cost).  A high RP relative to the current price means a
    large fraction of holders are underwater.  This is the building block for
    the classic capital-gains-overhang recipe
    ``capital_gains_overhang = (P_t - RP_t) / P_t``.
    """

    metadata = _metadata(
        "ts_turnover_reference_price",
        "换手存活加权平均持仓成本(仅用 t-1 及以前)。",
        ["price", "turnover", "window"],
        unit="price",
    )

    def _calculate_series(
        self, price: pd.DataFrame, turnover: pd.DataFrame, window: int = 60, **_: Any
    ) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, 0.75, 0.25)["ref"]


@register_operator(
    name="ts_turnover_cost_dispersion",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_cost_dispersion",
    source="turnover_survival",
)
class TsTurnoverCostDispersion(SeriesOperator):
    """Log-cost dispersion of the surviving chip distribution.

    ``Dispersion_t = sqrt( sum_n w_n * log(P_{t-n}/RP_t)^2 )``.  Low values mean
    holder costs are concentrated near the reference price; high values mean
    the cost base is spread widely across prices.
    """

    metadata = _metadata(
        "ts_turnover_cost_dispersion",
        "换手存活筹码的对数成本离散度。",
        ["price", "turnover", "window"],
        unit="ratio",
    )

    def _calculate_series(
        self, price: pd.DataFrame, turnover: pd.DataFrame, window: int = 60, **_: Any
    ) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, 0.75, 0.25)["disp"]


@register_operator(
    name="ts_turnover_profit_share",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_profit_share",
    source="turnover_survival",
)
class TsTurnoverProfitShare(SeriesOperator):
    """Share of surviving chips acquired below the current price.

    ``ProfitShare_t = sum_n w_n * 1(P_{t-n} < P_t)``.  Approximates the fraction
    of current holders that are in profit; complements the capital-gains
    overhang (which is the *size* of the average gain, not the breadth).
    """

    metadata = _metadata(
        "ts_turnover_profit_share",
        "获利筹码占比: 成本低于当前价的存活筹码权重比例。",
        ["price", "turnover", "window"],
        unit="ratio",
    )

    def _calculate_series(
        self, price: pd.DataFrame, turnover: pd.DataFrame, window: int = 60, **_: Any
    ) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, 0.75, 0.25)["profit"]


@register_operator(
    name="ts_turnover_holding_age",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_holding_age",
    source="turnover_survival",
)
class TsTurnoverHoldingAge(SeriesOperator):
    """Weighted average holding age (in trading days) of surviving chips.

    ``Age_t = sum_n w_n * n`` where ``n`` is the lag of each chip cohort.  A
    low age means the current float is dominated by recently-traded chips (fast
    rotation); a high age means a long-held, stale base.
    """

    metadata = _metadata(
        "ts_turnover_holding_age",
        "存活筹码加权平均持仓天数。",
        ["price", "turnover", "window"],
        unit="days",
    )

    def _calculate_series(
        self, price: pd.DataFrame, turnover: pd.DataFrame, window: int = 60, **_: Any
    ) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, 0.75, 0.25)["age"]


@register_operator(
    name="ts_turnover_near_cost_mass",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_near_cost_mass",
    source="turnover_survival",
)
class TsTurnoverNearCostMass(SeriesOperator):
    """Mass of surviving chips whose cost sits within ``band`` of the current price.

    ``NearMass_t = sum_n w_n * 1(|P_{t-n} - P_t| <= band * P_t)``.  High values
    indicate a dense supply/demand zone right around the price (support /
    resistance from concentrated holder costs).
    """

    metadata = _metadata(
        "ts_turnover_near_cost_mass",
        "当前价格附近 ±band 内的存活筹码密度。",
        ["price", "turnover", "window", "band"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        price: pd.DataFrame,
        turnover: pd.DataFrame,
        window: int = 60,
        band: float = 0.05,
        **_: Any,
    ) -> pd.DataFrame:
        return _run_all(price, turnover, window, band, 0.75, 0.25)["near"]


@register_operator(
    name="ts_turnover_cost_quantile_distance",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_cost_quantile_distance",
    source="turnover_survival",
)
class TsTurnoverCostQuantileDistance(SeriesOperator):
    """Relative distance between two weighted cost quantiles.

    ``QDist_t = (Q_{q_high} - Q_{q_low}) / RP_t`` where the quantiles are taken
    over the surviving-chip cost distribution.  A small distance means the cost
    base is tight (all holders acquired in a narrow price band); a large
    distance means heavily dispersed holder costs.
    """

    metadata = _metadata(
        "ts_turnover_cost_quantile_distance",
        "存活筹码成本分位间距 (Q_high-Q_low)/RP。",
        ["price", "turnover", "window", "q_high", "q_low"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        price: pd.DataFrame,
        turnover: pd.DataFrame,
        window: int = 60,
        q_high: float = 0.75,
        q_low: float = 0.25,
        **_: Any,
    ) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, q_high, q_low)["qdist"]


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "ts_turnover_reference_price",
            "ts_turnover_cost_dispersion",
            "ts_turnover_profit_share",
            "ts_turnover_holding_age",
            "ts_turnover_near_cost_mass",
            "ts_turnover_cost_quantile_distance",
        }
    )
    from cleaned_operators.rolling_pack import register_polars_bridge

    for _canon in (
        "ts_turnover_reference_price",
        "ts_turnover_cost_dispersion",
        "ts_turnover_profit_share",
        "ts_turnover_holding_age",
        "ts_turnover_near_cost_mass",
        "ts_turnover_cost_quantile_distance",
    ):
        register_polars_bridge(_canon)


_register_surface()
