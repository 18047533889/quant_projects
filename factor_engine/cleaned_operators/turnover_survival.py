# -*- coding: utf-8 -*-
"""Turnover-survival / chip-cost family (2026-08, derived from Gemini CGO ideas).

The family models the *survival* of past traded chips.  Chips bought on day
``t-n`` are still held today with approximate probability equal to the product
of the daily survival probabilities ``e^{-u}`` since then (Poisson replacement
hazard, so >100% turnover needs no clipping):

    w_{t-n} = (1 - e^{-u_{t-n}}) * prod_{j=1}^{n-1} e^{-u_{t-j}}   (raw survival weight)
    w_n     = w_{t-n} / sum_k w_{t-k}                              (normalised)

where ``u`` is the decimal free-float turnover rate.  The reference (average
acquisition) price is the weight-normalised mean of past prices:

    RP_t = sum_n w_n * P_{t-n}

Only strictly-past rows ``[t-W, t-1]`` enter the cost distribution; the current
``P_t`` never contributes to its own reference price.  All six operators share
one per-column kernel so survival weights, cost moments and cost quantiles are
computed once.

Missing-value policy (production contract)
    * NaN turnover is UNKNOWN, never ``u = 0``: a provider gap must not be
      read as a no-churn day, so the row emits NaN (``missing_policy="break"``).
      Genuine suspensions with an explicit 0 turnover still behave as no-churn.
    * NaN price at a lag contributes zero weight (you cannot acquire at a
      missing price) and is dropped before the weighted quantile is computed.
    * Output is NaN while fewer than ``min_periods`` valid prices exist in the
      trailing history, or when the total survival weight collapses to ~0, or
      when the reference price is not positive.
    * Turnover uses a Poisson replacement hazard (``surv = exp(-u)``), so
      >100% turnover is handled without clipping the survival product.

All operators are prefix-causal, trailing-window, and return the same panel
axes as their inputs.

Model parameters (versioned logical-definition metadata)
    * ``_MAX_OLD_MASS = 0.10`` (v3, R3-144): the maximum tolerated residual
      pre-window float mass ``M_old = prod_j exp(-u_j)`` (the Poisson-hazard
      survival of any pre-window holding through the window).  Above this the
      windowed weights describe only a CONDITIONAL distribution of recently
      traded chips, not the full holding distribution — renormalising to 1
      silently erases a large part of the float.  v1 (R6-138) used 0.30; the
      audit found 29% of unknown old chips still dropped+renormalised was a
      strong model assumption, so v2/v3 (R3-144) tightened it to 0.10 and
      exposed ``ts_turnover_old_mass`` as a diagnostic output.  Use a longer
      ``window`` (or the recursive stateful holder-age machine tracked for the
      future) to bring ``M_old`` under the threshold.
    * Negative turnover (``u < 0``) is a DATA ERROR, not a no-churn
      suspension (R3-142): it fails closed (whole row NaN) instead of being
      clamped to 0.
    * A day with POSITIVE turnover but a missing price (R3-143) is "chips
      traded but at an unknown cost": the whole chip distribution that day
      emits NaN rather than dropping that mass and renormalising the rest to
      100%.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12

# Fixed log-price bins (relative to the current price) for the chip-cost shape
# statistics.  Open outer boundaries; the inner edges span ±20% log-price.
_COST_LOG_EDGES = np.array([-0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20])


def _default_min_periods(window: int) -> int:
    """Minimum valid-price lags required; grows with the window."""
    return max(5, window // 10)


# R6-138 / R3-144: a finite window discards every chip acquired BEFORE
# [t-W, t-1].  The fraction of the current float that predates the window is
# the survival product over the window ``M_old = prod_j exp(-u_j)`` (the
# Poisson-hazard survival of any pre-window holding through the window).  When
# that residual old mass is large, the window estimate is a CONDITIONAL
# distribution of recently-traded chips, not the full holding distribution —
# renormalising to 1 silently erases a large part of the float.  Above this
# threshold the estimate is reported as NaN (fail-closed) instead of a
# mislabelled conditional value.  v1 = 0.30 (R6-138) was audited as still too
# loose (29% of unknown old chips were dropped and the rest renormalised);
# tightened to 0.10 in R3-144, with ``ts_turnover_old_mass`` exposing the
# diagnostic.  This is a MODEL PARAMETER (see module docstring, item 145).
_MAX_OLD_MASS = 0.10


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
    near-cost mass, cost-quantile distance, cost-shape statistics).
    """
    rows = price.shape[0]
    ref = np.full(rows, np.nan)
    disp = np.full(rows, np.nan)
    profit = np.full(rows, np.nan)
    age = np.full(rows, np.nan)
    near = np.full(rows, np.nan)
    qdist = np.full(rows, np.nan)
    cost_ent = np.full(rows, np.nan)
    mode_d = np.full(rows, np.nan)
    cost_sk = np.full(rows, np.nan)
    age_disp = np.full(rows, np.nan)
    old_mass = np.full(rows, np.nan)
    cost_ent_v = np.full(rows, np.nan)

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

        # P0-033: a missing turnover is UNKNOWN, never a real zero-turnover
        # day.  A suspension is a genuine no-trade event, but a provider gap is
        # not; since the operator cannot tell them apart it must fail closed and
        # emit NaN rather than read the gap as "no churn".
        if not np.all(np.isfinite(turns)):
            continue
        # R3-142: negative turnover is a DATA ERROR, never a no-churn
        # suspension.  ``maximum(turnover, 0)`` read -0.2 as a no-churn day and
        # silently inverted a data error into a bullish signal; fail closed
        # (whole row NaN) instead of clamping.
        if np.any(turns < 0.0):
            continue
        # R3-143: a day with POSITIVE turnover but a missing price means chips
        # changed hands at an UNKNOWN cost.  Dropping that mass and
        # renormalising the rest to 100% fabricates a full-cost distribution, so
        # the whole chip distribution that day fails closed.
        if np.any((turns > 0.0) & (~price_ok)):
            continue
        # P0-034: Poisson replacement hazard.  A-share turnover routinely
        # exceeds 100% (one float can change hands several times a day);
        # hard-clipping to [0, 1-eps] made any >100% turnover erase every old
        # chip in a single day.  With a Poisson hazard the per-day survival
        # factor is exp(-u): u=1 -> 36.8% survive, u=2 -> 13.5%, u=3 -> 5.0%.
        u = turns
        surv = np.exp(-u)

        # suffix survival: suf[k] = prod_{m=k}^{L-1} exp(-u[m]); suf[L] = 1.
        r = np.cumprod(surv[::-1])
        suf = np.empty(length + 1)
        suf[length] = 1.0
        suf[:length] = r[::-1]
        # raw weight at lag k: fraction replaced that day x still-held since.
        w = (1.0 - surv) * suf[1 : length + 1]
        w = np.where(price_ok, w, 0.0)

        total = float(w.sum())
        if not np.isfinite(total) or total <= _EPS:
            continue
        # R6-138 / R3-144: residual old mass = survival of ANY pre-window
        # holding through the window = suf[0].  If a large fraction of the float
        # predates the window, the normalised weights below describe only the
        # recent conditional distribution — fail closed instead of mislabelling
        # it.  The raw old mass is ALWAYS recorded as a diagnostic (even when
        # the fail-closed threshold trips) so callers can see how much of the
        # float was window-truncated.
        old_mass[t] = float(suf[0])
        if float(suf[0]) > _MAX_OLD_MASS:
            continue
        wn = w / total

        # Zero-weight rows carry NaN prices; ``0 * NaN == NaN`` would poison
        # the weighted mean, so evaluate over a masked price vector.
        rp = float(np.sum(wn * np.where(price_ok, prices, 0.0)))
        if not np.isfinite(rp) or rp <= 0.0:
            continue
        ref[t] = rp

        # cost dispersion: sqrt( sum wn * log(price/rp)^2 )
        with np.errstate(divide="ignore", invalid="ignore"):
            log_dist = np.log(np.where(price_ok, prices, rp) / rp)
        disp[t] = float(np.sqrt(np.sum(wn * log_dist * log_dist)))

        # profit share: weight below current price (strictly cheaper).
        profit[t] = float(np.sum(wn[price_ok & (prices < current)]))

        # holding age: lag n = L - k (days held).
        lags = length - np.arange(length, dtype=float)
        age[t] = float(np.sum(wn * lags))

        # --- deepening cost-shape statistics (P1) ---
        # R6-140: cost entropy must describe the SHAPE of the chip-cost
        # distribution, independent of today's price position.  The old bins
        # were relative to the *current* price, so the entropy mixed "how the
        # costs are spread" with "where the current price sits inside the
        # distribution" (a recipe-level concept, tracked separately by
        # ``ts_turnover_cost_mode_distance``).  Binning around the reference
        # price RP (a pure function of the chip history) makes the entropy a
        # standalone shape primitive.
        with np.errstate(divide="ignore", invalid="ignore"):
            z_cur = np.log(np.where(price_ok, prices, rp) / rp)
        bins_idx = np.digitize(z_cur, _COST_LOG_EDGES)
        nb = _COST_LOG_EDGES.shape[0] + 1
        mass = np.zeros(nb)
        for bb in range(nb):
            mass[bb] = float(np.sum(wn[bins_idx == bb]))
        mtot = float(mass.sum())
        if mtot > _EPS:
            p = mass / mtot
            p = p[p > 0.0]
            cost_ent[t] = -float(np.sum(p * np.log(p))) / np.log(nb)
            bmax = int(np.argmax(mass))
            sel = bins_idx == bmax
            sel_w = wn[sel]
            if float(sel_w.sum()) > _EPS:
                pmode = float(
                    np.sum(wn[sel] * np.where(price_ok, prices, 0.0)[sel]) / float(np.sum(wn[sel]))
                )
                if np.isfinite(pmode) and pmode > 0.0:
                    mode_d[t] = float(np.log(current / pmode))
        # R3-147: volatility-scaled chip-cost entropy — bin on log(P/RP)/sigma
        # with sigma = weighted cost dispersion.  The fixed ±2/5/10/20% bins of
        # the raw ``cost_ent`` are not comparable across differently-volatile
        # stocks (the same absolute log-distance is "tight" for a low-vol name
        # and "wide" for a high-vol name); standardising by the distribution's
        # own dispersion makes the entropy a scale-free shape measure.
        sigma = disp[t]
        if np.isfinite(sigma) and sigma > _EPS:
            z_std = z_cur / sigma
            bins_v = np.digitize(z_std, _COST_LOG_EDGES)
            mass_v = np.zeros(nb)
            for bb in range(nb):
                mass_v[bb] = float(np.sum(wn[bins_v == bb]))
            mtot_v = float(mass_v.sum())
            if mtot_v > _EPS:
                pv = mass_v / mtot_v
                pv = pv[pv > 0.0]
                cost_ent_v[t] = -float(np.sum(pv * np.log(pv))) / np.log(nb)
            else:
                cost_ent_v[t] = 0.0
        else:
            # degenerate cost distribution (every cost == RP): entropy is 0.
            cost_ent_v[t] = 0.0
        # weighted cost skew: third moment of z = log(P/RP) over normalised mass.
        m2 = float(np.sum(wn * log_dist * log_dist))
        m3 = float(np.sum(wn * log_dist * log_dist * log_dist))
        if np.isfinite(m2) and m2 > _EPS:
            cost_sk[t] = float(m3 / (m2 ** 1.5))
        # age dispersion: weighted std of holding lags around the mean age.
        ad = lags - age[t]
        age_disp[t] = float(np.sqrt(np.sum(wn * ad * ad)))

        # near-cost mass: weight within +/- band_pct of the current price.
        rel = np.abs(prices - current) / current
        near[t] = float(np.sum(wn[price_ok & (rel <= band_pct)]))

        # P0-035: weighted cost quantiles must drop zero-weight / non-finite
        # prices first.  Sorting the raw array kept missing prices inside the
        # value space (weight 0 but still interpolated against), which could
        # turn the quantile distance into NaN or a wrong level.
        qmask = price_ok & (wn > 0.0)
        if int(qmask.sum()) < 2:
            continue
        qs = prices[qmask]
        qw = wn[qmask]
        qorder = np.argsort(qs, kind="stable")
        qs = qs[qorder]
        qw = qw[qorder]
        qcdf = np.cumsum(qw)
        qcdf = qcdf / qcdf[-1]
        q = _weighted_quantile(qs, qcdf, (q_low, q_high))
        if np.all(np.isfinite(q)) and (q[1] - q[0]) > 0.0:
            qdist[t] = float((q[1] - q[0]) / rp)

    return {
        "ref": ref,
        "disp": disp,
        "profit": profit,
        "age": age,
        "near": near,
        "qdist": qdist,
        "cost_ent": cost_ent,
        "mode_d": mode_d,
        "cost_sk": cost_sk,
        "age_disp": age_disp,
        "old_mass": old_mass,
        "cost_ent_v": cost_ent_v,
    }


def _weighted_quantile(
    values: np.ndarray, cdf: np.ndarray, quantiles: tuple[float, ...]
) -> np.ndarray:
    """Weighted ECDF-inverse quantiles from a sorted (value, cdf) pair (R3-146).

    ``cdf`` is the cumulative weight (already normalised to end at ~1).  A
    quantile ``q`` maps to the FIRST OBSERVED value whose cumulative weight
    reaches ``q`` — never a linearly interpolated cost that no chip was actually
    acquired at.  Endpoints clamp to ``values[0]`` / ``values[-1]``: ``q <=
    cdf[0]`` maps to ``values[0]`` and ``q >= cdf[-1]`` maps to ``values[-1]``
    (P0-16 / P0-17: no below-first-value extrapolation, no fabricated cost).
    """
    out = np.empty(len(quantiles))
    for i, qq in enumerate(quantiles):
        qq = float(qq)
        if qq <= float(cdf[0]):
            out[i] = float(values[0])
        elif qq >= float(cdf[-1]):
            out[i] = float(values[-1])
        else:
            idx = int(np.searchsorted(cdf, qq, side="left"))
            idx = min(max(idx, 0), values.shape[0] - 1)
            out[i] = float(values[idx])
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
        "ref": [], "disp": [], "profit": [], "age": [], "near": [], "qdist": [],
        "cost_ent": [], "mode_d": [], "cost_sk": [], "age_disp": [],
        "old_mass": [], "cost_ent_v": [],
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
    from factor_engine.cleaned_operators.base import OperatorMetadata

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

    R6-139 (window-conditioned, documented honestly): a finite ``window``
    discards every acquisition before ``[t-W, t-1]``, so the age is the
    conditional mean over the *windowed* chip cohorts.  When the residual
    pre-window mass ``M_old`` is small the windowed estimate equals the true
    age; when ``M_old`` is large (e.g. an average holding age far beyond
    ``window``), the estimate is reported as NaN (R6-138 fail-closed) rather
    than a truncated number that understates the real holding period.  The
    semantic is therefore ``window-conditioned turnover holding age``; a fully
    recursive holder-age state machine would lift the truncation and is tracked
    as a future stateful operator.
    """

    metadata = _metadata(
        "ts_turnover_holding_age",
        "存活筹码加权平均持仓天数（window 条件化，见 R6-139）。",
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


@register_operator(
    name="ts_turnover_cost_entropy",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_cost_entropy",
    source="turnover_survival",
)
class TsTurnoverCostEntropy(SeriesOperator):
    """存活筹码成本分布的熵（固定 log-price 分箱，相对当前价）。

    ``H = -Σ_b p_b log p_b / log B``（B=9 个固定箱：8 条内边界产生 9 个区间，
    越界并到开区间箱）。低 =
    筹码高度集中在少数成本区（单一密集成本带）；高 = 筹码成本高度分散。与
    ``ts_turnover_cost_dispersion``（只看二阶尺度）互补——entropy 看整个质量
    分布形状。只用 t-1 及以前，PIT 安全。
    """

    metadata = _metadata(
        "ts_turnover_cost_entropy",
        "筹码成本分布熵（[0,1]，低=集中单一成本带）。",
        ["price", "turnover", "window"],
        unit="entropy",
    )

    def _calculate_series(self, price: pd.DataFrame, turnover: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, 0.75, 0.25)["cost_ent"]


@register_operator(
    name="ts_turnover_cost_mode_distance",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_cost_mode_distance",
    source="turnover_survival",
)
class TsTurnoverCostModeDistance(SeriesOperator):
    """最大筹码成本峰的相对位置 ``log(P_t / P_mode)``。

    在固定 log-price 分箱上找权重最大的成本峰，取该箱内筹码的加权平均成本
    P_mode；输出相对当前价的对数距离。正 = 最大筹码峰在现价下方（多数被套的
    密集成本带），负 = 峰在上方。比参考价 mean/median 更贴近"最大筹码峰在哪
    "，A 股尤有解释力。PIT 安全。
    """

    metadata = _metadata(
        "ts_turnover_cost_mode_distance",
        "最大成本峰相对当前价 log(P_t/P_mode)。",
        ["price", "turnover", "window"],
        unit="log",
    )

    def _calculate_series(self, price: pd.DataFrame, turnover: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, 0.75, 0.25)["mode_d"]


@register_operator(
    name="ts_turnover_cost_skew",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_cost_skew",
    source="turnover_survival",
)
class TsTurnoverCostSkew(SeriesOperator):
    """存活筹码对数成本的加权三阶矩（偏度）。

    ``Skew = Σ_n w_n z_n^3 / (Σ_n w_n z_n^2)^1.5``，``z_n = log(P_{t-n}/RP_t)``。
    同样 dispersion 下区分筹码主要拖在上方（z 右尾大 → 正偏）还是下方。PIT 安全。
    """

    metadata = _metadata(
        "ts_turnover_cost_skew",
        "筹码成本分布加权偏度（正=拖在上方）。",
        ["price", "turnover", "window"],
        unit="ratio",
    )

    def _calculate_series(self, price: pd.DataFrame, turnover: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, 0.75, 0.25)["cost_sk"]


@register_operator(
    name="ts_turnover_age_dispersion",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_age_dispersion",
    source="turnover_survival",
)
class TsTurnoverAgeDispersion(SeriesOperator):
    """存活筹码持仓年龄的加权离散度 ``sqrt(Σ w_n (n - Age)^2)``。

    平均持仓年龄相同的两只股票：A 全部筹码约 40 天、B 一半 5 天一半 75 天，
    AgeDisp 完全不同。与 ``ts_turnover_holding_age``（均值）互补。PIT 安全。
    """

    metadata = _metadata(
        "ts_turnover_age_dispersion",
        "存活筹码持仓年龄加权离散度（天）。",
        ["price", "turnover", "window"],
        unit="days",
    )

    def _calculate_series(self, price: pd.DataFrame, turnover: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, 0.75, 0.25)["age_disp"]


@register_operator(
    name="ts_turnover_old_mass",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_old_mass",
    source="turnover_survival",
)
class TsTurnoverOldMass(SeriesOperator):
    """Residual pre-window float mass ``M_old = prod_j exp(-u_j)`` (diagnostic).

    A finite ``window`` discards every acquisition before ``[t-W, t-1]``; the
    fraction of the current float that predates the window is the Poisson-hazard
    survival of any pre-window holding through the window.  When ``M_old`` is
    large the windowed estimates are CONDITIONAL on recently-traded chips, not
    the full holding distribution — ``_MAX_OLD_MASS`` (model parameter, R3-145)
    fail-closes them.  This operator exposes ``M_old`` itself so a caller can
    see exactly how much of the float was window-truncated instead of guessing.
    """

    metadata = _metadata(
        "ts_turnover_old_mass",
        "窗口外残留浮筹质量 M_old = prod exp(-u)（诊断输出）。",
        ["price", "turnover", "window"],
        unit="ratio",
    )

    def _calculate_series(self, price: pd.DataFrame, turnover: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, 0.75, 0.25)["old_mass"]


@register_operator(
    name="ts_turnover_cost_entropy_vol_scaled",
    category="time_series_chip_cost",
    business_category="time_series_chip_cost",
    canonical="ts_turnover_cost_entropy_vol_scaled",
    source="turnover_survival",
)
class TsTurnoverCostEntropyVolScaled(SeriesOperator):
    """波动率标度的筹码成本熵（按 log(P/RP)/sigma 分箱，R3-147）。

    The raw ``ts_turnover_cost_entropy`` bins on the ABSOLUTE log-price distance
    (fixed ±2/5/10/20% edges), so the same absolute distance is "tight" for a
    low-volatility name and "wide" for a high-volatility one — the fixed bins
    are incomparable across differently-volatile stocks.  This variant
    standardises ``z = log(P/RP)`` by the weighted cost dispersion
    ``sigma = sqrt(sum wn * z^2)`` (the distribution's own scale) and bins on
    ``z/sigma``, making the entropy a scale-free shape measure.  A degenerate
    cost distribution (``sigma ~ 0``, every cost == RP) has entropy exactly 0.
    """

    metadata = _metadata(
        "ts_turnover_cost_entropy_vol_scaled",
        "波动率标度筹码成本熵（log(P/RP)/disp 分箱，跨波动率可比）。",
        ["price", "turnover", "window"],
        unit="entropy",
    )

    def _calculate_series(self, price: pd.DataFrame, turnover: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        return _run_all(price, turnover, window, 0.05, 0.75, 0.25)["cost_ent_v"]


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_turnover_reference_price",
            "ts_turnover_cost_dispersion",
            "ts_turnover_profit_share",
            "ts_turnover_holding_age",
            "ts_turnover_near_cost_mass",
            "ts_turnover_cost_quantile_distance",
            "ts_turnover_cost_entropy",
            "ts_turnover_cost_mode_distance",
            "ts_turnover_cost_skew",
            "ts_turnover_age_dispersion",
            "ts_turnover_old_mass",
            "ts_turnover_cost_entropy_vol_scaled",
        })
    from factor_engine.cleaned_operators.rolling_pack import register_polars_bridge

    for _canon in (
        "ts_turnover_reference_price",
        "ts_turnover_cost_dispersion",
        "ts_turnover_profit_share",
        "ts_turnover_holding_age",
        "ts_turnover_near_cost_mass",
        "ts_turnover_cost_quantile_distance",
        "ts_turnover_cost_entropy",
        "ts_turnover_cost_mode_distance",
        "ts_turnover_cost_skew",
        "ts_turnover_age_dispersion",
        "ts_turnover_old_mass",
        "ts_turnover_cost_entropy_vol_scaled",
    ):
        register_polars_bridge(_canon)


_register_surface()
