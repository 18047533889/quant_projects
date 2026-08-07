# -*- coding: utf-8 -*-
"""Order-flow → price-impact microstructure primitives (minute → daily, 2026-08).

A thin but *honest* layer between raw minute bars and daily cross-sectional
factors.  None of these fabricates a tick-level signed flow it does not have:

* ``intraday_bvc_imbalance``        — BV-C (Bulk Volume Classification) signed
  flow imbalance, ``sum V_m*(2*Phi(z_m)-1) / sum V_m`` with ``z_m`` a scaled
  minute log-return.  Limit-locked bars default to *neutral* (zero flow); they
  are never force-classified as 100% buy / 100% sell.
* ``intraday_impact_beta``          — per-day price-impact regression slope
  ``r_m = alpha + lambda * q_m`` over a supplied signed-flow series.
* ``intraday_impact_asymmetry``     — buy-impact vs sell-impact asymmetry,
  ``(lambda_plus - |lambda_minus|) / (|lambda_plus| + |lambda_minus|)``.
* ``intraday_return_wasserstein_shift`` — Wasserstein-1 distance between today's
  minute-return distribution and the strictly-past multi-day history,
  standardised by the historical MAD.  Distinct from the intraday *volume
  profile* EMD (time-axis activity) — this measures return-distribution shift.
* ``micro_bvc_vpin``                — VPIN built from BV-C flow over
  equal-volume buckets, splitting boundary bars proportionally by volume.

All operators are minute-frequency-input, one-scalar-per-(date, symbol) output,
prefix-causal, and never look past the current row / day.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.microstructure.intraday_agg import (
    _as_panel,
    _daily_agg,
    _daily_agg_two,
    _log_returns,
)

_EPS = 1e-12
_PHI_GRID = np.linspace(0.0, 1.0, 512)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    return norm.cdf(x)


def _rolling_scale(returns: np.ndarray, scale_window: int, min_periods: int) -> np.ndarray:
    """Trailing-window std of returns (reset per day) used as the BVC scale."""
    n = returns.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - scale_window + 1)
        seg = returns[lo : i + 1]
        valid = seg[np.isfinite(seg)]
        if valid.size < min_periods:
            continue
        out[i] = float(np.std(valid, ddof=0))
    return out


def _bvc_flow(
    close_vals: np.ndarray,
    volume_vals: np.ndarray,
    locked: np.ndarray | None,
    scale_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """BV-C signed flow and volume arrays for one (instrument, day).

    Returns ``(of, vol)`` aligned arrays.  NaN return / missing scale / locked
    bars emit zero flow (neutral), never a forced direction.
    """
    n = close_vals.shape[0]
    r = _log_returns(close_vals)
    scale = _rolling_scale(r, max(2, int(scale_window)), max(2, int(scale_window) // 2))
    z = np.where(np.isfinite(r) & np.isfinite(scale), r / (scale + _EPS), 0.0)
    p = _normal_cdf(z)
    of = volume_vals * (2.0 * p - 1.0)
    if locked is not None:
        lk = np.asarray(locked, dtype=float)
        neutral = (np.isfinite(lk) & (lk != 0.0)) | ~np.isfinite(of)
        of = np.where(neutral, 0.0, of)
    of = np.where(np.isfinite(of), of, 0.0)
    vol = np.where(np.isfinite(volume_vals), volume_vals, 0.0)
    return of, vol


def _wasserstein_shift_series(
    daily_returns: list[np.ndarray], lookback_days: int
) -> np.ndarray:
    """Per-instrument daily Wasserstein-shift over a list of per-day arrays."""
    lb = max(2, int(lookback_days))
    rows = len(daily_returns)
    out = np.full(rows, np.nan)
    for i in range(rows):
        today = np.asarray(daily_returns[i], dtype=float)
        parts = [np.asarray(daily_returns[j], dtype=float) for j in range(max(0, i - lb), i)]
        parts = [p for p in parts if p.size]
        if today.size < 5 or not parts:
            out[i] = np.nan
            continue
        hist = np.concatenate(parts)
        if hist.size < 30:
            out[i] = np.nan
            continue
        mad = float(np.median(np.abs(hist - np.median(hist))))
        q_hist = np.quantile(hist, _PHI_GRID)
        q_today = np.quantile(today, _PHI_GRID)
        out[i] = float(np.mean(np.abs(q_today - q_hist))) / (mad + _EPS)
    return out


def _vpin_series(
    daily_close: list[np.ndarray],
    daily_volume: list[np.ndarray],
    scale_window: int,
    bucket_count: int,
) -> np.ndarray:
    """Per-instrument daily VPIN over lists of per-day close/volume arrays."""
    sw = max(2, int(scale_window))
    buckets = max(2, int(bucket_count))
    rows = len(daily_close)
    out = np.full(rows, np.nan)
    for i in range(rows):
        cv = np.asarray(daily_close[i], dtype=float)
        vv = np.asarray(daily_volume[i], dtype=float)
        if cv.size == 0 or not np.any(np.isfinite(cv)):
            out[i] = np.nan
            continue
        of, vol = _bvc_flow(cv, vv, None, sw)
        total_v = float(vol.sum())
        if total_v <= _EPS:
            out[i] = np.nan
            continue
        target = total_v / buckets
        acc = 0.0
        of_bucket = 0.0
        abs_of = 0.0
        finished = False
        for m in range(vol.shape[0]):
            vm = float(vol[m])
            if vm <= 0.0:
                continue
            ofm = float(of[m])
            if acc + vm > target and acc > 0.0 and not finished:
                first_share = (target - acc) / vm
                abs_of += abs(ofm * first_share) + abs(ofm * (1.0 - first_share))
                acc = vm * (1.0 - first_share)
                finished = True
                of_bucket = 0.0
            else:
                acc += vm
                of_bucket += ofm
                if acc >= target and not finished:
                    abs_of += abs(of_bucket)
                    acc = 0.0
                    of_bucket = 0.0
        abs_of += abs(of_bucket)
        out[i] = abs_of / total_v
    return out


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday", "daily_agg", "minute", "pit_safe", "causal", "typed_v2",
            "source_blocked", f"signature:{','.join(params)}->series",
            f"unit:{unit}", "cost:1",
        ],
    )


@register_operator(
    name="intraday_bvc_imbalance",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_bvc_imbalance",
    source="microstructure.flow_impact",
)
class IntradayBvcImbalance(SeriesOperator):
    """BV-C signed order-flow imbalance, one scalar per (date, symbol).

    ``z_m = r_m / (sigma_m + eps)`` (``sigma_m`` a trailing minute scale),
    ``p_m = Phi(z_m)``, ``OF_m = V_m*(2*p_m - 1)`` and
    ``BVCI = sum(OF) / sum(V)`` over the day (range roughly ``[-1, 1]``).
    ``locked`` is an optional 0/1 panel: bars marked locked are classified
    neutral (zero flow) — a price resting at the limit is never assumed to be
    all-buy or all-sell without tick-level trade direction.
    """

    metadata = _metadata(
        "intraday_bvc_imbalance",
        "BV-C 成交量分类买卖流不平衡 (sum V(2Phi(z)-1)/sum V)。",
        ["close", "volume", "scale_window", "locked"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        scale_window: int = 20,
        locked: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        close, volume = _as_panel(close), _as_panel(volume)
        locked_panel = _as_panel(locked) if isinstance(locked, pd.DataFrame) else None
        sw = max(2, int(scale_window))
        out: dict[str, pd.Series] = {}

        for inst in close.columns:
            c = close[inst].to_numpy(dtype=float)
            v = volume[inst].to_numpy(dtype=float)
            lk = None
            if locked_panel is not None and inst in locked_panel.columns:
                lk = locked_panel[inst].to_numpy(dtype=float)
            idx = close[inst].index
            per_day: dict[pd.Timestamp, float] = {}
            for day, group_idx in pd.Series(np.arange(len(idx)), index=idx).groupby(idx.normalize()):
                positions = np.asarray(group_idx, dtype=int)
                if positions.size == 0:
                    continue
                cv = c[positions]
                vv = v[positions]
                lv = lk[positions] if lk is not None else None
                if not np.any(np.isfinite(cv)):
                    per_day[day] = np.nan
                    continue
                of, vol = _bvc_flow(cv, vv, lv, sw)
                total_v = float(vol.sum())
                if total_v <= _EPS:
                    per_day[day] = np.nan
                else:
                    per_day[day] = float(of.sum()) / total_v
            out[inst] = pd.Series(per_day, dtype=float)
        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


@register_operator(
    name="intraday_impact_beta",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_impact_beta",
    source="microstructure.flow_impact",
)
class IntradayImpactBeta(SeriesOperator):
    """Per-day price-impact regression slope ``lambda``.

    Regresses minute returns on a supplied signed flow ``q``
    (``r_m = alpha + lambda*q_m + eps``) and returns ``lambda`` per
    (date, symbol).  The flow can come from BV-C classification
    (``intraday_bvc_imbalance``) or any other order-flow estimator; the
    regression primitive itself stays flow-agnostic.
    """

    metadata = _metadata(
        "intraday_impact_beta",
        "日内价格冲击回归斜率 lambda (r = alpha + lambda*q)。",
        ["returns", "flow", "min_periods"],
        unit="impact",
    )

    def _calculate_series(
        self,
        returns: pd.DataFrame,
        flow: pd.DataFrame,
        min_periods: int = 20,
        **_: Any,
    ) -> pd.DataFrame:
        mp = max(5, int(min_periods))

        def _fn(r_vals: np.ndarray, q_vals: np.ndarray) -> float:
            finite = np.isfinite(r_vals) & np.isfinite(q_vals)
            r = r_vals[finite].astype(float)
            q = q_vals[finite].astype(float)
            if r.size < mp:
                return np.nan
            qc = q - q.mean()
            var = float(np.sum(qc * qc))
            if var <= _EPS:
                return np.nan
            return float(np.sum(qc * (r - r.mean())) / var)

        return _daily_agg_two(returns, flow, _fn)


@register_operator(
    name="intraday_impact_asymmetry",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_impact_asymmetry",
    source="microstructure.flow_impact",
)
class IntradayImpactAsymmetry(SeriesOperator):
    """Buy vs sell price-impact asymmetry.

    ``lambda_plus`` is the impact slope on the positive-flow subsample and
    ``lambda_minus`` the slope on the negative-flow subsample; the output is
    ``(lambda_plus - |lambda_minus|) / (|lambda_plus| + |lambda_minus| + eps)``.
    A positive value means buying pressure moves price more (thin sell side); a
    negative value means selling pressure dominates.
    """

    metadata = _metadata(
        "intraday_impact_asymmetry",
        "买卖冲击不对称 (lambda_+ - |lambda_-|)/(|lambda_+|+|lambda_-|)。",
        ["returns", "flow", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        returns: pd.DataFrame,
        flow: pd.DataFrame,
        min_periods: int = 20,
        **_: Any,
    ) -> pd.DataFrame:
        mp = max(5, int(min_periods))
        side_min = max(3, mp // 3)

        def _slope(r: np.ndarray, q: np.ndarray) -> float | None:
            if r.size < side_min:
                return None
            qc = q - q.mean()
            var = float(np.sum(qc * qc))
            if var <= _EPS:
                return None
            return float(np.sum(qc * (r - r.mean())) / var)

        def _fn(r_vals: np.ndarray, q_vals: np.ndarray) -> float:
            finite = np.isfinite(r_vals) & np.isfinite(q_vals)
            r = r_vals[finite].astype(float)
            q = q_vals[finite].astype(float)
            if r.size < max(mp, side_min):
                return np.nan
            pos = q > 0.0
            neg = q < 0.0
            lam_plus = _slope(r[pos], q[pos])
            lam_minus = _slope(r[neg], q[neg])
            if lam_plus is None or lam_minus is None:
                return np.nan
            denom = abs(lam_plus) + abs(lam_minus) + _EPS
            return float((lam_plus - abs(lam_minus)) / denom)

        return _daily_agg_two(returns, flow, _fn)


@register_operator(
    name="intraday_return_wasserstein_shift",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_return_wasserstein_shift",
    source="microstructure.flow_impact",
)
class IntradayReturnWassersteinShift(SeriesOperator):
    """Wasserstein-1 shift between today's return distribution and the past.

    ``F_t`` is today's minute-return ECDF, ``F_hist`` the ECDF of the strictly
    past ``lookback_days`` trading days.  The output is
    ``W1(F_t, F_hist) / (MAD(F_hist) + eps)``, where ``W1`` is approximated on a
    fine quantile grid.  This is the return-distribution analogue of the volume
    *profile* EMD: it detects a shift in how returns are spread out, not in
    where trading activity sits.
    """

    metadata = _metadata(
        "intraday_return_wasserstein_shift",
        "当日分钟收益分布相对过去 N 日的 W1 距离(MAD 标准化)。",
        ["returns", "lookback_days"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        returns: pd.DataFrame,
        lookback_days: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        returns = _as_panel(returns)
        out: dict[str, pd.Series] = {}
        for inst in returns.columns:
            col = returns[inst]
            day_list: list[pd.Timestamp] = []
            day_returns: list[np.ndarray] = []
            for day, group in col.groupby(col.index.normalize()):
                vals = np.asarray(group, dtype=float)
                vals = vals[np.isfinite(vals)]
                day_list.append(day)
                day_returns.append(vals)
            values = _wasserstein_shift_series(day_returns, lookback_days)
            out[inst] = pd.Series(
                {d: v for d, v in zip(day_list, values)}, dtype=float
            )
        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


@register_operator(
    name="micro_bvc_vpin",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_bvc_vpin",
    source="microstructure.flow_impact",
)
class MicroBvcVpin(SeriesOperator):
    """VPIN built from BV-C flow over equal-volume buckets.

    The day's minute bars are aggregated into ``bucket_count`` equal-volume
    buckets; a bar that straddles a bucket boundary is split proportionally by
    volume, so the boundary minute never sits entirely on one side.  The output
    is ``sum_b |OF_b| / total_volume`` — the conventional VPIN range ``[0, 1]``.
    P2 / research-only: BV-C carries estimation error, minute bars are not ticks
    and the VPIN literature itself is contested.
    """

    metadata = _metadata(
        "micro_bvc_vpin",
        "BV-C 等量桶 VPIN (sum|OF_b|/total_volume), 边界分钟按量切分。",
        ["close", "volume", "scale_window", "bucket_count"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        scale_window: int = 40,
        bucket_count: int = 20,
        **_: Any,
    ) -> pd.DataFrame:
        close, volume = _as_panel(close), _as_panel(volume)
        out: dict[str, pd.Series] = {}

        for inst in close.columns:
            c = close[inst].to_numpy(dtype=float)
            v = volume[inst].to_numpy(dtype=float)
            idx = close[inst].index
            day_list: list[pd.Timestamp] = []
            day_close: list[np.ndarray] = []
            day_volume: list[np.ndarray] = []
            for day, group_idx in pd.Series(np.arange(len(idx)), index=idx).groupby(idx.normalize()):
                positions = np.asarray(group_idx, dtype=int)
                day_list.append(day)
                day_close.append(c[positions])
                day_volume.append(v[positions])
            values = _vpin_series(day_close, day_volume, scale_window, bucket_count)
            out[inst] = pd.Series(
                {d: val for d, val in zip(day_list, values)}, dtype=float
            )
        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    # The four intraday impact primitives are P1 daily targets.  ``micro_bvc_vpin``
    # is P2 / research-only (BV-C estimation error, minute bars are not ticks, the
    # VPIN literature is contested) and must never enter the default production
    # mining whitelist — it stays on the research surface.
    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "intraday_bvc_imbalance",
            "intraday_impact_beta",
            "intraday_impact_asymmetry",
            "intraday_return_wasserstein_shift",
        }
    )
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {"micro_bvc_vpin"}
    )


_register_surface()
