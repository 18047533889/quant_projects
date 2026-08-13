# -*- coding: utf-8 -*-
"""Intraday time-series structure operators (2026-08 final pack, group 6).

Minute panels in, daily panels out (one scalar per TradeDate, Symbol).  The
family covers: intraday bar-range shape persistence/deviation, tail-bar volume
share, price-volume alignment, U-shaped time effect (edge / midday RV share),
same-slot cross-day surprises (volume / amount / volatility), cross-sectional
ex-self lead-lag vs the market and vs the same-industry peer group, morning vs
afternoon return asymmetry, closing-window participation and the affinity of
the intraday high/low timing.

Contract
--------
* Causal: a day's scalar uses only that day's own minute data plus past days
  (slot/trailing references are ``shift(1)`` windows).  Nothing is emitted for
  day t before day t is complete.
* Missing-value policy: NaN is never treated as 0.  Constant windows, samples
  below ``min_obs`` and cross-sections with fewer than two valid members return
  NaN (fail-closed).
* Returns are computed only between bars that are consecutive within a session
  (gap <= ``max_gap_minutes``), so overnight gaps and the lunch break never
  inject a spurious one-bar move.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import register_operator
from cleaned_operators.intraday._core import (
    _EPS,
    DataDegeneracy,
    SessionAggregationOperator,
    as_panel,
    daily_agg,
    metadata,
    minute_of_day,
    np_errstate,
    register_surface,
    require_same_session_grid,
    seg_mask,
    session_local,
)

_MAX_GAP_MINUTES = 10
_CANONICALS: list[str] = []


# ---------------------------------------------------------------------------
# Shared kernels
# ---------------------------------------------------------------------------

def _intraday_returns(cv: np.ndarray, times: np.ndarray, max_gap_minutes: int = _MAX_GAP_MINUTES) -> np.ndarray:
    """Within-session log returns, NaN at every session/gap boundary."""
    cv = np.asarray(cv, dtype=float)
    r = np.full(len(cv), np.nan)
    if len(cv) < 2:
        return r
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.where(cv[:-1]) != 0, np.log(cv[1:] / cv[:-1]), np.nan)
    gaps = (
        np.asarray(times[1:], dtype="datetime64[s]").astype(np.int64)
        - np.asarray(times[:-1], dtype="datetime64[s]").astype(np.int64)
    ) // 60
    ok = (gaps >= 1) & (gaps <= int(max_gap_minutes)) & np.isfinite(logr)
    r[1:] = np.where(ok, logr, np.nan)
    return r


def _daily_pair_with_times(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """Per-(instrument, day) fn(a_vals, b_vals, times) with aligned minute bars."""
    frame_a, frame_b = as_panel(frame_a), as_panel(frame_b)
    # P0-08: a mismatched session grid would be silently compressed by the
    # concat+dropna below; fail closed instead.  The dropna only removes rows
    # where the PRIMARY column (``a``) is NaN.
    require_same_session_grid(frame_a, frame_b)
    out: dict[str, pd.Series] = {}
    for inst in frame_a.columns:
        if inst not in frame_b.columns:
            continue
        joined = pd.concat([frame_a[inst], frame_b[inst]], axis=1, keys=["a", "b"]).dropna(subset=["a"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals_a = np.asarray(group["a"], dtype=float)
            vals_b = np.asarray(group["b"], dtype=float)
            times = np.asarray(group.index, dtype="datetime64[ns]")
            if int(np.sum(np.isfinite(vals_a))) < 2:
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(vals_a, vals_b, times))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    return pd.DataFrame(out).sort_index()


def _slot_matrix(values: pd.Series) -> pd.DataFrame:
    """Per-stock minute series -> day x minute-of-day matrix (mean per slot)."""
    frame = values.to_frame("v")
    frame["slot"] = minute_of_day(frame.index.to_numpy(dtype="datetime64[ns]"))
    frame["day"] = frame.index.normalize()
    return frame.pivot_table(index="day", columns="slot", values="v", aggfunc="mean")


def _trailing_mean(mat: pd.DataFrame, window: int) -> pd.DataFrame:
    """Mean of the *past* ``window`` days per slot (causal, never today)."""
    w = max(2, int(window))
    return mat.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()


def _log_range_series(high_inst: pd.Series, low_inst: pd.Series) -> pd.Series:
    h = high_inst.to_numpy(dtype=float)
    l = low_inst.to_numpy(dtype=float)
    with np_errstate():
        r = np.where(
            (h > 0) & (l > 0) & np.isfinite(h) & np.isfinite(l),
            np.log(h / l),
            np.nan,
        )
    return pd.Series(r, index=high_inst.index)


def _slot_daily_metric(values: pd.DataFrame, window: int, metric: str) -> pd.DataFrame:
    """Compare today's per-slot profile to the trailing mean per slot.

    metric in {"cosine", "mean_diff", "mean_abs_diff"}.  Causality comes from
    ``_trailing_mean``; a slot needs >= 2 valid overlapping points or the day
    returns NaN.
    """
    out: dict[str, pd.Series] = {}
    for inst in values.columns:
        mat = _slot_matrix(values[inst])
        hist = _trailing_mean(mat, window)
        day_scores: dict[pd.Timestamp, float] = {}
        for day in mat.index:
            if day not in hist.index:
                day_scores[day] = np.nan
                continue
            a = mat.loc[day].to_numpy(dtype=float)
            b = hist.loc[day].to_numpy(dtype=float)
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 2:
                day_scores[day] = np.nan
                continue
            va, vb = a[valid], b[valid]
            if metric == "cosine":
                na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
                if na <= _EPS or nb <= _EPS:
                    day_scores[day] = np.nan
                    continue
                day_scores[day] = np.where((na * nb)) != 0, float(np.dot(va, vb) / (na * nb)), np.nan)
            elif metric == "mean_diff":
                day_scores[day] = float(np.mean(va - vb))
            else:  # mean_abs_diff
                day_scores[day] = float(np.mean(np.abs(va - vb)))
        out[inst] = pd.Series(day_scores, dtype=float)
    return pd.DataFrame(out).sort_index()


def _extreme_minutes(values: pd.Series, mode: str) -> pd.Series:
    """Per-day minute-of-day of the intraday high (mode='max') or low (mode='min')."""
    per_day: dict[pd.Timestamp, float] = {}
    for day, group in values.groupby(values.index.normalize()):
        v = np.asarray(group, dtype=float)
        if not np.any(np.isfinite(v)):
            per_day[day] = np.nan
            continue
        i = int(np.nanargmax(v) if mode == "max" else np.nanargmin(v))
        per_day[day] = float(
            minute_of_day(np.asarray(group.index, dtype="datetime64[ns]"))[i]
        )
    return pd.Series(per_day, dtype=float)


def _corr_skipna(a: np.ndarray, b: np.ndarray, min_obs: int = 10) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < int(min_obs):
        return np.nan
    va, vb = a[m], b[m]
    if float(np.std(va)) < 1e-12 or float(np.std(vb)) < 1e-12:
        return np.nan
    return float(np.corrcoef(va, vb)[0, 1])


def _label_ok(value: Any) -> bool:
    """True for a usable classification label (numeric or categorical)."""
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)) and not np.isfinite(float(value)):
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _lead_lag_panel(
    close: pd.DataFrame,
    lag: int,
    min_obs: int,
    industry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-day ex-self lead-lag score vs the market or same-industry peer mean.

    Score = mean_k [ corr(r_t, m_{t-k}) - corr(r_t, m_{t+k}) ].  Positive means
    the stock follows the peer benchmark (lags it); negative means it leads.
    ``m`` is the cross-sectional mean of the other instruments' minute returns,
    excluding the instrument itself and (when ``industry`` is given) excluding
    instruments with a different industry label on that day.
    """
    lag_v = max(1, int(lag))
    labels_map = None
    if industry is not None:
        ind = as_panel(industry)
        if isinstance(ind.index, pd.DatetimeIndex):
            ind = ind.groupby(ind.index.normalize()).first()
        labels_map = ind.to_dict(orient="index")
    out: dict[str, dict[pd.Timestamp, float]] = {}
    for day, day_frame in close.groupby(close.index.normalize()):
        if day_frame.shape[1] < 2:
            for inst in day_frame.columns:
                out.setdefault(inst, {})[day] = np.nan
            continue
        grid = day_frame.index.unique().sort_values()
        P = pd.DataFrame(index=grid, columns=day_frame.columns, dtype=float)
        for inst in day_frame.columns:
            P[inst] = day_frame[inst].reindex(grid).ffill()
        P = P.where(P > 0)
        with np_errstate():
            L = np.diff(np.log(P.to_numpy(dtype=float)), axis=0)
        n, ncols = L.shape
        if n < 4:
            for inst in day_frame.columns:
                out.setdefault(inst, {})[day] = np.nan
            continue
        for j, inst in enumerate(day_frame.columns):
            try:
                stock = L[:, j]
                if labels_map is not None:
                    lj = labels_map.get(day, {}).get(inst, np.nan)
                    if not _label_ok(lj):
                        out.setdefault(inst, {})[day] = np.nan
                        continue
                    peers = [
                        i
                        for i, oth in enumerate(day_frame.columns)
                        if i != j
                        and _label_ok(labels_map.get(day, {}).get(oth, np.nan))
                        and labels_map.get(day, {}).get(oth) == lj
                    ]
                else:
                    peers = [i for i in range(ncols) if i != j]
                if len(peers) < 1:
                    out.setdefault(inst, {})[day] = np.nan
                    continue
                m_ex = np.nanmean(L[:, peers], axis=1)
                scores: list[float] = []
                for k in range(1, lag_v + 1):
                    c_lag = _corr_skipna(stock[k:], m_ex[: n - k], min_obs)
                    c_lead = _corr_skipna(stock[: n - k], m_ex[k:], min_obs)
                    if np.isfinite(c_lag) and np.isfinite(c_lead):
                        scores.append(c_lag - c_lead)
                out.setdefault(inst, {})[day] = float(np.mean(scores)) if scores else np.nan
            except Exception:
                out.setdefault(inst, {})[day] = np.nan
    return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# Bar-range shape
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_bar_range_persistence",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_bar_range_persistence",
    source="intraday.time_structure_v2",
)
class IntraBarRangePersistence(SessionAggregationOperator):
    """当日分钟区间（high-low）形状与历史均值曲线的余弦相似度。"""

    metadata = metadata(
        "intra_bar_range_persistence", "日内 bar 区间曲线跨日延续（余弦）。",
        ["high", "low", "window"], unit="cosine",
    )

    def _calculate_series(self, high, low, window=20, session_tz=None, **_):
        high = session_local(high, session_tz)
        low = session_local(low, session_tz)
        ranges = pd.DataFrame({
            inst: _log_range_series(high[inst], low[inst])
            for inst in high.columns
        })
        return _slot_daily_metric(ranges, int(window), "cosine")


@register_operator(
    name="intra_bar_range_deviation",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_bar_range_deviation",
    source="intraday.time_structure_v2",
)
class IntraBarRangeDeviation(SessionAggregationOperator):
    """当日分钟区间水平相对历史均值的整体偏差（同槽位平均差异）。"""

    metadata = metadata(
        "intra_bar_range_deviation", "日内 bar 区间整体高于/低于历史均值。",
        ["high", "low", "window"], unit="level",
    )

    def _calculate_series(self, high, low, window=20, session_tz=None, **_):
        high = session_local(high, session_tz)
        low = session_local(low, session_tz)
        ranges = pd.DataFrame({
            inst: _log_range_series(high[inst], low[inst])
            for inst in high.columns
        })
        return _slot_daily_metric(ranges, int(window), "mean_diff")


# ---------------------------------------------------------------------------
# Volume / price structure
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_tail_volume_share",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_tail_volume_share",
    source="intraday.time_structure_v2",
)
class IntraTailVolumeShare(SessionAggregationOperator):
    """最大 |分钟收益| 尾部 bar 的成交量占全天比例。"""

    metadata = metadata(
        "intra_tail_volume_share", "极端分钟涨跌 bar 成交量占比。",
        ["close", "volume", "tail_quantile"], unit="ratio",
    )

    def _calculate_series(self, close, volume, tail_quantile=0.75, session_tz=None, **_):
        q = float(tail_quantile)
        if not 0.0 < q < 1.0:
            raise ValueError("tail_quantile must be in (0, 1)")
        close = session_local(close, session_tz)
        volume = session_local(volume, session_tz)

        def _fn(cv, vv, times):
            r = _intraday_returns(cv, times)
            m = np.isfinite(r) & np.isfinite(vv)
            if m.sum() < 4:
                return np.nan
            rr, v2 = r[m], vv[m]
            thr = float(np.quantile(np.abs(rr), q))
            tail = np.abs(rr) >= thr
            if not tail.any():
                return np.nan
            total = float(np.sum(v2))
            if total <= _EPS:
                return np.nan
            return np.where(total != 0, float(np.sum(v2[tail])) / total, np.nan)

        return _daily_pair_with_times(close, volume, _fn)


@register_operator(
    name="intra_volume_price_alignment",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_volume_price_alignment",
    source="intraday.time_structure_v2",
)
class IntraVolumePriceAlignment(SessionAggregationOperator):
    """分钟收益与分钟成交量的日内相关（上涨放量 vs 下跌放量）。"""

    metadata = metadata(
        "intra_volume_price_alignment", "价量日内相关：corr(minute_return, volume)。",
        ["close", "volume"], unit="ratio",
    )

    def _calculate_series(self, close, volume, session_tz=None, **_):
        close = session_local(close, session_tz)
        volume = session_local(volume, session_tz)

        def _fn(cv, vv, times):
            r = _intraday_returns(cv, times)
            m = np.isfinite(r) & np.isfinite(vv)
            if m.sum() < 10:
                return np.nan
            rr, v2 = r[m], vv[m]
            if float(np.std(rr)) < _EPS or float(np.std(v2)) < _EPS:
                return np.nan
            return float(np.corrcoef(rr, v2)[0, 1])

        return _daily_pair_with_times(close, volume, _fn)


# ---------------------------------------------------------------------------
# U-shaped time effect
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_ute_high",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_ute_high",
    source="intraday.time_structure_v2",
)
class IntraUteHigh(SessionAggregationOperator):
    """U 型时间效应-两翼：开收盘边缘窗口已实现方差占全天比例。"""

    metadata = metadata(
        "intra_ute_high", "开收盘边缘 RV 占比（U 型上翘强度）。",
        ["close", "edge_minutes"], unit="ratio",
    )

    def _calculate_series(self, close, edge_minutes=30, session_tz=None, **_):
        edge = int(edge_minutes)
        if edge < 1:
            raise ValueError("edge_minutes must be >= 1")
        close = session_local(close, session_tz)

        def _fn(cv, times):
            r = _intraday_returns(cv, times)
            r2 = r * r
            total = float(np.nansum(r2))
            if total <= _EPS:
                return np.nan
            minutes = minute_of_day(times)
            finite_minutes = minutes[np.isfinite(r)]
            if finite_minutes.size < 4:
                return np.nan
            s_open, s_close = int(finite_minutes.min()), int(finite_minutes.max())
            eff_edge = min(edge, (s_close - s_open) // 2)
            if eff_edge < 1:
                return np.nan
            open_mask = (minutes >= s_open) & (minutes <= s_open + eff_edge)
            close_mask = (minutes >= s_close - eff_edge) & (minutes <= s_close)
            edge_rv = float(np.nansum(r2[open_mask])) + float(np.nansum(r2[close_mask]))
            return np.where(total != 0, edge_rv / total, np.nan)

        return daily_agg(close, _fn)


@register_operator(
    name="intra_ute_low",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_ute_low",
    source="intraday.time_structure_v2",
)
class IntraUteLow(SessionAggregationOperator):
    """U 型时间效应-午间低谷：中午窗口已实现方差占全天比例。"""

    metadata = metadata(
        "intra_ute_low", "午间窗口 RV 占比（U 型低谷强度）。",
        ["close", "mid_start", "mid_end"], unit="ratio",
    )

    def _calculate_series(self, close, mid_start=660, mid_end=810, session_tz=None, **_):
        ms, me = int(mid_start), int(mid_end)
        if not 0 <= ms < me <= 1440:
            raise ValueError("require 0 <= mid_start < mid_end <= 1440")
        close = session_local(close, session_tz)

        def _fn(cv, times):
            r = _intraday_returns(cv, times)
            r2 = r * r
            total = float(np.nansum(r2))
            if total <= _EPS:
                return np.nan
            minutes = minute_of_day(times)
            mid = (minutes >= ms) & (minutes <= me)
            return np.where(total != 0, float(np.nansum(r2[mid])) / total, np.nan)

        return daily_agg(close, _fn)


# ---------------------------------------------------------------------------
# Same-slot cross-day surprises
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_slot_volume_surprise",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_slot_volume_surprise",
    source="intraday.time_structure_v2",
)
class IntraSlotVolumeSurprise(SessionAggregationOperator):
    """同槽位成交量相对历史均值的平均偏离（log 尺度）。"""

    metadata = metadata(
        "intra_slot_volume_surprise", "分钟槽成交量跨日意外度。",
        ["volume", "window"], unit="level",
    )

    def _calculate_series(self, volume, window=20, session_tz=None, **_):
        volume = session_local(volume, session_tz)
        with np_errstate():
            logvol = np.log1p(volume.to_numpy(dtype=float))
        frame = pd.DataFrame(logvol, index=volume.index, columns=volume.columns)
        return _slot_daily_metric(frame, int(window), "mean_abs_diff")


@register_operator(
    name="intra_slot_amount_surprise",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_slot_amount_surprise",
    source="intraday.time_structure_v2",
)
class IntraSlotAmountSurprise(SessionAggregationOperator):
    """同槽位成交额相对历史均值的平均偏离（log 尺度）。"""

    metadata = metadata(
        "intra_slot_amount_surprise", "分钟槽成交额跨日意外度。",
        ["amount", "window"], unit="level",
    )

    def _calculate_series(self, amount, window=20, session_tz=None, **_):
        amount = session_local(amount, session_tz)
        with np_errstate():
            logamt = np.log1p(amount.to_numpy(dtype=float))
        frame = pd.DataFrame(logamt, index=amount.index, columns=amount.columns)
        return _slot_daily_metric(frame, int(window), "mean_abs_diff")


@register_operator(
    name="intra_slot_volatility_surprise",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_slot_volatility_surprise",
    source="intraday.time_structure_v2",
)
class IntraSlotVolatilitySurprise(SessionAggregationOperator):
    """同槽位 bar 区间（波动代理）相对历史均值的平均偏离。"""

    metadata = metadata(
        "intra_slot_volatility_surprise", "分钟槽波动意外度（log range）。",
        ["high", "low", "window"], unit="level",
    )

    def _calculate_series(self, high, low, window=20, session_tz=None, **_):
        high = session_local(high, session_tz)
        low = session_local(low, session_tz)
        ranges = pd.DataFrame({
            inst: _log_range_series(high[inst], low[inst])
            for inst in high.columns
        })
        return _slot_daily_metric(ranges, int(window), "mean_abs_diff")


# ---------------------------------------------------------------------------
# Cross-sectional ex-self lead-lag
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_market_lead_lag_ex_self",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_market_lead_lag_ex_self",
    source="intraday.time_structure_v2",
)
class IntraMarketLeadLagExSelf(SessionAggregationOperator):
    """日内 ex-self 市场领先滞后：corr(r_t, m_{t-k}) - corr(r_t, m_{t+k})。"""

    metadata = metadata(
        "intra_market_lead_lag_ex_self", "个股 vs 剔除自身市场组合的日内领先滞后。",
        ["close", "lag", "min_obs"], unit="level", cost=9,
    )

    def _calculate_series(self, close, lag=3, min_obs=10, session_tz=None, **_):
        if int(lag) < 1:
            raise ValueError("lag must be >= 1")
        close = session_local(close, session_tz)
        return _lead_lag_panel(close, int(lag), int(min_obs), industry=None)


@register_operator(
    name="intra_industry_lead_lag_ex_self",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_industry_lead_lag_ex_self",
    source="intraday.time_structure_v2",
)
class IntraIndustryLeadLagExSelf(SessionAggregationOperator):
    """日内同行业（剔除自身）领先滞后。industry 为日频标签面板。"""

    metadata = metadata(
        "intra_industry_lead_lag_ex_self", "个股 vs 同行业（剔除自身）日内领先滞后。",
        ["close", "industry", "lag", "min_obs"], unit="level", cost=10,
        extra_tags=["allow_panel_broadcast"],
    )

    def _calculate_series(self, close, industry, lag=3, min_obs=10, session_tz=None, **_):
        if int(lag) < 1:
            raise ValueError("lag must be >= 1")
        close = session_local(close, session_tz)
        return _lead_lag_panel(close, int(lag), int(min_obs), industry=industry)


# ---------------------------------------------------------------------------
# Session structure
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_session_return_asymmetry",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_session_return_asymmetry",
    source="intraday.time_structure_v2",
)
class IntraSessionReturnAsymmetry(SessionAggregationOperator):
    """上午/下午绝对收益强度不对称：(morning - afternoon)/(morning + afternoon)。"""

    metadata = metadata(
        "intra_session_return_asymmetry", "日内上/下午绝对收益强度差。",
        ["close"], unit="ratio",
    )

    def _calculate_series(self, close, session_tz=None, **_):
        close = session_local(close, session_tz)

        def _fn(cv, times):
            r = np.abs(_intraday_returns(cv, times))
            morning = seg_mask(times, "morning")
            afternoon = seg_mask(times, "afternoon")
            m = float(np.nansum(r[morning]))
            a = float(np.nansum(r[afternoon]))
            denom = m + a
            if denom <= _EPS:
                return np.nan
            return np.where(denom != 0, (m - a) / denom, np.nan)

        return daily_agg(close, _fn)


@register_operator(
    name="intra_close_participation",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_close_participation",
    source="intraday.time_structure_v2",
)
class IntraCloseParticipation(SessionAggregationOperator):
    """收盘前 tail_minutes 分钟成交量占全天比例（尾盘参与度）。"""

    metadata = metadata(
        "intra_close_participation", "尾盘窗口成交量占比。",
        ["volume", "tail_minutes"], unit="ratio",
    )

    def _calculate_series(self, volume, tail_minutes=30, session_tz=None, **_):
        tail = int(tail_minutes)
        if tail < 1:
            raise ValueError("tail_minutes must be >= 1")
        volume = session_local(volume, session_tz)

        def _fn(vv, times):
            m = np.isfinite(vv)
            if m.sum() < 2:
                return np.nan
            minutes = minute_of_day(times)
            s_open, s_close = int(minutes[m].min()), int(minutes[m].max())
            eff = min(tail, s_close - s_open)
            if eff < 1:
                return np.nan
            mask = (minutes >= s_close - eff) & (minutes <= s_close)
            total = float(np.nansum(vv))
            if total <= _EPS:
                return np.nan
            return np.where(total != 0, float(np.nansum(vv[mask])) / total, np.nan)

        return daily_agg(volume, _fn)


@register_operator(
    name="intra_high_low_affinity",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_high_low_affinity",
    source="intraday.time_structure_v2",
)
class IntraHighLowAffinity(SessionAggregationOperator):
    """日内高点/低点出现时刻与历史均值时刻的相似度（越固定越接近 1）。"""

    metadata = metadata(
        "intra_high_low_affinity", "日内高低点出现时刻的跨日可重复性。",
        ["high", "low", "window"], unit="ratio", cost=7,
    )

    def _calculate_series(self, high, low, window=20, session_tz=None, **_):
        w = max(2, int(window))
        high = session_local(high, session_tz)
        low = session_local(low, session_tz)
        out: dict[str, pd.Series] = {}
        for inst in high.columns:
            toh = _extreme_minutes(high[inst], "max")
            tol = _extreme_minutes(low[inst], "min")
            m_toh = _trailing_mean(toh.to_frame("v"), w)["v"]
            m_tol = _trailing_mean(tol.to_frame("v"), w)["v"]
            days = toh.index.intersection(m_toh.index)
            scores: dict[pd.Timestamp, float] = {}
            for day in days:
                if day not in tol.index or day not in m_tol.index:
                    continue
                vals = (toh[day], m_toh[day], tol[day], m_tol[day])
                if not all(np.isfinite(v) for v in vals):
                    scores[day] = np.nan
                    continue
                dev = np.where(120.0 != 0, (abs(vals[0] - vals[1]) + abs(vals[2] - vals[3])) / 120.0, np.nan)
                scores[day] = np.where((1.0 + dev) != 0, 1.0 / (1.0 + dev), np.nan)
            out[inst] = pd.Series(scores, dtype=float)
        return pd.DataFrame(out).sort_index()


_CANONICALS.extend(
    [
        "intra_bar_range_persistence",
        "intra_bar_range_deviation",
        "intra_tail_volume_share",
        "intra_volume_price_alignment",
        "intra_ute_high",
        "intra_ute_low",
        "intra_slot_volume_surprise",
        "intra_slot_amount_surprise",
        "intra_slot_volatility_surprise",
        "intra_market_lead_lag_ex_self",
        "intra_industry_lead_lag_ex_self",
        "intra_session_return_asymmetry",
        "intra_close_participation",
        "intra_high_low_affinity",
    ]
)

register_surface(_CANONICALS)
