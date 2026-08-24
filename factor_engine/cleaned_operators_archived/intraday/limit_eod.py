# -*- coding: utf-8 -*-
"""Intraday limit-pressure and EOD-reversal decomposition operators (2026-08 R47).

Minute panels in, daily panels out (``SessionAggregationOperator`` from
``cleaned_operators/intraday/_core.py``): one scalar per (TradeDate, Symbol).
All kernels are causal and PIT-safe -- they consume only the day's own minute
data plus that day's daily limit panels, and form at that day's close.

``intra_limit_pre_hit_pressure_profile`` summarises the minute path in the
``pre_window`` bars immediately preceding a completed limit-hit event (or, with
``require_hit=False``, the nearest approach to the limit).

``intra_eod_reversal_decomposition`` decomposes the final ``window_minutes`` of
the trading day against a prior baseline window (the ``window_minutes`` before
it, or the whole morning session).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import register_operator
from cleaned_operators.intraday._core import (
    _EPS,
    SessionAggregationOperator,
    as_panel,
    broadcast_daily_panel,
    metadata,
    minute_of_day,
    register_surface,
    require_same_session_grid,
    session_local,
)

_CANONICALS: list[str] = []

_MORNING = (571, 690)  # 09:30 .. 11:30 minute-of-day


# ---------------------------------------------------------------------------
# intra_limit_pre_hit_pressure_profile
# ---------------------------------------------------------------------------
def _pre_hit_pressure_day(
    prices: np.ndarray,
    volumes: np.ndarray,
    minutes: np.ndarray,
    high_lim: float,
    low_lim: float,
    side: str,
    pre_window: int,
    require_hit: bool,
    output: str,
) -> float:
    """One (instrument, day) summary of the path before a limit-hit event.

    The pre-hit window stays INSIDE the session (morning/afternoon) that contains
    the hit bar -- the lunch boundary is never bridged.  ``minutes`` is the
    per-bar minute-of-day (int).
    """
    n = len(prices)
    if n == 0:
        return np.nan
    limit = high_lim if side == "up" else low_lim
    if not np.isfinite(limit):
        return np.nan
    if side == "up":
        hit_mask = prices >= limit
        distance = limit - prices  # >= 0 below the limit
    else:
        hit_mask = prices <= limit
        distance = prices - limit
    hits = np.flatnonzero(hit_mask)
    if len(hits) == 0:
        if require_hit:
            return np.nan
        # nearest approach by EOD: the bar with the minimum distance to limit
        d = np.where(np.isfinite(distance), distance, np.inf)
        if not np.any(np.isfinite(d)):
            return np.nan
        hit_idx = int(np.nanargmin(d))
    else:
        hit_idx = int(hits[0])

    # session boundary of the hit bar (do not bridge the lunch break)
    if minutes[hit_idx] <= _MORNING[1]:
        session_floor = 0
    else:
        session_floor = int(np.searchsorted(minutes, _MORNING[1] + 1))
    pre_start = max(session_floor, hit_idx - pre_window)
    if hit_idx - pre_start < pre_window:
        # fewer than ``pre_window`` valid pre-hit bars inside the same session
        return np.nan
    pre_prices = prices[pre_start:hit_idx]
    pre_volumes = volumes[pre_start:hit_idx]
    if len(pre_prices) < pre_window:
        return np.nan
    if not np.all(np.isfinite(pre_prices)):
        return np.nan
    if not np.all(np.isfinite(pre_volumes)):
        return np.nan

    if output == "return_accel":
        if len(pre_prices) < 2:
            return np.nan
        x = np.arange(len(pre_prices), dtype=float)
        return float(np.polyfit(x, np.log(pre_prices), 1)[0])
    if output == "volume_accel":
        if len(pre_volumes) < 2:
            return np.nan
        x = np.arange(len(pre_volumes), dtype=float)
        return float(np.polyfit(x, pre_volumes, 1)[0])
    if output == "distance_decay":
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.mean((limit - pre_prices) / limit))
    if output == "path_efficiency":
        if len(pre_prices) < 2:
            return np.nan
        net = abs(float(pre_prices[-1]) - float(pre_prices[0]))
        path = float(np.sum(np.abs(np.diff(pre_prices))))
        if path <= _EPS:
            return np.nan
        return float(net / path)
    return np.nan


def _limit_pressure_daily(
    price: pd.DataFrame,
    volume: pd.DataFrame,
    high_limit: pd.DataFrame,
    low_limit: pd.DataFrame,
    side: str,
    pre_window: int,
    require_hit: bool,
    output: str,
) -> pd.DataFrame:
    price = session_local(as_panel(price))
    volume = as_panel(volume)
    require_same_session_grid(price, volume)
    hl_bc = broadcast_daily_panel(price, high_limit)
    ll_bc = broadcast_daily_panel(price, low_limit)
    out: dict[str, pd.Series] = {}
    for inst in price.columns:
        pc = price[inst]
        vc = volume[inst]
        hc = hl_bc[inst]
        lc = ll_bc[inst]
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in pc.groupby(pc.index.normalize()):
            pv = group.to_numpy(dtype=float)
            vv = vc.reindex(group.index).to_numpy(dtype=float)
            hv = hc.reindex(group.index).to_numpy(dtype=float)
            lv = lc.reindex(group.index).to_numpy(dtype=float)
            minutes = minute_of_day(group.index.to_numpy(dtype="datetime64[ns]")).astype(int)
            hl_day = float(hv[np.isfinite(hv)][0]) if np.any(np.isfinite(hv)) else np.nan
            ll_day = float(lv[np.isfinite(lv)][0]) if np.any(np.isfinite(lv)) else np.nan
            per_day[day] = _pre_hit_pressure_day(
                pv, vv, minutes, hl_day, ll_day, side, pre_window, require_hit, output
            )
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


@register_operator(
    name="intra_limit_pre_hit_pressure_profile",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_limit_pre_hit_pressure_profile",
    source="intraday.limit_eod",
    backend="pandas_numpy",
    status="implemented",
)
class IntraLimitPreHitPressureProfile(SessionAggregationOperator):
    """Summarise the minute path immediately before a completed limit-hit event.

    ``price``/``volume`` are minute panels; ``high_limit``/``low_limit`` are
    DAILY panels broadcast to the day.  For a completed hit (close reaching the
    limit), the ``pre_window`` bars before the first hit bar are summarised:

    * ``return_accel``      -- linear slope of log-price over the pre-hit window.
    * ``volume_accel``      -- linear slope of volume over the pre-hit window.
    * ``distance_decay``    -- mean((limit - price)/limit) over the pre window.
    * ``path_efficiency``   -- |net move| / path length over the pre window.

    ``require_hit=True`` (default) emits NaN when the day never hits the limit;
    with ``require_hit=False`` the nearest approach by EOD is used as the event
    reference.  A zero-length pre-hit path (e.g. open-at-limit) emits NaN.  NaN
    when fewer than ``pre_window`` valid pre-hit bars exist.
    """

    metadata = metadata(
        "intra_limit_pre_hit_pressure_profile",
        "涨停/跌停首次触及前 pre_window 分钟的路径压力画像。",
        ["price", "volume", "high_limit", "low_limit", "side", "pre_window", "output", "require_hit"],
        unit="level",
        domain="intraday_limit",
        extra_tags=["allow_panel_broadcast"],
    )

    def _calculate_series(
        self,
        price: pd.DataFrame,
        volume: pd.DataFrame,
        high_limit: pd.DataFrame,
        low_limit: pd.DataFrame,
        side: str = "up",
        pre_window: int = 30,
        output: str = "return_accel",
        require_hit: bool = True,
        **_: Any,
    ) -> pd.DataFrame:
        if side not in ("up", "down"):
            raise ValueError(f"intra_limit_pre_hit_pressure_profile: side must be 'up' or 'down', got {side!r}")
        return _limit_pressure_daily(
            price, volume, high_limit, low_limit,
            str(side), max(1, int(pre_window)), bool(require_hit), str(output),
        )


# ---------------------------------------------------------------------------
# intra_eod_reversal_decomposition
# ---------------------------------------------------------------------------
def _eod_reversal_day(
    prices: np.ndarray,
    volumes: np.ndarray | None,
    minutes: np.ndarray,
    n_bars: int,
    baseline: str,
    output: str,
) -> float:
    n = len(prices)
    if n < 5:
        return np.nan
    # the close window is the final ``n_bars`` of the trading day (afternoon)
    cw_start = max(0, n - n_bars)
    cw = prices[cw_start:]
    if len(cw) < 5:
        return np.nan
    # afternoon session floor (minute-of-day strictly after the lunch break)
    afternoon_floor = int(np.searchsorted(minutes, _MORNING[1] + 1))
    if baseline == "morning":
        morning_mask = (minutes >= _MORNING[0]) & (minutes <= _MORNING[1])
        bw = prices[morning_mask]
    else:  # "prior_window": the ``n_bars`` before the close window, clamped to
        # the afternoon session so the lunch boundary is never bridged
        bw_start = max(afternoon_floor, cw_start - n_bars)
        bw = prices[bw_start:cw_start]
    if len(bw) < 5:
        return np.nan
    if not np.all(np.isfinite(cw)) or not np.all(np.isfinite(bw)):
        return np.nan

    cw_move = float(cw[-1]) - float(cw[0])
    bw_move = float(bw[-1]) - float(bw[0])
    rets = np.diff(np.log(bw))
    bw_std = float(np.std(rets)) if len(rets) >= 2 else np.nan
    if output == "pressure":
        if not np.isfinite(bw_std) or bw_std <= _EPS:
            return np.nan
        return float(cw_move / bw_std)
    if output == "reversal":
        sign_bw = 1.0 if bw_move >= 0.0 else -1.0
        return float(-sign_bw * cw_move / (abs(bw_move) + _EPS))
    if output == "retention":
        return float(cw_move / (abs(bw_move) + _EPS))
    if output == "participation_adjusted":
        if volumes is None or len(volumes) < n:
            return np.nan
        cw_vol = float(np.sum(volumes[cw_start:]))
        tot_vol = float(np.sum(volumes))
        if tot_vol <= _EPS:
            return np.nan
        if not np.isfinite(bw_std) or bw_std <= _EPS:
            return np.nan
        pressure = float(cw_move / bw_std)
        return float(pressure * cw_vol / tot_vol)
    return np.nan


def _bar_spacing_minutes(index: pd.DatetimeIndex) -> float:
    """Median bar spacing in minutes over the session (ignoring the lunch gap)."""
    if len(index) < 2:
        return 1.0
    # only consider intra-day (within same trading day) gaps
    mins = minute_of_day(index.to_numpy(dtype="datetime64[ns]")).astype(int)
    diffs = np.diff(mins)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))


def _eod_reversal_daily(
    price: pd.DataFrame,
    volume: pd.DataFrame | None,
    amount: pd.DataFrame | None,
    window_minutes: int,
    baseline: str,
    output: str,
) -> pd.DataFrame:
    price = session_local(as_panel(price))
    frames = [price]
    vol = None
    if volume is not None:
        vol = as_panel(volume)
        frames.append(vol)
    if amount is not None:
        frames.append(as_panel(amount))
    require_same_session_grid(*frames)
    out: dict[str, pd.Series] = {}
    for inst in price.columns:
        pc = price[inst]
        vc = vol[inst] if vol is not None else None
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in pc.groupby(pc.index.normalize()):
            pv = group.to_numpy(dtype=float)
            minutes = minute_of_day(group.index.to_numpy(dtype="datetime64[ns]")).astype(int)
            spacing = _bar_spacing_minutes(group.index)
            n_bars = max(1, int(round(window_minutes / spacing))) if spacing > 0 else 1
            vv = vc.reindex(group.index).to_numpy(dtype=float) if vc is not None else None
            per_day[day] = _eod_reversal_day(pv, vv, minutes, n_bars, baseline, output)
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


@register_operator(
    name="intra_eod_reversal_decomposition",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_eod_reversal_decomposition",
    source="intraday.limit_eod",
    backend="pandas_numpy",
    status="implemented",
)
class IntraEodReversalDecomposition(SessionAggregationOperator):
    """Decompose the final trading-window move.

    The close window is the last ``window_minutes`` of the trading day (mapped
    to bars by the session's median bar spacing).  ``baseline="prior_window"``
    uses the ``window_minutes`` before it; ``baseline="morning"`` uses the whole
    morning session.  Outputs (NaN with fewer than 5 bars in either window):

    * ``pressure``              -- (close[end]-close[start]) / baseline_std.
    * ``reversal``              -- -(sign of baseline move)*(close move)/(|baseline|+eps).
    * ``retention``             -- (close move)/(|baseline|+eps), positive = retains.
    * ``participation_adjusted``-- pressure * (close-window volume / day volume).
    """

    metadata = metadata(
        "intra_eod_reversal_decomposition",
        "收盘窗口相对前一窗口/午盘的动量分解（压力/反转/保持/参与度调整）。",
        ["price", "volume", "amount", "window_minutes", "baseline", "output"],
        unit="level",
        domain="intraday_eod",
        extra_tags=["allow_panel_broadcast"],
    )

    def _calculate_series(
        self,
        price: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        amount: pd.DataFrame | None = None,
        window_minutes: int = 30,
        baseline: str = "prior_window",
        output: str = "pressure",
        **_: Any,
    ) -> pd.DataFrame:
        if baseline not in ("prior_window", "morning"):
            raise ValueError(
                f"intra_eod_reversal_decomposition: baseline must be 'prior_window' "
                f"or 'morning', got {baseline!r}"
            )
        return _eod_reversal_daily(
            price, volume, amount,
            max(1, int(window_minutes)), str(baseline), str(output),
        )


_CANONICALS.extend(
    [
        "intra_limit_pre_hit_pressure_profile",
        "intra_eod_reversal_decomposition",
    ]
)
register_surface(_CANONICALS)
