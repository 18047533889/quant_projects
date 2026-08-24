# -*- coding: utf-8 -*-
"""R47 intraday event/response operators (2026-08 final pack, group 7).

Minute-frequency panels in, daily-frequency panels out (one scalar per
TradeDate, Symbol).  The family answers "given that this intraday event
happened, what is the subsequent response?":

* ``intra_event_window_reduce``           — fixed-slot window statistic around a
  selected event bar (mean / sum / std / max / min / last / slope).
* ``intra_event_pre_post_contrast``       — pre-event vs post-event metric
  contrast (mean / median / volatility / slope / range / activity).
* ``intra_impulse_event_detector``        — robust intraday price-impulse
  detection with episode merging (count / strength / timing / duration).
* ``intra_post_impulse_response``         — retention / giveback / drawdown /
  volume & VWAP ratios / recovery half-life after each impulse.
* ``intra_probe_outcome_score``           — composite observable probe score
  (strength / giveback / VWAP-hold / participation) per session.
* ``intra_supply_absorption_score``       — how well the market absorbed an
  impulse (price impact, price-per-amount, volume resilience).
* ``intra_consolidation_quality``         — post-trigger consolidation
  tightness / level / vol-compression / volume-dryup / rising floor.
* ``intra_response_curve_features``       — event-time response-curve shape
  (slope / curvature / auc / half-life / monotonicity / change count).
* ``intra_liquidity_resilience_curve_fit``— exponential activity-resilience
  fit after liquidity shocks (half-life / residual / asymptote).

Contract
--------
* Causal: a day's scalar uses only that day's own minute data (a session's
  events, windows and responses never leave the session; the lunch gap is never
  bridged — within-session bar adjacency only).
* Missing-value policy: NaN is never 0.  Constant windows, samples below
  ``min_obs``, and days with no event / insufficient data return NaN
  (fail-closed).  The single exception is the honest zero count of a detector
  (``intra_impulse_event_detector`` ``output="count"`` returns ``0.0`` when
  data is sufficient but no impulse fired).
* Deterministic: tie-breaks are explicit (``strongest`` picks the earliest
  event bar; running extremes take the first occurrence).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import register_operator
from factor_engine.cleaned_operators.intraday._core import (
    DataDegeneracy,
    SessionAggregationOperator,
    _EPS,
    as_panel,
    daily_agg,
    daily_agg_two,
    daily_agg_three,
    metadata,
    minute_of_day,
    np_errstate,
    register_surface,
    require_same_session_grid,
    safe_div,
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
        logr = np.log(cv[1:] / cv[:-1])
    gaps = (
        np.asarray(times[1:], dtype="datetime64[s]").astype(np.int64)
        - np.asarray(times[:-1], dtype="datetime64[s]").astype(np.int64)
    ) // 60
    ok = (gaps >= 1) & (gaps <= int(max_gap_minutes)) & np.isfinite(logr)
    r[1:] = np.where(ok, logr, np.nan)
    return r


def _bar_segments(times: np.ndarray, max_gap_minutes: int = _MAX_GAP_MINUTES) -> list[tuple[int, int]]:
    """Index ranges [start, end) of contiguous minute bars (no lunch bridging).

    A new segment starts whenever two consecutive bars are separated by more
    than ``max_gap_minutes`` (the A-share lunch break is 91 minutes, so morning
    and afternoon become distinct segments).
    """
    n = len(times)
    if n == 0:
        return []
    gaps = (
        np.asarray(times[1:], dtype="datetime64[s]").astype(np.int64)
        - np.asarray(times[:-1], dtype="datetime64[s]").astype(np.int64)
    ) // 60
    bounds = [0]
    for i, g in enumerate(gaps):
        if not (1 <= int(g) <= int(max_gap_minutes)):
            bounds.append(i + 1)
    bounds.append(n)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _segment_containing(e: int, segs: list[tuple[int, int]]) -> tuple[int, int] | None:
    for s, end in segs:
        if s <= e < end:
            return (s, end)
    return None


def _window_indices(e: int, seg: tuple[int, int], lo_off: int, hi_off: int) -> np.ndarray | None:
    """Bar indices [e+lo_off, e+hi_off] clipped to the same session segment."""
    seg_start, seg_end = seg
    lo = max(seg_start, e + lo_off)
    hi = min(seg_end - 1, e + hi_off)
    if lo > hi:
        return None
    return np.arange(lo, hi + 1, dtype=np.int64)


def _finite_with_times(x: np.ndarray, w: np.ndarray | None, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite values in window ``w`` plus their minute-of-day x-axis."""
    if w is None:
        return np.array([]), np.array([])
    vals = np.asarray(x, dtype=float)[w]
    ok = np.isfinite(vals)
    if not ok.any():
        return np.array([]), np.array([])
    minutes = minute_of_day(np.asarray(times, dtype="datetime64[ns]")[w])
    return vals[ok], minutes[ok].astype(float)


def _select_event(ev_all: np.ndarray, x: np.ndarray, event_select: str) -> int | None:
    """Pick one event bar per session.  ``strongest`` = max |x|, tie -> earliest."""
    if event_select == "first":
        return int(ev_all[0])
    if event_select == "last":
        return int(ev_all[-1])
    if event_select == "strongest":
        ax = np.abs(np.asarray(x, dtype=float)[ev_all])
        if not np.isfinite(ax).any():
            return None
        return int(ev_all[int(np.nanargmax(ax))])
    raise ValueError(f"unknown event_select {event_select!r}")


def _merge_episodes(ev_idx: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    """Merge event bars separated by <= ``merge_gap`` bars into episodes."""
    ev_idx = np.asarray(ev_idx, dtype=int)
    if ev_idx.size == 0:
        return []
    episodes: list[tuple[int, int]] = []
    start = prev = int(ev_idx[0])
    for i in ev_idx[1:]:
        i = int(i)
        if i - prev <= int(merge_gap):
            prev = i
        else:
            episodes.append((start, prev))
            start = prev = i
    episodes.append((start, prev))
    return episodes


def _ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if n < 2:
        return np.nan
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        return np.nan
    denom = float(np.sum((x - x.mean()) ** 2))
    if denom <= _EPS:
        return np.nan
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)


def _ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if n < 2 or not np.isfinite(y).all() or not np.isfinite(x).all():
        return np.nan
    design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= _EPS:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _quadratic_coef(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if n < 3 or not np.isfinite(y).all() or not np.isfinite(x).all():
        return np.nan
    design = np.column_stack([np.ones(n), x, x ** 2])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[2])


def _detect_impulse_episodes(
    cv: np.ndarray,
    times: np.ndarray,
    *,
    direction: str,
    threshold: str,
    z: float,
    merge_gap: int,
    max_gap_minutes: int = _MAX_GAP_MINUTES,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Detect intraday price impulses and merge them into episodes.

    Returns ``(episodes, evmask)`` where ``episodes`` is a list of ``(start,
    end)`` event-bar index pairs and ``evmask`` marks every event bar.
    """
    r = _intraday_returns(cv, times, max_gap_minutes)
    fin = np.isfinite(r)
    rvals = r[fin]
    n = len(r)
    if rvals.size == 0:
        return [], np.zeros(n, dtype=bool)
    z = float(z)
    if threshold == "robust_z":
        med = float(np.median(rvals))
        mad = float(np.median(np.abs(rvals - med)))
        thr = med + z * 1.4826 * mad
        thr_t = np.full(n, thr)
    elif threshold == "vol_scaled":
        roll_std = pd.Series(r).rolling(20, min_periods=5).std().to_numpy()
        thr_t = z * roll_std
    elif threshold == "quantile":
        q = float(np.clip(z, 0.0, 1.0))
        thr = float(np.quantile(np.abs(rvals), q, method="linear"))
        thr_t = np.full(n, thr)
    else:
        raise ValueError(f"unknown threshold {threshold!r}")
    direction = str(direction).lower()
    if direction == "up":
        ev = fin & (r > thr_t)
    elif direction == "down":
        ev = fin & (r < -thr_t)
    elif direction == "both":
        ev = fin & (np.abs(r) > thr_t)
    else:
        raise ValueError(f"unknown direction {direction!r}")
    ev_idx = np.flatnonzero(ev)
    episodes = _merge_episodes(ev_idx, merge_gap)
    return episodes, ev


# ---------------------------------------------------------------------------
# Daily aggregation with times (per instrument, per calendar day)
# ---------------------------------------------------------------------------

def _daily_with_times(
    frames: list[Any],
    fn: Callable[..., float],
    *,
    min_finite: int = 2,
) -> pd.DataFrame:
    """Per-(instrument, calendar-day) ``fn(*vals, times)`` on aligned minute bars.

    The primary frame's bar positions are preserved (rows are never dropped), so
    within-session adjacency / gap logic in the kernels sees the true minute
    grid.  NaN values inside a frame are left for the kernel to interpret.
    """
    frames = [as_panel(f) for f in frames]
    require_same_session_grid(*frames)
    base = frames[0]
    out: dict[str, pd.Series] = {}
    for inst in base.columns:
        if any(inst not in f.columns for f in frames[1:]):
            continue
        joined = pd.concat(
            [f[inst] for f in frames],
            axis=1,
            keys=[f"v{i}" for i in range(len(frames))],
        )
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals = [
                np.asarray(group[f"v{i}"], dtype=float) for i in range(len(frames))
            ]
            times = np.asarray(group.index, dtype="datetime64[ns]")
            if int(np.sum(np.isfinite(vals[0]))) < int(min_finite):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(*vals, times))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# 1. intra_event_window_reduce
# ---------------------------------------------------------------------------

def _event_window_reduce_day(
    x: np.ndarray, mask: np.ndarray, times: np.ndarray,
    pre: int, post: int, reducer: str, event_select: str, min_obs: int,
) -> float:
    ev_all = np.flatnonzero(np.isfinite(mask) & (mask > 0))
    if ev_all.size == 0:
        return np.nan
    e = _select_event(ev_all, x, event_select)
    if e is None:
        return np.nan
    segs = _bar_segments(times)
    seg = _segment_containing(e, segs)
    if seg is None:
        return np.nan
    w = _window_indices(e, seg, -pre, post)
    if w is None:
        return np.nan
    vals, minutes = _finite_with_times(x, w, times)
    cnt = int(vals.size)
    if cnt < int(min_obs):
        return np.nan
    if reducer == "mean":
        return float(np.mean(vals))
    if reducer == "sum":
        return float(np.sum(vals))
    if reducer == "std":
        if cnt < 2:
            return np.nan
        return float(np.std(vals, ddof=1))
    if reducer == "max":
        return float(np.max(vals))
    if reducer == "min":
        return float(np.min(vals))
    if reducer == "last":
        return float(vals[-1])
    if reducer == "slope":
        if cnt < 2:
            return np.nan
        return _ols_slope(vals, minutes)
    raise ValueError(f"unknown reducer {reducer!r}")


@register_operator(
    name="intra_event_window_reduce",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_event_window_reduce",
    source="intraday.event_response",
    backend="pandas_numpy",
    status="implemented",
)
class IntraEventWindowReduce(SessionAggregationOperator):
    """选定事件 bar 前后的固定槽位窗口统计量（不跨午间、截断于段边缘）。"""

    metadata = metadata(
        "intra_event_window_reduce",
        "选定事件 bar 的 [pre, post] 窗口 reducer（mean/sum/std/max/min/last/slope）。",
        ["x", "event_mask", "pre", "post", "reducer", "event_select", "min_obs"],
        unit="level",
    )

    def _calculate_series(
        self, x, event_mask, pre=5, post=15, reducer="mean",
        event_select="first", min_obs=3, session_tz=None, **_,
    ):
        pre, post, min_obs = int(pre), int(post), int(min_obs)
        if pre < 0 or post < 0:
            raise ValueError("pre and post must be >= 0")
        if min_obs < 1:
            raise ValueError("min_obs must be >= 1")
        reducer = str(reducer).lower()
        if reducer not in {"mean", "sum", "std", "max", "min", "last", "slope"}:
            raise ValueError(f"unknown reducer {reducer!r}")
        if event_select not in {"first", "last", "strongest"}:
            raise ValueError(f"unknown event_select {event_select!r}")
        x = session_local(x, session_tz)
        event_mask = session_local(event_mask, session_tz)
        return _daily_with_times(
            [x, event_mask],
            lambda a, b, t: _event_window_reduce_day(
                a, b, t, pre, post, reducer, event_select, min_obs,
            ),
        )


# ---------------------------------------------------------------------------
# 2. intra_event_pre_post_contrast
# ---------------------------------------------------------------------------

def _event_pre_post_contrast_day(
    x: np.ndarray, mask: np.ndarray, times: np.ndarray,
    pre: int, post: int, metric: str, event_select: str, min_obs: int,
) -> float:
    ev_all = np.flatnonzero(np.isfinite(mask) & (mask > 0))
    if ev_all.size == 0:
        return np.nan
    e = _select_event(ev_all, x, event_select)
    if e is None:
        return np.nan
    segs = _bar_segments(times)
    seg = _segment_containing(e, segs)
    if seg is None:
        return np.nan
    pre_w = _window_indices(e, seg, -pre, -1)
    post_w = _window_indices(e, seg, 1, post)
    pre_v, _ = _finite_with_times(x, pre_w, times)
    post_v, _ = _finite_with_times(x, post_w, times)
    if int(pre_v.size) < int(min_obs) or int(post_v.size) < int(min_obs):
        return np.nan
    if metric == "mean_diff":
        return float(np.mean(post_v) - np.mean(pre_v))
    if metric == "median_diff":
        return float(np.median(post_v) - np.median(pre_v))
    if metric == "vol_ratio":
        if pre_v.size < 2:
            return np.nan
        pre_std = float(np.std(pre_v, ddof=1))
        if pre_std <= _EPS:
            return np.nan
        return float(np.std(post_v, ddof=1)) / pre_std
    if metric == "slope_diff":
        pre_v2, pre_t = _finite_with_times(x, pre_w, times)
        post_v2, post_t = _finite_with_times(x, post_w, times)
        pre_s = _ols_slope(pre_v2, pre_t)
        post_s = _ols_slope(post_v2, post_t)
        if not np.isfinite(pre_s) or not np.isfinite(post_s):
            return np.nan
        return post_s - pre_s
    if metric == "range_ratio":
        pre_range = float(np.max(pre_v) - np.min(pre_v))
        if pre_range <= _EPS:
            return np.nan
        return float(np.max(post_v) - np.min(post_v)) / pre_range
    if metric == "activity_ratio":
        pre_norm = float(np.sum(np.abs(pre_v)))
        if pre_norm <= _EPS:
            return np.nan
        return float(np.sum(np.abs(post_v))) / pre_norm
    raise ValueError(f"unknown metric {metric!r}")


@register_operator(
    name="intra_event_pre_post_contrast",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_event_pre_post_contrast",
    source="intraday.event_response",
    backend="pandas_numpy",
    status="implemented",
)
class IntraEventPrePostContrast(SessionAggregationOperator):
    """事件前后窗口指标的对比（均值/中位数/波动/斜率/区间/活动度）。"""

    metadata = metadata(
        "intra_event_pre_post_contrast",
        "事件前 vs 事件后窗口指标对比（同段截断，不跨午间）。",
        ["x", "event_mask", "pre", "post", "metric", "event_select", "min_obs"],
        unit="level",
    )

    def _calculate_series(
        self, x, event_mask, pre=10, post=10, metric="mean_diff",
        event_select="first", min_obs=5, session_tz=None, **_,
    ):
        pre, post, min_obs = int(pre), int(post), int(min_obs)
        if pre < 0 or post < 0:
            raise ValueError("pre and post must be >= 0")
        if min_obs < 1:
            raise ValueError("min_obs must be >= 1")
        metric = str(metric).lower()
        if metric not in {
            "mean_diff", "median_diff", "vol_ratio", "slope_diff",
            "range_ratio", "activity_ratio",
        }:
            raise ValueError(f"unknown metric {metric!r}")
        if event_select not in {"first", "last", "strongest"}:
            raise ValueError(f"unknown event_select {event_select!r}")
        x = session_local(x, session_tz)
        event_mask = session_local(event_mask, session_tz)
        return _daily_with_times(
            [x, event_mask],
            lambda a, b, t: _event_pre_post_contrast_day(
                a, b, t, pre, post, metric, event_select, min_obs,
            ),
        )


# ---------------------------------------------------------------------------
# 3. intra_impulse_event_detector
# ---------------------------------------------------------------------------

def _impulse_detector_day(
    cv: np.ndarray, times: np.ndarray, event: str, threshold: str, z: float,
    min_bars: int, merge_gap: int, output: str,
) -> float:
    r = _intraday_returns(cv, times)
    fin = np.isfinite(r)
    if int(fin.sum()) < int(min_bars):
        return np.nan
    episodes, evmask = _detect_impulse_episodes(
        cv, times, direction=event, threshold=threshold, z=z, merge_gap=merge_gap,
    )
    if not episodes:
        if output == "count":
            return 0.0
        return np.nan
    nbars = float(len(cv))
    if output == "count":
        return float(len(episodes))
    if output == "strength":
        return float(np.sum(np.abs(r[evmask])))
    if output == "max_strength":
        best = 0.0
        for (s, e) in episodes:
            seg = np.abs(r[s:e + 1])
            seg = seg[np.isfinite(seg)]
            if seg.size:
                best = max(best, float(np.sum(seg)))
        return best
    first = int(episodes[0][0])
    last = int(episodes[-1][1])
    if output == "first_time":
        return first / nbars
    if output == "last_time":
        return last / nbars
    if output == "duration":
        return float(last - first + 1)
    raise ValueError(f"unknown output {output!r}")


@register_operator(
    name="intra_impulse_event_detector",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_impulse_event_detector",
    source="intraday.event_response",
    backend="pandas_numpy",
    status="implemented",
)
class IntraImpulseEventDetector(SessionAggregationOperator):
    """日内价格脉冲检测（robust_z / vol_scaled / quantile 阈值，事件归并）。"""

    metadata = metadata(
        "intra_impulse_event_detector",
        "日内价格脉冲检测：count/strength/max_strength/first_time/last_time/duration。",
        ["price", "volume", "event", "threshold", "z", "min_bars", "merge_gap", "output"],
        unit="count", cost=7,
    )

    def _calculate_series(
        self, price, volume=None, event="up", threshold="robust_z", z=3.0,
        min_bars=1, merge_gap=2, output="count", session_tz=None, **_,
    ):
        event = str(event).lower()
        if event not in {"up", "down", "both"}:
            raise ValueError(f"unknown event {event!r}")
        threshold = str(threshold).lower()
        if threshold not in {"robust_z", "vol_scaled", "quantile"}:
            raise ValueError(f"unknown threshold {threshold!r}")
        z = float(z)
        if not np.isfinite(z):
            raise ValueError("z must be finite")
        if threshold == "quantile" and not 0.0 <= z <= 1.0:
            raise ValueError("z must be in [0, 1] for quantile threshold")
        if int(min_bars) < 1:
            raise ValueError("min_bars must be >= 1")
        if int(merge_gap) < 0:
            raise ValueError("merge_gap must be >= 0")
        output = str(output).lower()
        if output not in {
            "count", "strength", "max_strength", "first_time", "last_time", "duration",
        }:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        day_fn = lambda cv, t: _impulse_detector_day(  # noqa: E731
            cv, t, event, threshold, z, int(min_bars), int(merge_gap), output,
        )
        if volume is None:
            return _daily_with_times([price], day_fn)
        volume = session_local(volume, session_tz)
        return _daily_with_times([price, volume], lambda a, b, t: day_fn(a, t))


# ---------------------------------------------------------------------------
# 4. intra_post_impulse_response
# ---------------------------------------------------------------------------

def _post_response_event(
    cv: np.ndarray, vv: np.ndarray, av: np.ndarray, r: np.ndarray,
    times: np.ndarray, segs: list[tuple[int, int]], e: int, horizon: int, output: str,
) -> float:
    seg = _segment_containing(e, segs)
    if seg is None:
        return np.nan
    pw = _window_indices(e, seg, 1, horizon)
    if pw is None:
        return np.nan
    pc = cv[pw]
    pc_ok = np.isfinite(pc)
    if int(pc_ok.sum()) < 2:
        return np.nan
    e_price = cv[e]
    if not np.isfinite(e_price) or e_price <= _EPS:
        return np.nan
    pc_fin = pc[pc_ok]
    pr = r[pw]
    pr_fin = pr[np.isfinite(pr)]
    pre_w = _window_indices(e, seg, -horizon, -1)
    pre_r = r[pre_w] if pre_w is not None else np.array([])
    pre_r = pre_r[np.isfinite(pre_r)]

    if output == "retention":
        return float(pc_fin[-1] / e_price - 1.0)

    path = np.concatenate([[e_price], pc_fin])
    running = np.maximum.accumulate(path)
    if output == "max_drawdown":
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = path / running - 1.0
        return float(np.min(dd))
    if output == "giveback":
        peak = float(np.max(path))
        peak_idx = int(np.argmax(path))
        min_after = float(np.min(path[peak_idx:]))
        if peak <= _EPS:
            return np.nan
        return min_after / peak - 1.0
    if output == "vol_ratio":
        if len(pr_fin) < 2 or len(pre_r) < 2:
            return np.nan
        pre_std = float(np.std(pre_r, ddof=1))
        if pre_std <= _EPS:
            return np.nan
        return float(np.std(pr_fin, ddof=1)) / pre_std
    if output == "volume_ratio":
        pv = vv[pw]
        pv = pv[np.isfinite(pv)]
        pre_v = vv[pre_w] if pre_w is not None else np.array([])
        pre_v = pre_v[np.isfinite(pre_v)]
        pre_sum = float(np.sum(pre_v))
        if pre_sum <= _EPS:
            return np.nan
        return float(np.sum(pv)) / pre_sum
    if output == "amount_ratio":
        pa = av[pw]
        pa = pa[np.isfinite(pa)]
        pre_a = av[pre_w] if pre_w is not None else np.array([])
        pre_a = pre_a[np.isfinite(pre_a)]
        pre_sum = float(np.sum(pre_a))
        if pre_sum <= _EPS:
            return np.nan
        return float(np.sum(pa)) / pre_sum
    if output == "vwap_hold":
        pv = vv[pw]
        pa = av[pw]
        ok = np.isfinite(pv) & np.isfinite(pa) & (pv > 0)
        if int(ok.sum()) == 0:
            return np.nan
        vwap = float(np.sum(pa[ok])) / float(np.sum(pv[ok]))
        if vwap <= _EPS:
            return np.nan
        return vwap / e_price - 1.0
    if output == "recovery_half_life":
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = path / running - 1.0
        trough_idx = int(np.argmin(dd))
        trough = path[trough_idx]
        peak_at_trough = running[trough_idx]
        if peak_at_trough - trough <= _EPS:
            return np.nan
        halfway = trough + 0.5 * (peak_at_trough - trough)
        for k in range(trough_idx + 1, len(path)):
            if path[k] >= halfway:
                return float(k)
        return np.nan
    raise ValueError(f"unknown output {output!r}")


def _post_impulse_response_day(
    cv: np.ndarray, vv: np.ndarray, av: np.ndarray, times: np.ndarray,
    direction: str, threshold: str, z: float, horizon: int, output: str,
    merge_gap: int = 2,
) -> float:
    r = _intraday_returns(cv, times)
    episodes, _ = _detect_impulse_episodes(
        cv, times, direction=direction, threshold=threshold, z=z, merge_gap=merge_gap,
    )
    if not episodes:
        return np.nan
    segs = _bar_segments(times)
    results: list[float] = []
    for (_, e) in episodes:
        val = _post_response_event(cv, vv, av, r, times, segs, int(e), int(horizon), output)
        if np.isfinite(val):
            results.append(float(val))
    if not results:
        return np.nan
    return float(np.mean(results))


@register_operator(
    name="intra_post_impulse_response",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_post_impulse_response",
    source="intraday.event_response",
    backend="pandas_numpy",
    status="implemented",
)
class IntraPostImpulseResponse(SessionAggregationOperator):
    """脉冲后响应：retention/giveback/max_drawdown/vol & volume & amount 比/vwap_hold/半恢复期。"""

    metadata = metadata(
        "intra_post_impulse_response",
        "每个脉冲事件后的 horizon 响应指标（事件结束时锚定，段内截断）。",
        ["price", "volume", "amount", "direction", "threshold", "z", "horizon", "output"],
        unit="ratio", cost=9,
    )

    def _calculate_series(
        self, price, volume, amount, direction="up", threshold="robust_z", z=3.0,
        horizon=30, output="retention", session_tz=None, **_,
    ):
        direction = str(direction).lower()
        if direction not in {"up", "down"}:
            raise ValueError(f"unknown direction {direction!r}")
        threshold = str(threshold).lower()
        if threshold not in {"robust_z", "vol_scaled", "quantile"}:
            raise ValueError(f"unknown threshold {threshold!r}")
        z = float(z)
        if not np.isfinite(z):
            raise ValueError("z must be finite")
        if threshold == "quantile" and not 0.0 <= z <= 1.0:
            raise ValueError("z must be in [0, 1] for quantile threshold")
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        output = str(output).lower()
        if output not in {
            "retention", "giveback", "max_drawdown", "vol_ratio",
            "volume_ratio", "amount_ratio", "vwap_hold", "recovery_half_life",
        }:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        volume = session_local(volume, session_tz)
        amount = session_local(amount, session_tz)
        return _daily_with_times(
            [price, volume, amount],
            lambda a, b, c, t: _post_impulse_response_day(
                a, b, c, t, direction, threshold, z, int(horizon), output,
            ),
        )


# ---------------------------------------------------------------------------
# 5. intra_probe_outcome_score
# ---------------------------------------------------------------------------

def _probe_outcome_score_day(
    cv: np.ndarray, vv: np.ndarray, av: np.ndarray, times: np.ndarray,
    direction: str, z: float, probe_horizon: int, response_horizon: int, output: str,
    merge_gap: int = 2,
) -> float:
    r = _intraday_returns(cv, times)
    episodes, evmask = _detect_impulse_episodes(
        cv, times, direction=direction, threshold="robust_z", z=z, merge_gap=merge_gap,
    )
    if len(episodes) < 2:
        return np.nan
    segs = _bar_segments(times)
    strength: list[float] = []
    givebacks: list[float] = []
    vwap_holds: list[float] = []
    parts: list[float] = []
    vol_fin = vv[np.isfinite(vv)]
    session_vol_sum = float(np.sum(vol_fin)) if vol_fin.size else np.nan
    for (_, e) in episodes:
        e = int(e)
        seg = _segment_containing(e, segs)
        if seg is None:
            continue
        probe_w = _window_indices(e, seg, 0, probe_horizon)
        if probe_w is None:
            continue
        rr = np.abs(r[probe_w])
        rr = rr[np.isfinite(rr)]
        if rr.size == 0:
            continue
        strength.append(float(np.max(rr)))
        gb = _post_response_event(cv, vv, av, r, times, segs, e, response_horizon, "giveback")
        vw = _post_response_event(cv, vv, av, r, times, segs, e, response_horizon, "vwap_hold")
        if not np.isfinite(gb) or not np.isfinite(vw):
            continue
        givebacks.append(gb)
        vwap_holds.append(vw)
        post_w = _window_indices(e, seg, 1, response_horizon)
        pv = vv[post_w] if post_w is not None else np.array([])
        pv = pv[np.isfinite(pv)]
        if not np.isfinite(session_vol_sum) or session_vol_sum <= _EPS:
            parts.append(np.nan)
        else:
            parts.append(float(np.sum(pv)) / session_vol_sum)
    if len(strength) < 2 or len(givebacks) < 2 or len(vwap_holds) < 2:
        return np.nan
    parts_arr = np.asarray(parts, dtype=float)
    if not np.isfinite(parts_arr).all():
        return np.nan

    def _zscore(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        sd = float(np.std(arr, ddof=1))
        if not np.isfinite(sd) or sd <= _EPS:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / sd

    s_z = _zscore(np.asarray(strength))
    g_z = _zscore(np.asarray(givebacks))
    v_z = _zscore(np.asarray(vwap_holds))
    if output == "score":
        per = 0.4 * s_z - 0.3 * g_z + 0.2 * v_z + 0.1 * parts_arr
        return float(np.mean(per))
    if output == "failure":
        return float(np.mean(g_z > 0))
    if output == "holding":
        return float(np.mean(v_z > 0))
    if output == "second_push":
        counts: list[float] = []
        for (_, e) in episodes:
            e = int(e)
            seg = _segment_containing(e, segs)
            if seg is None:
                counts.append(0.0)
                continue
            post_w = _window_indices(e, seg, 1, response_horizon)
            if post_w is None:
                counts.append(0.0)
                continue
            counts.append(1.0 if bool(np.any(evmask[post_w])) else 0.0)
        return float(np.mean(counts))
    raise ValueError(f"unknown output {output!r}")


@register_operator(
    name="intra_probe_outcome_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_probe_outcome_score",
    source="intraday.event_response",
    backend="pandas_numpy",
    status="implemented",
)
class IntraProbeOutcomeScore(SessionAggregationOperator):
    """复合可观测探针评分：强度/回吐/VWAP 持有/参与度的标准化组合。"""

    metadata = metadata(
        "intra_probe_outcome_score",
        "脉冲事件复合评分（0.4*strength_z - 0.3*giveback_z + 0.2*vwap_z + 0.1*participation）。",
        ["price", "volume", "amount", "direction", "z", "probe_horizon", "response_horizon", "output"],
        unit="level", cost=10,
    )

    def _calculate_series(
        self, price, volume, amount, direction="up", z=3.0, probe_horizon=10,
        response_horizon=30, output="score", session_tz=None, **_,
    ):
        direction = str(direction).lower()
        if direction not in {"up", "down", "both"}:
            raise ValueError(f"unknown direction {direction!r}")
        z = float(z)
        if not np.isfinite(z) or z <= 0:
            raise ValueError("z must be a positive finite number")
        if int(probe_horizon) < 0:
            raise ValueError("probe_horizon must be >= 0")
        if int(response_horizon) < 1:
            raise ValueError("response_horizon must be >= 1")
        output = str(output).lower()
        if output not in {"score", "failure", "holding", "second_push"}:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        volume = session_local(volume, session_tz)
        amount = session_local(amount, session_tz)
        return _daily_with_times(
            [price, volume, amount],
            lambda a, b, c, t: _probe_outcome_score_day(
                a, b, c, t, direction, z, int(probe_horizon), int(response_horizon), output,
            ),
        )


# ---------------------------------------------------------------------------
# 6. intra_supply_absorption_score
# ---------------------------------------------------------------------------

def _absorption_event(
    cv: np.ndarray, vv: np.ndarray, av: np.ndarray, r: np.ndarray,
    times: np.ndarray, segs: list[tuple[int, int]], e: int, horizon: int,
    output: str, session_mean_vol: float,
) -> float:
    seg = _segment_containing(e, segs)
    if seg is None:
        return np.nan
    pw = _window_indices(e, seg, 1, horizon)
    if pw is None:
        return np.nan
    pr = r[pw]
    pr = pr[np.isfinite(pr)]
    pv = vv[pw]
    pv = pv[np.isfinite(pv)]
    pa = av[pw]
    pa = pa[np.isfinite(pa)]
    if output == "price_per_amount":
        pa_sum = float(np.sum(pa))
        if pa_sum <= _EPS:
            return np.nan
        return float(np.sum(np.abs(pr))) / pa_sum
    if output == "downside_resilience":
        e_price = cv[e]
        pc = cv[pw]
        pc = pc[np.isfinite(pc)]
        if not np.isfinite(e_price) or e_price <= _EPS or pc.size < 1:
            return np.nan
        path = np.concatenate([[e_price], pc])
        running = np.maximum.accumulate(path)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = float(np.min(path / running - 1.0))
        return 1.0 - dd
    if output == "absorption":
        pv_sum = float(np.sum(pv))
        if pv_sum <= _EPS:
            return np.nan
        price_impact = float(np.sum(np.abs(pr))) / (pv_sum / session_mean_vol)
        return 1.0 / (1.0 + price_impact)
    if output == "volume_no_drop":
        pre_w = _window_indices(e, seg, -horizon, -1)
        pre_v = vv[pre_w] if pre_w is not None else np.array([])
        pre_v = pre_v[np.isfinite(pre_v)]
        if pre_v.size == 0 or float(np.mean(pre_v)) <= _EPS:
            return np.nan
        if pv.size == 0:
            return np.nan
        return float(np.mean(pv)) / float(np.mean(pre_v))
    raise ValueError(f"unknown output {output!r}")


def _supply_absorption_day(
    cv: np.ndarray, vv: np.ndarray, av: np.ndarray, times: np.ndarray,
    event: str, horizon: int, output: str, merge_gap: int = 2,
) -> float:
    r = _intraday_returns(cv, times)
    direction = {"up_impulse": "up", "down_impulse": "down", "all": "both"}[event]
    episodes, _ = _detect_impulse_episodes(
        cv, times, direction=direction, threshold="robust_z", z=3.0, merge_gap=merge_gap,
    )
    if not episodes:
        return np.nan
    vol_fin = vv[np.isfinite(vv)]
    if vol_fin.size == 0:
        return np.nan
    session_mean_vol = float(np.mean(vol_fin))
    if not np.isfinite(session_mean_vol) or session_mean_vol <= _EPS:
        return np.nan
    segs = _bar_segments(times)
    results: list[float] = []
    for (_, e) in episodes:
        val = _absorption_event(cv, vv, av, r, times, segs, int(e), int(horizon), output, session_mean_vol)
        if np.isfinite(val):
            results.append(float(val))
    if not results:
        return np.nan
    return float(np.mean(results))


@register_operator(
    name="intra_supply_absorption_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_supply_absorption_score",
    source="intraday.event_response",
    backend="pandas_numpy",
    status="implemented",
)
class IntraSupplyAbsorptionScore(SessionAggregationOperator):
    """供给侧吸收能力：价格冲击吸收 / 每元冲击 / 成交量不掉 / 下行韧性。"""

    metadata = metadata(
        "intra_supply_absorption_score",
        "脉冲后的流动性吸收（absorption/price_per_amount/volume_no_drop/downside_resilience）。",
        ["price", "volume", "amount", "event", "horizon", "output"],
        unit="ratio", cost=8,
    )

    def _calculate_series(
        self, price, volume, amount, event="all", horizon=30, output="absorption",
        session_tz=None, **_,
    ):
        event = str(event).lower()
        if event not in {"up_impulse", "down_impulse", "all"}:
            raise ValueError(f"unknown event {event!r}")
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        output = str(output).lower()
        if output not in {
            "absorption", "price_per_amount", "volume_no_drop", "downside_resilience",
        }:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        volume = session_local(volume, session_tz)
        amount = session_local(amount, session_tz)
        return _daily_with_times(
            [price, volume, amount],
            lambda a, b, c, t: _supply_absorption_day(a, b, c, t, event, int(horizon), output),
        )


# ---------------------------------------------------------------------------
# 7. intra_consolidation_quality
# ---------------------------------------------------------------------------

def _consolidation_event(
    cv: np.ndarray, vv: np.ndarray, av: np.ndarray, r: np.ndarray,
    times: np.ndarray, segs: list[tuple[int, int]], e: int, horizon: int, output: str,
) -> float:
    seg = _segment_containing(e, segs)
    if seg is None:
        return np.nan
    pw = _window_indices(e, seg, 1, horizon)
    if pw is None:
        return np.nan
    pc = cv[pw]
    pc_ok = np.isfinite(pc)
    if int(pc_ok.sum()) < 2:
        return np.nan
    pc_fin = pc[pc_ok]
    e_price = cv[e]
    if not np.isfinite(e_price) or e_price <= _EPS:
        return np.nan
    post_mean = float(np.mean(pc_fin))
    if post_mean <= _EPS:
        return np.nan
    post_range = float(np.max(pc_fin) - np.min(pc_fin))
    tightness = 1.0 - post_range / post_mean
    if output == "tightness":
        return tightness
    if output == "level":
        return post_mean / e_price - 1.0
    pr = r[pw]
    pr = pr[np.isfinite(pr)]
    pre_w = _window_indices(e, seg, -horizon, -1)
    pre_r = r[pre_w] if pre_w is not None else np.array([])
    pre_r = pre_r[np.isfinite(pre_r)]
    if output == "vol_compression":
        if len(pr) < 2 or len(pre_r) < 2:
            return np.nan
        pre_std = float(np.std(pre_r, ddof=1))
        if pre_std <= _EPS:
            return np.nan
        return float(np.std(pr, ddof=1)) / pre_std
    pv = vv[pw]
    pv = pv[np.isfinite(pv)]
    pre_v = vv[pre_w] if pre_w is not None else np.array([])
    pre_v = pre_v[np.isfinite(pre_v)]
    if output == "volume_dryup":
        if pre_v.size == 0 or float(np.mean(pre_v)) <= _EPS:
            return np.nan
        if pv.size == 0:
            return np.nan
        return float(np.mean(pv)) / float(np.mean(pre_v))
    pre_c = cv[pre_w] if pre_w is not None else np.array([])
    pre_c = pre_c[np.isfinite(pre_c)]
    if output == "rising_floor":
        min_pre = float(np.min(pre_c)) if pre_c.size else np.nan
        min_post = float(np.min(pc_fin))
        if not np.isfinite(min_pre) or min_pre <= _EPS:
            return np.nan
        return (min_post - min_pre) / min_pre
    if output == "breakout_readiness":
        start = float(pc_fin[0])
        end = float(pc_fin[-1])
        if start <= _EPS:
            return np.nan
        move = end / start - 1.0
        sign = 1.0 if move > 0 else (-1.0 if move < 0 else 0.0)
        return sign * tightness
    raise ValueError(f"unknown output {output!r}")


def _consolidation_quality_day(
    cv: np.ndarray, vv: np.ndarray, av: np.ndarray, times: np.ndarray,
    trigger_z: float, horizon: int, output: str, merge_gap: int = 2,
) -> float:
    r = _intraday_returns(cv, times)
    episodes, _ = _detect_impulse_episodes(
        cv, times, direction="both", threshold="robust_z", z=trigger_z, merge_gap=merge_gap,
    )
    if not episodes:
        return np.nan
    segs = _bar_segments(times)
    results: list[float] = []
    for (_, e) in episodes:
        val = _consolidation_event(cv, vv, av, r, times, segs, int(e), int(horizon), output)
        if np.isfinite(val):
            results.append(float(val))
    if not results:
        return np.nan
    return float(np.mean(results))


@register_operator(
    name="intra_consolidation_quality",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_consolidation_quality",
    source="intraday.event_response",
    backend="pandas_numpy",
    status="implemented",
)
class IntraConsolidationQuality(SessionAggregationOperator):
    """触发事件后的整固窗口质量（tightness/level/vol_compression/volume_dryup/rising_floor/breakout）。"""

    metadata = metadata(
        "intra_consolidation_quality",
        "脉冲后整固窗口质量（紧度/水平/波动压缩/量能枯竭/抬升地板/突破准备度）。",
        ["price", "volume", "amount", "trigger", "trigger_z", "horizon", "output"],
        unit="ratio", cost=8,
        available_at="session_close", same_session_usable=False,
    )

    def _calculate_series(
        self, price, volume, amount, trigger="impulse", trigger_z=3.0, horizon=30,
        output="tightness", session_tz=None, **_,
    ):
        trigger = str(trigger).lower()
        if trigger != "impulse":
            raise ValueError("only trigger='impulse' is supported")
        trigger_z = float(trigger_z)
        if not np.isfinite(trigger_z) or trigger_z <= 0:
            raise ValueError("trigger_z must be a positive finite number")
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        output = str(output).lower()
        if output not in {
            "tightness", "level", "vol_compression", "volume_dryup",
            "rising_floor", "breakout_readiness",
        }:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        volume = session_local(volume, session_tz)
        amount = session_local(amount, session_tz)
        return _daily_with_times(
            [price, volume, amount],
            lambda a, b, c, t: _consolidation_quality_day(a, b, c, t, trigger_z, int(horizon), output),
        )


# ---------------------------------------------------------------------------
# 8. intra_response_curve_features
# ---------------------------------------------------------------------------

def _response_curve(
    cv: np.ndarray, av: np.ndarray, r: np.ndarray, times: np.ndarray,
    segs: list[tuple[int, int]], e: int, horizon: int, curve: str,
) -> np.ndarray | None:
    seg = _segment_containing(e, segs)
    if seg is None:
        return None
    pw = _window_indices(e, seg, 1, horizon)
    if pw is None:
        return None
    e_price = cv[e]
    if not np.isfinite(e_price) or e_price <= _EPS:
        return None
    pc = cv[pw]
    if curve == "price":
        return pc / e_price - 1.0
    if curve == "drawdown":
        y = pc / e_price - 1.0
        run = np.minimum.accumulate(y)
        return np.minimum(run, 0.0)
    if curve == "volatility":
        pr = r[pw]
        return pd.Series(pr).rolling(5, min_periods=2).std().to_numpy()
    if curve == "activity":
        return av[pw].astype(float)
    raise ValueError(f"unknown curve {curve!r}")


def _curve_summary(y: np.ndarray, output: str) -> float:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if output == "auc":
        return float(np.sum(y))
    if output == "slope":
        return _ols_slope(y, np.arange(n, dtype=float))
    if output == "curvature":
        return _quadratic_coef(y, np.arange(n, dtype=float))
    if output == "half_life":
        maxy = float(np.max(np.abs(y)))
        if maxy <= _EPS:
            return np.nan
        for k in range(n):
            if abs(y[k]) <= 0.5 * maxy:
                return float(k)
        return np.nan
    if output == "monotonicity":
        last_move = y[-1] - y[0]
        if abs(last_move) <= _EPS:
            return 0.5
        target = 1.0 if last_move > 0 else -1.0
        signs = np.sign(np.diff(y))
        return float(np.mean(signs == target))
    if output == "change_count":
        flips = 0
        prev = 0.0
        for s in np.sign(np.diff(y)):
            if s == 0:
                continue
            if prev != 0 and s != prev:
                flips += 1
            prev = s
        return float(flips)
    raise ValueError(f"unknown output {output!r}")


def _response_curve_features_day(
    cv: np.ndarray, av: np.ndarray, times: np.ndarray,
    horizon: int, curve: str, output: str, merge_gap: int = 2,
) -> float:
    r = _intraday_returns(cv, times)
    episodes, _ = _detect_impulse_episodes(
        cv, times, direction="both", threshold="robust_z", z=3.0, merge_gap=merge_gap,
    )
    if not episodes:
        return np.nan
    segs = _bar_segments(times)
    results: list[float] = []
    for (_, e) in episodes:
        y = _response_curve(cv, av, r, times, segs, int(e), int(horizon), curve)
        if y is None:
            continue
        yf = y[np.isfinite(y)]
        if yf.size < 2:
            continue
        val = _curve_summary(yf, output)
        if np.isfinite(val):
            results.append(float(val))
    if not results:
        return np.nan
    return float(np.mean(results))


@register_operator(
    name="intra_response_curve_features",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_response_curve_features",
    source="intraday.event_response",
    backend="pandas_numpy",
    status="implemented",
)
class IntraResponseCurveFeatures(SessionAggregationOperator):
    """事件时间响应曲线的形状特征（slope/curvature/auc/half_life/monotonicity/change_count）。"""

    metadata = metadata(
        "intra_response_curve_features",
        "触发后响应曲线形状（price/drawdown/volatility/activity 曲线）。",
        ["price", "activity", "trigger", "horizon", "curve", "output"],
        unit="level", cost=9,
    )

    def _calculate_series(
        self, price, activity, trigger="impulse", horizon=30, curve="price",
        output="slope", session_tz=None, **_,
    ):
        trigger = str(trigger).lower()
        if trigger != "impulse":
            raise ValueError("only trigger='impulse' is supported")
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        curve = str(curve).lower()
        if curve not in {"price", "drawdown", "volatility", "activity"}:
            raise ValueError(f"unknown curve {curve!r}")
        output = str(output).lower()
        if output not in {
            "slope", "curvature", "auc", "half_life", "monotonicity", "change_count",
        }:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        activity = session_local(activity, session_tz)
        return _daily_with_times(
            [price, activity],
            lambda a, b, t: _response_curve_features_day(a, b, t, int(horizon), curve, output),
        )


# ---------------------------------------------------------------------------
# 9. intra_liquidity_resilience_curve_fit
# ---------------------------------------------------------------------------

def _resilience_fit_event(
    cv: np.ndarray, av: np.ndarray, r: np.ndarray, times: np.ndarray,
    segs: list[tuple[int, int]], e: int, horizon: int, output: str,
) -> float:
    seg = _segment_containing(e, segs)
    if seg is None:
        return np.nan
    pw = _window_indices(e, seg, 1, horizon)
    if pw is None:
        return np.nan
    a = av[pw]
    a = a[np.isfinite(a)]
    if a.size < 3:
        return np.nan
    a_inf = float(np.min(a))
    res0 = float(a[0]) - a_inf
    if res0 <= _EPS:
        return np.nan
    if output == "residual":
        return res0
    if output == "asymptote":
        return a_inf
    if output == "slope":
        return _ols_slope(a, np.arange(len(a), dtype=float))
    resid = a - a_inf
    usable = resid > _EPS
    if int(usable.sum()) < 3:
        return np.nan
    k = np.arange(len(a), dtype=float)[usable]
    log_res = np.log(resid[usable])
    slope_fit = _ols_slope(log_res, k)
    if not np.isfinite(slope_fit) or slope_fit >= 0.0:
        return np.nan
    tau = -1.0 / slope_fit
    if output == "half_life":
        return tau * np.log(2.0)
    if output == "r2":
        return _ols_r2(log_res, k)
    raise ValueError(f"unknown output {output!r}")


def _liquidity_resilience_day(
    cv: np.ndarray, av: np.ndarray, times: np.ndarray,
    shock_threshold: float, horizon: int, output: str, merge_gap: int = 2,
) -> float:
    r = _intraday_returns(cv, times)
    fin = np.isfinite(r)
    if int(fin.sum()) < 3:
        return np.nan
    session_std = float(np.std(r[fin], ddof=1))
    if not np.isfinite(session_std) or session_std <= _EPS:
        return np.nan
    shock = fin & (np.abs(r) > float(shock_threshold) * session_std)
    episodes = _merge_episodes(np.flatnonzero(shock), merge_gap)
    if not episodes:
        return np.nan
    segs = _bar_segments(times)
    results: list[float] = []
    for (_, e) in episodes:
        val = _resilience_fit_event(cv, av, r, times, segs, int(e), int(horizon), output)
        if np.isfinite(val):
            results.append(float(val))
    if not results:
        return np.nan
    return float(np.mean(results))


@register_operator(
    name="intra_liquidity_resilience_curve_fit",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_liquidity_resilience_curve_fit",
    source="intraday.event_response",
    backend="pandas_numpy",
    status="implemented",
)
class IntraLiquidityResilienceCurveFit(SessionAggregationOperator):
    """流动性冲击后的活动度指数衰减拟合（half_life/residual/asymptote/slope/r2）。"""

    metadata = metadata(
        "intra_liquidity_resilience_curve_fit",
        "冲击后活动度 exp 衰减最小二乘拟合的半恢复期等特征。",
        ["price", "activity", "shock_threshold", "horizon", "output"],
        unit="count", cost=10,
    )

    def _calculate_series(
        self, price, activity, shock_threshold=2.5, horizon=30, output="half_life",
        session_tz=None, **_,
    ):
        shock_threshold = float(shock_threshold)
        if not np.isfinite(shock_threshold) or shock_threshold <= 0:
            raise ValueError("shock_threshold must be a positive finite number")
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1")
        output = str(output).lower()
        if output not in {"half_life", "residual", "asymptote", "slope", "r2"}:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        activity = session_local(activity, session_tz)
        return _daily_with_times(
            [price, activity],
            lambda a, b, t: _liquidity_resilience_day(a, b, t, shock_threshold, int(horizon), output),
        )


_CANONICALS.extend(
    [
        "intra_event_window_reduce",
        "intra_event_pre_post_contrast",
        "intra_impulse_event_detector",
        "intra_post_impulse_response",
        "intra_probe_outcome_score",
        "intra_supply_absorption_score",
        "intra_consolidation_quality",
        "intra_response_curve_features",
        "intra_liquidity_resilience_curve_fit",
    ]
)

register_surface(_CANONICALS)
