# -*- coding: utf-8 -*-
"""Intraday slice-profile and volume-at-price operators (2026-08, R43-R45).

This module implements the minute -> daily "slice profile" family: mask-reduced
intraday slices, multi-resolution within-session resampling, same-slot
cross-day z-scores, session-boundary jump structure, and a set of
volume-at-price (VAP) profile geometries (entropy / peak structure / supply
structure / value area) plus round-price clustering and barrier-response
statistics.

Contract
--------
* Causal: a day's scalar uses only that day's own minute data plus past days
  (cross-day references are ``shift(1)``-anchored trailing windows).  Nothing
  is emitted for day t before day t is complete.
* Missing-value policy: NaN is never treated as 0.  Degenerate / constant /
  short windows return NaN (fail-closed).  Param validation raises ``ValueError``.
* Every scalar parameter carries a default in its ``_calculate_series``
  signature.
* The lunch break (11:30 -- 13:00) is never bridged: within-session lags,
  resampling buckets and boundary windows all restart at 09:31 / 13:01.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import register_operator
from cleaned_operators.intraday._core import (
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

_CANONICALS: list[str] = []

# A-share session structure: morning 09:30..11:30 (minute-of-day 570..690),
# afternoon 13:00..15:00 (780..900).  Bar data normally starts at 09:31 / 13:01.
_MORNING = (570, 690)
_AFTERNOON = (780, 900)
_TRADING_MINUTES = 240  # 120 morning + 120 afternoon
_MIN_PROFILE_BARS = 8


# ---------------------------------------------------------------------------
# Shared kernels
# ---------------------------------------------------------------------------

def _session_positions(times: np.ndarray) -> np.ndarray:
    """Continuous trading-minute position in [0, 1] within the session.

    Morning bars map to ``1/240 .. 120/240``, afternoon bars to
    ``121/240 .. 240/240``; the lunch break occupies the missing positions.
    Bars outside the trading grid get NaN.
    """
    minutes = minute_of_day(np.asarray(times, dtype="datetime64[ns]"))
    morning = (minutes >= _MORNING[0]) & (minutes <= _MORNING[1])
    afternoon = (minutes >= _AFTERNOON[0]) & (minutes <= _AFTERNOON[1])
    idx = np.where(
        morning,
        minutes - _MORNING[0],
        np.where(afternoon, minutes - _AFTERNOON[0] + 120, np.nan),
    )
    return idx / float(_TRADING_MINUTES)


def _segment_of(times: np.ndarray) -> np.ndarray:
    """0 = morning, 1 = afternoon, -1 = outside the trading grid."""
    minutes = minute_of_day(np.asarray(times, dtype="datetime64[ns]"))
    morning = (minutes >= _MORNING[0]) & (minutes <= _MORNING[1])
    afternoon = (minutes >= _AFTERNOON[0]) & (minutes <= _AFTERNOON[1])
    return np.where(morning, 0, np.where(afternoon, 1, -1))


def _slice_mask(times: np.ndarray, window: object, slice_center: object) -> np.ndarray:
    """Boolean mask over a day's bars for the selected intraday interval.

    * ``slice_center`` not None -> normalized window ``[slice-0.25, slice+0.25]``
      of the session's continuous trading-minute index.
    * ``window`` == "All" -> whole session.
    * ``window`` integer >= 1 -> the trailing ``window`` bars of the session.
    """
    n = len(times)
    if slice_center is not None:
        sc = float(slice_center)
        if not 0.0 <= sc <= 1.0:
            raise ValueError("slice must be in [0, 1] or None")
        pos = _session_positions(times)
        return (pos >= sc - 0.25) & (pos <= sc + 0.25)
    if isinstance(window, str):
        if str(window).lower() != "all":
            raise ValueError(f"window must be 'All' or an integer, got {window!r}")
        return np.ones(n, dtype=bool)
    w = int(window)
    if w < 1:
        raise ValueError("window must be 'All' or an integer >= 1")
    return np.arange(n) >= n - w


def _percentile_rank(vals: np.ndarray) -> np.ndarray:
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n <= 1:
        return np.full(n, 0.5)
    return np.asarray([np.sum(v < vi) for vi in v], dtype=float) / (n - 1)


def _mask_keep(mask_vals: np.ndarray, side: str, q: float) -> np.ndarray:
    ranks = _percentile_rank(mask_vals)
    if side == "high":
        return ranks >= q
    if side == "low":
        return ranks <= (1.0 - q)
    raise ValueError("mask_side must be 'high' or 'low'")


_REDUCERS = {
    "mean", "std", "sum", "skew", "kurtosis", "median", "slope",
    "last_minus_first", "positive_share",
}


def _reduce_1d(a: np.ndarray, reducer: str) -> float:
    a = np.asarray(a, dtype=float)
    n = len(a)
    if n == 0:
        return np.nan
    if reducer == "mean":
        return float(np.mean(a))
    if reducer == "sum":
        return float(np.sum(a))
    if reducer == "median":
        return float(np.median(a))
    if reducer == "positive_share":
        return float(np.mean(a > 0))
    if reducer == "last_minus_first":
        return float(a[-1] - a[0])
    if n < 2:
        return np.nan
    sd = float(np.std(a))
    if reducer == "std":
        if sd <= _EPS:
            return np.nan
        return float(np.std(a, ddof=1))
    if reducer == "slope":
        if sd <= _EPS:
            return np.nan
        idx = np.arange(n, dtype=float)
        return float(np.polyfit(idx, a, 1)[0])
    if reducer == "skew":
        if sd <= _EPS or n < 3:
            return np.nan
        m = float(np.mean(a))
        return float(np.mean((a - m) ** 3) / sd ** 3)
    if reducer == "kurtosis":
        if sd <= _EPS or n < 4:
            return np.nan
        m = float(np.mean(a))
        return float(np.mean((a - m) ** 4) / sd ** 4 - 3.0)
    raise ValueError(f"unknown reducer {reducer!r}")


def _two_panels_daily(
    a_frame: pd.DataFrame,
    b_frame: pd.DataFrame,
    fn,
    min_finite: int = 2,
) -> pd.DataFrame:
    """Per-(instrument, day) fn(a_vals, b_vals, times) with grid guard."""
    a_frame, b_frame = as_panel(a_frame), as_panel(b_frame)
    require_same_session_grid(a_frame, b_frame)
    out: dict[str, pd.Series] = {}
    for inst in a_frame.columns:
        a, b = a_frame[inst], b_frame[inst]
        days = pd.DatetimeIndex(sorted(set(a.index.normalize())))
        joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna(subset=["a"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            av = np.asarray(group["a"], dtype=float)
            bv = np.asarray(group["b"], dtype=float)
            times = np.asarray(group.index, dtype="datetime64[ns]")
            if int(np.sum(np.isfinite(av))) < int(min_finite):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(av, bv, times))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float).reindex(days)
    return pd.DataFrame(out).sort_index()


def _three_panels_daily(
    a_frame: pd.DataFrame,
    b_frame: pd.DataFrame,
    c_frame: pd.DataFrame,
    fn,
    min_finite: int = 2,
) -> pd.DataFrame:
    """Per-(instrument, day) fn(a_vals, b_vals, c_vals, times) with grid guard."""
    a_frame, b_frame, c_frame = as_panel(a_frame), as_panel(b_frame), as_panel(c_frame)
    require_same_session_grid(a_frame, b_frame, c_frame)
    out: dict[str, pd.Series] = {}
    for inst in a_frame.columns:
        days = pd.DatetimeIndex(sorted(set(a_frame[inst].index.normalize())))
        joined = pd.concat(
            [a_frame[inst], b_frame[inst], c_frame[inst]],
            axis=1,
            keys=["a", "b", "c"],
        ).dropna(subset=["a"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            av = np.asarray(group["a"], dtype=float)
            bv = np.asarray(group["b"], dtype=float)
            cv = np.asarray(group["c"], dtype=float)
            times = np.asarray(group.index, dtype="datetime64[ns]")
            if int(np.sum(np.isfinite(av))) < int(min_finite):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(av, bv, cv, times))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float).reindex(days)
    return pd.DataFrame(out).sort_index()


def _slot_matrix(values: pd.Series) -> pd.DataFrame:
    """Per-stock minute series -> day x minute-of-day matrix (mean per slot)."""
    frame = values.to_frame("v")
    frame["slot"] = minute_of_day(frame.index.to_numpy(dtype="datetime64[ns]"))
    frame["day"] = frame.index.normalize()
    return frame.pivot_table(index="day", columns="slot", values="v", aggfunc="mean")


def _run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    cur = 0
    for m in mask:
        if m:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
                cur = 0
    if cur > 0:
        lengths.append(cur)
    return lengths


def _ts_values(series: pd.Series | None, ts_array: np.ndarray) -> np.ndarray:
    if series is None or len(ts_array) == 0:
        return np.array([], dtype=float)
    return series.reindex(pd.DatetimeIndex(ts_array)).to_numpy(dtype=float)


def _daily_value(series: pd.Series | None, day) -> float:
    if series is None or day is None:
        return np.nan
    try:
        v = series.get(day, np.nan)
        fv = float(v)
        return fv if np.isfinite(fv) else np.nan
    except (TypeError, ValueError, KeyError):
        return np.nan


# ---------------------------------------------------------------------------
# 1. intra_slice_mask_reduce
# ---------------------------------------------------------------------------

def _slice_mask_reduce_day(
    xv, mv, times, window, slice_center, mask_side, mask_q, reducer, min_bars,
) -> float:
    sel = _slice_mask(times, window, slice_center)
    xs = xv[sel]
    ms = mv[sel]
    valid = np.isfinite(xs) & np.isfinite(ms)
    if int(valid.sum()) < int(min_bars):
        return np.nan
    xv2 = xs[valid]
    mv2 = ms[valid]
    keep = _mask_keep(mv2, mask_side, mask_q)
    rx = xv2[keep]
    if len(rx) == 0:
        return np.nan
    if reducer not in ("sum", "mean") and len(rx) < 2:
        return np.nan
    return _reduce_1d(rx, reducer)


@register_operator(
    name="intra_slice_mask_reduce",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_slice_mask_reduce",
    source="intraday.slice_profile",
)
class IntraSliceMaskReduce(SessionAggregationOperator):
    """日内切片内按 mask 分位保留 bar，再对 x 做归约。"""

    metadata = metadata(
        "intra_slice_mask_reduce", "切片+mask 分位筛选后的日内归约。",
        ["x", "mask_field", "window", "slice", "mask_side", "mask_q", "reducer", "min_bars"],
        unit="level",
    )

    def _calculate_series(
        self, x, mask_field, window="All", slice=None, mask_side="high", mask_q=0.7,
        reducer="mean", min_bars=10, session_tz=None, **_,
    ):
        if mask_side not in ("high", "low"):
            raise ValueError("mask_side must be 'high' or 'low'")
        if not 0.0 < float(mask_q) < 1.0:
            raise ValueError("mask_q must be in (0, 1)")
        if reducer not in _REDUCERS:
            raise ValueError(f"unknown reducer {reducer!r}")
        if not (isinstance(window, str) or int(window) >= 1):
            raise ValueError("window must be 'All' or an integer >= 1")
        mb = int(min_bars)
        if mb < 1:
            raise ValueError("min_bars must be >= 1")
        if slice is not None and not 0.0 <= float(slice) <= 1.0:
            raise ValueError("slice must be in [0, 1] or None")
        x = session_local(x, session_tz)
        mask_field = session_local(mask_field, session_tz)
        return _two_panels_daily(
            x, mask_field,
            lambda a, b, t: _slice_mask_reduce_day(
                a, b, t, window, slice, mask_side, float(mask_q), reducer, mb,
            ),
        )


# ---------------------------------------------------------------------------
# 2. intra_slice_mask_pair_reduce
# ---------------------------------------------------------------------------

_PAIR_REDUCERS = {"corr", "cov", "slope", "intercept", "r2", "euclidean", "cosine"}


def _pair_reduce(px: np.ndarray, py: np.ndarray, reducer: str, min_pairs: int) -> float:
    n = len(px)
    if n < int(min_pairs):
        return np.nan
    if reducer in ("corr", "cov", "slope", "intercept", "r2"):
        vx = float(np.std(px))
        vy = float(np.std(py))
        if vx <= _EPS or vy <= _EPS:
            return np.nan
        mx, my = float(np.mean(px)), float(np.mean(py))
        cov = float(np.mean((px - mx) * (py - my)))
        if reducer == "cov":
            return cov
        if reducer == "corr":
            return float(cov / (vx * vy))
        slope = cov / (vy * vy)
        if reducer == "slope":
            return float(slope)
        intercept = mx - slope * my
        if reducer == "intercept":
            return float(intercept)
        pred = slope * py + intercept
        ss_res = float(np.sum((px - pred) ** 2))
        ss_tot = float(np.sum((px - mx) ** 2))
        if ss_tot <= _EPS:
            return np.nan
        return float(1.0 - ss_res / ss_tot)
    if reducer == "euclidean":
        return float(np.sqrt(np.mean((px - py) ** 2)))
    if reducer == "cosine":
        nx = float(np.linalg.norm(px))
        ny = float(np.linalg.norm(py))
        if nx <= _EPS or ny <= _EPS:
            return np.nan
        return float(np.dot(px, py) / (nx * ny))
    raise ValueError(f"unknown reducer {reducer!r}")


def _slice_mask_pair_day(
    xv, yv, mv, times, window, slice_center, mask_side, mask_q, y_lag, reducer, min_pairs,
) -> float:
    sel = _slice_mask(times, window, slice_center)
    seg = _segment_of(times)
    sel_idx = np.where(sel)[0]
    m_valid = np.isfinite(xv) & np.isfinite(mv)
    valid_in_slice = sel_idx[m_valid[sel_idx]]
    if int(len(valid_in_slice)) < int(min_pairs):
        return np.nan
    keep = _mask_keep(mv[valid_in_slice], mask_side, mask_q)
    retained = valid_in_slice[keep]
    px: list[float] = []
    py: list[float] = []
    for i in retained:
        j = i - int(y_lag)
        if j < 0:
            continue
        if seg[i] != seg[j] or seg[i] < 0:
            continue  # never bridge lunch / overnight
        if np.isfinite(xv[i]) and np.isfinite(yv[j]):
            px.append(float(xv[i]))
            py.append(float(yv[j]))
    return _pair_reduce(np.asarray(px, dtype=float), np.asarray(py, dtype=float), reducer, min_pairs)


@register_operator(
    name="intra_slice_mask_pair_reduce",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_slice_mask_pair_reduce",
    source="intraday.slice_profile",
)
class IntraSliceMaskPairReduce(SessionAggregationOperator):
    """切片+mask 筛选后对 (x, y_lag) 做双变量归约。"""

    metadata = metadata(
        "intra_slice_mask_pair_reduce", "切片+mask 筛选后的 x 与滞后 y 配对归约。",
        ["x", "y", "mask_field", "window", "slice", "mask_side", "mask_q", "y_lag", "reducer", "min_pairs"],
        unit="level",
    )

    def _calculate_series(
        self, x, y, mask_field, window="All", slice=None, mask_side="high", mask_q=0.7,
        y_lag=0, reducer="corr", min_pairs=10, session_tz=None, **_,
    ):
        if mask_side not in ("high", "low"):
            raise ValueError("mask_side must be 'high' or 'low'")
        if not 0.0 < float(mask_q) < 1.0:
            raise ValueError("mask_q must be in (0, 1)")
        if reducer not in _PAIR_REDUCERS:
            raise ValueError(f"unknown reducer {reducer!r}")
        if int(y_lag) < 0:
            raise ValueError("y_lag must be >= 0")
        if int(min_pairs) < 1:
            raise ValueError("min_pairs must be >= 1")
        if not (isinstance(window, str) or int(window) >= 1):
            raise ValueError("window must be 'All' or an integer >= 1")
        if slice is not None and not 0.0 <= float(slice) <= 1.0:
            raise ValueError("slice must be in [0, 1] or None")
        x = session_local(x, session_tz)
        y = session_local(y, session_tz)
        mask_field = session_local(mask_field, session_tz)
        return _three_panels_daily(
            x, y, mask_field,
            lambda a, b, c, t: _slice_mask_pair_day(
                a, b, c, t, window, slice, mask_side, float(mask_q), int(y_lag), reducer, int(min_pairs),
            ),
        )


# ---------------------------------------------------------------------------
# 3. intra_multiresolution_resample_reduce
# ---------------------------------------------------------------------------

_RESAMPLE_REDUCERS = {"mean", "std", "slope", "skew", "kurtosis", "last_minus_first"}


def _resample_day_bars(xv: np.ndarray, times: np.ndarray, k: int):
    """Non-overlapping k-minute bars within each session (09:31 / 13:01 restart)."""
    minutes = minute_of_day(np.asarray(times, dtype="datetime64[ns]"))
    morning = (minutes >= 571) & (minutes <= 690)
    afternoon = (minutes >= 781) & (minutes <= 900)
    n_morning = int(np.ceil(120 / k))
    n_afternoon = int(np.ceil(120 / k))
    bars = np.full(n_morning + n_afternoon, np.nan, dtype=float)
    for b in range(n_morning):
        lo = 571 + b * k
        hi = min(lo + k, 571 + 120)
        m = morning & (minutes >= lo) & (minutes < hi)
        vals = xv[m]
        if np.isfinite(vals).any():
            bars[b] = float(np.mean(vals[np.isfinite(vals)]))
    for b in range(n_afternoon):
        lo = 781 + b * k
        hi = min(lo + k, 781 + 120)
        m = afternoon & (minutes >= lo) & (minutes < hi)
        vals = xv[m]
        if np.isfinite(vals).any():
            bars[n_morning + b] = float(np.mean(vals[np.isfinite(vals)]))
    return bars, n_morning + n_afternoon


def _bars_reduce(bars: np.ndarray, reducer: str) -> float:
    b = bars[np.isfinite(bars)]
    n = len(b)
    if n == 0:
        return np.nan
    if reducer == "mean":
        return float(np.mean(b))
    if reducer == "last_minus_first":
        return float(b[-1] - b[0])
    if n < 2:
        return np.nan
    sd = float(np.std(b))
    if reducer == "std":
        if sd <= _EPS:
            return np.nan
        return float(np.std(b, ddof=1))
    if reducer == "slope":
        if sd <= _EPS:
            return np.nan
        idx = np.arange(n, dtype=float)
        return float(np.polyfit(idx, b, 1)[0])
    if reducer == "skew":
        if sd <= _EPS or n < 3:
            return np.nan
        m = float(np.mean(b))
        return float(np.mean((b - m) ** 3) / sd ** 3)
    if reducer == "kurtosis":
        if sd <= _EPS or n < 4:
            return np.nan
        m = float(np.mean(b))
        return float(np.mean((b - m) ** 4) / sd ** 4 - 3.0)
    raise ValueError(f"unknown reducer {reducer!r}")


@register_operator(
    name="intra_multiresolution_resample_reduce",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_multiresolution_resample_reduce",
    source="intraday.slice_profile",
)
class IntraMultiresolutionResampleReduce(SessionAggregationOperator):
    """日内多分辨率重采样后逐日归约，再取滞后窗口均值。"""

    metadata = metadata(
        "intra_multiresolution_resample_reduce", "k 分钟重采样 bar 的日度形态归约。",
        ["x", "bar_minutes", "lookback_days", "reducer", "session_split", "min_coverage"],
        unit="level",
    )

    def _calculate_series(
        self, x, bar_minutes=10, lookback_days=10, reducer="mean", session_split=True,
        min_coverage=0.8, session_tz=None, **_,
    ):
        k = int(bar_minutes)
        lb = int(lookback_days)
        mc = float(min_coverage)
        if k < 1:
            raise ValueError("bar_minutes must be >= 1")
        if lb < 1:
            raise ValueError("lookback_days must be >= 1")
        if reducer not in _RESAMPLE_REDUCERS:
            raise ValueError(f"unknown reducer {reducer!r}")
        if not 0.0 < mc <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")
        if type(session_split) is not bool:
            raise ValueError("session_split must be a bool")
        x = session_local(x, session_tz)
        out: dict[str, pd.Series] = {}
        for inst in x.columns:
            col = x[inst]
            summaries: dict[pd.Timestamp, float] = {}
            for day, group in col.groupby(col.index.normalize()):
                vals = np.asarray(group, dtype=float)
                times = np.asarray(group.index, dtype="datetime64[ns]")
                if not session_split:
                    # continuous single grid; still only ever consumes real bars
                    bars, expected = _resample_day_bars(vals, times, k)
                else:
                    bars, expected = _resample_day_bars(vals, times, k)
                coverage = float(np.isfinite(bars).sum()) / expected
                if coverage < mc:
                    summaries[day] = np.nan
                    continue
                summaries[day] = _bars_reduce(bars, reducer)
            s = pd.Series(summaries, dtype=float).sort_index()
            out[inst] = s.shift(1).rolling(lb, min_periods=2).mean()
        return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# 4. intra_same_slot_zscore
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_same_slot_zscore",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_same_slot_zscore",
    source="intraday.slice_profile",
)
class IntraSameSlotZscore(SessionAggregationOperator):
    """同分钟槽位相对历史均值的日内 z 值。"""

    metadata = metadata(
        "intra_same_slot_zscore", "分钟槽位跨日 z 值（仅用过去日）。",
        ["x", "history_days", "ddof", "min_history"], unit="level",
    )

    def _calculate_series(
        self, x, history_days=20, ddof=1, min_history=10, session_tz=None, **_,
    ):
        if int(ddof) not in (0, 1):
            raise ValueError("ddof must be 0 or 1")
        hd = int(history_days)
        mh = int(min_history)
        if hd < 1:
            raise ValueError("history_days must be >= 1")
        if mh < 1:
            raise ValueError("min_history must be >= 1")
        x = session_local(x, session_tz)
        out: dict[str, pd.Series] = {}
        for inst in x.columns:
            days = pd.DatetimeIndex(sorted(set(x[inst].index.normalize())))
            mat = _slot_matrix(x[inst])
            if mat.shape[0] == 0:
                out[inst] = pd.Series(np.nan, index=days)
                continue
            prior_mean = mat.shift(1).rolling(hd, min_periods=mh).mean()
            prior_std = mat.shift(1).rolling(hd, min_periods=mh).std(ddof=int(ddof))
            with np_errstate():
                z = (mat - prior_mean) / prior_std
            z = z.where(prior_std > _EPS)
            out[inst] = z.mean(axis=1).reindex(days)
        return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# 5. intra_session_boundary_jump
# ---------------------------------------------------------------------------

def _boundary_series(
    price_col: pd.Series,
    vol_col: pd.Series | None,
    pre_close_col: pd.Series | None,
    boundary: str,
    pre_bars: int,
    post_bars: int,
    output: str,
) -> pd.Series:
    day_groups = list(price_col.groupby(price_col.index.normalize()))
    result: dict[pd.Timestamp, float] = {}
    for i, (day, group) in enumerate(day_groups):
        times = group.index.to_numpy(dtype="datetime64[ns]")
        prices = np.asarray(group, dtype=float)
        minutes = minute_of_day(times)
        fin = np.isfinite(prices)
        morning = (minutes >= _MORNING[0]) & (minutes <= _MORNING[1])
        afternoon = (minutes >= _AFTERNOON[0]) & (minutes <= _AFTERNOON[1])

        gap = np.nan
        boundary_price = np.nan
        pre_ts: np.ndarray = np.array([], dtype="datetime64[ns]")
        post_ts: np.ndarray = np.array([], dtype="datetime64[ns]")

        if boundary == "lunch_restart":
            m_idx = np.where(fin & morning)[0]
            a_idx = np.where(fin & afternoon)[0]
            if len(m_idx) < 1 or len(a_idx) < 1:
                result[day] = np.nan
                continue
            boundary_price = float(prices[m_idx[-1]])
            gap = float(prices[a_idx[0]]) / boundary_price - 1.0
            pre_idx = m_idx[-(pre_bars + 1):-1] if len(m_idx) > 1 else m_idx[:0]
            post_idx = a_idx[:post_bars]
            pre_ts = group.index[pre_idx].to_numpy()
            post_ts = group.index[post_idx].to_numpy()
        elif boundary == "open":
            if pre_close_col is None:
                result[day] = np.nan
                continue
            boundary_price = _daily_value(pre_close_col, day)
            if not np.isfinite(boundary_price) or boundary_price <= _EPS:
                result[day] = np.nan
                continue
            d_idx = np.where(fin)[0]
            if len(d_idx) < 1:
                result[day] = np.nan
                continue
            gap = float(prices[d_idx[0]]) / boundary_price - 1.0
            post_idx = d_idx[:post_bars]
            post_ts = group.index[post_idx].to_numpy()
            if i > 0:
                prev_day, prev_group = day_groups[i - 1]
                prev_times = prev_group.index.to_numpy(dtype="datetime64[ns]")
                prev_prices = np.asarray(prev_group, dtype=float)
                prev_minutes = minute_of_day(prev_times)
                prev_fin = np.isfinite(prev_prices)
                prev_morning = prev_fin & (prev_minutes >= _MORNING[0]) & (prev_minutes <= _MORNING[1])
                prev_afternoon = prev_fin & (prev_minutes >= _AFTERNOON[0]) & (prev_minutes <= _AFTERNOON[1])
                combined = np.where(prev_morning | prev_afternoon)[0]
                if len(combined) >= 1:
                    take = combined[-pre_bars:]
                    pre_ts = prev_group.index[take].to_numpy()
        else:  # "close"
            d_idx = np.where(fin)[0]
            if len(d_idx) < pre_bars + 1:
                result[day] = np.nan
                continue
            pre_idx = d_idx[-pre_bars:]
            boundary_price = float(prices[pre_idx[0]])
            gap = float(prices[d_idx[-1]]) / boundary_price - 1.0
            pre_ts = group.index[pre_idx].to_numpy()

        pre_vals = _ts_values(price_col, pre_ts)
        post_vals = _ts_values(price_col, post_ts)
        pre_std = float(np.std(pre_vals, ddof=1)) if len(pre_vals) >= 2 else np.nan
        out_val = np.nan
        if output == "gap":
            out_val = gap
        elif output == "normalized_gap":
            if np.isfinite(pre_std) and abs(pre_std) > _EPS and np.isfinite(gap):
                out_val = gap / pre_std
        elif output == "recovery":
            if len(post_vals) >= 1 and np.isfinite(boundary_price) and boundary_price > _EPS:
                out_val = float(post_vals[-1]) / boundary_price - 1.0 - gap
        elif output == "volume_jump":
            pre_v = _ts_values(vol_col, pre_ts)
            post_v = _ts_values(vol_col, post_ts)
            if len(pre_v) >= 1 and len(post_v) >= 1:
                pm = float(np.mean(pre_v[np.isfinite(pre_v)]))
                po = float(np.mean(post_v[np.isfinite(post_v)]))
                if np.isfinite(pm) and np.isfinite(po) and abs(pm) > _EPS:
                    out_val = po / pm
        result[day] = float(out_val) if np.isfinite(out_val) else np.nan
    return pd.Series(result, dtype=float).sort_index()


@register_operator(
    name="intra_session_boundary_jump",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_session_boundary_jump",
    source="intraday.slice_profile",
)
class IntraSessionBoundaryJump(SessionAggregationOperator):
    """会话边界跳空 / 缺口结构（开盘、午间、收盘）。"""

    metadata = metadata(
        "intra_session_boundary_jump", "开盘/午间/收盘边界跳空与恢复结构。",
        ["price", "volume", "pre_close", "boundary", "pre_bars", "post_bars", "output"],
        unit="ratio",
        extra_tags=["allow_panel_broadcast"],
    )

    def _calculate_series(
        self, price, volume=None, pre_close=None, boundary="lunch_restart", pre_bars=5,
        post_bars=5, output="gap", session_tz=None, **_,
    ):
        if boundary not in ("open", "lunch_restart", "close"):
            raise ValueError("boundary must be 'open', 'lunch_restart' or 'close'")
        if output not in ("gap", "normalized_gap", "recovery", "volume_jump"):
            raise ValueError(f"unknown output {output!r}")
        pb, po = int(pre_bars), int(post_bars)
        if pb < 1 or po < 1:
            raise ValueError("pre_bars and post_bars must be >= 1")
        price = session_local(price, session_tz)
        if volume is not None:
            volume = session_local(as_panel(volume), session_tz)
        pre_close_panel = None
        if pre_close is not None:
            pre_close_panel = as_panel(pre_close).copy()
            if isinstance(pre_close_panel.index, pd.DatetimeIndex):
                pre_close_panel.index = pre_close_panel.index.normalize()
        out: dict[str, pd.Series] = {}
        for inst in price.columns:
            vc = None if volume is None else volume[inst]
            pc = None if pre_close_panel is None else pre_close_panel[inst]
            out[inst] = _boundary_series(price[inst], vc, pc, boundary, pb, po, output)
        return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# Volume-at-price profile helpers (operators 6-9)
# ---------------------------------------------------------------------------

def _vap_day_profile(price_vals, volume_vals, bins, weighting, price_basis, normalize):
    """Normalized volume-at-price histogram -> (prof, pmin, pmax) or None."""
    p = np.asarray(price_vals, dtype=float)
    v = np.asarray(volume_vals, dtype=float)
    if price_basis == "close":
        basis = p
    elif price_basis in ("ohlc_typical", "vwap"):
        # The fixed signature only carries close + volume; no high/low/amount
        # panels are available, so these bases degrade to the close proxy.
        basis = p
    else:
        raise ValueError(f"price_basis must be one of close/ohlc_typical/vwap, got {price_basis!r}")
    if weighting == "volume":
        w = v
    elif weighting == "amount":
        w = p * v
    elif weighting == "time":
        w = np.ones_like(p)
    else:
        raise ValueError(f"weighting must be one of volume/amount/time, got {weighting!r}")
    valid = np.isfinite(basis) & np.isfinite(w) & (w > 0)
    if int(valid.sum()) < _MIN_PROFILE_BARS:
        return None
    b = basis[valid]
    ww = w[valid]
    total = float(np.sum(ww))
    if total <= _EPS:
        return None
    pmin, pmax = float(b.min()), float(b.max())
    if pmax <= pmin + _EPS:
        return None
    nb = int(bins)
    bin_idx = np.clip(((b - pmin) / (pmax - pmin) * nb).astype(int), 0, nb - 1)
    hist = np.zeros(nb)
    np.add.at(hist, bin_idx, ww)
    if float(np.sum(hist)) <= _EPS:
        return None
    # Distributional outputs always consume a normalized profile.
    prof = hist / float(np.sum(hist))
    return prof, pmin, pmax


def _vap_bin_price(i, pmin, pmax, bins):
    return pmin + (i + 0.5) * (pmax - pmin) / bins


def _vap_value_area(prof, target_mass):
    poc = int(np.argmax(prof))
    lo = hi = poc
    mass = float(prof[poc])
    while mass < target_mass and (lo > 0 or hi < len(prof) - 1):
        left = prof[lo - 1] if lo > 0 else -1.0
        right = prof[hi + 1] if hi < len(prof) - 1 else -1.0
        if right > left:
            hi += 1
            mass += float(prof[hi])
        else:
            lo -= 1
            mass += float(prof[lo])
    return lo, hi, mass


def _vap_local_maxima(prof: np.ndarray) -> list[int]:
    peaks: list[int] = []
    n = len(prof)
    for i in range(n):
        left = prof[i - 1] if i > 0 else -np.inf
        right = prof[i + 1] if i < n - 1 else -np.inf
        if prof[i] > left and prof[i] > right:
            peaks.append(i)
    return peaks


def _vap_prominence(prof: np.ndarray, i: int) -> float:
    h = float(prof[i])
    n = len(prof)
    if i == 0:
        right = float(prof[1:].min()) if n > 1 else h
        return h - right
    if i == n - 1:
        left = float(prof[:i].min())
        return h - left
    left = float(prof[:i].min())
    right = float(prof[i + 1:].min())
    return h - max(left, right)


def _vap_panel_daily(price, volume, bins, weighting, price_basis, normalize, output_fn):
    price = as_panel(price)
    volume = as_panel(volume)
    require_same_session_grid(price, volume)
    out: dict[str, pd.Series] = {}
    for inst in price.columns:
        days = pd.DatetimeIndex(sorted(set(price[inst].index.normalize())))
        joined = pd.concat([price[inst], volume[inst]], axis=1, keys=["p", "v"]).dropna(subset=["p"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            pv = np.asarray(group["p"], dtype=float)
            vv = np.asarray(group["v"], dtype=float)
            res = _vap_day_profile(pv, vv, bins, weighting, price_basis, normalize)
            if res is None:
                per_day[day] = np.nan
                continue
            prof, pmin, pmax = res
            try:
                per_day[day] = float(output_fn(prof, pmin, pmax, pv))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float).reindex(days)
    return pd.DataFrame(out).sort_index()


def _last_finite(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return float(v[-1]) if len(v) else np.nan


# ---------------------------------------------------------------------------
# 6. intra_volume_at_price_profile
# ---------------------------------------------------------------------------

_VAP6_OUTPUTS = {
    "entropy", "skew", "kurtosis", "poc_price", "value_area_width",
    "tail_mass", "dip", "concentration",
}


def _vap6_output(prof, bins, output):
    if output == "entropy":
        nz = prof[prof > 0]
        return float(-np.sum(nz * np.log(nz)) / np.log(bins))
    idx = np.arange(bins)
    m = float(np.sum(prof * idx))
    v = float(np.sum(prof * (idx - m) ** 2))
    if output == "skew":
        if v <= _EPS:
            return np.nan
        return float(np.sum(prof * (idx - m) ** 3) / v ** 1.5)
    if output == "kurtosis":
        if v <= _EPS:
            return np.nan
        return float(np.sum(prof * (idx - m) ** 4) / v ** 2 - 3.0)
    if output == "poc_price":
        return float((int(np.argmax(prof)) + 0.5) / bins)
    if output == "value_area_width":
        lo, hi, _ = _vap_value_area(prof, 0.7)
        return float((hi - lo + 1) / bins)
    if output == "tail_mass":
        lo_cut = int(np.floor(0.1 * bins))
        hi_cut = int(np.floor(0.9 * bins))
        return float(prof[:lo_cut].sum() + prof[hi_cut:].sum())
    if output == "dip":
        return float(1.0 - prof.max() * bins)
    if output == "concentration":
        return float(np.sum(prof * prof))
    raise ValueError(f"unknown output {output!r}")


@register_operator(
    name="intra_volume_at_price_profile",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_volume_at_price_profile",
    source="intraday.slice_profile",
)
class IntraVolumeAtPriceProfile(SessionAggregationOperator):
    """量价分布（VAP）统计：熵 / 偏度 / 峰度 / POC / VA 宽度等。"""

    metadata = metadata(
        "intra_volume_at_price_profile", "量价分布直方图统计。",
        ["price", "volume", "bins", "weighting", "price_basis", "normalize", "output"],
        unit="level", available_at="session_close", same_session_usable=False,
    )

    def _calculate_series(
        self, price, volume, bins=64, weighting="volume", price_basis="close",
        normalize=True, output="entropy", session_tz=None, **_,
    ):
        if int(bins) < 8:
            raise ValueError("bins must be >= 8")
        if weighting not in ("volume", "amount", "time"):
            raise ValueError(f"unknown weighting {weighting!r}")
        if price_basis not in ("close", "ohlc_typical", "vwap"):
            raise ValueError(f"unknown price_basis {price_basis!r}")
        if output not in _VAP6_OUTPUTS:
            raise ValueError(f"unknown output {output!r}")
        if type(normalize) is not bool:
            raise ValueError("normalize must be a bool")
        price = session_local(price, session_tz)
        volume = session_local(volume, session_tz)
        nb = int(bins)

        def _out(prof, pmin, pmax, pv):
            return _vap6_output(prof, nb, output)

        return _vap_panel_daily(price, volume, nb, weighting, price_basis, normalize, _out)


# ---------------------------------------------------------------------------
# 7. intra_volume_profile_peak_geometry
# ---------------------------------------------------------------------------

_PEAK_OUTPUTS = {
    "peak_count", "top_peak_mass", "second_peak_mass", "peak_ratio",
    "peak_distance", "top_peak_width", "nearest_peak_distance", "valley_depth",
}


def _smooth_profile(prof: np.ndarray, smooth: int) -> np.ndarray:
    s = int(smooth)
    if s <= 0:
        return prof
    kernel = np.ones(2 * s + 1) / (2 * s + 1)
    return np.convolve(prof, kernel, mode="same")


def _peak_output(prof, bins, smooth, min_prominence, output):
    sp = _smooth_profile(prof, smooth)
    max_v = float(sp.max()) if len(sp) else 0.0
    if max_v <= _EPS:
        return np.nan
    thr = float(min_prominence) * max_v
    peaks = [i for i in _vap_local_maxima(sp) if _vap_prominence(sp, i) >= thr]
    if not peaks:
        return np.nan
    peaks_sorted = sorted(peaks, key=lambda i: sp[i], reverse=True)
    top = peaks_sorted[0]
    if output == "peak_count":
        return float(len(peaks_sorted))
    top_mass = float(sp[top])
    if output == "top_peak_mass":
        return top_mass
    second = peaks_sorted[1] if len(peaks_sorted) >= 2 else None
    second_mass = float(sp[second]) if second is not None else 0.0
    if output == "second_peak_mass":
        return second_mass
    if output == "peak_ratio":
        if top_mass <= _EPS:
            return np.nan
        return second_mass / top_mass
    if output == "peak_distance":
        if second is None:
            return np.nan
        return float(abs(top - second) / bins)
    if output == "top_peak_width":
        half = 0.5 * top_mass
        return float(np.sum(sp >= half))
    if output == "nearest_peak_distance":
        if len(peaks_sorted) < 2:
            return np.nan
        order = sorted(peaks_sorted)
        gaps = [order[i + 1] - order[i] for i in range(len(order) - 1)]
        return float(min(gaps) / bins)
    if output == "valley_depth":
        if second is None:
            return np.nan
        lo, hi = min(top, second), max(top, second)
        valley = float(sp[lo + 1:hi].min()) if hi - lo > 1 else top_mass
        base = min(top_mass, second_mass)
        if base <= _EPS:
            return np.nan
        return float((base - valley) / base)
    raise ValueError(f"unknown output {output!r}")


@register_operator(
    name="intra_volume_profile_peak_geometry",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_volume_profile_peak_geometry",
    source="intraday.slice_profile",
)
class IntraVolumeProfilePeakGeometry(SessionAggregationOperator):
    """VAP 峰结构：峰数 / 主次峰 / 距离 / 谷深等。"""

    metadata = metadata(
        "intra_volume_profile_peak_geometry", "量价分布峰形几何。",
        ["price", "volume", "bins", "smooth", "min_prominence", "output"],
        unit="level", available_at="session_close", same_session_usable=False,
    )

    def _calculate_series(
        self, price, volume, bins=64, smooth=2, min_prominence=0.05, output="peak_count",
        session_tz=None, **_,
    ):
        if int(bins) < 8:
            raise ValueError("bins must be >= 8")
        if int(smooth) < 0:
            raise ValueError("smooth must be >= 0")
        if not 0.0 < float(min_prominence) < 1.0:
            raise ValueError("min_prominence must be in (0, 1)")
        if output not in _PEAK_OUTPUTS:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        volume = session_local(volume, session_tz)
        nb = int(bins)
        sm = int(smooth)
        mp = float(min_prominence)

        def _out(prof, pmin, pmax, pv):
            return _peak_output(prof, nb, sm, mp, output)

        return _vap_panel_daily(price, volume, nb, "volume", "close", True, _out)


# ---------------------------------------------------------------------------
# 8. intra_volume_profile_supply_structure
# ---------------------------------------------------------------------------

_SUPPLY_OUTPUTS = {
    "overhead_mass", "near_overhead_mass", "under_price_mass", "supply_vacuum",
    "nearest_upper_peak", "nearest_lower_peak", "distance_weighted_overhang",
}


def _supply_output(prof, pmin, pmax, current, bins, decay, output):
    if not np.isfinite(current):
        return np.nan
    centers = np.asarray([_vap_bin_price(i, pmin, pmax, bins) for i in range(bins)])
    above = centers > current
    below = centers < current
    if output == "overhead_mass":
        return float(prof[above].sum())
    if output == "near_overhead_mass":
        within = (centers > current) & (centers <= current * 1.05)
        return float(prof[within].sum())
    if output == "under_price_mass":
        return float(prof[below].sum())
    max_v = float(prof.max())
    if max_v <= _EPS:
        return np.nan
    peaks = [i for i in _vap_local_maxima(prof) if _vap_prominence(prof, i) >= 0.02 * max_v]
    upper = [i for i in peaks if centers[i] > current]
    lower = [i for i in peaks if centers[i] < current]
    if output == "supply_vacuum":
        best = None
        for i in upper:
            between = (centers > current) & (centers < centers[i])
            if float(prof[between].sum()) <= _EPS:
                d = (centers[i] - current) / current
                if best is None or d < best:
                    best = d
        return float(best) if best is not None else np.nan
    if output == "nearest_upper_peak":
        if not upper:
            return np.nan
        return float(np.min(np.abs(centers[upper] - current) / current))
    if output == "nearest_lower_peak":
        if not lower:
            return np.nan
        return float(np.min(np.abs(centers[lower] - current) / current))
    if output == "distance_weighted_overhang":
        d = np.abs(centers[above] - current) / current
        return float(np.sum(prof[above] * np.exp(-float(decay) * d)))
    raise ValueError(f"unknown output {output!r}")


@register_operator(
    name="intra_volume_profile_supply_structure",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_volume_profile_supply_structure",
    source="intraday.slice_profile",
)
class IntraVolumeProfileSupplyStructure(SessionAggregationOperator):
    """VAP 供给结构：头顶量 / 真空 / 上峰距离等。"""

    metadata = metadata(
        "intra_volume_profile_supply_structure", "量价分布的供给结构。",
        ["price", "volume", "bins", "decay", "output"], unit="ratio",
        available_at="session_close", same_session_usable=False,
    )

    def _calculate_series(
        self, price, volume, bins=64, decay=20.0, output="overhead_mass", session_tz=None, **_,
    ):
        if int(bins) < 8:
            raise ValueError("bins must be >= 8")
        if float(decay) <= 0:
            raise ValueError("decay must be > 0")
        if output not in _SUPPLY_OUTPUTS:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        volume = session_local(volume, session_tz)
        nb = int(bins)
        dc = float(decay)

        def _out(prof, pmin, pmax, pv):
            return _supply_output(prof, pmin, pmax, _last_finite(pv), nb, dc, output)

        return _vap_panel_daily(price, volume, nb, "volume", "close", True, _out)


# ---------------------------------------------------------------------------
# 9. intra_volume_profile_value_area
# ---------------------------------------------------------------------------

_VALUE_AREA_OUTPUTS = {
    "value_area_width", "poc_price", "value_area_high", "value_area_low",
    "value_area_mid", "in_value_area",
}


def _value_area_output(prof, pmin, pmax, price_vals, bins, target_mass, output):
    lo, hi, _ = _vap_value_area(prof, target_mass)
    if output == "value_area_width":
        return float((hi - lo + 1) / bins)
    if output == "poc_price":
        return float((int(np.argmax(prof)) + 0.5) / bins)
    if output == "value_area_high":
        return float((hi + 0.5) / bins)
    if output == "value_area_low":
        return float((lo + 0.5) / bins)
    if output == "value_area_mid":
        return float(((lo + 0.5) + (hi + 0.5)) / 2.0 / bins)
    if output == "in_value_area":
        c = np.asarray(price_vals, dtype=float)
        c = c[np.isfinite(c)]
        if len(c) == 0:
            return np.nan
        bin_idx = np.clip(((c - pmin) / (pmax - pmin) * bins).astype(int), 0, bins - 1)
        return float(np.mean((bin_idx >= lo) & (bin_idx <= hi)))
    raise ValueError(f"unknown output {output!r}")


@register_operator(
    name="intra_volume_profile_value_area",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_volume_profile_value_area",
    source="intraday.slice_profile",
)
class IntraVolumeProfileValueArea(SessionAggregationOperator):
    """VAP 价值区间：VA 宽度 / POC / VA 高低 / 区间内占比。"""

    metadata = metadata(
        "intra_volume_profile_value_area", "量价分布价值区间。",
        ["price", "volume", "bins", "target_mass", "output"], unit="ratio",
        available_at="session_close", same_session_usable=False,
    )

    def _calculate_series(
        self, price, volume, bins=64, target_mass=0.7, output="value_area_width",
        session_tz=None, **_,
    ):
        if int(bins) < 8:
            raise ValueError("bins must be >= 8")
        if not 0.0 < float(target_mass) < 1.0:
            raise ValueError("target_mass must be in (0, 1)")
        if output not in _VALUE_AREA_OUTPUTS:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        volume = session_local(volume, session_tz)
        nb = int(bins)
        tm = float(target_mass)

        def _out(prof, pmin, pmax, pv):
            return _value_area_output(prof, pmin, pmax, pv, nb, tm, output)

        return _vap_panel_daily(price, volume, nb, "volume", "close", True, _out)


# ---------------------------------------------------------------------------
# 10. intra_round_price_clustering_share
# ---------------------------------------------------------------------------

_CLUSTER_OUTPUTS = {"share", "excess_share", "run_length"}


def _round_tick(price_range: float, lattice: float) -> float:
    """A-share convention: 0.01 when range/lattice < 20, else 0.05."""
    if price_range / lattice < 20.0:
        return 0.01
    return 0.05


@register_operator(
    name="intra_round_price_clustering_share",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_round_price_clustering_share",
    source="intraday.slice_profile",
)
class IntraRoundPriceClusteringShare(SessionAggregationOperator):
    """整价位聚类：贴近整数的 bar 占比及其超额。"""

    metadata = metadata(
        "intra_round_price_clustering_share", "整价位聚类占比 / 超额 / 连串长度。",
        ["price", "lattice", "tolerance_ticks", "window", "output", "min_bars"],
        unit="ratio", available_at="session_close", same_session_usable=False,
    )

    def _calculate_series(
        self, price, lattice=1.0, tolerance_ticks=0.25, window="All", output="share",
        min_bars=30, session_tz=None, **_,
    ):
        if float(lattice) <= 0:
            raise ValueError("lattice must be > 0")
        if float(tolerance_ticks) <= 0:
            raise ValueError("tolerance_ticks must be > 0")
        if output not in _CLUSTER_OUTPUTS:
            raise ValueError(f"unknown output {output!r}")
        if not (isinstance(window, str) or int(window) >= 1):
            raise ValueError("window must be 'All' or an integer >= 1")
        mb = int(min_bars)
        if mb < 1:
            raise ValueError("min_bars must be >= 1")
        price = session_local(price, session_tz)
        lat = float(lattice)
        tol_t = float(tolerance_ticks)
        out: dict[str, pd.Series] = {}
        for inst in price.columns:
            col = price[inst]
            per_day: dict[pd.Timestamp, float] = {}
            for day, group in col.groupby(col.index.normalize()):
                vals = np.asarray(group, dtype=float)
                times = np.asarray(group.index, dtype="datetime64[ns]")
                sel = _slice_mask(times, window, None)
                p = vals[sel]
                p = p[np.isfinite(p)]
                if len(p) < mb:
                    per_day[day] = np.nan
                    continue
                prange = float(p.max() - p.min())
                if prange <= _EPS:
                    per_day[day] = np.nan
                    continue
                tick = _round_tick(prange, lat)
                tol = tol_t * tick
                with np_errstate():
                    dist = np.minimum(p % lat, lat - (p % lat))
                clustered = dist <= tol
                if output == "share":
                    per_day[day] = float(np.mean(clustered))
                elif output == "excess_share":
                    expected = 2.0 * tol_t * tick / prange
                    per_day[day] = float(np.mean(clustered) - expected)
                else:  # run_length
                    if not clustered.any():
                        per_day[day] = np.nan
                    else:
                        runs = _run_lengths(clustered)
                        per_day[day] = float(np.mean(runs))
            out[inst] = pd.Series(per_day, dtype=float)
        return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# 11. intra_round_price_barrier_response
# ---------------------------------------------------------------------------

_BARRIER_OUTPUTS = {"cross_rate", "bounce_rate", "magnet_strength", "asymmetry"}


def _nearest_level(x: float, lattice: float) -> float:
    return float(round(x / lattice) * lattice)


def _barrier_day_stats(prices, times, lattice, tolerance_ticks):
    """Per-day approach-event tallies for round-price barrier response."""
    minutes = minute_of_day(np.asarray(times, dtype="datetime64[ns]"))
    fin = np.isfinite(prices)
    seg = np.where(
        (minutes >= _MORNING[0]) & (minutes <= _MORNING[1]), 0,
        np.where((minutes >= _AFTERNOON[0]) & (minutes <= _AFTERNOON[1]), 1, -1),
    )
    valid_prices = prices[fin]
    if len(valid_prices) < 2:
        return None
    prange = float(valid_prices.max() - valid_prices.min())
    if prange <= _EPS:
        return None
    tick = _round_tick(prange, lattice)
    tol = float(tolerance_ticks) * tick
    stats = {
        "crossed": 0, "bounced": 0,
        "up_crossed": 0, "up_bounced": 0, "down_crossed": 0, "down_bounced": 0,
        "approach_move": 0.0, "approach_n": 0,
        "control_move": 0.0, "control_n": 0,
    }
    n = len(prices)
    for i in range(1, n):
        if not fin[i] or not fin[i - 1]:
            continue
        if seg[i] != seg[i - 1] or seg[i] < 0:
            continue  # never bridge lunch / overnight
        p_prev = prices[i - 1]
        p_i = prices[i]
        hi = max(p_prev, p_i)
        lo = min(p_prev, p_i)
        L_hi = _nearest_level(hi, lattice)
        L_lo = _nearest_level(lo, lattice)
        d_hi = abs(hi - L_hi)
        d_lo = abs(lo - L_lo)
        level = None
        direction = None
        if d_hi <= tol and not (lo <= L_hi <= hi):
            level = L_hi
            direction = "up" if L_hi > hi else "down"
        elif d_lo <= tol and not (lo <= L_lo <= hi):
            level = L_lo
            direction = "up" if L_lo > hi else "down"
        if level is None:
            # matched control bar (not near any round-price level)
            move = abs(p_i - p_prev)
            if move > _EPS:
                stats["control_move"] += float(move)
                stats["control_n"] += 1
            continue
        move = abs(p_i - p_prev)
        if move > _EPS and abs(p_i - level) < abs(p_prev - level):
            stats["approach_move"] += float(move)
        stats["approach_n"] += 1
        if i + 1 < n and fin[i + 1] and seg[i + 1] == seg[i]:
            p_next = prices[i + 1]
            if direction == "up":
                if p_next > level:
                    stats["crossed"] += 1
                    stats["up_crossed"] += 1
                else:
                    stats["bounced"] += 1
                    stats["up_bounced"] += 1
            else:
                if p_next < level:
                    stats["crossed"] += 1
                    stats["down_crossed"] += 1
                else:
                    stats["bounced"] += 1
                    stats["down_bounced"] += 1
    return stats


@register_operator(
    name="intra_round_price_barrier_response",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_round_price_barrier_response",
    source="intraday.slice_profile",
)
class IntraRoundPriceBarrierResponse(SessionAggregationOperator):
    """整价位屏障反应：接近 / 穿越 / 弹回 / 磁吸强度 / 不对称。"""

    metadata = metadata(
        "intra_round_price_barrier_response", "整价位屏障穿越率与磁吸强度。",
        ["price", "lattice", "lookback_days", "tolerance_ticks", "output", "min_events"],
        unit="ratio", available_at="session_close", same_session_usable=False,
    )

    def _calculate_series(
        self, price, lattice=1.0, lookback_days=20, tolerance_ticks=1.0,
        output="cross_rate", min_events=5, session_tz=None, **_,
    ):
        if float(lattice) <= 0:
            raise ValueError("lattice must be > 0")
        if float(tolerance_ticks) <= 0:
            raise ValueError("tolerance_ticks must be > 0")
        if int(lookback_days) < 1:
            raise ValueError("lookback_days must be >= 1")
        if int(min_events) < 1:
            raise ValueError("min_events must be >= 1")
        if output not in _BARRIER_OUTPUTS:
            raise ValueError(f"unknown output {output!r}")
        price = session_local(price, session_tz)
        lat = float(lattice)
        tol_t = float(tolerance_ticks)
        lb = int(lookback_days)
        me = int(min_events)
        out: dict[str, pd.Series] = {}
        _zero_stats = {
            "crossed": 0.0, "bounced": 0.0,
            "up_crossed": 0.0, "up_bounced": 0.0, "down_crossed": 0.0, "down_bounced": 0.0,
            "approach_move": 0.0, "approach_n": 0.0,
            "control_move": 0.0, "control_n": 0.0,
        }
        for inst in price.columns:
            col = price[inst]
            per_day: dict[pd.Timestamp, dict] = {}
            for day, group in col.groupby(col.index.normalize()):
                vals = np.asarray(group, dtype=float)
                times = np.asarray(group.index, dtype="datetime64[ns]")
                stats = _barrier_day_stats(vals, times, lat, tol_t)
                per_day[day] = stats if stats is not None else dict(_zero_stats)
            df = pd.DataFrame(per_day).T
            rolled = df.shift(1).rolling(lb, min_periods=1).sum()
            result: dict[pd.Timestamp, float] = {}
            for day in df.index:
                r = rolled.loc[day]
                crossed = float(r["crossed"]) if pd.notna(r.get("crossed")) else 0.0
                bounced = float(r["bounced"]) if pd.notna(r.get("bounced")) else 0.0
                total = crossed + bounced
                if total < me:
                    result[day] = np.nan
                    continue
                if output == "cross_rate":
                    result[day] = crossed / total
                elif output == "bounce_rate":
                    result[day] = bounced / total
                elif output == "magnet_strength":
                    apm = float(r["approach_move"]) if pd.notna(r.get("approach_move")) else 0.0
                    apn = float(r["approach_n"]) if pd.notna(r.get("approach_n")) else 0.0
                    cvm = float(r["control_move"]) if pd.notna(r.get("control_move")) else 0.0
                    cvn = float(r["control_n"]) if pd.notna(r.get("control_n")) else 0.0
                    if apn >= 1 and cvn >= 1 and cvm > _EPS:
                        result[day] = (apm / apn) / (cvm / cvn)
                    else:
                        result[day] = np.nan
                else:  # asymmetry
                    uc = float(r["up_crossed"]) if pd.notna(r.get("up_crossed")) else 0.0
                    ub = float(r["up_bounced"]) if pd.notna(r.get("up_bounced")) else 0.0
                    dc = float(r["down_crossed"]) if pd.notna(r.get("down_crossed")) else 0.0
                    db = float(r["down_bounced"]) if pd.notna(r.get("down_bounced")) else 0.0
                    up_rate = uc / (uc + ub) if (uc + ub) >= 1 else np.nan
                    down_rate = dc / (dc + db) if (dc + db) >= 1 else np.nan
                    if np.isfinite(up_rate) and np.isfinite(down_rate):
                        result[day] = up_rate - down_rate
                    else:
                        result[day] = np.nan
            out[inst] = pd.Series(result, dtype=float).sort_index()
        return pd.DataFrame(out).sort_index()


_CANONICALS.extend(
    [
        "intra_slice_mask_reduce",
        "intra_slice_mask_pair_reduce",
        "intra_multiresolution_resample_reduce",
        "intra_same_slot_zscore",
        "intra_session_boundary_jump",
        "intra_volume_at_price_profile",
        "intra_volume_profile_peak_geometry",
        "intra_volume_profile_supply_structure",
        "intra_volume_profile_value_area",
        "intra_round_price_clustering_share",
        "intra_round_price_barrier_response",
    ]
)

register_surface(_CANONICALS)
