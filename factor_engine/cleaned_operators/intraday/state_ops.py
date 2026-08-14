# -*- coding: utf-8 -*-
"""Intraday discrete-state operators (2026-08 R47 pack).

Minute panels in.  Most operators emit one scalar per (TradeDate, Symbol) —
a daily panel — per the intraday family contract.  Two operators
(``intra_neighbor_event_class``, ``intra_range_gap_flag``) are *per-bar*
classifiers by their R42 spec: they consume a minute panel and emit a
minute panel of the same shape (one integer/float code per bar).  That is an
honest deviation from the family "minute->daily" grain and is declared in
each operator's metadata (``output_grain="minute"``) rather than being
silently folded into a daily scalar.

Contract
--------
* Causal: a day's scalar uses only that day's own minute data plus past days.
  Trailing ``window_days`` references are ``shift(1)``-anchored — for
  ``window_days > 1`` the window is the completed days strictly before today
  ``[t - window_days, t - 1]``; for ``window_days == 1`` the scalar is
  today's own value.  Nothing is emitted for day t before day t is complete.
* Missing-value policy: NaN is never treated as 0.  Null state bars are
  excluded from counts; constant windows, samples below ``min_events`` /
  ``min_slots`` and zero denominators return NaN (fail-closed).
* Session discipline: bars are grouped into A-share sessions (morning
  minute-of-day 570..690, afternoon 780..900).  Event pairing, dwell runs,
  transition counting and neighbour searches never bridge the lunch break or
  a day boundary.
* Discrete state is expected as numeric codes (``int``/``float``); NaN marks
  a null/absent state bar.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    SeriesOperator,
    register_operator,
    strict_int_runtime,
)
from cleaned_operators.intraday._core import (
    _EPS,
    DataDegeneracy,
    SessionAggregationOperator,
    as_panel,
    daily_agg,
    metadata,
    minute_of_day,
    register_surface,
    require_same_session_grid,
    safe_div,
    session_local,
)

_MORNING = (570, 690)    # 09:30 .. 11:30 Shanghai minute-of-day
_AFTERNOON = (780, 900)  # 13:00 .. 15:00 Shanghai minute-of-day
_MAX_GAP_MINUTES = 10

_CANONICALS: list[str] = []


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _same_session_segment(m_i: int, m_j: int) -> bool:
    """True when two minute-of-day values lie in the same continuous session."""
    return (
        (_MORNING[0] <= m_i <= _MORNING[1] and _MORNING[0] <= m_j <= _MORNING[1])
        or (_AFTERNOON[0] <= m_i <= _AFTERNOON[1] and _AFTERNOON[0] <= m_j <= _AFTERNOON[1])
    )


def _is_lunch_boundary(m_prev: int, m_cur: int) -> bool:
    """Morning's last bar directly followed by the afternoon's first bar."""
    return (
        _MORNING[0] <= m_prev <= _MORNING[1]
        and _AFTERNOON[0] <= m_cur <= _AFTERNOON[1]
    )


def _common_cols(*frames: pd.DataFrame) -> list[str]:
    cols = set(frames[0].columns)
    for f in frames[1:]:
        cols &= set(f.columns)
    return sorted(cols)


def _int_param(value: Any, name: str, lower: int = 1) -> int:
    """Strict integer gate (rejects bool/string/fractional, enforces lower)."""
    return strict_int_runtime(value, name, lower=lower)


def _target_mask(vals: np.ndarray, target: Any) -> np.ndarray:
    """bool array: finite bars whose discrete state equals ``target``.

    ``target`` is an exact numeric match; a ``None`` target selects every
    finite bar.  Null (NaN) state bars are always excluded.
    """
    finite = np.isfinite(vals)
    if target is None:
        return finite
    return finite & (vals == target)


def _window_days(days: list[pd.Timestamp], i: int, wd: int) -> list[pd.Timestamp]:
    """Days contributing to output day ``i``.

    ``wd == 1`` -> today only.  ``wd > 1`` -> the ``wd`` completed days
    strictly before today (``shift(1)`` anchor; causal, never today/future).
    """
    if wd == 1:
        return [days[i]]
    return days[max(0, i - wd):i]


def _trailing_sum(daily: pd.DataFrame, wd: int) -> pd.DataFrame:
    """Shift(1)-anchored rolling sum of a daily panel over ``wd`` days."""
    w = max(1, int(wd))
    return daily.shift(1).rolling(w, min_periods=1).sum()


def _safe_div_df(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    """safe_div keeping the DataFrame axis (safe_div itself returns ndarray)."""
    arr = safe_div(num.to_numpy(dtype=float), den.to_numpy(dtype=float))
    return pd.DataFrame(arr, index=num.index, columns=num.columns)


def _trailing_ratio(num: pd.DataFrame, den: pd.DataFrame, wd: int) -> pd.DataFrame:
    """Pooled ratio over the trailing window: sum(num)/sum(den), shift(1)."""
    w = max(1, int(wd))
    s_num = num.shift(1).rolling(w, min_periods=1).sum()
    s_den = den.shift(1).rolling(w, min_periods=1).sum()
    return _safe_div_df(s_num, s_den)


def _aligned_agg(
    frames: list[pd.DataFrame],
    keys: list[str],
    fn: Callable[..., float],
    min_finite: int = 1,
) -> pd.DataFrame:
    """Per-(instrument, day) ``fn(a_vals, b_vals, ..., times)`` over aligned rows.

    Unlike ``daily_agg_two`` / ``daily_agg_three`` this does NOT drop rows
    where the primary panel is NaN — an all-NaN day is preserved as an NaN
    output row (fail-closed) instead of silently vanishing from the panel.
    Only ``DataDegeneracy`` / ``ZeroDivisionError`` / ``OverflowError`` map to
    NaN; a plain ``ValueError`` (bad parameter / kernel bug) propagates.
    """
    require_same_session_grid(*frames)
    out: dict[str, pd.Series] = {}
    for inst in _common_cols(*frames):
        joined = pd.concat([f[inst] for f in frames], axis=1, keys=keys)
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals = [np.asarray(group[k], dtype=float) for k in keys]
            if int(np.sum(np.isfinite(vals[0]))) < int(min_finite):
                per_day[day] = np.nan
                continue
            times = np.asarray(group.index, dtype="datetime64[ns]")
            try:
                per_day[day] = float(fn(*vals, times))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def _moment_of(x: np.ndarray, moment: str) -> float:
    """Population moments (ddof=0) of a pooled sample.  Zero variance -> NaN
    for the scale-free skew/kurtosis (division by ~0 is a degeneracy)."""
    mu = float(np.mean(x))
    m2 = float(np.mean((x - mu) ** 2))
    sd = float(np.sqrt(m2))
    if moment == "std":
        return sd
    if sd < _EPS:
        return np.nan
    m3 = float(np.mean((x - mu) ** 3))
    m4 = float(np.mean((x - mu) ** 4))
    if moment == "skew":
        return m3 / (sd ** 3)
    return m4 / (sd ** 4)  # kurtosis (Pearson m4/m2^2, not excess)


def _day_event_gaps(vals: np.ndarray, times: np.ndarray, target: Any) -> np.ndarray:
    """Trading-minute gaps between consecutive target events within a session.

    Gaps never cross the lunch break or a day boundary (per-day input already
    isolates the day).  Days with fewer than two target events contribute an
    empty array.
    """
    event = _target_mask(vals, target)
    idx = np.where(event)[0]
    if len(idx) < 2:
        return np.array([])
    minutes = minute_of_day(times)
    prev = minutes[idx[:-1]].astype(float)
    cur = minutes[idx[1:]].astype(float)
    gap = cur - prev
    keep = np.array(
        [_same_session_segment(int(prev[k]), int(cur[k])) for k in range(len(gap))]
    )
    return gap[keep]


def _day_follow_pairs(
    xv: np.ndarray, sv: np.ndarray, times: np.ndarray, target: Any, lead: int
) -> tuple[np.ndarray, np.ndarray]:
    """(x_event, x_lead) pairs for every target event with an in-session lead.

    The lead bar sits exactly ``lead`` positions ahead, must be on the same
    calendar day and in the same session segment (never across lunch).  Pairs
    with a non-finite x on either side are dropped (pairwise complete).
    """
    minutes = minute_of_day(times)
    days = times.astype("datetime64[D]")
    event = _target_mask(sv, target)
    idx = np.where(event)[0]
    xe: list[float] = []
    xl: list[float] = []
    for i in idx:
        j = i + lead
        if j >= len(xv):
            continue
        if days[j] != days[i]:
            continue
        m_i, m_j = int(minutes[i]), int(minutes[j])
        if not _same_session_segment(m_i, m_j):
            continue
        a, b = xv[i], xv[j]
        if np.isfinite(a) and np.isfinite(b):
            xe.append(float(a))
            xl.append(float(b))
    return np.asarray(xe, dtype=float), np.asarray(xl, dtype=float)


def _ols_slope(xe: np.ndarray, xl: np.ndarray, min_events: int) -> float:
    """OLS slope (with intercept) of x_lead on x_event."""
    m = np.isfinite(xe) & np.isfinite(xl)
    if m.sum() < int(min_events):
        return np.nan
    a, b = xe[m], xl[m]
    va = float(np.var(a))
    if va <= _EPS:
        return np.nan
    cov = float(np.mean((a - np.mean(a)) * (b - np.mean(b))))
    return cov / va


def _pearson(xe: np.ndarray, xl: np.ndarray, min_events: int) -> float:
    """Pearson correlation between x_event and x_lead."""
    m = np.isfinite(xe) & np.isfinite(xl)
    if m.sum() < int(min_events):
        return np.nan
    a, b = xe[m], xl[m]
    if float(np.std(a)) < _EPS or float(np.std(b)) < _EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _dwell_runs(vals: np.ndarray, times: np.ndarray, target_state: Any) -> np.ndarray:
    """Run lengths of consecutive bars satisfying the state condition.

    A run breaks on: a bar that fails the condition, a lunch/session boundary,
    or (when ``target_state`` is None) a change of the discrete value.
    Consecutive rows are treated as consecutive bars (a missing minute inside
    a session does not reset a dwell run).
    """
    n = len(vals)
    minutes = minute_of_day(times)
    lengths: list[float] = []
    run = 0
    prev_minute: int | None = None
    prev_value: float | None = None
    for i in range(n):
        v = vals[i]
        finite = bool(np.isfinite(v))
        cond = finite if target_state is None else (finite and v == target_state)
        cross = False
        if prev_minute is not None:
            cross = _is_lunch_boundary(prev_minute, int(minutes[i]))
        changed = False
        if cond and target_state is None and prev_value is not None and finite and v != prev_value:
            changed = True
        if cond:
            if run > 0 and not cross and not changed:
                run += 1
            else:
                if run > 0:
                    lengths.append(float(run))
                run = 1
        else:
            if run > 0:
                lengths.append(float(run))
            run = 0
        prev_minute = int(minutes[i])
        prev_value = v
    if run > 0:
        lengths.append(float(run))
    return np.asarray(lengths, dtype=float)


def _bar_metadata(name: str, description: str, params: list[str], *, unit: str) -> OperatorMetadata:
    """Metadata for per-bar (minute-in, minute-out) intraday operators."""
    tags = [
        "intraday", "minute", "pit_safe", "causal",
        "typed_v2", "source_blocked", "grain_minute_to_minute",
        f"signature:{','.join(params)}->series", f"unit:{unit}",
        "cost:3", "domain:intraday",
    ]
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=tags,
        input_grain="minute",
        output_grain="minute",
    )


# ---------------------------------------------------------------------------
# 1. intra_state_count
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_state_count",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_count",
    source="intraday.state_ops",
    research_only=True,
)
class IntraStateCount(SessionAggregationOperator):
    """统计日内处于 target 离散状态的分钟 bar 数量（含跨日窗口）。"""

    metadata = metadata(
        "intra_state_count", "日内目标状态分钟数。",
        ["state", "target", "window_days"], unit="count",
    )

    def _calculate_series(self, state, target=None, window_days=1, session_tz=None, **_):
        wd = _int_param(window_days, "window_days")
        state = session_local(as_panel(state), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals)
            if not finite.any():
                return np.nan
            if target is None:
                return float(finite.sum())
            return float(np.sum(finite & (vals == target)))

        daily = daily_agg(state, _kernel, min_finite=1)
        if wd == 1:
            return daily
        return _trailing_sum(daily, wd)


# ---------------------------------------------------------------------------
# 2. intra_state_sum
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_state_sum",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_sum",
    source="intraday.state_ops",
    research_only=True,
)
class IntraStateSum(SessionAggregationOperator):
    """对目标状态 bar 的 x 求和（含跨日窗口）。"""

    metadata = metadata(
        "intra_state_sum", "目标状态 bar 的 x 值之和。",
        ["x", "state", "target", "window_days"], unit="level",
    )

    def _calculate_series(self, x, state, target, window_days=1, session_tz=None, **_):
        wd = _int_param(window_days, "window_days")
        x = session_local(as_panel(x), session_tz)
        state = session_local(as_panel(state), session_tz)

        def _kernel(xv, sv, times):
            m = np.isfinite(xv) & np.isfinite(sv) & (sv == target)
            if not m.any():
                return np.nan
            return float(np.sum(xv[m]))

        daily = _aligned_agg([x, state], ["x", "s"], _kernel, min_finite=1)
        if wd == 1:
            return daily
        return _trailing_sum(daily, wd)


# ---------------------------------------------------------------------------
# 3. intra_state_vwap
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_state_vwap",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_vwap",
    source="intraday.state_ops",
    research_only=True,
)
class IntraStateVwap(SessionAggregationOperator):
    """目标状态 bar 的成交额加权均价（pooled VWAP）。"""

    metadata = metadata(
        "intra_state_vwap", "目标状态 bar 的 VWAP。",
        ["price", "volume", "state", "target", "window_days"], unit="price",
    )

    def _calculate_series(self, price, volume, state, target, window_days=1, session_tz=None, **_):
        wd = _int_param(window_days, "window_days")
        price = session_local(as_panel(price), session_tz)
        volume = session_local(as_panel(volume), session_tz)
        state = session_local(as_panel(state), session_tz)

        def _num(pv, vv, sv, times):
            m = np.isfinite(pv) & np.isfinite(vv) & (vv > 0) & np.isfinite(sv) & (sv == target)
            if not m.any():
                return np.nan
            return float(np.sum(pv[m] * vv[m]))

        def _den(pv, vv, sv, times):
            m = np.isfinite(pv) & np.isfinite(vv) & (vv > 0) & np.isfinite(sv) & (sv == target)
            if not m.any():
                return np.nan
            return float(np.sum(vv[m]))

        num_daily = _aligned_agg([price, volume, state], ["p", "v", "s"], _num, min_finite=1)
        den_daily = _aligned_agg([price, volume, state], ["p", "v", "s"], _den, min_finite=1)
        if wd == 1:
            return _safe_div_df(num_daily, den_daily)
        return _trailing_ratio(num_daily, den_daily, wd)


# ---------------------------------------------------------------------------
# 4. intra_state_interval_moment
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_state_interval_moment",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_interval_moment",
    source="intraday.state_ops",
    research_only=True,
)
class IntraStateIntervalMoment(SessionAggregationOperator):
    """目标状态事件间隔的跨日矩（std/skew/kurtosis）。"""

    metadata = metadata(
        "intra_state_interval_moment", "目标状态事件间隔矩。",
        ["state", "target", "moment", "window_days", "min_events"], unit="level",
    )

    def _calculate_series(self, state, target, moment="std", window_days=20, min_events=4, session_tz=None, **_):
        if moment not in ("std", "skew", "kurtosis"):
            raise ValueError("moment must be one of {'std','skew','kurtosis'}")
        wd = _int_param(window_days, "window_days")
        me = _int_param(min_events, "min_events")
        state = session_local(as_panel(state), session_tz)
        out: dict[str, pd.Series] = {}
        for inst in state.columns:
            col = state[inst]
            day_gaps: dict[pd.Timestamp, np.ndarray] = {}
            for day, group in col.groupby(col.index.normalize()):
                vals = np.asarray(group, dtype=float)
                times = np.asarray(group.index, dtype="datetime64[ns]")
                day_gaps[day] = _day_event_gaps(vals, times, target)
            days = sorted(day_gaps.keys())
            per_day: dict[pd.Timestamp, float] = {}
            for i, day in enumerate(days):
                w = _window_days(days, i, wd)
                pooled = np.concatenate([day_gaps[d] for d in w]) if w else np.array([])
                if pooled.size < me:
                    per_day[day] = np.nan
                    continue
                per_day[day] = _moment_of(pooled, moment)
            out[inst] = pd.Series(per_day, dtype=float)
        return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# 5/6/7. intra_state_follow_ratio / follow_beta / follow_corr
# ---------------------------------------------------------------------------

def _follow_pooled(
    x: pd.DataFrame,
    state: pd.DataFrame,
    target: Any,
    lead: int,
    wd: int,
    stat: Callable[[np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """Pool (x_event, x_lead) pairs over the trailing window, then apply stat."""
    out: dict[str, pd.Series] = {}
    for inst in _common_cols(x, state):
        xc, sc = x[inst], state[inst]
        joined = pd.concat([xc, sc], axis=1, keys=["x", "s"])
        day_pairs: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
        for day, group in joined.groupby(joined.index.normalize()):
            xv = np.asarray(group["x"], dtype=float)
            sv = np.asarray(group["s"], dtype=float)
            times = np.asarray(group.index, dtype="datetime64[ns]")
            day_pairs[day] = _day_follow_pairs(xv, sv, times, target, lead)
        days = sorted(day_pairs.keys())
        per_day: dict[pd.Timestamp, float] = {}
        for i, day in enumerate(days):
            w = _window_days(days, i, wd)
            if not w:
                per_day[day] = np.nan
                continue
            xe = np.concatenate([day_pairs[d][0] for d in w]) if w else np.array([])
            xl = np.concatenate([day_pairs[d][1] for d in w]) if w else np.array([])
            if xe.size == 0:
                per_day[day] = np.nan
                continue
            per_day[day] = stat(xe, xl)
        out[inst] = pd.Series(per_day, dtype=float)
    return pd.DataFrame(out).sort_index()


@register_operator(
    name="intra_state_follow_ratio",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_follow_ratio",
    source="intraday.state_ops",
    research_only=True,
)
class IntraStateFollowRatio(SessionAggregationOperator):
    """目标事件后 lead 期 x 的累计比值 sum(x_lead)/sum(x_event)。"""

    metadata = metadata(
        "intra_state_follow_ratio", "事件后 x 相对事件 x 的累计比值。",
        ["x", "state", "target", "lead_bars", "window_days"], unit="ratio",
    )

    def _calculate_series(self, x, state, target, lead_bars=1, window_days=20, session_tz=None, **_):
        lb = _int_param(lead_bars, "lead_bars")
        wd = _int_param(window_days, "window_days")
        x = session_local(as_panel(x), session_tz)
        state = session_local(as_panel(state), session_tz)

        def _ratio(xe, xl):
            den = float(np.sum(xe))
            if den <= _EPS:
                return np.nan
            return float(np.sum(xl)) / den

        return _follow_pooled(x, state, target, lb, wd, _ratio)


@register_operator(
    name="intra_state_follow_beta",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_follow_beta",
    source="intraday.state_ops",
    research_only=True,
)
class IntraStateFollowBeta(SessionAggregationOperator):
    """事件 x 对 lead 后 x 的 OLS 斜率（含截距）。"""

    metadata = metadata(
        "intra_state_follow_beta", "事件后 x 对事件 x 的 OLS 斜率。",
        ["x", "state", "target", "lead_bars", "window_days", "min_events"], unit="level",
    )

    def _calculate_series(self, x, state, target, lead_bars=1, window_days=20, min_events=8, session_tz=None, **_):
        lb = _int_param(lead_bars, "lead_bars")
        wd = _int_param(window_days, "window_days")
        me = _int_param(min_events, "min_events")
        x = session_local(as_panel(x), session_tz)
        state = session_local(as_panel(state), session_tz)
        return _follow_pooled(x, state, target, lb, wd, lambda xe, xl: _ols_slope(xe, xl, me))


@register_operator(
    name="intra_state_follow_corr",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_follow_corr",
    source="intraday.state_ops",
    research_only=True,
)
class IntraStateFollowCorr(SessionAggregationOperator):
    """事件 x 与 lead 后 x 的 Pearson 相关。"""

    metadata = metadata(
        "intra_state_follow_corr", "事件后 x 与事件 x 的相关。",
        ["x", "state", "target", "lead_bars", "window_days", "min_events"], unit="ratio",
    )

    def _calculate_series(self, x, state, target, lead_bars=1, window_days=20, min_events=8, session_tz=None, **_):
        lb = _int_param(lead_bars, "lead_bars")
        wd = _int_param(window_days, "window_days")
        me = _int_param(min_events, "min_events")
        x = session_local(as_panel(x), session_tz)
        state = session_local(as_panel(state), session_tz)
        return _follow_pooled(x, state, target, lb, wd, lambda xe, xl: _pearson(xe, xl, me))


# ---------------------------------------------------------------------------
# 8. intra_state_pair_same_slot_corr
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_state_pair_same_slot_corr",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_pair_same_slot_corr",
    source="intraday.state_ops",
    research_only=True,
)
class IntraStatePairSameSlotCorr(SessionAggregationOperator):
    """两个状态在相同时刻槽上的逐日命中计数向量的 Pearson 相关。"""

    metadata = metadata(
        "intra_state_pair_same_slot_corr", "同槽位双状态命中计数的相关。",
        ["state_a", "target_a", "state_b", "target_b", "window_days", "min_slots"], unit="ratio",
    )

    def _calculate_series(self, state_a, target_a, state_b, target_b, window_days=20, min_slots=8, session_tz=None, **_):
        wd = _int_param(window_days, "window_days")
        ms = _int_param(min_slots, "min_slots")
        state_a = session_local(as_panel(state_a), session_tz)
        state_b = session_local(as_panel(state_b), session_tz)
        require_same_session_grid(state_a, state_b)
        out: dict[str, pd.Series] = {}
        for inst in _common_cols(state_a, state_b):
            sa = state_a[inst].to_numpy(dtype=float)
            sb = state_b[inst].to_numpy(dtype=float)
            times = state_a[inst].index.to_numpy(dtype="datetime64[ns]")
            day_arr = times.astype("datetime64[D]")
            slot_arr = minute_of_day(times)
            n = len(times)
            cnt_a = (np.isfinite(sa) & (sa == target_a)).astype(int)
            obs_a = np.isfinite(sa).astype(int)
            cnt_b = (np.isfinite(sb) & (sb == target_b)).astype(int)
            obs_b = np.isfinite(sb).astype(int)
            day_records: dict[pd.Timestamp, dict[int, list[int]]] = defaultdict(dict)
            for i in range(n):
                d = pd.Timestamp(day_arr[i])
                rec = day_records[d].setdefault(int(slot_arr[i]), [0, 0, 0, 0])
                rec[0] += int(cnt_a[i])
                rec[1] += int(cnt_b[i])
                rec[2] += int(obs_a[i])
                rec[3] += int(obs_b[i])
            days = sorted(day_records.keys())
            per_day: dict[pd.Timestamp, float] = {}
            for i, day in enumerate(days):
                w = _window_days(days, i, wd)
                if not w:
                    per_day[day] = np.nan
                    continue
                agg: dict[int, list[int]] = {}
                for d in w:
                    for slot, rec in day_records[d].items():
                        a = agg.setdefault(slot, [0, 0, 0, 0])
                        a[0] += rec[0]
                        a[1] += rec[1]
                        a[2] += rec[2]
                        a[3] += rec[3]
                ca: list[float] = []
                cb: list[float] = []
                for slot in sorted(agg.keys()):
                    rec = agg[slot]
                    if rec[2] > 0 and rec[3] > 0:  # both states observed at this slot
                        ca.append(float(rec[0]))
                        cb.append(float(rec[1]))
                if len(ca) < ms:
                    per_day[day] = np.nan
                    continue
                va = np.asarray(ca, dtype=float)
                vb = np.asarray(cb, dtype=float)
                if float(np.std(va)) < _EPS or float(np.std(vb)) < _EPS:
                    per_day[day] = np.nan
                    continue
                per_day[day] = float(np.corrcoef(va, vb)[0, 1])
            out[inst] = pd.Series(per_day, dtype=float)
        return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# 9. intra_state_dwell_stats
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_state_dwell_stats",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_dwell_stats",
    source="intraday.state_ops",
    research_only=True,
)
class IntraStateDwellStats(SessionAggregationOperator):
    """状态驻留 run 长度统计（mean/max/cv/last/share）。"""

    metadata = metadata(
        "intra_state_dwell_stats", "状态驻留 run 统计量。",
        ["state", "target_state", "output", "min_slots"], unit="level",
    )

    def _calculate_series(self, state, target_state=None, output="mean", min_slots=20, session_tz=None, **_):
        if output not in ("mean", "max", "cv", "last", "share"):
            raise ValueError("output must be one of {'mean','max','cv','last','share'}")
        ms = _int_param(min_slots, "min_slots")
        state = session_local(as_panel(state), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals)
            valid_count = int(finite.sum())
            if valid_count < ms:
                return np.nan
            if output == "share":
                if target_state is None:
                    return 1.0
                target_bars = int(np.sum(finite & (vals == target_state)))
                return float(target_bars) / float(valid_count)
            lengths = _dwell_runs(vals, times, target_state)
            if lengths.size == 0:
                return np.nan
            if output == "mean":
                return float(np.mean(lengths))
            if output == "max":
                return float(np.max(lengths))
            if output == "cv":
                mean = float(np.mean(lengths))
                if mean <= _EPS:
                    return np.nan
                return float(np.std(lengths)) / mean
            return float(lengths[-1])  # last

        return daily_agg(state, _kernel, min_finite=1)


# ---------------------------------------------------------------------------
# 10. intra_state_transition_entropy
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_state_transition_entropy",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_state_transition_entropy",
    source="intraday.state_ops",
    research_only=True,
)
class IntraStateTransitionEntropy(SessionAggregationOperator):
    """相邻 bar 状态转移矩阵的香农熵（可去自转移 + 归一化）。"""

    metadata = metadata(
        "intra_state_transition_entropy", "状态转移矩阵熵。",
        ["state", "min_slots", "normalize", "include_self"], unit="entropy",
    )

    def _calculate_series(self, state, min_slots=30, normalize=True, include_self=True, session_tz=None, **_):
        ms = _int_param(min_slots, "min_slots")
        if type(normalize) is not bool:
            raise ValueError("normalize must be a boolean")
        if type(include_self) is not bool:
            raise ValueError("include_self must be a boolean")
        state = session_local(as_panel(state), session_tz)

        def _kernel(vals, times):
            finite = np.isfinite(vals)
            if finite.sum() < ms:
                return np.nan
            codes = vals[finite]
            unique = np.unique(codes)
            if unique.size < 2:
                return np.nan
            code_map = {c: i for i, c in enumerate(unique)}
            k = int(unique.size)
            T = np.zeros((k, k), dtype=float)
            minutes = minute_of_day(times)
            n = len(vals)
            for i in range(n - 1):
                if not (finite[i] and finite[i + 1]):
                    continue
                m_i, m_j = int(minutes[i]), int(minutes[i + 1])
                if not _same_session_segment(m_i, m_j):
                    continue
                T[code_map[vals[i]], code_map[vals[i + 1]]] += 1.0
            if not include_self:
                np.fill_diagonal(T, 0.0)
            total = float(T.sum())
            if total <= _EPS:
                return np.nan
            p = T / total
            with np.errstate(divide="ignore", invalid="ignore"):
                H = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
            if normalize:
                H = H / np.log(k)
            return H

        return daily_agg(state, _kernel, min_finite=1)


# ---------------------------------------------------------------------------
# 11. intra_neighbor_event_class (per-bar)
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_neighbor_event_class",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_neighbor_event_class",
    source="intraday.state_ops",
    research_only=True,
)
class IntraNeighborEventClass(SeriesOperator):
    """逐 bar 事件分类：孤立 / 簇集（同 session 邻域搜索）。"""

    metadata = _bar_metadata(
        "intra_neighbor_event_class",
        "逐 bar 事件孤立/簇集分类。",
        ["event_mask", "radius", "isolated_code", "clustered_code"],
        unit="code",
    )

    def _calculate_series(self, event_mask, radius=1, isolated_code=1, clustered_code=2, session_tz=None, **_):
        r = _int_param(radius, "radius")
        iso = float(isolated_code)
        clu = float(clustered_code)
        event_mask = session_local(as_panel(event_mask), session_tz)
        out: dict[str, pd.Series] = {}
        for inst in event_mask.columns:
            col = event_mask[inst]
            v = np.asarray(col, dtype=float)
            times = np.asarray(col.index, dtype="datetime64[ns]")
            minutes = minute_of_day(times)
            days = times.astype("datetime64[D]")
            n = len(v)
            res = np.full(n, np.nan)
            for i in range(n):
                if not np.isfinite(v[i]):
                    continue  # stays NaN
                if v[i] <= 0:
                    res[i] = 0.0
                    continue
                found = False
                for d in range(1, r + 1):
                    for j in (i - d, i + d):
                        if 0 <= j < n and np.isfinite(v[j]) and v[j] > 0:
                            if days[j] == days[i] and _same_session_segment(
                                int(minutes[j]), int(minutes[i])
                            ):
                                found = True
                                break
                    if found:
                        break
                res[i] = clu if found else iso
            out[inst] = pd.Series(res, index=col.index, dtype=float)
        return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 12. intra_range_gap_flag (per-bar)
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_range_gap_flag",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_range_gap_flag",
    source="intraday.state_ops",
    research_only=True,
)
class IntraRangeGapFlag(SeriesOperator):
    """逐 bar 事件前后价格区间是否出现严格跳空（不重叠）。"""

    metadata = _bar_metadata(
        "intra_range_gap_flag",
        "事件 bar 前后区间跳空标志。",
        ["high", "low", "event_mask", "neighbor_bars"],
        unit="code",
    )

    def _calculate_series(self, high, low, event_mask, neighbor_bars=1, session_tz=None, **_):
        nb = _int_param(neighbor_bars, "neighbor_bars")
        high = session_local(as_panel(high), session_tz)
        low = session_local(as_panel(low), session_tz)
        event_mask = session_local(as_panel(event_mask), session_tz)
        require_same_session_grid(high, low, event_mask)
        out: dict[str, pd.Series] = {}
        for inst in _common_cols(high, low, event_mask):
            h = np.asarray(high[inst], dtype=float)
            l = np.asarray(low[inst], dtype=float)
            m = np.asarray(event_mask[inst], dtype=float)
            times = np.asarray(high[inst].index, dtype="datetime64[ns]")
            minutes = minute_of_day(times)
            days = times.astype("datetime64[D]")
            n = len(h)
            res = np.full(n, np.nan)
            for i in range(n):
                if not np.isfinite(m[i]):
                    continue  # stays NaN
                if m[i] <= 0:
                    res[i] = 0.0
                    continue
                j_pre = i - nb
                j_post = i + nb

                def _ok(j):
                    return (
                        0 <= j < n
                        and days[j] == days[i]
                        and _same_session_segment(int(minutes[j]), int(minutes[i]))
                        and np.isfinite(h[j])
                        and np.isfinite(l[j])
                    )

                if not (_ok(j_pre) and _ok(j_post)):
                    continue  # stays NaN (missing / cross-session neighbour)
                if h[j_pre] < l[j_post] or h[j_post] < l[j_pre]:
                    res[i] = 1.0  # strict gap (ranges do not overlap)
                else:
                    res[i] = 0.0  # overlap / touch
            out[inst] = pd.Series(res, index=high[inst].index, dtype=float)
        return pd.DataFrame(out)


_CANONICALS.extend(
    [
        "intra_state_count",
        "intra_state_sum",
        "intra_state_vwap",
        "intra_state_interval_moment",
        "intra_state_follow_ratio",
        "intra_state_follow_beta",
        "intra_state_follow_corr",
        "intra_state_pair_same_slot_corr",
        "intra_state_dwell_stats",
        "intra_state_transition_entropy",
        "intra_neighbor_event_class",
        "intra_range_gap_flag",
    ]
)

register_surface(_CANONICALS)
