# -*- coding: utf-8 -*-
"""Marked-event dynamics (2026-08 market language, P1).

Events carry magnitude (the *mark*): a large-return shock's size, an abnormal
turnover's z-score, a limit event's strength.  These operators study how event
magnitudes and event timing interact:

* ``event_mark_autocorr``       — autocorrelation of event marks in *event
  index* space: do big events follow big events?
* ``event_interval_mark_coupling`` — correlation between the inter-event
  interval and the following event's mark: does a long quiet spell build up a
  bigger event, or do bursts come in clusters?

Both are prefix-causal trailing-window kernels, deterministic, fail-closed to
NaN when too few events are present in the window.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_udf

_EPS = 1e-12


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="marked_event",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "marked_event", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _validate_event_bool(ev: np.ndarray, operator: str) -> None:
    """Event inputs must be strict boolean indicators {0, 1} or NaN (review P1-47a).

    A probability / z-score / any other non-0/1 finite value is a caller bug:
    silently thresholding at 0.5 would reinterpret a non-boolean input as an
    event.  Raise loudly instead.
    """
    finite = ev[np.isfinite(ev)]
    bad = finite[(finite != 0.0) & (finite != 1.0)]
    if bad.size:
        raise ValueError(
            f"{operator} requires a strict boolean event indicator "
            f"(values 0/1 or NaN); found non-boolean finite value {float(bad[0])!r}"
        )


def _event_states(evc: np.ndarray, mkc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify event bars (``evc == 1``) by mark availability (review #43).

    Returns ``(idx, marks, has_mark)``.  ``has_mark`` is True for an event with
    a finite mark ("event occurred, mark available") and False for an event
    whose mark is NaN ("event occurred, mark unavailable").  This is
    deliberately distinct from "event did not occur" (``evc == 0``) and "event
    observation unknown" (``evc == NaN``), which are never returned as events.
    """
    idx = np.flatnonzero(evc == 1.0)
    marks = np.asarray([mkc[i] for i in idx], dtype=float)
    has_mark = np.isfinite(marks)
    return idx, marks, has_mark


def _mark_autocorr_1d(marks: np.ndarray, event_lag: int) -> float:
    """Corr(m_j, m_{j-lag}) over a contiguous, fully-available mark array."""
    n = marks.shape[0]
    if n <= event_lag + 3:
        return np.nan
    a = marks[event_lag:]
    b = marks[: n - event_lag]
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) <= event_lag + 2:
        return np.nan
    aa = a[ok]
    bb = b[ok]
    va = float(np.var(aa))
    vb = float(np.var(bb))
    if va <= _EPS or vb <= _EPS:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _mark_autocorr_chunk(evc: np.ndarray, mkc: np.ndarray, event_lag: int, mark_missing_policy: str = "censor") -> float:
    _, marks, has_mark = _event_states(evc, mkc)
    n = marks.shape[0]
    if n <= event_lag + 3:
        return np.nan
    if mark_missing_policy == "censor":
        # #43 censor: an event that occurred but whose mark is unavailable is a
        # *break* in the event-index sequence — magnitude memory across the gap
        # is unknown, so autocorrelation is computed within the longest
        # contiguous segment of available marks and events on either side of
        # the missing-mark event are never paired.
        best = np.nan
        best_size = -1
        seg_start = 0
        for j in range(n + 1):
            if j < n and has_mark[j]:
                continue
            seg = marks[seg_start:j]
            if seg.size >= 2:
                c = _mark_autocorr_1d(seg, event_lag)
                if np.isfinite(c) and seg.size > best_size:
                    best = c
                    best_size = seg.size
            seg_start = j + 1
        return best
    # drop: drop unavailable-mark events and re-index the survivors.
    return _mark_autocorr_1d(marks[has_mark], event_lag)


@register_operator(
    name="event_mark_autocorr",
    category="marked_event",
    business_category="marked_event",
    canonical="event_mark_autocorr",
    source="marked_event",
)
class EventMarkAutocorr(SeriesOperator):
    """事件 mark 的（事件序）自相关：大事件之后是否仍出大事件。

    窗口内抽出事件发生日的 mark 序列 ``m_1..m_K``，输出 ``Corr(m_j, m_{j-lag})``
    （lag 为事件序号间隔）。与 event-spacing 不同——这里研究的是事件强度记忆。
    P1。
    """

    metadata = _metadata(
        "event_mark_autocorr",
        "事件 mark 序列自相关（事件强度记忆）。",
        ["event", "mark", "history_window", "event_lag", "mark_missing_policy"],
        unit="corr",
        cost=4,
    )

    def _calculate_series(
        self,
        event: pd.DataFrame,
        mark: pd.DataFrame,
        history_window: int = 252,
        event_lag: int = 1,
        mark_missing_policy: str = "censor",
        **_: Any,
    ) -> pd.DataFrame:
        w = int(history_window)
        el = int(event_lag)
        if el < 1:
            raise ValueError("event_mark_autocorr requires event_lag >= 1")
        if w < el + 5:
            raise ValueError("event_mark_autocorr requires history_window >= event_lag + 5")
        if mark_missing_policy not in ("censor", "drop"):
            raise ValueError("event_mark_autocorr mark_missing_policy must be 'censor' or 'drop'")

        evv = event.to_numpy(dtype=float)
        _validate_event_bool(evv, "event_mark_autocorr")
        mkv = mark.to_numpy(dtype=float)
        rows, cols = evv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            for r in range(rows):
                lo = max(0, r - w + 1)
                out[r, c] = _mark_autocorr_chunk(
                    evv[lo : r + 1, c], mkv[lo : r + 1, c], el, mark_missing_policy
                )
        return frame_like(event, out)


def _corr_longest_run(xv: np.ndarray, yv: np.ndarray, ok: np.ndarray, min_pairs: int = 4) -> float:
    """Corr(x, y) within the longest contiguous run of valid observations.

    Used by the ``censor`` MarkMissingPolicy: an invalid observation (an
    interval spanning an unknown event observation, or an event whose mark is
    unavailable) *breaks* the sequence, so observations on either side of the
    gap belong to different epochs and are never pooled.  Fail-closed to NaN
    when no run is large enough.
    """
    best = np.nan
    best_size = -1
    j = 0
    n = ok.shape[0]
    while j < n:
        if not ok[j]:
            j += 1
            continue
        k = j
        while k < n and ok[k]:
            k += 1
        if k - j >= min_pairs:
            xs = xv[j:k]
            ys = yv[j:k]
            vx = float(np.var(xs))
            vy = float(np.var(ys))
            if vx > _EPS and vy > _EPS:
                c = float(np.corrcoef(xs, ys)[0, 1])
                if np.isfinite(c) and (k - j) > best_size:
                    best = c
                    best_size = k - j
        j = k
    return best


def _interval_mark_chunk(evc: np.ndarray, in_idx: np.ndarray, prev_idx: int | None, mkv: np.ndarray, mark_missing_policy: str = "censor") -> float:
    """Corr(interval -> following event mark) over a window's events.

    ``in_idx`` are the absolute event rows inside the window; ``prev_idx`` is the
    last event row strictly before the window (or None).  The pre-window ->
    first-in-window interval is *included* (paired with the first in-window
    event's mark); dropping it biases the estimate against long intervals that
    merely straddle the window boundary (review P1-47b).

    Review #42: an interval that spans an *unknown* event observation (a NaN in
    the event indicator strictly between the two bounding events) is censored —
    the interval length is unknown, so it must not connect the two events.
    Review #43: an event that occurred but whose mark is NaN is "event occurred,
    mark unavailable"; the ``mark_missing_policy`` decides whether it breaks the
    interval sequence (``censor``) or is simply dropped (``drop``).
    """
    if in_idx.size < 4:
        return np.nan
    marks = mkv[in_idx]
    if prev_idx is not None:
        starts = np.concatenate([[prev_idx], in_idx[:-1]])
        ends = in_idx
        marks_after = marks            # first interval pairs with marks[0]
    else:
        starts = in_idx[:-1]
        ends = in_idx[1:]
        marks_after = marks[1:]
    intervals = (ends - starts).astype(float)
    ok = np.isfinite(intervals) & np.isfinite(marks_after)
    # #42: censor any interval spanning an unknown event observation.
    for j in range(len(intervals)):
        a = int(starts[j])
        b = int(ends[j])
        if b - a > 1 and np.isnan(evc[a + 1 : b]).any():
            ok[j] = False
    if int(ok.sum()) < 4:
        return np.nan
    if mark_missing_policy == "censor":
        return _corr_longest_run(intervals, marks_after, ok, min_pairs=4)
    ti = intervals[ok]
    mi = marks_after[ok]
    vt = float(np.var(ti))
    vm = float(np.var(mi))
    if vt <= _EPS or vm <= _EPS:
        return np.nan
    return float(np.corrcoef(ti, mi)[0, 1])


@register_operator(
    name="event_interval_mark_coupling",
    category="marked_event",
    business_category="marked_event",
    canonical="event_interval_mark_coupling",
    source="marked_event",
)
class EventIntervalMarkCoupling(SeriesOperator):
    """事件间隔与事件强度耦合 Corr(τ_j, m_j)。

    τ_j = 相邻事件日期间隔，m_j = 事件 mark。正相关 = 长时间平静后憋出大事件；
    负相关 = 事件越密集越强。适合 limit event / abnormal return / turnover spike
    / financial surprise。P1。
    """

    metadata = _metadata(
        "event_interval_mark_coupling",
        "事件间隔与事件强度相关 Corr(τ_j, m_j)。",
        ["event", "mark", "window", "mark_missing_policy"],
        unit="corr",
        cost=4,
    )

    def _calculate_series(
        self,
        event: pd.DataFrame,
        mark: pd.DataFrame,
        window: int = 252,
        mark_missing_policy: str = "censor",
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        if w < 6:
            raise ValueError("event_interval_mark_coupling requires window >= 6")
        if mark_missing_policy not in ("censor", "drop"):
            raise ValueError("event_interval_mark_coupling mark_missing_policy must be 'censor' or 'drop'")

        evv = event.to_numpy(dtype=float)
        _validate_event_bool(evv, "event_interval_mark_coupling")
        mkv = mark.to_numpy(dtype=float)
        rows, cols = evv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            all_idx = np.flatnonzero(evv[:, c] == 1.0)
            for r in range(rows):
                lo = max(0, r - w + 1)
                start_pos = int(np.searchsorted(all_idx, lo, side="left"))
                end_pos = int(np.searchsorted(all_idx, r + 1, side="left"))
                in_idx = all_idx[start_pos:end_pos]
                prev_idx = int(all_idx[start_pos - 1]) if start_pos > 0 else None
                out[r, c] = _interval_mark_chunk(evv[:, c], in_idx, prev_idx, mkv[:, c], mark_missing_policy)
        return frame_like(event, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"event_mark_autocorr", "event_interval_mark_coupling"})
    for _canon in ("event_mark_autocorr", "event_interval_mark_coupling"):
        register_polars_udf(_canon)


_register_surface()
