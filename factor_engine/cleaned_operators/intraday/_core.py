# -*- coding: utf-8 -*-
"""Shared kernels for next-stage intraday operators.

Self-contained helpers (adapted from the reviewed ``microstructure`` contract):
every operator accepts minute-frequency panels (row index is a minute
timestamp, columns are instruments) and returns a daily-frequency panel
(row index is the calendar date, columns are instruments).

Contract
--------
* One scalar per (TradeDate, Symbol).  Never a row per minute.
* All kernels are causal: only the day's own minute data plus that day's daily
  limit / weight panels, never future bars or future days.
* Empty / all-NaN windows yield NaN, never Inf or a fabricated zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator

_EPS = 1e-12
_SESSION_TZ = "Asia/Shanghai"
_MORNING = (570, 690)   # 09:30 .. 11:30 minute-of-day
_AFTERNOON = (780, 900)  # 13:00 .. 15:00 minute-of-day
_SEGMENT_RANGES = {"morning": _MORNING, "afternoon": _AFTERNOON}


class DataDegeneracy(ValueError):
    """A kernel hit genuinely degenerate *data* (not a programming error).

    P1-18: the daily-aggregation wrappers map only this exception (plus
    ``ZeroDivisionError`` / ``OverflowError``, which are by definition data
    degeneracies) to ``NaN``.  A plain ``ValueError`` from a bad parameter or a
    kernel bug must propagate so it is never silently swallowed as a missing
    estimate.
    """


class SessionCoveragePolicy(str, Enum):
    """Coverage policy for accepting a session's partial-bar mask."""

    STRICT_FULL_SESSION = "strict_full_session"
    MIN_COVERAGE_095 = "min_coverage_095"
    PAIRWISE_ALLOWED = "pairwise_allowed"
    EVENT_SPARSE = "event_sparse"


@dataclass
class SessionGrid:
    """Session structure of a minute panel (P0-08).

    Describes the per-trade-date minute-slot grid so multi-input minute panels
    can be checked for alignment *before* a ``pd.concat(...).dropna(...)`` step
    silently compresses a mismatched axis into a shorter (fabricated) grid.

    Fields
    ------
    market:          market identifier (``A-share`` when the index is tz-aware).
    trade_date:      sorted tuple of unique trade dates (normalized).
    expected_slots:  number of minute slots in a full session (modal day width).
    session_slot:    per-row minute-of-day (int).
    present_mask:    per-row bool; True when the row lies on a full-width session
                     day (all expected slots observed).
    """

    market: str
    trade_date: tuple
    expected_slots: int
    session_slot: np.ndarray
    present_mask: np.ndarray
    day_slots: tuple = ()  # ((day, (minute-of-day slots...)), ...)

    def slot_structure(self) -> tuple:
        """Canonical (day -> minute-slot tuple) key for grid-equality checks.

        Two grids are equal only when every trade date maps to the SAME set of
        minute-of-day slots — comparing just the date list or the slot *count*
        would miss a panel whose bars sit at different minutes of the session.
        """
        return self.day_slots


def build_session_grid(index: Any, market: str | None = None) -> SessionGrid:
    """Build the session grid of a minute ``DatetimeIndex`` (P0-08).

    ``present_mask`` marks bars on days whose observed slot count equals the
    modal full-session width; partial days are still represented in
    ``trade_date`` but are not ``present``.  The grid is used by
    ``require_same_session_grid`` to fail closed on mismatched multi-input
    panels.
    """
    idx = pd.DatetimeIndex(index)
    if market is None:
        market = "A-share" if getattr(idx, "tz", None) is not None else "unknown"
    minutes = minute_of_day(idx.to_numpy(dtype="datetime64[ns]"))
    days = idx.normalize()
    day_slots: dict[pd.Timestamp, set[int]] = {}
    for day, m in zip(days, minutes):
        day_slots.setdefault(pd.Timestamp(day), set()).add(int(m))
    sorted_days = tuple(sorted(day_slots.keys()))
    slots_per_day = tuple((d, tuple(sorted(day_slots[d]))) for d in sorted_days)
    expected_slots = int(max((len(s) for _, s in slots_per_day), default=0))
    full_days = {d for d, s in slots_per_day if len(s) == expected_slots}
    present_mask = np.asarray([pd.Timestamp(day) in full_days for day in days], dtype=bool)
    return SessionGrid(
        market=market,
        trade_date=sorted_days,
        expected_slots=expected_slots,
        session_slot=minutes.astype(np.int64),
        present_mask=present_mask,
        day_slots=slots_per_day,
    )


def require_same_session_grid(*frames: Any) -> None:
    """Fail closed when minute panels do NOT share the same session grid (P0-08).

    A grid mismatch — different trade dates or different per-day slot
    resolution — would otherwise be silently compressed by the
    ``pd.concat(...).dropna(subset=[...])`` sites in the daily-aggregation
    kernels, producing a day estimate on a partial axis.  Raise instead.
    """
    if len(frames) < 2:
        return
    grids = [build_session_grid(f.index) for f in frames]
    first = grids[0]
    for g in grids[1:]:
        if g.slot_structure() != first.slot_structure():
            raise ValueError(
                "intraday panels do not share the same session grid: "
                f"{first.market} trade_dates={len(first.trade_date)} "
                f"expected_slots={first.expected_slots} vs "
                f"{g.market} trade_dates={len(g.trade_date)} "
                f"expected_slots={g.expected_slots} — refusing to splice a "
                "partial axis into a day estimate"
            )


def session_coverage_ok(present_mask: np.ndarray | None, policy: SessionCoveragePolicy) -> bool:
    """Return whether a session's present-bar mask satisfies ``policy`` (P0-09)."""
    if present_mask is None or len(present_mask) == 0:
        return False
    mask = np.asarray(present_mask, dtype=bool)
    n_present = int(np.sum(mask))
    n_total = int(len(mask))
    coverage = n_present / n_total if n_total else 0.0
    if policy is SessionCoveragePolicy.STRICT_FULL_SESSION:
        return n_present == n_total
    if policy is SessionCoveragePolicy.MIN_COVERAGE_095:
        return coverage >= 0.95
    if policy is SessionCoveragePolicy.PAIRWISE_ALLOWED:
        return n_present >= 2
    if policy is SessionCoveragePolicy.EVENT_SPARSE:
        return n_present >= 1
    return False


class SessionAggregationOperator(SeriesOperator):
    """Base class for intraday minute -> daily aggregation operators (P0-07).

    Every operator here genuinely changes time frequency: it consumes a
    minute-frequency panel and emits one scalar per (TradeDate, Symbol).  The
    grain contract is declared on the class and mirrored into
    ``OperatorMetadata.input_grain`` / ``output_grain`` by ``_core.metadata()``.
    ``calculate`` runs the shared framework validation gate then delegates to
    ``_calculate_series`` exactly like ``SeriesOperator`` — no kernel math is
    changed, only the frequency contract is made explicit.
    """

    input_grain: str = "minute"
    output_grain: str = "daily"

    # R5-02: ``calculate`` below routes through ``_prepare_call`` ->
    # ``validate_operator_call`` (identical to the framework ``SeriesOperator``
    # pattern), so this custom method must be declared contract-handling.
    _HANDLES_CALL_CONTRACT = True

    def calculate(self, *args, **kwargs) -> pd.DataFrame:
        processed_args, processed_kwargs = self._prepare_call(args, kwargs)
        return self._calculate_series(*processed_args, **processed_kwargs)


def metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    domain: str = "intraday",
    cost: int = 6,
    extra_tags: list[str] | None = None,
) -> OperatorMetadata:
    """Standard metadata for intraday -> daily aggregation operators."""
    tags = [
        "intraday", "daily_agg", "minute", "pit_safe", "causal",
        "typed_v2", "source_blocked", "grain_minute_to_daily",
        f"signature:{','.join(params)}->series", f"unit:{unit}",
        f"cost:{cost}", f"domain:{domain}",
    ]
    if extra_tags:
        tags.extend(extra_tags)
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=tags,
        input_grain="minute",
        output_grain="daily",
    )


def as_panel(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(getattr(x, "name", None) or "value")
    return x


def minute_of_day(times: np.ndarray) -> np.ndarray:
    """Timestamp array -> minute-of-day (integer)."""
    seconds = times.astype("datetime64[s]").astype("int64") % 86400
    return seconds // 60


def log_returns(vals: np.ndarray) -> np.ndarray:
    out = np.full(len(vals), np.nan)
    if len(vals) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.log(vals[1:] / vals[:-1])
    return out


def session_local(frame: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    """Convert a tz-aware index to session wall-clock (naive) for minute math.

    A-share COS minute data is stored in UTC; session segments are defined in
    Asia/Shanghai wall-clock.  Naive indexes are assumed to already be local.
    """
    frame = as_panel(frame)
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        tz = tz or _SESSION_TZ
        frame = frame.tz_convert(tz)
        frame.index = frame.index.tz_localize(None)
    return frame


def seg_mask(times: np.ndarray, segment: str) -> np.ndarray:
    lo, hi = _SEGMENT_RANGES[str(segment)]
    minutes = minute_of_day(times)
    return (minutes >= lo) & (minutes <= hi)


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.isfinite(den) & (np.abs(den) > _EPS), num / den, np.nan)
    return out


def daily_agg(
    frame: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    min_finite: int = 2,
) -> pd.DataFrame:
    """Apply per-(instrument, calendar-day) aggregation fn(vals, times).

    ``min_finite`` (P0-09) is the smallest number of finite minute values a day
    must have before a day estimate is produced; a single finite value can no
    longer manufacture one.  Only ``DataDegeneracy`` (P1-18) — plus genuine
    ``ZeroDivisionError`` / ``OverflowError`` — maps to ``NaN``; a plain
    ``ValueError`` from a parameter/kernel bug propagates.
    """
    frame = as_panel(frame)
    out: dict[str, pd.Series] = {}
    for inst in frame.columns:
        col = frame[inst]
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in col.groupby(col.index.normalize()):
            vals = np.asarray(group, dtype=float)
            times = np.asarray(group.index, dtype="datetime64[ns]")
            if int(np.sum(np.isfinite(vals))) < int(min_finite):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(vals, times))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def daily_agg_two(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    min_finite: int = 2,
) -> pd.DataFrame:
    """Apply fn(a_vals, b_vals) per (instrument, day).

    The two frames must share the same session grid (P0-08) — otherwise the
    ``pd.concat(...).dropna(subset=["a"])`` below would silently compress a
    mismatched axis.  The ``dropna`` is kept; it only removes rows where the
    PRIMARY column (``a``) is NaN, and slot compression is now guarded.
    """
    frame_a, frame_b = as_panel(frame_a), as_panel(frame_b)
    require_same_session_grid(frame_a, frame_b)
    out: dict[str, pd.Series] = {}
    for inst in frame_a.columns:
        a, b = frame_a[inst], frame_b[inst]
        joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna(subset=["a"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals_a = np.asarray(group["a"], dtype=float)
            vals_b = np.asarray(group["b"], dtype=float)
            if int(np.sum(np.isfinite(vals_a))) < int(min_finite):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(vals_a, vals_b))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def daily_agg_three(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    frame_c: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    *,
    min_finite: int = 2,
) -> pd.DataFrame:
    """Apply fn(a_vals, b_vals, c_vals) per (instrument, day).

    Same session-grid guard (P0-08) and ``DataDegeneracy`` catch (P1-18) as
    ``daily_agg_two``.
    """
    frame_a, frame_b, frame_c = as_panel(frame_a), as_panel(frame_b), as_panel(frame_c)
    require_same_session_grid(frame_a, frame_b, frame_c)
    out: dict[str, pd.Series] = {}
    for inst in frame_a.columns:
        joined = pd.concat(
            [frame_a[inst], frame_b[inst], frame_c[inst]], axis=1, keys=["a", "b", "c"]
        ).dropna(subset=["a"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals_a = np.asarray(group["a"], dtype=float)
            vals_b = np.asarray(group["b"], dtype=float)
            vals_c = np.asarray(group["c"], dtype=float)
            if int(np.sum(np.isfinite(vals_a))) < int(min_finite):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(vals_a, vals_b, vals_c))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def broadcast_daily_panel(close_frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Broadcast a daily panel (index=date) onto a minute panel by date."""
    daily = as_panel(daily)
    out: dict[str, pd.Series] = {}
    for inst in close_frame.columns:
        if inst not in daily.columns:
            continue
        close_col = close_frame[inst]
        daily_col = daily[inst]
        days = close_col.index.normalize()
        mapped = daily_col.reindex(pd.DatetimeIndex(days.unique()))
        out[inst] = pd.Series(
            mapped.reindex(pd.DatetimeIndex(days)).to_numpy(),
            index=close_col.index,
        )
    return pd.DataFrame(out)


def register_surface(canonicals: list[str]) -> None:
    """Append new canonicals to the reviewed extended surface.

    Union is order-independent so concurrent modules may each add their own
    names without clobbering one another.
    """
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(canonicals))


def register_research_surface(canonicals: list[str]) -> None:
    """Append new canonicals to the research surface (experimental operators)."""
    import cleaned_operators.operator_surface as _surface

    _surface.extend_research_only(set(canonicals))


# Silent-numpy context for statistic kernels that may overflow intermediate
# products on extreme minute moves.
_np_err = {"divide": "ignore", "invalid": "ignore", "over": "ignore"}


def np_errstate():
    return np.errstate(**_np_err)


def mu_p(p: float) -> float:
    """E|Z|^p for a standard normal Z."""
    return float(2.0 ** (p / 2.0) * math.gamma((p + 1.0) / 2.0) / math.gamma(0.5))


def tripower_scale() -> float:
    """mu_{4/3}^{-3} constant for tripower quarticity."""
    return float(mu_p(4.0 / 3.0) ** -3)
