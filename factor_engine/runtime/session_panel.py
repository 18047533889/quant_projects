# -*- coding: utf-8 -*-
"""Central physical-clock canonicalization for minute→daily operators.

R26-013..039.  Every minute→daily operator canonicalizes its raw minute panel
through :class:`SessionPanel` BEFORE any statistics are computed, so:

* bar frequency comes from a *declared* :class:`runtime.session_calendar.SessionCalendar`
  (its ``bar_freq``) / DataContract / SourceRef temporal grain — NEVER from a
  modal of observed minute deltas (R26-013/014);
* physically absent timestamps become explicit missing slots via reindex onto
  the official grid (R26-017..019);
* log-returns only exist between *adjacent official slots*; a gap is NaN, never
  a single fused bar (R26-020);
* completeness / coverage counts *unique valid official slots*, never observed
  row counts (R26-021);
* a duplicate official slot is a DQ failure (R26-021);
* the trade-date is the *session-local* calendar date, so UTC-stored data is
  split on the correct session day (R26-036..038).

Unknown source timezone fails closed (R26-038): an opaque tz-aware index cannot
be silently assumed session-local.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from runtime.session_calendar import SessionCalendar

_EPS = 1e-12


def _hhmm_minute(value: str) -> int:
    hour, minute = (int(part) for part in str(value).split(":"))
    return hour * 60 + minute


def declared_width_minutes(calendar: Any) -> int:
    """Structural bar width in minutes from the calendar's declared ``bar_freq``.

    R26-013/014: bar resolution is a data-contract property.  It comes from the
    calendar / DataContract / SourceMetadata, NEVER from a modal of observed
    minute deltas — a dataset that systematically drops every other bar (modal
    delta 2) must not be re-certified at 2-min resolution.  An unresolvable
    resolution fails closed (raises) so the caller never emits on a guessed grid.
    """
    bar_freq = getattr(calendar, "bar_freq", None)
    if not bar_freq:
        raise ValueError(
            "calendar must declare `bar_freq` (bar resolution); the resolution "
            "is never inferred from observed minute deltas (R26-014)"
        )
    text = str(bar_freq).strip().lower()
    if text.endswith("min"):
        text = text[:-3]
    elif text.endswith("m"):
        text = text[:-1]
    if not text.isdigit():
        raise ValueError(f"calendar bar_freq must be a minute resolution, got {bar_freq!r}")
    width = int(text)
    if width < 1:
        raise ValueError(f"calendar bar_freq must be >= 1 minute, got {bar_freq!r}")
    return width


def _calm_convention(calendar: Any) -> str:
    convention = str(getattr(calendar, "timestamp_convention", "bar_end")).lower()
    if convention not in {"bar_start", "bar_end"}:
        raise ValueError("calendar timestamp_convention must be bar_start or bar_end")
    return convention


@dataclass(frozen=True)
class MinuteGrid:
    """Official per-session minute grid, derived ONLY from a declared calendar.

    ``expected`` are the grid's wall-clock labels for ``trade_date``; ``slot_id``
    is the official 0-based ordinal of each slot (used for every time-position /
    duration / recovery statistic, never the observed row index).
    """

    market: str
    session: str
    bar_freq: str
    trade_date: pd.Timestamp
    expected_minutes: np.ndarray  # minute-of-day per slot (int)
    expected: np.ndarray          # datetime64[ns] per slot
    slot_id: np.ndarray           # 0..n_slots-1
    segment_id: np.ndarray        # 0..n_segments-1 per slot
    close_mod: int                # minute-of-day of the final session bar label

    @property
    def n_slots(self) -> int:
        return int(self.slot_id.size)

    @property
    def official_minutes(self) -> np.ndarray:
        return self.expected_minutes

    @classmethod
    def from_calendar(cls, calendar: Any, trade_date: pd.Timestamp) -> "MinuteGrid":
        segments = list(getattr(calendar, "segments", None) or ())
        if not segments:
            raise ValueError("calendar must define non-empty session segments")
        width = declared_width_minutes(calendar)
        convention = _calm_convention(calendar)
        minutes: list[int] = []
        segment_ids: list[int] = []
        for number, (start_text, stop_text) in enumerate(segments):
            start = _hhmm_minute(start_text)
            stop = _hhmm_minute(stop_text)
            if stop <= start:
                raise ValueError("calendar segments must be increasing intervals")
            if convention == "bar_start":
                seg_minutes = list(range(start, stop, width))
            else:
                seg_minutes = list(range(start + width, stop + 1, width))
            minutes.extend(seg_minutes)
            segment_ids.extend([number] * len(seg_minutes))
        if not minutes:
            raise ValueError("calendar segments must span at least one bar")
        arr = np.asarray(minutes, dtype=int)
        seg_arr = np.asarray(segment_ids, dtype=int)
        expected = np.asarray(
            [pd.Timestamp(trade_date) + pd.Timedelta(minutes=int(m)) for m in arr],
            dtype="datetime64[ns]",
        )
        last_stop = _hhmm_minute(segments[-1][1])
        close_mod = last_stop - width if convention == "bar_start" else last_stop
        return cls(
            market=str(getattr(calendar, "market", "US")),
            session=str(getattr(calendar, "session", "regular")),
            bar_freq=str(getattr(calendar, "bar_freq", "1min")),
            trade_date=pd.Timestamp(trade_date),
            expected_minutes=arr,
            expected=expected,
            slot_id=np.arange(len(arr), dtype=int),
            segment_id=seg_arr,
            close_mod=close_mod,
        )

    def minute_to_slot(self, minute_of_day: np.ndarray) -> np.ndarray:
        """Map observed minute-of-day values to official slot ordinals (-1 off-grid)."""
        out = np.full(minute_of_day.shape, -1, dtype=int)
        # expected_minutes is strictly increasing -> searchsorted.
        idx = np.searchsorted(self.expected_minutes, minute_of_day)
        idx = np.clip(idx, 0, self.n_slots - 1)
        hit = self.expected_minutes[idx] == minute_of_day
        out[hit] = idx[hit]
        return out


def _to_session_local(
    timestamps: np.ndarray,
    *,
    source_timezone: str | None,
    session_timezone: str,
) -> np.ndarray:
    """Convert observed timestamps to session-local naive wall-clock.

    R26-036..038: the source may store UTC while the session is
    Asia/Shanghai / America/New_York — the TradeDate / HHMM / slot math must run
    on session-local time.  A tz-aware index with an UNKNOWN source timezone
    fails closed (raise) rather than silently assuming session-local.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if ts.tz is None:
        # Naive = already session-local wall-clock.  The caller declares the
        # storage convention; a naive index has no UTC ambiguity.
        return ts.to_numpy(dtype="datetime64[ns]")
    if source_timezone is None:
        raise ValueError(
            "tz-aware minute timestamps require a declared `source_timezone`; "
            "unknown storage timezone fails closed (R26-038)"
        )
    local = ts.tz_convert(session_timezone)
    return local.tz_localize(None).to_numpy(dtype="datetime64[ns]")


def session_trade_dates(timestamps: np.ndarray, session_timezone: str) -> np.ndarray:
    """Session-local calendar date for each timestamp (naive/aware-safe)."""
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if ts.tz is not None:
        ts = ts.tz_convert(session_timezone)
        ts = ts.tz_localize(None)
    return ts.normalize().to_numpy(dtype="datetime64[ns]")


def _minute_of_day(times: np.ndarray) -> np.ndarray:
    seconds = times.astype("datetime64[s]").astype("int64") % 86400
    return seconds // 60


@dataclass(frozen=True)
class SessionPanel:
    """A single (instrument, session-local trade date) reindexed onto the grid.

    Every statistic (log-return, coverage, position, duration, recovery,
    concentration share) is computed from ``values`` aligned to ``grid`` — never
    from the raw observed array.
    """

    market: str
    session: str
    bar_freq: str
    trade_date: pd.Timestamp
    session_timezone: str
    source_timezone: str | None
    grid: MinuteGrid
    values: np.ndarray          # aligned to grid; NaN at absent / duplicate slots
    observed: np.ndarray        # observed timestamp per slot (NaT if absent)
    is_present: np.ndarray      # an observed bar exists at this slot
    is_duplicate: np.ndarray    # this official slot appeared >1x (DQ fail)

    @property
    def n_slots(self) -> int:
        return int(self.grid.n_slots)

    @property
    def is_valid_bar(self) -> np.ndarray:
        """Present AND unique AND on-grid (the only bars that count)."""
        return self.is_present & ~self.is_duplicate

    @property
    def slot_id(self) -> np.ndarray:
        return self.grid.slot_id

    def coverage(self, valid_mask: np.ndarray | None = None) -> float:
        """Fraction of unique valid official slots (R26-021).

        Denominator = official slot count (never observed rows).  A duplicate
        slot and an absent slot both fail; lunch-gap bars are off-grid and never
        counted.
        """
        if self.n_slots == 0:
            return 0.0
        if valid_mask is None:
            valid = self.is_valid_bar
        else:
            valid = self.is_valid_bar & np.asarray(valid_mask, dtype=bool)
        return float(valid.sum()) / float(self.n_slots)

    def log_returns(self) -> np.ndarray:
        """Grid-gated log returns (R26-017..020).

        Only slot-adjacent pairs with BOTH prices finite produce a return.  A
        physically absent bar is an explicit missing slot, so ``log(P_t/P_{t-1})``
        across a gap is NaN — never a fused single-bar return.  Non-positive
        prices are invalid and yield NaN (not a fabricated log-return).
        """
        out = np.full(self.n_slots, np.nan)
        if self.n_slots < 2:
            return out
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = self.values[1:] / self.values[:-1]
        ok = np.isfinite(self.values[1:]) & np.isfinite(self.values[:-1])
        ok &= (self.values[1:] > 0.0) & (self.values[:-1] > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.where(ok, np.log(ratio), np.nan)
        return out

    def value_at_minute(self, minute_of_day: int) -> float:
        """Value at an exact official minute (R26-025..027 EndpointPolicy.EXACT).

        Returns NaN unless the requested official slot exists AND carries a
        valid bar.  Never substitutes the nearest finite bar.
        """
        slot = int(self.grid.minute_to_slot(np.asarray([minute_of_day], dtype=int))[0])
        if slot < 0 or not self.is_valid_bar[slot]:
            return np.nan
        return float(self.values[slot])

    def contiguous_complete_run(self) -> np.ndarray:
        """Boolean mask of the longest trailing contiguous complete run.

        R26-028/029: path-style statistics break at the first interior missing
        slot; only an explicit trailing contiguous complete run is used.
        """
        valid = self.is_valid_bar & np.isfinite(self.values)
        if not np.any(valid):
            return np.zeros(self.n_slots, dtype=bool)
        last = int(np.nonzero(valid)[0][-1])
        run = np.zeros(self.n_slots, dtype=bool)
        run[last] = True
        i = last - 1
        while i >= 0 and valid[i]:
            run[i] = True
            i -= 1
        return run


@dataclass(frozen=True)
class _BuiltPanel:
    panel: SessionPanel
    trade_date: pd.Timestamp


def build_session_panel(
    timestamps: np.ndarray,
    values: np.ndarray,
    calendar: Any,
    *,
    market: str,
    session_timezone: str,
    source_timezone: str | None = None,
    trade_date: pd.Timestamp | None = None,
) -> SessionPanel:
    """Reindex one day's observed (timestamps, values) onto the official grid.

    ``timestamps`` may be tz-aware (stored UTC) — they are converted to
    ``session_timezone`` wall-clock first.  A duplicate official slot marks the
    slot DQ-failed; an off-grid bar (lunch recess) is dropped.
    """
    local = _to_session_local(timestamps, source_timezone=source_timezone, session_timezone=session_timezone)
    if trade_date is None:
        dates = session_trade_dates(local, session_timezone)
        if dates.size == 0:
            raise ValueError("cannot build a session panel from empty timestamps")
        trade_date = pd.Timestamp(np.datetime64(dates[0], "D"))
    grid = MinuteGrid.from_calendar(calendar, trade_date)
    mods = _minute_of_day(local)
    slots = grid.minute_to_slot(mods)
    on_grid = slots >= 0
    values = np.asarray(values, dtype=float)
    aligned = np.full(grid.n_slots, np.nan)
    observed = np.full(grid.n_slots, np.datetime64("NaT"), dtype="datetime64[ns]")
    present = np.zeros(grid.n_slots, dtype=bool)
    duplicate = np.zeros(grid.n_slots, dtype=bool)
    seen: set[int] = set()
    for s, v, t in zip(slots[on_grid], values[on_grid], local[on_grid]):
        s = int(s)
        if s in seen:
            duplicate[s] = True
            present[s] = True  # still present, but DQ-failed by duplication
            continue
        seen.add(s)
        present[s] = True
        aligned[s] = v
        observed[s] = t
    return SessionPanel(
        market=market,
        session=str(getattr(calendar, "session", "regular")),
        bar_freq=str(getattr(calendar, "bar_freq", "1min")),
        trade_date=pd.Timestamp(trade_date),
        session_timezone=session_timezone,
        source_timezone=source_timezone,
        grid=grid,
        values=aligned,
        observed=observed,
        is_present=present,
        is_duplicate=duplicate,
    )


def default_ashare_calendar(*, bar_freq: str = "1min") -> SessionCalendar:
    return SessionCalendar.ashare(bar_freq=bar_freq, timestamp_convention="bar_end")


__all__ = [
    "MinuteGrid",
    "SessionPanel",
    "build_session_panel",
    "declared_width_minutes",
    "default_ashare_calendar",
    "session_trade_dates",
    "to_session_local",
]


def to_session_local(
    timestamps: np.ndarray,
    *,
    source_timezone: str | None,
    session_timezone: str,
) -> np.ndarray:
    return _to_session_local(timestamps, source_timezone=source_timezone, session_timezone=session_timezone)
