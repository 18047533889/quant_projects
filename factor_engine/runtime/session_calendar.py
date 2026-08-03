# -*- coding: utf-8
"""Explicit exchange-session calendar helpers for native intraday runtimes.

The calendar owns timestamp labelling, lunch breaks and session resets.  It is
small on purpose: holiday selection remains the data source's responsibility,
while bars can never leak across a trading session or a market recess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from cleaned_operators.operator_policy import bar_freq_to_timedelta, bars_per_day


def _minute(value: str) -> int:
    hour, minute = (int(part) for part in str(value).split(":"))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid session time {value!r}")
    return hour * 60 + minute


@dataclass(frozen=True)
class SessionCalendar:
    """Trading-session definition used by minute-native helpers and runtimes.

    ``segments`` contains half-open wall-clock intervals.  A-share regular
    trading therefore has two segments and the lunch recess cannot become part
    of a multi-minute bar.  ``timestamp_convention`` describes the raw one-
    minute labels: ``bar_end`` uses ``(open, close]`` and ``bar_start`` uses
    ``[open, close)``.
    """

    market: str = "US"
    session: str = "regular"
    bar_freq: str = "1min"
    bars_per_session: int | None = None
    segments: tuple[tuple[str, str], ...] | None = None
    timestamp_convention: str = "bar_end"

    def __post_init__(self) -> None:
        market = str(self.market or "US").upper()
        convention = str(self.timestamp_convention).lower()
        if convention not in {"bar_end", "bar_start"}:
            raise ValueError("timestamp_convention must be bar_end or bar_start")
        segments = self.segments
        if segments is None:
            segments = (
                (("09:30", "11:30"), ("13:00", "15:00"))
                if market in {"CN", "ASHARE", "A_SHARE"}
                else (("09:30", "16:00"),)
            )
        parsed = tuple((str(start), str(stop)) for start, stop in segments)
        if not parsed or any(_minute(stop) <= _minute(start) for start, stop in parsed):
            raise ValueError("session segments must be non-empty increasing intervals")
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "session", str(self.session or "regular").lower())
        object.__setattr__(self, "bar_freq", str(self.bar_freq))
        object.__setattr__(self, "segments", parsed)
        object.__setattr__(self, "timestamp_convention", convention)
        if self.session_bars <= 0:
            raise ValueError("SessionCalendar requires a positive bar count")

    @classmethod
    def ashare(
        cls,
        *,
        bar_freq: str = "1min",
        timestamp_convention: str = "bar_end",
    ) -> "SessionCalendar":
        return cls(
            market="CN",
            bar_freq=bar_freq,
            timestamp_convention=timestamp_convention,
        )

    @property
    def bar_timedelta(self) -> pd.Timedelta:
        return bar_freq_to_timedelta(self.bar_freq)

    @property
    def session_minutes(self) -> int:
        return sum(_minute(stop) - _minute(start) for start, stop in self.segments or ())

    @property
    def session_bars(self) -> int:
        if self.bars_per_session is not None:
            return int(self.bars_per_session)
        try:
            return bars_per_day(self.bar_freq, market=self.market, session=self.session)
        except (KeyError, ValueError):
            width = max(1, int(self.bar_timedelta / pd.Timedelta(minutes=1)))
            return int(np.ceil(self.session_minutes / width))

    def session_key(self, timestamps: Iterable[object]) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(timestamps)).normalize()

    def minute_ordinal(self, timestamps: Iterable[object]) -> np.ndarray:
        """Return continuous in-session minute ordinals; recess/outside is -1."""
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        minute = ts.hour * 60 + ts.minute
        output = np.full(len(ts), -1, dtype=int)
        offset = 0
        for start_text, stop_text in self.segments or ():
            start, stop = _minute(start_text), _minute(stop_text)
            if self.timestamp_convention == "bar_end":
                valid = (minute > start) & (minute <= stop)
                local = minute - start - 1
            else:
                valid = (minute >= start) & (minute < stop)
                local = minute - start
            output[valid] = offset + local[valid]
            offset += stop - start
        return output

    def segment_ordinal(self, timestamps: Iterable[object]) -> np.ndarray:
        """Return minute ordinal within each segment; recess/outside is -1."""
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        minute = ts.hour * 60 + ts.minute
        output = np.full(len(ts), -1, dtype=int)
        for start_text, stop_text in self.segments or ():
            start, stop = _minute(start_text), _minute(stop_text)
            if self.timestamp_convention == "bar_end":
                valid = (minute > start) & (minute <= stop)
                local = minute - start - 1
            else:
                valid = (minute >= start) & (minute < stop)
                local = minute - start
            output[valid] = local[valid]
        return output

    def segment_id(self, timestamps: Iterable[object]) -> np.ndarray:
        """Return the segment number, with -1 for recess/outside timestamps."""
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        minute = ts.hour * 60 + ts.minute
        output = np.full(len(ts), -1, dtype=int)
        for number, (start_text, stop_text) in enumerate(self.segments or ()):
            start, stop = _minute(start_text), _minute(stop_text)
            if self.timestamp_convention == "bar_end":
                valid = (minute > start) & (minute <= stop)
            else:
                valid = (minute >= start) & (minute < stop)
            output[valid] = number
        return output

    def offset_bars(self, anchor: str | pd.Timestamp, n_bars: int) -> pd.Timestamp:
        return pd.Timestamp(anchor) + self.bar_timedelta * int(n_bars)

    def warmup_load_start(
        self,
        requested_start: str | pd.Timestamp,
        *,
        lookback_bars: int,
    ) -> pd.Timestamp:
        lb = max(0, int(lookback_bars))
        if lb <= 0:
            return pd.Timestamp(requested_start)
        anchor = pd.Timestamp(requested_start)
        if anchor == anchor.normalize():
            anchor += self.bar_timedelta * (self.session_bars - 1)
        return self.offset_bars(anchor, -lb)


# Backward-compatible public name used by existing callers.
SessionBarCalendar = SessionCalendar
