"""SessionSpec — abstract away A/US trading sessions.

Per the multi-market plan §35-§38: operators must never hardcode
``if minute >= 570``.  Intraday logic works off ``SessionSlotId`` /
``normalized_session_position`` derived from a :class:`SessionSpec`.

A-share (COS_ashare_lqtp_data_dictionary 2026-08-08):
    09:31-11:30 + 13:01-15:00 CST (UTC+8), ~240 bars/day, no 09:30/13:00 bars,
    no lunch bars.  QuoteTime is stored UTC.

US (COS_us_massive_data_dictionary 2026-08-08):
    Regular 09:30-16:00 Eastern, DST-aware via America/New_York; early-close
    days (is_early_close ~51 True days) must be down-weighted for volume/
    volatility factors.  No hardcoded UTC-5 offset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

TIMEZONE_ASHARE = "Asia/Shanghai"
TIMEZONE_US = "America/New_York"


@dataclass(frozen=True)
class SessionSegment:
    """One contiguous intraday trading segment (local wall-clock time)."""

    start: time
    end: time
    slot_offset: int = 0  # bar index of this segment's first minute


@dataclass(frozen=True)
class SessionSpec:
    """A market's intraday session contract."""

    session_id: str
    timezone: str
    segments: tuple[SessionSegment, ...]
    slot_count: int
    minute_bars: bool = True
    early_close_policy: str = "none"  # none | down_weight | exclude
    notes: str = ""
    bar_convention: str = "bar_start"  # bar_start -> [start, end), 390 bars
    # Known early-close dates (e.g. US half-days).  Populated by a Calendar
    # provider at runtime; empty here means "no early close known".
    early_close_dates: frozenset[str] = field(default_factory=frozenset)

    def slot_for_local(self, ts: datetime) -> int | None:
        """Map a local wall-clock datetime to a SessionSlotId (1-based bar).

        The session is ``[start, end)`` under ``bar_start`` (a bar at ``end``
        does not exist), so a computed slot beyond ``slot_count`` returns None —
        US 16:00 must NOT map to slot 391 in a 390-bar session (P1-17).
        """
        t = ts.time()
        for seg in self.segments:
            if seg.start <= t <= seg.end:
                delta = (t.hour * 60 + t.minute) - (
                    seg.start.hour * 60 + seg.start.minute
                )
                slot = seg.slot_offset + delta + 1
                if slot > self.slot_count:
                    return None
                return slot
        return None

    def slot_for_timestamp(self, ts: datetime) -> int | None:
        """Map a *timezone-aware* datetime to a SessionSlotId.

        Production must pass tz-aware timestamps (A-share QuoteTime is stored
        UTC, US is Eastern/DST-aware); a naive timestamp is rejected outright
        rather than silently interpreted in the wrong wall-clock (P1-16).
        """
        if ts.tzinfo is None:
            raise ValueError(
                f"{self.session_id} slot_for_timestamp requires a tz-aware "
                "datetime; got naive {ts!r}"
            )
        local = ts.astimezone(_tz(self.timezone))
        return self.slot_for_local(local)

    def for_date(self, trade_date: Any) -> "SessionSpec":
        """Return the session as-of ``trade_date`` (early-close aware, P1-18).

        Without a Calendar provider the base session is returned unchanged; a
        runtime calendar should subclass/replace ``early_close_dates`` so an
        early-close day gets a down-weighted/excluded session contract.
        """
        import pandas as pd

        try:
            key = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        except Exception:  # pragma: no cover - defensive
            return self
        if key in self.early_close_dates and self.early_close_policy != "none":
            # Half-day: use the segments but shrink the close edge.  For US a
            # typical early close ends 13:00; we expose the flag for the caller
            # (volume/volatility factors down-weight or exclude the day).
            return SessionSpec(
                session_id=self.session_id,
                timezone=self.timezone,
                segments=self.segments,
                slot_count=self.slot_count,
                minute_bars=self.minute_bars,
                early_close_policy=self.early_close_policy,
                notes=self.notes + f" [early-close {key}]",
                bar_convention=self.bar_convention,
                early_close_dates=self.early_close_dates,
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timezone": self.timezone,
            "segments": [
                {"start": s.start.strftime("%H:%M"), "end": s.end.strftime("%H:%M")}
                for s in self.segments
            ],
            "slot_count": self.slot_count,
            "minute_bars": self.minute_bars,
            "early_close_policy": self.early_close_policy,
            "bar_convention": self.bar_convention,
            "notes": self.notes,
        }


def _tz(name: str) -> Any:
    import zoneinfo

    return zoneinfo.ZoneInfo(name)


# A-share: 09:31-11:30 (120 bars) + 13:01-15:00 (120 bars) = 240 bars/day.
ASHARE_SESSION = SessionSpec(
    session_id="ASHARE_CONTINUOUS",
    timezone=TIMEZONE_ASHARE,
    segments=(
        SessionSegment(start=time(9, 31), end=time(11, 30), slot_offset=0),
        SessionSegment(start=time(13, 1), end=time(15, 0), slot_offset=120),
    ),
    slot_count=240,
    early_close_policy="none",
    notes="09:31-11:30 + 13:01-15:00 CST (UTC+8); no 09:30/13:00 bars; no lunch bars",
)

# US: regular 09:30-16:00 Eastern, DST-aware; early-close half-days flagged.
# Bar-START convention: 390 minute bars labelled 09:30..15:59; 16:00 is the
# exclusive session end (slot_for_local(16:00) -> None, never slot 391).
US_SESSION = SessionSpec(
    session_id="US_REGULAR",
    timezone=TIMEZONE_US,
    segments=(SessionSegment(start=time(9, 30), end=time(16, 0), slot_offset=0),),
    slot_count=390,
    early_close_policy="down_weight",
    bar_convention="bar_start",
    notes="09:30-16:00 Eastern (bar-start, 390 bars); DST via America/New_York; "
          "is_early_close ~51 days/year",
)

_SESSION_BY_MARKET = {"ashare": ASHARE_SESSION, "us": US_SESSION}


def session_for(market: str) -> SessionSpec:
    key = str(market).strip().lower()
    try:
        return _SESSION_BY_MARKET[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"no SessionSpec for market {market!r}") from exc


def session_for_date(market: str, trade_date: Any) -> SessionSpec:
    """Base session for ``market`` as-of ``trade_date`` (early-close aware)."""
    return session_for(market).for_date(trade_date)


__all__ = [
    "ASHARE_SESSION",
    "SessionSegment",
    "SessionSpec",
    "US_SESSION",
    "session_for",
    "session_for_date",
]
