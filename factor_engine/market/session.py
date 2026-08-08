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

from dataclasses import dataclass
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

    def slot_for_local(self, ts: datetime) -> int | None:
        """Map a local wall-clock datetime to a SessionSlotId (1-based bar)."""
        t = ts.time()
        for seg in self.segments:
            if seg.start <= t <= seg.end:
                delta = (t.hour * 60 + t.minute) - (
                    seg.start.hour * 60 + seg.start.minute
                )
                return seg.slot_offset + delta + 1
        return None

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
            "notes": self.notes,
        }


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
US_SESSION = SessionSpec(
    session_id="US_REGULAR",
    timezone=TIMEZONE_US,
    segments=(SessionSegment(start=time(9, 30), end=time(16, 0), slot_offset=0),),
    slot_count=390,
    early_close_policy="down_weight",
    notes="09:30-16:00 Eastern; DST via America/New_York; is_early_close ~51 days/year",
)

_SESSION_BY_MARKET = {"ashare": ASHARE_SESSION, "us": US_SESSION}


def session_for(market: str) -> SessionSpec:
    key = str(market).strip().lower()
    try:
        return _SESSION_BY_MARKET[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"no SessionSpec for market {market!r}") from exc


__all__ = [
    "ASHARE_SESSION",
    "SessionSegment",
    "SessionSpec",
    "US_SESSION",
    "session_for",
]
