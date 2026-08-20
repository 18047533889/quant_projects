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

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

TIMEZONE_ASHARE = "Asia/Shanghai"
TIMEZONE_US = "America/New_York"


class EarlyCloseNormalizationPolicy(str, enum.Enum):
    """R40 #237：early-close 日的明确归一化策略（替代裸字符串常量）。

    每个策略有明确的数学公式（按 session duration ratio）：

    * ``NONE`` — 不调整（1.0）；
    * ``DOWN_WEIGHT`` — 按可用时长比例缩放
      ``session_duration / full_duration``（US 半日市约 210/390）；
    * ``EXCLUDE`` — 该日从样本中剔除（权重 0.0）。

    ``applicable_operator_families`` 声明该策略适用的 operator 家族
    （volume / volatility / session-profile…）；``semantic_version`` 绑定公式
    版本，任何公式变更都会改变依赖它的 factor 语义身份。
    """

    NONE = "none"
    DOWN_WEIGHT = "down_weight"
    EXCLUDE = "exclude"

    def normalization_formula(
        self, full_duration_minutes: int, session_duration_minutes: int
    ) -> float:
        """按 session duration ratio 的明确公式。"""
        full = max(1, int(full_duration_minutes))
        sess = int(session_duration_minutes)
        if self is EarlyCloseNormalizationPolicy.NONE:
            return 1.0
        if self is EarlyCloseNormalizationPolicy.EXCLUDE:
            return 0.0
        # DOWN_WEIGHT: scale by the fraction of regular session time available.
        return float(max(0.0, sess) / full)

    @property
    def applicable_operator_families(self) -> tuple[str, ...]:
        if self is EarlyCloseNormalizationPolicy.DOWN_WEIGHT:
            return ("volume", "volatility", "session_profile", "amount_share")
        if self is EarlyCloseNormalizationPolicy.EXCLUDE:
            return ("volume", "volatility", "session_profile")
        return ()

    @property
    def semantic_version(self) -> str:
        return "r40#237-v1"

    @classmethod
    def from_value(cls, value: Any) -> "EarlyCloseNormalizationPolicy":
        if isinstance(value, EarlyCloseNormalizationPolicy):
            return value
        return cls(str(value or "none").strip().lower())


@dataclass(frozen=True)
class SessionSegment:
    """One contiguous intraday trading segment (local wall-clock time)."""

    start: time
    end: time
    slot_offset: int = 0  # bar index of this segment's first minute


@dataclass(frozen=True)
class SessionSpec:
    """A market's intraday session contract.

    R40 #232：本类是 **immutable projection** —— 业务规则的单一权威是
    :class:`market.exchange_session_calendar.ExchangeSessionCalendar`。
    ``SessionSpec`` 保留给既有调用方（minute slot 映射 / early-close 感知），
    但不再维护独立的 calendar 业务规则；真实 holiday / DST / early-close 来源
    应经 ``ExchangeSessionCalendar`` 生成，再投影成本对象。
    """

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
    # R17-054: actual per-date early-close times from the Calendar provider
    # (ISO date -> time-of-day); None when unknown.  ``for_date`` shortens the
    # session to these.
    _early_close_times: dict[str, Any] | None = field(
        default=None, repr=False, compare=False, hash=False
    )

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

        R17-054: an early-close date REALLY shortens the session — the Calendar
        provider records the actual close time, and the returned spec carries the
        correct segments / slot_count / expected_bar_count so volume/volatility /
        session-profile operators normalize on the real day instead of seeing a
        note-only "half day".  Without a Calendar provider the base session is
        returned unchanged.
        """
        import pandas as pd

        try:
            key = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        except Exception:  # pragma: no cover - defensive
            return self
        close_time = None
        if self._early_close_times is not None:
            close_time = self._early_close_times.get(key)
        if close_time is None:
            close_time = self._default_early_close_time()
        if key in self.early_close_dates and self.early_close_policy != "none":
            # Build the shortened session.  US typical early close ends 13:00;
            # ``close_time`` (from the Calendar provider) overrides.
            close_t = self._to_time(close_time)
            segments = tuple(
                seg if seg.end <= close_t
                else SessionSegment(start=seg.start, end=close_t, slot_offset=seg.slot_offset)
                for seg in self.segments
            )
            # Bar count = sum of minute deltas over the shortened segments.
            slot_count = sum(
                int(
                    (seg.end.hour * 60 + seg.end.minute)
                    - (seg.start.hour * 60 + seg.start.minute)
                )
                for seg in segments
            )
            return SessionSpec(
                session_id=self.session_id,
                timezone=self.timezone,
                segments=segments,
                slot_count=slot_count,
                minute_bars=self.minute_bars,
                early_close_policy=self.early_close_policy,
                notes=self.notes + f" [early-close {key} -> close {close_t:%H:%M}, {slot_count} bars]",
                bar_convention=self.bar_convention,
                early_close_dates=self.early_close_dates,
            )
        return self

    def with_early_close_times(self, times: dict[str, Any]) -> "SessionSpec":
        """Return a copy carrying the Calendar provider's actual close times.

        The supplied dates are also folded into ``early_close_dates`` so
        ``for_date`` recognizes them as early closes (R17-054).
        """
        return SessionSpec(
            session_id=self.session_id,
            timezone=self.timezone,
            segments=self.segments,
            slot_count=self.slot_count,
            minute_bars=self.minute_bars,
            early_close_policy=self.early_close_policy,
            notes=self.notes,
            bar_convention=self.bar_convention,
            early_close_dates=self.early_close_dates | frozenset(times.keys()),
            _early_close_times=dict(times),
        )

    def _default_early_close_time(self) -> Any:
        """Default early-close time when the Calendar provider is absent (13:00)."""
        try:
            from datetime import time as _time

            return _time(13, 0)
        except Exception:  # pragma: no cover - defensive
            return None

    @staticmethod
    def _to_time(value: Any) -> Any:
        from datetime import time as _time

        if isinstance(value, _time):
            return value
        try:
            return _time(value.hour, value.minute)
        except Exception:  # pragma: no cover - defensive
            return _time(13, 0)

    def normalization(self) -> EarlyCloseNormalizationPolicy:
        """R40 #237：把字符串 ``early_close_policy`` 解析为归一化策略对象。"""
        return EarlyCloseNormalizationPolicy.from_value(self.early_close_policy)

    def calendar_digest(self) -> str:
        """R40 #236：绑定 calendar version + early-close 来源 + segments +
        timezone 的 digest（供 factor identity / cache 使用）。"""
        h = hashlib.sha256()
        h.update(b"SessionSpecV2")
        h.update(b"\x00" + self.session_id.encode("utf-8"))
        h.update(b"\x00" + self.timezone.encode("utf-8"))
        for seg in self.segments:
            h.update(b"\x00" + f"{seg.start:%H:%M}-{seg.end:%H:%M}".encode("utf-8"))
        h.update(b"\x00" + str(self.slot_count).encode("utf-8"))
        h.update(b"\x00" + self.bar_convention.encode("utf-8"))
        h.update(b"\x00" + self.normalization().value.encode("utf-8"))
        h.update(b"\x00" + ",".join(sorted(self.early_close_dates)).encode("utf-8"))
        ec_times = dict(self._early_close_times or {})
        h.update(
            b"\x00"
            + ",".join(
                f"{d}:{t.strftime('%H:%M') if hasattr(t, 'strftime') else t}"
                for d, t in sorted(ec_times.items())
            ).encode("utf-8")
        )
        return h.hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        ec_times = dict(self._early_close_times or {})
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
            # R40 #236: early-close 信息必须进 to_dict（不再是 notes-only）。
            "early_close_dates": sorted(self.early_close_dates),
            "early_close_times": {
                d: (t.strftime("%H:%M") if hasattr(t, "strftime") else t)
                for d, t in sorted(ec_times.items())
            },
            "early_close_normalization_policy": self.normalization().value,
            "calendar_digest": self.calendar_digest(),
        }


def _tz(name: str) -> Any:
    import zoneinfo

    return zoneinfo.ZoneInfo(name)


# A-share: 09:31-11:30 (120 bars) + 13:01-15:00 (120 bars) = 240 bars/day.
# R17-053: the A-share COS minute mirror labels each bar with its END timestamp
# (09:31 = the 09:30-09:31 bar, 15:00 = the last bar) — bar_convention=bar_end,
# NOT the SessionSpec bar_start default.
ASHARE_SESSION = SessionSpec(
    session_id="ASHARE_CONTINUOUS",
    timezone=TIMEZONE_ASHARE,
    segments=(
        SessionSegment(start=time(9, 31), end=time(11, 30), slot_offset=0),
        SessionSegment(start=time(13, 1), end=time(15, 0), slot_offset=120),
    ),
    slot_count=240,
    early_close_policy="none",
    bar_convention="bar_end",
    notes="09:31-11:30 + 13:01-15:00 CST (UTC+8), BAR-END labels; no 09:30/13:00 "
          "bars; no lunch bars (R17-053)",
)

# US: regular 09:30-16:00 Eastern, DST-aware; early-close half-days flagged.
# Bar-START convention: 390 minute bars labelled 09:30..15:59; 16:00 is the
# exclusive session end (slot_for_local(16:00) -> None, never slot 391).
# R17-055: US early-close days are ~tens of TRUE dates in the whole history, not
# "~51 per year" — the wrong year-frequency is gone from the docs.
US_SESSION = SessionSpec(
    session_id="US_REGULAR",
    timezone=TIMEZONE_US,
    segments=(SessionSegment(start=time(9, 30), end=time(16, 0), slot_offset=0),),
    slot_count=390,
    early_close_policy="down_weight",
    bar_convention="bar_start",
    notes="09:30-16:00 Eastern (bar-start, 390 bars); DST via America/New_York; "
          "early close dates are ~tens of true days in all history, not per-year "
          "(R17-055)",
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
    "EarlyCloseNormalizationPolicy",
    "SessionSegment",
    "SessionSpec",
    "US_SESSION",
    "session_for",
    "session_for_date",
]
