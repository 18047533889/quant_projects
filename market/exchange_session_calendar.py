# -*- coding: utf-8 -*-
"""ExchangeSessionCalendar —— 交易所交易会话日历的**单一权威**（R40 #232）。

历史上市场/日历分裂成两套独立模型：

* :class:`market.session.SessionSpec` —— 有 early_close_dates / for_date() 感知，
  但没有权威节假日 / DST / 交易所语义；
* :class:`runtime.session_calendar.SessionCalendar` —— 有 segments / bar_freq /
  offset_bars，但固定 09:30-16:00、无 timezone / early close / DST。

R40 #232 引入 :class:`ExchangeSessionCalendar` 作为唯一维护业务规则的地方，
把 market / exchange / timezone / calendar_version / trade_date / segments /
auctions / lunch break / early close / timestamp convention / bar grids /
holidays / DST 全部收拢到一个对象。``SessionSpec`` / ``SessionCalendar`` 退化为
**不可变投影**：只读快照，不再维护独立业务规则。

``for_date(trade_date)`` 返回按当日生效的会话（regular close / early close
time / DST timezone / holiday）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any

import pandas as pd

#: calendar_version 的默认值。真实来源由 dataaccess 簇提供；此处给确定性默认。
DEFAULT_CALENDAR_VERSION = "exchange_v1"

#: A 股默认午休区间（11:30-13:00）。
ASHARE_LUNCH = ("11:30", "13:00")
#: US 无午休。
US_LUNCH = ()


@dataclass(frozen=True)
class AuctionWindow:
    """集合竞价窗口（开/收盘集合竞价）。"""

    label: str  # "open_auction" / "close_auction"
    start: time
    end: time


@dataclass(frozen=True)
class ExchangeSessionCalendar:
    """交易会话日历的单一权威（R40 #232）。

    ``segments`` 是会话内的连续交易段（本地 wall-clock）；``auctions`` 是集合
    竞价窗口；``lunch_break`` 是午休（A 股）；``early_close_dates`` /
    ``early_close_times`` 是半日市；``holidays`` 是休市日；``dst_aware`` 说明
    时区是否随 DST 变化（US yes / A 股 no）。``calendar_version`` 绑定任何
    calendar 派生 hash（identity / cache / checkpoint）。
    """

    market: str
    exchange: str
    timezone: str
    segments: tuple[tuple[str, str], ...] = (("09:30", "16:00"),)
    auctions: tuple[AuctionWindow, ...] = ()
    lunch_break: tuple[str, str] | None = None
    early_close_dates: frozenset[str] = frozenset()
    early_close_times: dict[str, str] = field(default_factory=dict)
    timestamp_convention: str = "bar_start"
    bar_freq: str = "1min"
    holidays: frozenset[str] = frozenset()
    dst_aware: bool = True
    calendar_version: str = DEFAULT_CALENDAR_VERSION
    notes: str = ""

    def __post_init__(self) -> None:
        market = str(self.market or "").strip().lower()
        if market in {"a_share", "cn", "china"}:
            market = "ashare"
        if market not in {"ashare", "us", "hk"}:
            raise ValueError(f"ExchangeSessionCalendar does not support market {market!r}")
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "timezone", str(self.timezone or "").strip())
        object.__setattr__(self, "exchange", str(self.exchange or "").strip().upper())
        convention = str(self.timestamp_convention or "bar_start").lower()
        if convention not in {"bar_start", "bar_end"}:
            raise ValueError("timestamp_convention must be bar_start or bar_end")
        object.__setattr__(self, "timestamp_convention", convention)
        object.__setattr__(
            self,
            "holidays",
            frozenset(pd.Timestamp(h).normalize() for h in (self.holidays or ())),
        )
        if not self.segments:
            raise ValueError("ExchangeSessionCalendar requires non-empty segments")
        for start, stop in self.segments:
            if self._hhmm(stop) <= self._hhmm(start):
                raise ValueError(f"segment {start}-{stop} must be increasing")

    @staticmethod
    def _hhmm(value: Any) -> int:
        if isinstance(value, time):
            return value.hour * 60 + value.minute
        h, m = (int(p) for p in str(value).split(":"))
        return h * 60 + m

    @staticmethod
    def _to_time(value: Any) -> time:
        if isinstance(value, time):
            return value
        try:
            h, m = (int(p) for p in str(value).split(":"))
            return time(h, m)
        except Exception:  # pragma: no cover - defensive
            return time(13, 0)

    # ------------------------------------------------------------------
    # per-date authoritative session
    # ------------------------------------------------------------------

    def close_time_for(self, trade_date: Any) -> time:
        """当日收盘时间（early close 感知）。"""
        key = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        if key in self.early_close_dates:
            raw = self.early_close_times.get(key, "13:00")
            return self._to_time(raw)
        return self._to_time(self.segments[-1][1])

    def for_date(self, trade_date: Any) -> "ExchangeSessionCalendar":
        """返回该 trade_date 生效的会话（early close 裁剪段 / slot_count）。"""
        key = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        if key not in self.early_close_dates:
            return self
        close_t = self.close_time_for(key)
        segments = tuple(
            seg if self._hhmm(seg[1]) <= self._hhmm(close_t)
            else (seg[0], close_t.strftime("%H:%M"))
            for seg in self.segments
        )
        return ExchangeSessionCalendar(
            market=self.market,
            exchange=self.exchange,
            timezone=self.timezone,
            segments=segments,
            auctions=self.auctions,
            lunch_break=self.lunch_break,
            early_close_dates=self.early_close_dates,
            early_close_times=self.early_close_times,
            timestamp_convention=self.timestamp_convention,
            bar_freq=self.bar_freq,
            holidays=self.holidays,
            dst_aware=self.dst_aware,
            calendar_version=self.calendar_version,
            notes=self.notes + f" [early-close {key} -> {close_t:%H:%M}]",
        )

    def is_trading_day(self, day: Any) -> bool:
        day = pd.Timestamp(day)
        if day.weekday() >= 5:
            return False
        return day.normalize() not in self.holidays

    def expected_sessions(self, start: Any, end: Any) -> pd.DatetimeIndex:
        """Return authoritative exchange sessions in the inclusive range."""
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            return pd.DatetimeIndex([])
        days = pd.date_range(start_ts, end_ts, freq="D")
        return pd.DatetimeIndex([day for day in days if self.is_trading_day(day)])

    def shift_session(self, session: Any, bars: int) -> pd.Timestamp:
        """Shift a session by exchange-session bars, independent of data gaps."""
        anchor = pd.Timestamp(session).normalize()
        if not self.is_trading_day(anchor):
            raise ValueError(f"{anchor.date()} is not an exchange session")
        remaining = abs(int(bars))
        direction = 1 if bars >= 0 else -1
        current = anchor
        while remaining:
            current += pd.Timedelta(days=direction)
            if self.is_trading_day(current):
                remaining -= 1
        return current

    def session_distance(self, start: Any, end: Any) -> int:
        """Number of exchange-session steps from ``start`` to ``end``."""
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if not self.is_trading_day(start_ts) or not self.is_trading_day(end_ts):
            raise ValueError("session_distance endpoints must be exchange sessions")
        if start_ts == end_ts:
            return 0
        if end_ts > start_ts:
            return len(self.expected_sessions(start_ts, end_ts)) - 1
        return -(len(self.expected_sessions(end_ts, start_ts)) - 1)

    def session_local(self, timestamps: Any) -> pd.DatetimeIndex:
        """把 timestamps 统一到 session-local timezone（naive 视为本地）。"""
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if ts.tz is not None:
            return ts.tz_convert(self.timezone).tz_localize(None)
        return ts

    def bar_slots(self, day: Any) -> list[pd.Timestamp]:
        """当日合法 bar slot（segment-aware，跳过午休 / early close）。"""
        cal = self.for_date(day)
        day_ts = pd.Timestamp(day).normalize()
        width = 1  # 1min grid
        slots: list[pd.Timestamp] = []
        for start, stop in cal.segments:
            s, e = self._hhmm(start), self._hhmm(stop)
            if cal.timestamp_convention == "bar_end":
                k = 1
                while s + k * width <= e:
                    slots.append(day_ts + pd.Timedelta(minutes=s + k * width))
                    k += 1
            else:
                k = 0
                while s + k * width < e:
                    slots.append(day_ts + pd.Timedelta(minutes=s + k * width))
                    k += 1
        return slots

    # ------------------------------------------------------------------
    # projections
    # ------------------------------------------------------------------

    def to_session_spec(self) -> Any:
        """投影为 ``market.session.SessionSpec``（只读快照，无独立规则）。"""
        from market.session import SessionSegment, SessionSpec

        segments = tuple(
            SessionSegment(
                start=self._to_time(s),
                end=self._to_time(e),
                slot_offset=sum(
                    self._hhmm(a) - self._hhmm(b) for a, b in self.segments[:i]
                ),
            )
            for i, (s, e) in enumerate(self.segments)
        )
        slot_count = sum(
            self._hhmm(e) - self._hhmm(s) for s, e in self.segments
        )
        return SessionSpec(
            session_id=f"{self.market.upper()}_{self.exchange}",
            timezone=self.timezone,
            segments=segments,
            slot_count=slot_count,
            early_close_policy="down_weight" if self.market == "us" else "none",
            notes=self.notes,
            bar_convention=self.timestamp_convention,
            early_close_dates=self.early_close_dates,
            _early_close_times={
                d: self._to_time(t) for d, t in self.early_close_times.items()
            },
        )

    def to_session_calendar(self) -> Any:
        """投影为 ``runtime.session_calendar.SessionCalendar``。"""
        from runtime.session_calendar import SessionCalendar

        return SessionCalendar(
            market=self.market.upper(),
            session="regular",
            bar_freq=self.bar_freq,
            segments=self.segments,
            timestamp_convention=self.timestamp_convention,
            holidays=self.holidays,
            timezone=self.timezone,
            authoritativeness="EXCHANGE_CERTIFIED",
        )

    def calendar_digest(self) -> str:
        """绑定 calendar version + early-close 来源 + segments + timezone 的 hash。"""
        import hashlib

        payload = {
            "market": self.market,
            "exchange": self.exchange,
            "timezone": self.timezone,
            "calendar_version": self.calendar_version,
            "segments": self.segments,
            "timestamp_convention": self.timestamp_convention,
            "early_close_dates": sorted(self.early_close_dates),
            "early_close_times": sorted(self.early_close_times.items()),
            "holidays": sorted(str(h) for h in self.holidays),
            "dst_aware": self.dst_aware,
            "lunch_break": self.lunch_break,
        }
        raw = ",".join(f"{k}={v}" for k, v in sorted(payload.items()))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def default_exchange_session_calendar(market: str) -> ExchangeSessionCalendar:
    """R40 #235：US / A 股 / HK 的权威会话日历。

    US：America/New_York（DST-aware）09:30-16:00；
    A 股：Asia/Shanghai（无 DST）09:30-11:30 + 13:00-15:00，午休；
    HK：Asia/Hong_Kong 09:30-12:00 + 13:00-16:00，午休。
    """
    key = str(market).strip().lower()
    if key in {"a_share", "cn", "china"}:
        key = "ashare"
    if key == "ashare":
        return ExchangeSessionCalendar(
            market="ashare",
            exchange="SSE",
            timezone="Asia/Shanghai",
            segments=(("09:30", "11:30"), ("13:00", "15:00")),
            auctions=(
                AuctionWindow("open_auction", time(9, 15), time(9, 25)),
                AuctionWindow("close_auction", time(14, 57), time(15, 0)),
            ),
            lunch_break=ASHARE_LUNCH,
            timestamp_convention="bar_end",
            dst_aware=False,
            calendar_version=DEFAULT_CALENDAR_VERSION,
            notes="A-share 09:30-11:30 + 13:00-15:00 CST, bar-end labels",
        )
    if key == "us":
        return ExchangeSessionCalendar(
            market="us",
            exchange="NYSE",
            timezone="America/New_York",
            segments=(("09:30", "16:00"),),
            auctions=(
                AuctionWindow("open_auction", time(9, 30), time(9, 30)),
                AuctionWindow("close_auction", time(15, 55), time(16, 0)),
            ),
            lunch_break=None,
            timestamp_convention="bar_start",
            dst_aware=True,
            calendar_version=DEFAULT_CALENDAR_VERSION,
            notes="US regular 09:30-16:00 Eastern, DST-aware; early close ~13:00",
        )
    if key == "hk":
        return ExchangeSessionCalendar(
            market="hk",
            exchange="HKEX",
            timezone="Asia/Hong_Kong",
            segments=(("09:30", "12:00"), ("13:00", "16:00")),
            lunch_break=("12:00", "13:00"),
            timestamp_convention="bar_start",
            dst_aware=False,
            calendar_version=DEFAULT_CALENDAR_VERSION,
            notes="HK 09:30-12:00 + 13:00-16:00 HKT, no DST",
        )
    raise KeyError(f"no ExchangeSessionCalendar for market {market!r}")


__all__ = [
    "ASHARE_LUNCH",
    "AuctionWindow",
    "DEFAULT_CALENDAR_VERSION",
    "ExchangeSessionCalendar",
    "default_exchange_session_calendar",
]
