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


class CalendarAuthorityError(RuntimeError):
    """R40 #233：production 需要 EXCHANGE_CERTIFIED 日历，但当前是
    WEEKDAY_APPROXIMATION（无真实节假日集）—— hard fail。"""


#: 日历权威性：``EXCHANGE_CERTIFIED``（真实交易所节假日集）vs
#: ``WEEKDAY_APPROXIMATION``（仅周一到周五近似，无真实 holidays）。
EXCHANGE_CERTIFIED = "EXCHANGE_CERTIFIED"
WEEKDAY_APPROXIMATION = "WEEKDAY_APPROXIMATION"


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
    #: R32-P0-004: 跨日偏移的交易日过滤。缺省按周一到周五（A 股常规）；真实
    #: 节假日日历由上层 TradingCalendar 提供（``holidays`` 冻结日期集）。
    holidays: frozenset[str] = frozenset()
    #: R40 #234：会话本地时区（必填，默认 UTC）。session_key / minute_ordinal /
    #: segment_id / bar_slots 先统一到该时区再算。
    timezone: str = "UTC"
    #: R40 #233：日历权威性。默认 WEEKDAY_APPROXIMATION（向后兼容既有调用方）；
    #: production 的 offset_bars / warmup_load_start 要求 EXCHANGE_CERTIFIED。
    authoritativeness: str = WEEKDAY_APPROXIMATION

    def __post_init__(self) -> None:
        market = str(self.market or "US").upper()
        convention = str(self.timestamp_convention).lower()
        if convention not in {"bar_end", "bar_start"}:
            raise ValueError("timestamp_convention must be bar_end or bar_start")
        segments = self.segments
        if segments is None:
            if market in {"CN", "ASHARE", "A_SHARE"}:
                segments = (("09:30", "11:30"), ("13:00", "15:00"))
            elif market in {"HK", "HONG_KONG"}:
                segments = (("09:30", "12:00"), ("13:00", "16:00"))
            else:
                segments = (("09:30", "16:00"),)
        parsed = tuple((str(start), str(stop)) for start, stop in segments)
        if not parsed or any(_minute(stop) <= _minute(start) for start, stop in parsed):
            raise ValueError("session segments must be non-empty increasing intervals")
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "session", str(self.session or "regular").lower())
        object.__setattr__(self, "bar_freq", str(self.bar_freq))
        object.__setattr__(self, "segments", parsed)
        object.__setattr__(self, "timestamp_convention", convention)
        object.__setattr__(
            self,
            "holidays",
            frozenset(
                pd.Timestamp(h).normalize()
                for h in (self.holidays or ())
            ),
        )
        if self.session_bars <= 0:
            raise ValueError("SessionCalendar requires a positive bar count")
        auth = str(self.authoritativeness or "").strip().upper()
        if auth not in {EXCHANGE_CERTIFIED, WEEKDAY_APPROXIMATION}:
            raise ValueError(
                f"authoritativeness must be {EXCHANGE_CERTIFIED} or "
                f"{WEEKDAY_APPROXIMATION}, got {self.authoritativeness!r}"
            )
        object.__setattr__(self, "authoritativeness", auth)
        object.__setattr__(self, "timezone", str(self.timezone or "UTC"))

    def _session_local_ts(self, timestamps: Iterable[object]) -> pd.DatetimeIndex:
        """R40 #234：把 timestamps 统一到 session-local timezone。

        naive 输入视为已是会话本地；tz-aware 输入先 tz_convert 到
        ``self.timezone`` 再剥离 tz（本地 wall-clock）。
        """
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if ts.tz is not None:
            return ts.tz_convert(self.timezone).tz_localize(None)
        return ts

    def require_exchange_certified(self, *, mode: str = "production") -> None:
        """R40 #233：production 要求 EXCHANGE_CERTIFIED 才允许
        offset_bars / warmup_load_start。WEEKDAY_APPROXIMATION（空节假日集
        的 weekday 近似）→ hard fail，不再继续用近似日历跨日偏移。
        """
        if str(mode).strip().lower() == "production" and self.authoritativeness != EXCHANGE_CERTIFIED:
            raise CalendarAuthorityError(
                f"production requires an EXCHANGE_CERTIFIED calendar for "
                f"offset_bars / warmup_load_start; current calendar for "
                f"{self.market!r} is {self.authoritativeness} (holidays set "
                f"size={len(self.holidays)}) — a weekday approximation cannot "
                "certify cross-day bar offsets"
            )

    def _is_trading_day(self, day: pd.Timestamp) -> bool:
        """R32-P0-004: 交易日过滤 —— 周末 + 显式 holidays 非交易日。"""
        if day.weekday() >= 5:
            return False
        return day.normalize() not in self.holidays

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
        """R40 #234: 统一到 session-local timezone 后取交易日。"""
        return self._session_local_ts(timestamps).normalize()

    def minute_ordinal(self, timestamps: Iterable[object]) -> np.ndarray:
        """Return continuous in-session minute ordinals; recess/outside is -1.

        R40 #234: timestamps 先统一到 session-local timezone。
        """
        ts = self._session_local_ts(timestamps)
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
        """Return minute ordinal within each segment; recess/outside is -1.

        R40 #234: timestamps 先统一到 session-local timezone。
        """
        ts = self._session_local_ts(timestamps)
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
        """Return the segment number, with -1 for recess/outside timestamps.

        R40 #234: timestamps 先统一到 session-local timezone。
        """
        ts = self._session_local_ts(timestamps)
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

    def bar_slots(self, day: str | pd.Timestamp) -> list[pd.Timestamp]:
        """单交易日的合法 bar slot 时间戳（session-slot aware，跳过午休）。

        R32-P0-004: 会话结构（09:30 开盘 / 11:30–13:00 午休 / 集合竞价 / 半日
        市 / early close / DST）决定合法 bar grid —— 不再是「交易日零点 +
        bar_duration × N」的线性算术。
        """
        day = pd.Timestamp(day).normalize()
        width = max(1, int(self.bar_timedelta / pd.Timedelta(minutes=1)))
        slots: list[pd.Timestamp] = []
        for start_text, stop_text in self.segments or ():
            start, stop = _minute(start_text), _minute(stop_text)
            if self.timestamp_convention == "bar_end":
                # bar_end 用 (open, close]：slot = start + k*width，k>=1。
                k = 1
                while start + k * width <= stop:
                    slots.append(day + pd.Timedelta(minutes=start + k * width))
                    k += 1
            else:
                # bar_start 用 [open, close)：slot = start + k*width，k>=0。
                k = 0
                while start + k * width < stop:
                    slots.append(day + pd.Timedelta(minutes=start + k * width))
                    k += 1
        return slots

    def offset_bars(self, anchor: str | pd.Timestamp, n_bars: int, *, mode: str = "research") -> pd.Timestamp:
        """R32-P0-004: session-slot-aware bar offset（跳过午休 / 跨交易日）。

        anchor 若不在合法 slot 上，先按偏移方向吸附到最近的合法 slot；再沿
        session grid 逐 slot 偏移 n_bars。跨日时进入下一交易日（business day
        近似；节假日裁剪由上层 TradingCalendar 负责）。n_bars 有界（lookback
        通常 ≤ 数千），逐 slot 走是廉价且正确的。

        R40 #233：``mode="production"`` 时要求 EXCHANGE_CERTIFIED 日历，否则
        hard fail——跨日偏移不能靠 weekday 近似在 production 里静默继续。
        R40 #234：anchor 先统一到 session-local timezone。
        """
        self.require_exchange_certified(mode=mode)
        n = int(n_bars)
        ts = self._session_local_ts([pd.Timestamp(anchor)])[0]
        if n == 0:
            return ts
        sign = 1 if n > 0 else -1
        steps = abs(n)
        day = ts.normalize()
        slots = self.bar_slots(day)
        # 定位 anchor 在当前日 slot grid 中的位置（按偏移方向吸附）。
        pos = 0
        if slots:
            if sign > 0:
                pos = next((i for i, s in enumerate(slots) if s >= ts), len(slots))
            else:
                pos = next(
                    (i for i in range(len(slots) - 1, -1, -1) if slots[i] <= ts),
                    -1,
                )
        cur = ts
        while steps > 0:
            if sign > 0:
                if slots and pos < len(slots) - 1:
                    pos += 1
                    cur = slots[pos]
                    steps -= 1
                else:
                    day += pd.Timedelta(days=1)
                    while not self._is_trading_day(day):
                        day += pd.Timedelta(days=1)
                    slots = self.bar_slots(day)
                    if slots:
                        pos = 0
                        cur = slots[0]
                        steps -= 1
            else:
                if slots and pos > 0:
                    pos -= 1
                    cur = slots[pos]
                    steps -= 1
                else:
                    day -= pd.Timedelta(days=1)
                    while not self._is_trading_day(day):
                        day -= pd.Timedelta(days=1)
                    slots = self.bar_slots(day)
                    if slots:
                        pos = len(slots) - 1
                        cur = slots[-1]
                        steps -= 1
        return cur

    def warmup_load_start(
        self,
        requested_start: str | pd.Timestamp,
        *,
        lookback_bars: int,
        mode: str = "research",
    ) -> pd.Timestamp:
        """R40 #233：``mode="production"`` 要求 EXCHANGE_CERTIFIED（warmup
        load 依赖跨日 offset，weekday 近似不能用于 production 预加载窗口）。"""
        self.require_exchange_certified(mode=mode)
        lb = max(0, int(lookback_bars))
        if lb <= 0:
            return pd.Timestamp(requested_start)
        anchor = self._session_local_ts([pd.Timestamp(requested_start)])[0]
        if anchor == anchor.normalize():
            anchor += self.bar_timedelta * (self.session_bars - 1)
        return self.offset_bars(anchor, -lb, mode=mode)


# Backward-compatible public name used by existing callers.
SessionBarCalendar = SessionCalendar
