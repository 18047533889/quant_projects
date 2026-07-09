# -*- coding: utf-8
"""日内 session bar 日历：分钟/小时频 warmup 按 bar 精确扩窗。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cleaned_operators.operator_policy import bar_freq_to_timedelta, bars_per_day


@dataclass(frozen=True)
class SessionBarCalendar:
    """按 bar 时长做 timestamp 级偏移（不依赖完整 RTH 列表）。"""

    bar_freq: str
    bars_per_session: int | None = None
    market: str = "US"
    session: str = "regular"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bar_freq", str(self.bar_freq))
        object.__setattr__(self, "market", str(self.market or "US").upper())
        object.__setattr__(self, "session", str(self.session or "regular").lower())
        bpd = self.bars_per_session
        if bpd is None:
            bpd = bars_per_day(self.bar_freq, market=self.market, session=self.session)
        if bpd <= 0:
            raise ValueError(
                f"SessionBarCalendar 无效 bar_freq={self.bar_freq!r} "
                f"market={self.market!r} session={self.session!r}"
            )

    @property
    def bar_timedelta(self) -> pd.Timedelta:
        return bar_freq_to_timedelta(self.bar_freq)

    @property
    def session_bars(self) -> int:
        if self.bars_per_session is not None:
            return self.bars_per_session
        return bars_per_day(self.bar_freq, market=self.market, session=self.session)

    def offset_bars(self, anchor: str | pd.Timestamp, n_bars: int) -> pd.Timestamp:
        """相对 anchor 偏移 n 个 bar（n<0 向历史）。"""
        base = pd.Timestamp(anchor)
        return base + self.bar_timedelta * int(n_bars)

    def warmup_load_start(
        self,
        requested_start: str | pd.Timestamp,
        *,
        lookback_bars: int,
    ) -> pd.Timestamp:
        """从请求起点向前扩 lookback_bars 个 bar。"""
        lb = max(0, int(lookback_bars))
        if lb <= 0:
            return pd.Timestamp(requested_start)
        anchor = pd.Timestamp(requested_start)
        if anchor.hour == 0 and anchor.minute == 0 and anchor.second == 0:
            anchor = anchor + self.bar_timedelta * (self.session_bars - 1)
        return self.offset_bars(anchor, -lb)


# 别名：审查清单中的 SessionCalendar
SessionCalendar = SessionBarCalendar
