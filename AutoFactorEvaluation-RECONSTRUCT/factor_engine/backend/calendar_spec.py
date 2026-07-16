# -*- coding: utf-8
"""交易日历与时区契约（日期类算子不得由 emitter 自行猜测）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SessionCutoff = Literal["market_close", "market_open", "utc_midnight"]
DateSemantics = Literal["exchange_local_date", "utc_date"]


@dataclass(frozen=True)
class CalendarSpec:
    """日频第一阶段默认美股日历；微观结构/session 算子须显式引用。"""

    exchange: str = "XNYS"
    timezone: str = "America/New_York"
    session_cutoff: SessionCutoff = "market_close"
    date_semantics: DateSemantics = "exchange_local_date"
    early_close_policy: str = "exchange_calendar"
    holiday_source: str = "exchange_calendar"
    after_hours_announcement: Literal["same_trading_day", "next_trading_day"] = "next_trading_day"


DEFAULT_CALENDAR = CalendarSpec()

# 未来日期类算子须绑定 CalendarSpec
CALENDAR_BOUND_OPS: frozenset[str] = frozenset(
    """
    day_of_week month quarter year days_since business_day_count
    session_id is_month_end is_quarter_end
    """.split()
)


def calendar_spec_for(*, exchange: str | None = None) -> CalendarSpec:
    if exchange and exchange != DEFAULT_CALENDAR.exchange:
        return CalendarSpec(exchange=exchange)
    return DEFAULT_CALENDAR


def calendar_ops_require_explicit_spec() -> bool:
    return True
