# -*- coding: utf-8 -*-
"""R25 T-TIME-001..005 —— availability/PIT 测试（P0-005/017）。

    T-TIME-001  next_trading_day + calendar=None + strict → raise
    T-TIME-002  US filing_date=2024-05-10 00:00 → date_label，语义日期仍 2024-05-10
    T-TIME-003  true UTC news instant → 正确时区换算
    T-TIME-004  Friday filing date-only → next real session
    T-TIME-005  calendar right boundary → hard fail
"""
from __future__ import annotations

import datetime as _dt
from datetime import date

import pytest

from data_access.core.exceptions import CalendarUnavailableError
from data_access.read.session_calendar import (
    MarketCalendar,
    build_us_session,
    compile_available_from,
    compile_available_from_result,
)


def test_ttime001_calendar_none_strict_raises():
    """P0-005：next_trading_day + calendar=None + strict → CalendarUnavailableError。"""
    k = _dt.datetime(2024, 5, 10, 0, 0)
    with pytest.raises(CalendarUnavailableError, match="日历不可用"):
        compile_available_from(k, "next_trading_day", calendar=None, strict=True)


def test_ttime001b_research_degraded_result():
    """P0-005：research 返回 degraded AvailabilityResult（authoritative=False）。"""
    k = _dt.datetime(2024, 5, 10, 0, 0)
    r = compile_available_from_result(
        k, "next_trading_day", calendar=None, strict=False
    )
    assert r.authoritative is False
    assert r.degradation_reason is not None


def test_ttime002_us_filing_date_label():
    """P0-017：filing_date 是 date_label，naive 00:00 不做 UTC→纽约时区换算。

    date_label 在 ``compile_available_from`` 里 calendar 必须存在才能到下一步；
    这里直接验证 ``_to_local_date_time`` 的 date_label 分支：date 标签不提前一天。
    """
    from data_access.read.session_calendar import _to_local_date_time

    k = _dt.datetime(2024, 5, 10, 0, 0)
    d, t = _to_local_date_time(k, "America/New_York", time_representation="date_label")
    assert d == date(2024, 5, 10)  # 不是 2024-05-09


def test_ttime003_utc_news_instant_conversion():
    """US news published_utc 是 instant：UTC→America/New_York 正确换算。"""
    from data_access.read.session_calendar import _to_local_date_time

    # 2024-05-10 20:00 UTC = 2024-05-10 16:00 NY（EDT）
    k = _dt.datetime(2024, 5, 10, 20, 0)
    d, t = _to_local_date_time(k, "America/New_York", time_representation="instant")
    assert d == date(2024, 5, 10)
    assert t == _dt.time(16, 0)


def test_ttime004_friday_date_only_next_session():
    """Friday filing date-only → next real session（下一交易日，跳过周末）。"""
    # 2024-05-10 是周五；下交易日 2024-05-13（周一）。
    calendar = MarketCalendar(
        "us",
        trading_days=[
            date(2024, 5, 9),
            date(2024, 5, 10),
            date(2024, 5, 13),
            date(2024, 5, 14),
        ],
        timezone="America/New_York",
        source="explicit",
    )
    k = _dt.datetime(2024, 5, 10, 0, 0)  # date-only filing
    out = compile_available_from(
        k, "next_trading_day", calendar=calendar, time_representation="date_label"
    )
    assert out == date(2024, 5, 13)  # 周一开盘


def test_ttime005_calendar_right_boundary_hard_fail():
    """T-TIME-005：日历右边界不可证明 → strict hard fail。"""
    calendar = MarketCalendar(
        "us",
        trading_days=[date(2024, 5, 10), date(2024, 5, 13)],
        timezone="America/New_York",
        source="explicit",
    )
    # knowledge 在最后一个交易日之后（右边界）→ next_trading_day None → strict fail
    k = _dt.datetime(2024, 5, 14, 0, 0)
    with pytest.raises(Exception, match="右边界|无法证明"):
        compile_available_from(k, "next_trading_day", calendar=calendar, strict=True)
