# -*- coding: utf-8 -*-
"""R40 #232-237: ExchangeSessionCalendar 单一权威 / calendar authoritativeness /
timezone 统一 / US early-close+DST / to_dict early-close / EarlyCloseNormalizationPolicy."""
from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from factor_engine.market.exchange_session_calendar import (
    ExchangeSessionCalendar,
    default_exchange_session_calendar,
)
from factor_engine.market.session import EarlyCloseNormalizationPolicy, SessionSpec, US_SESSION
from factor_engine.runtime.session_calendar import (
    EXCHANGE_CERTIFIED,
    WEEKDAY_APPROXIMATION,
    CalendarAuthorityError,
    SessionCalendar,
)


def test_exchange_session_calendar_sole_authority() -> None:
    """#232: ExchangeSessionCalendar 是唯一权威；SessionSpec 是其投影。"""
    cal = default_exchange_session_calendar("us")
    spec = cal.to_session_spec()
    assert spec.timezone == "America/New_York"
    assert spec.slot_count == 390
    # SessionCalendar 投影
    sc = cal.to_session_calendar()
    assert sc.timezone == "America/New_York"
    assert sc.authoritativeness == EXCHANGE_CERTIFIED
    # SessionSpec 是 immutable projection：仍能 for_date（投影行为）
    assert US_SESSION.for_date("2024-07-03").slot_count == 390  # 无 early close times 则不变


def test_offset_bars_fails_without_holiday_set_in_production() -> None:
    """#233: WEEKDAY_APPROXIMATION 在 production 下 offset_bars hard fail。"""
    approx = SessionCalendar(
        market="CN", segments=(("09:30", "11:30"), ("13:00", "15:00")),
        timezone="Asia/Shanghai", authoritativeness=WEEKDAY_APPROXIMATION,
    )
    assert approx.authoritativeness == WEEKDAY_APPROXIMATION
    # research 默认仍可偏移（向后兼容）
    approx.offset_bars("2024-01-02 09:31", 5)
    with pytest.raises(CalendarAuthorityError):
        approx.offset_bars("2024-01-02 09:31", 5, mode="production")
    with pytest.raises(CalendarAuthorityError):
        approx.warmup_load_start("2024-01-02", lookback_bars=5, mode="production")
    # EXCHANGE_CERTIFIED 放行
    cert = SessionCalendar(
        market="CN", segments=(("09:30", "11:30"), ("13:00", "15:00")),
        timezone="Asia/Shanghai", holidays=frozenset(["2024-01-01"]),
        authoritativeness=EXCHANGE_CERTIFIED,
    )
    assert cert.offset_bars("2024-01-02 09:31", 5, mode="production") is not None


def test_session_key_correct_with_non_local_input_timestamps() -> None:
    """#234: session_key/minute_ordinal 先统一到 session-local timezone。"""
    cal = SessionCalendar(
        market="CN", segments=(("09:30", "11:30"), ("13:00", "15:00")),
        timezone="Asia/Shanghai", timestamp_convention="bar_end",
    )
    # UTC 09:31 = CST 17:31（非交易时段）—— 但 session_key 是交易日归一。
    ts = pd.to_datetime(["2024-01-02 01:31:00+00:00"])  # CST 09:31
    assert str(cal.session_key(ts)[0].date()) == "2024-01-02"
    # minute_ordinal：CST 09:31 是第一根 bar（bar_end -> ordinal 0）
    ord_ = cal.minute_ordinal(pd.to_datetime(["2024-01-02 01:31:00+00:00"]))
    assert ord_[0] == 0


def test_us_early_close_session_by_date() -> None:
    """#235: US early close 日返回缩短会话。"""
    us = default_exchange_session_calendar("us")
    early = ExchangeSessionCalendar(
        market="us", exchange="NYSE", timezone="America/New_York",
        segments=(("09:30", "16:00"),),
        early_close_dates=frozenset({"2024-07-03"}),
        early_close_times={"2024-07-03": "13:00"},
        timestamp_convention="bar_start", dst_aware=True,
    )
    regular = early.for_date("2024-07-01")
    half = early.for_date("2024-07-03")
    assert regular.segments == (("09:30", "16:00"),)
    assert half.segments == (("09:30", "13:00"),)
    assert early.close_time_for("2024-07-03") == time(13, 0)
    assert early.close_time_for("2024-07-01") == time(16, 0)


def test_us_dst_session_timezone() -> None:
    """#235: US DST-aware 时区（America/New_York）。"""
    us = default_exchange_session_calendar("us")
    assert us.dst_aware is True
    assert us.timezone == "America/New_York"
    # tz-aware 输入统一到 session-local
    local = us.session_local(pd.to_datetime(["2024-07-03 13:30:00-04:00"]))
    assert local[0].hour == 13  # EDT -> 13:30 local


def test_session_spec_to_dict_includes_early_close_info() -> None:
    """#236: to_dict 包含 early_close_dates / early_close_times / digest。"""
    us = US_SESSION.with_early_close_times({"2024-07-03": time(13, 0)})
    d = us.to_dict()
    assert "2024-07-03" in d["early_close_dates"]
    assert d["early_close_times"]["2024-07-03"] == "13:00"
    assert "calendar_digest" in d and len(d["calendar_digest"]) == 16
    assert d["early_close_normalization_policy"] == "down_weight"


def test_early_close_normalization_policy_formula() -> None:
    """#237: EarlyCloseNormalizationPolicy 按 session duration ratio 的公式。"""
    assert EarlyCloseNormalizationPolicy.NONE.normalization_formula(390, 210) == 1.0
    assert EarlyCloseNormalizationPolicy.EXCLUDE.normalization_formula(390, 210) == 0.0
    assert EarlyCloseNormalizationPolicy.DOWN_WEIGHT.normalization_formula(390, 210) == pytest.approx(210 / 390)
    assert "volume" in EarlyCloseNormalizationPolicy.DOWN_WEIGHT.applicable_operator_families
    assert EarlyCloseNormalizationPolicy.from_value("down_weight") is EarlyCloseNormalizationPolicy.DOWN_WEIGHT
