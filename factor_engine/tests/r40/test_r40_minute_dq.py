# -*- coding: utf-8 -*-
"""R40 #238-243: SessionDQReport / 分层异常 / multi-date / off-grid /
market required / US-HK calendar 适配."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.microstructure.intraday_agg import (
    MarketRequiredError,
    _daily_agg,
    _declared_calendar,
)
from runtime.session_calendar import SessionCalendar
from runtime.session_panel import (
    DuplicateSlotError,
    MultipleSessionDatesError,
    build_session_panel,
)


def _ashare_cal() -> SessionCalendar:
    return SessionCalendar(
        market="CN", segments=(("09:30", "11:30"), ("13:00", "15:00")),
        timestamp_convention="bar_end", timezone="Asia/Shanghai",
    )


def _panel(values):
    idx = pd.date_range("2024-01-02 09:31:00", periods=len(values), freq="1min")
    return pd.DataFrame({"A": np.asarray(values, dtype=float)}, index=idx)


def test_duplicate_slot_hard_dq_fail_in_production() -> None:
    """#238: production 下 duplicate official slot -> hard DQ fail。"""
    ts = pd.to_datetime(["2024-01-02 09:31:00", "2024-01-02 09:31:00"])
    p = build_session_panel(ts, np.array([1.0, 2.0]), _ashare_cal(),
                            market="ashare", session_timezone="Asia/Shanghai")
    assert p.dq.has_duplicate_official_slot is True
    with pytest.raises(DuplicateSlotError):
        p.dq.hard_dq_fail()
    # 非 production 聚合：duplicate 不再静默 NaN（报告可查）
    frame = _panel([1.0, 2.0])
    frame.index = pd.to_datetime(["2024-01-02 09:31:00", "2024-01-02 09:31:00"])
    out = _daily_agg(frame, lambda panel: float(np.nanmean(panel.values)), market="ashare")
    assert out.notna().all().all()


def test_aggregator_distinguishes_dq_fail_from_numeric_nan() -> None:
    """#239: production duplicate -> DuplicateSlotError 传播（非 NaN 吞掉）。"""
    frame = _panel([1.0, 2.0])
    frame.index = pd.to_datetime(["2024-01-02 09:31:00", "2024-01-02 09:31:00"])
    with pytest.raises(DuplicateSlotError):
        _daily_agg(frame, lambda panel: float(np.nanmean(panel.values)),
                   market="ashare", mode="production")


def test_build_session_panel_rejects_multi_date_input() -> None:
    """#240: trade_date=None 且输入跨多日 -> MultipleSessionDatesError。"""
    ts = pd.to_datetime(["2024-01-02 09:31:00", "2024-01-03 09:31:00"])
    with pytest.raises(MultipleSessionDatesError):
        build_session_panel(ts, np.array([1.0, 2.0]), _ashare_cal(),
                            market="ashare", session_timezone="Asia/Shanghai")


def test_off_grid_bars_recorded_and_fail_on_threshold() -> None:
    """#241: off-grid bars 记录在 DQ report；超阈值 hard fail。"""
    ts = pd.to_datetime(["2024-01-02 12:30:00"])  # 午休 off-grid
    p = build_session_panel(ts, np.array([5.0]), _ashare_cal(),
                            market="ashare", session_timezone="Asia/Shanghai")
    assert p.dq.off_grid_count == 1
    assert p.dq.off_grid_ratio > 0.0
    # 构造全部 off-grid 的场景 -> ratio 超阈值
    ts2 = pd.to_datetime(["2024-01-02 12:30:00", "2024-01-02 12:31:00"])
    p2 = build_session_panel(ts2, np.array([5.0, 6.0]), _ashare_cal(),
                             market="ashare", session_timezone="Asia/Shanghai")
    assert p2.dq.off_grid_count == 2
    with pytest.raises(DuplicateSlotError):
        p2.dq.hard_dq_fail(off_grid_floor=0.0)


def test_minute_aggregator_fails_without_explicit_market_in_production() -> None:
    """#242: market=None -> MarketRequiredError（不再回退 ashare）。"""
    with pytest.raises(MarketRequiredError):
        _declared_calendar(None, "1min")
    frame = _panel([1.0, 2.0, 3.0])
    with pytest.raises(MarketRequiredError):
        _daily_agg(frame, lambda panel: 0.0, mode="production")  # market 未传


def test_declared_calendar_adapts_us_hk_markets() -> None:
    """#243: US -> America/New_York 09:30-16:00；HK -> Asia/Hong_Kong 双段。"""
    us = _declared_calendar("us", "1min")
    assert us.timezone == "America/New_York"
    assert us.segments == (("09:30", "16:00"),)
    assert us.authoritativeness == "EXCHANGE_CERTIFIED"
    hk = _declared_calendar("hk", "1min")
    assert hk.timezone == "Asia/Hong_Kong"
    assert hk.segments == (("09:30", "12:00"), ("13:00", "16:00"))
    with pytest.raises(ValueError):
        _declared_calendar("jp", "1min")
