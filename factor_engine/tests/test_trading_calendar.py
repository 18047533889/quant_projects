"""交易日历与增量窗口（A 股 / 美股 vs pandas bdate）测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from runtime.incremental import build_incremental_plan
from storage.time_window import business_day_offset, resolve_incremental_window
from storage.trading_calendar import (
    TradingCalendar,
    clear_trading_calendar_cache,
    get_trading_calendar,
    infer_market,
    register_trading_calendar,
    trading_day_offset,
)


def _ashare_cny_fixture() -> TradingCalendar:
    """合成 A 股日历：2024 春节前后（跳过 2/9–2/18 休市）。"""
    days = [
        "2024-02-05",
        "2024-02-06",
        "2024-02-07",
        "2024-02-08",
        "2024-02-19",
        "2024-02-20",
        "2024-02-21",
    ]
    return TradingCalendar(days)


def test_trading_calendar_offset_forward_backward():
    cal = _ashare_cny_fixture()
    assert cal.offset("2024-02-08", 1) == pd.Timestamp("2024-02-19")
    assert cal.offset("2024-02-19", -1) == pd.Timestamp("2024-02-08")
    assert cal.offset("2024-02-08", 0) == pd.Timestamp("2024-02-08")


def test_trading_day_offset_differs_from_pandas_bdate_around_cny():
    cal = _ashare_cny_fixture()
    # pandas 会把 2/9 当作下一个工作日；A 股日历应跳到 2/19
    assert trading_day_offset("2024-02-08", 1, calendar=cal) == pd.Timestamp("2024-02-19")
    assert business_day_offset("2024-02-08", 1) == pd.Timestamp("2024-02-09")


def test_resolve_incremental_window_uses_calendar():
    cal = _ashare_cny_fixture()
    w = resolve_incremental_window(
        watermark_end="2024-02-20",
        lookback_bars=2,
        recompute_tail_bars=1,
        calendar=cal,
    )
    assert w["output_start"] == pd.Timestamp("2024-02-19")
    assert w["load_start"] == pd.Timestamp("2024-02-07")


def test_build_incremental_plan_with_synthetic_calendar():
    cal = _ashare_cny_fixture()
    plan = build_incremental_plan(
        factor_id="mom_v1",
        analysis_lookback=1,
        watermark={"end_date": "2024-02-20"},
        lookback_extra=0,
        recompute_tail_bars=1,
        calendar=cal,
    )
    assert plan.is_full_run is False
    assert plan.output_start == pd.Timestamp("2024-02-19")
    assert plan.load_start == pd.Timestamp("2024-02-07")


@pytest.mark.parametrize(
    ("universe", "dataset", "expected"),
    [
        ("ASHARE_DAILY", None, "ashare"),
        ("US_MASSIVE", None, "us"),
        (None, "ashare_daily", "ashare"),
        (None, "us_stock_daily", "us"),
        (None, "other", None),
    ],
)
def test_infer_market(universe, dataset, expected):
    assert infer_market(universe=universe, dataset=dataset) == expected


def test_register_trading_calendar_cache():
    clear_trading_calendar_cache()
    cal = _ashare_cny_fixture()
    register_trading_calendar("ashare", cal)
    assert get_trading_calendar("ashare") is cal
    clear_trading_calendar_cache()
    assert get_trading_calendar("ashare") is None or get_trading_calendar("ashare") is not cal


def test_trading_calendar_clamps_out_of_range():
    cal = TradingCalendar(["2024-01-02", "2024-01-03"])
    assert cal.offset("2024-01-02", -5) == pd.Timestamp("2024-01-02")
    assert cal.offset("2024-01-03", 10) == pd.Timestamp("2024-01-03")
