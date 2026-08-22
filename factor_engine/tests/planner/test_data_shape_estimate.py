# -*- coding: utf-8 -*-
"""§27-29: DataShapeEstimate tests —— metadata-only shape 推断（不加载大数据也能规划）。"""

import pandas as pd
import pytest

from planner.data_shape_estimate import (
    DataShapeEstimate,
    estimate_dates_from_calendar,
    estimate_instruments_from_universe,
    estimate_shape_from_metadata,
    shape_to_cost_context,
)
from storage.trading_calendar import TradingCalendar


class TestEstimateInstrumentsFromUniverse:
    """§28: 去掉固定 3000 instruments —— 按 universe 名称推断仪器数（保守先验）。"""

    def test_universe_prior_csi300(self):
        """CSI300 匹配 300 instruments。"""
        assert estimate_instruments_from_universe("CSI300") == 300
        assert estimate_instruments_from_universe("csi300") == 300
        assert estimate_instruments_from_universe("CSI-300") == 300

    def test_universe_prior_csi500(self):
        """CSI500 匹配 500 instruments。"""
        assert estimate_instruments_from_universe("CSI500") == 500
        assert estimate_instruments_from_universe("ZZ500") == 500

    def test_universe_prior_all_a(self):
        """ALL_A 匹配 5500 instruments。"""
        assert estimate_instruments_from_universe("ALL_A") == 5500
        assert estimate_instruments_from_universe("ASHARE_ALL") == 5500
        assert estimate_instruments_from_universe("ALL_ASHARE") == 5500

    def test_universe_prior_us(self):
        """US universe priors。"""
        assert estimate_instruments_from_universe("SP500") == 500
        assert estimate_instruments_from_universe("NASDAQ") == 3000
        assert estimate_instruments_from_universe("US_ALL") == 8000

    def test_universe_metadata_overrides_prior(self):
        """真实 metadata.instrument_count 覆盖 prior。"""

        class MockMetadata:
            instrument_count = 1234

        assert estimate_instruments_from_universe(
            "CSI300", universe_metadata=MockMetadata()
        ) == 1234

    def test_unknown_universe_returns_default(self):
        """未知 universe 返回 default（3000）。"""
        assert estimate_instruments_from_universe("UNKNOWN_UNIVERSE") == 3000
        assert estimate_instruments_from_universe(None) == 3000

    def test_custom_default(self):
        """自定义 default。"""
        assert estimate_instruments_from_universe("UNKNOWN", default=5000) == 5000


class TestEstimateDatesFromCalendar:
    """§29: 日期数优先真实交易日历 —— calendar.session_count；fallback 才普通工作日估计。"""

    def test_real_calendar_session_count(self):
        """真实 TradingCalendar session_count。"""
        days = pd.date_range("2020-01-01", "2020-12-31", freq="B")
        cal = TradingCalendar(days.tolist())
        count = estimate_dates_from_calendar(
            "2020-01-01", "2020-12-31", calendar=cal
        )
        assert count == len(days)

    def test_calendar_partial_range(self):
        """日历部分范围。"""
        days = pd.date_range("2020-01-01", "2020-12-31", freq="B")
        cal = TradingCalendar(days.tolist())
        count = estimate_dates_from_calendar(
            "2020-06-01", "2020-06-30", calendar=cal
        )
        expected = len(pd.date_range("2020-06-01", "2020-06-30", freq="B"))
        assert count == expected

    def test_bdate_range_fallback(self):
        """无 calendar 时 fallback 到 bdate_range。"""
        count = estimate_dates_from_calendar(
            "2020-01-01", "2020-12-31", allow_approximate_calendar=True
        )
        expected = len(pd.date_range("2020-01-01", "2020-12-31", freq="B"))
        assert count == expected

    def test_no_calendar_fail_closed(self):
        """production 模式下无 calendar 且 allow_approximate_calendar=False 返回 0。"""
        count = estimate_dates_from_calendar(
            "2020-01-01", "2020-12-31", allow_approximate_calendar=False
        )
        assert count == 0

    def test_invalid_date_range(self):
        """start > end 返回 0。"""
        count = estimate_dates_from_calendar(
            "2020-12-31", "2020-01-01", allow_approximate_calendar=True
        )
        assert count == 0

    def test_none_dates(self):
        """None dates 返回 0。"""
        assert estimate_dates_from_calendar(None, None) == 0
        assert estimate_dates_from_calendar("2020-01-01", None) == 0


class TestEstimateShapeFromMetadata:
    """§27: metadata-only shape 估算（不加载大数据也能规划）。"""

    def test_parquet_metadata_num_rows(self):
        """Parquet metadata num_rows。"""

        class MockParquetMetadata:
            num_rows = 100_000
            total_byte_size = 10_000_000

        shape = estimate_shape_from_metadata(
            universe="CSI300",
            start_date="2020-01-01",
            end_date="2020-12-31",
            columns=["close", "volume"],
            parquet_metadata=MockParquetMetadata(),
            allow_approximate_calendar=True,
        )
        assert shape.estimated_rows == 100_000
        assert shape.estimated_bytes == 10_000_000
        assert shape.estimated_instruments == 300
        assert shape.estimated_columns == 2
        assert shape.average_row_width_bytes == 100.0

    def test_data_access_manifest(self):
        """DataAccess manifest selected_bytes。"""

        class MockManifest:
            selected_bytes = 5_000_000
            instrument_count = 500

        shape = estimate_shape_from_metadata(
            start_date="2020-01-01",
            end_date="2020-12-31",
            columns=["close", "volume", "open", "high", "low"],
            data_access_manifest=MockManifest(),
            average_row_width_bytes=200.0,
            allow_approximate_calendar=True,
        )
        assert shape.estimated_bytes == 5_000_000
        assert shape.estimated_rows == 25_000
        assert shape.estimated_instruments == 500
        assert shape.estimated_columns == 5

    def test_default_shape_inference(self):
        """无 metadata 时推断默认 shape（date × instrument）。"""
        shape = estimate_shape_from_metadata(
            universe="CSI500",
            start_date="2020-01-01",
            end_date="2020-12-31",
            columns=["close", "volume"],
            allow_approximate_calendar=True,
        )
        # 估算日期数 ≈ 252 工作日
        assert shape.estimated_dates > 200
        assert shape.estimated_instruments == 500
        assert shape.estimated_rows == shape.estimated_dates * 500
        assert shape.estimated_columns == 2

    def test_minute_frequency_bars_per_session(self):
        """分钟频率 × bars_per_session。"""
        shape = estimate_shape_from_metadata(
            universe="CSI300",
            start_date="2020-01-01",
            end_date="2020-12-31",
            frequency="minute",
            bars_per_session=240,
            allow_approximate_calendar=True,
        )
        assert shape.frequency == "minute"
        assert shape.bars_per_session == 240
        # rows = dates × instruments × bars_per_session
        expected_rows = shape.estimated_dates * 300 * 240
        assert shape.estimated_rows == expected_rows

    def test_remote_storage_kind(self):
        """remote / storage_kind 元数据。"""
        shape = estimate_shape_from_metadata(
            universe="CSI300",
            start_date="2020-01-01",
            end_date="2020-12-31",
            remote=True,
            storage_kind="cos",
            allow_approximate_calendar=True,
        )
        assert shape.remote is True
        assert shape.storage_kind == "cos"

    def test_density_and_average_row_width(self):
        """density / average_row_width_bytes 自定义。"""
        shape = estimate_shape_from_metadata(
            universe="CSI300",
            start_date="2020-01-01",
            end_date="2020-12-31",
            density=0.85,
            average_row_width_bytes=512.0,
            allow_approximate_calendar=True,
        )
        assert shape.density == 0.85
        assert shape.average_row_width_bytes == 512.0

    def test_to_dict_serialization(self):
        """to_dict 序列化。"""
        shape = estimate_shape_from_metadata(
            universe="CSI300",
            start_date="2020-01-01",
            end_date="2020-12-31",
            columns=["close", "volume"],
            allow_approximate_calendar=True,
        )
        d = shape.to_dict()
        assert isinstance(d, dict)
        assert d["estimated_instruments"] == 300
        assert d["frequency"] == "daily"
        assert isinstance(d["estimated_rows"], int)


class TestShapeToCostContext:
    """shape_to_cost_context 转换为 CostContext。"""

    def test_shape_to_cost_context(self):
        """DataShapeEstimate → CostContext 字典。"""
        shape = DataShapeEstimate(
            estimated_rows=100_000,
            estimated_dates=252,
            estimated_instruments=500,
            estimated_columns=10,
            estimated_bytes=10_000_000,
            average_row_width_bytes=100.0,
            density=0.95,
            frequency="daily",
            bars_per_session=240,
            group_count=10,
        )
        ctx = shape_to_cost_context(shape)
        assert ctx["rows"] == 100_000
        assert ctx["instruments"] == 500
        assert ctx["expected_density"] == 0.95
        assert ctx["session_bars"] == 240
        assert ctx["group_count"] == 10


class TestDataShapeEstimateIntegration:
    """§27-29 集成测试 —— metadata-only 不加载大数据也能规划。"""

    def test_csi300_daily_one_year(self):
        """CSI300 日频 1 年。"""
        shape = estimate_shape_from_metadata(
            universe="CSI300",
            start_date="2020-01-01",
            end_date="2020-12-31",
            frequency="daily",
            allow_approximate_calendar=True,
        )
        assert shape.estimated_instruments == 300
        assert 200 <= shape.estimated_dates <= 260
        assert shape.estimated_rows == shape.estimated_dates * 300

    def test_all_a_minute_one_month(self):
        """ALL_A 分钟频 1 月。"""
        shape = estimate_shape_from_metadata(
            universe="ALL_A",
            start_date="2020-01-01",
            end_date="2020-01-31",
            frequency="minute",
            bars_per_session=240,
            allow_approximate_calendar=True,
        )
        assert shape.estimated_instruments == 5500
        assert shape.frequency == "minute"
        # 1 月约 20 交易日
        assert 15 <= shape.estimated_dates <= 25
        expected_rows = shape.estimated_dates * 5500 * 240
        assert shape.estimated_rows == expected_rows

    def test_us_sp500_daily_five_years(self):
        """US SP500 日频 5 年。"""
        shape = estimate_shape_from_metadata(
            universe="SP500",
            start_date="2016-01-01",
            end_date="2020-12-31",
            frequency="daily",
            market="us",
            allow_approximate_calendar=True,
        )
        assert shape.estimated_instruments == 500
        # 5 年约 1260 工作日
        assert 1200 <= shape.estimated_dates <= 1300
        assert shape.estimated_rows == shape.estimated_dates * 500

    def test_no_dates_fallback_252_days(self):
        """无日期范围时 fallback 到 252 交易日（1 年）。"""
        shape = estimate_shape_from_metadata(
            universe="CSI300",
            frequency="daily",
        )
        assert shape.estimated_dates == 252
        assert shape.estimated_rows == 252 * 300
