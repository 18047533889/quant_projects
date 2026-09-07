"""时间窗口与增量计划单元测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.runtime.incremental import build_incremental_plan, slice_factor_result_for_incremental
from factor_engine.storage.time_window import (
    WindowedDataSource,
    business_day_offset,
    resolve_incremental_window,
    slice_series_time_window,
)
from tests.helpers import InMemorySeriesSource


def _panel(values: list[float], dates: list[str] | None = None) -> pd.Series:
    dates = dates or ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), ["A"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series(values, index=idx)


@pytest.mark.parametrize('zone', [None, 'Asia/Shanghai'])
def test_windowed_lazy_scan_matches_pandas_boundaries(zone):
    pl = pytest.importorskip('polars')
    dates = pd.date_range('2024-01-01', periods=5, tz=zone)
    values = pd.Series(range(5), index=pd.MultiIndex.from_product([dates, ['A']], names=['timestamp', 'instrument']))
    class Source:
        def load_column(self, name):return values
        def scan_polars_long(self, columns):
            return pl.from_pandas(values.rename('close').reset_index()).lazy()
    wrapped = WindowedDataSource(Source(), start_date='2024-01-02', end_date='2024-01-04')
    lazy = wrapped.scan_polars_long(['close'])
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect()['close'].to_list() == wrapped.load_column('close').tolist() == [1,2,3]


def test_windowed_lazy_scan_refuses_ambiguous_axis():
    pl = pytest.importorskip('polars')
    class Source:
        def scan_polars_long(self, columns):
            return pl.DataFrame({'a':[pd.Timestamp('2024-01-01')], 'b':[pd.Timestamp('2024-01-01')], 'close':[1]}).lazy()
    with pytest.raises(NotImplementedError, match='unambiguous'):
        WindowedDataSource(Source(), start_date='2024-01-02').scan_polars_long(['close'])


def test_business_day_offset_forward_and_backward():
    assert business_day_offset("2024-01-05", 1) == pd.Timestamp("2024-01-08")
    assert business_day_offset("2024-01-08", -1) == pd.Timestamp("2024-01-05")


def test_resolve_incremental_window_no_watermark_is_full():
    w = resolve_incremental_window(watermark_end=None, lookback_bars=5)
    assert w["load_start"] is None
    assert w["output_start"] is None


def test_resolve_incremental_window_with_watermark():
    w = resolve_incremental_window(
        watermark_end="2024-01-10",
        lookback_bars=3,
        recompute_tail_bars=2,
    )
    assert w["output_start"] == business_day_offset("2024-01-10", -2)
    assert w["load_start"] == business_day_offset(w["output_start"], -3)


def test_slice_series_time_window():
    s = _panel([1.0, 2.0, 3.0, 4.0, 5.0])
    out = slice_series_time_window(s, start=pd.Timestamp("2024-01-04"), end=pd.Timestamp("2024-01-08"))
    assert len(out) == 3
    assert out.iloc[-1] == pytest.approx(5.0)


def test_windowed_data_source_slices_columns():
    close = _panel([10.0, 11.0, 12.0, 13.0, 14.0])
    inner = InMemorySeriesSource(data={"close": close})
    wrapped = WindowedDataSource(
        inner,
        start_date="2024-01-04",
        end_date="2024-01-08",
    )
    sliced = wrapped.load_column("close")
    assert len(sliced) == 3
    assert sliced.iloc[0] == pytest.approx(12.0)


def test_build_incremental_plan_from_watermark():
    plan = build_incremental_plan(
        factor_id="mom_v1",
        analysis_lookback=2,
        watermark={"end_date": "2024-01-08", "start_date": "2024-01-02"},
        lookback_extra=0,
        recompute_tail_bars=1,
    )
    assert plan.is_full_run is False
    assert plan.output_start == business_day_offset("2024-01-08", -1)
    assert plan.lookback_bars == 3  # 2 + 1 lag_buffer + 0 extra


def test_slice_factor_result_for_incremental():
    s = _panel([1.0, 2.0, 3.0, 4.0, 5.0])
    plan = build_incremental_plan(
        factor_id="x",
        analysis_lookback=0,
        watermark={"end_date": "2024-01-05"},
        lookback_extra=0,
        recompute_tail_bars=0,
    )
    out = slice_factor_result_for_incremental(s, plan)
    assert out.index.get_level_values(0).min() >= pd.Timestamp("2024-01-05")
