"""增量 tail 默认、IO 下推、缓存隔离测试。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.incremental import build_incremental_plan
from factor_engine.storage.cache import CacheManager
from factor_engine.storage.kline_parquet_source import KlineParquetSource
from factor_engine.storage.time_window import narrow_data_source_for_window
from tests.helpers import InMemorySeriesSource


def test_build_incremental_plan_default_tail_uses_analysis_lookback_not_load_buffer():
    plan = build_incremental_plan(
        factor_id="x",
        analysis_lookback=20,
        watermark={"end_date": "2024-02-01"},
        lookback_extra=5,
    )
    # load = 20+1+5=26; tail 默认 = 21（非 26）
    assert plan.lookback_bars == 26
    assert plan.output_start == pd.Timestamp("2024-01-03")  # 21 bdays back from 2024-02-01


def test_build_incremental_plan_empty_watermark_end_is_full_run():
    plan = build_incremental_plan(
        factor_id="x",
        analysis_lookback=2,
        watermark={"end_date": "", "start_date": "2024-01-01"},
    )
    assert plan.is_full_run is True


def test_narrow_data_source_pushes_dates_to_kline_source():
    src = KlineParquetSource(root="/tmp", start_date="2024-01-01", end_date="2024-12-31")
    narrowed = narrow_data_source_for_window(
        src,
        start_date="2024-06-01",
        end_date="2024-06-30",
    )
    assert isinstance(narrowed, KlineParquetSource)
    assert narrowed.start_date == "2024-06-01"
    assert narrowed.end_date == "2024-06-30"


@dataclass
class CountingSource:
    data: dict[str, pd.Series]
    calls: dict[str, int] = field(default_factory=dict)

    def load_column(self, name: str):
        self.calls[name] = self.calls.get(name, 0) + 1
        return self.data[name]


def test_incremental_run_ignores_stale_subplan_cache():
    """全量 cache 开启后，增量必须用 fresh_cache 重算，不能命中旧子计划。"""
    dates = pd.bdate_range("2024-01-02", periods=6)
    idx = pd.MultiIndex.from_product([dates, ["X"]], names=["timestamp", "instrument"])

    close_full = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], index=idx)
    close_window = close_full.copy()
    close_window.loc[(dates[-1], "X")] = 999.0  # 仅最后一天不同

    factor = Factor(name="last_close", expr=col("close"))
    cache = CacheManager()
    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close_full}),
        cache=cache,
    )
    full_last = engine.run(factor)["result"].loc[(dates[-1], "X")]
    assert full_last == pytest.approx(100.0)

    engine2 = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close_window}),
        cache=cache,
    )
    inc = engine2.run_incremental(
        factor,
        factor_id="close_v1",
        since=str(dates[-2].date()),
        lookback_extra=0,
        recompute_tail_bars=1,
    )
    inc_last = inc["result"].loc[(dates[-1], "X")]

    assert inc_last == pytest.approx(999.0)
    assert inc_last != pytest.approx(100.0)


def test_slice_series_time_window_respects_utc_index():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True), ["A"]],
        names=["timestamp", "instrument"],
    )
    s = pd.Series([1.0, 2.0], index=idx)
    from factor_engine.storage.time_window import slice_series_time_window

    out = slice_series_time_window(s, start=pd.Timestamp("2024-01-03"))
    assert len(out) == 1
    assert out.iloc[0] == pytest.approx(2.0)
