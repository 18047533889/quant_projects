# -*- coding: utf-8
"""RunWindow 与全量 auto_warmup 扩窗单元测试。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from factor_engine.api import ts_mean
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.cleaned_operators.operator_policy import effective_lookback
from factor_engine.runtime.run_window import (
    build_full_run_window,
    extract_source_date_bounds,
)
from factor_engine.storage.time_window import business_day_offset, slice_series_time_window
from tests.helpers import InMemorySeriesSource


def test_build_full_run_window_expands_start():
    w = build_full_run_window(
        requested_start="2024-01-08",
        requested_end="2024-01-15",
        lookback_bars=3,
    )
    assert w.requested_start == "2024-01-08"
    assert w.actual_load_start == business_day_offset("2024-01-08", -3).strftime("%Y-%m-%d")
    assert w.warmup_bars == 3
    assert w.trim_output is True


def test_build_full_run_window_no_start_no_expand():
    w = build_full_run_window(
        requested_start=None,
        requested_end="2024-01-15",
        lookback_bars=5,
    )
    assert w.actual_load_start is None
    assert w.warmup_bars == 0


def test_build_full_run_window_zero_lookback():
    w = build_full_run_window(
        requested_start="2024-01-08",
        requested_end="2024-01-15",
        lookback_bars=0,
    )
    assert w.actual_load_start == "2024-01-08"
    assert w.warmup_bars == 0


def test_extract_source_date_bounds_from_attributes():
    @dataclass
    class _Src:
        start_date: str = "2024-01-02"
        end_date: str = "2024-03-31"

    start, end = extract_source_date_bounds(_Src())
    assert start == "2024-01-02"
    assert end == "2024-03-31"


def test_extract_source_date_bounds_from_time_range():
    class _Src:
        def time_range(self):
            return "2024-02-01", "2024-02-29"

    start, end = extract_source_date_bounds(_Src())
    assert start == "2024-02-01"
    assert end == "2024-02-29"


@dataclass
class _BoundedInMemorySource(InMemorySeriesSource):
    """带日期边界的内存数据源，供 auto_warmup 测试读取请求窗口。"""

    start_date: str | None = None
    end_date: str | None = None

    def time_range(self):
        return self.start_date, self.end_date


def _close_panel(dates: list[str], assets: list[str] | None = None) -> pd.Series:
    assets = assets or ["AAA"]
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), assets],
        names=["timestamp", "instrument"],
    )
    return pd.Series([float(i + 1) for i in range(len(idx))], index=idx)


def test_auto_warmup_trims_to_requested_window():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    close = _close_panel(dates)
    source = _BoundedInMemorySource(
        data={"close": close},
        start_date="2024-01-04",
        end_date="2024-01-08",
    )
    factor = Factor(name="mean2", expr=ts_mean(col("close"), 2))
    engine = FactorEngine(backend=PandasBackend(), data_source=source)

    out = engine.run(factor, auto_warmup=True, trim_warmup=True)
    rw = out["run_window"]
    lb = effective_lookback(out["analysis"].lookback)
    assert rw is not None
    assert rw["requested_start"] == "2024-01-04"
    assert rw["actual_load_start"] == business_day_offset("2024-01-04", -lb).strftime("%Y-%m-%d")
    assert rw["warmup_bars"] == lb

    trimmed = slice_series_time_window(
        out["result"],
        start=pd.Timestamp("2024-01-04"),
        end=pd.Timestamp("2024-01-08"),
    )
    pd.testing.assert_series_equal(out["result"], trimmed, check_names=False)
    assert out["result"].notna().all()


def test_auto_warmup_off_does_not_attach_run_window():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    close = _close_panel(dates)
    source = _BoundedInMemorySource(
        data={"close": close},
        start_date="2024-01-04",
        end_date="2024-01-05",
    )
    factor = Factor(name="mean2", expr=ts_mean(col("close"), 2))
    engine = FactorEngine(backend=PandasBackend(), data_source=source)

    out_no_warmup = engine.run(factor, auto_warmup=False)
    out_warmup = engine.run(factor, auto_warmup=True, trim_warmup=True)
    assert "run_window" not in out_no_warmup
    assert "run_window" in out_warmup
    # 无 warmup 时内存源仍返回全量；有 warmup+trim 时裁剪到请求窗口
    assert len(out_no_warmup["result"]) == len(close)
    assert len(out_warmup["result"]) == 2
