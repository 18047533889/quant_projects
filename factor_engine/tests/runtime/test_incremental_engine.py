"""增量引擎端到端测试：lookback 边界 + 落盘 upsert。"""

from __future__ import annotations

import pandas as pd
import pytest

from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from storage.materializer import ParquetMaterializer
from tests.helpers import InMemorySeriesSource


def _build_close_panel(n_days: int = 10) -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["timestamp", "instrument"])
    values = [float(i + 1) for i in range(len(idx))]
    return pd.Series(values, index=idx)


def _mom_factor() -> Factor:
    return Factor(name="mom_2_rank", expr=rank(ts_mean(col("close"), 2)))


def test_run_incremental_full_run_without_watermark():
    close = _build_close_panel()
    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    )
    out = engine.run_incremental(_mom_factor(), factor_id="mom_test")
    assert out["incremental"]["is_full_run"] is True
    assert len(out["result"]) == len(close)


def test_run_incremental_matches_full_at_tail(tmp_path):
    close = _build_close_panel()
    source = InMemorySeriesSource(data={"close": close})
    engine = FactorEngine(backend=PandasBackend(), data_source=source)
    factor = _mom_factor()
    fid = "mom_incr"

    full = engine.run(factor)["result"]
    lake = tmp_path / "lake"
    mat = ParquetMaterializer(lake_root=lake)
    mat.materialize(factor_id=fid, result=full, ast_hash="h1")

    inc = engine.run_incremental(factor, factor_id=fid, lake_root=lake, lookback_extra=0)
    assert inc["incremental"]["is_full_run"] is False

    # 增量切片应覆盖 tail；与全量 tail 数值一致
    overlap_idx = inc["result"].index.intersection(full.index)
    assert len(overlap_idx) > 0
    pd.testing.assert_series_equal(
        inc["result"].loc[overlap_idx],
        full.loc[overlap_idx],
        check_names=False,
    )


def test_materialize_incremental_appends_new_dates(tmp_path):
    dates_old = pd.bdate_range("2024-01-02", periods=5)
    dates_all = pd.bdate_range("2024-01-02", periods=7)
    idx_old = pd.MultiIndex.from_product([dates_old, ["AAA"]], names=["timestamp", "instrument"])
    idx_all = pd.MultiIndex.from_product([dates_all, ["AAA"]], names=["timestamp", "instrument"])
    close_old = pd.Series([float(i) for i in range(len(idx_old))], index=idx_old)
    close_all = pd.Series([float(i) for i in range(len(idx_all))], index=idx_all)

    factor = Factor(name="delay1", expr=col("close"))
    fid = "delay1_v1"
    lake = tmp_path / "lake"

    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close_old}),
    )
    engine.materialize(factor, factor_id=fid, lake_root=lake, expression='col("close")')

    engine2 = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close_all}),
    )
    out = engine2.materialize_incremental(
        factor,
        factor_id=fid,
        lake_root=lake,
        lookback_extra=0,
        recompute_tail_bars=0,
        expression='col("close")',
    )
    assert out["materialization"]["rows_written"] >= 2

    pq = lake / "factors" / fid / "year=2024" / "data.parquet"
    df = pd.read_parquet(pq)
    assert len(df) == 7


def test_incremental_lookback_window_produces_correct_rolling_mean():
    """窗口加载 + tail 重算：ts_mean(2) 在边界日与全量一致。"""
    dates = pd.bdate_range("2024-01-02", periods=8)
    idx = pd.MultiIndex.from_product([dates, ["X"]], names=["timestamp", "instrument"])
    close = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0], index=idx)

    factor = Factor(name="mean2", expr=ts_mean(col("close"), 2))
    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    )
    full = engine.run(factor)["result"]

    # 模拟 watermark 停在倒数第 3 个交易日
    wm_date = dates[-3]
    inc = engine.run_incremental(
        factor,
        factor_id="mean2_v1",
        since=str(wm_date.date()),
        lookback_extra=0,
        recompute_tail_bars=2,
    )
    overlap = inc["result"].index.intersection(full.index)
    pd.testing.assert_series_equal(
        inc["result"].loc[overlap],
        full.loc[overlap],
        check_names=False,
    )


def test_run_incremental_infers_market_from_data_source():
    close = _build_close_panel(5)
    source = InMemorySeriesSource(data={"close": close})
    source.dataset = "us_stock_daily"  # type: ignore[attr-defined]
    factor = Factor(name="delay1", expr=col("close"), universe="ASHARE_DAILY")
    engine = FactorEngine(backend=PandasBackend(), data_source=source)
    out = engine.run_incremental(
        factor,
        factor_id="delay1",
        since="2024-01-05",
        lookback_extra=0,
        recompute_tail_bars=0,
    )
    # universe 优先于 dataset
    assert out["incremental"]["market"] == "ashare"
