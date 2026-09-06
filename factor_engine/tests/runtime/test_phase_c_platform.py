"""Phase C：Numba 扩面、Polars lazy、materialize_sharded 扩展。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from factor_engine.api import ts_mean, ts_std
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.backend.polars_backend import PolarsBackend
from factor_engine.backend.routing import numba_enabled_for_op
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.shard_materialize import (
    month_date_bounds,
    month_keys_between,
    shard_bucket_values,
    shard_time_months,
)
from tests.helpers import InMemorySeriesSource


def _panel(n: int = 30) -> dict:
    dates = pd.bdate_range("2024-01-02", periods=n)
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=len(idx)), index=idx)
    return {"close": s}


def test_numba_enabled_for_ts_mean_with_env(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_USE_NUMBA", "1")
    assert numba_enabled_for_op("ts_mean", window=20) is True
    monkeypatch.delenv("FACTOR_ENGINE_USE_NUMBA", raising=False)
    assert numba_enabled_for_op("ts_mean", window=20) is False


def test_ts_mean_numba_matches_pandas(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_USE_NUMBA", "1")
    data = _panel(40)
    panel = data["close"].unstack(level="instrument")
    from factor_engine.cleaned_operators.common.time_series import TSMean

    op = TSMean()
    expected = panel.rolling(window=5, min_periods=1).mean()
    got = op._calculate_series(panel, window=5)
    pd.testing.assert_frame_equal(got, expected, rtol=1e-4, atol=1e-4)


def test_ts_std_numba_matches_pandas(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_USE_NUMBA", "1")
    data = _panel(40)
    panel = data["close"].unstack(level="instrument")
    eng = FactorEngine(backend=PandasBackend(), data_source=InMemorySeriesSource(data=data))
    f = Factor(name="std_test", expr=ts_std(col("close"), 5))
    out = eng.run(f)
    direct = panel.rolling(window=5, min_periods=1).std()
    got_panel = out["result"].unstack(level="instrument")
    pd.testing.assert_frame_equal(got_panel, direct, rtol=1e-4, atol=1e-4)


def test_polars_backend_execute_lazy_sets_flag():
    from factor_engine.backend.context import ExecutionContext

    backend = PolarsBackend(use_lazy=True)
    assert backend.use_lazy is True
    src = MagicMock()
    src.enable_lazy_scan = MagicMock()
    ctx = ExecutionContext(data_source=src, panel_cache={})
    with patch.object(PandasBackend, "_eval", return_value=pd.Series([1.0])):
        backend.execute(MagicMock(), ctx)
    src.enable_lazy_scan.assert_called_once_with(True)


def test_data_access_source_lazy_scan_path():
    from factor_engine.storage.data_access_source import DataAccessSource

    src = DataAccessSource(dataset="ds", read_auto=True)
    src.enable_lazy_scan(True)
    assert src.lazy_scan is True
    assert src.read_auto is True


def test_shard_bucket_values():
    b0 = shard_bucket_values(8, shard_index=0, shard_count=2)
    b1 = shard_bucket_values(8, shard_index=1, shard_count=2)
    assert sorted(b0 + b1) == list(range(8))
    assert not set(b0) & set(b1)


def test_shard_time_months():
    months = month_keys_between("2024-01-15", "2024-03-10")
    assert months == ["2024-01", "2024-02", "2024-03"]
    start, end = month_date_bounds(["2024-01", "2024-02"])
    assert start == "2024-01-01"
    assert end == "2024-02-29"
    m0 = shard_time_months(months, shard_index=0, shard_count=2)
    m1 = shard_time_months(months, shard_index=1, shard_count=2)
    assert sorted(m0 + m1) == months


def test_materialize_sharded_time_month_scopes_dates(tmp_path, monkeypatch):
    data = _panel(10)
    src = InMemorySeriesSource(data=data)
    eng = FactorEngine(backend=PandasBackend(), data_source=src)
    factors = [Factor(name="f1", expr=ts_mean(col("close"), 2))]

    written: list[dict] = []

    def _fake_execute(engine, factor, output, **kwargs):
        written.append({"start": getattr(engine.data_source, "start_date", None)})
        return {"materialization": {"rows_written": 1}}

    monkeypatch.setattr(
        "factor_engine.runtime.materialize_service.execute_materialize",
        _fake_execute,
    )
    out = eng.materialize_sharded(
        factors,
        factor_ids=["fid1"],
        shard_by="time_month",
        shard_index=0,
        shard_count=1,
        time_months=["2024-01"],
        lake_root=tmp_path / "lake",
        run_many_batch=False,
    )
    assert out["shard_by"] == "time_month"
    assert out["time_months"] == ["2024-01"]
    assert written and written[0]["start"] == "2024-01-01"


def test_scan_dataset_columns_mock():
    from factor_engine.backend.polars_lazy import scan_dataset_columns

    import pyarrow as pa

    table = pa.table(
        {
            "dt": ["2024-01-02", "2024-01-03"],
            "sym": ["A", "A"],
            "close": [1.0, 2.0],
        }
    )
    import polars as pl

    mock_lf = pl.from_arrow(table).lazy()
    store = MagicMock()
    store.scan = None  # 走 scan_polars 回退路径
    store.scan_polars.return_value = mock_lf
    out = scan_dataset_columns(
        store,
        "ds",
        physical_columns=["close"],
        time_column="dt",
        instrument_column="sym",
        time_range=None,
        instrument_filter=None,
        output_names=None,
        normalize_timestamp=False,
        timestamp_unit=None,
        params={},
    )
    assert "close" in out
    store.scan_polars.assert_called_once()
