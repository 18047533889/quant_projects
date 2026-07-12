# -*- coding: utf-8
"""DQ preserve_invalid_rows 与 ClickHouse 企业路径测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime.dq_gates import assert_factor_dq, evaluate_factor_dq


def _series(values):
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02"]), ["A", "B", "C"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series(values, index=idx)


def test_preserve_invalid_rows_skips_inf_gate():
    s = _series([1.0, float("inf"), float("nan")])
    report = evaluate_factor_dq(s, preserve_invalid_rows=True)
    inf_check = next(c for c in report.checks if c.name == "inf_ratio")
    assert inf_check.passed is True


def test_strict_dq_fails_on_inf_without_preserve():
    s = _series([1.0, float("inf"), 2.0])
    report = evaluate_factor_dq(s, preserve_invalid_rows=False)
    assert not report.passed


def test_preserve_invalid_rows_with_strict_dq_passes():
    s = _series([1.0, float("inf"), float("nan")])
    report = assert_factor_dq(
        s,
        raise_on_fail=True,
        preserve_invalid_rows=True,
    )
    assert report.passed


def test_effective_lookback_scales_intraday_to_daily():
    from cleaned_operators.operator_policy import bars_per_day, effective_lookback

    assert bars_per_day("5m") == 78
    daily = effective_lookback(20, factor_freq="1d", source_bar_freq="1d")
    intraday = effective_lookback(20, factor_freq="1d", source_bar_freq="5m")
    assert intraday > daily
    assert intraday >= 20 * 78


def test_staging_clickhouse_target_writes_staging_only_in_materializer(tmp_path, monkeypatch):
    """materializer 层：staging_clickhouse 仅 upsert staging（CH 由 engine 层追加）。"""
    from storage.materializer import ParquetMaterializer

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["A"]],
        names=["timestamp", "instrument"],
    )
    series = pd.Series([1.0, 2.0], index=idx)
    staging_calls = []

    class _FakeStore:
        def upsert(self, dataset, table, **kwargs):
            staging_calls.append(dataset)
            return {"rows_upserted": 2}

    monkeypatch.setattr("data_access.get_store", lambda: _FakeStore())
    mat = ParquetMaterializer(lake_root=tmp_path)
    result = mat.materialize(
        factor_id="dual_test",
        result=series,
        ast_hash="h1",
        write_target="staging_clickhouse",
    )
    assert result["write_target"] == "staging_clickhouse"
    assert "staging" in result
    assert len(staging_calls) == 1
    assert not (tmp_path / "factors" / "dual_test").exists()


def test_materialize_incremental_staging_clickhouse_dual_write(tmp_path, monkeypatch):
    """增量路径：staging_clickhouse 应 upsert staging 并追加 CH 写入。"""
    from api.columns import col
    from api.factor import Factor
    from backend.pandas_backend import PandasBackend
    from runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource
    from unittest.mock import MagicMock, patch

    dates_old = pd.bdate_range("2024-01-02", periods=5)
    dates_all = pd.bdate_range("2024-01-02", periods=7)
    idx_old = pd.MultiIndex.from_product([dates_old, ["AAA"]], names=["timestamp", "instrument"])
    idx_all = pd.MultiIndex.from_product([dates_all, ["AAA"]], names=["timestamp", "instrument"])
    close_old = pd.Series([float(i) for i in range(len(idx_old))], index=idx_old)
    close_all = pd.Series([float(i) for i in range(len(idx_all))], index=idx_all)

    factor = Factor(name="delay1", expr=col("close"))
    fid = "incr_dual"
    lake = tmp_path / "lake"

    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close_old}),
    )
    staging_calls = []

    class _FakeStore:
        def upsert(self, dataset, table, **kwargs):
            staging_calls.append(dataset)
            return {"rows_upserted": len(dataset)}

    monkeypatch.setattr("data_access.get_store", lambda: _FakeStore())
    mock_insert = MagicMock(return_value=5)
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.write.insert_factor_dataframe", mock_insert):
            engine.materialize(
                factor,
                factor_id=fid,
                lake_root=lake,
                write_target="staging_clickhouse",
                expression='col("close")',
            )

    engine2 = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close_all}),
    )
    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.write.insert_factor_dataframe", mock_insert):
            out = engine2.materialize_incremental(
                factor,
                factor_id=fid,
                lake_root=lake,
                write_target="staging_clickhouse",
                lookback_extra=0,
                recompute_tail_bars=0,
                expression='col("close")',
            )

    assert out["materialization"]["write_target"] == "staging_clickhouse"
    assert "clickhouse" in out["materialization"]
    assert len(staging_calls) >= 2
    assert mock_insert.call_count >= 2


def test_incremental_plan_scales_intraday_source_to_calendar_days():
    from runtime.incremental import build_incremental_plan

    plan = build_incremental_plan(
        factor_id="x",
        analysis_lookback=20,
        watermark={"end_date": "2024-06-01"},
        lookback_extra=0,
        recompute_tail_bars=5,
        factor_freq="1d",
        source_bar_freq="5m",
    )
    assert plan.lookback_bars >= 20 * 78
    assert plan.source_bar_freq == "5m"
    assert plan.load_start is not None
    assert plan.window_mode == "intraday_tick_precise"
    from cleaned_operators.operator_policy import bar_freq_to_timedelta, bars_per_day

    bpd = bars_per_day("5m")
    bar_td = bar_freq_to_timedelta("5m")
    end_anchor = pd.Timestamp("2024-06-01") + bar_td * bpd
    assert end_anchor - plan.load_start >= bar_td * plan.lookback_bars - bar_td


def test_dual_write_clickhouse_failure_raises_with_partial_summary(tmp_path, monkeypatch):
    from api.columns import col
    from api.factor import Factor
    from backend.pandas_backend import PandasBackend
    from runtime.engine import FactorEngine
    from storage.exceptions import DualWriteError
    from tests.helpers import InMemorySeriesSource
    from unittest.mock import MagicMock, patch

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=3), ["A"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([1.0, 2.0, 3.0], index=idx)
    factor = Factor(name="d", expr=col("close"))
    staging_calls = []

    class _FakeStore:
        def upsert(self, dataset, table, **kwargs):
            staging_calls.append(len(dataset))
            return {"rows_upserted": len(dataset)}

    monkeypatch.setattr("data_access.get_store", lambda: _FakeStore())
    mock_cfg = MagicMock()

    def _boom(**kwargs):
        raise RuntimeError("ch down")

    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    )
    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.write.insert_factor_dataframe", side_effect=_boom):
            with pytest.raises(DualWriteError) as excinfo:
                engine.materialize(
                    factor,
                    factor_id="dual_fail",
                    lake_root=tmp_path,
                    write_target="staging_clickhouse",
                    expression='col("close")',
                )
    assert excinfo.value.summary.get("partial_write") is True
    assert excinfo.value.summary.get("primary_write_completed") is True
    assert len(staging_calls) == 1
    from storage.materializer import ParquetMaterializer

    mat = ParquetMaterializer(lake_root=tmp_path)
    assert mat.catalog.get_watermark("dual_fail") is None


def test_defer_watermark_commits_after_ch_success(tmp_path, monkeypatch):
    from api.columns import col
    from api.factor import Factor
    from backend.pandas_backend import PandasBackend
    from runtime.engine import FactorEngine
    from storage.materializer import ParquetMaterializer
    from tests.helpers import InMemorySeriesSource
    from unittest.mock import MagicMock, patch

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=3), ["A"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([1.0, 2.0, 3.0], index=idx)
    factor = Factor(name="ok", expr=col("close"))

    class _FakeStore:
        def upsert(self, dataset, table, **kwargs):
            return {"rows_upserted": len(table)}

    monkeypatch.setattr("data_access.get_store", lambda: _FakeStore())
    mock_cfg = MagicMock()
    engine = FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": close}),
    )
    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.write.insert_factor_dataframe", return_value=3):
            out = engine.materialize(
                factor,
                factor_id="dual_ok",
                lake_root=tmp_path,
                write_target="staging_clickhouse",
            )
    assert out["materialization"].get("dual_write_committed") is True
    mat = ParquetMaterializer(lake_root=tmp_path)
    wm = mat.catalog.get_watermark("dual_ok")
    assert wm is not None
    assert wm["row_count"] == 3


def test_clickhouse_materializer_uses_insert_dataframe(tmp_path):
    from storage.clickhouse_materializer import ClickHouseMaterializer
    from unittest.mock import MagicMock, patch

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["A"]],
        names=["ts", "inst"],
    )
    series = pd.Series([1.0, 2.0], index=idx)
    mock_insert = MagicMock(return_value=2)
    mock_cfg = MagicMock()

    with patch("data_access.clickhouse.panel.ClickHouseConfig.from_env", return_value=mock_cfg):
        with patch("data_access.clickhouse.write.insert_factor_dataframe", mock_insert):
            mat = ClickHouseMaterializer(table="fv")
            summary = mat.materialize("fid", series, ensure_table=False)

    assert summary.rows_written == 2
    mock_insert.assert_called_once()
    frame = mock_insert.call_args.kwargs["frame"]
    assert "is_valid" in frame.columns
    assert "invalid_reason" in frame.columns
