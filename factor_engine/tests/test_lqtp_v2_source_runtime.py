from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_logical_native_bridge_uses_certified_batch_reader_and_exact_keys():
    from factor_engine.api.source_ref import source_col
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource
    name = source_col("StockIncome", "NetProfit").name
    index = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-08-02"), "B"),
        (pd.Timestamp("2024-08-01"), "A")], names=["timestamp", "instrument"])
    class Source(LQTPLogicalDataSource):
        def load_columns(self, names):
            self.requested = names
            return {name: pd.Series([20., np.nan], index=index),
                    "close": pd.Series([5., 7.], index=index[::-1])}
    source = Source(type("Inner", (), {})())
    frame = source.scan_polars_long([name, "close"]).collect().to_pandas()
    assert source.requested == [name, "close"]
    assert frame["inst"].tolist() == ["A", "B"]
    assert np.isnan(frame[name].iloc[0]) and frame[name].iloc[1] == 20
    assert frame["close"].tolist() == [5., 7.]
    source.load_columns = lambda names: {name: pd.Series([1.,2.], index=index[[0,0]])}
    with pytest.raises(ValueError, match="unique date/instrument"):
        source.scan_polars_long([name])


def test_rewrapped_logical_source_reuses_certified_native_wave():
    import polars as pl
    from factor_engine.api.source_ref import source_col
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource
    from factor_engine.storage.sources.wave_prefetched_source import WavePrefetchedSourceAdapter
    name = source_col("StockIncome", "NetProfit").name
    class NoReads:
        def load_columns(self, names):
            raise AssertionError("must not reopen or request pandas")
    adapter = WavePrefetchedSourceAdapter(NoReads())
    frame = pl.DataFrame({"ts":[1,2], "inst":["A","B"], name:[5.,None], "AdjClose":[3.,4.]})
    adapter.publish_native(7,[name,"AdjClose"],frame)
    source = LQTPLogicalDataSource(adapter)
    assert source.scan_polars_long([name,"AdjClose"]).collect().equals(frame)
    assert adapter.has_native_columns([name])
    assert not adapter.has_native_columns(["absent"])
    adapter.release_wave(7)
    assert not adapter.has_native_columns([name])


def test_yaml_dialect_version_is_validated_and_bridged(tmp_path: Path) -> None:
    from factor_engine.runtime.config import load_config
    path = tmp_path / "factor.yaml"
    path.write_text(
        """
factor:
  name: x
  expr: safe_log(close)
  surface: daily
  dialect: lqtp
  dialect_version: '2026-07-19'
data_source:
  type: data_access
  dataset: ashare_stock_daily_adj
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.factor.dialect == "lqtp"
    assert cfg.factor.dialect_version == "2026-07-19"
    assert cfg.factor.surface == "lqtp"  # bridge for existing from_loaded_config API

    path.write_text(path.read_text().replace("2026-07-19", "2025-01-01"), encoding="utf-8")
    with pytest.raises(ValueError, match="dialect_version"):
        load_config(path)


def test_source_ref_disables_full_sql_pushdown() -> None:
    from factor_engine.api.source_ref import source_col
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.planner.sql_lowerer import lower_to_physical_plan

    ref = source_col("BenchmarkIndexDailyBar", "Close", index="000985.SH")
    name = ref.name
    node = PlanNode(
        op="add",
        inputs=(
            PlanNode(op="column", inputs=(), attrs={"name": name}),
            PlanNode(op="literal", inputs=(), attrs={"value": 1.0}),
        ),
        attrs={},
    )
    physical = lower_to_physical_plan(node, mode="research")
    assert physical.fully_sql is False


def test_financial_lag_updates_when_old_quarter_is_revised() -> None:
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    class DummyInner:
        start_date = None
        end_date = None

    class FixtureSource(LQTPLogicalDataSource):
        def _anchor_index(self):
            return pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-08-01"), "A"),
                    (pd.Timestamp("2024-08-20"), "A"),
                ],
                names=["timestamp", "instrument"],
            )

        def _financial_raw(self, dataset: str, field: str):
            return pd.DataFrame(
                {
                    "Symbol": ["A", "A", "A"],
                    "ReportPeriodEndDate": ["2024-03-31", "2024-06-30", "2024-03-31"],
                    "PubDate": ["2024-04-30", "2024-07-31", "2024-08-15"],
                    field: [10.0, 20.0, 11.0],
                }
            )

    source = FixtureSource(DummyInner())
    out = source._financial("dummy", "Metric", "financial_lag", {"quarters": 1})
    np.testing.assert_allclose(out.to_numpy(), [10.0, 11.0], equal_nan=True)


def test_load_source_refs_batch_coalesces_financial_fields() -> None:
    """#10：同表多个财务 SourceRef 合并成一次 store 读，逐字段 PIT join。"""
    from factor_engine.api.source_ref import make_source_ref, encode_source_ref
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    reads: list[list[str]] = []

    class DummyInner:
        start_date = None
        end_date = None

    class FixtureSource(LQTPLogicalDataSource):
        def _anchor_index(self):
            return pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-08-01"), "A"),
                    (pd.Timestamp("2024-08-20"), "A"),
                ],
                names=["timestamp", "instrument"],
            )

        def _financial_raw_multi(self, dataset: str, fields: list[str]):
            reads.append(list(fields))
            return pd.DataFrame(
                {
                    "Symbol": ["A", "A"],
                    "ReportPeriodEndDate": ["2024-03-31", "2024-06-30"],
                    "PubDate": ["2024-04-30", "2024-08-05"],
                    "TotalAssets": [100.0, 200.0],
                    "TotalLiability": [40.0, 90.0],
                }
            )

    source = FixtureSource(DummyInner())
    a = encode_source_ref(make_source_ref("StockBalance", "TotalAssets"))
    b = encode_source_ref(make_source_ref("StockBalance", "TotalLiability"))
    out = source.load_source_refs_batch([a, b])
    # 两个字段 → 一次 _financial_raw_multi 读
    assert reads == [["TotalAssets", "TotalLiability"]]
    np.testing.assert_allclose(out[a].to_numpy(), [100.0, 200.0])
    np.testing.assert_allclose(out[b].to_numpy(), [40.0, 90.0])


def test_financial_batch_collapse_keeps_conflicts_but_drops_snapshot_repeats() -> None:
    import pyarrow as pa
    from factor_engine.storage.sources.lqtp_logical_source_v2 import (
        _collapse_financial_batches,
    )

    columns = ["Symbol", "ReportPeriodEndDate", "PubDate", "TotalAssets"]
    first = pa.RecordBatch.from_pylist([
        {"Symbol": "A", "ReportPeriodEndDate": "2024-03-31", "PubDate": "2024-04-30", "TotalAssets": 100.0},
        {"Symbol": "A", "ReportPeriodEndDate": "2024-03-31", "PubDate": "2024-04-30", "TotalAssets": 100.0},
    ])
    second = pa.RecordBatch.from_pylist([
        {"Symbol": "A", "ReportPeriodEndDate": "2024-03-31", "PubDate": "2024-04-30", "TotalAssets": 100.0},
        {"Symbol": "A", "ReportPeriodEndDate": "2024-03-31", "PubDate": "2024-04-30", "TotalAssets": 101.0},
    ])
    out = _collapse_financial_batches([first, second], columns)
    assert out.to_dict("records") == [
        {"Symbol": "A", "ReportPeriodEndDate": "2024-03-31", "PubDate": "2024-04-30", "TotalAssets": 100.0},
        {"Symbol": "A", "ReportPeriodEndDate": "2024-03-31", "PubDate": "2024-04-30", "TotalAssets": 101.0},
    ]


def test_financial_raw_multi_streams_bounded_and_closes(monkeypatch) -> None:
    import pyarrow as pa
    from factor_engine.storage.sources import data_access_source
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    calls = {}
    class Handle:
        snapshot = type("Snapshot", (), {"snapshot_id": "snap-1"})()
        def stream(self, *, batch_size):
            calls["stream_batch_size"] = batch_size
            yield pa.RecordBatch.from_pylist([{
                "Symbol": "A", "ReportPeriodEndDate": "2024-03-31",
                "PubDate": "2024-04-30", "TotalAssets": 100.0,
            }])
        def close(self):
            calls["closed"] = True
    class Store:
        def get_dataset(self, dataset):
            return type("Dataset", (), {"instrument_column": "Symbol"})()
        def read(self, dataset, **kwargs):
            calls["read"] = (dataset, kwargs)
            return Handle()
    class Inner:
        end_date = "2026-04-30"
        instrument_filter = ["A"]
        run_mode = "interactive_research"

    monkeypatch.setattr(data_access_source, "_get_store", lambda: Store())
    source = LQTPLogicalDataSource(Inner())
    out = source._financial_raw_multi("ashare_stock_balance", ["TotalAssets"])
    dataset, kwargs = calls["read"]
    assert dataset == "ashare_stock_balance"
    assert kwargs["columns"] == [
        "Symbol", "ReportPeriodEndDate", "PubDate", "TotalAssets"]
    assert kwargs["time_range"] == (None, "2026-04-30")
    assert kwargs["instrument_filter"] == ["A"]
    assert kwargs["result"] == "stream"
    assert kwargs["batch_size"] == calls["stream_batch_size"] == 50_000
    assert calls["closed"] is True
    assert out["TotalAssets"].tolist() == [100.0]
    dependency = source.collect_source_dependencies()[0]
    assert dependency["snapshot_id"] == "snap-1"
    assert dependency["availability_column"] == "PubDate"


def test_minute_resample_does_not_silently_collapse_to_daily() -> None:
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource
    from factor_engine.storage.sources.data_access_source import MissingDataDependencyError

    class DailyInner:
        def load_column(self, name: str):
            if name != "close":
                raise KeyError(name)
            idx = pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-02"), "A"),
                    (pd.Timestamp("2024-01-03"), "A"),
                ],
                names=["timestamp", "instrument"],
            )
            return pd.Series([1.0, 2.0], index=idx, name="close")

    source = LQTPLogicalDataSource(DailyInner(), factor_freq="1d")
    with pytest.raises(MissingDataDependencyError, match="intraday bar sequence"):
        source._minute_daily("Close", "minute_resample", {"period": 5})


def test_multiminute_vwap_is_amount_over_volume() -> None:
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    source = LQTPLogicalDataSource(object(), factor_freq="1d")
    amount = pd.Series([1000.0], index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")], names=["timestamp", "instrument"]
    ))
    volume = pd.Series([10.0], index=amount.index)

    def fake_minute(field: str, transform: str, params: dict):
        return {"Amount": amount, "Volume": volume}[field]

    source._minute_daily = fake_minute
    out = LQTPLogicalDataSource._minute_weighted_vwap(source, "minute_bar", {"period": 5})
    assert out.iloc[0] == pytest.approx(100.0)


def test_blocked_lqtp_names_are_classified_not_unknown() -> None:
    from factor_engine.api.dsl_parser import DSLParseError, parse_expr
    for formula in [
        "l2_sum(close)",
        "l2_count(close)",
        "group_minmax(close, industry)",
        "group_quantile_mask(close, industry, 0.2, 0.8)",
    ]:
        with pytest.raises(DSLParseError, match="recognized but not executable"):
            parse_expr(formula, surface="lqtp")


def test_machine_readable_manifest_distinguishes_blocked_from_production() -> None:
    from factor_engine.api.lqtp_capabilities import build_lqtp_capability_manifest
    manifest = build_lqtp_capability_manifest()
    assert manifest["dialect_version"] == "2026-07-19"
    assert "ts_sumac" in manifest["recognized_blocked"]
    assert "l2_sum" in manifest["recognized_blocked"]
    # Stable neutralize one-arg forms are sourced, not blocked.
    assert "size_neutralize" not in manifest["recognized_blocked"]
    assert "industry_size_neutralize" not in manifest["recognized_blocked"]
    assert "industry_neutralize(x)" in manifest["source_aware"]
    assert "size_neutralize(x)" in manifest["source_aware"]
    assert "industry_size_neutralize(x)" in manifest["source_aware"]
    assert "size_neutralize" in manifest["canonical_operators"]
    assert "industry_size_neutralize" in manifest["canonical_operators"]
