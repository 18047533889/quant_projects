"""read/aggregation.py —— 分钟→日聚合 DuckDB 下推（#14）。"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.read.aggregation import (
    AggregationSpec,
    aggregate_minute_to_daily,
    parse_aggregation_spec,
)
from data_access.registry import load_registry
from data_access.store import DataAccessStore


@pytest.fixture()
def minute_store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    ts = pd.date_range("2024-01-02 09:30", periods=5, freq="2min").append(
        pd.date_range("2024-01-02 10:00", periods=3, freq="2min")
    )
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["A"] * 8,
            "volume": [100, 200, 300, 400, 500, 600, 700, 800],
            "close": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
        }
    )
    pq.write_table(pa.table(df), str(tmp_path / "m.parquet"))
    (tmp_path / "datasets.yaml").write_text(
        f"""
minute_ds:
  kind: static
  access_mode: published
  layout: plain
  format: parquet
  root: {tmp_path}
  glob: "*.parquet"
  time_column: timestamp
  instrument_column: symbol
  schema:
    timestamp: timestamp
    symbol: string
    volume: int
    close: double
""",
        encoding="utf-8",
    )
    return DataAccessStore(
        registry=load_registry(tmp_path / "datasets.yaml"),
        engine=DuckDBEngine(threads=2, enable_object_cache=False),
    )


def test_minute_range_sum_pushdown(minute_store):
    """#14 minute_range 09:30~10:00 volume sum（含 10:00 整点）在 DuckDB 内完成。"""
    spec = AggregationSpec(aggregation="minute_range", start="09:30", end="10:00")
    # #7 返回带 snapshot 的 ReadHandle
    t = aggregate_minute_to_daily(minute_store, "minute_ds", "volume", spec).to_arrow()
    rows = t.to_pydict()
    assert rows["inst"] == ["A"]
    assert int(rows["value"][0]) == 100 + 200 + 300 + 400 + 500 + 600  # 含 10:00


def test_minute_at_first_and_last(minute_store):
    """#14 minute_at：close 取该分钟最后一笔（语义 last），open 取第一笔。"""
    t = aggregate_minute_to_daily(
        minute_store, "minute_ds", "close",
        AggregationSpec(aggregation="minute_at", hhmm="09:30"),
    ).to_arrow()
    assert float(t.to_pydict()["value"][0]) == pytest.approx(10.0)  # 09:30 只有一笔
    # 10:00 只有 10:00 一笔 → last = 10.5
    t2 = aggregate_minute_to_daily(
        minute_store, "minute_ds", "close",
        AggregationSpec(aggregation="minute_at", hhmm="10:00"),
    ).to_arrow()
    assert float(t2.to_pydict()["value"][0]) == pytest.approx(10.5)


def test_minute_range_instrument_filter(minute_store):
    """#14 instrument_filter 下推到聚合。"""
    spec = AggregationSpec(aggregation="minute_range", start="09:30", end="09:40")
    t = aggregate_minute_to_daily(
        minute_store, "minute_ds", "volume", spec,
        instrument_filter=["NOPE"],
    ).to_arrow()
    assert t.num_rows == 0


def test_parse_aggregation_spec_defaults():
    """归一化：字符串 / dict / 实例。"""
    spec = parse_aggregation_spec("minute_at", field="volume")
    assert spec.aggregation == "minute_at"
    assert spec.metric == "sum"  # volume → sum
    spec2 = parse_aggregation_spec(
        {"aggregation": "minute_range", "start": "09:31", "end": "10:00"}, field="close"
    )
    assert spec2.metric is None  # 显式 dict 不推断 metric


def test_parse_aggregation_spec_strict_hhmm_rejects_bad_time():
    """#P0-56 非法 HH:MM（含非时间字符串）必须 ValidationError，不再静默接受。"""
    from data_access.core.exceptions import ValidationError

    for bad in ("a", "b", "9:31", "09:61", "24:00", 931):
        with pytest.raises(ValidationError):
            parse_aggregation_spec(
                {"aggregation": "minute_range", "start": bad, "end": "10:00"},
                field="close",
            )


def test_parse_aggregation_spec_strict_period_index():
    """#P0-56 period<=0 / index<0 / unknown key 必须 fail-closed，禁止 silent clamp。"""
    from data_access.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        parse_aggregation_spec({"period": -100}, field="close")
    with pytest.raises(ValidationError):
        parse_aggregation_spec({"period": 5, "index": -1}, field="close")
    with pytest.raises(ValidationError):
        parse_aggregation_spec({"aggregation": "minute_at", "bogus": 1}, field="close")
    with pytest.raises(ValidationError):
        parse_aggregation_spec({"timezone": "Not/AZone"}, field="close")
