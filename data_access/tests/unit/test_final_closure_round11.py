"""#收官轮（FE 集成复查）DataAccess 侧回归测试。

覆盖（closure ledger ID，对应外部 AI 复查项）：
    P0-2  真 PyArrow backend（engine="pyarrow"）与 DuckDB / Polars 真实 parity：
          date-only end 含完整一天、空股票池（[]）→ 0 行——三引擎结果一致，
          不再测 mode="arrow"（那是 DuckDB→Arrow 表，不是真 PyArrow 引擎）。
    9     DataRequest 静态 universe 空集 → 0 行 typed result（不是 None=全市场）。
    10    read_factors 的 date-only end 包含最后一天（datetime timestamp 列）。
    11/12 factor 元数据 universe/frequency/snapshot roundtrip（factor lake）。
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.read.data_request import DataRequest
from data_access.registry import load_registry
from data_access.store import DataAccessStore

_PKG = "/home/shw/quant_projects/dataaccess"


@pytest.fixture(autouse=True)
def _no_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)


def _ts_store(tmp_path, name="ds"):
    """timestamp 时间列（含白天行）的静态数据集 + store。"""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for d, sym, v in [
        ("2024-01-02 09:30:00", "AAA", 1.0),
        ("2024-01-02 14:00:00", "BBB", 2.0),
        ("2024-01-03 09:30:00", "AAA", 3.0),   # 最后一天白天行
        ("2024-01-03 15:30:00", "BBB", 4.0),  # 最后一天白天行
        ("2024-01-04 09:30:00", "AAA", 5.0),
    ]:
        rows.append({"d": _dt.datetime.fromisoformat(d), "s": sym, "v": v})
    pq.write_table(pa.Table.from_pylist(rows), str(root / "part.parquet"))
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
{name}:
  kind: static
  access_mode: published
  layout: plain
  root: {root}
  glob: "*.parquet"
  time_column: d
  instrument_column: s
  schema:
    d: timestamp
    s: string
    v: double
""",
        encoding="utf-8",
    )
    return DataAccessStore(
        registry=load_registry(cfg), engine=DuckDBEngine(threads=2)
    )


def _nrows(store, ds, *, engine, time_range, instrument_filter=None):
    """三引擎统一读取，返回行数。"""
    handle = store.read(
        ds,
        columns=["d", "s", "v"],
        time_range=time_range,
        instrument_filter=instrument_filter,
        engine=engine,
    )
    return handle.to_arrow().num_rows


# ---------------------------------------------------------------------------
# P0-2 真 PyArrow backend 三引擎 parity
# ---------------------------------------------------------------------------


def test_three_engine_parity_date_only_end(tmp_path):
    """date-only end 含完整一天：三引擎（duckdb/polars/pyarrow）行数一致。

    time_range end = "2024-01-03" 必须包含 01-03 的白天行（09:30 / 15:30），
    不能 ``<= 2024-01-03 00:00`` 丢掉。
    """
    store = _ts_store(tmp_path)
    got = {
        eng: _nrows(
            store, "ds", engine=eng,
            time_range=("2024-01-02", "2024-01-03"),
        )
        for eng in ("duckdb", "polars", "pyarrow")
    }
    assert got == {"duckdb": 4, "polars": 4, "pyarrow": 4}, got


def test_three_engine_parity_datetime_end(tmp_path):
    """带时间的 end 原样 ``<=``（不展开），三引擎一致。"""
    store = _ts_store(tmp_path)
    end = "2024-01-03 15:00:00"
    got = {
        eng: _nrows(
            store, "ds", engine=eng,
            time_range=("2024-01-02", end),
        )
        for eng in ("duckdb", "polars", "pyarrow")
    }
    # 01-02 两行 + 01-03 09:30 一行（15:30 在 end 之后）
    assert got == {"duckdb": 3, "polars": 3, "pyarrow": 3}, got


def test_three_engine_parity_empty_instrument_filter(tmp_path):
    """instrument_filter=[] → 空股票池 0 行（不是全市场），三引擎一致。"""
    store = _ts_store(tmp_path)
    for eng in ("duckdb", "polars", "pyarrow"):
        n = _nrows(
            store, "ds", engine=eng,
            time_range=("2024-01-02", "2024-01-04"),
            instrument_filter=[],
        )
        assert n == 0, f"engine={eng} 空股票池应 0 行，实际 {n}"
    # None 仍 = 全市场（对照）
    assert _nrows(store, "ds", engine="duckdb",
                  time_range=("2024-01-02", "2024-01-04")) == 5


def test_three_engine_parity_instrument_filter(tmp_path):
    """单标的过滤三引擎一致。"""
    store = _ts_store(tmp_path)
    got = {
        eng: _nrows(
            store, "ds", engine=eng,
            time_range=("2024-01-02", "2024-01-04"),
            instrument_filter=["AAA"],
        )
        for eng in ("duckdb", "polars", "pyarrow")
    }
    assert got == {"duckdb": 3, "polars": 3, "pyarrow": 3}, got


# ---------------------------------------------------------------------------
# 9) DataRequest 静态 universe 空集 → 0 行（不是 None=全市场）
# ---------------------------------------------------------------------------


def test_data_request_static_empty_universe_not_full_market(tmp_path):
    root = tmp_path / "a"
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([
            {"d": _dt.date(2024, 1, 2), "s": "AAA", "v": 1.5},
        ]),
        str(root / "part.parquet"),
    )
    uroot = tmp_path / "u"
    uroot.mkdir(parents=True, exist_ok=True)  # 目录存在但无任何 parquet → 空成分
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
a:
  kind: static
  access_mode: published
  layout: plain
  root: {root}
  glob: "*.parquet"
  time_column: d
  instrument_column: s
  schema:
    d: date
    s: string
    v: double
u:
  kind: static
  access_mode: published
  layout: plain
  root: {uroot}
  glob: "*.parquet"
  time_column: d
  instrument_column: s
  schema:
    d: date
    s: string
""",
        encoding="utf-8",
    )
    store = DataAccessStore(registry=load_registry(cfg), engine=DuckDBEngine(threads=2))
    plan = store.plan(
        DataRequest(
            fields=["v"],
            start="2024-01-02",
            end="2024-01-02",
            universe="u",
            time_varying_universe=False,
        )
    )
    handle = plan.execute()
    t = handle.to_arrow()
    assert t.num_rows == 0
    assert "v" in t.column_names  # typed schema 保留


# ---------------------------------------------------------------------------
# 10) read_factors date-only end 包含完整最后一天（timestamp datetime 列）
# ---------------------------------------------------------------------------


def test_read_factors_date_only_end_includes_last_day(tmp_path):
    lake = tmp_path / "lake"
    d = lake / "factors" / "mom_3d" / "year=2024"
    d.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([
            {"datetime": _dt.datetime(2024, 1, 2, 9, 30), "asset": "AAPL", "value": 1.0},
            {"datetime": _dt.datetime(2024, 1, 3, 15, 30), "asset": "AAPL", "value": 2.0},
            {"datetime": _dt.datetime(2024, 1, 4, 9, 30), "asset": "AAPL", "value": 3.0},
        ]),
        str(d / "data.parquet"),
    )
    (lake / "factors" / "mom_3d" / "_factor_meta.json").write_text(
        json.dumps({"factor_version": "v1", "frequency": "daily",
                    "status": "active", "universe": "us"}),
        encoding="utf-8",
    )
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
factor_lake:
  kind: parametric
  access_mode: published
  layout: hive
  root_template: {lake}/factors/{{factor_id}}
  glob_template: "year=*/data.parquet"
  partition_columns: [year]
  params_schema:
    factor_id: str
  time_column: datetime
  instrument_column: asset
  hive_partitioning: true
  union_by_name: true
  schema:
    datetime: timestamp
    asset: string
    value: double
""",
        encoding="utf-8",
    )
    store = DataAccessStore(registry=load_registry(cfg), engine=DuckDBEngine(threads=2))
    # date-only end=2024-01-03 → 必须包含 01-03 15:30 白天行（旧 ``<= 00:00`` 会丢）
    handle = store.read_factors(
        ["mom_3d"], columns=["datetime", "asset", "value"],
        time_range=("2024-01-02", "2024-01-03"),
    )
    assert handle.to_arrow().num_rows == 2


# ---------------------------------------------------------------------------
# 11/12) factor 元数据 universe/frequency/snapshot roundtrip
# ---------------------------------------------------------------------------


def test_factor_meta_universe_frequency_snapshot_roundtrip(tmp_path):
    lake = tmp_path / "lake"
    d = lake / "factors" / "mom_3d" / "year=2024"
    d.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([
            {"datetime": _dt.datetime(2024, 1, 2, 9, 30), "asset": "AAPL", "value": 1.0,
             "factor_version": "v1", "data_snapshot_id": "snap-abc"},
        ]),
        str(d / "data.parquet"),
    )
    meta = {
        "factor_version": "v1", "frequency": "daily", "status": "active",
        "universe": "us", "data_snapshot_id": "snap-abc",
    }
    (lake / "factors" / "mom_3d" / "_factor_meta.json").write_text(
        json.dumps(meta), encoding="utf-8",
    )
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
factor_lake:
  kind: parametric
  access_mode: published
  layout: hive
  root_template: {lake}/factors/{{factor_id}}
  glob_template: "year=*/data.parquet"
  partition_columns: [year]
  params_schema:
    factor_id: str
  time_column: datetime
  instrument_column: asset
  hive_partitioning: true
  union_by_name: true
  schema:
    datetime: timestamp
    asset: string
    value: double
    factor_version: string
    data_snapshot_id: string
""",
        encoding="utf-8",
    )
    store = DataAccessStore(registry=load_registry(cfg), engine=DuckDBEngine(threads=2))
    catalog = store.get_factor_catalog()
    assert "mom_3d" in catalog
    f = catalog.get("mom_3d")
    assert f is not None
    assert f.frequency == "daily"
    assert f.universe == "us"
    assert f.factor_version == "v1"
    # snapshot 贯穿到读取结果
    handle = store.read_factors(["mom_3d"], columns=["datetime", "asset", "value",
                                                     "factor_version", "data_snapshot_id"])
    t = handle.to_arrow()
    assert t.column("data_snapshot_id").to_pylist() == ["snap-abc"]


def _mk_request(**kw):
    from data_access.read.data_request import DataRequest

    return DataRequest(**kw)
