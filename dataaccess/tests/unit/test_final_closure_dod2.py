"""最终收官轮（closure）组合场景 DoD：并发写、plan 冻结、物理 pin、组合预算、
availability 编译器、PIT fail-closed、derived fail-closed、语义歧义、重复列、
FormatSpec 严格校验。

对应「最终收官轮」用户清单 1–11 + 组合 DoD 场景。全部在本地临时数据集上验证，
不依赖生产数据 / GitHub。
"""
from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access import DataRequest
from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import (
    AmbiguousSemanticFieldError,
    SnapshotBuildError,
    ValidationError,
)
from data_access.registry import load_registry
from data_access.store import DataAccessStore


def _store(tmp_path, datasets: dict[str, dict]) -> DataAccessStore:
    cfg = tmp_path / "datasets.yaml"
    lines = []
    for name, body in datasets.items():
        lines.append(f"{name}:")
        for k, v in body.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for sk, sv in v.items():
                    lines.append(f"    {sk}: {json.dumps(str(sv))}")
            else:
                lines.append(f"  {k}: {json.dumps(str(v))}")
    cfg.write_text("\n".join(lines), encoding="utf-8")
    return DataAccessStore(load_registry(cfg), DuckDBEngine(threads=2))


def _writable_ds(root: Path, name: str) -> dict[str, dict]:
    return {
        name: {
            "kind": "static", "access_mode": "staging", "layout": "plain",
            "root": str(root), "glob": "part-*.parquet",
            "time_column": "ts", "instrument_column": "sym",
            "schema": {"ts": "date", "sym": "string", "val": "double"},
        }
    }


# ---------------------------------------------------------------------------
# 1) 两个并发 writer：mutation → 整事务同锁，最终 manifest fresh + generation 一致
# ---------------------------------------------------------------------------

def test_concurrent_writers_serialize_fresh_manifest(tmp_path):
    """统一事务后并发写不竞态：两个线程写同一数据集，串行完成；最终 manifest
    source_epoch == manifest_built_epoch（fresh），generation 双写一致。"""
    d = tmp_path / "d"
    d.mkdir()
    store = _store(tmp_path, _writable_ds(d, "myds"))
    store.write_arrow("myds", pa.table(
        {"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [1.0]}))
    store.build_dataset_manifest("myds")

    results: list[int] = []
    errors: list[Exception] = []

    def _w(i: int) -> None:
        try:
            # mode="append"：不互清，验证两笔并发写都串行落盘
            store.write_arrow(
                "myds",
                pa.table({"ts": [dt.date(2024, 1, 2 + i)], "sym": ["A"],
                          "val": [float(i)]}),
                mode="append",
            )
            results.append(i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_w, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert sorted(results) == [0, 1]
    # 两笔都落盘（warmup + 2 append）+ manifest fresh（最后一个事务 rebuild 追平）
    assert len(list(d.glob("part-*.parquet"))) == 3
    tok = store.manifest_version("myds")
    assert tok["has_manifest"]
    assert tok["fresh"] is True, tok
    assert tok.get("manifest_generation_id")
    # generation 双写一致：parquet schema metadata 与 JSON sidecar 同代
    from data_access.read.manifest import (
        _MANIFEST_META_FILENAME,
        MANIFEST_FILENAME,
        _row_groups_generation,
    )
    import pyarrow.parquet as _pq

    gen_parquet = _row_groups_generation(d / MANIFEST_FILENAME)
    meta = json.loads((d / _MANIFEST_META_FILENAME).read_text(encoding="utf-8"))
    assert gen_parquet == meta["manifest_generation_id"]


# ---------------------------------------------------------------------------
# 2) plan 后改嵌套 filter/join/anchor/aggregation → execute 完全不变
# ---------------------------------------------------------------------------

def test_plan_deep_freeze_survives_post_plan_mutation(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    store = _store(tmp_path, _writable_ds(d, "myds"))
    store.write_arrow("myds", pa.table(
        {"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [1.0]}))
    req = DataRequest(fields=["ts", "sym", "val"])
    plan = store.plan(req)
    plan.execute().to_arrow()
    # plan 之后篡改活的 request 的嵌套结构
    req.anchor = "evil"
    req.filters = {"ts": {"<": dt.date(2000, 1, 1)}}  # type: ignore[assignment]
    req.joins = {"other": {"policy": "exact"}}  # type: ignore[assignment]
    tbl = plan.execute().to_arrow()
    assert tbl.num_rows == 1  # 仍是 plan 时的语义（无 filter）

    req2 = DataRequest(fields=["ts", "sym", "val"])
    req2.aggregations = None  # type: ignore[assignment]
    plan2 = store.plan(req2)
    # aggregation 项里的 dict 也要冻结：plan 时改 req2.aggregations 内部不生效
    req3 = DataRequest(
        fields=["ts", "sym", "val"],
        aggregations=[{"field": "val", "spec": {"aggregation": "sum"}}],
    )
    plan3 = store.plan(req3)
    req3.aggregations[0]["field"] = "sym"  # type: ignore[index]
    assert plan3.compiled.aggregations[0]["field"] == "val"


# ---------------------------------------------------------------------------
# 3) plan 后删除/替换 manifest → pin / fail_if_changed 正确拒绝
# ---------------------------------------------------------------------------

def test_pin_rejects_file_replace_and_manifest_disappear(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    store = _store(tmp_path, _writable_ds(d, "myds"))
    store.write_arrow("myds", pa.table(
        {"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [1.0]}))
    store.build_dataset_manifest("myds")

    plan = store.plan(DataRequest(fields=["ts", "sym", "val"], snapshot_policy="pin"))
    plan.execute().to_arrow()
    # 外部系统直接替换 parquet（不 bump epoch）
    fp = sorted(d.glob("part-*.parquet"))[0]
    pq.write_table(pa.table(
        {"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [999.0]}), str(fp))
    with pytest.raises(SnapshotBuildError):
        plan.execute()

    plan2 = store.plan(
        DataRequest(fields=["ts", "sym", "val"], snapshot_policy="fail_if_changed"))
    plan2.execute()
    (d / "_manifest.json").unlink()  # manifest 消失
    with pytest.raises(SnapshotBuildError):
        plan2.execute()


# ---------------------------------------------------------------------------
# 4) composed aggregation+join 与 read_joined 同一 merged budget
# ---------------------------------------------------------------------------

def test_composed_aggregation_join_budget_parity(tmp_path):
    minute = tmp_path / "minute"
    fin = tmp_path / "fin"
    minute.mkdir()
    fin.mkdir()
    rows = []
    for day in (1, 2):
        for s in ("A",):
            for i in range(3):
                rows.append((dt.datetime(2024, 1, day, 5 + i), s, 100.0, i + 1))
    pq.write_table(
        pa.table({"ts": [r[0] for r in rows], "inst": [r[1] for r in rows],
                  "Close": [r[2] for r in rows], "Volume": [r[3] for r in rows]}),
        str(minute / "m.parquet"),
    )
    # fin_ds 拆成两个文件，query_policy.max_scan_files=1 → 组合读必须拒绝
    pq.write_table(
        pa.table({"financial_time": [dt.date(2024, 1, 1)] * 2, "inst": ["A", "A"],
                  "Eps": [5.0, 6.0]}),
        str(fin / "f1.parquet"),
    )
    pq.write_table(
        pa.table({"financial_time": [dt.date(2024, 1, 2)] * 2, "inst": ["A", "A"],
                  "Eps": [8.0, 9.0]}),
        str(fin / "f2.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "minute_ds": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(minute), "glob": "*.parquet",
                "time_column": "ts", "instrument_column": "inst",
                "schema": {"ts": "timestamp", "inst": "string",
                           "Close": "double", "Volume": "int"},
            },
            "fin_ds": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(fin), "glob": "*.parquet",
                "time_column": "financial_time", "instrument_column": "inst",
                "schema": {"financial_time": "date", "inst": "string", "Eps": "double"},
                "query_policy": {"max_scan_files": 1},
            },
        },
    )
    from data_access.read.aggregation import AggregationItem, AggregationSpec

    spec = AggregationSpec(aggregation="minute_range", start="13:00", end="15:00",
                           metric="sum", market="ashare")
    req = DataRequest(
        fields=["minute_ds.Volume", "fin_ds.Eps"],
        start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2),
        instruments=["A"], anchor="minute_ds",
        aggregations=[AggregationItem("Volume", spec, "total_vol")],
        joins={"fin_ds": {"policy": "asof", "knowledge_time": "financial_time"}},
    )
    with pytest.raises(ValidationError, match="max_scan_files|扫描匹配"):
        store.plan(req).execute()


# ---------------------------------------------------------------------------
# 5) max_scan_files 在聚合路径同样生效
# ---------------------------------------------------------------------------

def test_aggregation_respects_max_scan_files(tmp_path):
    from data_access.read.aggregation import aggregate_minute_bundle, AggregationItem, AggregationSpec

    minute = tmp_path / "minute"
    minute.mkdir()
    for i in (1, 2):
        pq.write_table(
            pa.table({"ts": [dt.datetime(2024, 1, 1, 5)], "inst": ["A"],
                      "Volume": [i]}),
            str(minute / f"m{i}.parquet"),
        )
    store = _store(
        tmp_path,
        {
            "minute_ds": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(minute), "glob": "*.parquet",
                "time_column": "ts", "instrument_column": "inst",
                "schema": {"ts": "timestamp", "inst": "string", "Volume": "int"},
                "query_policy": {"max_scan_files": 1},
            },
        },
    )
    spec = AggregationSpec(aggregation="minute_range", start="00:00", end="23:59",
                           metric="sum", market="ashare")
    with pytest.raises(ValidationError, match="max_scan_files|扫描匹配"):
        aggregate_minute_bundle(
            store, "minute_ds", [AggregationItem("Volume", spec, "vol")],
            time_range=(dt.date(2024, 1, 1), dt.date(2024, 1, 1)),
            instrument_filter=["A"],
        )


# ---------------------------------------------------------------------------
# 6) availability 编译器：next_bar / 午休 / 收盘 / 周末 / early-close
# ---------------------------------------------------------------------------

def test_availability_compiler_next_bar_full_matrix():
    from data_access.read.session_calendar import (
        MarketCalendar,
        build_ashare_session,
        build_us_session,
        compile_available_from,
    )

    days = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4),
            dt.date(2024, 1, 5), dt.date(2024, 1, 8)]
    cal = MarketCalendar("ashare", trading_days=days, session=build_ashare_session())
    from zoneinfo import ZoneInfo

    TZ = ZoneInfo("Asia/Shanghai")
    L = lambda y, mo, d, h, mi: dt.datetime(y, mo, d, h, mi, tzinfo=TZ)  # noqa: E731

    def nb(k):
        return compile_available_from(k, "next_bar", calendar=cal)

    assert nb(L(2024, 1, 2, 10, 15)) == dt.datetime(2024, 1, 2, 10, 16)   # 盘中
    assert nb(L(2024, 1, 2, 11, 30)) == dt.datetime(2024, 1, 2, 13, 1)    # 午休跨段
    assert nb(L(2024, 1, 2, 11, 29)) == dt.datetime(2024, 1, 2, 11, 30)   # 段内
    assert nb(L(2024, 1, 2, 15, 0)) == dt.datetime(2024, 1, 3, 9, 31)     # 收盘→次日
    assert nb(L(2024, 1, 2, 12, 0)) == dt.datetime(2024, 1, 2, 13, 1)     # 午休
    assert nb(L(2024, 1, 2, 9, 0)) == dt.datetime(2024, 1, 2, 9, 31)      # 盘前→当日
    assert nb(L(2024, 1, 6, 10, 0)) == dt.datetime(2024, 1, 8, 9, 31)     # 周末→下交易日
    # latency 叠加
    assert compile_available_from(
        L(2024, 1, 2, 10, 15), "next_bar", calendar=cal, latency=2
    ) == dt.datetime(2024, 1, 2, 10, 18)
    # early-close：美股 half day 13:00 收市 → 12:59 下一根 13:00
    TZ2 = ZoneInfo("America/New_York")
    us = MarketCalendar(
        "us", trading_days=days,
        session=build_us_session(early_close_dates={dt.date(2024, 1, 3)}),
    )
    assert compile_available_from(
        dt.datetime(2024, 1, 3, 12, 59, tzinfo=TZ2), "next_bar", calendar=us
    ) == dt.datetime(2024, 1, 3, 13, 0)


# ---------------------------------------------------------------------------
# 7) 事件 asof 消费统一 availability（next_bar 可见性 + latency）
# ---------------------------------------------------------------------------

def test_event_asof_uses_calendar_available_from(tmp_path):
    from data_access.cos_event_runtime import _resolve_event_availability, _select
    from data_access.read.session_calendar import MarketCalendar, build_ashare_session
    import pandas as pd
    import numpy as np
    from data_access.cos_contract import COSDatasetContract

    contract = COSDatasetContract(
        name="ev", market="ashare", temporal_model="E1", join_policy="event",
        instrument_column="Symbol", panel_policy="event_only", pit_policy="strict",
        availability_column="knowledge", period_column=None, event_column=None,
    )
    days = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    cal = MarketCalendar("ashare", trading_days=days, session=build_ashare_session())
    # knowledge = 2024-01-02 11:30 上海本地（=03:30 UTC），盘中 → next_bar =
    # 13:01 本地 = 05:01 UTC（午休跨段）。
    events = pd.DataFrame({
        "Symbol": ["A"],
        "knowledge": [pd.Timestamp("2024-01-02 11:30", tz="Asia/Shanghai")],
        "val": [7.0],
    })
    # 13:00 上海 = 05:00 UTC < 05:01 UTC → 未到下一根 bar，不可见
    decisions = pd.DataFrame({
        "__pit_position": np.arange(1, dtype=np.int64),
        "instrument": ["A"],
        "decision_timestamp": [pd.Timestamp("2024-01-02 05:00", tz="UTC")],
    })
    selected = _select(
        decisions, events, contract, "decision_timestamp", "instrument",
        "knowledge", "latest_available", availability="next_bar", calendar=cal,
    )
    assert selected[0] is None
    # 13:03 上海 = 05:03 UTC ≥ 05:01 UTC → 可见
    decisions2 = decisions.copy()
    decisions2["decision_timestamp"] = [pd.Timestamp("2024-01-02 05:03", tz="UTC")]
    selected2 = _select(
        decisions2, events, contract, "decision_timestamp", "instrument",
        "knowledge", "latest_available", availability="next_bar", calendar=cal,
    )
    assert selected2[0] is not None


# ---------------------------------------------------------------------------
# 8) PIT index：glob 枚举失败 → complete=False；null ticker/filing → reject
# ---------------------------------------------------------------------------

def test_pit_index_enumeration_failure_fail_closed(tmp_path):
    from data_access.read.pit_event_index import build_pit_event_index

    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"Ticker": ["A"], "filing_date": [dt.date(2024, 1, 1)]}),
        str(d / "a.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "us_balance": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "filing_date", "instrument_column": "Ticker",
                "schema": {"Ticker": "string", "filing_date": "date"},
            }
        },
    )

    real_execute = store._engine.execute_arrow

    def flaky_execute(sql, params, deadline_ms=None):
        if "glob(" in str(sql):
            raise RuntimeError("glob broken")
        return real_execute(sql, params, deadline_ms=deadline_ms)

    store._engine.execute_arrow = flaky_execute  # type: ignore[method-assign]
    try:
        idx = build_pit_event_index(store, "us_balance", force=True)
    finally:
        store._engine.execute_arrow = real_execute  # type: ignore[method-assign]
    assert idx.metadata.complete is False
    assert idx.metadata.glob_failed is True
    assert idx.metadata.is_authoritative is False
    assert any(str(f).startswith("glob:") for f in idx.metadata.failed_files)


def test_pit_index_null_ticker_rejected(tmp_path):
    from data_access.read.pit_event_index import build_pit_event_index

    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"Ticker": [None], "filing_date": [dt.date(2024, 1, 1)]}),
        str(d / "a.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "us_balance": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "filing_date", "instrument_column": "Ticker",
                "schema": {"Ticker": "string", "filing_date": "date"},
            }
        },
    )
    with pytest.raises(ValidationError, match="null ticker"):
        build_pit_event_index(store, "us_balance", force=True)


def test_pit_index_null_filing_rejected(tmp_path):
    from data_access.read.pit_event_index import build_pit_event_index

    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"Ticker": ["A"], "filing_date": [None]}),
        str(d / "a.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "us_balance": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "filing_date", "instrument_column": "Ticker",
                "schema": {"Ticker": "string", "filing_date": "date"},
            }
        },
    )
    with pytest.raises(ValidationError, match="filing_date"):
        build_pit_event_index(store, "us_balance", force=True)


# ---------------------------------------------------------------------------
# 9) derived 字段 fail-closed：catalog load 拒 mining_allowed=true；planner 拒读取
# ---------------------------------------------------------------------------

def test_derived_field_fail_closed_catalog_and_planner(tmp_path):
    from data_access.read.semantic_catalog import parse_semantic_field

    # catalog load：derived + mining_allowed=true → 拒绝
    with pytest.raises(ValidationError, match="DerivedFieldCompiler|derived"):
        parse_semantic_field("us_market_cap_daily", {
            "market": "us", "dataset": None,
            "derived_expression": "a.Close * b.shares",
            "mining_allowed": True,
        })
    # derived + mining_allowed=false → 可加载（fail-closed 正确姿势）
    f = parse_semantic_field("us_market_cap_daily", {
        "market": "us", "derived_expression": "a.Close * b.shares",
        "mining_allowed": False,
    })
    assert f.derived_expression and not f.mining_allowed

    # planner：直接读取 derived 字段 → 清晰 ValidationError
    d = tmp_path / "d"
    d.mkdir()
    store = _store(tmp_path, _writable_ds(d, "myds"))
    store.write_arrow("myds", pa.table(
        {"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [1.0]}))
    with pytest.raises(ValidationError, match="derived"):
        store.plan(DataRequest(fields=["us_market_cap_daily"]))


# ---------------------------------------------------------------------------
# 10) 物理列语义歧义 → production 拒绝
# ---------------------------------------------------------------------------

def test_resolve_by_physical_ambiguous_production_reject(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    from data_access.read.semantic_catalog import SemanticFieldCatalog, SemanticField

    cat = SemanticFieldCatalog({
        "close": SemanticField(
            logical_name="close", dataset="us_daily", physical_name="Close",
            market="us"),
        "adjusted_close": SemanticField(
            logical_name="adjusted_close", dataset="us_daily", physical_name="Close",
            market="us", scale=2.0),
    })
    with pytest.raises(AmbiguousSemanticFieldError):
        cat.resolve_by_physical("us_daily", "Close")


# ---------------------------------------------------------------------------
# 11) duplicate Arrow 列单位归一化不丢列
# ---------------------------------------------------------------------------

def test_normalize_units_preserves_duplicate_columns(tmp_path):
    from data_access.read.semantic_catalog import SemanticField, normalize_table_units

    f1 = SemanticField(logical_name="a", dataset="d", physical_name="x",
                       scale=100.0, source_unit="percent", canonical_unit="ratio")
    f2 = SemanticField(logical_name="b", dataset="d", physical_name="x")
    tbl = pa.Table.from_arrays([pa.array([1.0, 2.0]), pa.array([3.0, 4.0])],
                               names=["x", "x"])
    out = normalize_table_units(tbl, [f1, f2], column_of=[f1, f2])
    assert out.num_columns == 2  # 不丢列
    assert out.column(0).to_pylist() == [100.0, 200.0]
    assert out.column(1).to_pylist() == [3.0, 4.0]  # 只归一化 f1 那列
    assert out.column_names == ["x", "x"]


# ---------------------------------------------------------------------------
# 12) FormatSpec typo / 非法 option 类型 → registry load fail
# ---------------------------------------------------------------------------

def test_format_spec_strict_validation(tmp_path):
    from data_access.read.formats import FormatSpec

    with pytest.raises(ValidationError, match="未知 key"):
        FormatSpec.from_yaml({"type": "csv", "delimeter": ","})
    with pytest.raises(ValidationError, match="类型非法"):
        FormatSpec.from_yaml({"type": "csv", "extra": {"sample_size": "many"}})
    with pytest.raises(ValidationError, match="重复"):
        FormatSpec.from_yaml({"type": "csv", "compression": "gzip",
                              "extra": {"compression": "gzip"}})
    with pytest.raises(ValidationError, match="不能是布尔"):
        FormatSpec.from_yaml({"type": "csv", "extra": {"skip": True}})
    # columns 结构值被正确渲染（不 str() 化）
    from data_access.read.formats import get_format_adapter

    spec = FormatSpec.from_yaml(
        {"type": "csv", "extra": {"columns": {"a": "INTEGER"}}})
    sql = get_format_adapter(spec).build_from_clause(
        "x.csv", hive_partitioning=False, union_by_name=False)
    assert "columns={'a': 'INTEGER'}" in sql


# ---------------------------------------------------------------------------
# 13) 写路径统一事务：rebuild 失败不再被吞（write caller 能看见）
# ---------------------------------------------------------------------------

def test_write_surfaces_manifest_rebuild_failure(tmp_path, monkeypatch):
    d = tmp_path / "d"
    d.mkdir()
    store = _store(tmp_path, _writable_ds(d, "myds"))
    store.write_arrow("myds", pa.table(
        {"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [1.0]}))
    store.build_dataset_manifest("myds")

    from data_access.read import manifest as manifest_mod

    real_build = manifest_mod.build_manifest_for_dataset

    def boom(*a, **k):
        raise RuntimeError("manifest rebuild broke")

    monkeypatch.setattr(manifest_mod, "build_manifest_for_dataset", boom)
    try:
        store.write_arrow("myds", pa.table(
            {"ts": [dt.date(2024, 1, 2)], "sym": ["A"], "val": [2.0]}))
        raise SystemExit("FAIL: rebuild failure must surface")
    except RuntimeError:
        pass
