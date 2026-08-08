"""第三轮整改回归：第三层增量问题的正确性锚点。

对应审计清单（P0 1-23 / P1 24-40）在 `6d66e9c` 之上的新修复：
    - #1/#2/#3 组合执行器：effective_join_specs 单一事实源 + result="stream" +
      compiled 不可变语义（explain == execute）
    - #4 plan 后篡改 request 不影响 execute（CompiledDataRequest 冻结）
    - #5 snapshot_policy=fail_if_changed/pin：版本变化拒绝执行
    - #6/#7 glob 冻结成精确文件清单；ScanHandle collect 前 revalidate + strict
      fail-closed（native_lazyframe / lazyframe）
    - #11/#12 日历：UTC→交易所本地时区；非交易日不生成当天开盘
    - #14/#15 契约门升级为谓词约束：Ne/IsNotNull/OR-部分分支不再算「已约束」
    - #17/#19 read_uri：精确 URI 物理范围；格式匹配不再无条件放行 parquet
    - #21/#22 serving 聚合一次扫描 + 行级谓词
    - #28 ScanHandle.native_lazyframe() production/strict fail-closed
    - #32 next_trading_day O(N) zip；#33 bool "false" 正确解析；#34 类型校验
    - #35 ContractIR external；#36 required_filters 完整合并 + 冲突检测
    - #39 StorageSpec(type=非法) 构造即抛
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access import DataRequest
from data_access.core.engine import DuckDBEngine
from data_access.read.predicate_ast import (
    And,
    Eq,
    Gt,
    In,
    IsNotNull,
    Ne,
    Or,
    filter_constraint_status,
    filter_restricts_column,
)
from data_access.read.session_calendar import (
    MarketCalendar,
    build_us_session,
    get_market_calendar,
    reset_calendars,
)
from data_access.read.temporal_join import parse_join_spec
from data_access.registry import load_registry
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _no_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)


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


def _daily_ds(root: Path, name: str = "ashare_stock_daily") -> dict[str, dict]:
    return {
        name: {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(root), "glob": "*.parquet",
            "time_column": "TradeDate", "instrument_column": "Symbol",
            "schema": {"TradeDate": "date", "Symbol": "string",
                       "Close": "double", "Volume": "int"},
        }
    }


# ---------------------------------------------------------------------------
# #14 / #15 谓词约束门
# ---------------------------------------------------------------------------

def test_predicate_restricts_column():
    assert filter_restricts_column(Eq("t", "q"), "t")
    assert filter_restricts_column(In("t", ("q",)), "t")
    assert not filter_restricts_column(Ne("t", "q"), "t")
    assert not filter_restricts_column(IsNotNull("t"), "t")
    assert not filter_restricts_column(Or([Eq("t", "q"), Gt("p", 0)]), "t")
    assert filter_restricts_column(And([Eq("t", "q"), Gt("p", 0)]), "t")
    assert filter_constraint_status(Ne("t", "q"), "t") == "weak"
    assert filter_constraint_status(IsNotNull("t"), "t") == "weak"
    assert filter_constraint_status(Eq("t", "q"), "t") == "positive"
    assert filter_constraint_status(Gt("p", 0), "t") == "absent"


def test_required_filter_weak_forms_rejected(tmp_path, monkeypatch):
    """#14：Ne/IsNotNull/OR-部分分支不能证明维度被限定 → strict 拒绝。"""
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    ind = tmp_path / "i"
    ind.mkdir()
    pq.write_table(
        pa.table({"TradeDate": [dt.date(2024, 1, 1)], "Symbol": ["A"],
                  "IndustrySource": ["sw_l1"], "Industry": ["银行"], "Close": [5.0]}),
        str(ind / "f.parquet"),
    )
    store = _store(tmp_path, {
        "ashare_stock_industry": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(ind), "glob": "*.parquet",
            "time_column": "TradeDate", "instrument_column": "Symbol",
            "schema": {"TradeDate": "date", "Symbol": "string",
                       "IndustrySource": "string", "Industry": "string",
                       "Close": "double"},
        }
    })
    _cols = ["TradeDate", "Symbol", "IndustrySource", "Industry", "Close"]
    # 正向：Eq 约束满足 required_filters + allowed_filter_values
    store.read_arrow(
        "ashare_stock_industry", columns=_cols,
        filters={"IndustrySource": "sw_l1"},
    )
    # Ne 不能证明 → 拒绝
    with pytest.raises(Exception):
        store.read_arrow(
            "ashare_stock_industry", columns=_cols,
            filters={"IndustrySource": {"ne": "sw_l1"}},
        )
    # IsNotNull 不能证明 → 拒绝
    with pytest.raises(Exception):
        store.read_arrow(
            "ashare_stock_industry", columns=_cols,
            filters={"IndustrySource": {"isnotnull": True}},
        )
    # OR 只有一支约束 → 拒绝
    with pytest.raises(Exception):
        store.read_arrow(
            "ashare_stock_industry", columns=_cols,
            filters=Or([Eq("IndustrySource", "sw_l1"), Gt("Close", 0)]),
        )


def test_allowed_filter_values_weak_rejected(tmp_path, monkeypatch):
    """#15：契约维度被弱过滤（Ne）→ 无法证明 result ⊆ allowed → strict 拒绝。"""
    from data_access.read.semantic_catalog import SemanticField

    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    store = _store(tmp_path, {
        "ashare_stock_industry": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(tmp_path / "i"), "glob": "*.parquet",
            "time_column": "TradeDate", "instrument_column": "Symbol",
            "schema": {"TradeDate": "date", "Symbol": "string", "Industry": "string"},
        }
    })
    f = SemanticField(
        logical_name="industry", dataset="ashare_stock_industry",
        physical_name="Industry",
        required_filters=("IndustrySource",),
    )
    # params 满足 required；但 filter 是 Ne → allowed 维度无法证明子集
    with pytest.raises(Exception):
        store._validate_allowed_filter_values(
            [f],
            {"ashare_stock_industry": {"IndustrySource": "sw_l1"}},
            filters=Ne("IndustrySource", "sw_l1"),
        )
    # 正向：Eq 值在 allowed 内 → 通过
    store._validate_allowed_filter_values(
        [f],
        {"ashare_stock_industry": {"IndustrySource": "sw_l1"}},
        filters=Eq("IndustrySource", "sw_l1"),
    )


# ---------------------------------------------------------------------------
# #33 / #34 TemporalJoinSpec 解析硬化
# ---------------------------------------------------------------------------

def test_temporal_join_spec_bool_strings():
    spec = parse_join_spec({
        "policy": "pit_asof", "knowledge_time": "PubDate",
        "deduplicate": "false", "future_cutoff": "false",
    })
    assert spec.deduplicate is False
    assert spec.future_cutoff is False
    spec2 = parse_join_spec({"policy": "pit_asof", "deduplicate": "0"})
    assert spec2.deduplicate is False
    spec3 = parse_join_spec({"policy": "pit_asof", "deduplicate": "true"})
    assert spec3.deduplicate is True


def test_temporal_join_spec_type_validation():
    from data_access.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        parse_join_spec({"revision_order": 2026})
    with pytest.raises(ValidationError):
        parse_join_spec({"primary_key": 7})
    with pytest.raises(ValidationError):
        parse_join_spec({"availability_latency": "abc"})


# ---------------------------------------------------------------------------
# #11 / #12 / #32 交易日历
# ---------------------------------------------------------------------------

def test_calendar_local_timezone_and_non_trading_day():
    reset_calendars()
    days = [dt.date(2024, 1, 4), dt.date(2024, 1, 5), dt.date(2024, 1, 8)]
    cal = MarketCalendar(
        "us", trading_days=days, timezone="America/New_York",
        session=build_us_session(),
    )
    # #11：Mon 13:30 UTC = 08:30 ET（盘前）→ 当日 09:30 开盘，不是下一交易日
    r = cal.available_from(dt.datetime(2024, 1, 8, 13, 30), "next_session_open")
    assert r == dt.datetime(2024, 1, 8, 9, 30)
    # tz-aware UTC 同样转本地
    r_tz = cal.available_from(
        dt.datetime(2024, 1, 8, 13, 30, tzinfo=dt.timezone.utc),
        "next_session_open",
    )
    assert r_tz == dt.datetime(2024, 1, 8, 9, 30)
    # #12：周六 08:00（= 周六 03:00 ET）→ 下周一 09:30，绝不生成「周六开盘」
    r2 = cal.available_from(dt.datetime(2024, 1, 6, 8, 0), "next_session_open")
    assert r2 == dt.datetime(2024, 1, 8, 9, 30)


def test_next_trading_day_join_zip_adjacent():
    days = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 5)]
    cal = MarketCalendar("us", trading_days=days)
    sql = cal.sql_next_trading_day_join()
    assert "(DATE '2024-01-01', DATE '2024-01-02')" in sql
    assert "(DATE '2024-01-02', DATE '2024-01-05')" in sql


# ---------------------------------------------------------------------------
# #17 / #19 read_uri 精确 URI + 格式匹配
# ---------------------------------------------------------------------------

def test_read_uri_physical_scope_exact_file(tmp_path, monkeypatch):
    """#17：strict read_uri 只读指定文件，不读整个 dataset glob。"""
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"TradeDate": [dt.date(2024, 1, 1)], "Symbol": ["A"],
                  "Close": [1.0], "Volume": [10]}),
        str(d / "f1.parquet"),
    )
    pq.write_table(
        pa.table({"TradeDate": [dt.date(2024, 1, 2)], "Symbol": ["B"],
                  "Close": [2.0], "Volume": [20]}),
        str(d / "f2.parquet"),
    )
    store = _store(tmp_path, _daily_ds(d))
    tbl = store.read_uri(
        str(d / "f1.parquet"), format="parquet",
        columns=["TradeDate", "Symbol", "Close", "Volume"],
    ).to_arrow()
    assert tbl.column("TradeDate").to_pylist() == [dt.date(2024, 1, 1)]
    assert tbl.column("Symbol").to_pylist() == ["A"]


def test_read_uri_format_mismatch_not_candidate(tmp_path, monkeypatch):
    """#19：format='parquet' 不会让 CSV 注册数据集成为候选。"""
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    d = tmp_path / "d"
    d.mkdir()
    (d / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    store = _store(tmp_path, {
        "mydata": {
            "kind": "static", "access_mode": "published", "layout": "plain",
            "root": str(d), "glob": "*.csv", "format": "csv",
            "schema": {"a": "int", "b": "int"},
        }
    })
    with pytest.raises(Exception):
        store.read_uri(str(d / "data.csv"), format="parquet")


# ---------------------------------------------------------------------------
# #5 snapshot pin + #4 compiled immutable
# ---------------------------------------------------------------------------

def _writable_ds(root: Path, name: str) -> dict[str, dict]:
    return {
        name: {
            "kind": "static", "access_mode": "staging", "layout": "plain",
            # write_arrow 写 part-*.parquet；glob 只匹配数据文件，避开 _manifest.*
            "root": str(root), "glob": "part-*.parquet",
            "time_column": "ts", "instrument_column": "sym",
            "schema": {"ts": "date", "sym": "string", "val": "double"},
        }
    }


def test_snapshot_pin_fail_if_changed(tmp_path):
    """#5：plan 后数据版本变化 → fail_if_changed 拒绝 execute。"""
    d = tmp_path / "d"
    d.mkdir()
    store = _store(tmp_path, _writable_ds(d, "myds"))
    store.write_arrow("myds", pa.table(
        {"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [1.0]}
    ))
    # 显式启用 manifest（rebuild 只在已有 sidecar 时发生）
    store.build_dataset_manifest("myds")
    req = DataRequest(fields=["ts", "sym", "val"], snapshot_policy="fail_if_changed")
    plan = store.plan(req)
    plan.execute()  # 版本未变 → 正常
    store.touch_manifest_epoch("myds")
    from data_access.core.exceptions import SnapshotBuildError

    with pytest.raises(SnapshotBuildError):
        plan.execute()


def test_compiled_request_immutable(tmp_path):
    """#4：plan 后篡改 request.limit 不影响 execute（消费 compiled）。"""
    d = tmp_path / "d"
    d.mkdir()
    store = _store(tmp_path, _writable_ds(d, "myds"))
    store.write_arrow("myds", pa.table(
        {"ts": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
         "sym": ["A", "B"], "val": [1.0, 2.0]}
    ))
    req = DataRequest(fields=["ts", "sym", "val"], limit=1)
    plan = store.plan(req)
    req.limit = 999  # 调用方在 plan 之后篡改
    tbl = plan.execute().to_arrow()
    assert tbl.num_rows == 1  # 仍是 plan 时的 limit=1


# ---------------------------------------------------------------------------
# #28 ScanHandle governed lazy fail-closed
# ---------------------------------------------------------------------------

def test_scan_handle_native_lazyframe_strict(tmp_path, monkeypatch):
    """#28：production/strict 下 native_lazyframe()/lazyframe() 拒绝裸 LazyFrame。"""
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"ts": [dt.date(2024, 1, 1)], "sym": ["A"], "val": [1.0]}),
        str(d / "part-1.parquet"),
    )
    store = _store(tmp_path, _writable_ds(d, "myds"))
    sh = store.scan("myds", columns=["ts", "sym", "val"])
    from data_access.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        sh.native_lazyframe()
    with pytest.raises(ValidationError):
        sh.lazyframe()
    res = sh.collect()  # 受控 collect 仍可用
    assert res.table.num_rows == 1


# ---------------------------------------------------------------------------
# #21/#22 serving 聚合一次扫描 + 行级谓词
# ---------------------------------------------------------------------------

def test_serving_aggregate_instrument_filter(tmp_path):
    """#22：materialize_daily_aggregate 只聚合请求的 instrument。"""
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(
        pa.table({
            "dt": [dt.date(2024, 1, 1), dt.date(2024, 1, 1),
                   dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "sym": ["A", "B", "A", "B"],
            "val": [1.0, 10.0, 2.0, 20.0],
        }),
        str(src / "f.parquet"),
    )
    svc = tmp_path / "svc"
    svc.mkdir()
    store = _store(tmp_path, {
        **_daily_ds(src, "src_ds"),
        "svc_ds": {
            "kind": "static", "access_mode": "staging", "layout": "plain",
            "root": str(svc), "glob": "*.parquet",
            "time_column": "dt", "instrument_column": "sym",
            "schema": {"dt": "date", "sym": "string", "v": "double"},
        },
    })
    result = store.materialize_daily_aggregate(
        source_dataset="src_ds",
        serving_dataset="svc_ds",
        time_column="dt",
        instrument_column="sym",
        groupby=["dt", "sym"],
        value_columns={"v": {"columns": ["val"], "agg": "sum"}},
        instrument_filter=["A"],
    )
    assert result["rows_computed"] == 2  # 只有 A 的两个日期
    out = store.read_arrow("svc_ds")
    assert out.column("sym").to_pylist() == ["A", "A"]


# ---------------------------------------------------------------------------
# #35 / #36 ContractIR external + required_filters 完整合并
# ---------------------------------------------------------------------------

def test_contract_ir_external_and_required_filters_merge(tmp_path):
    from data_access.cos_contract import COSDatasetContract
    from data_access.read.contract_ir import build_contract_ir

    d = tmp_path / "d"
    d.mkdir()
    store = _store(tmp_path, _daily_ds(d))
    contracts = {
        "external_us_ds": COSDatasetContract(
            "external_us_ds", "us", "E2", "asof", "ticker",
            "event_only", "strict",
            availability_column="filing_date", period_column="period_end",
            required_event_filters=("timeframe",),
            allowed_filter_values=(("timeframe", ("quarterly", "annual")),),
        )
    }
    ir = build_contract_ir(
        store.registry, contracts=contracts, catalog=None,
        external_contract_datasets=["external_us_ds"],
    )
    # #35 external → audit() 不再误报
    assert ir.audit() == []
    entry = ir.get("external_us_ds")
    assert entry.external is True
    # #36 required_event_filters 并入 required_filters
    assert "timeframe" in entry.required_filters
    assert entry.allowed_filter_values.get("timeframe") == ("quarterly", "annual")


# ---------------------------------------------------------------------------
# #39 StorageSpec 非法 type 构造即抛
# ---------------------------------------------------------------------------

def test_storage_spec_invalid_type_fails_closed():
    from data_access.core.exceptions import ValidationError
    from data_access.core.storage import StorageSpec

    with pytest.raises(ValidationError):
        StorageSpec(type="cosss")
    # 合法类型不受影响
    assert StorageSpec(type="cos").backend.value == "cos"


# ---------------------------------------------------------------------------
# #2 组合执行器 honor result="stream"
# ---------------------------------------------------------------------------

def test_composed_result_stream(tmp_path):
    """#2：聚合 + join 组合请求 result='stream' 返回流式句柄（不物化整表）。"""
    from data_access.read.aggregation import AggregationItem, AggregationSpec

    minute = tmp_path / "minute"
    fin = tmp_path / "fin"
    minute.mkdir()
    fin.mkdir()
    rows = []
    for day in (1, 2):
        for s in ("A", "B"):
            for i in range(2):
                rows.append((dt.datetime(2024, 1, day, 5 + i, 0), s, float(day), i))
    pq.write_table(
        pa.table({"ts": [r[0] for r in rows], "inst": [r[1] for r in rows],
                  "Close": [r[2] for r in rows], "Volume": [r[3] for r in rows]}),
        str(minute / "m.parquet"),
    )
    pq.write_table(
        pa.table({"financial_time": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)] * 2,
                  "inst": ["A", "A", "B", "B"], "Eps": [1.0, 2.0, 3.0, 4.0]}),
        str(fin / "f.parquet"),
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
            },
        },
    )
    spec = AggregationSpec(aggregation="minute_range", start="13:00", end="15:00",
                           metric="sum", market="ashare")
    req = DataRequest(
        fields=["minute_ds.Volume", "fin_ds.Eps"],
        start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2),
        instruments=["A", "B"], anchor="minute_ds",
        aggregations=[AggregationItem("Volume", spec, "total_vol")],
        joins={"fin_ds": {"policy": "asof", "knowledge_time": "financial_time"}},
        result="stream",
    )
    handle = store.plan(req).execute()
    batches = list(handle.stream())
    tbl = pa.Table.from_batches(batches)
    assert tbl.num_rows == 4  # 2 天 × 2 标的


# ---------------------------------------------------------------------------
# #6 冻结 glob + #30 格式感知路由 + #40 能力矩阵
# ---------------------------------------------------------------------------

def test_expand_glob_paths_frozen(tmp_path):
    """#6：glob 展开一次成精确文件清单，不再二次 mutable glob。"""
    d = tmp_path / "d"
    d.mkdir()
    for n in ("a.parquet", "b.parquet"):
        (d / n).write_bytes(b"")
    store = _store(tmp_path, {})
    frozen = store._expand_glob_paths([str(d / "*.parquet")])
    assert frozen == sorted([str(d / "a.parquet"), str(d / "b.parquet")])
    # 目录下新增文件不影响已经冻结的清单
    (d / "c.parquet").write_bytes(b"")
    assert store._expand_glob_paths(frozen) == frozen


def test_suggest_read_strategy_arrow_to_pyarrow():
    """#30：arrow/feather 数据集 auto 路由必须选 pyarrow（不能 scan_parquet）。"""
    from data_access.read.scan_cost import ScanCost, suggest_read_strategy

    cost = ScanCost(
        dataset="x", file_count=1, total_bytes=100, estimated_rows=100,
        projected_columns=2, total_columns=2, remote=False,
        file_format="arrow",
    )
    engine, result = suggest_read_strategy(cost, engine="auto", result="auto")
    assert engine == "pyarrow"
    cost2 = ScanCost(
        dataset="x", file_count=1, total_bytes=100, estimated_rows=100,
        projected_columns=2, total_columns=2, remote=False,
        file_format="parquet",
    )
    assert suggest_read_strategy(cost2, engine="auto", result="auto")[0] == "duckdb"


def test_storage_capability_matrix():
    """#40：后端能力矩阵如实声明；未实现后端 fail-closed。"""
    from data_access.core.storage import (
        StorageBackend,
        backend_readable,
        storage_backend_capabilities,
    )

    assert storage_backend_capabilities(StorageBackend.COS).authorization == "s3_prefix"
    assert storage_backend_capabilities(StorageBackend.LOCAL).authorization == "local_authorizer"
    assert backend_readable(StorageBackend.HTTP) is False
    assert backend_readable(StorageBackend.COS) is True
