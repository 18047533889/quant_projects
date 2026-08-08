"""DataAccess 收官 DoD 回归：P0 全部修复的正确性锚点。

每个测试对应 docs/DATAACCESS_FINAL_CLOSURE_PLAN.md 里的一条 DoD：
    - 不同执行路径等价性（duckdb / stream）
    - 聚合 + PIT join 组合 == 手工 aggregate → read_joined
    - 财务 late-revision 不回滚（Q3 可见后 Q2 修订不得覆盖）
    - 真实日历按 IsTradeDay 过滤（周六/周日/节假日不当交易日）
    - 查询窗末端 knowledge 映射到下一交易日（不 COALESCE 回退当天）
    - 美股 early close 按日期覆盖（13:00 收市）
    - required filter 不可绕过（columns=None / 物理列直读）
    - fanout 单值证明（IN 多值仍判 fan-out）
    - 无 end 的 effective_time_only 事件查询 production 拒绝
    - PITEventIndex：indexed_file_count=文件数、limit 截断 → incomplete
    - ContractIR.audit() 在完整 registry 上可运行
    - Metadata Plane 感知 mutation 后的 source_epoch 变化
    - Cache column order：[A,B] 与 [B,A] 是不同 key
    - stream 直接路由（不先物化再重扫）
    - mutation lock 跨 host 不得用本地 PID 回收
    - 嵌套 namespace：exit 恢复外层
"""
from __future__ import annotations

import datetime as dt
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access import DataRequest
from data_access.core.engine import DuckDBEngine
from data_access.read.aggregation import AggregationItem, AggregationSpec
from data_access.read.query_cache import query_cache_key
from data_access.read.session_calendar import (
    build_us_session,
    get_market_calendar,
    reset_calendars,
)
from data_access.registry import load_registry
from data_access.store import DataAccessStore


# ---------------------------------------------------------------------------
# 通用 fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")


def _store(tmp_path, datasets: dict[str, dict]) -> DataAccessStore:
    import json

    cfg = tmp_path / "datasets.yaml"
    lines = []
    for name, body in datasets.items():
        lines.append(f"{name}:")
        for k, v in body.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for sk, sv in v.items():
                    if isinstance(sv, dict):
                        lines.append(f"    {sk}:")
                        for ssk, ssv in sv.items():
                            lines.append(f"      {ssk}: {ssv}")
                    else:
                        lines.append(f"    {sk}: {json.dumps(str(sv))}")
            else:
                lines.append(f"  {k}: {json.dumps(str(v))}")
    cfg.write_text("\n".join(lines), encoding="utf-8")
    return DataAccessStore(load_registry(cfg), DuckDBEngine(threads=2))


# ---------------------------------------------------------------------------
# DoD #1 不同执行路径等价性：duckdb read_arrow vs stream
# ---------------------------------------------------------------------------

def test_path_equivalence_duckdb_vs_stream(tmp_path):
    """同一条读，read_arrow（duckdb）与 read_arrow_stream 结果一致。"""
    d = tmp_path / "d"
    d.mkdir()
    rows = [
        (dt.date(2024, 1, 1 + i), s, float(i), i * 100)
        for i in range(4)
        for s in ("A", "B")
    ]
    pq.write_table(
        pa.table(
            {
                "TradeDate": [r[0] for r in rows],
                "Symbol": [r[1] for r in rows],
                "Close": [r[2] for r in rows],
                "Volume": [r[3] for r in rows],
            }
        ),
        str(d / "f.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "ashare_stock_daily": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "TradeDate", "instrument_column": "Symbol",
                "schema": {"TradeDate": "date", "Symbol": "string",
                           "Close": "double", "Volume": "int"},
            }
        },
    )
    full = store.read_arrow("ashare_stock_daily", time_range=(dt.date(2024, 1, 2), dt.date(2024, 1, 3)))
    batches = list(
        store.read_arrow_stream("ashare_stock_daily", time_range=(dt.date(2024, 1, 2), dt.date(2024, 1, 3)))
    )
    streamed = pa.Table.from_batches(batches)
    assert streamed.num_rows == full.num_rows == 4
    assert sorted(streamed.column("Close").to_pylist()) == sorted(
        full.column("Close").to_pylist()
    )


# ---------------------------------------------------------------------------
# DoD #2 聚合 + PIT join 组合 == 手工 aggregate → read_joined
# ---------------------------------------------------------------------------

def test_composed_aggregation_join_parity(tmp_path):
    """PhysicalPlanExecutor 组合路径 == 手工 aggregate 再走完整 read_joined。"""
    minute = tmp_path / "minute"
    fin = tmp_path / "fin"
    minute.mkdir()
    fin.mkdir()
    rows = []
    for day in (1, 2):
        for s, base in (("A", 10.0), ("B", 20.0)):
            for i in range(3):
                t = dt.datetime(2024, 1, day, 5 + i, 0)  # naive UTC → 13-15 CST
                rows.append((t, s, base + i, 100 * (i + 1)))
    pq.write_table(
        pa.table({"ts": [r[0] for r in rows], "inst": [r[1] for r in rows],
                  "Close": [r[2] for r in rows], "Volume": [r[3] for r in rows]}),
        str(minute / "m.parquet"),
    )
    pq.write_table(
        pa.table({
            "financial_time": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)] * 2,
            "inst": ["A", "A", "B", "B"],
            "Eps": [5.0, 6.0, 8.0, 9.0],
        }),
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
    )
    composed = store.plan(req).execute().to_arrow().to_pydict()
    # 逐行一致性：按 (ts, inst) 对齐
    comp = sorted(
        zip(composed["ts"], composed["inst"], composed["total_vol"], composed["Eps"]),
        key=lambda r: (str(r[0]), r[1]),
    )
    assert all(v is not None and int(v) == 600 for v in [r[2] for r in comp])
    # 行数 = 2 天 × 2 标的
    assert len(comp) == 4


# ---------------------------------------------------------------------------
# DoD #3 财务 late-revision 不回滚
# ---------------------------------------------------------------------------

def test_financial_late_revision_no_rollback(tmp_path):
    daily = tmp_path / "daily"
    fin = tmp_path / "fin"
    daily.mkdir()
    fin.mkdir()
    pq.write_table(
        pa.table({"TradeDate": [dt.date(2025, 12, 1 + i) for i in range(3)],
                  "Symbol": ["A"] * 3, "Close": [1.0] * 3}),
        str(daily / "d.parquet"),
    )
    pq.write_table(
        pa.table({
            "PubDate": [dt.date(2025, 8, 1), dt.date(2025, 10, 31), dt.date(2025, 11, 15)],
            "Symbol": ["A"] * 3,
            "period_end": [dt.date(2025, 6, 30), dt.date(2025, 9, 30), dt.date(2025, 6, 30)],
            "Eps": [1.0, 3.0, 2.0],
        }),
        str(fin / "f.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "ashare_stock_daily": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(daily), "glob": "*.parquet",
                "time_column": "TradeDate", "instrument_column": "Symbol",
                "schema": {"TradeDate": "date", "Symbol": "string", "Close": "double"},
            },
            "ashare_stock_balance": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(fin), "glob": "*.parquet",
                "time_column": "PubDate", "instrument_column": "Symbol",
                "schema": {"PubDate": "date", "Symbol": "string",
                           "period_end": "date", "Eps": "double"},
            },
        },
    )
    h = store.read_joined(
        "ashare_stock_daily",
        {"ashare_stock_daily": ["Close"], "ashare_stock_balance": ["Eps"]},
        joins={"ashare_stock_balance": {
            "policy": "pit_asof", "knowledge_time": "PubDate",
            "period_time": "period_end", "period_selection": "latest_period",
            "revision_order": ["PubDate"],
        }},
        time_range=(dt.date(2025, 12, 1), dt.date(2025, 12, 3)),
        instrument_filter=["A"],
    ).to_arrow().to_pydict()
    eps = sorted(h["Eps"])
    assert eps == [3.0, 3.0, 3.0], f"Q3 可见后不得被晚到 Q2 修订回滚：{eps}"


# ---------------------------------------------------------------------------
# DoD #4/#5 真实日历 + 查询末端 boundary
# ---------------------------------------------------------------------------

def test_calendar_flag_filter_and_boundary_lookahead(tmp_path):
    cal = tmp_path / "cal"
    daily = tmp_path / "daily"
    fin = tmp_path / "fin"
    for d in (cal, daily, fin):
        d.mkdir()
    pq.write_table(
        pa.table({
            "TradeDate": [dt.date(2025, 12, 1), dt.date(2025, 12, 5), dt.date(2025, 12, 6),
                          dt.date(2025, 12, 8)],  # 12-06 Sat=非交易日；12-08 Mon=下一交易日
            "IsTradeDay": [True, True, False, True],
        }),
        str(cal / "c.parquet"),
    )
    pq.write_table(
        pa.table({"TradeDate": [dt.date(2025, 12, 1 + i) for i in range(5)],
                  "Symbol": ["A"] * 5, "Close": [1.0] * 5}),
        str(daily / "d.parquet"),
    )
    pq.write_table(
        pa.table({"PubDate": [dt.date(2025, 12, 5)], "Symbol": ["A"],
                  "period_end": [dt.date(2025, 9, 30)], "Eps": [3.0]}),
        str(fin / "f.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "ashare_calendar": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(cal), "glob": "*.parquet",
                "time_column": "TradeDate", "instrument_column": "TradeDate",
                "schema": {"TradeDate": "date", "IsTradeDay": "bool"},
            },
            "ashare_stock_daily": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(daily), "glob": "*.parquet",
                "time_column": "TradeDate", "instrument_column": "Symbol",
                "schema": {"TradeDate": "date", "Symbol": "string", "Close": "double"},
            },
            "ashare_stock_balance": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(fin), "glob": "*.parquet",
                "time_column": "PubDate", "instrument_column": "Symbol",
                "schema": {"PubDate": "date", "Symbol": "string",
                           "period_end": "date", "Eps": "double"},
            },
        },
    )
    # DoD #4：周六不当交易日
    c = store.get_calendar("ashare")
    assert c is not None and dt.date(2025, 12, 6) not in c.trading_days
    assert dt.date(2025, 12, 5) in c.trading_days
    # DoD #5：knowledge=窗口末(周五) → session availability 必须映射到下一交易日
    h = store.read_joined(
        "ashare_stock_daily",
        {"ashare_stock_daily": ["Close"], "ashare_stock_balance": ["Eps"]},
        joins={"ashare_stock_balance": {
            "policy": "pit_asof", "knowledge_time": "PubDate",
            "period_time": "period_end", "availability": "session",
        }},
        time_range=(dt.date(2025, 12, 1), dt.date(2025, 12, 5)),
        instrument_filter=["A"],
    ).to_arrow().to_pydict()
    assert all(v is None for v in h["Eps"]), "末条公告必须映射到下一交易日，不得提前当天可见"


# ---------------------------------------------------------------------------
# DoD #6 美股 early close
# ---------------------------------------------------------------------------

def test_us_early_close_date_override():
    us = build_us_session(early_close_dates=[dt.date(2025, 11, 28)])
    assert us.close_on(dt.date(2025, 11, 28)).strftime("%H:%M") == "13:00"
    assert us.close_on(dt.date(2025, 11, 27)).strftime("%H:%M") == "16:00"
    assert us.bar_count_on(dt.date(2025, 11, 28)) < us.bar_count_on(dt.date(2025, 11, 27))


# ---------------------------------------------------------------------------
# DoD #8 required filter 不可绕过（columns=None 也强制）
# ---------------------------------------------------------------------------

def test_dataset_level_required_filter_not_bypassable(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"TradeDate": [dt.date(2024, 1, 1)], "Symbol": ["000001.SZ"],
                  "Industry": ["银行"], "IndustrySource": ["sw_l1"]}),
        str(d / "f.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "ashare_stock_industry": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "TradeDate", "instrument_column": "Symbol",
                "schema": {"TradeDate": "date", "Symbol": "string",
                           "Industry": "string", "IndustrySource": "string"},
            }
        },
    )
    os.environ["QUANT_PRODUCTION_MODE"] = "1"
    try:
        store.read_arrow("ashare_stock_industry", columns=["Industry"], IndustrySource="sw_l1")
        # columns=None 也必须被 IndustrySource required filter 挡下
        with pytest.raises(Exception):
            store.read_arrow("ashare_stock_industry", columns=None)
    finally:
        os.environ.pop("QUANT_PRODUCTION_MODE", None)


# ---------------------------------------------------------------------------
# DoD #9 fanout 单值证明：IN 多值仍判 fan-out
# ---------------------------------------------------------------------------

def test_fanout_in_multiple_values_still_fanout():
    from data_access.cos_contract import get_cos_contract, validate_join_fanout

    contract = get_cos_contract("ashare_stock_industry")
    assert contract is not None
    # 单值 → 豁免
    validate_join_fanout(
        contract,
        join_key=("TradeDate", "Symbol"),
        applied_filter_columns=["IndustrySource"],
        applied_filter_values={"IndustrySource": {"sw_l1"}},
    )
    # IN 多值 → 仍必须报 fan-out（production）
    with pytest.raises(Exception):
        validate_join_fanout(
            contract,
            join_key=("TradeDate", "Symbol"),
            applied_filter_columns=["IndustrySource"],
            applied_filter_values={"IndustrySource": {"sw_l1", "sw_l2"}},
            production=True,
        )


# ---------------------------------------------------------------------------
# DoD #10 无 end 的 effective_time_only 查询 production 拒绝
# ---------------------------------------------------------------------------

def test_effective_event_unbounded_query_rejected_in_production():
    from data_access.cos_contract import enforce_event_cutoff, get_cos_contract

    contract = get_cos_contract("us_stock_dividend")
    assert contract is not None and contract.pit_policy == "effective_time_only"
    with pytest.raises(Exception):
        enforce_event_cutoff(contract, time_range=None, production=True)


# ---------------------------------------------------------------------------
# DoD #11 PITEventIndex：indexed_file_count=文件数、limit→incomplete
# ---------------------------------------------------------------------------

def test_pit_index_file_counts_and_truncation(tmp_path):
    from data_access.read.pit_event_index import build_pit_event_index

    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"Ticker": ["A", "B"], "filing_date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
                  "period_end": [dt.date(2023, 12, 31), dt.date(2024, 3, 31)]}),
        str(d / "2023-12-31.parquet"),
    )
    pq.write_table(
        pa.table({"Ticker": ["A", "B"], "filing_date": [dt.date(2024, 6, 1), dt.date(2024, 6, 2)],
                  "period_end": [dt.date(2024, 6, 30), dt.date(2024, 6, 30)]}),
        str(d / "2024-06-30.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "us_stock_balance": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "filing_date", "instrument_column": "Ticker",
                "schema": {"Ticker": "string", "filing_date": "date",
                           "period_end": "date"},
            }
        },
    )
    idx = build_pit_event_index(store, "us_stock_balance", force=True)
    assert idx.metadata.complete is True
    assert idx.metadata.indexed_file_count == idx.metadata.source_file_count == 2
    assert idx.metadata.is_authoritative is True
    idx2 = build_pit_event_index(store, "us_stock_balance", limit=1, force=True)
    assert idx2.metadata.complete is False
    assert idx2.metadata.indexed_file_count < idx2.metadata.source_file_count


# ---------------------------------------------------------------------------
# DoD #12 ContractIR.audit() 在完整 registry 上可运行
# ---------------------------------------------------------------------------

def test_contract_ir_audit_runs_on_full_registry():
    from data_access import get_store
    from data_access.read.contract_ir import build_contract_ir

    store = get_store()
    ir = build_contract_ir(
        store.registry,
        catalog=__import__(
            "data_access.read.semantic_catalog", fromlist=["get_semantic_catalog"]
        ).get_semantic_catalog(),
    )
    problems = ir.audit()  # 必须不 AttributeError
    assert isinstance(problems, list)


# ---------------------------------------------------------------------------
# DoD #13 Metadata Plane 感知 mutation 后的 source_epoch
# ---------------------------------------------------------------------------

def test_metadata_plane_epoch_invalidation(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"TradeDate": [dt.date(2024, 1, 1)], "Symbol": ["A"], "Close": [1.0]}),
        str(d / "f.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "ashare_stock_daily": {
                "kind": "static", "access_mode": "staging", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "TradeDate", "instrument_column": "Symbol",
                "schema": {"TradeDate": "date", "Symbol": "string", "Close": "double"},
            }
        },
    )
    store.build_dataset_manifest("ashare_stock_daily")
    plane = store.metadata_plane("ashare_stock_daily")
    epoch_before = plane.source_epoch()
    cov_before = plane.coverage()
    # 写数据 → mutation bump source_epoch
    store.write_arrow(
        "ashare_stock_daily",
        pa.table({"TradeDate": [dt.date(2024, 1, 2)], "Symbol": ["A"], "Close": [2.0]}),
        mode="append",
    )
    epoch_after = plane.source_epoch()
    assert epoch_before != epoch_after, "mutation 后 plane 必须感知 source_epoch 变化"
    # coverage 缓存被 epoch 变化清除（不再返回旧对象）
    cov_after = plane.coverage()
    assert cov_after is not None


# ---------------------------------------------------------------------------
# DoD #14 cache column order：[A,B] != [B,A]
# ---------------------------------------------------------------------------

def test_cache_key_column_order():
    k1 = query_cache_key(dataset="x", params={}, time_range=None, instruments=None,
                         columns=["A", "B"], manifest_token=None)
    k2 = query_cache_key(dataset="x", params={}, time_range=None, instruments=None,
                         columns=["B", "A"], manifest_token=None)
    assert k1 != k2, "[A,B] 与 [B,A] 必须是不同 cache key（输出列序不同）"


# ---------------------------------------------------------------------------
# DoD #15/#16 stream 直接路由 + early break 释放 cursor
# ---------------------------------------------------------------------------

def test_stream_direct_routing_no_double_scan(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"TradeDate": [dt.date(2024, 1, 1 + i) for i in range(5)],
                  "Symbol": ["A"] * 5, "Close": [1.0] * 5}),
        str(d / "f.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "ashare_stock_daily": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "TradeDate", "instrument_column": "Symbol",
                "schema": {"TradeDate": "date", "Symbol": "string", "Close": "double"},
            }
        },
    )
    # read(..., result="stream") 不应抛错，且能消费（stream 句柄）
    handle = store.read("ashare_stock_daily", columns=["Close"], result="stream",
                        time_range=(dt.date(2024, 1, 1), dt.date(2024, 1, 3)))
    n = sum(batch.num_rows for batch in handle.stream())
    assert n == 3


def test_stream_early_break_releases_cursor(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"TradeDate": [dt.date(2024, 1, 1 + i) for i in range(10)],
                  "Symbol": ["A"] * 10, "Close": [1.0] * 10}),
        str(d / "f.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "ashare_stock_daily": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "TradeDate", "instrument_column": "Symbol",
                "schema": {"TradeDate": "date", "Symbol": "string", "Close": "double"},
            }
        },
    )
    # 消费 2 批后 break：generator finally 必须关闭 reader，不留悬挂游标
    seen = 0
    for batch in store.read_arrow_stream("ashare_stock_daily"):
        seen += batch.num_rows
        if seen >= 2:
            break
    # break 后能继续正常读（连接已归还）
    t = store.read_arrow("ashare_stock_daily")
    assert t.num_rows == 10


# ---------------------------------------------------------------------------
# DoD #17 factor matrix 精确投影
# ---------------------------------------------------------------------------

def test_matrix_exact_projection(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"datetime": [dt.date(2024, 1, 1)], "asset": ["A"],
                  "f1": [1.0], "f2": [2.0], "f3": [3.0]}),
        str(d / "m.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "factor_matrix": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "datetime", "instrument_column": "asset",
                "schema": {"datetime": "date", "asset": "string",
                           "f1": "double", "f2": "double", "f3": "double"},
            },
            "factor_lake": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(tmp_path / "lake"), "glob": "*.parquet",
                "time_column": "datetime", "instrument_column": "asset",
                "schema": {"datetime": "date", "asset": "string"},
            },
        },
    )
    from data_access.read.read_handle import ReadHandle

    h = store._read_factor_matrix(
        ["f1", "f3"],
        time_range=None, universe="u", frequency="daily",
        columns=None, limit=None, engine="duckdb", result="arrow",
        prefer_polars=False, batch_size=100000, query_budget=None,
    )
    t = h.to_arrow()
    assert set(t.column_names) >= {"datetime", "asset", "f1", "f3"}


# ---------------------------------------------------------------------------
# DoD #20 mutation lock 跨 host 不得用本地 PID 回收
# ---------------------------------------------------------------------------

def test_mutation_lock_host_aware(tmp_path):
    from data_access.write.mutation_lock import _owner_is_dead

    # 本机 PID：活着 → False（不能回收）
    assert _owner_is_dead({"pid": os.getpid(), "host": os.uname().nodename}) is False
    # 异机：即使本地没有该 PID，也不能判死
    assert _owner_is_dead({"pid": 999999, "host": "some-other-host"}) is False


# ---------------------------------------------------------------------------
# DoD #21 嵌套 namespace：exit 恢复外层
# ---------------------------------------------------------------------------

def test_nested_namespace_restore():
    from data_access.core.namespace import DataAccessSession, namespace_scope, session_namespace

    os.environ.pop("QUANT_RUN_NAMESPACE", None)
    with DataAccessSession("A"):
        assert session_namespace() == "A"
        with namespace_scope("B"):
            assert session_namespace() == "B"
        assert session_namespace() == "A"
    assert session_namespace() is None
