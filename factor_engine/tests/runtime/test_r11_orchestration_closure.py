# -*- coding: utf-8 -*-
"""R11 收官批：增量调度 / 依赖 Catalog / Factor Identity / factor_matrix 8 项。

覆盖（外部 AI 复查 #1-#8）：
    #1  incremental_event_service.normalize_data_event 别名修复（不再自递归）。
    #2  scheduler 以 affected_start/affected_end 为修订重算窗口（而非 updated_date）。
    #3  deleted_keys → FactorUpdatePlan → tombstone（materializer 落盘为 NaN）。
    #4  dependency edge 以 resolved physical dataset 为 source_dataset（非 anchor）。
    #5  成功 materialize 原子持久化 full factor definition。
    #6  register() factor_version = semantic identity digest；production 冲突拒绝。
    #7  factor_matrix 分区 read-merge-write（partial 因子/日期不再覆盖整月）。
    #8  record_factor_manifest 单事务原子替换 + delete_factor 补删新表。

注意：本文件刻意不 import ``api.dsl_parser`` / 不解析因子公式——当前树里并发
会话的 cleaned_operators WIP 会让 ``load_all()`` 抛 R4-100 arity 审计错误。
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from runtime.dependency_catalog import DependencyCatalog, FactorDependencyEdge
from runtime.incremental_scheduler import (
    DataEvent,
    _coerce_deleted_keys,
    _edges_from_analysis,
    plan_updates_from_data_event,
)
from storage.catalog import FactorCatalog
from storage.exceptions import FactorSemanticIdentityMismatchError
from storage.materializer import ParquetMaterializer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ser(data: dict[tuple, float]) -> pd.Series:
    idx = pd.MultiIndex.from_tuples(
        list(data.keys()), names=["datetime", "instrument"]
    )
    return pd.Series(list(data.values()), index=idx, dtype="float64")


def _mk_catalog(tmp_path) -> tuple[FactorCatalog, DependencyCatalog]:
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    return catalog, DependencyCatalog(catalog)


def _register(catalog: FactorCatalog, fid: str, *, version: str | None = None) -> None:
    catalog.register(
        fid,
        author="tester",
        frequency="1d",
        ast_hash="h_" + fid,
        expression='col("close")',
        semantic_identity_digest=version,
    )


# ---------------------------------------------------------------------------
# #1 incremental_event_service 别名修复
# ---------------------------------------------------------------------------


def test_event_service_normalize_no_recursion():
    from runtime.incremental_event_service import normalize_data_event

    ev = normalize_data_event(
        {"dataset": "d", "column": "c", "updated_date": "2026-08-09"}
    )
    assert ev.dataset == "d"
    assert ev.updated_date == "2026-08-09"


# ---------------------------------------------------------------------------
# #2 affected_start/end 修订重算窗口
# ---------------------------------------------------------------------------


def test_plan_window_uses_affected_dates(tmp_path):
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_hist")
    dep.record_factor_edges(
        "f_hist",
        edges=[FactorDependencyEdge("f_hist", "ashare_stock_daily", "close")],
        lookback=2,
        frequency="1d",
        source_dataset="ashare_stock_daily",
    )
    event = DataEvent(
        dataset="ashare_stock_daily",
        column="close",
        updated_date="2026-08-09",  # 事件抵达日期
        field_id="close",
        revision_kind="revision",
        affected_start="2024-06-01",  # 实际受影响数据
        affected_end="2024-06-15",
    )
    plans = plan_updates_from_data_event(dep, event, lookback_extra=0)
    assert len(plans) == 1
    plan = plans[0]
    # 重算窗口从 affected_start 起，而不是 updated_date
    assert plan.since == "2024-06-01"
    assert plan.end_date == "2024-06-15"


def test_plan_window_backward_compat_uses_updated_date(tmp_path):
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_plain")
    dep.record_factor_edges(
        "f_plain",
        edges=[FactorDependencyEdge("f_plain", "ds_x", "close")],
        lookback=1,
        frequency="1d",
        source_dataset="ds_x",
    )
    plans = plan_updates_from_data_event(
        dep, DataEvent(dataset="ds_x", column="close", updated_date="2026-07-09")
    )
    assert plans[0].since == "2026-07-09"
    assert plans[0].end_date is None  # 无 affected_end → 传播到最新


# ---------------------------------------------------------------------------
# #3 deleted_keys → FactorUpdatePlan → tombstone
# ---------------------------------------------------------------------------


def test_coerce_deleted_keys_pairs_and_bare():
    ev = DataEvent(
        dataset="d",
        column="c",
        updated_date="2026-08-09",
        affected_start="2024-06-01",
        deleted_keys=(("2024-06-01", "AAA"), "BBB"),
    )
    keys = _coerce_deleted_keys(ev)
    assert keys == (("2024-06-01", "AAA"), ("2024-06-01", "BBB"))
    # 无 deleted_keys → None
    assert _coerce_deleted_keys(DataEvent("d", "c", "2026-08-09")) is None


def test_plan_carries_deleted_keys(tmp_path):
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_del")
    dep.record_factor_edges(
        "f_del",
        edges=[FactorDependencyEdge("f_del", "ds_x", "close")],
        lookback=1,
        frequency="1d",
        source_dataset="ds_x",
    )
    event = DataEvent(
        dataset="ds_x",
        column="close",
        updated_date="2026-08-09",
        deleted_keys=(("2024-06-01", "AAA"),),
    )
    plans = plan_updates_from_data_event(dep, event)
    assert plans[0].deleted_keys == (("2024-06-01", "AAA"),)
    d = plans[0].to_dict()
    assert d["deleted_keys"] == [["2024-06-01", "AAA"]]


def test_materializer_deleted_keys_writes_tombstone(tmp_path):
    m = ParquetMaterializer(lake_root=tmp_path / "lake")
    d1 = pd.Timestamp("2024-01-02")
    m.materialize(
        "f_tomb",
        _ser({(d1, "AAA"): 1.23, (d1, "BBB"): 2.5}),
        frequency="1d",
        ast_hash="a" * 64,
        author="tester",
    )
    # 第二次：增量重算只覆盖 AAA，BBB 被源删除 → deleted_keys 转 tombstone
    m.materialize(
        "f_tomb",
        _ser({(d1, "AAA"): 3.0}),
        frequency="1d",
        ast_hash="a" * 64,
        author="tester",
        deleted_keys=[(d1, "BBB")],
    )
    frame = _read_factor_lake(tmp_path / "lake", "f_tomb")
    row_aaa = frame[frame["asset"] == "AAA"]
    row_bbb = frame[frame["asset"] == "BBB"]
    assert float(row_aaa["value"].iloc[0]) == 3.0
    # BBB 已被 tombstone（NaN / is_valid=0），而不是残留 2.5
    assert pd.isna(float(row_bbb["value"].iloc[0]))
    assert int(row_bbb["is_valid"].iloc[0]) == 0


def _read_factor_lake(lake_root: Path, fid: str) -> pd.DataFrame:
    parts = sorted((lake_root / "factors" / fid).rglob("data.parquet"))
    assert parts, "factor lake 无分区文件"
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


# ---------------------------------------------------------------------------
# #4 dependency edge 用 resolved physical dataset
# ---------------------------------------------------------------------------


def test_edges_from_analysis_use_physical_dataset():
    spec = SimpleNamespace(
        dataset="ashare_stock_valuation_daily",  # DataAccess 物理 dataset
        table="StockValuationDaily",  # FactorEngine 逻辑表
        field_id="pe_ratio",
        source_name="PeRatio",
    )
    analysis = SimpleNamespace(
        referenced_fields={"pe_ratio": spec},
        referenced_columns={"pe_ratio"},
        lookback=3,
    )
    data_source = SimpleNamespace(dataset="ashare_stock_daily")  # anchor 不同
    edges = _edges_from_analysis("composite_f", analysis, data_source)
    assert len(edges) == 1
    edge = edges[0]
    # source_dataset 必须是物理 dataset，不是 anchor
    assert edge.source_dataset == "ashare_stock_valuation_daily"
    assert edge.logical_table == "StockValuationDaily"
    assert edge.field_id == "pe_ratio"
    assert edge.physical_field == "PeRatio"


def test_edges_fallback_when_no_dataset():
    spec = SimpleNamespace(
        dataset=None,
        table="StockDailyBar",
        field_id="close",
        source_name="Close",
    )
    analysis = SimpleNamespace(referenced_fields={"close": spec})
    data_source = SimpleNamespace(dataset="ashare_stock_daily")
    edges = _edges_from_analysis("f", analysis, data_source)
    # #收官轮 P0：spec.dataset 缺失 → TableSpec(spec.table).dataset 解析成物理
    # dataset（ashare_stock_daily），**绝不写 logical table name**（否则 DataEvent
    # 匹配不上、增量重算失效）；anchor 只兜底。
    assert edges[0].source_dataset == "ashare_stock_daily"
    assert edges[0].logical_table == "StockDailyBar"


def test_edges_unresolved_logical_table_skips_research_fails_production():
    """P0-40/P0-41：逻辑表名解析不到物理 dataset 时**绝不 anchor-fallback**。

    anchor-fallback 会让 valuation 事件永远匹配不到依赖它的因子（增量重算静默
    失效）——旧实现「回落 anchor」在此被废止。research 跳过该 edge 并 warning；
    production fail-closed 抛 ``DependencyDatasetResolutionError``。
    """
    spec = SimpleNamespace(
        dataset=None,
        table="UnknownLogicalTable999",
        field_id="close",
        source_name="Close",
    )
    analysis = SimpleNamespace(referenced_fields={"close": spec})
    data_source = SimpleNamespace(dataset="ashare_stock_daily")
    # research：edge 被跳过（不能写错的 source_dataset）
    edges = _edges_from_analysis("f", analysis, data_source, production=False)
    assert edges == []
    # production：fail-closed
    from runtime.incremental_scheduler import DependencyDatasetResolutionError

    with pytest.raises(DependencyDatasetResolutionError):
        _edges_from_analysis("f", analysis, data_source, production=True)


# ---------------------------------------------------------------------------
# #5 full factor definition 原子持久化
# ---------------------------------------------------------------------------


def test_record_factor_manifest_writes_full_definition(tmp_path):
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_full", version="digest_v1")
    dep.record_factor_manifest(
        "f_full",
        edges=[FactorDependencyEdge("f_full", "ashare_stock_daily", "close")],
        referenced_columns=["close"],
        lookback=5,
        frequency="1d",
        source_dataset="ashare_stock_daily",
        full_definition={
            "factor_id": "f_full",
            "expression": 'col("close")',
            "surface": "lqtp",
            "dialect": "lqtp",
            "market": "A",
            "universe": "CSI300",
            "run_mode": "production",
            "backend": "hybrid",
            "pit_enforce": True,
            "data_source_config": {"type": "parquet", "dataset": "ashare_stock_daily"},
            "ast_hash": "h_f_full",
        },
    )
    info = catalog.get_factor_info("f_full")
    # data_source_json 被同步
    assert "ashare_stock_daily" in info["data_source_json"]
    # factor_full_definition 行存在
    spec = dep.full_factor_definition("f_full")
    assert spec["surface"] == "lqtp"
    assert spec["dialect"] == "lqtp"
    assert spec["universe"] == "CSI300"
    assert spec["run_mode"] == "production"
    assert spec["backend"] == "hybrid"
    assert spec["pit_enforce"] is True


# ---------------------------------------------------------------------------
# #6 register factor_version = semantic identity digest
# ---------------------------------------------------------------------------


def test_register_factor_version_from_identity(tmp_path):
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    catalog.register(
        "f_v",
        author="tester",
        frequency="1d",
        ast_hash="a" * 64,
        semantic_identity_digest="digest_v1",
    )
    assert catalog.get_factor_info("f_v")["factor_version"] == "digest_v1"[:16]
    # 缺省 digest → ast_hash 全量 SHA-256（NEW-P0-57：权威 catalog 存全量 digest，
    # 16 位前缀只用于 Parquet/ClickHouse/matrix 显示侧）。
    catalog.register(
        "f_legacy",
        author="tester",
        frequency="1d",
        ast_hash="b" * 64,
    )
    assert catalog.get_factor_info("f_legacy")["factor_version"] == ("b" * 64)


def test_register_production_rejects_identity_change(tmp_path):
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    catalog.register(
        "f_p",
        author="tester",
        frequency="1d",
        ast_hash="a" * 64,
        semantic_identity_digest="digest_v1",
        production=True,
    )
    # 同一 factor_id、同一 AST、语义身份变了 → production 拒绝
    with pytest.raises(FactorSemanticIdentityMismatchError):
        catalog.register(
            "f_p",
            author="tester",
            frequency="1d",
            ast_hash="a" * 64,
            semantic_identity_digest="digest_v2",
            production=True,
        )
    # research：允许覆盖，记录新版本
    catalog.register(
        "f_p",
        author="tester",
        frequency="1d",
        ast_hash="a" * 64,
        semantic_identity_digest="digest_v2",
        production=False,
    )
    assert catalog.get_factor_info("f_p")["factor_version"] == "digest_v2"[:16]
    # 相同身份 → 不冲突
    catalog.register(
        "f_p",
        author="tester",
        frequency="1d",
        ast_hash="a" * 64,
        semantic_identity_digest="digest_v2",
        production=True,
    )


# ---------------------------------------------------------------------------
# #8 record_factor_manifest 原子替换 + delete_factor 补删
# ---------------------------------------------------------------------------


def test_record_factor_edges_atomic_replace(tmp_path):
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_at")
    dep.record_factor_edges(
        "f_at",
        edges=[FactorDependencyEdge("f_at", "ds_old", "close")],
        lookback=2,
        frequency="1d",
        source_dataset="ds_old",
    )
    assert len(dep.get_factor_dependency_edges("f_at")) == 1
    # 替换：旧 edge 消失，新 edge 出现；legacy 行镜像新列集合
    dep.record_factor_edges(
        "f_at",
        edges=[
            FactorDependencyEdge("f_at", "ds_new", "pe_ratio"),
            FactorDependencyEdge("f_at", "ds_new", "close"),
        ],
        lookback=3,
        frequency="1d",
        source_dataset="ds_new",
    )
    edges = dep.get_factor_dependency_edges("f_at")
    assert {e.field_id for e in edges} == {"pe_ratio", "close"}
    assert {e.source_dataset for e in edges} == {"ds_new"}
    flat = catalog.get_factor_dependency("f_at")
    assert set(flat["referenced_columns"]) == {"pe_ratio", "close"}


def test_delete_factor_cleans_edge_and_full_definition(tmp_path):
    catalog, dep = _mk_catalog(tmp_path)
    _register(catalog, "f_del8")
    dep.record_factor_manifest(
        "f_del8",
        edges=[FactorDependencyEdge("f_del8", "ds_x", "close")],
        lookback=1,
        frequency="1d",
        source_dataset="ds_x",
        full_definition={"factor_id": "f_del8", "expression": 'col("close")'},
    )
    assert dep.get_factor_dependency_edges("f_del8")
    assert dep.full_factor_definition("f_del8") is not None
    catalog.delete_factor("f_del8")
    assert not dep.get_factor_dependency_edges("f_del8")
    assert dep.full_factor_definition("f_del8") is None
    assert catalog.get_factor_info("f_del8") is None


# ---------------------------------------------------------------------------
# #7 factor_matrix 分区 read-merge-write（partial 不覆盖整月）
# ---------------------------------------------------------------------------


def test_matrix_partial_update_merges_not_overwrites(tmp_path):
    from storage.materialize.factor_matrix_materializer import (
        FactorMatrixMaterializer,
    )

    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-02")
    d2 = pd.Timestamp("2024-01-03")
    # 第一次：A/B/C 全月
    m.materialize(
        {
            "A": _ser({(d1, "AAA"): 1.0, (d1, "BBB"): 2.0, (d2, "AAA"): 3.0}),
            "B": _ser({(d1, "AAA"): 4.0, (d1, "BBB"): 5.0, (d2, "AAA"): 6.0}),
            "C": _ser({(d1, "AAA"): 7.0, (d1, "BBB"): 8.0, (d2, "AAA"): 9.0}),
        },
        universe="u1",
    )
    # 第二次：只增量 D + 只 1/3 单日 —— 之前会整月覆盖丢掉 A/B/C 与 1/2
    m.materialize(
        {"D": _ser({(d2, "AAA"): 99.0})},
        universe="u1",
    )
    frame = FactorMatrixMaterializer.load_matrix(
        tmp_path / "matrix", universe="u1"
    )
    assert {"A", "B", "C", "D"} <= set(frame.columns)
    # 旧日期保留
    assert set(pd.to_datetime(frame["datetime"]) >= {d1, d2}
    # 新值写入了 D
    row = frame[(frame["datetime"] == d2) & (frame["asset"] == "AAA")]
    assert float(row["D"].iloc[0]) == 99.0
    # 旧值保留
    row = frame[(frame["datetime"] == d2) & (frame["asset"] == "AAA")]
    assert float(row["A"].iloc[0]) == 3.0
