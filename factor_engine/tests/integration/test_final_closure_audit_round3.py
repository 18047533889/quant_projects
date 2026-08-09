# -*- coding: utf-8
"""#收官轮（HEAD eccc35d 复查）第二批评收 adversarial 验收测试。

覆盖外部 AI 复查确认的 6+2（DataAccess Core 已冻结，问题集中在 FactorEngine
物化 / matrix orchestration 最后一层）：

    A1  factor_matrix 部分日期 upsert 不丢历史（整月 → 只重算某一天 → 其它
        日期保留）。
    A2  factor_matrix 某一天显式 NaN/tombstone → 只有该 key 被 tombstone。
    A3  factor_matrix 旧分区读失败 → quarantine + hard fail，绝不 replace。
    A4  engine.run_mode=production + 环境变量 research → 物化器仍 production
        （orchestrator 一锤定音），local 直写拒绝。
    A5  同 factor_id：1d → 5m → semantic version 变化，production 拒绝沿用。
    A6  catalog/full-definition 写入失败 → production 抛
        MaterializedButCatalogCommitFailed（IN_DOUBT），不是完整 success。
    A7  不传 data_source_config 直接 materialize → 仍能恢复完整 live source
        config（execution_spec 还原）。
    A8  spec.dataset=None + table=StockDailyBar → edge.source_dataset =
        ashare_stock_daily（绝不写 logical table name）。
    A9  factor_matrix production 走 staging→publish（P1-14），并发 CAS。
"""
from __future__ import annotations

import glob
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from api import col, ts_mean
from api.factor import Factor
from ir.nodes import IRNode
from runtime import materialize_service
from runtime.factor_identity import compute_identity_from_materialize_ctx
from runtime.incremental_scheduler import _edges_from_analysis
from runtime.lineage import RunLineage
from storage.catalog import FactorCatalog
from storage.exceptions import (
    FactorSemanticIdentityMismatchError,
    MaterializedButCatalogCommitFailed,
)
from storage.materialize.factor_matrix_materializer import (
    FactorMatrixMaterializer,
    FactorMatrixReadError,
)
from storage.materializer import ParquetMaterializer

pytest.importorskip("pandas")


def _series(rows):
    frame = pd.DataFrame(
        [(pd.Timestamp(d), a, v) for d, a, v in rows],
        columns=["timestamp", "instrument", "value"],
    )
    return frame.set_index(["timestamp", "instrument"])["value"].sort_index()


def _fake_analysis():
    return SimpleNamespace(
        ir=IRNode(
            op="ts_mean",
            attrs={"d": 2},
            inputs=(IRNode(op="column", attrs={"name": "close"}),),
        ),
        referenced_fields={},
        referenced_columns=set(),
        lookback=0,
    )


class _FakeSource:
    dataset = "mock_ds"
    pit_enforce = False

    def execution_spec(self):
        return {"type": "data_access", "dataset": "mock_ds"}


class _FakeEngine:
    def __init__(self, run_mode="research"):
        self.run_mode = run_mode
        self.data_source = _FakeSource()


_FAKE_SUMMARY = {
    "factor_id": "f",
    "rows_written": 1,
    "partitions": [2024],
    "partition_keys": ["year=2024"],
    "partitions_failed": [],
    "partition_keys_failed": [],
    "partitions_skipped": [],
    "partition_keys_skipped": [],
    "watermark": {"start_date": "2024-01-01", "end_date": "2024-01-01", "row_count": 1},
    "dq_report": None,
    "run_id": "rid",
    "checkpoint_run_id": "rid",
    "write_target": "local",
    "watermark_deferred": False,
    "storage_format": "long",
    "partition_columns": ["year"],
    "force_tombstones": False,
    "identity_digest": "digest",
    "run_generation": "0",
}


def _call_execute_materialize(
    engine,
    monkeypatch,
    *,
    data_source_config=None,
    target="local",
    capture=None,
    lake_root=None,
):
    """Spy ParquetMaterializer.materialize 并调用 execute_materialize（快，无真实落盘）。"""
    captured = {} if capture is None else capture
    orig = ParquetMaterializer.materialize

    def spy(self, *a, **kw):
        for k in (
            "production",
            "semantic_identity",
            "data_source_config",
            "frequency",
            "write_target",
        ):
            if k in kw:
                captured[k] = kw[k]
        return dict(_FAKE_SUMMARY)

    monkeypatch.setattr(ParquetMaterializer, "materialize", spy)

    def fake_lineage(**kwargs):
        return RunLineage(
            run_id="rid",
            factor_id="f",
            factor_name="f",
            ast_hash="h",
            operator_catalog_hash="o",
        )

    monkeypatch.setattr(
        materialize_service.lineage_service, "build_materialize_lineage", fake_lineage
    )
    factor = Factor(name="f", expr=ts_mean(col("close"), 2), freq="1d")
    output = {"analysis": _fake_analysis(), "result": _series([("2024-01-01", "A", 1.0)])}
    return materialize_service.execute_materialize(
        engine,
        factor,
        output,
        target=target,
        lake_root=lake_root or "/tmp/fe_audit_lake",
        staging_dataset="factor_lake_staging",
        factor_id="f",
        author="audit",
        frequency=None,
        description=None,
        expression=None,
        dq_check=False,
        dq_strict=True,
        dq_thresholds=None,
        write_metadata=False,
        data_source_config=data_source_config,
        resume_materialize=False,
        isolate_partition_failures=True,
        preserve_invalid_rows=False,
        value_dtype="float32",
        clickhouse_table=None,
        ch_ensure_table=True,
        ch_host=None,
        ch_port=None,
        ch_database=None,
        ch_username=None,
        ch_password=None,
        ch_secure=None,
    )


# ---------------------------------------------------------------------------
# A1/A2 factor_matrix 部分日期 upsert 与 tombstone
# ---------------------------------------------------------------------------


def test_matrix_partial_date_upsert_preserves_other_dates(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2, d3 = (
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    )
    # 第一次：A 整月
    m.materialize(
        {"A": _series([("2024-01-01", "AAA", 1.0), ("2024-01-02", "AAA", 2.0), ("2024-01-03", "AAA", 3.0)])},
        universe="u1",
    )
    # 只重算 A 的 1/3 → 1/1、1/2 必须保留，1/3 用新值
    m.materialize({"A": _series([("2024-01-03", "AAA", 30.0)])}, universe="u1")
    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u1")
    A = frame.set_index(["datetime", "asset"])["A"]
    assert float(A[(d1, "AAA")]) == 1.0
    assert float(A[(d2, "AAA")]) == 2.0
    assert float(A[(d3, "AAA")]) == 30.0
    # 新增列 D 不影响 A
    m.materialize({"D": _series([("2024-01-01", "AAA", 99.0)])}, universe="u1")
    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u1")
    assert {"A", "D"} <= set(frame.columns)


def test_matrix_explicit_nan_tombstone_only_that_key(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    m.materialize(
        {"A": _series([("2024-01-01", "AAA", 1.0), ("2024-01-02", "AAA", 2.0)])},
        universe="u1",
    )
    # 显式 NaN = tombstone：只有该 key 被置空，其它 key 保留
    m.materialize({"A": _series([("2024-01-02", "AAA", float("nan"))])}, universe="u1")
    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u1")
    A = frame.set_index(["datetime", "asset"])["A"]
    assert pd.isna(float(A[(d2, "AAA")]))
    assert float(A[(d1, "AAA")]) == 1.0


# ---------------------------------------------------------------------------
# A3 factor_matrix 旧分区读失败 → quarantine + hard fail
# ---------------------------------------------------------------------------


def test_matrix_read_failure_fail_closed_no_replace(tmp_path):
    """旧分区读失败 → quarantine + hard fail，绝不 replace（research / production）。"""
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1 = pd.Timestamp("2024-01-01")

    def _corrupt():
        part = sorted(
            glob.glob(
                str(tmp_path / "matrix" / "universe=u1" / "freq=1d" / "**" / "data.parquet"),
                recursive=True,
            )
        )[0]
        with open(part, "wb") as fh:
            fh.write(b"NOT PARQUET")
        return part

    def _quarantined(part):
        from pathlib import Path

        qdir = Path(part).parent / ".quarantine"
        return qdir.is_dir() and bool(list(qdir.glob("*-data.parquet")))

    # research：hard fail + 隔离（文件移入 quarantine，不 replace）
    m.materialize({"A": _series([("2024-01-01", "AAA", 1.0)])}, universe="u1")
    part = _corrupt()
    with pytest.raises(FactorMatrixReadError):
        m.materialize({"A": _series([("2024-01-01", "AAA", 5.0)])}, universe="u1")
    assert _quarantined(part), "research read failure must quarantine, not replace"

    # production：同样 hard fail + 隔离
    m.materialize({"A": _series([("2024-01-01", "AAA", 7.0)])}, universe="u1")
    part = _corrupt()
    with pytest.raises(FactorMatrixReadError):
        m.materialize(
            {"A": _series([("2024-01-01", "AAA", 9.0)])},
            universe="u1",
            production=True,
        )
    assert _quarantined(part), "production read failure must quarantine, not replace"


# ---------------------------------------------------------------------------
# A4 engine.run_mode → materializer.production 显式传播
# ---------------------------------------------------------------------------


def test_production_local_materialize_rejected_directly(tmp_path):
    """真实 ParquetMaterializer：production=True + write_target=local → 拒绝。"""
    mat = ParquetMaterializer(lake_root=tmp_path / "lake")
    with pytest.raises(ValueError, match="direct-local"):
        mat.materialize(
            "f",
            _series([("2024-01-01", "A", 1.0)]),
            ir_node=_fake_analysis().ir,
            production=True,
            write_target="local",
        )


def test_execute_materialize_propagates_production_from_engine(tmp_path, monkeypatch):
    """engine.run_mode=production + env research → materialize 仍收 production=True。"""
    os.environ["FACTOR_ENGINE_RUN_MODE"] = "research"
    captured: dict = {}
    _call_execute_materialize(
        _FakeEngine(run_mode="production"),
        monkeypatch,
        data_source_config=None,
        capture=captured,
    )
    assert captured.get("production") is True, captured
    si = captured.get("semantic_identity")
    assert si is not None and si.frequency == "1d", captured


def test_execute_materialize_research_gets_production_false(tmp_path, monkeypatch):
    captured: dict = {}
    _call_execute_materialize(
        _FakeEngine(run_mode="research"),
        monkeypatch,
        data_source_config=None,
        capture=captured,
    )
    assert captured.get("production") is False, captured


# ---------------------------------------------------------------------------
# A5 同 factor_id：1d → 5m semantic version 变化 + production reject
# ---------------------------------------------------------------------------


def _identity_digest(frequency):
    return compute_identity_from_materialize_ctx(
        ir_node=_fake_analysis().ir,
        ast_hash="ast_x",
        data_source_config={"type": "x"},
        run_lineage=None,
        frequency=frequency,
    ).identity_digest()


def test_semantic_identity_frequency_changes_digest():
    assert _identity_digest("1d") != _identity_digest("5m")


def test_semantic_version_change_production_rejects_same_factor_id(tmp_path):
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    d1 = _identity_digest("1d")
    d5 = _identity_digest("5m")
    catalog.register(
        "alpha", "me", "1d", "ast_x", semantic_identity_digest=d1, production=True
    )
    # 同 factor_id、同 AST，但 freq 1d→5m → production 拒绝沿用
    with pytest.raises(FactorSemanticIdentityMismatchError):
        catalog.register(
            "alpha", "me", "5m", "ast_x", semantic_identity_digest=d5, production=True
        )
    # research 允许（显式全量重建豁免）
    catalog2 = FactorCatalog(tmp_path / "b.sqlite")
    catalog2.register(
        "alpha", "me", "1d", "ast_x", semantic_identity_digest=d1, production=True
    )
    catalog2.register(
        "alpha", "me", "5m", "ast_x", semantic_identity_digest=d5, production=False
    )


# ---------------------------------------------------------------------------
# A6 catalog/full-definition 写入失败 → production 不是完整 success
# ---------------------------------------------------------------------------


def test_catalog_commit_failure_fail_closed_in_production(tmp_path, monkeypatch):
    import runtime.incremental_scheduler as isch

    def _boom(*a, **k):
        raise RuntimeError("sqlite full")

    monkeypatch.setattr(isch, "record_factor_dependency_from_analysis", _boom)
    eng = _FakeEngine(run_mode="production")
    with pytest.raises(MaterializedButCatalogCommitFailed) as ei:
        _call_execute_materialize(eng, monkeypatch, data_source_config=None)
    assert ei.value.materialization  # 携带已落盘 summary 供对账


def test_catalog_commit_failure_research_only_warns(tmp_path, monkeypatch):
    import runtime.incremental_scheduler as isch

    def _boom(*a, **k):
        raise RuntimeError("sqlite full")

    monkeypatch.setattr(isch, "record_factor_dependency_from_analysis", _boom)
    eng = _FakeEngine(run_mode="research")
    out = _call_execute_materialize(eng, monkeypatch, data_source_config=None)
    assert out["materialization"]["factor_id"] == "f"


# ---------------------------------------------------------------------------
# A7 不传 data_source_config → 恢复完整 live source config（execution_spec）
# ---------------------------------------------------------------------------


def test_effective_data_source_config_recovers_from_live_source():
    cfg = materialize_service._effective_data_source_config(None, _FakeSource())
    assert cfg == {"type": "data_access", "dataset": "mock_ds"}
    # 显式 config 优先
    assert (
        materialize_service._effective_data_source_config(
            {"type": "explicit"}, _FakeSource()
        )
        == {"type": "explicit"}
    )


def test_execute_materialize_recovers_execution_spec(tmp_path, monkeypatch):
    captured: dict = {}
    _call_execute_materialize(
        _FakeEngine(run_mode="research"),
        monkeypatch,
        data_source_config=None,
        capture=captured,
    )
    assert captured.get("data_source_config") == {
        "type": "data_access",
        "dataset": "mock_ds",
    }


# ---------------------------------------------------------------------------
# A8 dependency edge 用 resolved physical dataset，绝不写 logical table name
# ---------------------------------------------------------------------------


def test_edges_fallback_resolves_physical_dataset():
    spec = SimpleNamespace(
        dataset=None, table="StockDailyBar", field_id="close", source_name="Close"
    )
    analysis = SimpleNamespace(
        referenced_fields={"close": spec}, referenced_columns={"close"}, lookback=0
    )
    edges = _edges_from_analysis("f", analysis, SimpleNamespace(dataset="anchor_ds"))
    assert edges[0].source_dataset == "ashare_stock_daily"
    assert edges[0].logical_table == "StockDailyBar"


def test_edges_fallback_unknown_table_uses_anchor():
    spec = SimpleNamespace(
        dataset=None,
        table="UnknownLogicalTable999",
        field_id="close",
        source_name="Close",
    )
    analysis = SimpleNamespace(referenced_fields={"close": spec})
    edges = _edges_from_analysis("f", analysis, SimpleNamespace(dataset="ashare_stock_daily"))
    assert edges[0].source_dataset == "ashare_stock_daily"


# ---------------------------------------------------------------------------
# A9 factor_matrix production 走 staging→publish
# ---------------------------------------------------------------------------


def test_matrix_production_staging_publish(tmp_path):
    m = FactorMatrixMaterializer(matrix_root=tmp_path / "matrix")
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    out = m.materialize(
        {"A": _series([("2024-01-01", "AAA", 1.0), ("2024-01-02", "AAA", 2.0)])},
        universe="u1",
        production=True,
    )
    assert out["manifest_version"] == 1
    # 没有残留 .staging（已原子 publish）
    leftovers = glob.glob(
        str(tmp_path / "matrix" / "universe=u1" / "freq=1d" / "**" / ".staging" / "**"),
        recursive=True,
    )
    assert not leftovers, leftovers
    frame = FactorMatrixMaterializer.load_matrix(tmp_path / "matrix", universe="u1")
    A = frame.set_index(["datetime", "asset"])["A"]
    assert float(A[(d1, "AAA")]) == 1.0 and float(A[(d2, "AAA")]) == 2.0
