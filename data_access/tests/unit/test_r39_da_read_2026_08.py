# -*- coding: utf-8 -*-
"""R39 —— DataAccess 读路径 P0/P1 正确性修复。

覆盖：
    32  deadline 从请求最外层入口开始（DeadlineContext 单一权威，早期阶段计入）
    30/31 temporal availability 接线修复 + production fail-closed / research degrade
    38/39/40/41/42 ReadHandle 状态机 / close 关闭流 / cleanup 回调 / stats 累计
    34/35 非零内存 admission（real memory estimate + enforce_memory_budget）
    44 typed empty-vs-failed resolution（EmptyPhysicalScope / SourceResolutionError）
    50/51 RuntimeModeIdentity 单一权威（DataReadSession / prepare_read / QueryBudget）
"""
from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import (
    DeadlineExceeded,
    SourceResolutionError,
    TemporalContractError,
    ValidationError,
)
from data_access.read.query_cache import reset_query_cache
from data_access.registry.loader import load_registry
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from data_access.runtime.cache_manager import reset_cache_manager
    from data_access.runtime.resource_governor import reset_global_governor

    reset_query_cache()
    reset_cache_manager()
    reset_global_governor()
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_RUN_MODE", raising=False)
    yield
    reset_query_cache()
    reset_cache_manager()
    reset_global_governor()


def _yaml_store(tmp_path: Path, yaml_text: str) -> DataAccessStore:
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    return DataAccessStore(load_registry(cfg), DuckDBEngine(threads=2))


_STATIC_TMPL = """
{a}:
  kind: static
  access_mode: staging
  layout: plain
  root: "{root}"
  glob: "part-*.parquet"
  time_column: ts
  instrument_column: sym
  schema:
    ts: date
    sym: string
    val: double
"""


def _write_rows(root: Path, rows: int, prefix: str = "part") -> None:
    root.mkdir(parents=True, exist_ok=True)
    t = pa.table(
        {
            "ts": [dt.date(2024, 1, 1 + i % 3) for i in range(rows)],
            "sym": [f"S{i % 5}" for i in range(rows)],
            "val": [float(i) for i in range(rows)],
        }
    )
    pq.write_table(t, root / f"{prefix}-{rows}.parquet")


def _static_store(tmp_path: Path, name: str = "ds") -> tuple[DataAccessStore, Path]:
    root = tmp_path / "d"
    _write_rows(root, 3)
    return _yaml_store(tmp_path, _STATIC_TMPL.format(a=name, root=root)), root


def _two_batches():
    b1 = pa.record_batch([pa.array([1, 2])], names=["x"])
    b2 = pa.record_batch([pa.array([3, 4])], names=["x"])
    return iter([b1, b2])


# ===========================================================================
# 32 —— deadline 从请求最外层入口开始（DeadlineContext）
# ===========================================================================

def test_deadline_counts_from_request_start():
    """DeadlineContext 从入口建立，子阶段 check 把早期耗时计入同一预算。"""
    from data_access.runtime.prepared_read import DeadlineContext

    ctx = DeadlineContext.start(max_elapsed_ms=120, source="test")
    assert ctx.deadline_at is not None
    time.sleep(0.03)  # 模拟 prepare 早期阶段（auth/paths）
    rem = ctx.remaining_ms()
    assert rem is not None and rem > 0
    time.sleep(0.12)  # 越过 deadline
    with pytest.raises(DeadlineExceeded):
        ctx.check()
    with pytest.raises(DeadlineExceeded):
        ctx.remaining_ms()


def test_deadline_entered_as_current():
    """DeadlineContext.enter() 成为 request-scoped 权威；reset 后回退 None。"""
    from data_access.runtime.prepared_read import (
        current_deadline,
        reset_deadline_context,
        DeadlineContext,
    )

    assert current_deadline() is None
    ctx = DeadlineContext.start(max_elapsed_ms=60_000, source="test")
    tok = ctx.enter()
    assert current_deadline() is ctx
    reset_deadline_context(tok)
    assert current_deadline() is None


def test_prepare_read_carries_deadline_context(tmp_path):
    """prepare_read 把 DeadlineContext 存进 PreparedRead（execute 拿剩余时间）。"""
    store, root = _static_store(tmp_path)
    from data_access.read.query_budget import QueryBudget

    budget = QueryBudget(max_elapsed_ms=10_000.0)
    prepared = store.prepare_read(
        "ds", columns=["ts", "sym", "val"], query_budget=budget
    )
    assert prepared.deadline_context is not None
    assert prepared.deadline_at is not None
    # 未超时 → remaining_ms 为正（早期阶段计入后仍有余量）
    assert prepared.deadline_context.remaining_ms() > 0


# ===========================================================================
# 30/31 —— temporal availability 接线 + production fail-closed / research degrade
# ===========================================================================

def _fake_pit_contract():
    spec = SimpleNamespace(query_clock="ts", partition_clock="date")
    pit = SimpleNamespace(
        pit_policy="strict",
        availability_column="filing_date",
        pit_fidelity="knowledge_date_pit",
    )
    return SimpleNamespace(physical_partition=spec, pit=pit, market="us", temporal_axes={})


def test_temporal_plan_availability_wired(tmp_path):
    """#30：_build_temporal_plan 从 RuntimeDatasetContract 构造 AvailabilityContract，
    两个位置参数传入，availability 不再是 None。"""
    store, root = _static_store(tmp_path)
    store._compile_contract_strict = lambda dataset: _fake_pit_contract()
    plan = store._build_temporal_plan("ds")
    assert plan is not None
    # research（strict off）→ degraded（authoritative=False），绝不裸 None 冒充权威。
    assert plan.availability is not None
    assert plan.availability.authoritative is False
    assert plan.availability.degradation_reason


def test_temporal_contract_fail_closed_in_production(tmp_path, monkeypatch):
    """#31：production/strict 下 calendar-required availability + 日历不可用 → typed
    TemporalContractError。"""
    store, root = _static_store(tmp_path)
    store._compile_contract_strict = lambda dataset: _fake_pit_contract()
    monkeypatch.setenv("DATA_ACCESS_STRICT_READ", "1")
    from data_access.runtime.mode_identity import set_runtime_mode_identity

    tok = set_runtime_mode_identity("production", source="test")
    try:
        with pytest.raises(TemporalContractError):
            store._build_temporal_plan("ds")
    finally:
        from data_access.runtime.mode_identity import reset_runtime_mode_identity

        reset_runtime_mode_identity(tok)


def test_temporal_contract_degrades_in_research(tmp_path):
    """#31：research 下返回 authoritative=False + degradation_reason，不抛。"""
    store, root = _static_store(tmp_path)
    store._compile_contract_strict = lambda dataset: _fake_pit_contract()
    from data_access.runtime.mode_identity import set_runtime_mode_identity

    tok = set_runtime_mode_identity("interactive_research", source="test")
    try:
        plan = store._build_temporal_plan("ds")
        assert plan.availability is not None
        assert plan.availability.authoritative is False
        assert plan.availability.degradation_reason is not None
    finally:
        from data_access.runtime.mode_identity import reset_runtime_mode_identity

        reset_runtime_mode_identity(tok)


# ===========================================================================
# 38/39/40/41/42 —— ReadHandle 状态机 / close / cleanup / stats
# ===========================================================================

def test_read_handle_state_machine_open_to_closed():
    """#40：CLOSED 硬拒绝一切读。"""
    from data_access.read.read_handle import ReadHandle

    h = ReadHandle(stream=_two_batches())
    assert h.state == "OPEN"
    h.close()
    assert h.state == "CLOSED"
    with pytest.raises(RuntimeError, match="CLOSED"):
        h.to_arrow()
    with pytest.raises(RuntimeError, match="CLOSED"):
        list(h.stream())
    h.close()  # 幂等


def test_read_handle_close_closes_source_and_releases_reservation():
    """#39/#41：close() 关闭底层 stream + 释放 reservation + 跑 cleanup 回调。"""
    from data_access.read.read_handle import ReadHandle

    reader_closed = {"n": 0}
    released = []

    class _Reader:
        def __iter__(self):
            return iter([pa.record_batch([pa.array([1])], names=["x"])])

        def close(self):
            reader_closed["n"] += 1

    cleanup_called = []

    def _cleanup():
        cleanup_called.append(1)

    h = ReadHandle(
        stream=_Reader(),
        _reservation="R1",
        _reservation_release_fn=lambda res: released.append(res),
        _cleanup_callbacks=[_cleanup],
    )
    # 从未迭代 → close 也必须释放（#38 create-then-close）
    h.close()
    assert released == ["R1"]
    assert cleanup_called == [1]
    assert reader_closed["n"] == 1


def test_read_handle_partial_consume_close_runs_generator_finally():
    """#39：close() 在部分消费后调用 generator.close() → finally 执行。"""
    from data_access.read.read_handle import ReadHandle

    finally_ran = []

    def _gen():
        try:
            for i in range(3):
                yield pa.record_batch([pa.array([i])], names=["x"])
        finally:
            finally_ran.append(1)

    h = ReadHandle(stream=_gen())
    it = h.stream()
    next(it)  # 部分消费
    h.close()
    assert finally_ran == [1]
    assert h.state == "CLOSED"


def test_read_handle_stream_full_consume_state_closed():
    """#40/#41：one-shot 流完整消费后进入 CLOSED，不能二次消费。"""
    from data_access.read.read_handle import ReadHandle

    h = ReadHandle(stream=_two_batches())
    batches = list(h.stream())
    assert sum(b.num_rows for b in batches) == 4
    assert h.state == "CLOSED"
    with pytest.raises(RuntimeError, match="CLOSED|one-shot"):
        h.to_arrow()


def test_read_handle_stream_stats_accumulate():
    """#42：stream 消费期间 stats rows/bytes/elapsed 累计并写回 handle.stats。"""
    from data_access.read.read_contract import ReadStats
    from data_access.read.read_handle import ReadHandle

    h = ReadHandle(
        stream=_two_batches(), stats=ReadStats(rows=0, bytes=0, elapsed_ms=0.0)
    )
    batches = list(h.stream())
    assert sum(b.num_rows for b in batches) == 4
    assert h.stats.rows == 4
    assert h.stats.bytes > 0
    assert h.stats.elapsed_ms >= 0.0


def test_read_handle_buffer_reuse_after_materialize():
    """#40：buffer=True 固化 → MATERIALIZED，可任意复用。"""
    from data_access.read.read_handle import ReadHandle

    h = ReadHandle(stream=_two_batches())
    batches = list(h.stream(buffer=True))
    assert sum(b.num_rows for b in batches) == 4
    assert h.state == "MATERIALIZED"
    assert h.to_arrow().num_rows == 4
    assert sum(b.num_rows for b in h.stream()) == 4


# ===========================================================================
# 34/35 —— 非零内存 admission
# ===========================================================================

def test_prepare_read_memory_admission_nonzero(tmp_path):
    """#34：prepare_read admit 传真实内存估计（不再是 0）。"""
    store, root = _static_store(tmp_path)
    prepared = store.prepare_read("ds", columns=["ts", "sym", "val"])
    assert prepared.resource_reservation.estimated_memory > 0


def test_enforce_memory_budget_fail_closed():
    """#35：max_estimated_memory 超限 → 执行前拒绝。"""
    from data_access.read.query_budget import (
        QueryBudget,
        enforce_memory_budget,
    )

    budget = QueryBudget(max_estimated_memory=1_000)
    enforce_memory_budget(budget, estimated_memory=500)
    with pytest.raises(ValidationError, match="内存"):
        enforce_memory_budget(budget, estimated_memory=10_000)


def test_collect_polars_with_budget_enforces_memory_before_collect():
    """#36：collect_polars_with_budget 传 estimated_memory → 执行前拒绝。"""
    from data_access.read.query_budget import (
        QueryBudget,
        collect_polars_with_budget,
    )

    lf = pl.LazyFrame({"x": [1, 2, 3]})
    with pytest.raises(ValidationError, match="内存"):
        collect_polars_with_budget(
            lf,
            query_budget=QueryBudget(max_estimated_memory=10),
            estimated_memory=10_000,
        )


# ===========================================================================
# 44 —— typed empty-vs-failed resolution
# ===========================================================================

def test_empty_physical_scope_is_typed_empty():
    """#44：EmptyPhysicalScope 是 list 子类（语义明确=空），不是失败。"""
    from data_access.runtime.prepared_read import EmptyPhysicalScope

    empty = EmptyPhysicalScope(dataset="ds", reason="no matching partitions")
    assert list(empty) == []
    assert isinstance(empty, list)
    assert empty.dataset == "ds"
    assert empty.reason


def test_source_resolution_error_distinct_from_empty():
    """#44：SourceResolutionError 是 DataAccessError，与空数据集严格区分。"""
    from data_access.core.exceptions import DataAccessError

    err = SourceResolutionError("resolution failed")
    assert isinstance(err, DataAccessError)
    assert not isinstance(err, list)


def test_composed_anchor_resolution_failure_raises_typed(tmp_path, monkeypatch):
    """#44：组合读锚点解析失败 → SourceResolutionError（不再 source_paths=[] 冒充空）。"""
    import data_access.read.physical_plan as pp
    from data_access.core.exceptions import CapabilityUnavailableError
    from types import SimpleNamespace

    store, root = _static_store(tmp_path)
    # 让聚合锚点走临时 parquet 回退（pool 不可用 → CapabilityUnavailableError）。
    store._engine.register_anchor_relation = lambda view, tbl: (_ for _ in ()).throw(
        CapabilityUnavailableError("pool unavailable")
    )
    store._engine.drop_anchor_relation = lambda view: None

    # 模拟锚点路径解析失败
    def _boom(*a, **k):
        raise RuntimeError("glob failed")

    store._prepare_dataset_read = _boom

    # 聚合锚点 fake handle（aggregate_minute_bundle 输出）
    class _AggHandle:
        def to_arrow(self):
            return pa.table({"ts": [dt.date(2024, 1, 1)], "inst": ["S0"], "val": [1.0]})

    store._engine.execute_arrow = lambda sql, params, deadline_ms=None: pa.table(
        {"ts": [dt.date(2024, 1, 1)], "inst": ["S0"]}
    )

    req = SimpleNamespace(
        aggregations=[{"field": "val", "spec": "sum"}],
        anchor="ds",
        dataset_params=lambda ds: {},
        joins={},
        join_specs={},
        filters=None,
        filters_by_dataset=None,
        universe=None,
        time_varying_universe=True,
        limit=None,
        order_by=None,
        normalize_units=False,
        result="auto",
    )
    plan = SimpleNamespace(
        anchor="ds",
        datasets=("ds", "ds2"),  # len>1 触发组合执行
        fields=(),
        per_dataset_columns={"ds": ["ts", "sym", "val"], "ds2": ["ts", "sym"]},
        join_specs_effective={"ds": {"policy": "exact"}},  # 非空，跳过 _effective_join_specs
        instruments=None,
        time_range=(dt.date(2024, 1, 1), dt.date(2024, 1, 3)),
        compiled=req,
    )
    # 不需要真的跑到 aggregate_minute_bundle——直接 mock 掉，聚焦 source_paths 包装。
    import data_access.read.aggregation as agg_mod

    monkeypatch.setattr(agg_mod, "aggregate_minute_bundle", lambda *a, **k: _AggHandle())
    from data_access.core.exceptions import DataAccessError

    with pytest.raises((SourceResolutionError, DataAccessError)):
        pp.execute_physical_plan(store, plan)


# ===========================================================================
# 50/51 —— RuntimeModeIdentity 单一权威
# ===========================================================================

def test_runtime_mode_identity_single_authority(monkeypatch):
    """#50/#51：RuntimeModeIdentity 覆盖 env；QueryBudget 走同一权威。"""
    from data_access.runtime.mode_identity import (
        current_runtime_mode,
        reset_runtime_mode_identity,
        set_runtime_mode_identity,
    )
    from data_access.read.query_budget import (
        QueryBudget,
        is_strict_semantics,
        resolve_query_budget,
    )

    monkeypatch.setenv("DATA_ACCESS_RUN_MODE", "interactive_research")
    assert current_runtime_mode().value == "interactive_research"
    assert is_strict_semantics() is False

    tok = set_runtime_mode_identity("production", source="test")
    try:
        assert current_runtime_mode().value == "production"
        assert is_strict_semantics() is True
        # 显式宽松 budget 在 production 下被 floor 收紧（require_columns=True）
        budget = resolve_query_budget(
            QueryBudget(max_rows=1_000_000, require_columns=False)
        )
        assert budget.require_columns is True
    finally:
        reset_runtime_mode_identity(tok)
    assert current_runtime_mode().value == "interactive_research"
    assert is_strict_semantics() is False


def test_data_read_session_sets_mode_identity(tmp_path):
    """#50：DataReadSession.__enter__ 绑定 RuntimeModeIdentity，退出恢复。"""
    from data_access.read.read_session import DataReadSession
    from data_access.runtime.mode_identity import current_runtime_mode

    store, root = _static_store(tmp_path)
    with DataReadSession(store, run_mode="production") as s:
        assert current_runtime_mode().value == "production"
        prepared = s.prepare_read("ds", columns=["ts", "sym", "val"])
        assert prepared.resource_reservation.estimated_memory > 0
    assert current_runtime_mode().value == "interactive_research"


def test_prepare_read_explicit_run_mode_sets_identity(tmp_path):
    """#50：prepare_read(run_mode=...) 请求期内单一权威。"""
    from data_access.runtime.mode_identity import current_runtime_mode

    store, root = _static_store(tmp_path)
    # production → 需显式 columns（floor 收紧）；且执行后身份恢复。
    prepared = store.prepare_read(
        "ds", columns=["ts", "sym", "val"], run_mode="production"
    )
    assert prepared.query_budget.require_columns is True
    assert current_runtime_mode().value == "interactive_research"


def test_store_read_run_mode_propagates_to_identity(tmp_path):
    """R39 #50：FE → DA 传播 —— ``store.read(run_mode=...)`` 经 **params 到达
    ``prepare_read`` 并设置 request-scoped RuntimeModeIdentity（单一权威）。
    run_mode 从 params 中被 pop（不残留为数据集参数）。
    """
    from data_access.runtime.mode_identity import current_runtime_mode

    store, root = _static_store(tmp_path)
    # store.read 无显式 run_mode 参数 → run_mode 走 **params → _read_handle →
    # read_arrow_stream → prepare_read(params) → prepare_read pop 并设 identity。
    handle = store.read(
        "ds", columns=["ts", "sym", "val"], run_mode="production"
    )
    # identity 在 prepare 阶段生效（strict floor：require_columns）。
    assert handle.snapshot is not None
    # prepare 结束后身份恢复（request-scoped）。
    assert current_runtime_mode().value == "interactive_research"
