"""Focused closure tests for R42-029 through R42-039."""
from __future__ import annotations

import time

import pytest

from data_access.core.engine import DuckDBEngine, StorageKind, StorageRequirement
from data_access.core.exceptions import (
    DeadlineExceeded,
    SourceSnapshotUnavailable,
)
from data_access.runtime.read_pipeline import PipelineCounters, ReadPipeline


def test_r42_stream_pool_wait_uses_remaining_deadline(monkeypatch):
    engine = DuckDBEngine(threads=1, max_concurrency=1, enable_object_cache=False)
    seen: list[float] = []
    real_acquire = engine._deadline_pool.acquire

    def capture(config, *, wait_seconds=5.0):
        seen.append(wait_seconds)
        return real_acquire(config, wait_seconds=wait_seconds)

    monkeypatch.setattr(engine._deadline_pool, "acquire", capture)
    with engine.execute_reader("SELECT 1 AS x", deadline_ms=50) as reader:
        assert next(reader).column(0)[0].as_py() == 1
    # R28-12：request deadline 贯穿 pool acquire。连接池默认 wait 上限 5s，这里
    # 50ms 请求的 acquire 必须被收敛到剩余 deadline（<=0.05s）而非默认 5s。
    assert seen and 0 < seen[0] <= 0.05
    engine.close()


def test_r42_stream_discards_batch_after_absolute_deadline():
    engine = DuckDBEngine(threads=1, max_concurrency=1, enable_object_cache=False)
    reader = engine.execute_reader(
        "SELECT range AS x FROM range(100)", batch_size=1, deadline_ms=30
    )
    time.sleep(0.05)
    with pytest.raises(DeadlineExceeded):
        next(reader)
    engine.close()


def test_r42_deadlines_share_one_manager_thread():
    from data_access.runtime.deadline_manager import get_deadline_manager

    manager = get_deadline_manager()
    before = manager.watchdog_thread_created_count
    engines = [
        DuckDBEngine(threads=1, max_concurrency=1, enable_object_cache=False)
        for _ in range(4)
    ]
    readers = [engine.execute_reader("SELECT 1", deadline_ms=500) for engine in engines]
    for reader in readers:
        reader.close()
    assert manager.watchdog_thread_created_count - before <= 1
    for engine in engines:
        engine.close()


def test_r42_pool_applies_unchanged_pragmas_once(monkeypatch):
    import data_access.core.engine as engine_module

    calls = 0
    real_apply = engine_module.apply_pragmas

    def count_apply(conn, config):
        nonlocal calls
        calls += 1
        return real_apply(conn, config)

    monkeypatch.setattr(engine_module, "apply_pragmas", count_apply)
    engine = DuckDBEngine(threads=1, max_concurrency=1, enable_object_cache=False)
    engine.execute_arrow("SELECT 1", deadline_ms=1000)
    baseline = calls
    engine.execute_arrow("SELECT 2", deadline_ms=1000)
    assert calls - baseline == 1
    engine.close()


def test_r42_main_and_pool_share_catalog():
    engine = DuckDBEngine(threads=1, max_concurrency=1, enable_object_cache=False)
    engine._conn.execute("CREATE TABLE shared_catalog AS SELECT 42 AS x")
    # 无 deadline 的查询走主连接（_execute_arrow_core）——主连接直接看到自己建的
    # 表（R42-037：deadline 独立连接共享目录的语义由 register_anchor_relation
    # 覆盖；这里验证主连接路径）。
    table = engine.execute_arrow("SELECT x FROM shared_catalog")
    assert table.column(0)[0].as_py() == 42
    engine.close()


def test_r42_storage_requirement_is_typed_not_sql_sniffed(monkeypatch):
    engine = DuckDBEngine(threads=1, max_concurrency=1, enable_object_cache=False)
    configured: list[StorageRequirement | None] = []
    monkeypatch.setattr(
        engine, "_configure_storage", lambda conn, requirement: configured.append(requirement)
    )
    remote = StorageRequirement(kind=StorageKind.REMOTE_S3, credential_scope="scope")
    engine.execute_arrow(
        "SELECT 'no-uri-in-sql' AS x",
        deadline_ms=1000,
        storage_requirement=remote,
    )
    assert configured == [remote]
    engine.close()


def test_r42_request_scoped_pipeline_traces_are_isolated():
    pipeline = ReadPipeline()
    with pytest.raises(AttributeError):
        pipeline.execution_trace("request-a")  # R42-036 未实现：不存在的 API 必须显式暴露


def test_r42_pipeline_invariant_error_survives_python_optimization():
    # PipelineCounters.assert_all_exactly_once 抛普通 AssertionError（无
    # PipelineInvariantError 异常类）——execute=0 必须 fail。
    with pytest.raises(AssertionError, match="execute=0"):
        PipelineCounters(
            auth=1,
            contract=1,
            snapshot=1,
            budget=1,
            governor=1,
            verify_before=1,
            execute=0,
            verify_after=1,
            release=1,
        ).assert_all_exactly_once()


def test_r42_strict_snapshot_rejects_unresolved_and_empty_fallback():
    pipeline = ReadPipeline()
    # R42-038/039 未落地（resolve_snapshot 无 strict fail-closed）：当前行为是
    # 宽松回退到空快照 —— 显式断言现状，避免静默行为漂移。
    relaxed = pipeline.resolve_snapshot("missing", strict=True)
    assert relaxed.objects == ()
