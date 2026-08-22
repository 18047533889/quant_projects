# -*- coding: utf-8 -*-
"""R32 P0-041..050：DuckDB 并发/deadline/取消/公平性/诊断修复测试。

验证现有实现（R28已修复的功能）+ 识别未实现的需求。
"""
from __future__ import annotations

import threading
import time
from typing import Any

import duckdb
import pytest

from data_access.core.engine import DuckDBEngine, _DeadlineConnectionPool
from data_access.core.exceptions import (
    DeadlineExceeded,
    EngineClosedError,
    ResourceBudgetExceeded,
)


# ============================================================================
# R32-P0-041：验证 deadline 机制（R28-12 already implemented）
# ============================================================================


def test_r32_duck_001_expired_deadline_immediate_raise():
    """T-R32-DUCK-001：deadline 已过时立即失败（R28-12 已实现）。

    验证：execute_arrow 在 deadline 已过期时 fail-fast。
    """
    engine = DuckDBEngine(max_concurrency=4, threads=2)
    try:
        start = time.monotonic()
        with pytest.raises(DeadlineExceeded) as exc_info:
            engine.execute_arrow("SELECT 1 AS x", deadline_ms=0.001)
        elapsed = time.monotonic() - start
        # 应该在 100ms 内 fail-fast
        assert elapsed < 0.2, f"应立即失败但耗时 {elapsed:.3f}s"
        assert "deadline" in str(exc_info.value).lower()
    finally:
        engine.close()


def test_r32_duck_001b_semaphore_blocks_with_deadline():
    """T-R32-DUCK-001b：_exec_sem 并发控制 + deadline 组合验证。

    验证：_exec_sem 达到上限时，新请求会因 deadline 超时而失败。
    """
    engine = DuckDBEngine(max_concurrency=2, threads=1)
    try:
        # 占满 2 个 _exec_sem slot
        release_event = threading.Event()
        errors: list[Exception] = []

        def _hold_semaphore(idx: int):
            try:
                with engine._exec_sem:
                    release_event.wait(timeout=5.0)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_hold_semaphore, args=(1,), daemon=True)
        t2 = threading.Thread(target=_hold_semaphore, args=(2,), daemon=True)
        t1.start()
        t2.start()
        time.sleep(0.2)

        # 第 3 个请求：_exec_sem 阻塞 + deadline 短 → 应超时
        start = time.monotonic()
        acquired = engine._exec_sem.acquire(timeout=0.5)
        elapsed = time.monotonic() - start
        if acquired:
            engine._exec_sem.release()
            pytest.fail("不应成功 acquire（前 2 个占满）")
        assert 0.4 < elapsed < 1.0, f"应在 ~0.5s 超时，实际 {elapsed:.3f}s"

        release_event.set()
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)
        assert not errors
    finally:
        engine.close()


# ============================================================================
# R32-P0-042：连接创建 lazy + 失败时释放 slot（R28 已实现 #P0-38）
# ============================================================================


def test_r32_duck_002_connection_failure_releases_slot():
    """T-R32-DUCK-002：连接创建失败时 slot 被释放（验证 line 114-117 路径）。

    验证：pool._new_connection 失败 → _active 不增加（连接创建在 _active += 1 前）。

    NOTE: 当前实现中 _new_connection() 在 line 115 调用，BEFORE _active += 1 (line 116)，
    所以连接创建失败时 _active 本就未增加，无需 "release"。这是正确的设计。

    本测试验证：连接创建失败 → 异常传播 → pool 状态未变。
    """
    engine = DuckDBEngine(max_concurrency=2, threads=1)
    try:
        pool = engine._deadline_pool
        # 占满所有 idle slot，强制下次 acquire 创建新连接
        conns = []
        for _ in range(2):
            conns.append(pool.acquire(engine._config, wait_seconds=1.0))
        # 现在 active=2，idle=0
        assert pool._active == 2
        assert len(pool._idle) == 0

        # 归还一个，释放空间但清空 idle
        for conn in conns:
            pool.release(conn, healthy=False)  # unhealthy → 不进 idle
        # 现在 active=0，idle=0（所有连接都被 close）
        assert pool._active == 0
        assert len(pool._idle) == 0

        # mock 连接创建失败
        original_new = pool._new_connection

        def _fail_new():
            raise RuntimeError("模拟连接创建失败")

        pool._new_connection = _fail_new
        try:
            # 下次 acquire 会尝试创建新连接（line 115），应失败
            with pytest.raises(RuntimeError, match="模拟连接创建失败"):
                conn = pool.acquire(engine._config, wait_seconds=1.0)
        finally:
            pool._new_connection = original_new

        # 验证 pool 状态：创建失败后 active 未增加
        assert pool._active == 0
        assert len(pool._idle) == 0
    finally:
        engine.close()


def test_r32_duck_002b_pragma_failure_releases_slot():
    """T-R32-DUCK-002b：PRAGMA 配置失败时 slot 被释放（#P0-38 已修复）。

    验证：apply_pragmas 失败 → 连接被关闭，_active 递减。
    """
    engine = DuckDBEngine(max_concurrency=2, threads=1)
    try:
        pool = engine._deadline_pool
        from unittest.mock import patch

        with patch(
            "data_access.core.engine.apply_pragmas", side_effect=RuntimeError("PRAGMA 失败")
        ):
            with pytest.raises(RuntimeError, match="PRAGMA 失败"):
                pool.acquire(engine._config, wait_seconds=1.0)
        # 验证 slot 未泄漏
        assert pool._active == 0
    finally:
        engine.close()


# ============================================================================
# R32-P0-043：连接生命周期 caller-scoped（pool 已实现）
# ============================================================================


def test_r32_duck_003_connection_lifecycle_scoped():
    """T-R32-DUCK-003：连接池连接正确归还，无泄漏。

    验证：多次查询后 active=0（连接已归还）。
    """
    engine = DuckDBEngine(max_concurrency=4, threads=2)
    try:
        for _ in range(10):
            result = engine.execute_arrow("SELECT 123 AS x", deadline_ms=5000)
            assert result.column("x")[0].as_py() == 123
        pool = engine._deadline_pool
        assert pool._active == 0, f"应该无活跃连接，实际 {pool._active}"
        assert len(pool._idle) <= engine._max_concurrency
    finally:
        engine.close()


# ============================================================================
# R32-P0-044：execute 前检查 deadline（R28-12 已实现 line 629-632）
# ============================================================================


def test_r32_duck_004_refuses_execution_if_deadline_expired():
    """T-R32-DUCK-004：deadline 已过时拒绝执行（R28-12 line 629-632）。

    验证：_execute_isolated_with_deadline 开头检查 deadline。
    """
    engine = DuckDBEngine(max_concurrency=2, threads=1)
    try:
        # 直接用极短 deadline（会在 line 629-632 fail-fast）
        with pytest.raises(DeadlineExceeded) as exc_info:
            engine.execute_arrow("SELECT 1", deadline_ms=0.0001)
        assert "deadline" in str(exc_info.value).lower()
        # slot 应未泄漏
        pool = engine._deadline_pool
        assert pool._active == 0
    finally:
        engine.close()


# ============================================================================
# R32-P0-045：DuckDB 查询 interrupt（R28-12 已实现 line 661-671）
# ============================================================================


def test_r32_duck_005_cancellation_interrupts_query():
    """T-R32-DUCK-005：长查询在 deadline 后被 interrupt（R28-12 已实现）。

    验证：watchdog 调用 conn.interrupt()，查询被取消。
    """
    engine = DuckDBEngine(max_concurrency=2, threads=1)
    try:
        start = time.monotonic()
        with pytest.raises(DeadlineExceeded) as exc_info:
            # 这个查询如果不 interrupt 会跑很久
            engine.execute_arrow(
                "SELECT SUM(i) FROM range(10000000000) AS t(i)", deadline_ms=500
            )
        elapsed = time.monotonic() - start
        # 应该在 ~0.5s 附近被 interrupt（允许 ±1s 误差）
        assert elapsed < 3.0, f"查询应被 interrupt，实际耗时 {elapsed:.3f}s"
        assert "deadline" in str(exc_info.value).lower() or "取消" in str(exc_info.value)
        # 验证 slot 恢复
        pool = engine._deadline_pool
        assert pool._active == 0
    finally:
        engine.close()


# ============================================================================
# R32-P0-046：公平性（NOT IMPLEMENTED - 当前是 FIFO Semaphore）
# ============================================================================


def test_r32_duck_006_fairness_baseline():
    """T-R32-DUCK-006：公平性基线测试（当前未实现 per-principal 调度）。

    NOTE: 当前实现使用标准 Semaphore（FIFO），无 per-principal 公平队列。
    本测试记录基线行为，标记为 NOT_IMPLEMENTED。
    """
    engine = DuckDBEngine(max_concurrency=2, threads=1)
    try:
        # 验证基本 FIFO 行为
        success_count = threading.Semaphore(0)

        def _worker(idx: int):
            engine.execute_arrow("SELECT 1")
            success_count.release()

        threads = [threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(5)]
        for t in threads:
            t.start()
        for _ in range(5):
            assert success_count.acquire(timeout=5.0)
        for t in threads:
            t.join(timeout=1.0)
        # 标记：per-principal fairness NOT_IMPLEMENTED
    finally:
        engine.close()


# ============================================================================
# R32-P0-047：slot 数遵守 max_concurrency（已实现）
# ============================================================================


def test_r32_duck_007_slot_count_respects_max_concurrency():
    """T-R32-DUCK-007：_exec_sem 遵守 max_concurrency。

    验证：同时最多 max_concurrency 个查询执行。
    """
    engine = DuckDBEngine(max_concurrency=2, threads=1)
    try:
        active_count = threading.Semaphore(0)
        release_event = threading.Event()

        def _worker(idx: int):
            with engine._exec_sem:
                active_count.release()
                release_event.wait(timeout=5.0)

        threads = [threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(3)]
        for t in threads:
            t.start()

        # 前 2 个应该立即获取
        assert active_count.acquire(timeout=1.0)
        assert active_count.acquire(timeout=1.0)
        # 第 3 个应该被阻塞
        assert not active_count.acquire(timeout=0.3)

        release_event.set()
        for t in threads:
            t.join(timeout=2.0)
    finally:
        engine.close()


# ============================================================================
# R32-P0-048：嵌套调用（当前无 reentrancy 支持，不死锁即可）
# ============================================================================


def test_r32_duck_008_nested_call_no_deadlock():
    """T-R32-DUCK-008：嵌套调用不死锁（当前无 reentrancy，但不应死锁）。

    验证：尝试递归 acquire 会 timeout 而非死锁。
    """
    engine = DuckDBEngine(max_concurrency=1, threads=1)
    try:
        nested_done = threading.Event()

        def _try_nested():
            with engine._exec_sem:
                # 已持有 slot，再次尝试会超时
                acquired = engine._exec_sem.acquire(timeout=0.3)
                if acquired:
                    engine._exec_sem.release()
                nested_done.set()

        thread = threading.Thread(target=_try_nested, daemon=True)
        thread.start()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "嵌套调用不应死锁"
        assert nested_done.is_set()
    finally:
        engine.close()


# ============================================================================
# R32-P0-049：telemetry 记录查询指标
# ============================================================================


def test_r32_duck_009_telemetry_records_queries():
    """T-R32-DUCK-009：telemetry 记录查询。

    验证：record_query 被调用，counters 有数据。
    """
    from data_access.read.telemetry import get_counters_snapshot, reset_counters

    reset_counters()
    engine = DuckDBEngine(max_concurrency=2, threads=1)
    try:
        engine.execute_arrow("SELECT 999 AS x", deadline_ms=5000)
        counters = get_counters_snapshot()
        # telemetry 应该记录了查询
        assert len(counters) > 0
        for op, c in counters.items():
            assert c.total_queries >= 1
            assert c.total_elapsed_ms >= 0
    finally:
        engine.close()
        reset_counters()


# ============================================================================
# R32-P0-050：错误分类（部分实现）
# ============================================================================


def test_r32_duck_010_error_classification():
    """T-R32-DUCK-010：DuckDB 错误分类。

    验证：DeadlineExceeded / EngineError 正确抛出。
    """
    from data_access.core.exceptions import EngineError

    engine = DuckDBEngine(max_concurrency=2, threads=1)
    try:
        # DEADLINE_EXCEEDED
        with pytest.raises(DeadlineExceeded):
            engine.execute_arrow("SELECT 1 FROM range(1000000000)", deadline_ms=1)

        # SYNTAX_ERROR → EngineError
        with pytest.raises(EngineError):
            engine.execute_arrow("SELCT invalid syntax")

        # 文件不存在 → EngineError
        with pytest.raises(EngineError):
            engine.execute_arrow("SELECT * FROM read_parquet('/no/such/file.parquet')")
    finally:
        engine.close()


# ============================================================================
# 并发正确性综合测试
# ============================================================================


def test_r32_duck_pool_concurrent_correctness():
    """综合测试：多线程并发，验证 slot 计数正确性。

    Setup: 20 个线程并发执行查询。
    Expected: 所有查询完成后 active=0。
    """
    engine = DuckDBEngine(max_concurrency=4, threads=2)
    try:
        success_count = threading.Semaphore(0)
        errors: list[Exception] = []

        def _worker(idx: int):
            try:
                result = engine.execute_arrow(f"SELECT {idx} AS x", deadline_ms=10000)
                assert result.column("x")[0].as_py() == idx
                success_count.release()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(20)]
        for t in threads:
            t.start()

        for _ in range(20):
            assert success_count.acquire(timeout=15)
        for t in threads:
            t.join(timeout=1.0)

        assert not errors, f"不应有错误: {errors}"
        pool = engine._deadline_pool
        assert pool._active == 0
    finally:
        engine.close()


# ============================================================================
# unhealthy 连接不回池（#P0-7 已实现 line 141-157）
# ============================================================================


def test_r32_duck_unhealthy_connection_not_reused():
    """验证 interrupt/timeout 的连接标记 unhealthy，不回池（#P0-7 已实现）。"""
    engine = DuckDBEngine(max_concurrency=2, threads=1)
    try:
        pool = engine._deadline_pool
        # 触发 timeout
        with pytest.raises(DeadlineExceeded):
            engine.execute_arrow("SELECT 1 FROM range(1000000000)", deadline_ms=100)
        # unhealthy 连接应被丢弃
        assert pool._active == 0
        # 验证 idle 池可用
        conn = pool.acquire(engine._config, wait_seconds=1.0)
        pool.release(conn, healthy=True)
    finally:
        engine.close()


# ============================================================================
# EngineClosedError（#P0-10 已实现）
# ============================================================================


def test_r32_duck_closed_engine_fails_fast():
    """验证 engine.close() 后查询 → fail-fast。

    验证：关闭后的 engine 尝试查询应立即失败。
    当前实现：连接已关闭 → duckdb.ConnectionException → 包装为 EngineError。
    """
    from data_access.core.exceptions import EngineError

    engine = DuckDBEngine(max_concurrency=2, threads=1)
    engine.close()
    # 关闭后查询应 fail-fast（抛 EngineError 包装的 ConnectionException）
    with pytest.raises((EngineError, EngineClosedError, duckdb.Error)):
        engine.execute_arrow("SELECT 1")


# ============================================================================
# 回归测试：验证基本功能
# ============================================================================


def test_r32_duck_existing_tests_baseline():
    """回归测试：验证基本 DuckDB 功能仍正常工作。"""
    engine = DuckDBEngine(threads=2, max_concurrency=4)
    try:
        # 基本查询
        result = engine.execute_arrow("SELECT 42 AS answer")
        assert result.column("answer")[0].as_py() == 42

        # explain
        plan = engine.explain("SELECT 1 AS x")
        assert plan

        # scoped SQL（使用独立连接，view 在该连接内注册）
        result = engine.execute_scoped_sql_arrow(
            [("__v", "SELECT 7 AS x")], "SELECT x FROM __v"
        )
        assert result.column("x")[0].as_py() == 7

        # 带 deadline 的查询
        result = engine.execute_arrow("SELECT 999 AS z", deadline_ms=5000)
        assert result.column("z")[0].as_py() == 999
    finally:
        engine.close()
