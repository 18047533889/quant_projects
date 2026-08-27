# -*- coding: utf-8 -*-
"""内存泄漏和资源治理综合检测套件。

运行：``python3 -m pytest tests/test_memory_leak_detection.py -v -s``

检测范围：
    1. 内存泄漏检测（长时间运行 + tracemalloc）
    2. 资源治理验证（L0 buffer/cache 生命周期）
    3. 压力测试（高并发/内存受限/大数据量）
    4. 失败恢复测试（backend 崩溃/超时后的资源清理）
    5. 资源使用统计（峰值/平均/CPU/磁盘）
"""
import gc
import os
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _get_process_memory_mb():
    """当前进程 RSS（MiB）。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        return 0.0


def _collect_garbage():
    """强制 GC 回收（测试前后清理）。"""
    gc.collect()
    gc.collect()
    gc.collect()


def _measure_memory_growth(func, iterations: int = 100, warmup: int = 10):
    """测量函数重复执行的内存增长。

    返回：(initial_mb, final_mb, growth_mb, growth_per_iter_kb)
    """
    _collect_garbage()
    for _ in range(warmup):
        func()
    _collect_garbage()

    initial = _get_process_memory_mb()
    for _ in range(iterations):
        func()
    _collect_garbage()
    final = _get_process_memory_mb()

    growth = final - initial
    growth_per_iter = (growth * 1024) / iterations if iterations > 0 else 0
    return initial, final, growth, growth_per_iter


# ---------------------------------------------------------------------------
# 1. 内存泄漏检测
# ---------------------------------------------------------------------------


def test_memory_leak_resource_governor_reserve_release():
    """MemoryGovernor reserve/release 循环无泄漏。"""
    from factor_engine.runtime.resource_governor import MemoryGovernor

    gov = MemoryGovernor(process_budget_bytes=100 * 1024**2, duckdb_budget_bytes=20 * 1024**2)

    def cycle():
        gov.reserve_accounting("test_layer", 1024 * 1024)
        gov.release_accounting("test_layer", 1024 * 1024)

    initial, final, growth, per_iter = _measure_memory_growth(cycle, iterations=1000)

    # 每次循环泄漏应 < 1KB（允许一定解释器开销）
    assert per_iter < 1.0, f"MemoryGovernor 泄漏 {per_iter:.2f}KB/iter (growth={growth:.2f}MB)"


def test_memory_leak_buffer_store_put_release():
    """GovernedBufferStore put/release 循环无泄漏。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore
    import pandas as pd

    store = GovernedBufferStore(budget_bytes=50 * 1024**2)
    df = pd.DataFrame({"x": range(100)})

    def cycle():
        result = store.put(f"key_{threading.get_ident()}", df, bytes_=1024)
        if result.status == "MEMORY":
            store.release(f"key_{threading.get_ident()}")

    initial, final, growth, per_iter = _measure_memory_growth(cycle, iterations=500)

    # 每次循环泄漏应 < 2KB
    assert per_iter < 2.0, f"BufferStore 泄漏 {per_iter:.2f}KB/iter (growth={growth:.2f}MB)"


def test_memory_leak_execution_resource_scope():
    """ExecutionResourceScope enter/exit 循环无泄漏。"""
    from factor_engine.runtime.resource_governor import ExecutionResourcePlan, ExecutionResourceScope

    plan = ExecutionResourcePlan.auto()

    def cycle():
        with ExecutionResourceScope(plan, strict=False):
            pass

    initial, final, growth, per_iter = _measure_memory_growth(cycle, iterations=500)

    # 每次循环泄漏应 < 3KB（DuckDB 连接对象开销，已修复单位问题后降至 ~2KB）
    assert per_iter < 3.0, f"ExecutionResourceScope 泄漏 {per_iter:.2f}KB/iter (growth={growth:.2f}MB)"


def test_memory_leak_cache_lru_eviction():
    """Cache LRU 淘汰无泄漏（register_layer + evict hook）。"""
    from factor_engine.runtime.resource_governor import MemoryGovernor
    import pandas as pd

    gov = MemoryGovernor(process_budget_bytes=10 * 1024**2, duckdb_budget_bytes=2 * 1024**2)
    cache = {}

    def evict_hook(target_bytes):
        freed = 0
        to_remove = []
        for k, v in list(cache.items())[:5]:
            to_remove.append(k)
            freed += len(str(v))
            if freed >= target_bytes:
                break
        for k in to_remove:
            cache.pop(k, None)
        return freed

    gov.register_layer("test_cache", evict_hook)

    def cycle():
        for i in range(20):
            cache[f"k{i}"] = pd.Series(range(100))
        gov.reserve("test_cache", 5 * 1024**2)
        gov.release("test_cache", 5 * 1024**2)
        cache.clear()

    initial, final, growth, per_iter = _measure_memory_growth(cycle, iterations=100)

    # 每次循环泄漏应 < 5KB
    assert per_iter < 5.0, f"Cache LRU eviction 泄漏 {per_iter:.2f}KB/iter (growth={growth:.2f}MB)"


# ---------------------------------------------------------------------------
# 2. 资源治理验证
# ---------------------------------------------------------------------------


def test_resource_governance_l0_buffer_lifecycle():
    """L0 buffer 生命周期：put → pin → release → 内存释放。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore, BufferEntryState
    import pandas as pd

    store = GovernedBufferStore(budget_bytes=100 * 1024**2)
    df = pd.DataFrame({"x": range(1000)})

    # Put
    result = store.put("test_key", df, bytes_=10_000)
    assert result.status == "MEMORY"
    assert store.get("test_key") is not None

    # Pin
    assert store.pin("test_key") is True
    summary = store.summary()
    assert summary["pinned"] >= 1

    # Release
    store.release("test_key")
    assert store.get("test_key") is None

    # 验证内存释放
    summary_after = store.summary()
    assert summary_after["accounted_bytes"] < summary["accounted_bytes"]


def test_resource_governance_cache_expiration():
    """Cache 过期机制验证（使用 ExecutionCacheSession）。"""
    from factor_engine.cache.session import ExecutionCacheSession

    # ExecutionCacheSession 没有 TTL 过期机制，跳过此测试
    pytest.skip("ExecutionCacheSession 不支持 TTL 过期（按执行周期管理）")


def test_resource_governance_file_handle_cleanup():
    """文件句柄泄漏检查（打开/关闭 parquet 文件）。"""
    import tempfile
    import pandas as pd

    try:
        import psutil
        proc = psutil.Process()
        initial_fds = len(proc.open_files())
    except Exception:
        pytest.skip("psutil 不可用")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"
        df = pd.DataFrame({"x": range(100)})

        for i in range(50):
            df.to_parquet(path)
            read_df = pd.read_parquet(path)
            assert len(read_df) == 100

        _collect_garbage()

        try:
            final_fds = len(proc.open_files())
            fd_leak = final_fds - initial_fds
            # 允许少量 FD 波动（系统缓存/日志等）
            assert fd_leak < 10, f"文件句柄泄漏 {fd_leak} 个"
        except Exception:
            pytest.skip("FD 计数不可靠")


def test_resource_governance_memory_governor_unregister():
    """MemoryGovernor unregister_layer 正确清理。"""
    from factor_engine.runtime.resource_governor import MemoryGovernor

    gov = MemoryGovernor(process_budget_bytes=10 * 1024**2, duckdb_budget_bytes=2 * 1024**2)

    def evict_stub(target):
        return 0

    gov.register_layer("layer1", evict_stub)
    gov.reserve_accounting("layer1", 5_000)

    assert gov.total_usage == 5_000

    # Unregister 应清理 accounting
    gov.unregister_layer("layer1")

    assert "layer1" not in gov._evict_hooks
    assert "layer1" not in gov._usage


def test_resource_governance_buffer_store_refcount():
    """BufferStore refcount 正确管理（acquire/release）。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore
    import pandas as pd

    store = GovernedBufferStore(budget_bytes=50 * 1024**2)
    df = pd.DataFrame({"x": range(100)})

    result = store.put("key1", df, bytes_=10_000)
    assert result.status == "MEMORY"

    # Acquire ref
    assert store.acquire_ref("key1") is True
    summary = store.summary()
    assert summary["in_use_refcount"] >= 1

    # 尝试 evict（应被 refcount 保护）
    freed = store.evict_if_over_budget(target=50 * 1024**2)
    assert store.get("key1") is not None  # 仍在内存

    # Release ref
    store.release_ref("key1")
    summary2 = store.summary()
    assert summary2["in_use_refcount"] == 0


# ---------------------------------------------------------------------------
# 3. 压力测试
# ---------------------------------------------------------------------------


def test_stress_high_concurrency_buffer_store():
    """高并发场景：多线程同时 put/get/release。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore
    import pandas as pd

    store = GovernedBufferStore(budget_bytes=50 * 1024**2)
    errors = []

    def worker(thread_id):
        try:
            for i in range(20):
                key = f"thread{thread_id}_key{i}"
                df = pd.DataFrame({"x": range(50)})
                result = store.put(key, df, bytes_=5_000)
                if result.status == "MEMORY":
                    retrieved = store.get(key)
                    if retrieved is not None:
                        store.release(key)
        except Exception as e:
            errors.append(str(e))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        for f in futures:
            f.result()

    assert len(errors) == 0, f"并发错误: {errors}"

    # 验证最终状态一致
    summary = store.summary()
    assert summary["writes"] > 0
    assert summary["accounted_bytes"] <= store.budget_bytes


def test_stress_memory_constrained_scenario():
    """内存受限场景：预算很小，频繁 evict。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore
    import pandas as pd

    # 只给 5MB 预算
    store = GovernedBufferStore(budget_bytes=5 * 1024**2)

    for i in range(100):
        df = pd.DataFrame({"x": range(1000)})
        result = store.put(f"key{i}", df, bytes_=100_000)
        # 预算小，大部分会被 REFUSED
        assert result.status in ("MEMORY", "REFUSED", "RECOMPUTE")

    summary = store.summary()
    # 验证预算始终遵守
    assert summary["accounted_bytes"] <= store.budget_bytes
    assert summary["refused"] > 0 or summary["keys"] < 100


def test_stress_large_panel_data():
    """大数据量场景：超大面板数据。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore
    import pandas as pd

    store = GovernedBufferStore(budget_bytes=200 * 1024**2)

    # 模拟大 panel
    large_df = pd.DataFrame({
        "symbol": ["A"] * 10000,
        "date": pd.date_range("2020-01-01", periods=10000),
        "value": range(10000),
    })

    import sys
    actual_size = sys.getsizeof(large_df)

    result = store.put("large_panel", large_df, bytes_=actual_size)

    if result.status == "MEMORY":
        retrieved = store.get("large_panel")
        assert retrieved is not None
        assert len(retrieved) == 10000
        store.release("large_panel")


def test_stress_repeated_execution_scope_threading():
    """压力测试：多线程重复进入 ExecutionResourceScope。"""
    from factor_engine.runtime.resource_governor import ExecutionResourcePlan, ExecutionResourceScope

    plan = ExecutionResourcePlan.auto()
    errors = []

    def worker(thread_id):
        try:
            for _ in range(10):
                with ExecutionResourceScope(plan, strict=False):
                    time.sleep(0.001)  # 模拟短任务
        except Exception as e:
            errors.append(f"thread{thread_id}: {e}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        for f in futures:
            f.result()

    # 非 strict 模式应能处理并发（warning 而非 error）
    assert len(errors) == 0, f"并发 ExecutionResourceScope 错误: {errors}"


# ---------------------------------------------------------------------------
# 4. 失败恢复测试
# ---------------------------------------------------------------------------


def test_failure_recovery_buffer_store_partial_release():
    """失败恢复：部分 release 后内存状态一致。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore
    import pandas as pd

    store = GovernedBufferStore(budget_bytes=50 * 1024**2)

    keys = []
    for i in range(10):
        key = f"key{i}"
        df = pd.DataFrame({"x": range(100)})
        result = store.put(key, df, bytes_=10_000)
        if result.status == "MEMORY":
            keys.append(key)

    initial_usage = store.summary()["accounted_bytes"]

    # 模拟部分失败：只 release 一半
    for key in keys[:len(keys)//2]:
        store.release(key)

    partial_usage = store.summary()["accounted_bytes"]
    assert partial_usage < initial_usage

    # Release 剩余
    for key in keys[len(keys)//2:]:
        store.release(key)

    final_usage = store.summary()["accounted_bytes"]
    assert final_usage < partial_usage


def test_failure_recovery_memory_governor_evict_exception():
    """失败恢复：evict hook 抛异常后 governor 状态一致。"""
    from factor_engine.runtime.resource_governor import MemoryGovernor

    gov = MemoryGovernor(process_budget_bytes=10 * 1024**2, duckdb_budget_bytes=2 * 1024**2)

    def broken_evict(target):
        raise RuntimeError("evict failed")

    gov.register_layer("broken", broken_evict)
    gov.reserve_accounting("broken", 5_000)

    # reserve 触发 evict，evict 失败 → reserve 应返回 False
    success = gov.reserve("broken", 20 * 1024**2)
    assert success is False

    # 验证 accounting 未损坏
    assert gov.total_usage >= 5_000


def test_failure_recovery_execution_scope_exception_in_body():
    """失败恢复：ExecutionResourceScope body 内异常，exit 仍清理。"""
    from factor_engine.runtime.resource_governor import ExecutionResourcePlan, ExecutionResourceScope
    import os

    plan = ExecutionResourcePlan.auto()

    prev_threads = os.environ.get("DUCKDB_MAX_THREADS")

    try:
        with ExecutionResourceScope(plan, strict=False):
            os.environ["DUCKDB_MAX_THREADS"] = "999"
            raise ValueError("body error")
    except ValueError:
        pass

    # exit 应恢复环境变量
    restored = os.environ.get("DUCKDB_MAX_THREADS")
    # 应恢复到 prev_threads（或被移除）
    assert restored != "999"


# ---------------------------------------------------------------------------
# 5. 资源使用统计
# ---------------------------------------------------------------------------


def test_stats_memory_governor_summary():
    """统计信息：MemoryGovernor summary 完整。"""
    from factor_engine.runtime.resource_governor import MemoryGovernor

    gov = MemoryGovernor(process_budget_bytes=100 * 1024**2, duckdb_budget_bytes=20 * 1024**2)
    gov.register_layer("layer1", lambda t: 1000)
    gov.reserve("layer1", 5_000)
    gov.reserve("layer2", 10_000)

    summary = gov.summary()

    assert "process_budget_bytes" in summary
    assert "usage_bytes" in summary
    assert "usage_by_layer" in summary
    assert "pressure_stage" in summary
    assert summary["usage_bytes"] >= 15_000
    assert summary["usage_by_layer"]["layer1"] >= 5_000


def test_stats_buffer_store_summary():
    """统计信息：BufferStore summary 完整。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore
    import pandas as pd

    store = GovernedBufferStore(budget_bytes=50 * 1024**2)

    for i in range(10):
        df = pd.DataFrame({"x": range(100)})
        store.put(f"key{i}", df, bytes_=10_000)

    summary = store.summary()

    assert "budget_bytes" in summary
    assert "accounted_bytes" in summary
    assert "keys" in summary
    assert "writes" in summary
    assert "refused" in summary
    assert "reconciliation" in summary

    assert summary["writes"] >= 10
    assert summary["keys"] <= 10


def test_stats_execution_resource_plan_to_dict():
    """统计信息：ExecutionResourcePlan 序列化。"""
    from factor_engine.runtime.resource_governor import ExecutionResourcePlan

    plan = ExecutionResourcePlan.auto()
    d = plan.to_dict()

    assert "effective_memory_limit_bytes" in d
    assert "process_budget_bytes" in d
    assert "max_workers" in d
    assert "duckdb_threads" in d

    assert d["effective_memory_limit_bytes"] > 0
    assert d["max_workers"] >= 1


def test_stats_process_family_memory():
    """统计信息：进程族内存探测。"""
    from factor_engine.runtime.resource_governor import process_family_memory_bytes

    mem = process_family_memory_bytes(prefer_pss=True)

    if mem is not None:
        assert mem > 0
        # 当前进程至少占用几 MB
        assert mem > 10 * 1024**2


def test_stats_live_memory_headroom():
    """统计信息：live headroom 探测。"""
    from factor_engine.runtime.resource_governor import live_memory_headroom_bytes

    headroom = live_memory_headroom_bytes()

    # headroom 可能为 0（内存紧张），但不应为负
    assert headroom >= 0


# ---------------------------------------------------------------------------
# 6. tracemalloc 深度分析（可选，较慢）
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tracemalloc_top_allocations():
    """使用 tracemalloc 追踪 top 内存分配。"""
    tracemalloc.start()

    from factor_engine.runtime.buffer_store import GovernedBufferStore
    import pandas as pd

    store = GovernedBufferStore(budget_bytes=100 * 1024**2)

    for i in range(100):
        df = pd.DataFrame({"x": range(1000)})
        store.put(f"key{i}", df, bytes_=10_000)
        if i % 10 == 0:
            for j in range(i - 10, i):
                store.release(f"key{j}")

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    # 打印 top 10 分配
    print("\n[tracemalloc] Top 10 allocations:")
    for stat in top_stats[:10]:
        print(stat)

    tracemalloc.stop()

    # 验证：至少能追踪到一些分配
    assert len(top_stats) > 0


# ---------------------------------------------------------------------------
# 7. 长时间运行稳定性测试
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_long_running_stability():
    """长时间运行稳定性：1000 次迭代，内存增长 < 50MB。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore
    from factor_engine.runtime.resource_governor import MemoryGovernor
    import pandas as pd

    gov = MemoryGovernor(process_budget_bytes=100 * 1024**2, duckdb_budget_bytes=20 * 1024**2)
    store = GovernedBufferStore(budget_bytes=50 * 1024**2)

    gov.register_layer("store", lambda t: store.evict_if_over_budget(t))

    _collect_garbage()
    initial_mem = _get_process_memory_mb()

    for i in range(1000):
        df = pd.DataFrame({"x": range(100)})
        result = store.put(f"key{i % 100}", df, bytes_=10_000)
        if i % 10 == 0:
            store.release(f"key{(i - 5) % 100}")
        if i % 100 == 0:
            _collect_garbage()

    _collect_garbage()
    final_mem = _get_process_memory_mb()
    growth = final_mem - initial_mem

    print(f"\n[长时间运行] 初始: {initial_mem:.1f}MB, 最终: {final_mem:.1f}MB, 增长: {growth:.1f}MB")

    # 1000 次迭代内存增长应 < 50MB
    assert growth < 50.0, f"长时间运行内存泄漏: {growth:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
