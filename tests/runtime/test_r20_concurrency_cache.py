# -*- coding: utf-8 -*-
"""R20 并发 / 资源治理 / 缓存正确性 regression tests.

覆盖：
    - R20-461  admission probe 无 ``__probe__`` ghost accounting
    - R20-462  ``check_pre_warmup`` 只在 normal 档返回 True（stop_warmup 及以上禁止）
    - R20-463  ``effective_cpu_slots`` 受 hard CPU limit clamp（容器 2 CPU / explicit 64 → 2）
    - R20-464  并发 Polars lazy/non-lazy：最终 source state == initial，结果 deterministic
    - R20-465  空 MultiIndex Series 持久化 round-trip 保留轴语义
    - R20-468  仅修改 lowering constant → lowering semantic hash 变化 → cache invalidated
    - LRU/MRU  overwrite 已有 key 移到 MRU（plain dict / OrderedDict）
    - MemoryGovernor / CacheManager 多线程 set/get/overwrite/evict/clear/scope stress
    - accounting invariant：``cache._bytes == sum(resident sizes)`` == ``governor.usage[layer]``
    - R20-114..118  ExecutionContext wrapper 线程隔离（不 setattr 到共享 inner）
    - ExecutionResourceScope ``strict`` 参数不再从 env 猜
"""
from __future__ import annotations

import threading

import pytest

# ---------------------------------------------------------------------------
# R20-461：check_admit / can_admit 无副作用，不留 __probe__ ghost accounting
# ---------------------------------------------------------------------------


def test_admission_probe_no_ghost():
    from factor_engine.runtime.resource_governor import MemoryGovernor

    gov = MemoryGovernor(process_budget_bytes=10_000, duckdb_budget_bytes=1_000)
    gov.register_layer("none", lambda target: 0)  # evict hook 不释放任何东西
    gov.reserve_accounting("real", 5_000)
    for _ in range(1000):
        # 5000 + 6000 > 10000 且 evict 无效 → 拒绝
        assert gov.check_admit(6_000) is False
    assert "__probe__" not in gov._usage
    assert gov.total_usage == 5_000  # 不被探测污染

    for _ in range(1000):
        assert gov.can_admit(6_000) is False
    assert "__probe__" not in gov._usage
    assert gov.total_usage == 5_000
    # 成功路径也不记账
    assert gov.can_admit(1_000) is True
    assert "__probe__" not in gov._usage
    assert gov.total_usage == 5_000


def test_check_admit_can_evict_real_layer():
    from factor_engine.runtime.resource_governor import MemoryGovernor

    gov = MemoryGovernor(process_budget_bytes=10_000, duckdb_budget_bytes=1_000)
    gov.register_layer("lru", lambda target: 4_000)  # evict 释放 4000
    gov.reserve_accounting("lru", 6_000)
    # 6000 + 5000 > 10000 → evict 4000 → 6000 + 1000 <= 10000 成立
    assert gov.check_admit(5_000) is True
    assert "__probe__" not in gov._usage
    assert gov.total_usage == 2_000  # 6000 - 4000


# ---------------------------------------------------------------------------
# R20-462：check_pre_warmup 只在 normal 档返回 True
# ---------------------------------------------------------------------------


def _gov_with_rss(budget: int):
    from factor_engine.runtime.resource_governor import MemoryGovernor

    rss = {"v": 0}
    gov = MemoryGovernor(
        process_budget_bytes=budget,
        duckdb_budget_bytes=budget // 10,
        rss_probe=lambda: rss["v"],
    )
    return gov, rss


def _set_frac(rss, budget, frac):
    rss["v"] = int(budget * frac)


def test_check_pre_warmup_only_normal():
    from factor_engine.runtime.resource_governor import _STAGE_STOP_WARMUP

    budget = 1_000_000
    gov, rss = _gov_with_rss(budget)

    # 低于 stop_warmup 阈值 → normal → 允许预热
    _set_frac(rss, budget, _STAGE_STOP_WARMUP - 1e-6)
    assert gov.pressure_stage() == "normal"
    assert gov.check_pre_warmup() is True

    # 等于阈值 → stop_warmup → 禁止
    _set_frac(rss, budget, _STAGE_STOP_WARMUP)
    assert gov.pressure_stage() == "stop_warmup"
    assert gov.check_pre_warmup() is False

    # 高于阈值 → 仍然 stop_warmup（下一个档位是 evict_lru）
    _set_frac(rss, budget, _STAGE_STOP_WARMUP + 0.05)
    assert gov.pressure_stage() == "stop_warmup"
    assert gov.check_pre_warmup() is False


def test_pressure_stage_all_threshold_boundaries():
    from factor_engine.runtime.resource_governor import (
        _STAGE_CRITICAL,
        _STAGE_EVICT_LRU,
        _STAGE_SPILL,
        _STAGE_STOP_WARMUP,
        _STAGE_THROTTLE,
    )

    budget = 1_000_000
    gov, rss = _gov_with_rss(budget)

    def stage_at(frac: float) -> str:
        _set_frac(rss, budget, frac)
        return gov.pressure_stage()

    # -epsilon / exact / +epsilon 行为验证
    assert stage_at(_STAGE_STOP_WARMUP - 1e-6) == "normal"
    assert stage_at(_STAGE_STOP_WARMUP) == "stop_warmup"
    assert stage_at(_STAGE_STOP_WARMUP + 1e-6) == "stop_warmup"

    assert stage_at(_STAGE_EVICT_LRU - 1e-6) == "stop_warmup"
    assert stage_at(_STAGE_EVICT_LRU) == "evict_lru"
    assert stage_at(_STAGE_EVICT_LRU + 1e-6) == "evict_lru"

    assert stage_at(_STAGE_SPILL - 1e-6) == "evict_lru"
    assert stage_at(_STAGE_SPILL) == "spill"
    assert stage_at(_STAGE_SPILL + 1e-6) == "spill"

    assert stage_at(_STAGE_THROTTLE - 1e-6) == "spill"
    assert stage_at(_STAGE_THROTTLE) == "throttle"
    assert stage_at(_STAGE_THROTTLE + 1e-6) == "throttle"

    assert stage_at(_STAGE_CRITICAL - 1e-6) == "throttle"
    assert stage_at(_STAGE_CRITICAL) == "critical"


# ---------------------------------------------------------------------------
# R20-463：effective_cpu_slots 受 hard CPU limit clamp
# ---------------------------------------------------------------------------


def test_effective_cpu_slots_hard_clamp():
    from factor_engine.runtime.resource_governor import effective_cpu_slots

    # 容器 2 CPU / explicit 64 → 2（不能再 explicit 直接 return）
    assert effective_cpu_slots(explicit=64, hard_limit=2) == 2
    # env 同样被 clamp
    assert effective_cpu_slots(env="FACTOR_ENGINE_CPU_BUDGET", hard_limit=2) == 2
    assert effective_cpu_slots(explicit=1, hard_limit=2) == 1
    assert effective_cpu_slots(hard_limit=2) == 2
    # hard_limit 兜底探测
    from factor_engine.runtime.resource_governor import _probe_hard_cpu_limit

    assert effective_cpu_slots() == _probe_hard_cpu_limit()


# ---------------------------------------------------------------------------
# R20-464：Polars lazy/non-lazy 并发线程安全（per-thread override）
# ---------------------------------------------------------------------------


def test_lazy_scan_thread_safety_final_state():
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    s = DataAccessSource(dataset="dummy", params={"lazy_scan": False, "read_auto": False})
    initial_lazy = s.lazy_scan
    initial_auto = s.read_auto
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker():
        try:
            barrier.wait()
            s.enable_lazy_scan(True)
            # 本线程视图为 lazy
            assert s.lazy_scan is True
            assert s.read_auto is True
            # 模拟 polars_backend finally 中的 restore
            s._lazy_scan = initial_lazy
            s.read_auto = initial_auto
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # 最终 source state == initial（主线程视图不被 worker 污染）
    assert s.lazy_scan == initial_lazy
    assert s.read_auto == initial_auto


def test_lazy_scan_mixed_lazy_nonlazy_deterministic():
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    s = DataAccessSource(dataset="dummy", params={"lazy_scan": False, "read_auto": False})
    barrier = threading.Barrier(6)
    errors: list[BaseException] = []

    def worker(lazy: bool):
        try:
            barrier.wait()
            s.enable_lazy_scan(lazy)
            v = (s.lazy_scan, s.read_auto)
            if lazy:
                assert v == (True, True)
            else:
                assert v == (False, False) or (False, True)
            s._lazy_scan = False
            s.read_auto = False
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i % 2 == 0,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert s.lazy_scan is False
    assert s.read_auto is False


# ---------------------------------------------------------------------------
# R20-465：空 MultiIndex Series 持久化 round-trip 保留轴语义
# ---------------------------------------------------------------------------


def test_persistent_cache_empty_multiindex_roundtrip(tmp_path):
    import pandas as pd

    from factor_engine.storage.cache import PersistentPlanCache

    idx = pd.MultiIndex.from_arrays([[], []], names=["timestamp", "instrument"])
    series = pd.Series(dtype="float64", index=idx)
    root = tmp_path / "plan_cache"
    cache = PersistentPlanCache(root, data_scope="s1")
    cache.set("empty_key", series)

    reloaded = PersistentPlanCache(root, data_scope="s1")
    hit = reloaded.get("empty_key")
    assert hit is not None
    assert isinstance(hit.index, pd.MultiIndex), "空 Series 必须保留 MultiIndex 轴"
    assert list(hit.index.names) == ["timestamp", "instrument"]
    assert len(hit) == 0


def test_persistent_cache_empty_single_index_roundtrip(tmp_path):
    import pandas as pd

    from factor_engine.storage.cache import PersistentPlanCache

    series = pd.Series(
        dtype="float64",
        index=pd.DatetimeIndex([], name="timestamp").tz_localize("UTC"),
    )
    root = tmp_path / "plan_cache2"
    cache = PersistentPlanCache(root, data_scope="s1")
    cache.set("empty_ts", series)
    reloaded = PersistentPlanCache(root, data_scope="s1")
    hit = reloaded.get("empty_ts")
    assert hit is not None
    assert len(hit) == 0
    assert getattr(hit.index, "name", None) == "timestamp"
    assert getattr(hit.index, "tz", None) is not None


# ---------------------------------------------------------------------------
# R20-468：lowering semantic hash 变化 → cache invalidated
# ---------------------------------------------------------------------------


def test_lowering_semantic_hash_cache_invalidation(tmp_path):
    import factor_engine.storage.cache as cache_mod
    from factor_engine.storage.cache import PersistentPlanCache, _compiler_semantic_hash

    root = tmp_path / "plan_cache3"
    ns_before = PersistentPlanCache(root, data_scope="s1")._namespace_root()
    h_before = _compiler_semantic_hash()

    original = cache_mod._LOWERING_SEMANTIC_VERSION
    try:
        cache_mod._LOWERING_SEMANTIC_VERSION = "2"
        ns_after = PersistentPlanCache(root, data_scope="s1")._namespace_root()
        h_after = _compiler_semantic_hash()
        assert h_after != h_before
        assert ns_after != ns_before
    finally:
        cache_mod._LOWERING_SEMANTIC_VERSION = original


def test_optimizer_semantic_version_cache_invalidation(tmp_path):
    import factor_engine.storage.cache as cache_mod
    from factor_engine.storage.cache import PersistentPlanCache, _compiler_semantic_hash

    root = tmp_path / "plan_cache4"
    ns_before = PersistentPlanCache(root, data_scope="s1")._namespace_root()
    h_before = _compiler_semantic_hash()
    original = cache_mod._OPTIMIZER_COMPILER_SEMANTIC_VERSION
    try:
        cache_mod._OPTIMIZER_COMPILER_SEMANTIC_VERSION = "2"
        assert _compiler_semantic_hash() != h_before
        assert PersistentPlanCache(root, data_scope="s1")._namespace_root() != ns_before
    finally:
        cache_mod._OPTIMIZER_COMPILER_SEMANTIC_VERSION = original


# ---------------------------------------------------------------------------
# LRU / MRU：overwrite 已有 key 移到 MRU
# ---------------------------------------------------------------------------


def test_cache_manager_overwrite_moves_to_mru():
    from factor_engine.storage.cache import CacheManager

    cache = CacheManager(budget_bytes=1024 * 1024)
    cache.set("a", [1.0])
    cache.set("b", [2.0])
    cache.set("c", [3.0])
    # 直接覆盖 a → a 应移到 MRU（末尾），b 成为 LRU（开头）
    cache.set("a", [4.0])
    assert list(cache._cache.keys()) == ["b", "c", "a"], "overwrite 必须把 key 移到 MRU"


def test_panel_cache_overwrite_moves_to_mru():
    import pandas as pd

    from factor_engine.cache.panel_cache import PanelCache

    cache = PanelCache(budget_bytes=1024 * 1024)
    cache.set("a", pd.Series([1.0]))
    cache.set("b", pd.Series([2.0]))
    cache.set("a", pd.Series([3.0]))
    assert list(cache._store.keys()) == ["b", "a"], "PanelCache overwrite 必须 move_to_end"


def test_expression_cache_overwrite_moves_to_mru():
    import pandas as pd

    from factor_engine.cache.expression_cache import ExpressionCache

    cache = ExpressionCache(budget_bytes=1024 * 1024)
    cache.set("s1", pd.Series([1.0]))
    cache.set("s2", pd.Series([2.0]))
    cache.set("s1", pd.Series([3.0]))
    assert list(cache._store.keys()) == ["s2", "s1"], "ExpressionCache overwrite 必须 move_to_end"


# ---------------------------------------------------------------------------
# MemoryGovernor / CacheManager 多线程 stress + accounting invariant
# ---------------------------------------------------------------------------


def test_cache_bytes_accounting_invariant():
    import factor_engine.runtime.resource_governor as rrg
    from factor_engine.runtime.resource_governor import estimate_object_bytes

    from factor_engine.storage.cache import CacheManager

    cache = CacheManager(budget_bytes=100_000, layer_name="invariant")
    cache.set("a", [0.0] * 10)
    cache.set("b", [0.0] * 20)
    cache.set("a", [0.0] * 30)  # overwrite
    resident = sum(estimate_object_bytes(v) for v in cache._cache.values())
    assert cache._bytes == resident
    assert cache._bytes >= 0
    # governor accounting 与 resident 一致（层名注册）
    gov = rrg.global_memory_governor()
    assert gov._usage.get("invariant", 0) == resident


def test_memory_governor_cache_concurrent_stress():
    import random

    import factor_engine.runtime.resource_governor as rrg
    from factor_engine.runtime.resource_governor import MemoryGovernor, estimate_object_bytes

    from factor_engine.storage.cache import CacheManager

    original = rrg._GLOBAL_GOVERNOR
    try:
        gov = MemoryGovernor(process_budget_bytes=300_000, duckdb_budget_bytes=50_000)
        rrg._GLOBAL_GOVERNOR = gov
        cache = CacheManager(budget_bytes=120_000, layer_name="stress")
        gov.register_layer("stress", cache.evict_if_over_budget)

        barrier = threading.Barrier(8)
        errors: list[BaseException] = []

        def worker(seed: int):
            rng = random.Random(seed)
            try:
                barrier.wait()
                for _ in range(1500):
                    op = rng.random()
                    key = f"k{rng.randint(0, 30)}"
                    val = [0.0] * rng.randint(1, 400)
                    if op < 0.5:
                        cache.set(key, val)
                    elif op < 0.7:
                        cache.get(key)
                    elif op < 0.85:
                        # 通过 governor 触发逐出（走 ``_evict_for`` 统一记账路径）
                        gov.can_admit(rng.randint(1, 50_000))
                    else:
                        cache.clear_memory()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        # 不漂移 / 非负 / accounting invariant（锁内快照避免读到半更新状态）
        with gov.lock:
            resident = sum(estimate_object_bytes(v) for v in cache._cache.values())
            assert cache._bytes >= 0
            assert cache._bytes == resident
            assert gov._usage.get("stress", 0) == resident
    finally:
        rrg._GLOBAL_GOVERNOR = original


def test_expression_cache_release_evict_lock_order():
    """set/release/evict 并发不得锁反转死锁（governor → cache 顺序恒一）。"""
    import random

    import pandas as pd

    from factor_engine.cache.expression_cache import ExpressionCache
    from factor_engine.runtime.resource_governor import MemoryGovernor

    gov = MemoryGovernor(process_budget_bytes=50_000, duckdb_budget_bytes=5_000)
    cache = ExpressionCache(budget_bytes=30_000, layer_name="expr_stress")
    gov.register_layer("expr_stress", cache.evict_if_over_budget)

    barrier = threading.Barrier(6)
    errors: list[BaseException] = []

    def worker(seed: int):
        rng = random.Random(seed)
        try:
            barrier.wait()
            for _ in range(1500):
                op = rng.random()
                sid = f"s{rng.randint(0, 10)}"
                if op < 0.5:
                    cache.set(sid, pd.Series([1.0] * rng.randint(1, 200)))
                elif op < 0.7:
                    cache.release(sid)
                else:
                    cache.evict_if_over_budget(rng.randint(0, 30_000))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # 无死锁即通过；记账非负
    assert cache._bytes >= 0
    assert gov._usage.get("expr_stress", 0) >= 0


def test_memory_governor_reserve_release_thread_safety():
    from factor_engine.runtime.resource_governor import MemoryGovernor

    gov = MemoryGovernor(process_budget_bytes=1_000_000, duckdb_budget_bytes=100_000)
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker():
        try:
            barrier.wait()
            for _ in range(2000):
                if gov.reserve("t", 100):
                    gov.release("t", 100)
                gov.reserve_accounting("a", 10)
                gov.release_accounting("a", 10)
                gov.throttle("stop_warmup")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # 记账最终平衡（a 层净 0）
    assert gov._usage.get("a", 0) == 0
    assert gov.total_usage == 0
    # 并发 register/unregister 不炸
    gov2 = MemoryGovernor(process_budget_bytes=100_000, duckdb_budget_bytes=10_000)

    def reg_worker():
        for i in range(500):
            gov2.register_layer(f"l{i}", lambda t: 0)
            gov2.unregister_layer(f"l{i}")

    regs = [threading.Thread(target=reg_worker) for _ in range(4)]
    for t in regs:
        t.start()
    for t in regs:
        t.join()
    assert len(gov2._evict_hooks) == 0


# ---------------------------------------------------------------------------
# R20-114..118：ExecutionContext wrapper 线程隔离（不 setattr 到共享 inner）
# ---------------------------------------------------------------------------


def test_execution_context_wrapper_thread_isolation():
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    inner = DataAccessSource(dataset="dummy", params={})
    barrier = threading.Barrier(4)
    results: dict[str, bool] = {}
    errors: list[BaseException] = []

    def worker(tid: int):
        try:
            barrier.wait()
            ctx = ExecutionContext(data_source=inner)
            results[f"is_wrapper_{tid}"] = ctx.data_source is not inner
            results[f"has_logical_wrapper_{tid}"] = ctx._logical_wrapper is ctx.data_source
            # 绝不能把 pointer setattr 到共享 inner 的实例 dict
            results[f"shared_attr_{tid}"] = "_factor_engine_lqtp_wrapper" in inner.__dict__
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for tid in range(4):
        assert results[f"is_wrapper_{tid}"] is True
        assert results[f"has_logical_wrapper_{tid}"] is True
        assert results[f"shared_attr_{tid}"] is False


def test_lineage_wrapper_thread_isolation():
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.storage.sources.data_access_source import (
        DataAccessSource,
        _logical_wrapper_for,
    )

    inner = DataAccessSource(dataset="dummy", params={})
    barrier = threading.Barrier(4)
    results: dict[int, bool] = {}
    errors: list[BaseException] = []

    def worker(tid: int):
        try:
            barrier.wait()
            ctx = ExecutionContext(data_source=inner)
            # 模拟 lineage_service._logical_wrapper 同线程查找
            found = _logical_wrapper_for(inner)
            results[tid] = found is ctx._logical_wrapper
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for tid in range(4):
        assert results[tid] is True, f"thread {tid} 应看到自己的 logical wrapper"


# ---------------------------------------------------------------------------
# R20-132..137：ExecutionResourceScope 显式 strict，不从 env 猜
# ---------------------------------------------------------------------------


def test_execution_resource_scope_strict_explicit():
    from factor_engine.runtime.resource_governor import ExecutionResourceScope

    assert ExecutionResourceScope(strict=True).strict is True
    assert ExecutionResourceScope(strict=False).strict is False


def test_execution_resource_scope_strict_env_fallback(monkeypatch):
    from factor_engine.runtime.resource_governor import ExecutionResourceScope

    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    assert ExecutionResourceScope().strict is True
    monkeypatch.delenv("FACTOR_ENGINE_RUN_MODE", raising=False)
    assert ExecutionResourceScope().strict is False
    # 显式 strict 优先于 env
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    assert ExecutionResourceScope(strict=False).strict is False


# ---------------------------------------------------------------------------
# R20-111：corrupted-state marker（research warning / production abort）
# ---------------------------------------------------------------------------


def test_corrupted_state_marker():
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    s = DataAccessSource(dataset="dummy", params={})
    assert s._corrupted_state is None
    s._mark_corrupted("restore failed")
    assert s._corrupted_state == "restore failed"
    # research：warning 不抛
    s.run_mode = "research"
    s._assert_healthy()
    # production：必须 abort
    s.run_mode = "production"
    with pytest.raises(RuntimeError):
        s._assert_healthy()


def test_restore_on_closed_source_poisons():
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    s = DataAccessSource(dataset="dummy", params={"lazy_scan": True})
    s.close()
    # 模拟 polars_backend restore：直接回写 _lazy_scan/read_auto（不经 _assert_open）
    s._lazy_scan = False
    s.read_auto = False
    assert s._corrupted_state is not None, "closed-source restore 必须记录 corrupted-state"
    # production 下后续使用 abort
    s.run_mode = "production"
    with pytest.raises(RuntimeError):
        s._assert_open()
