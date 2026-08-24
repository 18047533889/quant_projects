# -*- coding: utf-8 -*-
"""R30-P0-005 / R30-P1-013 / R30-P1-014 测试。

覆盖：
    - SourceBlockCache LRU / evict / spillable 偏好 / stats
    - R30ReadSession：prepare_once 只 prepare 一次；source-block 命中不重复
      execute；CostModel 决策正确；prepared_cache key 含 security_scope；
      A/B 两个 session 的 resolution cache 不串（threading + ContextVar）
    - CacheHierarchy 层级完整 + invalidation_policy 绑定 security/snapshot
    - ExecutionLease acquire / release / child / deadline / envelope

全部用 fake store（可计 prepare/execute 调用次数），不依赖真实 Store/DB。
"""
from __future__ import annotations

import threading
import time

import pyarrow as pa
import pytest

from data_access.core.exceptions import ValidationError
from data_access.r30.session import (
    CostModel,
    R30ReadSession,
    SourceBlockCache,
    source_block_key,
)
from data_access.r30.cache_hierarchy import (
    CACHE_HIERARCHY,
    CacheLayer,
    cache_scope_id,
    invalidation_policy,
)
from data_access.r30.execution_lease import DEFAULT_ENVELOPE, ExecutionLease


# ---- fake store / prepared / result ----

class _FakeSnapshot:
    snapshot_id = "snap-1"
    content_digest = "digest-1"


class _UnsupportedSnapshot:
    pass


class _TypedSnapshot:
    def __init__(self, snapshot_id):
        self.snapshot_id = snapshot_id


class _FakePrepared:
    def __init__(self, dataset: str, security_digest: str | None = None) -> None:
        self.dataset = dataset
        self.security_digest = security_digest
        self.credential_scope_id = "cred-1"
        self.resolved_source_snapshot = _FakeSnapshot()
        self.physical_scope = ("p1.parquet",)
        self.request_identity = "req"


class _FakeResult:
    def __init__(self, rows: int = 10, cols: int = 3) -> None:
        self.table = pa.table({f"c{i}": [1] * rows for i in range(cols)})
        self.snapshot = _FakeSnapshot()

    def to_arrow(self):
        return self.table


class _FakeStore:
    def __init__(self, scope_digest: str = "scope-digest") -> None:
        self.scope_digest = scope_digest
        self.prepare_calls = 0
        self.execute_calls = 0
        self.lock_calendars_calls = 0

    def _effective_security_digest(self) -> str:
        return self.scope_digest

    def lock_calendars(self) -> None:
        self.lock_calendars_calls += 1

    def prepare_read(self, dataset: str, **kwargs):
        self.prepare_calls += 1
        return _FakePrepared(dataset, security_digest=self.scope_digest)

    def execute_prepared_read(self, prepared, *, verify_after: bool = True):
        self.execute_calls += 1
        return _FakeResult()


class _AlwaysMaterialize:
    """cost model 桩：永远物化（测缓存命中用）。"""

    def should_materialize(self, **kwargs):
        return "materialize"

    def estimate_memory(self, rows: int, columns: int) -> int:
        return rows * columns * 8


def _args():
    return dict(
        dataset="ds",
        columns=["a", "b"],
        time_range=("2024-01-01", "2024-01-02"),
        instrument_filter=["A", "B"],
        filters={"a": 1},
        params={"k": "v"},
    )


# ---- SourceBlockCache ----

def test_source_block_cache_lru_hit_stats():
    cache = SourceBlockCache(max_bytes=10_000, max_entries=16)
    cache.put("a", "va", size=10)
    cache.put("b", "vb", size=20)
    cache.put("c", "vc", size=30)
    assert cache.get("a") == "va"  # hit → MRU
    assert cache.get("zz") is None  # miss
    stats = cache.stats()
    assert stats["entries"] == 3
    assert stats["size"] == 60
    assert stats["hit"] == 1
    assert stats["miss"] == 1
    assert stats["reuse_count"] == 1  # only "a" touched
    # 再次 get("a") → reuse_count 再 +1，a 移到 MRU
    assert cache.get("a") == "va"
    assert cache.stats()["reuse_count"] == 2
    assert cache.stats()["hit"] == 2


def test_source_block_cache_lru_evict():
    cache = SourceBlockCache(max_bytes=100, max_entries=16)
    cache.put("a", "va", size=30)
    cache.put("b", "vb", size=30)
    cache.put("c", "vc", size=30)
    cache.get("a")  # a → MRU
    cache.put("d", "vd", size=40)  # 总 130 > 100 → evict LRU（b，再 c 若仍超）
    stats = cache.stats()
    # 30+30+30+40=130 → evict b(30) → 100 刚好 ≤ 100 停
    assert "b" not in cache._data
    assert cache.stats()["size"] <= 100
    assert stats["evictions"] == 1


def test_source_block_cache_spillable_not_evicted_first():
    cache = SourceBlockCache(max_bytes=100, max_entries=16)
    cache.put("keep", "vkeep", size=80, spillable=False)
    cache.put("a", "va", size=30)
    cache.put("b", "vb", size=30)
    # 总 140 > 100：先 evict spillable（a / b），keep(非 spillable) 保留
    assert "keep" in cache._data
    assert "a" not in cache._data or "b" not in cache._data
    # 只剩 keep 也不超过 max_bytes 就不会再动 keep
    assert cache.stats()["size"] <= 100


def test_source_block_cache_evict_until_bytes_and_clear():
    cache = SourceBlockCache(max_bytes=10_000)
    cache.put("a", "va", size=30)
    cache.put("b", "vb", size=30)
    cache.put("c", "vc", size=30)
    remaining = cache.evict(until_bytes=40)
    assert remaining <= 40
    assert cache.stats()["entries"] <= 2
    cache.clear()
    assert cache.stats()["entries"] == 0
    assert cache.stats()["size"] == 0
    assert cache.get("a") is None
    assert cache.stats()["hit"] == 0  # clear 重置计数


def test_source_block_key_stable_and_scoped():
    base = dict(
        dataset="ds",
        snapshot="snap-1",
        time_range=("2024-01-01", "2024-01-02"),
        universe=["A", "B"],
        canonical_fields=["open", "close"],
        filters={"a": 1},
        pit_contract="pit-c",
        security_digest="sec-A",
        price_basis="close",
    )
    k1 = source_block_key(**base)
    k2 = source_block_key(**base)
    assert k1 == k2
    assert isinstance(k1, str) and len(k1) == 64
    # 不同 security_digest → 不同 key
    k3 = source_block_key(**{**base, "security_digest": "sec-B"})
    assert k1 != k3
    # 不同 price_basis → 不同 key
    k4 = source_block_key(**{**base, "price_basis": "vwap"})
    assert k1 != k4
    # snapshot 对象身份提取
    k5 = source_block_key(**{**base, "snapshot": _FakeSnapshot()})
    assert k1 == k5


def test_source_block_key_uses_strict_canonical_identity_and_fails_closed():
    base = dict(
        dataset="ds",
        snapshot="snap-1",
        time_range=("2024-01-01", "2024-01-02"),
        universe=["A", "B"],
        canonical_fields=["open", "close"],
        filters={"a": 1},
        pit_contract="pit-c",
        security_digest="sec-A",
        price_basis="close",
    )
    assert source_block_key(**base) == source_block_key(**base)
    assert source_block_key(**base) != source_block_key(**{**base, "snapshot": "snap-2"})
    # CanonicalIdentityEncoder currently preserves sequence order but does not tag
    # list versus tuple; this test follows that authority's actual semantics.
    assert source_block_key(**base) == source_block_key(
        **{**base, "universe": ("A", "B")}
    )
    assert source_block_key(**base) != source_block_key(
        **{**base, "universe": ["B", "A"]}
    )
    assert source_block_key(**base) == source_block_key(
        **{**base, "snapshot": _TypedSnapshot("snap-1")}
    )
    assert source_block_key(**base) != source_block_key(
        **{**base, "snapshot": _TypedSnapshot(1)}
    )
    assert source_block_key(**{**base, "snapshot": _TypedSnapshot(1)}) != source_block_key(
        **{**base, "snapshot": _TypedSnapshot("1")}
    )
    assert source_block_key(**base) != source_block_key(
        **{**base, "filters": {"a": True}}
    )
    with pytest.raises(ValidationError, match="repr fallback"):
        source_block_key(**{**base, "snapshot": _UnsupportedSnapshot()})



def test_cost_model_decisions():
    cm = CostModel()
    # 高复用 → materialize（5×500=2500 > 1500）
    assert (
        cm.should_materialize(
            reuse_count=5,
            estimated_size=1000,
            rescan_cost=500,
            remote_cost=1500,
            available_memory=10**6,
            materialize_cost=1500,
        )
        == "materialize"
    )
    # 低复用 → rescan
    assert (
        cm.should_materialize(
            reuse_count=1,
            estimated_size=1000,
            rescan_cost=500,
            remote_cost=1500,
            available_memory=10**6,
            materialize_cost=1500,
        )
        == "rescan"
    )
    # 内存装不下 + 重扫比远端贵 → relation（不物化，保留 relation 复用）
    assert (
        cm.should_materialize(
            reuse_count=10,
            estimated_size=5000,
            rescan_cost=3000,
            remote_cost=500,
            available_memory=1000,
            materialize_cost=1500,
        )
        == "relation"
    )
    # 内存装不下 + 重扫不比远端贵 → rescan
    assert (
        cm.should_materialize(
            reuse_count=10,
            estimated_size=5000,
            rescan_cost=500,
            remote_cost=2000,
            available_memory=1000,
            materialize_cost=1500,
        )
        == "rescan"
    )
    assert cm.estimate_memory(100, 10) == 8000
    assert cm.estimate_memory(0, 10) == 0


# ---- R30ReadSession ----

def test_prepare_once_calls_prepare_only_once():
    store = _FakeStore()
    with R30ReadSession(store) as s:
        p1 = s.prepare_once(**_args())
        p2 = s.prepare_once(**_args())
        assert p1 is p2
        assert store.prepare_calls == 1
    assert store.lock_calendars_calls == 1


def test_prepared_cache_key_includes_security_scope():
    store = _FakeStore()
    with R30ReadSession(store) as s:
        default_scope = s._scope_digest
        assert default_scope  # resolve_execution_context 总是给一个 digest
        s.prepare_once(**_args())
        assert store.prepare_calls == 1
        # 同 scope 再问 → 命中
        s.prepare_once(**_args())
        assert store.prepare_calls == 1
        # 显式传与默认相同 scope digest → 与默认 key 相同 → 命中
        s.prepare_once(**_args(), security_digest=default_scope)
        assert store.prepare_calls == 1
        # 不同 security_digest → 不同 key → 重新 prepare
        s.prepare_once(**_args(), security_digest="other-scope")
        assert store.prepare_calls == 2


def test_read_source_block_hits_cache_no_execute():
    store = _FakeStore()
    session = R30ReadSession(store, cost_model=_AlwaysMaterialize())
    with session as s:
        t1 = s.read_source_block(**_args())
        assert store.execute_calls == 1
        assert store.prepare_calls == 1
        # 二次读：source-block 命中 → execute 不再调
        t2 = s.read_source_block(**_args())
        assert store.execute_calls == 1
        assert store.prepare_calls == 1
        assert t1 is t2
        assert isinstance(t1, pa.Table)
        # 缓存 stats 记录命中
        sc = s.source_block_cache.stats()
        assert sc["entries"] == 1
        assert sc["hit"] == 1
        assert sc["miss"] == 1


def test_read_source_block_no_materialize_not_cached():
    store = _FakeStore()
    session = R30ReadSession(store, cost_model=CostModel())
    # 默认 CostModel：一次请求 reuse_count=1 → rescan，不缓存
    with session as s:
        s.read_source_block(**_args())
        assert s.source_block_cache.stats()["entries"] == 0
        assert store.execute_calls == 1
        # 二次：prepared 命中但 source-block 未缓存 → 仍 execute
        s.read_source_block(**_args())
        assert store.execute_calls == 2
        assert s.source_block_cache.stats()["entries"] == 0


def test_read_source_block_default_cost_materializes_after_reuse():
    store = _FakeStore()
    session = R30ReadSession(store, cost_model=CostModel())
    # 默认 rescan_cost=size, materialize_cost=size*2 → reuse_count>2 时 materialize。
    # 前 2 次 rescan（reuse_count=1,2），第 3 次 materialize（reuse_count=3:
    # 3*size > 2*size），第 4 次命中缓存。
    with session as s:
        for _ in range(3):
            s.read_source_block(**_args())
        assert store.execute_calls == 3
        assert s.source_block_cache.stats()["entries"] == 1
        s.read_source_block(**_args())
        assert store.execute_calls == 3  # 命中缓存，不再 execute
        assert s.source_block_cache.stats()["hit"] == 1


def test_read_source_block_expected_fields():
    store = _FakeStore()
    session = R30ReadSession(store, cost_model=_AlwaysMaterialize())
    with session as s:
        t = s.read_source_block(**_args(), expected_fields=["c0"])
        assert t is not None
        with pytest.raises(ValueError):
            s.read_source_block(**_args(), expected_fields=["missing-field"])


def test_session_contextvar_isolation_a_b():
    from data_access.runtime.read_session_context import get_resolution_cache

    store = _FakeStore()
    sA = R30ReadSession(store)
    sB = R30ReadSession(store)
    results: dict[str, bool] = {}

    with sA:
        sA.resolution_cache["a"] = 1
        # A 线程当前 ContextVar 绑定 A 的 dict
        assert get_resolution_cache() is sA.resolution_cache

        def worker():
            with sB:
                # B 线程 ContextVar 绑定 B 的 dict（A 的 cache 不串进来）
                results["b_cache_is_sB"] = get_resolution_cache() is sB.resolution_cache
                results["b_has_a_key"] = "a" in get_resolution_cache()
                sB.resolution_cache["b"] = 2

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        # B 退出后 A 线程 ContextVar 仍是 A 的 dict
        results["a_cache_still_sA"] = get_resolution_cache() is sA.resolution_cache
        results["a_not_leaked_b"] = "b" not in sA.resolution_cache

    # A、B 各自的 dict 不串
    assert results["b_cache_is_sB"] is True
    assert results["b_has_a_key"] is False
    assert results["a_cache_still_sA"] is True
    assert results["a_not_leaked_b"] is True
    assert sA.resolution_cache is not sB.resolution_cache


def test_session_stats_and_read_convenience():
    store = _FakeStore()
    with R30ReadSession(store, cost_model=_AlwaysMaterialize()) as s:
        s.read(**_args())
        s.read_source_block(**_args())
        stats = s.stats()
        assert stats["prepared_cache"]["hits"] == 1
        assert stats["prepared_cache"]["misses"] == 1
        assert stats["source_block_cache"]["entries"] == 1
        assert stats["source_block_cache"]["hit"] == 1


def test_session_closed_cannot_enter_again():
    store = _FakeStore()
    s = R30ReadSession(store)
    with s:
        pass
    with pytest.raises(RuntimeError):
        with s:
            pass


# ---- CacheHierarchy ----

def test_cache_hierarchy_layers_complete():
    assert len(CACHE_HIERARCHY) == 5
    assert [l.level for l in CACHE_HIERARCHY] == ["L0", "L1", "L2", "L3", "L4"]
    for layer in CACHE_HIERARCHY:
        assert isinstance(layer, CacheLayer)
        assert layer.purpose
        assert layer.identity
        assert layer.security_scope
        assert layer.snapshot_scope
        assert layer.invalidation_trigger
    # L1 是源块缓存（本 R30 关注层）
    assert "source-block" in CACHE_HIERARCHY[1].purpose
    assert "dataset+snapshot+security+price_basis" == CACHE_HIERARCHY[1].identity


def test_invalidation_policy_binds_security_and_snapshot():
    for layer in CACHE_HIERARCHY:
        policy = invalidation_policy(layer)
        assert policy["security_bound"] is True
        assert policy["snapshot_bound"] is True
        assert policy["invalidation_trigger"]
        assert "freshness_authority" in policy


def test_cache_scope_id_stable_and_scoped():
    storeA = _FakeStore(scope_digest="A")
    storeB = _FakeStore(scope_digest="B")
    layer = CACHE_HIERARCHY[1]
    c1 = cache_scope_id(layer, ["ds", "snap-1"], storeA)
    c2 = cache_scope_id(layer, ["ds", "snap-1"], storeA)
    assert c1 == c2
    assert len(c1) == 64  # stable_digest_full
    # 不同 security scope → 不同缓存 scope（绝不跨 scope 共享）
    assert c1 != cache_scope_id(layer, ["ds", "snap-1"], storeB)
    # 不同 identity（snapshot）→ 不同 scope
    assert c1 != cache_scope_id(layer, ["ds", "snap-2"], storeA)


# ---- ExecutionLease ----

def _lease(lease_id: str = "root", deadline: float | None = None):
    return ExecutionLease(
        lease_id=lease_id,
        memory=1000,
        scan_bytes=500,
        remote_slots=4,
        duckdb_slots=2,
        temp_disk=1000,
        spill_budget=200,
        absolute_deadline=deadline,
    )


def test_execution_lease_acquire_release_remaining():
    lease = _lease()
    envelope = {
        "memory": 1000,
        "scan_bytes": 500,
        "remote_slots": 4,
        "duckdb_slots": 2,
        "temp_disk": 1000,
        "spill_budget": 200,
    }
    assert lease.acquire(envelope) is lease
    assert lease.remaining(time.monotonic()) == float("inf")
    d = lease.to_dict()
    assert d["acquired"] is True
    assert d["released"] is False
    assert d["memory"] == 1000
    lease.release()
    assert lease.to_dict()["released"] is True
    lease.release()  # 幂等
    assert lease.to_dict()["released"] is True


def test_execution_lease_acquire_over_envelope_fails_closed():
    lease = _lease()
    with pytest.raises(RuntimeError):
        lease.acquire({"memory": 500})  # 请求 1000 > 500
    lease2 = _lease()
    lease2.acquire({"memory": 1000})
    with pytest.raises(RuntimeError):
        lease2.acquire({"memory": 1000})  # 二次 acquire


def test_execution_lease_request_child():
    lease = _lease()
    lease.acquire({"memory": 1000, "scan_bytes": 500})
    child = lease.request_child({"memory": 100, "scan_bytes": 50})
    assert isinstance(child, ExecutionLease)
    assert child.lease_id == "root/child-1"
    assert child.to_dict()["memory"] == 100
    assert child.to_dict()["scan_bytes"] == 50
    # 父租约剩余扣减
    parent_state = lease.to_dict()
    assert parent_state["remaining"]["memory"] == 900
    assert parent_state["remaining"]["scan_bytes"] == 450
    # 子租约继承 deadline
    assert child.absolute_deadline == lease.absolute_deadline
    child.release()
    lease.release()
    assert lease.to_dict()["children"] == []


def test_execution_lease_request_child_insufficient():
    lease = _lease()
    lease.acquire({"memory": 1000, "scan_bytes": 500})
    with pytest.raises(RuntimeError):
        lease.request_child({"memory": 999_999})
    with pytest.raises(ValueError):
        lease.request_child({"cpu": 4})  # 未知维度
    lease.release()
    with pytest.raises(RuntimeError):
        lease.request_child({"memory": 10})  # 父已释放


def test_execution_lease_deadline():
    deadline = time.monotonic() + 10.0
    lease = _lease(deadline=deadline)
    assert lease.remaining(time.monotonic()) > 0
    assert lease.remaining(deadline + 5) <= 0  # 已过 deadline


def test_execution_lease_from_resource_envelope():
    env = {
        "memory_bytes": 2000,
        "scan_bytes": 1000,
        "remote_slots": 8,
        "duckdb_slots": 4,
        "temp_disk_bytes": 5000,
        "spill_budget_bytes": 300,
    }
    lease = ExecutionLease.from_resource_envelope(env)
    d = lease.to_dict()
    assert d["memory"] == 2000
    assert d["scan_bytes"] == 1000
    assert d["remote_slots"] == 8
    assert d["duckdb_slots"] == 4
    assert d["temp_disk"] == 5000
    assert d["spill_budget"] == 300
    # envelope 不可用 → 保守默认值（不抛）
    default_lease = ExecutionLease.from_resource_envelope(None)
    assert default_lease.memory == DEFAULT_ENVELOPE["memory"]
    assert default_lease.remote_slots == DEFAULT_ENVELOPE["remote_slots"]
