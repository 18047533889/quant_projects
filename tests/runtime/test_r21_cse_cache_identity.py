# -*- coding: utf-8 -*-
"""R21-CSE-CACHE-IDENTITY: Regression tests for CSE cache fail-closed governance.

Covers:
    1. Cache key binds full semantic identity (CSECacheKey) — same node id but
       different DataSnapshot/Universe/Market/DecisionClock/params/pi/contract
       do NOT reuse a cached value.
    2. size_bytes unknown/<=0 is REJECTED (CacheCapacityExceeded) — not 0, never
       silently unbounded.
    3. Updating an existing key removes the old key from _lru_order before
       re-append — no duplicate LRU entries / no size accounting leak.
    4. All-pinned + over-capacity raises CacheCapacityExceeded (backpressure) —
       never silently breaks the hard limit.
    5. LIRS (not implemented) fails closed with UnsupportedEvictionPolicy —
       never silently falls back to LRU.

Thread env vars: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import pytest

from factor_engine.runtime.multibackend.cse_cache_optimizer import (
    CSECacheKey,
    CSECacheOptimizer,
    CacheCapacityExceeded,
    UnsupportedEvictionPolicy,
)


def _key(
    logical_node="node_a",
    bound_params="p:default",
    data_source="ds:ashare",
    universe="uni:all",
    market="A",
    decision_clock="dc:t_close",
    physical_impl="pi:v3:abc",
    semantic_contract="sc:1",
) -> CSECacheKey:
    return CSECacheKey(
        logical_node=logical_node,
        bound_params=bound_params,
        data_source=data_source,
        universe=universe,
        market=market,
        decision_clock=decision_clock,
        physical_impl=physical_impl,
        semantic_contract=semantic_contract,
    )


# ---------------------------------------------------------------------------
# 1. Key binds full semantic identity
# ---------------------------------------------------------------------------
def test_same_node_id_different_semantic_dimension_do_not_reuse():
    """同一 shared node id 但任一语义维度不同 → 不能复用缓存值。"""
    cache = CSECacheOptimizer(max_cache_bytes=1024**2)

    base = _key()
    cache.put(base, "value_A", size_bytes=100)
    # 相同 key 命中
    assert cache.get(base) == "value_A"

    # 每个维度单独变化 → 必须 miss（不能复用）
    for field in [
        "bound_params",
        "data_source",
        "universe",
        "market",
        "decision_clock",
        "physical_impl",
        "semantic_contract",
    ]:
        cache.put(_key(**{field: "OTHER"}), f"other:{field}", size_bytes=100)
        assert cache.get(_key(**{field: "OTHER"})) == f"other:{field}", field

    # 原 key 仍是 base 的值（未被维度变化污染）
    assert cache.get(base) == "value_A"


def test_full_key_deterministic_and_distinct():
    """CSECacheKey.to_key/digest 确定性；任两维度不同即不同 key。"""
    a = _key()
    assert a.to_key() == _key().to_key()
    assert a.digest() == _key().digest()
    assert len(a.digest()) == 64  # full 256-bit sha256, no truncation

    distinct = [
        _key(market="US"),
        _key(universe="uni:CSI300"),
        _key(decision_clock="dm:t+2"),
        _key(physical_impl="pi:v3:def"),
        _key(semantic_contract="contract:2"),
        _key(data_source="ds:2"),
        _key(bound_params="p:w=20"),
    ]
    digests = {k.digest() for k in [a, *distinct]}
    assert len(digests) == 1 + len(distinct), "distinct dimensions must not collide"


# ---------------------------------------------------------------------------
# 2. size_bytes unknown -> rejected, never unbounded
# ---------------------------------------------------------------------------
def test_size_bytes_zero_rejected():
    cache = CSECacheOptimizer(max_cache_bytes=1024)
    with pytest.raises(CacheCapacityExceeded):
        cache.put(_key(), "v", size_bytes=0)


def test_size_bytes_none_rejected():
    cache = CSECacheOptimizer(max_cache_bytes=1024)
    with pytest.raises(CacheCapacityExceeded):
        cache.put(_key(), "v", size_bytes=None)


def test_size_bytes_negative_rejected():
    cache = CSECacheOptimizer(max_cache_bytes=1024)
    with pytest.raises(CacheCapacityExceeded):
        cache.put(_key(), "v", size_bytes=-5)


def test_size_bytes_known_positive_ok():
    cache = CSECacheOptimizer(max_cache_bytes=1024)
    cache.put(_key(), "v", size_bytes=100)
    assert cache.get(_key()) == "v"


# ---------------------------------------------------------------------------
# 3. Update removes old key from _lru_order (no duplicate LRU / no accounting leak)
# ---------------------------------------------------------------------------
def test_update_key_removes_old_lru_entry():
    cache = CSECacheOptimizer(max_cache_bytes=1024 << 10)
    k = _key()
    cache.put(k, "v1", size_bytes=100)
    cache.put(k, "v2", size_bytes=100)  # update existing key

    # key must appear exactly once in _lru_order
    occurrences = [i for i, x in enumerate(cache._lru_order) if x == k]
    assert len(occurrences) == 1, f"duplicate LRU entries: {cache._lru_order}"

    # size accounting: only one entry counted
    assert cache.metrics().total_size_bytes == 100


def test_update_then_evict_no_stale_lru_reference():
    """更新后逐出一致：_lru_order 无残留旧引用，逐出不会 IndexError/重复。"""
    cache = CSECacheOptimizer(max_cache_bytes=500)
    a = _key(logical_node="a")
    b = _key(logical_node="b")

    cache.put(a, "a", size_bytes=200)
    cache.put(b, "b", size_bytes=200)
    # 更新 a（旧 size 200 -> 新 200，总 400 <= 500），不应产生重复 LRU 项
    cache.put(a, "a2", size_bytes=200)

    assert len(cache._lru_order) == 2
    assert [k for k in cache._lru_order].count(a) == 1
    assert [k for k in cache._lru_order].count(b) == 1


# ---------------------------------------------------------------------------
# 4. All-pinned + over-capacity -> backpressure, never silent break
# ---------------------------------------------------------------------------
def test_all_pinned_over_capacity_raises():
    cache = CSECacheOptimizer(max_cache_bytes=500)
    a = _key(logical_node="a")
    b = _key(logical_node="b")

    cache.put(a, "a", size_bytes=300)
    cache.put(b, "b", size_bytes=200)
    cache.pin(a)
    cache.pin(b)

    # 新条目使总量超上限，但唯一可逐出项（全部 pinned）不可逐出 → fail-closed
    with pytest.raises(CacheCapacityExceeded):
        cache.put(_key(logical_node="c"), "c", size_bytes=300)


def test_all_pinned_memory_pressure_evict_raises():
    cache = CSECacheOptimizer(max_cache_bytes=1024)
    cache.put(_key(logical_node="a"), "a", size_bytes=400)
    cache.put(_key(logical_node="b"), "b", size_bytes=400)
    cache.pin(_key(logical_node="a"))
    cache.pin(_key(logical_node="b"))

    with pytest.raises(CacheCapacityExceeded):
        cache.evict_by_memory_pressure(target_bytes=100)


def test_unpin_allows_eviction_again():
    cache = CSECacheOptimizer(max_cache_bytes=500)
    a = _key(logical_node="a")
    b = _key(logical_node="b")
    cache.put(a, "a", size_bytes=300)
    cache.put(b, "b", size_bytes=200)
    cache.pin(a)
    cache.pin(b)

    with pytest.raises(CacheCapacityExceeded):
        cache.put(_key(logical_node="c"), "c", size_bytes=300)

    # 释放一个 pin → 可逐出
    cache.unpin(a)
    cache.put(_key(logical_node="c"), "c", size_bytes=300)  # 不再抛


# ---------------------------------------------------------------------------
# 5. LIRS not implemented -> fail closed, never LRU fallback
# ---------------------------------------------------------------------------
def test_lirs_raises_unsupported_eviction_policy():
    with pytest.raises(UnsupportedEvictionPolicy):
        CSECacheOptimizer(max_cache_bytes=1024, eviction_policy="lirs")


def test_unknown_eviction_policy_raises_not_lru_fallback():
    with pytest.raises(UnsupportedEvictionPolicy):
        CSECacheOptimizer(max_cache_bytes=1024, eviction_policy="mru")


def test_lru_and_lfu_still_valid():
    CSECacheOptimizer(max_cache_bytes=1024, eviction_policy="lru")
    CSECacheOptimizer(max_cache_bytes=1024, eviction_policy="lfu")
