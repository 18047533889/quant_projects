# -*- coding: utf-8 -*-
"""Regression tests for region limit accounting consistency.

Covers:
- Rebalance must update BOTH pool._limit_bytes AND RegionMemoryStats.limit_bytes
- New-region default limits must not oversell (fit within remaining budget)
- Rebalance keeps aggregate limits within total memory (no overselling)
"""
from __future__ import annotations

from runtime.multibackend.concurrent_region_isolation import (
    ConcurrentRegionIsolationManager,
)


def test_rebalance_updates_stats_limit_bytes() -> None:
    """Rebalance must update both pool._limit_bytes and stats.limit_bytes."""
    mgr = ConcurrentRegionIsolationManager(total_memory_bytes=1_000_000)
    pool_a = mgr.create_region("a")
    pool_b = mgr.create_region("b")

    # Sanity: both views initially agree
    assert pool_a._limit_bytes == pool_a.stats().limit_bytes
    assert pool_b._limit_bytes == pool_b.stats().limit_bytes

    mgr.rebalance_limits()
    fair_share = 1_000_000 // 2

    # The public stats view must reflect the rebalanced limit, not the stale one
    assert pool_a._limit_bytes == fair_share
    assert pool_a.stats().limit_bytes == fair_share
    assert pool_b.stats().limit_bytes == fair_share

    # stats() returns a copy, so the internal stats record must also be updated
    # (i.e. a fresh stats() call must still show the new limit)
    assert pool_a.stats().limit_bytes == fair_share


def test_new_region_default_limits_do_not_oversell() -> None:
    """Default region limits must fit within the remaining budget."""
    mgr = ConcurrentRegionIsolationManager(total_memory_bytes=100_000)
    pool_a = mgr.create_region("a")

    # First region gets full budget
    assert pool_a._limit_bytes == 100_000
    assert pool_a.stats().limit_bytes == 100_000

    # Second default region must not push the total over the budget
    pool_b = mgr.create_region("b")
    total_limit = pool_a._limit_bytes + pool_b._limit_bytes
    assert total_limit <= 100_000

    # Third default region keeps the sum within budget
    pool_c = mgr.create_region("c")
    total_limit = (
        pool_a._limit_bytes + pool_b._limit_bytes + pool_c._limit_bytes
    )
    assert total_limit <= 100_000


def test_rebalance_keeps_aggregate_within_total_memory() -> None:
    """After rebalance, the sum of all region limits must not exceed total."""
    mgr = ConcurrentRegionIsolationManager(total_memory_bytes=100_000)
    mgr.create_region("a")
    mgr.create_region("b")
    mgr.create_region("c")

    mgr.rebalance_limits()

    total_limit = sum(
        mgr.get_region(rid).stats().limit_bytes  # type: ignore[union-attr]
        for rid in ("a", "b", "c")
    )
    assert total_limit <= 100_000
