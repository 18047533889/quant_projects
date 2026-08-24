# -*- coding: utf-8 -*-
"""R21 regression: token preemption actually cancels running tasks + memory budget fix.

Covers:
1. Preemption carries a RevocationToken; the preempted task checks generation at
   checkpoint and stops. (Direct preempt_tokens path + automatic preempt path.)
2. Memory budget = min(cgroup/host available, hard_limit) — never the old x4
   amplification of hard_memory_limit.
3. Negative cpu_tokens / memory_bytes are rejected (ValueError) instead of being
   silently clamped.
4. Startup ResourceBudget detects cgroup CPU quota / cpuset / cores / RAM.
"""
from __future__ import annotations

import pytest

from factor_engine.runtime.multibackend.concurrent_token_manager import (
    ConcurrentTokenManager,
    RevocationToken,
    TokenReservation,
)
from factor_engine.runtime.resource_governor import (
    detect_resource_budget,
    effective_cpu_slots,
    effective_memory_limit_bytes,
    resolve_token_budget,
)


def test_memory_budget_is_never_4x_hard_limit() -> None:
    """memory budget must equal detected hard limit, never a 4x amplification."""
    detected = effective_memory_limit_bytes()
    mgr = ConcurrentTokenManager(memory_budget_bytes=None)  # default probe path
    assert mgr.memory_budget_bytes == detected
    assert mgr.memory_budget_bytes <= 32 * 1024**3 + 1  # no x4 blowup on host RAM
    assert mgr.memory_budget_bytes > 0
    # cpu budget also derived from real detection, not hardcoded 8.
    assert mgr.cpu_budget == effective_cpu_slots()
    assert mgr.io_budget == max(1, effective_cpu_slots())


def test_preempted_task_stops_at_checkpoint() -> None:
    """Preemption carries a RevocationToken; the preempted task checks generation
    and stops instead of continuing to run."""
    mgr = ConcurrentTokenManager(
        cpu_budget=4, io_budget=2, memory_budget_bytes=1024**3
    )
    low_rid = mgr.acquire_tokens(2, 1, 256 * 1024**2, task_id="low", priority=0)
    assert low_rid is not None
    low_reservation = mgr._reservations[low_rid]

    # 1) Simulate a checkpoint before preemption: not revoked yet.
    assert mgr.is_revoked(low_reservation) is False

    # 2) High-priority task preempts the low-priority reservation directly.
    rev = mgr.preempt_tokens(
        low_rid, priority=5, reason="high-priority task arrived"
    )
    assert isinstance(rev, RevocationToken)
    assert rev.reservation_id == low_rid
    assert rev.generation == 1

    # 3) The running task checks the revocation at its next checkpoint and stops.
    assert mgr.is_revoked(rev) is True
    assert mgr.is_revoked(low_reservation) is True

    # 4) The reservation was actually removed (resources freed for the preemptor).
    assert low_rid not in mgr._reservations
    assert mgr.metrics().preempted == 1
    assert mgr.metrics().preemptions == 1


def test_automatic_preempt_also_revokes_running_task() -> None:
    """The acquire-then-preempt path must also revoke the victim (not just drop it
    from the dict) so the running task observes the generation bump."""
    mgr = ConcurrentTokenManager(
        cpu_budget=2, io_budget=2, memory_budget_bytes=1024**3
    )
    low_rid = mgr.acquire_tokens(2, 1, 256 * 1024**2, task_id="low", priority=0)
    assert low_rid is not None
    low_reservation = mgr._reservations[low_rid]

    high_rid = mgr.acquire_tokens(2, 1, 256 * 1024**2, task_id="high", priority=5)
    assert high_rid is not None

    # low reservation was preempted out of the dict AND its generation is stale.
    assert low_rid not in mgr._reservations
    assert mgr.is_revoked(low_reservation) is True
    assert mgr.metrics().preempted == 1
    assert mgr.metrics().revoked_generation == 1


def test_negative_tokens_are_rejected() -> None:
    """Negative cpu_tokens / memory_bytes raise ValueError (not silently clamped)."""
    mgr = ConcurrentTokenManager(
        cpu_budget=4, io_budget=4, memory_budget_bytes=1024**3
    )
    with pytest.raises(ValueError):
        mgr.acquire_tokens(-1, 0, 0)
    with pytest.raises(ValueError):
        mgr.acquire_tokens(1, 0, -1)
    # io_tokens positional misuse: check all three named params reject negatives.
    with pytest.raises(ValueError):
        mgr.acquire_tokens(0, -2, 0)
    with pytest.raises(ValueError):
        mgr.acquire_tokens(0, 0, -3)

    # constructor budgets must be positive too
    with pytest.raises(ValueError):
        ConcurrentTokenManager(cpu_budget=-1)
    with pytest.raises(ValueError):
        ConcurrentTokenManager(memory_budget_bytes=-5)


def test_startup_resource_budget_detects_cpu_and_ram() -> None:
    """Startup ResourceBudget detects cgroup quota / cpuset / cores / RAM.

    cpuset cores may be None (not readable in some sandboxes); hard_cpu_cores and
    hard_memory_limit_bytes must always be positive and consistent with the
    effective-limit functions.
    """
    budget = detect_resource_budget()
    assert budget.hard_cpu_cores >= 1
    assert budget.hard_memory_limit_bytes >= 1
    assert budget.hard_memory_limit_bytes == effective_memory_limit_bytes()
    assert budget.hard_cpu_cores == effective_cpu_slots()

    # cpuset detection, when available, is reflected in the hard CPU count.
    if budget.cpuset_cores is not None:
        assert budget.cpuset_cores >= 1
        assert budget.hard_cpu_cores <= budget.cpuset_cores or (
            budget.cgroup_cpu_quota_cores is not None
            and budget.hard_cpu_cores <= budget.cgroup_cpu_quota_cores
        )

    d = budget.to_dict()
    assert d["hard_cpu_cores"] == budget.hard_cpu_cores
    assert d["hard_memory_limit_bytes"] == budget.hard_memory_limit_bytes


def test_resolve_token_budget_rejects_negative() -> None:
    """resolve_token_budget must raise ValueError on negative/zero invalid values."""
    with pytest.raises(ValueError):
        resolve_token_budget(cpu_tokens=-1)
    with pytest.raises(ValueError):
        resolve_token_budget(memory_bytes=-1)
    with pytest.raises(ValueError):
        resolve_token_budget(cpu_tokens=0)
    # None -> probes real resources (positive)
    cpu, io, mem = resolve_token_budget()
    assert cpu >= 1
    assert io >= 1
    assert mem >= 1
