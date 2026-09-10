from __future__ import annotations

import threading
import time

import pytest

from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
from factor_engine.runtime.exceptions import (
    Cancellation,
    CancellationToken,
    reset_active_cancellation_token,
    set_active_cancellation_token,
)
from factor_engine.runtime.resource_broker import ResourceBroker


def _broker(monkeypatch: pytest.MonkeyPatch, budget: int = 100) -> ResourceBroker:
    broker = ResourceBroker(hard_memory_limit=budget, cpu_slots=2)
    monkeypatch.setattr(broker, "execution_budget", lambda: budget)
    return broker


def _release_pair(leases: tuple[object, object]) -> None:
    for lease in leases:
        lease.release()


def test_protected_egress_waiters_acquire_in_fifo_order(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker(monkeypatch)
    blocker = broker.acquire_memory(MemoryLeaseKind.CSE_CACHE, 80, lease_id="blocker")
    assert blocker is not None
    entered = [threading.Event(), threading.Event()]
    acquired: list[str] = []

    def attempt(index: int, lease_id: str) -> None:
        with broker.protected_egress_waiter(lease_id, time.monotonic() + 2.0):
            entered[index].set()
            while True:
                leases = broker.acquire_protected_egress(15, 15, lease_id=lease_id)
                if leases is not None:
                    acquired.append(lease_id)
                    _release_pair(leases)
                    return
                broker.wait_for_resource_change(0.2)

    first = threading.Thread(target=attempt, args=(0, "first"))
    second = threading.Thread(target=attempt, args=(1, "second"))
    first.start()
    assert entered[0].wait(1.0)
    second.start()
    assert entered[1].wait(1.0)
    blocker.release()
    first.join(2.0)
    second.join(2.0)

    assert not first.is_alive() and not second.is_alive()
    assert acquired == ["first", "second"]
    assert broker._memory_leases == {}


def test_cancelled_head_is_removed_and_wakes_next_waiter(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker(monkeypatch)
    token = CancellationToken()
    head_entered = threading.Event()
    next_entered = threading.Event()
    cancelled: list[bool] = []
    cancellation_elapsed: list[float] = []
    next_acquired = threading.Event()

    def head() -> None:
        reset = set_active_cancellation_token(token)
        try:
            with broker.protected_egress_waiter("head", time.monotonic() + 2.0):
                head_entered.set()
                wait_started = time.monotonic()
                broker.wait_for_resource_change(5.0)
        except Cancellation:
            cancelled.append(True)
            cancellation_elapsed.append(time.monotonic() - wait_started)
        finally:
            reset_active_cancellation_token(reset)

    def next_waiter() -> None:
        with broker.protected_egress_waiter("next", time.monotonic() + 2.0):
            next_entered.set()
            while True:
                leases = broker.acquire_protected_egress(10, 10, lease_id="next")
                if leases is not None:
                    next_acquired.set()
                    _release_pair(leases)
                    return
                broker.wait_for_resource_change(0.2)

    first = threading.Thread(target=head)
    second = threading.Thread(target=next_waiter)
    first.start()
    assert head_entered.wait(1.0)
    second.start()
    assert next_entered.wait(1.0)
    token.cancel()
    first.join(1.0)
    second.join(1.0)

    assert cancelled == [True]
    assert cancellation_elapsed[0] < 0.25
    assert next_acquired.is_set()
    assert broker._memory_leases == {}


def test_wait_stops_at_waiter_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker(monkeypatch)
    started = time.monotonic()
    with broker.protected_egress_waiter("deadline", started + 0.05):
        assert broker.wait_for_resource_change(1.0) is False
    elapsed = time.monotonic() - started
    assert 0.03 <= elapsed < 0.5


def test_real_lease_release_wakes_waiter_without_lost_wakeup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _broker(monkeypatch)
    # First realize the protected sink reserve, then use the remaining pool.
    # Ordinary compute/cache admission may not consume the unrealized reserve.
    seed_egress = broker.acquire_protected_egress(5, 5, lease_id="seed-egress")
    assert seed_egress is not None
    blocker = broker.acquire_memory(MemoryLeaseKind.CSE_CACHE, 90, lease_id="blocker")
    assert blocker is not None
    failed_acquire = threading.Event()
    result: list[bool] = []

    def wait() -> None:
        with broker.protected_egress_waiter("waiting", time.monotonic() + 2.0):
            assert broker.acquire_protected_egress(1, 1, lease_id="waiting") is None
            failed_acquire.set()
            result.append(broker.wait_for_resource_change(1.0))

    thread = threading.Thread(target=wait)
    thread.start()
    assert failed_acquire.wait(1.0)
    blocker.release()
    thread.join(1.0)

    assert result == [True]
    _release_pair(seed_egress)
    assert broker._memory_leases == {}


def test_non_waiter_cannot_bypass_fifo_head_and_failed_attempt_leaks_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _broker(monkeypatch)
    with broker.protected_egress_waiter("head", time.monotonic() + 1.0):
        assert broker.acquire_protected_egress(10, 10, lease_id="bypass") is None
        assert broker._memory_leases == {}
    assert not broker._protected_egress_waiters


def test_cancelled_request_cannot_acquire_even_when_capacity_is_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _broker(monkeypatch)
    token = CancellationToken()
    token.cancel()
    reset = set_active_cancellation_token(token)
    try:
        with broker.protected_egress_waiter("cancelled", time.monotonic() + 1.0):
            with pytest.raises(Cancellation):
                broker.acquire_protected_egress(10, 10, lease_id="cancelled")
    finally:
        reset_active_cancellation_token(reset)
    assert broker._memory_leases == {}


def test_expired_stalled_head_is_pruned(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker(monkeypatch)
    head_entered = threading.Event()
    release_head = threading.Event()

    def stalled_head() -> None:
        with broker.protected_egress_waiter("stalled", time.monotonic() + 0.05):
            head_entered.set()
            release_head.wait(1.0)

    thread = threading.Thread(target=stalled_head)
    thread.start()
    assert head_entered.wait(1.0)
    time.sleep(0.07)
    with broker.protected_egress_waiter("next", time.monotonic() + 1.0):
        leases = broker.acquire_protected_egress(10, 10, lease_id="next")
        assert leases is not None
        _release_pair(leases)
    release_head.set()
    thread.join(1.0)
    assert broker._memory_leases == {}
