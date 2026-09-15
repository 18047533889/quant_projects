from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from factor_engine.cache.session import ExecutionCacheSession
from factor_engine.runtime.batch_service import _bind_cse_budget_authority
from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.buffer_store import GovernedBufferStore
from factor_engine.runtime.resource_broker_ipc import ParentBrokerIPC


class _BudgetAuthority:
    def __init__(self, budget: int) -> None:
        self.budget = budget

    def current_cse_budget(self) -> int:
        return self.budget


def test_zero_cse_budget_is_fail_closed_not_unbounded():
    store = GovernedBufferStore(budget_bytes=0)
    result = store.put("zero", np.arange(8), bytes_=64)
    assert result.status == "REFUSED"
    assert store.summary()["budget_bytes"] == 0
    assert store.summary()["accounted_bytes"] == 0


def test_bound_authority_replaces_static_plan_and_shrinks_with_pressure():
    authority = _BudgetAuthority(100)
    store = GovernedBufferStore(budget_bytes=10_000)
    store.bind_budget_authority(authority)
    assert store.put("resident", np.arange(10), bytes_=80).status == "MEMORY"
    authority.budget = 40
    store.refresh_budget()
    observed = store.summary()
    assert observed["budget_bytes"] == 40
    assert observed["accounted_bytes"] == 0
    assert observed["budget_authority"] == "_BudgetAuthority.current_cse_budget"
    assert observed["budget_refreshes"] >= 2


def test_pressure_shrink_preserves_in_use_entry_until_consumer_releases():
    authority = _BudgetAuthority(100)
    store = GovernedBufferStore()
    store.bind_budget_authority(authority)
    assert store.put("live", np.arange(10), bytes_=80).status == "MEMORY"
    assert store.acquire_ref("live")
    authority.budget = 0
    store.refresh_budget()
    assert store.get("live") is not None
    assert store.summary()["accounted_bytes"] == 80
    assert store.put("new", np.arange(1), bytes_=8).status == "REFUSED"
    store.release_ref("live")
    store.refresh_budget()
    assert store.summary()["accounted_bytes"] == 0


def test_batch_context_binds_store_to_real_broker_authority():
    authority = _BudgetAuthority(25)
    store = GovernedBufferStore(budget_bytes=999)
    ctx = SimpleNamespace(shared_buffers=store, runtime_stats={})
    _bind_cse_budget_authority(ctx, authority)
    assert store.budget_bytes == 25
    assert ctx.runtime_stats["cse_cache_budget"]["budget_bytes"] == 25
    assert ctx.runtime_stats["cse_cache_budget"]["authority"] == (
        "_BudgetAuthority.current_cse_budget"
    )


def test_session_accepts_an_ordinary_resource_broker_as_authority(monkeypatch):
    broker = ResourceBroker(hard_memory_limit=1024**3, min_host_reserve_gb=0,
                            min_host_reserve_fraction=0)
    monkeypatch.setattr(broker, "current_cse_budget", lambda: 31)
    session = ExecutionCacheSession(cse_budget_bytes=999,
                                    cse_budget_authority=broker, strict=False)
    try:
        assert session.buffer_store.budget_bytes == 31
        assert session.buffer_store.summary()["budget_authority"] == (
            "ResourceBroker.current_cse_budget"
        )
    finally:
        session.release()


def test_ipc_proxy_exposes_parent_cse_budget_without_local_authority():
    authority = _BudgetAuthority(73)
    ipc = ParentBrokerIPC(authority, rpc_timeout_seconds=1.0)
    proxy = ipc.create_proxy()
    try:
        assert proxy.current_cse_budget() == 73
        authority.budget = 0
        assert proxy.current_cse_budget() == 0
    finally:
        ipc.close()

