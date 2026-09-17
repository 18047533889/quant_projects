"""Explicit cache budgets and independent residency accounting."""
import threading
from collections import defaultdict
from types import SimpleNamespace

import pandas as pd
import pytest

from factor_engine.storage.cache import CacheManager


@pytest.mark.parametrize("budget", [True, False, -1, 1.5, "100"])
def test_invalid_explicit_budget_is_rejected(budget):
    with pytest.raises((TypeError, ValueError)):
        CacheManager(budget_bytes=budget)


def test_zero_budget_never_admits_even_unknown_size(monkeypatch):
    from factor_engine.runtime import resource_governor
    cache = CacheManager(budget_bytes=0)
    assert cache.budget_bytes == 0
    monkeypatch.setattr(resource_governor, "estimate_object_bytes", lambda value: 0)
    cache.set("opaque", object())
    assert cache.get("opaque") is None
    assert cache._bytes == 0
    assert cache.with_scope("other").budget_bytes == 0


def test_clear_releases_only_this_backing_store(monkeypatch):
    from factor_engine.storage import cache as module
    used = defaultdict(int)
    def reserve(name, count):
        used[name] += count
    def release(name, count):
        used[name] -= count
        assert used[name] >= 0
    gov = SimpleNamespace(
        lock=threading.RLock(), reserve_accounting=reserve,
        release_accounting=release,
        release_all=lambda name: used.__setitem__(name, 0),
    )
    monkeypatch.setattr(module, "_governor", lambda: gov)
    left = CacheManager(budget_bytes=4096)
    right = CacheManager(budget_bytes=4096)
    left.set("left", pd.Series([1., 2.]))
    right.set("right", pd.Series([3., 4.]))
    right_bytes = right._bytes
    assert used["l2_subplan"] == left._bytes + right_bytes
    alias = left.with_scope("alias")
    alias.clear_memory()
    assert left._bytes == alias._bytes == 0
    assert used["l2_subplan"] == right_bytes
    alias.clear_memory()
    assert used["l2_subplan"] == right_bytes
    assert right.get("right") is not None
    right.clear_memory()
    assert used["l2_subplan"] == 0
