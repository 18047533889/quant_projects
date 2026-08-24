# -*- coding: utf-8 -*-
"""R40 #120/#122/#124: persistent-cache save-lock refcount, cache session
execution_id uniqueness, and runtime_stats merge."""
from __future__ import annotations

import threading

import pytest
from factor_engine.storage.sources.datasource import DataSource


class _MinimalSource(DataSource):
    def load_column(self, name: str):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# #120: _save_lock_for must never evict an actively-held lock.
# ---------------------------------------------------------------------------
class TestSaveLockRefcount:
    def test_save_lock_never_evicts_active_lock(self):
        import factor_engine.storage.cache as cache

        # Clear module-level lock registry for a deterministic test.
        cache._SAVE_LOCKS.clear()
        cache._SAVE_LOCKS_ORDER.clear()
        cache._SAVE_LOCK_REFS.clear()

        held = cache._save_lock_for("active-key")
        held.__enter__()  # refcount = 1
        try:
            # Saturate the registry well past the bound with OTHER keys; the
            # LRU must skip "active-key" because its refcount is 1.
            for i in range(cache._SAVE_LOCK_MAX * 2):
                w = cache._save_lock_for(f"other-{i}")
                # entering every wrapper would make them all "active"; instead
                # just fetch (refcount stays 0 → evictable).
                del w
            assert "active-key" in cache._SAVE_LOCKS, "active lock must never be evicted"
            assert len(cache._SAVE_LOCKS) <= cache._SAVE_LOCK_MAX + 1
        finally:
            held.__exit__(None, None, None)
            # Cleanup
            cache._SAVE_LOCKS.clear()
            cache._SAVE_LOCKS_ORDER.clear()
            cache._SAVE_LOCK_REFS.clear()

    def test_save_lock_wrapper_serializes_same_key(self):
        import factor_engine.storage.cache as cache

        cache._SAVE_LOCKS.clear()
        cache._SAVE_LOCKS_ORDER.clear()
        cache._SAVE_LOCK_REFS.clear()
        try:
            a = cache._save_lock_for("k")
            b = cache._save_lock_for("k")
            # Enter once: refcount increments; the wrapper references the SAME
            # underlying registry lock for the key as any other wrapper.
            a.__enter__()
            assert cache._SAVE_LOCK_REFS.get("k") == 1
            assert cache._SAVE_LOCKS["k"] is a.lock
            a.__exit__(None, None, None)
            assert "k" not in cache._SAVE_LOCK_REFS

            # A second wrapper entering after release reuses the same lock.
            b.__enter__()
            assert cache._SAVE_LOCK_REFS.get("k") == 1
            assert cache._SAVE_LOCKS["k"] is b.lock
            b.__exit__(None, None, None)
            assert "k" not in cache._SAVE_LOCK_REFS
        finally:
            cache._SAVE_LOCKS.clear()
            cache._SAVE_LOCKS_ORDER.clear()
            cache._SAVE_LOCK_REFS.clear()

    def test_lock_registry_bounded_without_enter(self):
        # Existing r32 gate: calling _save_lock_for without entering must stay bounded.
        import factor_engine.storage.cache as cache

        cache._SAVE_LOCKS.clear()
        cache._SAVE_LOCKS_ORDER.clear()
        cache._SAVE_LOCK_REFS.clear()
        try:
            for i in range(cache._SAVE_LOCK_MAX * 2):
                cache._save_lock_for(f"bounded-{i}")
            assert len(cache._SAVE_LOCKS) <= cache._SAVE_LOCK_MAX
        finally:
            cache._SAVE_LOCKS.clear()
            cache._SAVE_LOCKS_ORDER.clear()
            cache._SAVE_LOCK_REFS.clear()


# ---------------------------------------------------------------------------
# #122: wrap_context must preserve existing runtime_stats.
# ---------------------------------------------------------------------------
class TestWrapContext:
    def test_wrap_context_preserves_existing_runtime_stats(self):
        from factor_engine.backend.context import ExecutionContext
        from factor_engine.cache.layers import CacheHitStats
        from factor_engine.cache.session import ExecutionCacheSession

        ds = _MinimalSource()
        ctx = ExecutionContext(data_source=ds)
        ctx.runtime_stats = {"existing_layer": {"hits": 7}}
        session = ExecutionCacheSession(execution_id="uniq-xyz")
        wrapped = session.wrap_context(ctx)
        assert wrapped.runtime_stats["existing_layer"] == {"hits": 7}
        assert wrapped.runtime_stats["cache"] is session.stats
        assert wrapped.runtime_stats["cache"] is not None


# ---------------------------------------------------------------------------
# #124: default execution_id must be unique per session.
# ---------------------------------------------------------------------------
class TestExecutionId:
    def test_default_execution_id_is_unique_per_session(self):
        from factor_engine.cache.session import ExecutionCacheSession

        s1 = ExecutionCacheSession()
        s2 = ExecutionCacheSession()
        assert s1.execution_id != "session"
        assert s1.execution_id != s2.execution_id
        # governor layer names are therefore unique too
        assert s1._l0_layer != s2._l0_layer

    def test_explicit_execution_id_preserved(self):
        from factor_engine.cache.session import ExecutionCacheSession

        s = ExecutionCacheSession(execution_id="my-run-1")
        assert s.execution_id == "my-run-1"
        assert s._l0_layer == "my-run-1:l0_cse"
