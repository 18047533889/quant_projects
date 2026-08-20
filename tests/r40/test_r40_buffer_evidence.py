# -*- coding: utf-8 -*-
"""R40 #11/#14/#25/#63 测试：buffer store 状态机/多因子淘汰 + 证据完备性 + 证据有效性缓存。"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# #11：条目状态机 + 原子 refcount；evict_if_over_budget 跳过 in-use 条目
# ---------------------------------------------------------------------------


def test_l0_pinned_entry_survives_eviction():
    from runtime.buffer_store import BufferEntryState, GovernedBufferStore

    store = GovernedBufferStore(budget_bytes=2000)
    store.put("big_expensive", list(range(100)), bytes_=800, recompute_cost_ms=500.0)
    store.put("cheap", list(range(10)), bytes_=100, recompute_cost_ms=1.0)
    store.put("pin_me", list(range(50)), bytes_=300, recompute_cost_ms=400.0)
    assert store.pin("pin_me") is True

    freed = store.evict_if_over_budget(target=900)
    assert freed >= 800  # big_expensive + cheap 至少被释放
    # pin_me 必须存活（PINNED / refcount>0 不可淘汰）
    assert store.get("pin_me") is not None
    summary = store.summary()
    assert summary["pinned"] >= 1
    assert summary["in_use_refcount"] >= 1
    # cheap（最低淘汰成本）应该先被淘汰
    assert store.get("cheap") is None


def test_l0_in_use_acquire_ref_survives_eviction():
    from runtime.buffer_store import GovernedBufferStore

    store = GovernedBufferStore(budget_bytes=2000)
    store.put("in_use", [1, 2, 3], bytes_=150, recompute_cost_ms=200.0)
    store.put("other", [4, 5, 6], bytes_=150, recompute_cost_ms=200.0)
    assert store.acquire_ref("in_use") is True
    store.evict_if_over_budget(target=1000)
    assert store.get("in_use") is not None  # refcount>0 → 跳过
    store.release_ref("in_use")
    store.evict_if_over_budget(target=1000)
    assert store.get("in_use") is None  # refcount 归零 → 可淘汰


def test_buffer_entry_state_machine():
    from runtime.buffer_store import BufferEntryState, GovernedBufferStore

    store = GovernedBufferStore(budget_bytes=2000)
    store.put("k", [1], bytes_=10)
    entry = store._entries["k"]
    assert entry.state == BufferEntryState.RECLAIMABLE
    store.pin("k")
    assert entry.state == BufferEntryState.PINNED
    assert entry.refcount >= 1
    store.release("k")
    # release 归还 hold → refcount 0 → 条目被 drop
    assert "k" not in store._entries


# ---------------------------------------------------------------------------
# #14：SpillDecisionEngine 多因子打分（recompute/reload/write/future/stale）
# ---------------------------------------------------------------------------


def test_spill_eviction_ranks_by_recompute_cost():
    from runtime.buffer_store import _Entry, SpillDecisionEngine

    engine = SpillDecisionEngine()
    entries = {
        "cheap": _Entry(value=[1], bytes=10, recompute_cost_ms=1.0),
        "mid": _Entry(value=[2], bytes=10, recompute_cost_ms=50.0),
        "expensive": _Entry(value=[3], bytes=10, recompute_cost_ms=900.0),
    }
    ranked = engine.evict_candidate(entries.items())
    keys = [k for k, _ in ranked]
    assert keys[0] == "cheap"       # 最低 recompute cost 最先淘汰
    assert keys[1] == "mid"
    assert keys[-1] == "expensive"


def test_spill_decision_engine_skips_in_use():
    from runtime.buffer_store import _Entry, SpillDecisionEngine
    from runtime.buffer_store import BufferEntryState

    engine = SpillDecisionEngine()
    entries = [
        ("free", _Entry(value=[1], bytes=10, recompute_cost_ms=1.0)),
        ("pinned", _Entry(value=[2], bytes=10, recompute_cost_ms=1.0,
                          state=BufferEntryState.PINNED, refcount=1)),
        ("held", _Entry(value=[3], bytes=10, recompute_cost_ms=1.0, refcount=2)),
    ]
    ranked = engine.evict_candidate(entries)
    keys = [k for k, _ in ranked]
    assert keys == ["free"]  # 只有 free 可淘汰


def test_spill_eviction_wired_into_store():
    from runtime.buffer_store import GovernedBufferStore

    store = GovernedBufferStore(budget_bytes=2000)
    store.put("cheap", [1], bytes_=100, recompute_cost_ms=1.0)
    store.put("expensive", [2], bytes_=100, recompute_cost_ms=900.0)
    freed = store.evict_if_over_budget(target=60, pressure=0.0)
    assert freed >= 100
    assert store.get("cheap") is None      # 低 recompute cost 先淘汰
    assert store.get("expensive") is not None


# ---------------------------------------------------------------------------
# #25：参数域认证证据全维度完备性审计（check_* 函数复用）
# ---------------------------------------------------------------------------


def test_evidence_cross_product_completeness():
    import itertools

    from scripts.audit_r40_evidence_completeness import (
        REQUIRED_AXES,
        check_evidence_axis_completeness,
        check_evidence_cross_product_completeness,
    )

    combos = list(itertools.product(
        sorted(REQUIRED_AXES["backend"]),
        sorted(REQUIRED_AXES["execution_variant"]),
        sorted(REQUIRED_AXES["grain"]),
        sorted(REQUIRED_AXES["source_context"]),
        sorted(REQUIRED_AXES["dtype"]),
    ))
    complete_points = [
        {
            "canonical": "ts_mean",
            "semantic_version": "2",
            "backend": b,
            "execution_variant": v,
            "source_context": sc,
            "parameter_point": (("window", 5),),
            "dtype": d,
            "grain": g,
            "passed": True,
        }
        for b, v, g, sc, d in combos
    ]
    res = check_evidence_cross_product_completeness(complete_points)
    assert res["all_complete"] is True
    assert res["canonicals"]["ts_mean"]["ok"] is True

    # 缺 duckdb_sql 轴 → incomplete
    incomplete_points = [
        p for p in complete_points if p["backend"] != "duckdb_sql"
    ]
    res2 = check_evidence_cross_product_completeness(incomplete_points)
    assert res2["all_complete"] is False
    assert res2["canonicals"]["ts_mean"]["ok"] is False
    assert any(m[0] == "duckdb_sql" for m in res2["canonicals"]["ts_mean"]["missing"])

    # 轴完整性：缺轴 → incomplete
    bad = [
        {"canonical": "ts_mean", "passed": True, "backend": "", "semantic_version": ""},
        {"canonical": "abs", "passed": True, "backend": "pandas_numpy",
         "semantic_version": "2", "execution_variant": "reference",
         "source_context": "memory", "dtype": "float64", "grain": "daily"},
    ]
    ax = check_evidence_axis_completeness(bad)
    assert ax["complete"] is False


# ---------------------------------------------------------------------------
# #63：证据有效性缓存 key 绑定版本维度（HEAD/build/tcb/artifact）
# ---------------------------------------------------------------------------


def test_evidence_validity_cache_version_key_invalidation():
    import runtime.evidence_truth as et

    et.invalidate_evidence_validity_cache()
    calls = {"n": 0}
    orig = et._evidence_validity_uncached

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    et._evidence_validity_uncached = counting
    try:
        k1 = ("headA", "buildA", "tcbA", "artA")
        r1 = et.evidence_validity_cached(k1)
        r2 = et.evidence_validity_cached(k1)
        assert calls["n"] == 1  # 同 key → 缓存命中，不重算
        k2 = ("headB", "buildA", "tcbA", "artA")
        et.evidence_validity_cached(k2)
        assert calls["n"] == 2  # HEAD 维度变化 → 新 key → 重算
        k3 = ("headA", "buildB", "tcbA", "artA")
        et.evidence_validity_cached(k3)
        assert calls["n"] == 3  # build 维度变化 → 重算
    finally:
        et._evidence_validity_uncached = orig
        et.invalidate_evidence_validity_cache()


def test_evidence_validity_version_key_changes_with_components():
    import runtime.evidence_truth as et

    k_a = et.evidence_validity_version_key(
        head_sha="h1", build_manifest_digest_val="b1", tcb_hash="t1", artifact_hash="a1"
    )
    k_b = et.evidence_validity_version_key(
        head_sha="h2", build_manifest_digest_val="b1", tcb_hash="t1", artifact_hash="a1"
    )
    assert k_a != k_b
    assert k_a[0] == "h1" and k_b[0] == "h2"
