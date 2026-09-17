from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.batch_service import (
    _materialize_shared_subplan,
    _release_all_cse_sids,
    _release_consumed_sids,
    _setup_cse_refcounts,
    validate_cse_refcount_integrity,
)
from factor_engine.runtime.buffer_store import GovernedBufferStore


class _Backend:
    def __init__(self, value):
        self.value = value

    def execute(self, _plan, _ctx):
        return self.value


def _ctx(store, refcounts):
    return SimpleNamespace(
        shared_result_cache={},
        shared_buffers=store,
        expression_cache=None,
        runtime_stats={},
        run_mode="production",
        _cse_refcounts=dict(refcounts),
        cse_certificate_store=None,
    )


def test_materialization_pin_survives_forced_eviction_until_last_consumer() -> None:
    value = np.arange(128, dtype=np.float64)
    store = GovernedBufferStore({}, budget_bytes=value.nbytes)
    ctx = _ctx(store, {"shared": 2})

    assert _materialize_shared_subplan(
        _Backend(value), PlanNode("literal", attrs={"value": value}), ctx, "shared"
    )
    assert store.summary()["in_use_refcount"] == 1
    assert store.evict_if_over_budget(target=value.nbytes) == 0
    assert store.get_ref("shared") is value

    root = PlanNode("plan_ref", attrs={"sid": "shared"})
    _release_consumed_sids(ctx, root)
    assert store.get_ref("shared") is value
    _release_consumed_sids(ctx, root)
    assert store.get_ref("shared") is None


def test_unpinned_overwrite_clears_stale_pinned_membership() -> None:
    first = np.arange(8, dtype=np.float64)
    second = first + 1.0
    store = GovernedBufferStore({}, budget_bytes=first.nbytes * 2)

    assert store.put("shared", first, pin=True).status == "MEMORY"
    assert "shared" in store._pinned
    assert store.put("shared", second, pin=False).status == "MEMORY"
    assert "shared" not in store._pinned
    assert store.evict_if_over_budget(target=second.nbytes) == second.nbytes
    assert store.get_ref("shared") is None


def test_nested_shared_parent_releases_child_without_lifetime_gap() -> None:
    child_value = np.arange(32, dtype=np.float64)
    parent_value = child_value + 1.0
    store = GovernedBufferStore({}, budget_bytes=child_value.nbytes + parent_value.nbytes)
    ctx = _ctx(store, {"child": 1, "parent": 1})
    child = PlanNode("literal", attrs={"value": child_value})
    parent = PlanNode("plan_ref", attrs={"sid": "child"})

    assert _materialize_shared_subplan(_Backend(child_value), child, ctx, "child")
    assert _materialize_shared_subplan(_Backend(parent_value), parent, ctx, "parent")
    assert store.get_ref("child") is None
    assert store.get_ref("parent") is parent_value
    assert store.evict_if_over_budget(target=parent_value.nbytes) == 0

    _release_consumed_sids(ctx, PlanNode("plan_ref", attrs={"sid": "parent"}))
    assert store.get_ref("parent") is None


def test_shared_value_matches_independent_numeric_execution() -> None:
    index = pd.date_range("2025-01-01", periods=8)
    base = pd.Series(np.arange(8, dtype=float), index=index)
    shared = base.rolling(3, min_periods=1).mean()
    store = GovernedBufferStore({}, budget_bytes=shared.memory_usage(deep=True) * 2)
    ctx = _ctx(store, {"mean": 2})
    assert _materialize_shared_subplan(
        _Backend(shared), PlanNode("literal", attrs={"value": shared}), ctx, "mean"
    )

    actual_a = store.get_ref("mean") + 1.0
    actual_b = store.get_ref("mean") * 2.0
    pd.testing.assert_series_equal(actual_a, base.rolling(3, min_periods=1).mean() + 1.0)
    pd.testing.assert_series_equal(actual_b, base.rolling(3, min_periods=1).mean() * 2.0)


def test_abort_cleanup_releases_unconsumed_materialization_pins() -> None:
    value = np.arange(64, dtype=np.float64)
    store = GovernedBufferStore({}, budget_bytes=value.nbytes * 2)
    ctx = _ctx(store, {"first": 2, "second": 1})
    assert _materialize_shared_subplan(_Backend(value), PlanNode("literal"), ctx, "first")
    assert _materialize_shared_subplan(_Backend(value + 1), PlanNode("literal"), ctx, "second")

    # Models sink failure/cancellation before either root consumes its sid.
    _release_all_cse_sids(ctx, {"first": PlanNode("literal"), "second": PlanNode("literal")})
    assert store.get_ref("first") is None
    assert store.get_ref("second") is None
    assert store.summary()["accounted_bytes"] == 0


def test_orphan_shared_task_is_counted_when_it_consumes_live_inner_sid() -> None:
    child_ref = PlanNode("plan_ref", attrs={"sid": "child"})
    parent_ref = PlanNode("plan_ref", attrs={"sid": "parent"})
    shared = {
        "child": PlanNode("literal", attrs={"value": 1.0}),
        "parent": PlanNode("abs", inputs=(child_ref,)),
        # The physical lowerer currently schedules every shared definition.
        "orphan": PlanNode("neg", inputs=(child_ref,)),
    }
    roots = [parent_ref, child_ref]
    store = GovernedBufferStore({}, budget_bytes=1024)
    ctx = _ctx(store, {})

    _setup_cse_refcounts(ctx, roots, shared_nodes=shared)
    assert ctx._cse_refcounts == {"parent": 1, "child": 3}
    ctx.run_mode = "research"
    report = validate_cse_refcount_integrity(
        SimpleNamespace(roots=roots, shared_nodes=shared), ctx
    )
    assert not report["ok"]
    assert any("orphaned" in issue for issue in report["issues"])

    assert _materialize_shared_subplan(
        _Backend(np.arange(4.0)), shared["child"], ctx, "child"
    )
    _release_consumed_sids(ctx, shared["parent"])
    _release_consumed_sids(ctx, shared["orphan"])
    assert store.get_ref("child") is not None
    _release_consumed_sids(ctx, roots[1])
    assert store.get_ref("child") is None
