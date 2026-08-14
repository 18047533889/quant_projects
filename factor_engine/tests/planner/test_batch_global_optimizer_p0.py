from __future__ import annotations

from types import SimpleNamespace

from planner.batch_global_optimizer import BatchGlobalOptimizer
from planner.logical_plan import PlanNode


def _ctx(rows: int | None):
    return SimpleNamespace(
        run_mode="research",
        runtime_stats={} if rows is None else {"row_count_estimate": rows},
    )


def test_nested_descendants_are_discovered_without_node_graph_entries():
    leaf = PlanNode("column", attrs={"name": "x"}, node_id="leaf")
    child = PlanNode("abs", inputs=(leaf,), node_id="child")
    root = PlanNode("neg", inputs=(child,), node_id="root")

    result = BatchGlobalOptimizer().optimize_batch(
        {"factor": root}, {}, {}, _ctx(100)
    )

    assert {"factor", "child", "leaf"}.issubset(result.per_node_choices)


def test_shared_child_gets_one_persisted_assignment():
    shared = PlanNode("abs", inputs=(PlanNode("column", node_id="source"),), node_id="shared")
    left = PlanNode("neg", inputs=(shared,), node_id="left")
    right = PlanNode("log", inputs=(shared,), node_id="right")

    result = BatchGlobalOptimizer().optimize_batch(
        {"left": left, "right": right}, {"shared": shared}, {}, _ctx(100)
    )

    assert "shared" in result.per_node_choices
    assert result.shared_benefits["shared"].consumer_count == 2


def test_backend_region_constructor_uses_current_contract():
    root = PlanNode("abs", inputs=(PlanNode("column"),))
    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100)
    )

    assert result.physical_plan.regions
    region = result.physical_plan.regions[0]
    assert region.required_properties is None
    assert region.state_contract is None


def test_unknown_row_estimate_is_not_production_ready():
    root = PlanNode("abs", inputs=(PlanNode("column"),))
    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(None)
    )

    assert result.production_ready is False
    assert result.readiness_reason == "row-count estimate unavailable"
