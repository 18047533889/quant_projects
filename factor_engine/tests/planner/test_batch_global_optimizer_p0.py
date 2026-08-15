from __future__ import annotations

from types import SimpleNamespace

import pytest

from planner.backend_region import PhysicalBackend, Representation
from backend.contracts import ExecutionKind
from backend.operator_capability import UnsupportedOperatorBackendError
from planner.batch_global_optimizer import BatchGlobalOptimizer
from planner.logical_plan import PlanNode


def _ctx(rows: int | None, *, mode: str = "research", complete: bool = False):
    stats = {} if rows is None else {"row_count_estimate": rows}
    if complete:
        stats.update(estimated_bytes=rows * 16, estimated_memory_bytes=rows * 8)
    return SimpleNamespace(run_mode=mode, runtime_stats=stats)


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
    assert region.required_properties is not None
    assert region.output_properties is not None
    assert region.state_contract is None


def test_unknown_row_estimate_is_not_production_ready():
    root = PlanNode("abs", inputs=(PlanNode("column"),))
    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(None)
    )

    assert result.production_ready is False
    assert result.readiness_reason == "row-count estimate unavailable"


def test_known_rows_alone_are_not_production_ready():
    root = PlanNode("abs", inputs=(PlanNode("column"),))

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100)
    )

    assert result.production_ready is False
    assert result.readiness_reason == "byte estimate unavailable"


def test_complete_estimates_and_contracts_are_production_ready():
    root = PlanNode("abs", inputs=(PlanNode("column"),))

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, complete=True)
    )

    assert result.production_ready is True
    assert result.readiness_reason == ""
    assert result.physical_plan.peak_memory_bytes == 800


def test_empty_plan_is_not_fake_production_ready():
    result = BatchGlobalOptimizer().optimize_batch(
        {}, {}, {}, _ctx(100, complete=True)
    )

    assert result.production_ready is False
    assert result.readiness_reason == "empty logical plan"
    assert result.physical_plan.regions == ()


def test_production_unsupported_operator_fails_closed(monkeypatch):
    monkeypatch.setattr("backend.operator_capability.supports_pandas", lambda *a, **k: False)
    monkeypatch.setattr("backend.operator_capability.supports_polars", lambda *a, **k: False)
    root = PlanNode("synthetic_uncertified", inputs=(PlanNode("column"),))

    with pytest.raises(UnsupportedOperatorBackendError, match="production-certified"):
        BatchGlobalOptimizer().optimize_batch(
            {"root": root}, {}, {}, _ctx(100, mode="production", complete=True)
        )


def test_research_polars_delegate_is_explicit_and_not_ready(monkeypatch):
    monkeypatch.setattr("backend.operator_capability.supports_pandas", lambda *a, **k: False)
    monkeypatch.setattr("backend.operator_capability.supports_polars", lambda *a, **k: True)
    monkeypatch.setattr("backend.polars_backend_kind.canonical_polars_is_delegate", lambda *a, **k: True)
    root = PlanNode("synthetic_delegate", inputs=(PlanNode("column"),))

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, complete=True)
    )

    assert result.per_node_choices["root"].execution_kind == ExecutionKind.POLARS_PANDAS_DELEGATE
    assert result.production_ready is False
    assert result.readiness_reason == "delegate fallback assigned: root"


def test_source_node_inherits_consumer_backend_when_neutral(monkeypatch):
    monkeypatch.setattr("backend.operator_capability.supports_pandas", lambda *a, **k: False)
    monkeypatch.setattr("backend.operator_capability.supports_polars", lambda *a, **k: True)
    monkeypatch.setattr("backend.polars_backend_kind.canonical_polars_is_delegate", lambda *a, **k: False)
    source = PlanNode("column", node_id="source")
    root = PlanNode("synthetic_polars", inputs=(source,), node_id="root")

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, complete=True)
    )

    assert result.per_node_choices["source"].backend == PhysicalBackend.POLARS_PANEL
    assert result.per_node_choices["source"].representation == Representation.POLARS_LONG
    assert result.physical_plan.edges == ()


def test_same_backend_representation_change_has_typed_transfer(monkeypatch):
    monkeypatch.setattr("backend.operator_capability.supports_pandas", lambda *a, **k: False)
    monkeypatch.setattr("backend.operator_capability.supports_polars", lambda *a, **k: True)
    monkeypatch.setattr("backend.polars_backend_kind.canonical_polars_is_delegate", lambda *a, **k: False)
    source = PlanNode(
        "column",
        attrs={
            "source_backend": "polars_panel",
            "source_representation": "polars_wide",
        },
        node_id="source",
    )
    root = PlanNode("synthetic_polars", inputs=(source,), node_id="root")

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, complete=True)
    )

    assert len(result.physical_plan.edges) == 1
    edge = result.physical_plan.edges[0]
    assert edge.source_backend == edge.target_backend == PhysicalBackend.POLARS_PANEL
    assert edge.source_representation == Representation.POLARS_WIDE
    assert edge.target_representation == Representation.POLARS_LONG
    assert result.physical_plan.backend_switch_count == 0


def test_invalid_source_residency_fails_closed():
    source = PlanNode(
        "column", attrs={"source_backend": "typo_backend"}, node_id="source"
    )

    with pytest.raises(ValueError, match="unknown source_backend"):
        BatchGlobalOptimizer().optimize_batch(
            {"root": source}, {}, {}, _ctx(100, complete=True)
        )


def test_explicit_source_residency_creates_typed_transfer(monkeypatch):
    monkeypatch.setattr("backend.operator_capability.supports_pandas", lambda *a, **k: True)
    monkeypatch.setattr("backend.operator_capability.supports_polars", lambda *a, **k: False)
    monkeypatch.setattr("backend.operator_capability.capability_for", lambda *a, **k: SimpleNamespace(
        execution_kind=ExecutionKind.PANDAS_REFERENCE,
        is_production_eligible=lambda: True,
    ))
    source = PlanNode(
        "column",
        attrs={
            "source_backend": "polars_panel",
            "source_representation": "polars_wide",
        },
        node_id="source",
    )
    root = PlanNode("synthetic_pandas", inputs=(source,), node_id="root")

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, complete=True)
    )

    assert len(result.physical_plan.edges) == 1
    edge = result.physical_plan.edges[0]
    assert edge.source_backend == PhysicalBackend.POLARS_PANEL
    assert edge.target_backend == PhysicalBackend.PANDAS_NUMPY
    assert edge.estimated_bytes == 1600
    assert result.production_ready is True
