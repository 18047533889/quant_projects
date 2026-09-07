from __future__ import annotations

from types import SimpleNamespace

import pytest

from factor_engine.planner.backend_region import PhysicalBackend, Representation
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.operator_capability import UnsupportedOperatorBackendError
from factor_engine.planner.batch_global_optimizer import BatchGlobalOptimizer
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.multibackend.batch_global_optimizer import (
    NodeBackendChoice,
    PhysicalBatchGlobalOptimizer,
)


def _choice(node_id, backend, cost):
    return NodeBackendChoice(
        node_id=node_id,
        backend=backend,
        compute_cost_ms=cost,
        transfer_from_children_ms=0.0,
        total_cost_ms=cost,
        representation=(
            Representation.PANDAS_LONG
            if backend == PhysicalBackend.PANDAS_NUMPY
            else Representation.POLARS_LONG
        ),
        execution_kind=ExecutionKind.PANDAS_REFERENCE,
        production_certified=True,
    )


def test_complete_incumbents_keep_pandas_and_native_baselines():
    candidates = {
        "a": (_choice("a", PhysicalBackend.PANDAS_NUMPY, 2.0), _choice("a", PhysicalBackend.POLARS_LONG, 1.0)),
        "b": (_choice("b", PhysicalBackend.PANDAS_NUMPY, 2.0), _choice("b", PhysicalBackend.POLARS_LONG, 1.0)),
    }

    names = {name for name, _ in PhysicalBatchGlobalOptimizer._complete_incumbents(candidates)}

    assert "single:pandas_numpy:pandas_long" in names
    assert "single:polars_long:polars_long" in names
    assert "legal_mixed" in names


def test_exact_search_candidate_cap_returns_legal_incumbent(monkeypatch):
    optimizer = PhysicalBatchGlobalOptimizer(
        exact_search_max_ambiguous_nodes=20,
        max_candidate_plans=3,
        max_optimization_ms=250.0,
    )
    candidates = {
        node_id: (_choice(node_id, PhysicalBackend.PANDAS_NUMPY, 2.0), _choice(node_id, PhysicalBackend.POLARS_LONG, 1.0))
        for node_id in ("root", "a", "b", "c")
    }
    monkeypatch.setattr(optimizer, "_eligible_choices", lambda node_id, node, rows, ctx: candidates[node_id])
    nodes = {node_id: PlanNode("literal") for node_id in candidates}
    graph = {"root": ["a", "b", "c"], "a": [], "b": [], "c": []}

    choices, basis, count, selected = optimizer._optimize_graph(
        nodes, graph, 100, _ctx(100), {node_id: (100, 800, 800) for node_id in nodes}
    )

    assert set(choices) == set(nodes)
    assert basis == "bounded_incumbent_timeout"
    assert count <= 3
    assert selected in {"mixed", "legal_mixed", "single:polars_long:polars_long"}


def test_explicit_pandas_policy_filters_auto_candidates(monkeypatch):
    optimizer = PhysicalBatchGlobalOptimizer(forced_backend="pandas")
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_pandas", lambda *a, **k: True)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_polars", lambda *a, **k: True)
    monkeypatch.setattr("factor_engine.backend.polars_backend_kind.canonical_polars_is_delegate", lambda *a, **k: False)
    choices = optimizer._eligible_choices("root", PlanNode("abs"), 100, _ctx(100))
    assert choices
    assert {choice.backend for choice in choices} == {PhysicalBackend.PANDAS_NUMPY}


def _ctx(rows: int | None, *, mode: str = "research", complete: bool = False):
    stats = {} if rows is None else {"row_count_estimate": rows}
    if complete:
        stats.update(estimated_bytes=rows * 16, estimated_memory_bytes=rows * 8)
    return SimpleNamespace(run_mode=mode, runtime_stats=stats)


def _bound_column(**extra):
    from data_access.read.data_read_identity import DataReadIdentity, ResolvedFieldIdentity
    from factor_engine.runtime.physical_source_binding import PhysicalSourceBinding
    identity = DataReadIdentity(
        dataset="equity_daily", revision="r1", calendar_identity="ashare_daily",
        universe_snapshot="all_a_2026", source_snapshot="snapshot_2026",
        fields=(ResolvedFieldIdentity(logical_name="close", physical_name="close",
                                      dataset="equity_daily", availability="same_day"),),
    )
    attrs = {
        "name": "close",
        "physical_source_binding": PhysicalSourceBinding(identity, "close", "catalog:v1"),
    }
    attrs.update(extra)
    return PlanNode("column", attrs=attrs)


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
    root = PlanNode("abs", inputs=(_bound_column(),))
    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100)
    )

    assert result.physical_plan.regions
    region = result.physical_plan.regions[0]
    assert region.required_properties is not None
    assert region.output_properties is not None
    assert region.state_contract is None


def test_unknown_row_estimate_is_not_production_ready():
    root = PlanNode("abs", inputs=(_bound_column(),))
    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(None)
    )

    assert result.production_ready is False
    assert result.readiness_reason == "row-count estimate unavailable"


def test_actual_cse_plan_ref_adds_sid_dependency():
    shared = PlanNode(
        "abs",
        inputs=(PlanNode("column", node_id="source"),),
        node_id="shared-node",
    )
    ref = PlanNode("plan_ref", attrs={"sid": "shared-sid"}, node_id="ref")
    root = PlanNode("neg", inputs=(ref,), node_id="root")

    result = BatchGlobalOptimizer().optimize_batch(
        {"factor": root}, {"shared-sid": shared}, {}, _ctx(100, complete=True)
    )

    assert {"factor", "ref", "shared-sid", "source"}.issubset(
        result.per_node_choices
    )
    assert result.physical_plan.logical_node_count == 4
    ref_region = next(
        region for region in result.physical_plan.regions if "ref" in region.node_ids
    )
    shared_region = next(
        region
        for region in result.physical_plan.regions
        if "shared-sid" in region.node_ids
    )
    assert ref_region.region_id == shared_region.region_id or any(
        edge.producer_region == shared_region.region_id
        and edge.consumer_region == ref_region.region_id
        for edge in result.physical_plan.edges
    )


def test_dangling_plan_ref_sid_fails_closed():
    root = PlanNode("plan_ref", attrs={"sid": "missing"}, node_id="ref")

    with pytest.raises(ValueError, match="dangling plan_ref sid 'missing'"):
        BatchGlobalOptimizer().optimize_batch(
            {"factor": root}, {}, {}, _ctx(100, complete=True)
        )


def test_cyclic_plan_ref_sid_fails_closed():
    ref_a = PlanNode("plan_ref", attrs={"sid": "b"}, node_id="ref-a")
    ref_b = PlanNode("plan_ref", attrs={"sid": "a"}, node_id="ref-b")

    with pytest.raises(ValueError, match="cyclic plan dependency"):
        BatchGlobalOptimizer().optimize_batch(
            {"factor": ref_a}, {"a": ref_a, "b": ref_b}, {}, _ctx(100, complete=True)
        )


def test_direct_plan_node_input_cycle_fails_closed():
    first = PlanNode("abs", node_id="first")
    second = PlanNode("neg", inputs=(first,), node_id="second")
    object.__setattr__(first, "inputs", (second,))

    with pytest.raises(ValueError, match="cyclic plan dependency"):
        BatchGlobalOptimizer().optimize_batch(
            {"factor": first}, {}, {}, _ctx(100, complete=True)
        )


def test_known_rows_alone_are_not_production_ready():
    root = PlanNode("abs", inputs=(PlanNode("column"),))

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100)
    )

    assert result.production_ready is False
    assert result.readiness_reason == "byte estimate unavailable"


def test_complete_estimates_and_contracts_are_production_ready(monkeypatch):
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_pandas", lambda *a, **k: True)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_polars", lambda *a, **k: False)
    monkeypatch.setattr("factor_engine.backend.operator_capability.capability_for", lambda *a, **k: SimpleNamespace(
        execution_kind=ExecutionKind.PANDAS_REFERENCE,
        is_production_eligible=lambda: True,
    ))
    root = PlanNode("abs", inputs=(_bound_column(),))

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, mode="production", complete=True)
    )

    assert result.production_ready is True
    assert result.readiness_reason == ""
    assert result.physical_plan.peak_memory_bytes == 800


def test_research_mode_never_certifies_production_readiness():
    root = PlanNode("abs", inputs=(_bound_column(),))

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, mode="research", complete=True)
    )

    assert result.production_ready is False
    assert result.readiness_reason == "production readiness requires production run mode"


def test_empty_plan_is_not_fake_production_ready():
    result = BatchGlobalOptimizer().optimize_batch(
        {}, {}, {}, _ctx(100, complete=True)
    )

    assert result.production_ready is False
    assert result.readiness_reason == "empty logical plan"
    assert result.physical_plan.regions == ()


def test_production_unsupported_operator_fails_closed(monkeypatch):
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_pandas", lambda *a, **k: False)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_polars", lambda *a, **k: False)
    root = PlanNode("synthetic_uncertified", inputs=(PlanNode("column"),))

    with pytest.raises(UnsupportedOperatorBackendError, match="production-certified"):
        BatchGlobalOptimizer().optimize_batch(
            {"root": root}, {}, {}, _ctx(100, mode="production", complete=True)
        )


def test_research_polars_delegate_is_explicit_and_not_ready(monkeypatch):
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_pandas", lambda *a, **k: False)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_polars", lambda *a, **k: True)
    monkeypatch.setattr("factor_engine.backend.polars_backend_kind.canonical_polars_is_delegate", lambda *a, **k: True)
    root = PlanNode("synthetic_delegate", inputs=(PlanNode("column"),))

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, complete=True)
    )

    assert result.per_node_choices["root"].execution_kind == ExecutionKind.POLARS_PANDAS_DELEGATE
    assert result.production_ready is False
    assert result.readiness_reason == "delegate fallback assigned: root"


def test_source_node_inherits_consumer_backend_when_neutral(monkeypatch):
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_pandas", lambda *a, **k: False)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_polars", lambda *a, **k: True)
    monkeypatch.setattr("factor_engine.backend.polars_backend_kind.canonical_polars_is_delegate", lambda *a, **k: False)
    source = PlanNode("column", node_id="source")
    root = PlanNode("synthetic_polars", inputs=(source,), node_id="root")

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, complete=True)
    )

    assert result.per_node_choices["source"].backend == PhysicalBackend.POLARS_PANEL
    assert result.per_node_choices["source"].representation == Representation.POLARS_LONG
    assert result.physical_plan.edges == ()


def test_same_backend_representation_change_has_typed_transfer(monkeypatch):
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_pandas", lambda *a, **k: False)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_polars", lambda *a, **k: True)
    monkeypatch.setattr("factor_engine.backend.polars_backend_kind.canonical_polars_is_delegate", lambda *a, **k: False)
    source = _bound_column(
        source_backend="polars_panel",
        source_representation="polars_wide",
    )
    object.__setattr__(source, "node_id", "source")
    root = PlanNode("synthetic_polars", inputs=(source,), node_id="root")

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, mode="production", complete=True)
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
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_pandas", lambda *a, **k: True)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_polars", lambda *a, **k: False)
    monkeypatch.setattr("factor_engine.backend.operator_capability.capability_for", lambda *a, **k: SimpleNamespace(
        execution_kind=ExecutionKind.PANDAS_REFERENCE,
        is_production_eligible=lambda: True,
    ))
    source = _bound_column(
        source_backend="polars_panel",
        source_representation="polars_wide",
    )
    object.__setattr__(source, "node_id", "source")
    root = PlanNode("synthetic_pandas", inputs=(source,), node_id="root")

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, mode="production", complete=True)
    )

    assert len(result.physical_plan.edges) == 1
    edge = result.physical_plan.edges[0]
    assert edge.source_backend == PhysicalBackend.POLARS_PANEL
    assert edge.target_backend == PhysicalBackend.PANDAS_NUMPY
    assert edge.estimated_bytes == 1600
    assert result.production_ready is True


def test_region_and_transfer_estimates_use_local_node_evidence(monkeypatch):
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_pandas", lambda *a, **k: True)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_polars", lambda *a, **k: False)
    monkeypatch.setattr(
        "factor_engine.backend.operator_capability.capability_for",
        lambda *a, **k: SimpleNamespace(
            execution_kind=ExecutionKind.PANDAS_REFERENCE,
            is_production_eligible=lambda: True,
        ),
    )
    source = PlanNode(
        "column",
        attrs={
            "source_backend": "polars_panel",
            "source_representation": "polars_wide",
            "estimated_rows": 7,
            "estimated_bytes": 111,
            "estimated_memory_bytes": 222,
        },
        node_id="source",
    )
    root = PlanNode(
        "synthetic_pandas",
        inputs=(source,),
        attrs={
            "estimated_rows": 13,
            "estimated_bytes": 333,
            "estimated_memory_bytes": 444,
        },
        node_id="root",
    )

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(100, mode="production", complete=True)
    )

    regions = {
        node_id: region
        for region in result.physical_plan.regions
        for node_id in region.node_ids
    }
    assert regions["source"].estimated_rows == 7
    assert regions["source"].estimated_memory_bytes == 222
    assert regions["root"].estimated_rows == 13
    assert regions["root"].estimated_memory_bytes == 444
    edge = result.physical_plan.edges[0]
    assert edge.estimated_rows == 7
    assert edge.estimated_bytes == 111


def test_native_fraction_is_derived_from_execution_kinds(monkeypatch):
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_pandas", lambda *a, **k: True)
    monkeypatch.setattr("factor_engine.backend.operator_capability.supports_polars", lambda *a, **k: False)
    monkeypatch.setattr(
        "factor_engine.backend.operator_capability.capability_for",
        lambda *a, **k: SimpleNamespace(
            execution_kind=ExecutionKind.NATIVE_EXPR,
            is_production_eligible=lambda: True,
        ),
    )
    source = PlanNode("column", node_id="source")
    root = PlanNode("synthetic_native", inputs=(source,), node_id="root")

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx(10, mode="production", complete=True)
    )

    assert result.physical_plan.native_fraction == 0.5


def test_disconnected_same_residency_nodes_remain_separate():
    left = PlanNode(
        "column",
        attrs={"source_backend": "pandas_numpy", "source_representation": "pandas_long"},
        node_id="left",
    )
    right = PlanNode(
        "column",
        attrs={"source_backend": "pandas_numpy", "source_representation": "pandas_long"},
        node_id="right",
    )

    result = BatchGlobalOptimizer().optimize_batch(
        {"left": left, "right": right}, {}, {}, _ctx(10, complete=True)
    )

    assert len(result.physical_plan.regions) == 2
    assert {tuple(region.node_ids) for region in result.physical_plan.regions} == {
        ("left",),
        ("right",),
    }


def test_exact_search_is_used_at_explicit_threshold():
    source = PlanNode("column", node_id="source")
    root = PlanNode("column", inputs=(source,), node_id="root")

    result = BatchGlobalOptimizer(
        exact_search_max_ambiguous_nodes=2
    ).optimize_batch({"root": root}, {}, {}, _ctx(10, complete=True))

    assert result.optimization_basis == "dp_global_exact"
    assert set(result.per_node_choices) == {"root", "source"}


def test_large_ambiguous_dag_uses_bounded_deterministic_fallback(monkeypatch):
    node = PlanNode("column", node_id="node-0")
    for index in range(1, 18):
        node = PlanNode("column", inputs=(node,), node_id=f"node-{index}")

    optimizer = BatchGlobalOptimizer(
        exact_search_max_ambiguous_nodes=4,
        approximate_max_passes=2,
    )
    monkeypatch.setattr(
        optimizer,
        "_exact_assignment",
        lambda *args, **kwargs: pytest.fail("large DAG entered exact enumeration"),
    )
    local_cost_calls = 0
    original_local_cost = optimizer._local_choice_cost

    def counted_local_cost(*args, **kwargs):
        nonlocal local_cost_calls
        local_cost_calls += 1
        return original_local_cost(*args, **kwargs)

    monkeypatch.setattr(optimizer, "_local_choice_cost", counted_local_cost)
    first = optimizer.optimize_batch(
        {"root": node}, {}, {}, _ctx(10, complete=True)
    )
    first_signature = {
        node_id: (choice.backend, choice.representation)
        for node_id, choice in first.per_node_choices.items()
    }
    first_call_count = local_cost_calls

    local_cost_calls = 0
    second = optimizer.optimize_batch(
        {"root": node}, {}, {}, _ctx(10, complete=True)
    )
    second_signature = {
        node_id: (choice.backend, choice.representation)
        for node_id, choice in second.per_node_choices.items()
    }

    assert first.optimization_basis == "dp_global_approximate"
    assert second.optimization_basis == "dp_global_approximate"
    assert len(first.per_node_choices) == 18
    assert first_signature == second_signature
    assert len(first.physical_plan.regions) == 1
    assert first.physical_plan.edges == ()
    assert first_call_count <= 2 * 18 * 2
    assert local_cost_calls == first_call_count


def test_search_configuration_rejects_unbounded_values():
    with pytest.raises(ValueError, match="exact_search_max_ambiguous_nodes"):
        BatchGlobalOptimizer(exact_search_max_ambiguous_nodes=-1)
    with pytest.raises(ValueError, match="approximate_max_passes"):
        BatchGlobalOptimizer(approximate_max_passes=0)
