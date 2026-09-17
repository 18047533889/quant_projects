from types import SimpleNamespace

import pytest

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.multibackend.batch_global_optimizer import (
    PhysicalBatchGlobalOptimizer,
    plan_proves_bounded_scalar,
)


def _derive(nodes, graph, *, costs=None):
    return PhysicalBatchGlobalOptimizer()._derive_node_estimates(
        nodes,
        graph,
        roots=(),
        global_rows=10_000,
        global_bytes=80_000,
        global_memory=80_000,
        column_scan_costs=costs or {},
    )


def test_scalar_ref_and_where_shape_contracts_do_not_borrow_global_panel_rows():
    one = PlanNode("literal", attrs={"value": 1})
    two = PlanNode("literal", attrs={"value": 2.0})
    scalar = PlanNode("safe_div_null", inputs=(one, two))
    scalar_ref = PlanNode("plan_ref", attrs={"sid": "scalar"})
    panel = PlanNode("column", attrs={"name": "close"})
    panel_ref = PlanNode("plan_ref", attrs={"sid": "panel"})
    where = PlanNode("where", inputs=(panel_ref, scalar_ref, panel_ref))
    nodes = {
        "one": one,
        "two": two,
        "scalar": scalar,
        "scalar_ref": scalar_ref,
        "panel": panel,
        "panel_ref": panel_ref,
        "where": where,
    }
    graph = {
        "one": [],
        "two": [],
        "scalar": ["one", "two"],
        "scalar_ref": ["scalar"],
        "panel": [],
        "panel_ref": ["panel"],
        "where": ["panel_ref", "scalar_ref", "panel_ref"],
    }
    costs = {"close": SimpleNamespace(estimated_rows=100, projection_bytes=800)}
    out = _derive(nodes, graph, costs=costs)
    assert out["scalar"] == (1, 32, 32)
    assert out["scalar_ref"] == out["scalar"]
    assert out["panel_ref"] == (100, 800, 800)
    assert out["where"] == (100, 800, 800)


def test_unknown_source_or_container_literal_cannot_be_masked_by_known_where_branch():
    panel = PlanNode("column", attrs={"name": "close"})
    unknown = PlanNode("column", attrs={"name": "other", "dataset": "secondary"})
    container = PlanNode("literal", attrs={"value": [1, 2, 3]})
    bad_source = PlanNode("where", inputs=(panel, unknown, panel))
    bad_literal = PlanNode("where", inputs=(panel, container, panel))
    nodes = {
        "panel": panel,
        "unknown": unknown,
        "container": container,
        "bad_source": bad_source,
        "bad_literal": bad_literal,
    }
    graph = {
        "panel": [],
        "unknown": [],
        "container": [],
        "bad_source": ["panel", "unknown", "panel"],
        "bad_literal": ["panel", "container", "panel"],
    }
    costs = {"close": SimpleNamespace(estimated_rows=100, projection_bytes=800)}
    out = _derive(nodes, graph, costs=costs)
    assert out["unknown"] == (0, 0, 0)
    assert out["bad_source"] == (0, 0, 0)
    assert out["bad_literal"] == (0, 0, 0)


def test_declared_sequence_parameter_is_shape_neutral_but_not_scalar_proof():
    import factor_engine.cleaned_operators.multiscale_trend  # noqa: F401

    panel = PlanNode("column", attrs={"name": "close"})
    window = PlanNode("literal", attrs={"value": 60})
    scales = PlanNode("literal", attrs={"value": [5, 10, 20, 40]})
    configured = PlanNode(
        "ts_multiscale_trend_consensus", inputs=(panel, window, scales)
    )
    nodes = {
        "panel": panel,
        "window": window,
        "scales": scales,
        "configured": configured,
    }
    graph = {
        "panel": [],
        "window": [],
        "scales": [],
        "configured": ["panel", "window", "scales"],
    }
    costs = {"close": SimpleNamespace(estimated_rows=100, projection_bytes=800)}

    out = _derive(nodes, graph, costs=costs)
    assert out["scales"][0] == 1
    assert out["configured"] == (100, 800, 800)
    assert not plan_proves_bounded_scalar(scales)


def test_declared_sequence_shared_with_panel_position_remains_unknown():
    import factor_engine.cleaned_operators.multiscale_trend  # noqa: F401

    panel = PlanNode("column", attrs={"name": "close"})
    window = PlanNode("literal", attrs={"value": 60})
    scales = PlanNode("literal", attrs={"value": [5, 10, 20, 40]})
    configured = PlanNode(
        "ts_multiscale_trend_consensus", inputs=(panel, window, scales)
    )
    unsafe = PlanNode("where", inputs=(panel, scales, panel))
    nodes = {
        "panel": panel,
        "window": window,
        "scales": scales,
        "configured": configured,
        "unsafe": unsafe,
    }
    graph = {
        "panel": [],
        "window": [],
        "scales": [],
        "configured": ["panel", "window", "scales"],
        "unsafe": ["panel", "scales", "panel"],
    }
    costs = {"close": SimpleNamespace(estimated_rows=100, projection_bytes=800)}

    out = _derive(nodes, graph, costs=costs)
    assert out["scales"] == (0, 0, 0)
    assert out["configured"] == (0, 0, 0)
    assert out["unsafe"] == (0, 0, 0)


def test_declared_sequence_rejects_cycles_and_oversized_reference_counts():
    import factor_engine.cleaned_operators.multiscale_trend  # noqa: F401

    panel = PlanNode("column", attrs={"name": "close"})
    window = PlanNode("literal", attrs={"value": 60})
    cyclic = []
    cyclic.append(cyclic)
    for value in (cyclic, [5] * 4097):
        scales = PlanNode("literal", attrs={"value": value})
        configured = PlanNode(
            "ts_multiscale_trend_consensus", inputs=(panel, window, scales)
        )
        nodes = {
            "panel": panel,
            "window": window,
            "scales": scales,
            "configured": configured,
        }
        graph = {
            "panel": [],
            "window": [],
            "scales": [],
            "configured": ["panel", "window", "scales"],
        }
        costs = {"close": SimpleNamespace(estimated_rows=100, projection_bytes=800)}

        out = _derive(nodes, graph, costs=costs)
        assert out["scales"] == (0, 0, 0)
        assert out["configured"] == (0, 0, 0)


def test_plan_ref_requires_exactly_one_sid_dependency():
    one = PlanNode("literal", attrs={"value": 1})
    two = PlanNode("literal", attrs={"value": 2})
    ref = PlanNode("plan_ref", attrs={"sid": "one"})
    nodes = {"one": one, "two": two, "ref": ref}
    with pytest.raises(ValueError, match="exactly one SID dependency"):
        _derive(nodes, {"one": [], "two": [], "ref": ["one", "two"]})


def test_arbitrary_or_axis_changing_literal_operator_does_not_seed_scalar_shape():
    one = PlanNode("literal", attrs={"value": 1})
    arbitrary = PlanNode("vector_expand", inputs=(one,))
    explicit_axis = PlanNode(
        "safe_div_null",
        inputs=(one, one),
        attrs={"output_axis": "time"},
    )
    nodes = {"one": one, "arbitrary": arbitrary, "explicit_axis": explicit_axis}
    graph = {"one": [], "arbitrary": ["one"], "explicit_axis": ["one", "one"]}
    out = _derive(nodes, graph)
    assert out["arbitrary"] == (0, 0, 0)
    assert out["explicit_axis"] == (0, 0, 0)


def test_bounded_string_control_is_shape_neutral_but_not_scalar_output_proof():
    panel = PlanNode("column", attrs={"name": "close"})
    policy = PlanNode("literal", attrs={"value": "peaks"})
    configured = PlanNode(
        "ts_extrema_divergence_strength", inputs=(panel, panel, policy)
    )
    nodes = {"panel": panel, "policy": policy, "configured": configured}
    graph = {
        "panel": [],
        "policy": [],
        "configured": ["panel", "panel", "policy"],
    }
    costs = {"close": SimpleNamespace(estimated_rows=100, projection_bytes=800)}

    assert _derive(nodes, graph, costs=costs)["configured"] == (100, 800, 800)
    assert not plan_proves_bounded_scalar(policy)


def test_scalar_proof_is_iterative_memoized_and_cycle_closed():
    leaf = PlanNode("literal", attrs={"value": True})
    deep = leaf
    for _ in range(3000):
        deep = PlanNode("not", inputs=(deep,))
    assert plan_proves_bounded_scalar(deep)

    left = PlanNode("plan_ref", attrs={"sid": "right"})
    right = PlanNode("plan_ref", attrs={"sid": "left"})
    assert not plan_proves_bounded_scalar(left, {"left": left, "right": right})


def test_scalar_proof_rejects_numeric_subclasses():
    class IntSubclass(int):
        pass

    assert not plan_proves_bounded_scalar(
        PlanNode("literal", attrs={"value": IntSubclass(1)})
    )


def test_shape_derivation_does_not_scan_root_tuple_per_node():
    class CountingRoots(tuple):
        contains_calls = 0

        def __contains__(self, item):
            type(self).contains_calls += 1
            return super().__contains__(item)

    roots = CountingRoots(f"root_{index}" for index in range(100))
    nodes = {
        node_id: PlanNode("literal", attrs={"value": index})
        for index, node_id in enumerate(roots)
    }
    graph = {node_id: [] for node_id in nodes}
    PhysicalBatchGlobalOptimizer()._derive_node_estimates(
        nodes,
        graph,
        roots=roots,
        global_rows=1,
        global_bytes=8,
        global_memory=8,
    )

    assert CountingRoots.contains_calls == 0
