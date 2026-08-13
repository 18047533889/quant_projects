"""
Tests for metric dependency graph.
"""

import pytest

from quant_evaluator.planner.dependency_plan import (
    MetricNode,
    MetricDependencyGraph,
    MetricKind,
    resolve_metric_dependencies,
)
from quant_evaluator.contracts.errors import InvalidContractError


class TestMetricNode:
    def test_node_creation(self):
        node = MetricNode(
            metric_id="ic_mean",
            metric_kind=MetricKind.IC,
            estimated_cost=10.0,
        )

        assert node.metric_id == "ic_mean"
        assert node.metric_kind == MetricKind.IC
        assert node.estimated_cost == 10.0
        assert len(node.dependencies) == 0

    def test_add_dependency(self):
        node = MetricNode(
            metric_id="ic_mean",
            metric_kind=MetricKind.IC,
        )

        node.add_dependency("ic_daily")
        assert node.has_dependency("ic_daily")
        assert not node.has_dependency("other")


class TestMetricDependencyGraph:
    def test_add_node(self):
        graph = MetricDependencyGraph()
        node = MetricNode("coverage", MetricKind.COVERAGE)

        graph.add_node(node)

        assert "coverage" in graph.nodes
        assert graph.get_node("coverage") == node

    def test_duplicate_node_raises(self):
        graph = MetricDependencyGraph()
        node = MetricNode("coverage", MetricKind.COVERAGE)

        graph.add_node(node)

        with pytest.raises(InvalidContractError, match="already exists"):
            graph.add_node(node)

    def test_add_dependency(self):
        graph = MetricDependencyGraph()
        node1 = MetricNode("ic_daily", MetricKind.IC)
        node2 = MetricNode("ic_mean", MetricKind.IC)

        graph.add_node(node1)
        graph.add_node(node2)

        graph.add_dependency("ic_mean", "ic_daily")

        assert node2.has_dependency("ic_daily")

    def test_add_dependency_unknown_node_raises(self):
        graph = MetricDependencyGraph()
        node = MetricNode("ic_mean", MetricKind.IC)
        graph.add_node(node)

        with pytest.raises(InvalidContractError, match="not found"):
            graph.add_dependency("ic_mean", "unknown")

    def test_cycle_detection_no_cycle(self):
        graph = MetricDependencyGraph()

        # Linear chain: A -> B -> C
        graph.add_node(MetricNode("A", MetricKind.CUSTOM))
        graph.add_node(MetricNode("B", MetricKind.CUSTOM))
        graph.add_node(MetricNode("C", MetricKind.CUSTOM))

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        has_cycle, cycle_path = graph.has_cycle()
        assert not has_cycle
        assert cycle_path is None

    def test_cycle_detection_simple_cycle(self):
        graph = MetricDependencyGraph()

        # Cycle: A -> B -> A
        graph.add_node(MetricNode("A", MetricKind.CUSTOM))
        graph.add_node(MetricNode("B", MetricKind.CUSTOM))

        graph.add_dependency("B", "A")
        graph.add_dependency("A", "B")

        has_cycle, cycle_path = graph.has_cycle()
        assert has_cycle
        assert cycle_path is not None
        assert "A" in cycle_path and "B" in cycle_path

    def test_cycle_detection_self_cycle(self):
        graph = MetricDependencyGraph()

        # Self cycle: A -> A
        node = MetricNode("A", MetricKind.CUSTOM)
        node.add_dependency("A")
        graph.add_node(node)

        has_cycle, cycle_path = graph.has_cycle()
        assert has_cycle
        assert cycle_path is not None

    def test_topological_sort_simple(self):
        graph = MetricDependencyGraph()

        # A -> B -> C
        graph.add_node(MetricNode("A", MetricKind.CUSTOM, estimated_cost=1.0))
        graph.add_node(MetricNode("B", MetricKind.CUSTOM, estimated_cost=2.0))
        graph.add_node(MetricNode("C", MetricKind.CUSTOM, estimated_cost=3.0))

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        sorted_ids = graph.topological_sort()

        # A must come before B, B must come before C
        assert sorted_ids.index("A") < sorted_ids.index("B")
        assert sorted_ids.index("B") < sorted_ids.index("C")

    def test_topological_sort_diamond(self):
        graph = MetricDependencyGraph()

        # Diamond: A -> B, A -> C, B -> D, C -> D
        graph.add_node(MetricNode("A", MetricKind.CUSTOM))
        graph.add_node(MetricNode("B", MetricKind.CUSTOM))
        graph.add_node(MetricNode("C", MetricKind.CUSTOM))
        graph.add_node(MetricNode("D", MetricKind.CUSTOM))

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "A")
        graph.add_dependency("D", "B")
        graph.add_dependency("D", "C")

        sorted_ids = graph.topological_sort()

        # A must come before B and C
        assert sorted_ids.index("A") < sorted_ids.index("B")
        assert sorted_ids.index("A") < sorted_ids.index("C")

        # B and C must come before D
        assert sorted_ids.index("B") < sorted_ids.index("D")
        assert sorted_ids.index("C") < sorted_ids.index("D")

    def test_topological_sort_cycle_raises(self):
        graph = MetricDependencyGraph()

        # Cycle: A -> B -> A
        graph.add_node(MetricNode("A", MetricKind.CUSTOM))
        graph.add_node(MetricNode("B", MetricKind.CUSTOM))

        graph.add_dependency("B", "A")
        graph.add_dependency("A", "B")

        with pytest.raises(InvalidContractError, match="cycle"):
            graph.topological_sort()

    def test_execution_levels_simple(self):
        graph = MetricDependencyGraph()

        # A, B (parallel) -> C
        graph.add_node(MetricNode("A", MetricKind.CUSTOM))
        graph.add_node(MetricNode("B", MetricKind.CUSTOM))
        graph.add_node(MetricNode("C", MetricKind.CUSTOM))

        graph.add_dependency("C", "A")
        graph.add_dependency("C", "B")

        levels = graph.get_execution_levels()

        # Level 0: A and B (can run in parallel)
        # Level 1: C
        assert len(levels) == 2
        assert set(levels[0]) == {"A", "B"}
        assert set(levels[1]) == {"C"}

    def test_execution_levels_complex(self):
        graph = MetricDependencyGraph()

        # Level 0: A, B
        # Level 1: C (depends on A), D (depends on B)
        # Level 2: E (depends on C and D)
        graph.add_node(MetricNode("A", MetricKind.CUSTOM))
        graph.add_node(MetricNode("B", MetricKind.CUSTOM))
        graph.add_node(MetricNode("C", MetricKind.CUSTOM))
        graph.add_node(MetricNode("D", MetricKind.CUSTOM))
        graph.add_node(MetricNode("E", MetricKind.CUSTOM))

        graph.add_dependency("C", "A")
        graph.add_dependency("D", "B")
        graph.add_dependency("E", "C")
        graph.add_dependency("E", "D")

        levels = graph.get_execution_levels()

        assert len(levels) == 3
        assert set(levels[0]) == {"A", "B"}
        assert set(levels[1]) == {"C", "D"}
        assert set(levels[2]) == {"E"}

    def test_estimate_total_cost(self):
        graph = MetricDependencyGraph()

        graph.add_node(MetricNode("A", MetricKind.CUSTOM, estimated_cost=10.0))
        graph.add_node(MetricNode("B", MetricKind.CUSTOM, estimated_cost=20.0))
        graph.add_node(MetricNode("C", MetricKind.CUSTOM, estimated_cost=5.0))

        total_cost = graph.estimate_total_cost()
        assert total_cost == 35.0

    def test_critical_path_linear(self):
        graph = MetricDependencyGraph()

        # A (10) -> B (20) -> C (5)
        graph.add_node(MetricNode("A", MetricKind.CUSTOM, estimated_cost=10.0))
        graph.add_node(MetricNode("B", MetricKind.CUSTOM, estimated_cost=20.0))
        graph.add_node(MetricNode("C", MetricKind.CUSTOM, estimated_cost=5.0))

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        path, cost = graph.get_critical_path()

        assert path == ["A", "B", "C"]
        assert cost == 35.0

    def test_critical_path_branching(self):
        graph = MetricDependencyGraph()

        # A (10) -> B (20) -> D (5)
        #        -> C (30) -> D (5)
        # Critical path should be A -> C -> D (45)
        graph.add_node(MetricNode("A", MetricKind.CUSTOM, estimated_cost=10.0))
        graph.add_node(MetricNode("B", MetricKind.CUSTOM, estimated_cost=20.0))
        graph.add_node(MetricNode("C", MetricKind.CUSTOM, estimated_cost=30.0))
        graph.add_node(MetricNode("D", MetricKind.CUSTOM, estimated_cost=5.0))

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "A")
        graph.add_dependency("D", "B")
        graph.add_dependency("D", "C")

        path, cost = graph.get_critical_path()

        assert "A" in path and "C" in path and "D" in path
        assert cost == 45.0


class TestResolveMetricDependencies:
    def test_simple_specs(self):
        specs = [
            {
                "metric_id": "coverage",
                "metric_kind": "coverage",
                "estimated_cost": 1.0,
            },
            {
                "metric_id": "ic_daily",
                "metric_kind": "ic",
                "dependencies": ["coverage"],
                "estimated_cost": 10.0,
            },
        ]

        graph = resolve_metric_dependencies(specs)

        assert "coverage" in graph.nodes
        assert "ic_daily" in graph.nodes
        assert graph.nodes["ic_daily"].has_dependency("coverage")

    def test_missing_metric_id_raises(self):
        specs = [{"metric_kind": "coverage"}]

        with pytest.raises(InvalidContractError, match="metric_id is required"):
            resolve_metric_dependencies(specs)

    def test_missing_metric_kind_raises(self):
        specs = [{"metric_id": "coverage"}]

        with pytest.raises(InvalidContractError, match="metric_kind is required"):
            resolve_metric_dependencies(specs)

    def test_unknown_dependency_raises(self):
        specs = [
            {
                "metric_id": "ic_daily",
                "metric_kind": "ic",
                "dependencies": ["unknown_metric"],
            },
        ]

        with pytest.raises(InvalidContractError, match="unknown metric"):
            resolve_metric_dependencies(specs)

    def test_cycle_raises(self):
        specs = [
            {
                "metric_id": "A",
                "metric_kind": "custom",
                "dependencies": ["B"],
            },
            {
                "metric_id": "B",
                "metric_kind": "custom",
                "dependencies": ["A"],
            },
        ]

        with pytest.raises(InvalidContractError, match="cycle"):
            resolve_metric_dependencies(specs)

    def test_string_metric_kind(self):
        specs = [
            {"metric_id": "coverage", "metric_kind": "coverage"},
            {"metric_id": "ic", "metric_kind": "ic"},
            {"metric_id": "custom", "metric_kind": "unknown_kind"},  # Should become CUSTOM
        ]

        graph = resolve_metric_dependencies(specs)

        assert graph.nodes["coverage"].metric_kind == MetricKind.COVERAGE
        assert graph.nodes["ic"].metric_kind == MetricKind.IC
        assert graph.nodes["custom"].metric_kind == MetricKind.CUSTOM

    def test_metadata_preserved(self):
        specs = [
            {
                "metric_id": "ic_daily",
                "metric_kind": "ic",
                "metadata": {"method": "spearman"},
                "parallelizable": False,
                "requires_full_batch": True,
            },
        ]

        graph = resolve_metric_dependencies(specs)

        node = graph.nodes["ic_daily"]
        assert node.metadata == {"method": "spearman"}
        assert node.parallelizable is False
        assert node.requires_full_batch is True
