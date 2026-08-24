"""
Tests for central review P0 fixes.

Covers correctness issues fixed in central review response.
"""

import pytest

from factor_assets.selection import (
    GateResult,
    CompositeGate,
    ThresholdGate,
    MetricBinding,
    MetricDirection,
    MetricEvidence,
)
from factor_assets.clustering import ModularityClustering, HierarchicalClustering
from factor_assets.graph import SparseCorrelationGraph, CorrelationEdge


def test_composite_gate_explicit_binding():
    """CompositeGate supports explicit metric bindings."""
    gate1 = ThresholdGate(gate_name="minimum_ic", threshold=0.02, higher_is_better=True)
    gate2 = ThresholdGate(gate_name="maximum_turnover", threshold=0.5, higher_is_better=False)

    composite = CompositeGate(
        gate_name="admission_gates",
        gates=[gate1, gate2],
        require_all=True,
        metric_bindings={
            "minimum_ic": MetricBinding("ic_mean", "correlation", MetricDirection.HIGHER_IS_BETTER),
            "maximum_turnover": MetricBinding("turnover", "fraction", MetricDirection.LOWER_IS_BETTER),
        }
    )

    metrics = {
        "ic_mean": MetricEvidence(0.03, "correlation", MetricDirection.HIGHER_IS_BETTER),
        "turnover": MetricEvidence(0.4, "fraction", MetricDirection.LOWER_IS_BETTER),
    }

    composite_eval, sub_evals = composite.evaluate_all(
        factor_id="F001",
        evidence_id="EVD_001",
        metrics=metrics,
    )

    assert composite_eval.result == GateResult.PASS
    assert len(sub_evals) == 2
    assert all(ev.passed for ev in sub_evals)


def test_modularity_clustering_production_fail_closed():
    """ModularityClustering fails closed without allow_toy_algorithm."""
    edges = [
        CorrelationEdge("F1", "F2", 0.8),
        CorrelationEdge("F2", "F3", 0.7),
    ]
    graph = SparseCorrelationGraph(edges)

    # Should fail closed without explicit opt-in
    with pytest.raises(RuntimeError, match="toy/reference implementation"):
        ModularityClustering(graph)


def test_modularity_clustering_allows_explicit_opt_in():
    """ModularityClustering allows testing with explicit flag."""
    edges = [
        CorrelationEdge("F1", "F2", 0.8),
    ]
    graph = SparseCorrelationGraph(edges)

    clustering = ModularityClustering(graph, allow_toy_algorithm=True)
    result = clustering.cluster()
    assert result.num_clusters >= 1


def test_hierarchical_clustering_max_factors_default():
    """HierarchicalClustering enforces default max_factors=500."""
    edges = []
    for i in range(501):
        if i > 0:
            edges.append(CorrelationEdge(f"F{i}", f"F{i-1}", 0.5))

    graph = SparseCorrelationGraph(edges)

    # Should fail with > 500 factors
    with pytest.raises(ValueError, match="exceeds max_factors"):
        HierarchicalClustering(graph)


def test_hierarchical_clustering_explicit_override():
    """HierarchicalClustering can override max_factors."""
    edges = []
    for i in range(50):
        if i > 0:
            edges.append(CorrelationEdge(f"F{i}", f"F{i-1}", 0.5))

    graph = SparseCorrelationGraph(edges)

    clustering = HierarchicalClustering(graph, max_factors=100)
    dendrogram = clustering.build_dendrogram()
    assert len(dendrogram.factor_ids) == 50

