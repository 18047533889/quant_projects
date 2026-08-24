"""Tests for Pareto frontier."""

import pytest
from factor_assets.optimizer.pareto import (
    ParetoPoint,
    ParetoFrontier,
    ParetoOptimizer,
    compute_pareto_frontier,
)


def test_pareto_point_creation():
    """Test Pareto point creation."""
    point = ParetoPoint(
        point_id="point_001",
        objectives={"rank_ic": 0.05, "sharpe": 1.5},
        metadata={"asset_id": "asset_123"},
    )

    assert point.point_id == "point_001"
    assert point.objectives["rank_ic"] == 0.05
    assert point.metadata["asset_id"] == "asset_123"


def test_pareto_point_dominance():
    """Test dominance relation."""
    point_a = ParetoPoint(
        point_id="a",
        objectives={"rank_ic": 0.05, "sharpe": 1.5},
    )

    point_b = ParetoPoint(
        point_id="b",
        objectives={"rank_ic": 0.03, "sharpe": 1.2},
    )

    objectives = ["rank_ic", "sharpe"]

    # A dominates B (better on both)
    assert point_a.dominates(point_b, objectives)
    assert not point_b.dominates(point_a, objectives)


def test_pareto_point_no_dominance():
    """Test no dominance when tradeoff exists."""
    point_a = ParetoPoint(
        point_id="a",
        objectives={"rank_ic": 0.05, "sharpe": 1.2},
    )

    point_b = ParetoPoint(
        point_id="b",
        objectives={"rank_ic": 0.03, "sharpe": 1.5},
    )

    objectives = ["rank_ic", "sharpe"]

    # Neither dominates (tradeoff)
    assert not point_a.dominates(point_b, objectives)
    assert not point_b.dominates(point_a, objectives)


def test_pareto_optimizer_compute_frontier():
    """Test Pareto frontier computation."""
    optimizer = ParetoOptimizer(objectives=["rank_ic", "sharpe"])

    points = [
        {
            "point_id": "a",
            "objectives": {"rank_ic": 0.05, "sharpe": 1.5},
        },
        {
            "point_id": "b",
            "objectives": {"rank_ic": 0.03, "sharpe": 1.2},
        },
        {
            "point_id": "c",
            "objectives": {"rank_ic": 0.04, "sharpe": 1.6},
        },
    ]

    frontier = optimizer.compute_frontier(points)

    # Only a and c should be on frontier (b is dominated by a)
    assert len(frontier.points) == 2
    frontier_ids = {p.point_id for p in frontier.points}
    assert "a" in frontier_ids
    assert "c" in frontier_ids
    assert "b" not in frontier_ids


def test_pareto_optimizer_all_non_dominated():
    """Test frontier when all points non-dominated."""
    optimizer = ParetoOptimizer(objectives=["rank_ic", "sharpe"])

    points = [
        {"point_id": "a", "objectives": {"rank_ic": 0.05, "sharpe": 1.2}},
        {"point_id": "b", "objectives": {"rank_ic": 0.03, "sharpe": 1.5}},
    ]

    frontier = optimizer.compute_frontier(points)

    # Both are on frontier (tradeoff)
    assert len(frontier.points) == 2


def test_pareto_optimizer_is_dominated():
    """Test dominance check against frontier."""
    optimizer = ParetoOptimizer(objectives=["rank_ic", "sharpe"])

    frontier = ParetoFrontier(
        points=[
            ParetoPoint("a", {"rank_ic": 0.05, "sharpe": 1.5}),
            ParetoPoint("b", {"rank_ic": 0.04, "sharpe": 1.6}),
        ],
        objectives=["rank_ic", "sharpe"],
    )

    dominated_point = ParetoPoint("c", {"rank_ic": 0.03, "sharpe": 1.2})
    non_dominated_point = ParetoPoint("d", {"rank_ic": 0.06, "sharpe": 1.4})

    assert optimizer.is_dominated(dominated_point, frontier)
    assert not optimizer.is_dominated(non_dominated_point, frontier)


def test_pareto_optimizer_select_diverse_subset():
    """Test diverse subset selection."""
    optimizer = ParetoOptimizer(objectives=["rank_ic", "sharpe"])

    frontier = ParetoFrontier(
        points=[
            ParetoPoint("a", {"rank_ic": 0.05, "sharpe": 1.5}),
            ParetoPoint("b", {"rank_ic": 0.04, "sharpe": 1.6}),
            ParetoPoint("c", {"rank_ic": 0.03, "sharpe": 1.7}),
            ParetoPoint("d", {"rank_ic": 0.06, "sharpe": 1.4}),
        ],
        objectives=["rank_ic", "sharpe"],
    )

    selected = optimizer.select_diverse_subset(frontier, max_size=2)

    assert len(selected) == 2
    assert all(isinstance(s, str) for s in selected)


def test_pareto_optimizer_select_diverse_all():
    """Test diverse selection when max_size >= frontier size."""
    optimizer = ParetoOptimizer(objectives=["rank_ic", "sharpe"])

    frontier = ParetoFrontier(
        points=[
            ParetoPoint("a", {"rank_ic": 0.05, "sharpe": 1.5}),
            ParetoPoint("b", {"rank_ic": 0.04, "sharpe": 1.6}),
        ],
        objectives=["rank_ic", "sharpe"],
    )

    selected = optimizer.select_diverse_subset(frontier, max_size=10)

    # Should return all points
    assert len(selected) == 2


def test_compute_pareto_frontier_convenience():
    """Test convenience function."""
    points = [
        {"point_id": "a", "objectives": {"rank_ic": 0.05, "sharpe": 1.5}},
        {"point_id": "b", "objectives": {"rank_ic": 0.03, "sharpe": 1.2}},
    ]

    frontier = compute_pareto_frontier(points, objectives=["rank_ic", "sharpe"])

    assert isinstance(frontier, ParetoFrontier)
    assert len(frontier.points) >= 1


def test_pareto_frontier_serialization():
    """Test frontier serialization."""
    frontier = ParetoFrontier(
        points=[
            ParetoPoint("a", {"rank_ic": 0.05, "sharpe": 1.5}),
        ],
        objectives=["rank_ic", "sharpe"],
        dominated_count=2,
    )

    data = frontier.to_dict()
    assert data["dominated_count"] == 2
    assert len(data["points"]) == 1
