"""Negative tests for Pareto frontier."""

import pytest
from factor_assets.optimizer.pareto import (
    ParetoPoint,
    ParetoOptimizer,
)


def test_pareto_empty_points():
    """Test Pareto frontier with empty points."""
    optimizer = ParetoOptimizer(objectives=["rank_ic", "sharpe"])
    frontier = optimizer.compute_frontier([])

    assert len(frontier.points) == 0
    assert frontier.dominated_count == 0


def test_pareto_single_point():
    """Test Pareto frontier with single point."""
    optimizer = ParetoOptimizer(objectives=["rank_ic"])

    points = [{"point_id": "a", "objectives": {"rank_ic": 0.05}}]
    frontier = optimizer.compute_frontier(points)

    assert len(frontier.points) == 1


def test_pareto_missing_objectives_are_ineligible():
    optimizer = ParetoOptimizer(objectives=["rank_ic", "sharpe"])

    frontier = optimizer.compute_frontier([
        {"point_id": "a", "objectives": {"rank_ic": 0.05}},
        {"point_id": "b", "objectives": {"rank_ic": 0.03, "sharpe": 1.5}},
    ])
    assert [p.point_id for p in frontier.points] == ["b"]
    assert frontier.dominated_count == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_pareto_nonfinite_objectives_are_ineligible(bad):
    optimizer = ParetoOptimizer(objectives=["rank_ic", "sharpe"])
    frontier = optimizer.compute_frontier([
        {"point_id": "bad", "objectives": {"rank_ic": bad, "sharpe": 99.0}},
        {"point_id": "good", "objectives": {"rank_ic": 0.03, "sharpe": 1.5}},
    ])
    assert [p.point_id for p in frontier.points] == ["good"]


def test_pareto_all_dominated():
    """Test when all points dominated by one."""
    optimizer = ParetoOptimizer(objectives=["rank_ic", "sharpe"])

    points = [
        {"point_id": "best", "objectives": {"rank_ic": 0.10, "sharpe": 2.0}},
        {"point_id": "worse1", "objectives": {"rank_ic": 0.05, "sharpe": 1.5}},
        {"point_id": "worse2", "objectives": {"rank_ic": 0.03, "sharpe": 1.0}},
    ]

    frontier = optimizer.compute_frontier(points)

    assert len(frontier.points) == 1
    assert frontier.points[0].point_id == "best"
    assert frontier.dominated_count == 2


def test_pareto_identical_points():
    """Test identical points (neither dominates)."""
    optimizer = ParetoOptimizer(objectives=["rank_ic", "sharpe"])

    points = [
        {"point_id": "a", "objectives": {"rank_ic": 0.05, "sharpe": 1.5}},
        {"point_id": "b", "objectives": {"rank_ic": 0.05, "sharpe": 1.5}},
    ]

    frontier = optimizer.compute_frontier(points)

    # Both on frontier since neither strictly dominates
    assert len(frontier.points) == 2


def test_pareto_diverse_subset_empty():
    """Test diverse subset from empty frontier."""
    optimizer = ParetoOptimizer(objectives=["rank_ic"])

    from factor_assets.optimizer.pareto import ParetoFrontier
    frontier = ParetoFrontier(points=[], objectives=["rank_ic"])

    selected = optimizer.select_diverse_subset(frontier, max_size=5)
    assert len(selected) == 0


def test_pareto_diverse_subset_zero_max():
    """Test diverse subset with zero max_size."""
    optimizer = ParetoOptimizer(objectives=["rank_ic"])

    from factor_assets.optimizer.pareto import ParetoFrontier
    frontier = ParetoFrontier(
        points=[ParetoPoint("a", {"rank_ic": 0.05})],
        objectives=["rank_ic"],
    )

    selected = optimizer.select_diverse_subset(frontier, max_size=0)
    assert len(selected) == 0


def test_pareto_no_objectives():
    """Test with empty objectives list."""
    optimizer = ParetoOptimizer(objectives=[])

    points = [
        {"point_id": "a", "objectives": {"rank_ic": 0.05}},
        {"point_id": "b", "objectives": {"rank_ic": 0.03}},
    ]

    # With no objectives, no point can dominate
    frontier = optimizer.compute_frontier(points)
    assert len(frontier.points) == 2
