"""Tests for Pareto frontier tracking."""

import pytest

from factor_optimizer.search.pareto import ParetoPoint, ParetoFrontier, ParetoArchive


def test_pareto_point_creation():
    point = ParetoPoint(trial_id="t1", objectives=(0.8, 0.6))

    assert point.trial_id == "t1"
    assert point.objectives == (0.8, 0.6)
    assert len(point.metadata) == 0


def test_pareto_point_validation():
    with pytest.raises(ValueError, match="objectives cannot be empty"):
        ParetoPoint(trial_id="t1", objectives=())

    with pytest.raises(ValueError, match="must be numeric"):
        ParetoPoint(trial_id="t1", objectives=("a", "b"))

    with pytest.raises(ValueError, match="must be finite"):
        ParetoPoint(trial_id="t1", objectives=(float("nan"), 1.0))
    with pytest.raises(ValueError, match="must be finite"):
        ParetoPoint(trial_id="t1", objectives=(float("inf"), 1.0))


def test_dominated_cache_is_cleared_for_improved_trial():
    frontier = ParetoFrontier()
    assert frontier.add_point(ParetoPoint("t1", (2.0, 2.0)))
    assert not frontier.add_point(ParetoPoint("t2", (1.0, 1.0)))
    improved = ParetoPoint("t2", (3.0, 3.0))
    assert frontier.add_point(improved)
    assert not frontier.is_dominated(improved)


def test_pareto_point_dominates():
    p1 = ParetoPoint(trial_id="t1", objectives=(0.8, 0.7))
    p2 = ParetoPoint(trial_id="t2", objectives=(0.6, 0.5))
    p3 = ParetoPoint(trial_id="t3", objectives=(0.9, 0.4))

    # p1 dominates p2 (better on both)
    assert p1.dominates(p2)
    assert not p2.dominates(p1)

    # p1 and p3 don't dominate each other
    assert not p1.dominates(p3)
    assert not p3.dominates(p1)


def test_pareto_point_dominates_equal():
    p1 = ParetoPoint(trial_id="t1", objectives=(0.8, 0.7))
    p2 = ParetoPoint(trial_id="t2", objectives=(0.8, 0.7))

    # Equal points don't dominate each other
    assert not p1.dominates(p2)
    assert not p2.dominates(p1)


def test_pareto_point_distance():
    p1 = ParetoPoint(trial_id="t1", objectives=(0.0, 0.0))
    p2 = ParetoPoint(trial_id="t2", objectives=(3.0, 4.0))

    distance = p1.distance_to(p2)
    assert distance == 5.0


def test_pareto_frontier_initialization():
    frontier = ParetoFrontier(objective_names=["ic", "sharpe"])

    assert frontier.size == 0
    assert frontier.num_dimensions == 0
    assert frontier.objective_names == ["ic", "sharpe"]


def test_pareto_frontier_add_point():
    frontier = ParetoFrontier()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.5, 0.5))
    added = frontier.add_point(p1)

    assert added
    assert frontier.size == 1


def test_pareto_frontier_dominated_point():
    frontier = ParetoFrontier()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.8, 0.7))
    p2 = ParetoPoint(trial_id="t2", objectives=(0.6, 0.5))

    frontier.add_point(p1)
    added = frontier.add_point(p2)

    # p2 is dominated by p1
    assert not added
    assert frontier.size == 1


def test_pareto_frontier_remove_dominated():
    frontier = ParetoFrontier()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.6, 0.5))
    p2 = ParetoPoint(trial_id="t2", objectives=(0.8, 0.7))

    frontier.add_point(p1)
    frontier.add_point(p2)

    # p2 dominates p1, so p1 should be removed
    assert frontier.size == 1
    assert frontier.get_point("t2") is not None
    assert frontier.get_point("t1") is None


def test_pareto_frontier_non_dominated():
    frontier = ParetoFrontier()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.9, 0.4))
    p2 = ParetoPoint(trial_id="t2", objectives=(0.4, 0.9))
    p3 = ParetoPoint(trial_id="t3", objectives=(0.6, 0.6))

    frontier.add_point(p1)
    frontier.add_point(p2)
    frontier.add_point(p3)

    # All three are non-dominated
    assert frontier.size == 3


def test_pareto_frontier_is_dominated():
    frontier = ParetoFrontier()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.8, 0.7))
    p2 = ParetoPoint(trial_id="t2", objectives=(0.6, 0.5))

    frontier.add_point(p1)

    assert not frontier.is_dominated(p1)
    assert frontier.is_dominated(p2)


def test_pareto_frontier_get_point():
    frontier = ParetoFrontier()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.5, 0.5))
    frontier.add_point(p1)

    retrieved = frontier.get_point("t1")
    assert retrieved is not None
    assert retrieved.trial_id == "t1"

    not_found = frontier.get_point("t999")
    assert not_found is None


def test_pareto_frontier_extremes():
    frontier = ParetoFrontier()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.9, 0.4))
    p2 = ParetoPoint(trial_id="t2", objectives=(0.4, 0.9))
    p3 = ParetoPoint(trial_id="t3", objectives=(0.6, 0.6))

    frontier.add_point(p1)
    frontier.add_point(p2)
    frontier.add_point(p3)

    extremes = frontier.extremes()

    assert extremes[0].trial_id == "t1"  # Best on first objective
    assert extremes[1].trial_id == "t2"  # Best on second objective


def test_pareto_frontier_hypervolume_2d():
    frontier = ParetoFrontier()

    p1 = ParetoPoint(trial_id="t1", objectives=(1.0, 0.5))
    p2 = ParetoPoint(trial_id="t2", objectives=(0.5, 1.0))

    frontier.add_point(p1)
    frontier.add_point(p2)

    reference = (0.0, 0.0)
    hv = frontier.hypervolume(reference)

    # Approximate check
    assert hv > 0
    assert hv < 2.0


def test_pareto_frontier_hypervolume_not_2d():
    frontier = ParetoFrontier()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.5, 0.5, 0.5))
    frontier.add_point(p1)

    with pytest.raises(NotImplementedError):
        frontier.hypervolume((0.0, 0.0, 0.0))


def test_pareto_frontier_coverage():
    frontier1 = ParetoFrontier()
    frontier2 = ParetoFrontier()

    # frontier1 dominates frontier2
    frontier1.add_point(ParetoPoint(trial_id="t1", objectives=(0.9, 0.8)))
    frontier2.add_point(ParetoPoint(trial_id="t2", objectives=(0.5, 0.5)))

    coverage = frontier1.coverage(frontier2)
    assert coverage == 1.0

    # No coverage
    frontier3 = ParetoFrontier()
    frontier3.add_point(ParetoPoint(trial_id="t3", objectives=(0.3, 0.3)))

    coverage = frontier3.coverage(frontier1)
    assert coverage == 0.0


def test_pareto_frontier_spacing():
    frontier = ParetoFrontier()

    # Evenly spaced points
    frontier.add_point(ParetoPoint(trial_id="t1", objectives=(1.0, 0.0)))
    frontier.add_point(ParetoPoint(trial_id="t2", objectives=(0.5, 0.5)))
    frontier.add_point(ParetoPoint(trial_id="t3", objectives=(0.0, 1.0)))

    spacing = frontier.spacing()
    assert spacing >= 0


def test_pareto_frontier_serialization():
    frontier = ParetoFrontier(objective_names=["obj1", "obj2"])

    p1 = ParetoPoint(trial_id="t1", objectives=(0.8, 0.6), metadata={"gen": 1})
    frontier.add_point(p1)

    data = frontier.to_dict()

    assert "points" in data
    assert "objective_names" in data
    assert data["size"] == 1
    assert data["num_dimensions"] == 2

    # Deserialize
    restored = ParetoFrontier.from_dict(data)
    assert restored.size == 1
    assert restored.objective_names == ["obj1", "obj2"]
    assert restored.get_point("t1") is not None


def test_pareto_archive_initialization():
    archive = ParetoArchive(objective_names=["ic", "sharpe"])

    assert archive.objective_names == ["ic", "sharpe"]
    assert archive.current_frontier.size == 0
    assert len(archive.frontiers) == 0


def test_pareto_archive_add_point():
    archive = ParetoArchive()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.5, 0.5))
    added = archive.add_point(p1)

    assert added
    assert archive.current_frontier.size == 1


def test_pareto_archive_snapshot():
    archive = ParetoArchive()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.5, 0.5))
    archive.add_point(p1)

    archive.snapshot(iteration=0)

    assert len(archive.frontiers) == 1

    # Add another point
    p2 = ParetoPoint(trial_id="t2", objectives=(0.8, 0.8))
    archive.add_point(p2)

    archive.snapshot(iteration=1)

    assert len(archive.frontiers) == 2


def test_pareto_archive_get_frontier():
    archive = ParetoArchive()

    p1 = ParetoPoint(trial_id="t1", objectives=(0.5, 0.5))
    archive.add_point(p1)
    archive.snapshot(iteration=0)

    # Get historical frontier
    frontier_0 = archive.get_frontier(iteration=0)
    assert frontier_0.size == 1

    # Get current frontier
    current = archive.get_frontier()
    assert current is archive.current_frontier


def test_pareto_archive_get_frontier_invalid():
    archive = ParetoArchive()

    with pytest.raises(ValueError):
        archive.get_frontier(iteration=999)


def test_pareto_archive_frontier_growth():
    archive = ParetoArchive()

    archive.add_point(ParetoPoint(trial_id="t1", objectives=(0.5, 0.5)))
    archive.snapshot(iteration=0)

    archive.add_point(ParetoPoint(trial_id="t2", objectives=(0.8, 0.8)))
    archive.snapshot(iteration=1)

    growth = archive.frontier_growth()

    assert len(growth) == 3  # 2 snapshots + current
    assert growth[0] == (0, 1)
    assert growth[1] == (1, 1)  # Second point dominated first
    assert growth[2][1] == 1


def test_pareto_archive_hypervolume_progress():
    archive = ParetoArchive()

    archive.add_point(ParetoPoint(trial_id="t1", objectives=(0.5, 0.5)))
    archive.snapshot(iteration=0)

    archive.add_point(ParetoPoint(trial_id="t2", objectives=(0.8, 0.3)))
    archive.snapshot(iteration=1)

    reference = (0.0, 0.0)
    progress = archive.hypervolume_progress(reference)

    assert len(progress) == 3
    assert all(hv >= 0 for _, hv in progress)


def test_pareto_frontier_empty_operations():
    frontier = ParetoFrontier()

    assert frontier.extremes() == {}
    assert frontier.hypervolume((0.0, 0.0)) == 0.0
    assert frontier.spacing() == 0.0


def test_pareto_point_distance_different_dimensions():
    p1 = ParetoPoint(trial_id="t1", objectives=(0.5, 0.5))
    p2 = ParetoPoint(trial_id="t2", objectives=(0.5, 0.5, 0.5))

    with pytest.raises(ValueError):
        p1.distance_to(p2)


def test_pareto_point_dominates_different_dimensions():
    p1 = ParetoPoint(trial_id="t1", objectives=(0.5, 0.5))
    p2 = ParetoPoint(trial_id="t2", objectives=(0.5, 0.5, 0.5))

    with pytest.raises(ValueError):
        p1.dominates(p2)
