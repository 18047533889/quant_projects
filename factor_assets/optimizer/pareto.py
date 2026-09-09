"""
True Pareto frontier computation with dominance relation.

No weighted-score approximation: uses strict multi-objective dominance.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set


@dataclass
class ParetoPoint:
    """
    Point in multi-objective space.

    Attributes:
        point_id: Unique identifier
        objectives: Objective values (higher is better)
        metadata: Additional point metadata
    """

    point_id: str
    objectives: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def dominates(self, other: "ParetoPoint", objectives: List[str]) -> bool:
        """
        Check if this point dominates another.

        Point A dominates B if:
        - A is at least as good as B on all objectives
        - A is strictly better than B on at least one objective

        Args:
            other: Other point to compare
            objectives: List of objective names to consider

        Returns:
            True if this point dominates other
        """
        at_least_as_good = True
        strictly_better = False

        for obj_name in objectives:
            if obj_name not in self.objectives or obj_name not in other.objectives:
                return False
            self_val = self.objectives[obj_name]
            other_val = other.objectives[obj_name]
            if not math.isfinite(float(self_val)) or not math.isfinite(float(other_val)):
                return False

            if self_val < other_val:
                at_least_as_good = False
                break

            if self_val > other_val:
                strictly_better = True

        return at_least_as_good and strictly_better

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "point_id": self.point_id,
            "objectives": dict(self.objectives),
            "metadata": dict(self.metadata),
        }


@dataclass
class ParetoFrontier:
    """
    Non-dominated Pareto frontier.

    Attributes:
        points: Points on the frontier
        objectives: Objective names
        dominated_count: Number of dominated points excluded
    """

    points: List[ParetoPoint]
    objectives: List[str]
    dominated_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "points": [p.to_dict() for p in self.points],
            "objectives": list(self.objectives),
            "dominated_count": self.dominated_count,
        }


class ParetoOptimizer:
    """
    True Pareto frontier computation.

    Uses dominance relation, not weighted scoring.
    """

    def __init__(self, objectives: List[str]):
        """
        Initialize optimizer.

        Args:
            objectives: List of objective names (all maximization)
        """
        self.objectives = objectives

    def compute_frontier(
        self,
        points: List[Dict[str, Any]],
        point_id_key: str = "point_id",
        objective_key: str = "objectives",
    ) -> ParetoFrontier:
        """
        Compute non-dominated Pareto frontier.

        Args:
            points: List of point dictionaries
            point_id_key: Key for point ID
            objective_key: Key for objectives dict

        Returns:
            ParetoFrontier with non-dominated points
        """
        if not points:
            return ParetoFrontier(points=[], objectives=self.objectives)

        # Convert to ParetoPoint objects
        pareto_points = []
        invalid_count = 0
        for p in points:
            point_id = p.get(point_id_key, "")
            objectives = p.get(objective_key, {})
            metadata = {k: v for k, v in p.items() if k not in {point_id_key, objective_key}}

            if (
                not point_id
                or not isinstance(objectives, dict)
                or any(name not in objectives for name in self.objectives)
                or any(
                    isinstance(objectives.get(name), bool)
                    or not isinstance(objectives.get(name), (int, float))
                    or not math.isfinite(float(objectives[name]))
                    for name in self.objectives
                )
            ):
                invalid_count += 1
                continue
            pareto_points.append(
                ParetoPoint(
                    point_id=point_id,
                    objectives=objectives,
                    metadata=metadata,
                )
            )

        # Find non-dominated points
        non_dominated = []
        dominated_count = invalid_count

        for i, point in enumerate(pareto_points):
            is_dominated = False

            for j, other in enumerate(pareto_points):
                if i == j:
                    continue

                if other.dominates(point, self.objectives):
                    is_dominated = True
                    dominated_count += 1
                    break

            if not is_dominated:
                non_dominated.append(point)

        return ParetoFrontier(
            points=non_dominated,
            objectives=self.objectives,
            dominated_count=dominated_count,
        )

    def is_dominated(
        self,
        point: ParetoPoint,
        frontier: ParetoFrontier,
    ) -> bool:
        """
        Check if point is dominated by any point in frontier.

        Args:
            point: Point to check
            frontier: Frontier to check against

        Returns:
            True if point is dominated
        """
        for frontier_point in frontier.points:
            if frontier_point.dominates(point, self.objectives):
                return True
        return False

    def select_diverse_subset(
        self,
        frontier: ParetoFrontier,
        max_size: int,
    ) -> List[str]:
        """
        Select diverse subset from frontier.

        Uses greedy selection to maximize spread across objective space.

        Args:
            frontier: Frontier to select from
            max_size: Maximum number of points to select

        Returns:
            List of selected point IDs
        """
        if max_size <= 0 or not frontier.points:
            return []

        if len(frontier.points) <= max_size:
            return [p.point_id for p in frontier.points]

        selected: List[ParetoPoint] = []
        remaining = list(frontier.points)

        # Select first point: best on first objective
        if remaining:
            first_obj = self.objectives[0]
            best_point = max(remaining, key=lambda p: p.objectives.get(first_obj, float('-inf')))
            selected.append(best_point)
            remaining.remove(best_point)

        # Greedily select points that maximize minimum distance to selected
        while len(selected) < max_size and remaining:
            best_point = None
            best_min_distance = float('-inf')

            for candidate in remaining:
                # Compute minimum distance to selected points
                min_distance = float('inf')
                for selected_point in selected:
                    distance = self._compute_distance(candidate, selected_point)
                    min_distance = min(min_distance, distance)

                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_point = candidate

            if best_point:
                selected.append(best_point)
                remaining.remove(best_point)
            else:
                break

        return [p.point_id for p in selected]

    def _compute_distance(self, p1: ParetoPoint, p2: ParetoPoint) -> float:
        """
        Compute normalized Euclidean distance between points.

        Args:
            p1: First point
            p2: Second point

        Returns:
            Normalized distance in objective space
        """
        sum_sq_diff = 0.0
        count = 0

        for obj_name in self.objectives:
            v1 = p1.objectives.get(obj_name, 0.0)
            v2 = p2.objectives.get(obj_name, 0.0)
            sum_sq_diff += (v1 - v2) ** 2
            count += 1

        if count == 0:
            return 0.0

        return (sum_sq_diff / count) ** 0.5


def compute_pareto_frontier(
    points: List[Dict[str, Any]],
    objectives: List[str],
    point_id_key: str = "point_id",
    objective_key: str = "objectives",
) -> ParetoFrontier:
    """
    Convenience function to compute Pareto frontier.

    Args:
        points: List of point dictionaries
        objectives: List of objective names
        point_id_key: Key for point ID
        objective_key: Key for objectives dict

    Returns:
        ParetoFrontier with non-dominated points
    """
    optimizer = ParetoOptimizer(objectives=objectives)
    return optimizer.compute_frontier(
        points=points,
        point_id_key=point_id_key,
        objective_key=objective_key,
    )
