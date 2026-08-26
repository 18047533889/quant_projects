"""Pareto frontier tracking for multi-objective optimization."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ParetoPoint:
    """
    A point in the Pareto frontier.

    Attributes:
        trial_id: Reference to the trial
        objectives: Objective values (higher is better for all)
        metadata: Additional point metadata
    """
    trial_id: str
    objectives: Tuple[float, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.objectives:
            raise ValueError("objectives cannot be empty")
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in self.objectives
        ):
            # bool is an int subclass — True would silently rank as objective 1.0
            raise ValueError("all objectives must be numeric non-boolean numbers")
        if not all(math.isfinite(float(v)) for v in self.objectives):
            raise ValueError("all objectives must be finite")

    def dominates(self, other: "ParetoPoint") -> bool:
        """
        Check if this point dominates another.

        A point dominates another if it's >= on all objectives and strictly > on at least one.
        """
        if len(self.objectives) != len(other.objectives):
            raise ValueError("cannot compare points with different dimensions")

        better_or_equal = all(
            s >= o for s, o in zip(self.objectives, other.objectives)
        )
        strictly_better = any(
            s > o for s, o in zip(self.objectives, other.objectives)
        )

        return better_or_equal and strictly_better

    def distance_to(self, other: "ParetoPoint") -> float:
        """Euclidean distance to another point."""
        if len(self.objectives) != len(other.objectives):
            raise ValueError("cannot compute distance for different dimensions")

        return sum(
            (s - o) ** 2 for s, o in zip(self.objectives, other.objectives)
        ) ** 0.5


class ParetoFrontier:
    """
    Tracks the Pareto frontier for multi-objective optimization.

    The frontier contains non-dominated points: no point on the frontier
    is strictly worse than another point on all objectives.
    """

    def __init__(self, objective_names: Optional[List[str]] = None):
        """
        Initialize Pareto frontier.

        Args:
            objective_names: Names of objectives (for reporting)
        """
        self.points: List[ParetoPoint] = []
        self.objective_names = objective_names
        self._dominated_cache: Set[str] = set()

    @property
    def num_dimensions(self) -> int:
        """Number of objectives."""
        if not self.points:
            return 0
        return len(self.points[0].objectives)

    @property
    def size(self) -> int:
        """Number of points on the frontier."""
        return len(self.points)

    def add(self, point: ParetoPoint) -> bool:
        """
        Add a point to the frontier if non-dominated.

        Alias for :meth:`add_point` that returns whether the new point
        survives (is not dominated).  Provided as a concise, discoverable
        name for the winner-selector flow.

        Returns:
            True if point was added (non-dominated), False if dominated
        """
        return self.add_point(point)

    def dominated(self, points) -> Set[str]:
        """
        Return the set of trial_ids among ``points`` that are dominated by
        the frontier.

        Args:
            points: An iterable of :class:`ParetoPoint` (or a set of them).

        Returns:
            Set of trial_ids that are dominated by at least one frontier point.
        """
        dominated_ids: Set[str] = set()
        for point in points:
            if self.is_dominated(point):
                dominated_ids.add(point.trial_id)
        return dominated_ids

    def frontier(self) -> List[ParetoPoint]:
        """
        Return the set of non-dominated points currently on the frontier.

        Returns:
            A list of the non-dominated :class:`ParetoPoint` objects.
        """
        return list(self.points)

    def add_point(self, point: ParetoPoint) -> bool:
        """
        Add a point to the frontier if non-dominated.

        Returns:
            True if point was added (non-dominated), False if dominated
        """
        # Check if new point is dominated by existing frontier
        for existing in self.points:
            if existing.dominates(point):
                self._dominated_cache.add(point.trial_id)
                return False

        # Remove points dominated by the new point
        self.points = [
            p for p in self.points
            if not point.dominates(p)
        ]

        # A trial ID may be retried with improved objectives.  Do not let a
        # prior dominated result poison the replacement's domination query.
        self._dominated_cache.discard(point.trial_id)

        # Add new point
        self.points.append(point)
        return True

    def is_dominated(self, point: ParetoPoint) -> bool:
        """Check if a point is dominated by the frontier."""
        if point.trial_id in self._dominated_cache:
            return True

        for existing in self.points:
            if existing.dominates(point):
                self._dominated_cache.add(point.trial_id)
                return True

        return False

    def get_point(self, trial_id: str) -> Optional[ParetoPoint]:
        """Retrieve a point by trial ID."""
        for point in self.points:
            if point.trial_id == trial_id:
                return point
        return None

    def extremes(self) -> Dict[int, ParetoPoint]:
        """
        Get extreme points for each objective.

        Returns:
            Dictionary mapping objective index to point with max value
        """
        if not self.points:
            return {}

        extremes = {}
        for i in range(self.num_dimensions):
            # Tie-break on trial_id: max() returns the FIRST maximal
            # element, and self.points order follows insertion (trial
            # completion) order — equal-objective extremes would flip with
            # a differently-ordered but equivalent frontier.
            best_point = max(
                self.points, key=lambda p: (p.objectives[i], p.trial_id)
            )
            extremes[i] = best_point

        return extremes

    def hypervolume(self, reference_point: Tuple[float, ...]) -> float:
        """
        Compute hypervolume indicator (approximation for 2D).

        For 2D objectives only. Higher is better.

        Args:
            reference_point: Reference point (typically worst case)

        Returns:
            Hypervolume covered by the frontier
        """
        if not self.points:
            # Empty frontier needs dimension check from reference
            if len(reference_point) != 2:
                raise NotImplementedError("hypervolume only implemented for 2D")
            return 0.0

        if self.num_dimensions != 2:
            raise NotImplementedError("hypervolume only implemented for 2D")

        # Sort points by first objective (descending)
        sorted_points = sorted(
            self.points,
            key=lambda p: p.objectives[0],
            reverse=True
        )

        volume = 0.0
        prev_y = reference_point[1]

        for point in sorted_points:
            x, y = point.objectives
            if x <= reference_point[0] or y <= reference_point[1]:
                continue

            width = x - reference_point[0]
            # Defensive only: a frontier maintained through add_point
            # cannot yield y < prev_y (such a point would be dominated
            # and removed), but a directly-constructed or duplicated
            # point list could — clamp so a degenerate slice adds zero,
            # never a negative volume that silently shrinks the
            # indicator.
            height = max(0.0, y - prev_y)
            volume += width * height
            if y > prev_y:
                prev_y = y

        return volume

    def coverage(self, other: "ParetoFrontier") -> float:
        """
        Compute coverage metric: fraction of other's points dominated by this frontier.

        Args:
            other: Another Pareto frontier

        Returns:
            Fraction in [0, 1]
        """
        if not other.points:
            return 1.0

        dominated_count = sum(
            1 for other_point in other.points
            if any(self_point.dominates(other_point) for self_point in self.points)
        )

        return dominated_count / len(other.points)

    def spacing(self) -> float:
        """
        Compute spacing metric: uniformity of point distribution.

        Lower is better (more uniform).
        """
        if len(self.points) < 2:
            return 0.0

        # Compute all pairwise distances
        distances = []
        for i, p1 in enumerate(self.points):
            min_dist = float('inf')
            for j, p2 in enumerate(self.points):
                if i != j:
                    dist = p1.distance_to(p2)
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)

        # Compute variance of distances
        mean_dist = sum(distances) / len(distances)
        variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)

        return variance ** 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize frontier to dictionary."""
        return {
            "points": [
                {
                    "trial_id": p.trial_id,
                    "objectives": list(p.objectives),
                    "metadata": p.metadata,
                }
                for p in self.points
            ],
            "objective_names": self.objective_names,
            "size": self.size,
            "num_dimensions": self.num_dimensions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParetoFrontier":
        """Deserialize frontier from dictionary."""
        frontier = cls(objective_names=data.get("objective_names"))
        for point_data in data.get("points", []):
            point = ParetoPoint(
                trial_id=point_data["trial_id"],
                objectives=tuple(point_data["objectives"]),
                metadata=point_data.get("metadata", {}),
            )
            frontier.add_point(point)
        return frontier


class ParetoArchive:
    """
    Archive tracking multiple Pareto frontiers over time.

    Useful for tracking frontier evolution during search.
    """

    def __init__(self, objective_names: Optional[List[str]] = None):
        """
        Initialize archive.

        Args:
            objective_names: Names of objectives
        """
        self.objective_names = objective_names
        self.frontiers: List[Tuple[int, ParetoFrontier]] = []
        self.current_frontier = ParetoFrontier(objective_names)

    def snapshot(self, iteration: int) -> None:
        """
        Take a snapshot of the current frontier.

        Args:
            iteration: Search iteration number
        """
        # Create a copy of current frontier
        snapshot = ParetoFrontier(self.objective_names)
        for point in self.current_frontier.points:
            snapshot.add_point(point)

        self.frontiers.append((iteration, snapshot))

    def add_point(self, point: ParetoPoint) -> bool:
        """Add a point to the current frontier."""
        return self.current_frontier.add_point(point)

    def get_frontier(self, iteration: Optional[int] = None) -> ParetoFrontier:
        """
        Get frontier at a specific iteration.

        Args:
            iteration: Iteration number (None for current)

        Returns:
            Pareto frontier
        """
        if iteration is None:
            return self.current_frontier

        for iter_num, frontier in self.frontiers:
            if iter_num == iteration:
                return frontier

        raise ValueError(f"no frontier snapshot at iteration {iteration}")

    def frontier_growth(self) -> List[Tuple[int, int]]:
        """
        Track frontier size over iterations.

        Returns:
            List of (iteration, size) tuples
        """
        growth = [(iter_num, frontier.size) for iter_num, frontier in self.frontiers]
        growth.append((len(self.frontiers), self.current_frontier.size))
        return growth

    def hypervolume_progress(
        self, reference_point: Tuple[float, ...]
    ) -> List[Tuple[int, float]]:
        """
        Track hypervolume over iterations.

        Args:
            reference_point: Reference point for hypervolume

        Returns:
            List of (iteration, hypervolume) tuples
        """
        if self.current_frontier.num_dimensions != 2:
            raise NotImplementedError("hypervolume only for 2D")

        progress = []
        for iter_num, frontier in self.frontiers:
            hv = frontier.hypervolume(reference_point)
            progress.append((iter_num, hv))

        hv = self.current_frontier.hypervolume(reference_point)
        progress.append((len(self.frontiers), hv))

        return progress
