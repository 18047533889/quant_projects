"""
Parameter plateau detection and neighbor survival analysis.

Detects stable parameter regions via perturbation and robustness scoring.

DEPRECATED (DLIB-FA-001): factor_optimizer (FO) owns search-stopping plateau
detection. FA does not run a search loop. This module is retained for
compatibility (§108) and marked RESEARCH_ONLY. It will be removed after the
deprecation window. No FA production path consumes it.
"""

import math
import warnings as _warnings

_warnings.warn(
    "factor_assets.optimizer.plateau is deprecated: FO owns search-stopping "
    "plateau detection. FA does not run a search loop. This module is "
    "retained for compatibility and will be removed.",
    DeprecationWarning,
    stacklevel=2,
)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParameterNeighbor:
    """
    Parameter neighbor from perturbation.

    Attributes:
        neighbor_id: Unique identifier
        base_parameters: Original parameters
        perturbed_parameters: Perturbed parameters
        perturbation_info: Information about perturbation
        metric_value: Metric value for this neighbor
        relative_performance: Performance relative to base
    """

    neighbor_id: str
    base_parameters: Dict[str, Any]
    perturbed_parameters: Dict[str, Any]
    perturbation_info: Dict[str, Any]
    metric_value: Optional[float] = None
    relative_performance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "neighbor_id": self.neighbor_id,
            "base_parameters": dict(self.base_parameters),
            "perturbed_parameters": dict(self.perturbed_parameters),
            "perturbation_info": dict(self.perturbation_info),
            "metric_value": self.metric_value,
            "relative_performance": self.relative_performance,
        }


@dataclass
class PlateauAnalysis:
    """
    Analysis of parameter plateau.

    Attributes:
        base_metric: Metric value at base parameters
        worst_neighbor_metric: Worst neighbor metric value
        mean_neighbor_metric: Mean neighbor metric value
        local_sensitivity: Max relative degradation
        plateau_stability: Fraction of neighbors within tolerance
        neighbor_survival_rate: Fraction of neighbors above threshold
        neighbors: All evaluated neighbors
    """

    base_metric: float
    worst_neighbor_metric: float
    mean_neighbor_metric: float
    local_sensitivity: float
    plateau_stability: float
    neighbor_survival_rate: float
    neighbors: List[ParameterNeighbor] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "base_metric": self.base_metric,
            "worst_neighbor_metric": self.worst_neighbor_metric,
            "mean_neighbor_metric": self.mean_neighbor_metric,
            "local_sensitivity": self.local_sensitivity,
            "plateau_stability": self.plateau_stability,
            "neighbor_survival_rate": self.neighbor_survival_rate,
            "neighbor_count": len(self.neighbors),
        }


class ParameterPlateauDetector:
    """
    Detect parameter plateaus via neighbor perturbation.

    Generates neighbors and analyzes metric stability.
    """

    def __init__(
        self,
        perturbation_scale: float = 0.1,
        stability_tolerance: float = 0.05,
        survival_threshold: float = 0.8,
    ):
        """
        Initialize detector.

        Args:
            perturbation_scale: Scale of perturbations (relative to range)
            stability_tolerance: Tolerance for plateau stability
            survival_threshold: Threshold for neighbor survival (relative to base)
        """
        self.perturbation_scale = perturbation_scale
        self.stability_tolerance = stability_tolerance
        self.survival_threshold = survival_threshold

    def generate_neighbors(
        self,
        parameters: Dict[str, Any],
        max_neighbors: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter neighbors via perturbation.

        Args:
            parameters: Base parameters
            max_neighbors: Maximum number of neighbors

        Returns:
            List of perturbed parameter dictionaries
        """
        neighbors = []

        # Generate neighbors by perturbing each numeric parameter
        for param_name, param_value in parameters.items():
            if isinstance(param_value, bool) or not isinstance(param_value, (int, float)):
                continue

            # Estimate reasonable perturbation range
            if param_value == 0:
                perturbation = self.perturbation_scale
            else:
                perturbation = abs(param_value) * self.perturbation_scale

            # Generate neighbors in both directions
            for direction in [-1, 1]:
                if len(neighbors) >= max_neighbors:
                    break

                new_value = param_value + direction * perturbation

                # Create perturbed parameters
                new_params = dict(parameters)
                new_params[param_name] = new_value

                neighbors.append({
                    "parameters": new_params,
                    "perturbation_info": {
                        "param_name": param_name,
                        "direction": direction,
                        "magnitude": perturbation,
                        "original_value": param_value,
                        "new_value": new_value,
                    },
                })

            if len(neighbors) >= max_neighbors:
                break

        return neighbors[:max_neighbors]

    def analyze_plateau(
        self,
        base_parameters: Dict[str, Any],
        base_metric: float,
        neighbor_results: List[Tuple[Dict[str, Any], float]],
    ) -> PlateauAnalysis:
        """
        Analyze parameter plateau from neighbor evaluations.

        Args:
            base_parameters: Base parameters
            base_metric: Metric value at base parameters
            neighbor_results: List of (parameters, metric) tuples for neighbors

        Returns:
            PlateauAnalysis with stability metrics
        """
        if not neighbor_results:
            return PlateauAnalysis(
                base_metric=base_metric,
                worst_neighbor_metric=base_metric,
                mean_neighbor_metric=base_metric,
                local_sensitivity=0.0,
                plateau_stability=1.0,
                neighbor_survival_rate=1.0,
                neighbors=[],
            )

        # Create neighbor objects
        neighbors = []
        for i, (params, metric) in enumerate(neighbor_results):
            relative_perf = (metric - base_metric) / abs(base_metric) if base_metric != 0 else 0.0

            neighbor = ParameterNeighbor(
                neighbor_id=f"neighbor_{i}",
                base_parameters=base_parameters,
                perturbed_parameters=params,
                perturbation_info={},
                metric_value=metric,
                relative_performance=relative_perf,
            )
            neighbors.append(neighbor)

        # Compute statistics
        neighbor_metrics = [n.metric_value for n in neighbors if n.metric_value is not None]
        if not neighbor_metrics:
            neighbor_metrics = [base_metric]

        # Non-finite neighbor metrics (NaN/±inf from an overflowed or failed
        # evaluation) must not be silently aggregated: a NaN poisons
        # min/mean with order-dependent results, and an inf worst-metric
        # makes local_sensitivity -inf, whose max(0, 1-inf)=inf sensitivity
        # score then clamps the overall robustness to a perfect 1.0.
        # Fail closed: exclude them from the finite statistics but keep
        # every neighbor in the stability/survival denominators so a
        # broken region scores low.
        finite_metrics = [m for m in neighbor_metrics if math.isfinite(m)]
        n_total = len(neighbor_metrics)
        if not finite_metrics:
            # Every neighbor evaluation was non-finite: the region is
            # unverifiable, not maximally stable.
            return PlateauAnalysis(
                base_metric=base_metric,
                worst_neighbor_metric=float("nan"),
                mean_neighbor_metric=float("nan"),
                local_sensitivity=float("inf"),
                plateau_stability=0.0,
                neighbor_survival_rate=0.0,
                neighbors=neighbors,
            )

        worst_metric = min(finite_metrics)
        mean_metric = sum(finite_metrics) / len(finite_metrics)

        # Compute local sensitivity (max degradation)
        if base_metric != 0:
            local_sensitivity = (base_metric - worst_metric) / abs(base_metric)
        else:
            local_sensitivity = 0.0

        # Compute plateau stability (fraction within tolerance); NaN/inf
        # neighbors are never "within tolerance" (comparison is False).
        stable_count = sum(
            1 for m in neighbor_metrics
            if abs(m - base_metric) / abs(base_metric) <= self.stability_tolerance
        ) if base_metric != 0 else len(finite_metrics)
        plateau_stability = stable_count / n_total

        # Compute neighbor survival rate (fraction above threshold);
        # NaN/inf neighbors never count as survivors (comparison False —
        # +inf survives only via finite-metric gating above, but an inf
        # metric is an overflow, not a good neighbour).
        survival_count = sum(
            1 for m in neighbor_metrics
            if math.isfinite(m) and m >= base_metric * self.survival_threshold
        )
        neighbor_survival_rate = survival_count / n_total

        return PlateauAnalysis(
            base_metric=base_metric,
            worst_neighbor_metric=worst_metric,
            mean_neighbor_metric=mean_metric,
            local_sensitivity=local_sensitivity,
            plateau_stability=plateau_stability,
            neighbor_survival_rate=neighbor_survival_rate,
            neighbors=neighbors,
        )


class NeighborSurvivalAnalyzer:
    """
    Analyze neighbor survival and rank robustness.

    Provides robustness scoring based on neighbor performance.
    """

    def __init__(self, survival_threshold: float = 0.8):
        """
        Initialize analyzer.

        Args:
            survival_threshold: Threshold for neighbor survival
        """
        self.survival_threshold = survival_threshold

    def compute_robustness_score(
        self,
        plateau_analysis: PlateauAnalysis,
    ) -> float:
        """
        Compute robustness score from plateau analysis.

        Args:
            plateau_analysis: Plateau analysis results

        Returns:
            Robustness score in [0, 1].  A non-finite local_sensitivity
            (±inf, from an all-non-finite neighbor region, direct
            construction, or overflow) contributes a sensitivity score of
            0.0 — overflowed measurements are never admittable evidence
            of robustness, even when they point "upward" (−inf).
        """
        # Weight different factors
        stability_weight = 0.4
        survival_weight = 0.4
        sensitivity_weight = 0.2

        # Compute weighted score
        stability_score = plateau_analysis.plateau_stability
        survival_score = plateau_analysis.neighbor_survival_rate
        # A non-finite sensitivity (inf from an all-non-finite neighbor
        # region) means "unverifiable", which must score as maximally
        # UNrobust, not clamp up to 1.0.  This also intentionally zeroes
        # −inf (a directly constructed or overflowed analysis where
        # sensitivity improved without bound): an overflowed measurement
        # in EITHER direction is not admittable evidence of robustness.
        sensitivity = plateau_analysis.local_sensitivity
        if not math.isfinite(sensitivity):
            sensitivity_score = 0.0
        else:
            sensitivity_score = max(0.0, 1.0 - sensitivity)

        robustness = (
            stability_weight * stability_score +
            survival_weight * survival_score +
            sensitivity_weight * sensitivity_score
        )

        return max(0.0, min(1.0, robustness))

    def rank_by_robustness(
        self,
        candidates: List[Dict[str, Any]],
        plateau_key: str = "plateau_analysis",
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates by robustness.

        Args:
            candidates: List of candidate dictionaries
            plateau_key: Key for plateau analysis in candidate dict

        Returns:
            Candidates sorted by robustness (descending)
        """
        scored_candidates = []

        for candidate in candidates:
            plateau_analysis = candidate.get(plateau_key)
            if plateau_analysis:
                if isinstance(plateau_analysis, dict):
                    # Convert dict to PlateauAnalysis
                    plateau_analysis = PlateauAnalysis(**plateau_analysis)

                robustness = self.compute_robustness_score(plateau_analysis)
            else:
                robustness = 0.0

            scored_candidates.append({
                **candidate,
                "robustness_score": robustness,
            })

        scored_candidates.sort(key=lambda c: c["robustness_score"], reverse=True)
        return scored_candidates

    def filter_by_robustness(
        self,
        candidates: List[Dict[str, Any]],
        min_robustness: float,
        plateau_key: str = "plateau_analysis",
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates by minimum robustness.

        Args:
            candidates: List of candidate dictionaries
            min_robustness: Minimum robustness score
            plateau_key: Key for plateau analysis

        Returns:
            Filtered candidates
        """
        ranked = self.rank_by_robustness(candidates, plateau_key)
        return [c for c in ranked if c.get("robustness_score", 0.0) >= min_robustness]
