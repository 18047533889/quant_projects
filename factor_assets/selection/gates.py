"""
Evidence threshold gates for factor selection.

Gates evaluate evidence against thresholds without duplicating evaluation logic.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol
from datetime import datetime, timezone


class GateResult(Enum):
    """Result of gate evaluation."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass(frozen=True)
class GateEvaluation:
    """
    Result of evaluating a gate against evidence.

    Immutable record of gate application with full provenance.
    """
    gate_name: str
    factor_id: str
    result: GateResult
    timestamp: str  # ISO 8601
    evidence_id: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    message: Optional[str] = None
    gate_version: Optional[str] = None

    def __post_init__(self):
        if not self.gate_name:
            raise ValueError("gate_name is required")
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.timestamp:
            raise ValueError("timestamp is required")

    @property
    def passed(self) -> bool:
        """Check if gate passed."""
        return self.result == GateResult.PASS

    @property
    def failed(self) -> bool:
        """Check if gate failed."""
        return self.result == GateResult.FAIL


class EvidenceGate(Protocol):
    """
    Protocol for evidence-based admission gates.

    Gates evaluate evidence against thresholds without accessing raw values.
    """

    def evaluate(
        self,
        factor_id: str,
        evidence_id: str,
        metric_name: str,
        metric_value: float,
    ) -> GateEvaluation:
        """
        Evaluate gate against evidence.

        Args:
            factor_id: Factor identifier
            evidence_id: Evidence identifier
            metric_name: Metric name
            metric_value: Metric value to evaluate

        Returns:
            GateEvaluation result
        """
        ...

    @property
    def gate_name(self) -> str:
        """Get gate name."""
        ...

    @property
    def gate_version(self) -> str:
        """Get gate version."""
        ...


class ThresholdGate:
    """
    Simple threshold gate for numeric metrics.

    Passes if metric_value >= threshold (for higher-is-better metrics)
    or metric_value <= threshold (for lower-is-better metrics).
    """

    def __init__(
        self,
        gate_name: str,
        threshold: float,
        higher_is_better: bool = True,
        gate_version: str = "1.0",
    ):
        """
        Initialize threshold gate.

        Args:
            gate_name: Gate identifier
            threshold: Threshold value
            higher_is_better: If True, pass when value >= threshold;
                            if False, pass when value <= threshold
            gate_version: Gate version identifier
        """
        if not gate_name:
            raise ValueError("gate_name is required")

        self._gate_name = gate_name
        self._threshold = threshold
        self._higher_is_better = higher_is_better
        self._gate_version = gate_version

    @property
    def gate_name(self) -> str:
        """Get gate name."""
        return self._gate_name

    @property
    def gate_version(self) -> str:
        """Get gate version."""
        return self._gate_version

    @property
    def threshold(self) -> float:
        """Get threshold value."""
        return self._threshold

    def evaluate(
        self,
        factor_id: str,
        evidence_id: str,
        metric_name: str,
        metric_value: float,
    ) -> GateEvaluation:
        """Evaluate threshold gate."""
        now = datetime.now(timezone.utc).isoformat()

        if self._higher_is_better:
            passed = metric_value >= self._threshold
            message = (
                f"{metric_name}={metric_value:.4f} >= {self._threshold:.4f}"
                if passed
                else f"{metric_name}={metric_value:.4f} < {self._threshold:.4f}"
            )
        else:
            passed = metric_value <= self._threshold
            message = (
                f"{metric_name}={metric_value:.4f} <= {self._threshold:.4f}"
                if passed
                else f"{metric_name}={metric_value:.4f} > {self._threshold:.4f}"
            )

        return GateEvaluation(
            gate_name=self._gate_name,
            factor_id=factor_id,
            result=GateResult.PASS if passed else GateResult.FAIL,
            timestamp=now,
            evidence_id=evidence_id,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=self._threshold,
            message=message,
            gate_version=self._gate_version,
        )


class CompositeGate:
    """
    Composite gate combining multiple sub-gates.

    Supports AND (all must pass) and OR (any must pass) logic.
    """

    def __init__(
        self,
        gate_name: str,
        gates: list[EvidenceGate],
        require_all: bool = True,
        gate_version: str = "1.0",
    ):
        """
        Initialize composite gate.

        Args:
            gate_name: Composite gate identifier
            gates: List of sub-gates to evaluate
            require_all: If True, all gates must pass (AND);
                        if False, any gate can pass (OR)
            gate_version: Gate version identifier
        """
        if not gate_name:
            raise ValueError("gate_name is required")
        if not gates:
            raise ValueError("gates list cannot be empty")

        self._gate_name = gate_name
        self._gates = gates
        self._require_all = require_all
        self._gate_version = gate_version

    @property
    def gate_name(self) -> str:
        """Get gate name."""
        return self._gate_name

    @property
    def gate_version(self) -> str:
        """Get gate version."""
        return self._gate_version

    def evaluate_all(
        self,
        factor_id: str,
        evidence_id: str,
        metrics: dict[str, float],
    ) -> tuple[GateEvaluation, list[GateEvaluation]]:
        """
        Evaluate all sub-gates and compute composite result.

        Args:
            factor_id: Factor identifier
            evidence_id: Evidence identifier
            metrics: Dict of metric_name -> metric_value

        Returns:
            Tuple of (composite_evaluation, sub_evaluations)
        """
        sub_evaluations = []

        for gate in self._gates:
            # Try to find matching metric for this gate
            matched = False
            gate_name_lower = gate.gate_name.lower()

            # Strategy 1: Look for metric name contained in gate name
            # e.g., "minimum_ic" contains "ic", "maximum_turnover" contains "turnover"
            for metric_name, metric_value in metrics.items():
                metric_lower = metric_name.lower()
                if metric_lower in gate_name_lower or gate_name_lower in metric_lower:
                    try:
                        eval_result = gate.evaluate(
                            factor_id=factor_id,
                            evidence_id=evidence_id,
                            metric_name=metric_name,
                            metric_value=metric_value,
                        )
                        sub_evaluations.append(eval_result)
                        matched = True
                        break
                    except Exception:
                        continue

            # Strategy 2: If no match found, try reverse (gate name prefix matches metric)
            if not matched:
                for metric_name, metric_value in metrics.items():
                    if gate_name_lower.startswith(metric_name.lower()):
                        try:
                            eval_result = gate.evaluate(
                                factor_id=factor_id,
                                evidence_id=evidence_id,
                                metric_name=metric_name,
                                metric_value=metric_value,
                            )
                            sub_evaluations.append(eval_result)
                            matched = True
                            break
                        except Exception:
                            continue

            # Strategy 3: Fallback - try first available metric
            if not matched:
                for metric_name, metric_value in metrics.items():
                    try:
                        eval_result = gate.evaluate(
                            factor_id=factor_id,
                            evidence_id=evidence_id,
                            metric_name=metric_name,
                            metric_value=metric_value,
                        )
                        sub_evaluations.append(eval_result)
                        break  # Only evaluate each gate once
                    except Exception:
                        continue

        now = datetime.now(timezone.utc).isoformat()

        # Compute composite result
        if not sub_evaluations:
            return (
                GateEvaluation(
                    gate_name=self._gate_name,
                    factor_id=factor_id,
                    result=GateResult.ERROR,
                    timestamp=now,
                    evidence_id=evidence_id,
                    message="No sub-gates evaluated",
                    gate_version=self._gate_version,
                ),
                sub_evaluations,
            )

        if self._require_all:
            # AND logic: all must pass
            all_passed = all(ev.passed for ev in sub_evaluations)
            result = GateResult.PASS if all_passed else GateResult.FAIL
            passed_count = sum(1 for ev in sub_evaluations if ev.passed)
            message = f"{passed_count}/{len(sub_evaluations)} gates passed (require all)"
        else:
            # OR logic: any can pass
            any_passed = any(ev.passed for ev in sub_evaluations)
            result = GateResult.PASS if any_passed else GateResult.FAIL
            passed_count = sum(1 for ev in sub_evaluations if ev.passed)
            message = f"{passed_count}/{len(sub_evaluations)} gates passed (require any)"

        composite = GateEvaluation(
            gate_name=self._gate_name,
            factor_id=factor_id,
            result=result,
            timestamp=now,
            evidence_id=evidence_id,
            message=message,
            gate_version=self._gate_version,
        )

        return composite, sub_evaluations


class MinimumICGate(ThresholdGate):
    """
    Gate for minimum information coefficient threshold.

    Passes if absolute IC >= threshold (since IC can be negative but magnitude matters).
    """

    def __init__(
        self,
        threshold: float = 0.02,
        gate_version: str = "1.0",
    ):
        """
        Initialize minimum IC gate.

        Args:
            threshold: Minimum absolute IC value
            gate_version: Gate version identifier
        """
        if threshold < 0:
            raise ValueError("IC threshold must be non-negative")

        super().__init__(
            gate_name="minimum_ic",
            threshold=threshold,
            higher_is_better=True,
            gate_version=gate_version,
        )

    def evaluate(
        self,
        factor_id: str,
        evidence_id: str,
        metric_name: str,
        metric_value: float,
    ) -> GateEvaluation:
        """Evaluate IC gate using absolute value."""
        abs_value = abs(metric_value)
        now = datetime.now(timezone.utc).isoformat()

        passed = abs_value >= self._threshold
        message = (
            f"|{metric_name}|={abs_value:.4f} >= {self._threshold:.4f} (raw={metric_value:.4f})"
            if passed
            else f"|{metric_name}|={abs_value:.4f} < {self._threshold:.4f} (raw={metric_value:.4f})"
        )

        return GateEvaluation(
            gate_name=self._gate_name,
            factor_id=factor_id,
            result=GateResult.PASS if passed else GateResult.FAIL,
            timestamp=now,
            evidence_id=evidence_id,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=self._threshold,
            message=message,
            gate_version=self._gate_version,
        )


class MaximumTurnoverGate(ThresholdGate):
    """
    Gate for maximum turnover threshold.

    Passes if turnover <= threshold (lower is better for turnover).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        gate_version: str = "1.0",
    ):
        """
        Initialize maximum turnover gate.

        Args:
            threshold: Maximum acceptable turnover
            gate_version: Gate version identifier
        """
        if threshold < 0:
            raise ValueError("Turnover threshold must be non-negative")

        super().__init__(
            gate_name="maximum_turnover",
            threshold=threshold,
            higher_is_better=False,
            gate_version=gate_version,
        )


class MinimumCoverageGate(ThresholdGate):
    """
    Gate for minimum coverage threshold.

    Passes if coverage >= threshold (coverage should be in [0, 1]).
    """

    def __init__(
        self,
        threshold: float = 0.8,
        gate_version: str = "1.0",
    ):
        """
        Initialize minimum coverage gate.

        Args:
            threshold: Minimum coverage fraction (0 to 1)
            gate_version: Gate version identifier
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Coverage threshold must be in [0, 1]")

        super().__init__(
            gate_name="minimum_coverage",
            threshold=threshold,
            higher_is_better=True,
            gate_version=gate_version,
        )


class MinimumObservationsGate(ThresholdGate):
    """
    Gate for minimum number of observations.

    Passes if observation count >= threshold.
    """

    def __init__(
        self,
        threshold: int = 100,
        gate_version: str = "1.0",
    ):
        """
        Initialize minimum observations gate.

        Args:
            threshold: Minimum number of observations
            gate_version: Gate version identifier
        """
        if threshold < 1:
            raise ValueError("Observations threshold must be at least 1")

        super().__init__(
            gate_name="minimum_observations",
            threshold=float(threshold),
            higher_is_better=True,
            gate_version=gate_version,
        )

    def evaluate(
        self,
        factor_id: str,
        evidence_id: str,
        metric_name: str,
        metric_value: float,
    ) -> GateEvaluation:
        """Evaluate observations gate with integer formatting."""
        now = datetime.now(timezone.utc).isoformat()

        passed = metric_value >= self._threshold
        message = (
            f"{metric_name}={int(metric_value)} >= {int(self._threshold)}"
            if passed
            else f"{metric_name}={int(metric_value)} < {int(self._threshold)}"
        )

        return GateEvaluation(
            gate_name=self._gate_name,
            factor_id=factor_id,
            result=GateResult.PASS if passed else GateResult.FAIL,
            timestamp=now,
            evidence_id=evidence_id,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=self._threshold,
            message=message,
            gate_version=self._gate_version,
        )


class ParetoDominanceGate:
    """
    Gate checking if a factor is Pareto dominated.

    A factor is Pareto dominated if another factor exists that is:
    - Better or equal on all objectives
    - Strictly better on at least one objective

    Passes if the factor is NOT Pareto dominated.
    """

    def __init__(
        self,
        gate_name: str = "pareto_dominance",
        higher_is_better_metrics: Optional[list[str]] = None,
        lower_is_better_metrics: Optional[list[str]] = None,
        gate_version: str = "1.0",
    ):
        """
        Initialize Pareto dominance gate.

        Args:
            gate_name: Gate identifier
            higher_is_better_metrics: List of metric names where higher is better
            lower_is_better_metrics: List of metric names where lower is better
            gate_version: Gate version identifier
        """
        self._gate_name = gate_name
        self._higher_is_better = set(higher_is_better_metrics or [])
        self._lower_is_better = set(lower_is_better_metrics or [])
        self._gate_version = gate_version

        # Validate no overlap
        overlap = self._higher_is_better & self._lower_is_better
        if overlap:
            raise ValueError(f"Metrics cannot be both higher and lower is better: {overlap}")

    @property
    def gate_name(self) -> str:
        """Get gate name."""
        return self._gate_name

    @property
    def gate_version(self) -> str:
        """Get gate version."""
        return self._gate_version

    def is_dominated(
        self,
        candidate_metrics: dict[str, float],
        reference_metrics: dict[str, float],
    ) -> bool:
        """
        Check if candidate is dominated by reference.

        Args:
            candidate_metrics: Metrics of the candidate factor
            reference_metrics: Metrics of the reference factor

        Returns:
            True if candidate is dominated by reference
        """
        # Get common metrics that have known directions
        common_metrics = set(candidate_metrics.keys()) & set(reference_metrics.keys())
        known_metrics = common_metrics & (self._higher_is_better | self._lower_is_better)

        if not known_metrics:
            return False

        better_or_equal_count = 0
        strictly_better_count = 0

        for metric in known_metrics:
            candidate_val = candidate_metrics[metric]
            reference_val = reference_metrics[metric]

            # Determine if reference is better
            if metric in self._higher_is_better:
                if reference_val > candidate_val:
                    strictly_better_count += 1
                    better_or_equal_count += 1
                elif reference_val == candidate_val:
                    better_or_equal_count += 1
                else:
                    # Reference is worse, not dominated
                    return False
            elif metric in self._lower_is_better:
                if reference_val < candidate_val:
                    strictly_better_count += 1
                    better_or_equal_count += 1
                elif reference_val == candidate_val:
                    better_or_equal_count += 1
                else:
                    # Reference is worse, not dominated
                    return False

        # Dominated if reference is better or equal on all known metrics,
        # and strictly better on at least one
        return better_or_equal_count == len(known_metrics) and strictly_better_count > 0

    def evaluate(
        self,
        factor_id: str,
        evidence_id: str,
        candidate_metrics: dict[str, float],
        reference_factors: list[tuple[str, dict[str, float]]],
    ) -> GateEvaluation:
        """
        Evaluate if factor is Pareto dominated.

        Args:
            factor_id: Candidate factor identifier
            evidence_id: Evidence identifier
            candidate_metrics: Metrics of the candidate factor
            reference_factors: List of (factor_id, metrics) tuples to compare against

        Returns:
            GateEvaluation result (PASS if not dominated, FAIL if dominated)
        """
        now = datetime.now(timezone.utc).isoformat()

        dominated_by = []
        for ref_id, ref_metrics in reference_factors:
            if self.is_dominated(candidate_metrics, ref_metrics):
                dominated_by.append(ref_id)

        if dominated_by:
            message = f"Factor dominated by {len(dominated_by)} other(s): {', '.join(dominated_by[:3])}"
            if len(dominated_by) > 3:
                message += f" (+{len(dominated_by) - 3} more)"

            return GateEvaluation(
                gate_name=self._gate_name,
                factor_id=factor_id,
                result=GateResult.FAIL,
                timestamp=now,
                evidence_id=evidence_id,
                message=message,
                gate_version=self._gate_version,
            )
        else:
            return GateEvaluation(
                gate_name=self._gate_name,
                factor_id=factor_id,
                result=GateResult.PASS,
                timestamp=now,
                evidence_id=evidence_id,
                message=f"Not Pareto dominated (checked against {len(reference_factors)} factors)",
                gate_version=self._gate_version,
            )
