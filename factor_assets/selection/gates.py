"""
Evidence threshold gates for factor selection.

Gates evaluate evidence against thresholds without duplicating evaluation logic.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Protocol, Union
from datetime import datetime, timezone


class MetricDirection(Enum):
    """Comparison direction declared by a metric contract."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    ABSOLUTE_HIGHER_IS_BETTER = "absolute_higher_is_better"


@dataclass(frozen=True)
class MetricBinding:
    """Exact metric schema required by one gate."""

    metric_id: str
    unit: str
    direction: MetricDirection

    def __post_init__(self):
        if not self.metric_id:
            raise ValueError("metric_id is required")
        if not self.unit:
            raise ValueError("metric unit is required")
        if not isinstance(self.direction, MetricDirection):
            raise TypeError("metric direction must be a MetricDirection")


@dataclass(frozen=True)
class MetricEvidence:
    """Typed metric value supplied by an evidence producer."""

    value: float
    unit: str
    direction: MetricDirection

    def __post_init__(self):
        if not self.unit:
            raise ValueError("metric evidence unit is required")
        if not isinstance(self.direction, MetricDirection):
            raise TypeError("metric evidence direction must be a MetricDirection")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("metric evidence value must be a non-boolean number")
        if not math.isfinite(self.value):
            raise ValueError(
                "metric evidence value must be finite (NaN/±inf is an "
                "overflowed or failed measurement, not an admittable score)"
            )


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
    def metric_direction(self) -> MetricDirection:
        """Return the metric direction enforced by this gate."""
        return (
            MetricDirection.HIGHER_IS_BETTER
            if self._higher_is_better
            else MetricDirection.LOWER_IS_BETTER
        )

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
        metric_bindings: Optional[Mapping[str, MetricBinding]] = None,
    ):
        """
        Initialize composite gate.

        Args:
            gate_name: Composite gate identifier
            gates: List of sub-gates to evaluate
            require_all: If True, all gates must pass (AND);
                        if False, any gate can pass (OR)
            gate_version: Gate version identifier
            metric_bindings: Exact gate_name -> typed metric schema mapping.
                           Missing, duplicate, or invalid bindings fail closed.
        """
        if not gate_name:
            raise ValueError("gate_name is required")
        if not gates:
            raise ValueError("gates list cannot be empty")

        self._gate_name = gate_name
        self._gates = gates
        self._require_all = require_all
        self._gate_version = gate_version
        self._metric_bindings = dict(metric_bindings or {})

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
        metrics: Mapping[str, MetricEvidence],
    ) -> tuple[GateEvaluation, list[GateEvaluation]]:
        """
        Evaluate all sub-gates and compute composite result.

        Args:
            factor_id: Factor identifier
            evidence_id: Evidence identifier
            metrics: Exact metric_id -> typed metric evidence mapping

        Returns:
            Tuple of (composite_evaluation, sub_evaluations)
        """
        sub_evaluations = []
        binding_errors = False

        def fail_closed(gate: EvidenceGate, message: str, metric_name: Optional[str] = None) -> None:
            nonlocal binding_errors
            binding_errors = True
            sub_evaluations.append(
                GateEvaluation(
                    gate_name=gate.gate_name,
                    factor_id=factor_id,
                    result=GateResult.FAIL,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    evidence_id=evidence_id or None,
                    metric_name=metric_name,
                    message=message,
                    gate_version=gate.gate_version,
                )
            )

        gate_names = [gate.gate_name for gate in self._gates]
        duplicate_names = {name for name in gate_names if gate_names.count(name) > 1}

        for gate in self._gates:
            if not evidence_id:
                fail_closed(gate, "Missing evidence_id")
                continue
            if gate.gate_name in duplicate_names:
                fail_closed(gate, f"Ambiguous duplicate gate name: {gate.gate_name}")
                continue

            binding = self._metric_bindings.get(gate.gate_name)
            if binding is None:
                fail_closed(gate, f"Missing metric binding for gate: {gate.gate_name}")
                continue
            if not isinstance(binding, MetricBinding):
                fail_closed(gate, f"Invalid untyped metric binding for gate: {gate.gate_name}")
                continue

            evidence = metrics.get(binding.metric_id)
            if evidence is None:
                fail_closed(
                    gate,
                    f"Missing evidence for bound metric: {binding.metric_id}",
                    binding.metric_id,
                )
                continue
            if not isinstance(evidence, MetricEvidence):
                fail_closed(
                    gate,
                    f"Invalid untyped evidence for bound metric: {binding.metric_id}",
                    binding.metric_id,
                )
                continue
            if evidence.unit != binding.unit:
                fail_closed(
                    gate,
                    f"Metric unit mismatch for {binding.metric_id}: expected {binding.unit}, got {evidence.unit}",
                    binding.metric_id,
                )
                continue
            if evidence.direction is not binding.direction:
                fail_closed(
                    gate,
                    f"Metric direction mismatch for {binding.metric_id}: expected {binding.direction.value}, got {evidence.direction.value}",
                    binding.metric_id,
                )
                continue

            expected_direction = getattr(gate, "metric_direction", None)
            if expected_direction is None or expected_direction is not binding.direction:
                expected = expected_direction.value if isinstance(expected_direction, MetricDirection) else "declared gate direction"
                fail_closed(
                    gate,
                    f"Binding direction incompatible with gate {gate.gate_name}: expected {expected}, got {binding.direction.value}",
                    binding.metric_id,
                )
                continue

            try:
                sub_evaluations.append(
                    gate.evaluate(
                        factor_id=factor_id,
                        evidence_id=evidence_id,
                        metric_name=binding.metric_id,
                        metric_value=evidence.value,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — fail closed on any
                # sub-gate crash (a gate that raises is a failed
                # evaluation, never a passing one; letting an unexpected
                # exception type propagate would abort the whole
                # composite run instead of recording one bad gate).
                fail_closed(
                    gate,
                    f"Gate evaluation failed for {binding.metric_id}: {type(exc).__name__}",
                    binding.metric_id,
                )

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

        if binding_errors:
            result = GateResult.FAIL
            passed_count = sum(1 for ev in sub_evaluations if ev.passed)
            message = f"{passed_count}/{len(sub_evaluations)} gates passed; metric evidence validation failed"
        elif self._require_all:
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
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("IC threshold must be a finite non-boolean number")
        if threshold < 0:
            raise ValueError("IC threshold must be non-negative")

        super().__init__(
            gate_name="minimum_ic",
            threshold=threshold,
            higher_is_better=True,
            gate_version=gate_version,
        )

    @property
    def metric_direction(self) -> MetricDirection:
        """IC magnitude is the admitted quantity."""
        return MetricDirection.ABSOLUTE_HIGHER_IS_BETTER

    def evaluate(
        self,
        factor_id: str,
        evidence_id: str,
        metric_name: str,
        metric_value: float,
    ) -> GateEvaluation:
        """Evaluate IC gate using absolute value."""
        now = datetime.now(timezone.utc).isoformat()

        if isinstance(metric_value, bool) or not math.isfinite(metric_value):
            # NaN/±inf IC is an overflowed or failed measurement, not
            # evidence of skill: fail closed instead of |inf| >= threshold.
            return GateEvaluation(
                gate_name=self._gate_name,
                factor_id=factor_id,
                result=GateResult.FAIL,
                timestamp=now,
                evidence_id=evidence_id,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=self._threshold,
                message=(
                    f"|{metric_name}| non-finite ({metric_value!r}) — "
                    "fail closed"
                ),
                gate_version=self._gate_version,
            )

        abs_value = abs(metric_value)
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
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("Turnover threshold must be a finite non-boolean number")
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
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("Coverage threshold must be a finite non-boolean number")
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

            # A non-numeric value on either side would raise TypeError in
            # math.isfinite / comparisons; guard so a standalone
            # evaluate() call cannot crash on broken input data
            # (undefined comparisons prove nothing → "not dominated").
            if (
                isinstance(candidate_val, bool) or not isinstance(candidate_val, (int, float))
                or isinstance(reference_val, bool) or not isinstance(reference_val, (int, float))
            ):
                return False
            # A non-finite side makes the comparison undefined — fail
            # closed by returning "not dominated" and letting evaluate()
            # record the contamination.  The behavior change vs the old
            # code is ±inf, not NaN (NaN comparisons were already all
            # False, which hit the "reference is worse" return False):
            # a −inf candidate against a clearly dominating reference
            # used to count as dominated (True); an undefined metric
            # proves nothing, so it no longer does.
            if not math.isfinite(candidate_val) or not math.isfinite(reference_val):
                return False

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

        # Any non-finite TRACKED metric (in higher/lower_is_better) makes the
        # Pareto comparison undefined: fail closed (FAIL) rather than letting
        # NaN comparison semantics silently pass an unmeasurable factor.
        # Untracked metrics are never compared, so they are ignored here
        # (they cannot contaminate the dominance check).
        tracked = self._higher_is_better | self._lower_is_better
        nonfinite = []
        for metric in tracked & set(candidate_metrics.keys()):
            value = candidate_metrics[metric]
            # bool is an int subclass but not a valid metric value; a
            # non-numeric value would raise TypeError in the comparisons
            # later — treat both as undefined and fail closed.  Note:
            # numpy float64 passes (float subclass); numpy float32/int64
            # do NOT (cast at the boundary, not here).
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                nonfinite.append(metric)
        if nonfinite:
            return GateEvaluation(
                gate_name=self._gate_name,
                factor_id=factor_id,
                result=GateResult.FAIL,
                timestamp=now,
                evidence_id=evidence_id,
                message=(
                    f"Candidate has non-finite/non-numeric tracked metrics, "
                    f"Pareto comparison undefined: {', '.join(sorted(nonfinite)[:5])} — fail closed"
                ),
                gate_version=self._gate_version,
            )

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
