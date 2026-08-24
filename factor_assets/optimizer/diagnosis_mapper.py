"""
Diagnosis-driven repair: maps QE diagnostic signals to FO mutation recommendations.

Protocol-based bridge to factor_optimizer repair engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DiagnosisKind(Enum):
    """Type of diagnostic signal."""

    POOR_RANK_IC = "poor_rank_ic"
    HIGH_TURNOVER = "high_turnover"
    PARAMETER_INSTABILITY = "parameter_instability"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    REGIME_SENSITIVITY = "regime_sensitivity"
    COMPUTATION_TIMEOUT = "computation_timeout"
    NUMERICAL_INSTABILITY = "numerical_instability"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    UNKNOWN = "unknown"


@dataclass
class DiagnosisEvidence:
    """
    Evidence supporting a diagnosis.

    Attributes:
        kind: Type of diagnosis
        severity: Severity score [0,1]
        metrics: Supporting metric values
        context: Additional context (time range, split, etc.)
        confidence: Confidence in diagnosis [0,1]
        source: Source of diagnosis (QE module)
    """

    kind: DiagnosisKind
    severity: float
    metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "unknown"

    def __post_init__(self):
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(f"severity must be in [0,1], got {self.severity}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "kind": self.kind.value,
            "severity": self.severity,
            "metrics": dict(self.metrics),
            "context": dict(self.context),
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class RepairCandidate:
    """
    Mutation candidate for repair.

    Attributes:
        mutation_type: Type of mutation from FO grammar
        parameters: Mutation parameters
        priority: Priority score [0,1]
        rationale: Human-readable rationale
        expected_improvement: Expected metric improvement
        risk_score: Risk of introducing new issues [0,1]
    """

    mutation_type: str
    parameters: Dict[str, Any]
    priority: float
    rationale: str
    expected_improvement: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.5

    def __post_init__(self):
        if not (0.0 <= self.priority <= 1.0):
            raise ValueError(f"priority must be in [0,1], got {self.priority}")
        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError(f"risk_score must be in [0,1], got {self.risk_score}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mutation_type": self.mutation_type,
            "parameters": dict(self.parameters),
            "priority": self.priority,
            "rationale": self.rationale,
            "expected_improvement": dict(self.expected_improvement),
            "risk_score": self.risk_score,
        }


class RepairStrategy(ABC):
    """Protocol for diagnosis-specific repair strategies."""

    @abstractmethod
    def applicable_to(self, diagnosis: DiagnosisEvidence) -> bool:
        """Check if strategy applies to this diagnosis."""
        pass

    @abstractmethod
    def generate_repairs(
        self,
        diagnosis: DiagnosisEvidence,
        asset_context: Dict[str, Any],
    ) -> List[RepairCandidate]:
        """Generate repair candidates for this diagnosis."""
        pass


class DiagnosisMapper:
    """
    Maps diagnostic signals to repair mutations.

    Delegates to registered RepairStrategy implementations.
    """

    def __init__(self):
        self.strategies: List[RepairStrategy] = []

    def register_strategy(self, strategy: RepairStrategy) -> None:
        """Register a repair strategy."""
        self.strategies.append(strategy)

    def map_to_repairs(
        self,
        diagnoses: List[DiagnosisEvidence],
        asset_context: Dict[str, Any],
        max_repairs: int = 10,
    ) -> List[RepairCandidate]:
        """
        Map diagnoses to repair candidates.

        Args:
            diagnoses: List of diagnostic evidence
            asset_context: Context about the asset being repaired
            max_repairs: Maximum number of repairs to generate

        Returns:
            List of repair candidates, sorted by priority
        """
        all_repairs: List[RepairCandidate] = []

        for diagnosis in diagnoses:
            for strategy in self.strategies:
                if strategy.applicable_to(diagnosis):
                    repairs = strategy.generate_repairs(diagnosis, asset_context)
                    all_repairs.extend(repairs)

        all_repairs.sort(key=lambda r: r.priority, reverse=True)
        return all_repairs[:max_repairs]


class DefaultRepairStrategy(RepairStrategy):
    """
    Default repair strategy for common diagnoses.

    Provides baseline mutations for standard issues.
    """

    def applicable_to(self, diagnosis: DiagnosisEvidence) -> bool:
        """Apply to most diagnosis kinds."""
        return diagnosis.kind in {
            DiagnosisKind.POOR_RANK_IC,
            DiagnosisKind.HIGH_TURNOVER,
            DiagnosisKind.PARAMETER_INSTABILITY,
            DiagnosisKind.OVERFITTING,
        }

    def generate_repairs(
        self,
        diagnosis: DiagnosisEvidence,
        asset_context: Dict[str, Any],
    ) -> List[RepairCandidate]:
        """Generate default repairs based on diagnosis kind."""
        repairs = []

        if diagnosis.kind == DiagnosisKind.POOR_RANK_IC:
            repairs.append(
                RepairCandidate(
                    mutation_type="adjust_window",
                    parameters={"direction": "increase", "scale": 1.5},
                    priority=0.7 * diagnosis.severity,
                    rationale="Increase window to capture more signal",
                    expected_improvement={"rank_ic": 0.05},
                    risk_score=0.3,
                )
            )
            repairs.append(
                RepairCandidate(
                    mutation_type="change_aggregation",
                    parameters={"method": "robust"},
                    priority=0.6 * diagnosis.severity,
                    rationale="Use robust aggregation to reduce noise",
                    expected_improvement={"rank_ic": 0.03},
                    risk_score=0.4,
                )
            )

        elif diagnosis.kind == DiagnosisKind.HIGH_TURNOVER:
            repairs.append(
                RepairCandidate(
                    mutation_type="add_smoothing",
                    parameters={"window": 5, "method": "ewma"},
                    priority=0.8 * diagnosis.severity,
                    rationale="Add smoothing to reduce turnover",
                    expected_improvement={"turnover": -0.2},
                    risk_score=0.2,
                )
            )

        elif diagnosis.kind == DiagnosisKind.PARAMETER_INSTABILITY:
            repairs.append(
                RepairCandidate(
                    mutation_type="regularize",
                    parameters={"strength": "moderate"},
                    priority=0.7 * diagnosis.severity,
                    rationale="Add regularization for stability",
                    expected_improvement={"stability": 0.1},
                    risk_score=0.3,
                )
            )

        elif diagnosis.kind == DiagnosisKind.OVERFITTING:
            repairs.append(
                RepairCandidate(
                    mutation_type="simplify_model",
                    parameters={"complexity_reduction": 0.3},
                    priority=0.8 * diagnosis.severity,
                    rationale="Simplify model to reduce overfitting",
                    expected_improvement={"validation_ic": 0.05},
                    risk_score=0.4,
                )
            )

        return repairs


def create_default_mapper() -> DiagnosisMapper:
    """Create a DiagnosisMapper with default strategies."""
    mapper = DiagnosisMapper()
    mapper.register_strategy(DefaultRepairStrategy())
    return mapper
