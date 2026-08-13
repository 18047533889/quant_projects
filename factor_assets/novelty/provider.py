"""
QE evidence protocol boundary.

FA does NOT duplicate QE's evaluation logic or metric computation.
Evidence is obtained through a protocol boundary from QE adapters.
"""

from dataclasses import dataclass
from typing import Protocol, Optional


@dataclass(frozen=True)
class EvidenceQuery:
    """
    Query specification for QE evidence lookup.

    Specifies what evidence to retrieve without containing the evidence itself.
    """
    factor_id: str
    evaluation_run_id: Optional[str] = None
    metric_names: tuple[str, ...] = ()
    universe_ref: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")


@dataclass(frozen=True)
class EvidenceResult:
    """
    Evidence retrieval result from QE.

    Contains only bounded summary statistics, not full metric arrays.
    Full evidence lives in QE; FA stores only references and summaries.
    """
    factor_id: str
    evaluation_run_id: str
    evidence_id: str
    timestamp: str  # ISO 8601
    available: bool
    # Bounded summary metrics (not full arrays)
    primary_metric_name: Optional[str] = None
    primary_metric_value: Optional[float] = None
    secondary_metrics: dict[str, float] = None
    universe_ref: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    qe_version: Optional[str] = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.evaluation_run_id:
            raise ValueError("evaluation_run_id is required")
        if self.secondary_metrics is None:
            object.__setattr__(self, 'secondary_metrics', {})

    @property
    def has_warnings(self) -> bool:
        """Check if evidence has warnings."""
        return len(self.warnings) > 0

    @property
    def is_valid(self) -> bool:
        """Check if evidence is available and valid."""
        return self.available and not self.has_warnings


class EvidenceProvider(Protocol):
    """
    Protocol for obtaining evaluation evidence from QE.

    FA does not implement this — it is provided by an optional QE adapter.
    This protocol defines the boundary without creating a hard dependency.
    """

    def get_evidence(self, query: EvidenceQuery) -> Optional[EvidenceResult]:
        """
        Retrieve evaluation evidence for a factor.

        Args:
            query: Evidence query specification

        Returns:
            EvidenceResult if found, None otherwise

        Raises:
            ValueError: If query is invalid
        """
        ...

    def has_evidence(self, factor_id: str, evaluation_run_id: Optional[str] = None) -> bool:
        """
        Check if evidence exists for a factor.

        Args:
            factor_id: Factor identifier
            evaluation_run_id: Optional specific evaluation run

        Returns:
            True if evidence exists, False otherwise
        """
        ...

    def get_latest_evidence(self, factor_id: str) -> Optional[EvidenceResult]:
        """
        Get most recent evidence for a factor.

        Args:
            factor_id: Factor identifier

        Returns:
            Most recent EvidenceResult if found, None otherwise
        """
        ...


class MockEvidenceProvider:
    """
    Mock evidence provider for testing.

    Stores evidence in memory without requiring real QE integration.
    """

    def __init__(self):
        self._evidence: dict[str, list[EvidenceResult]] = {}

    def add_evidence(self, result: EvidenceResult) -> None:
        """
        Add evidence result to mock store.

        Args:
            result: Evidence result to store
        """
        if result.factor_id not in self._evidence:
            self._evidence[result.factor_id] = []
        self._evidence[result.factor_id].append(result)

    def get_evidence(self, query: EvidenceQuery) -> Optional[EvidenceResult]:
        """Get evidence matching query."""
        if query.factor_id not in self._evidence:
            return None

        results = self._evidence[query.factor_id]

        # Filter by evaluation_run_id if specified
        if query.evaluation_run_id:
            results = [r for r in results if r.evaluation_run_id == query.evaluation_run_id]

        # Filter by metric names if specified
        if query.metric_names:
            results = [
                r for r in results
                if r.primary_metric_name in query.metric_names
            ]

        # Return most recent if multiple matches
        if results:
            return sorted(results, key=lambda r: r.timestamp, reverse=True)[0]

        return None

    def has_evidence(self, factor_id: str, evaluation_run_id: Optional[str] = None) -> bool:
        """Check if evidence exists."""
        if factor_id not in self._evidence:
            return False

        if evaluation_run_id:
            return any(
                r.evaluation_run_id == evaluation_run_id
                for r in self._evidence[factor_id]
            )

        return True

    def get_latest_evidence(self, factor_id: str) -> Optional[EvidenceResult]:
        """Get most recent evidence."""
        if factor_id not in self._evidence:
            return None

        results = self._evidence[factor_id]
        if not results:
            return None

        return sorted(results, key=lambda r: r.timestamp, reverse=True)[0]

    def count(self) -> int:
        """Get total number of evidence results stored."""
        return sum(len(results) for results in self._evidence.values())

    def clear(self) -> None:
        """Clear all stored evidence."""
        self._evidence.clear()
