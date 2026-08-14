"""
Selection subsystem: gates and policies.

Evidence-based admission gates and selection decision records.
"""

from factor_assets.selection.gates import (
    GateResult,
    GateEvaluation,
    EvidenceGate,
    MetricDirection,
    MetricBinding,
    MetricEvidence,
    ThresholdGate,
    CompositeGate,
    MinimumICGate,
    MaximumTurnoverGate,
    MinimumCoverageGate,
    MinimumObservationsGate,
    ParetoDominanceGate,
)
from factor_assets.selection.policy import (
    SelectionDecision,
    SelectionReason,
    SelectionPolicy,
)

__all__ = [
    "GateResult",
    "GateEvaluation",
    "EvidenceGate",
    "MetricDirection",
    "MetricBinding",
    "MetricEvidence",
    "ThresholdGate",
    "CompositeGate",
    "MinimumICGate",
    "MaximumTurnoverGate",
    "MinimumCoverageGate",
    "MinimumObservationsGate",
    "ParetoDominanceGate",
    "SelectionDecision",
    "SelectionReason",
    "SelectionPolicy",
]
