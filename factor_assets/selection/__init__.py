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
from factor_assets.selection.decision import (CandidateEvidence, DecisionArtifact,
    DecisionProvider, DecisionRequest, DecisionStatus, GateReceipt, MetricRule, ReplacementRoleRule,
    JointUtilityEvidence, RawJointMetricEvidence, Relationship, SelectionPolicySpec, UtilityDirection)

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
    "CandidateEvidence", "DecisionArtifact", "DecisionProvider", "DecisionRequest",
    "DecisionStatus", "GateReceipt", "MetricRule", "ReplacementRoleRule", "JointUtilityEvidence", "RawJointMetricEvidence", "Relationship",
    "SelectionPolicySpec", "UtilityDirection",
]
