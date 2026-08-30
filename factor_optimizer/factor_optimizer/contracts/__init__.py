"""Contracts for mutation proposals, search budgets, and trials."""

from .candidate_mutation import CandidateMutation
from .objective import (
    DEFAULT_METRIC_NAME,
    OBJECTIVE_DIRECTIONS,
    ObjectiveDirection,
    ObjectiveSpec,
)
from .search_budget import SearchBudget, BudgetTracker
from .trial import Trial, TrialStatus
from .splits import EvaluationProtocol, SplitPlan
from .validator import TrialValidatorIdentity, MutationGrammarValidator
from .library_snapshot_ref import LibrarySnapshotRef
from .treatment_integrity import (
    EVIDENCE_SCHEMA_VERSION,
    RAW_TREATMENT_KIND,
    IntegrityCheckResult,
    TreatmentIntegrityEvidence,
    TreatmentIntegrityStatus,
    build_integrity_evidence,
    describe_integrity_problem,
    digest_value,
    require_integrity_evidence,
)

__all__ = [
    "CandidateMutation",
    "ObjectiveSpec",
    "ObjectiveDirection",
    "OBJECTIVE_DIRECTIONS",
    "DEFAULT_METRIC_NAME",
    "SearchBudget",
    "BudgetTracker",
    "Trial",
    "TrialStatus",
    "SplitPlan",
    "EvaluationProtocol",
    "TrialValidatorIdentity",
    "MutationGrammarValidator",
    "LibrarySnapshotRef",
    # Treatment integrity evidence (R55 P0-9)
    "EVIDENCE_SCHEMA_VERSION",
    "RAW_TREATMENT_KIND",
    "IntegrityCheckResult",
    "TreatmentIntegrityEvidence",
    "TreatmentIntegrityStatus",
    "build_integrity_evidence",
    "describe_integrity_problem",
    "digest_value",
    "require_integrity_evidence",
]
