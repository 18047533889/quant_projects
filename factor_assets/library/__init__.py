"""Library governance decision gates (QRP-P7).

Re-exports the promotion/rollback decision pure functions and their immutable
artifacts. This package adds no second Authority: the gates only *decide*;
the ``FactorLibraryVersionArtifact`` remains the single source of truth for the
library body, and promotion / rollback (upstream, by the library operator)
still only change the active pointer.
"""

from factor_assets.library.promotion_gate import (
    PromotionDecision,
    PromotionReasonCode,
    VWAP_TO_VWAP_BASIS,
    EVIDENCE_NOT_COMPUTED,
    DEFAULT_MIN_RANK_IC,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_REJECT_DUPLICATES,
    CandidateEvaluationRef,
    PromotionDecisionArtifact,
    RollbackDecisionArtifact,
    PromotionGate,
    RollbackGate,
)

__all__ = [
    "PromotionDecision",
    "PromotionReasonCode",
    "VWAP_TO_VWAP_BASIS",
    "EVIDENCE_NOT_COMPUTED",
    "DEFAULT_MIN_RANK_IC",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_REJECT_DUPLICATES",
    "CandidateEvaluationRef",
    "PromotionDecisionArtifact",
    "RollbackDecisionArtifact",
    "PromotionGate",
    "RollbackGate",
]