"""
Factor Assets Optimizer: typed mutation and diagnosis-driven repair coordination.

This module coordinates with factor_optimizer (FO) through public contracts.
It does NOT implement mutations, search, or evaluation - those belong to FO.
Instead, it provides FA-specific orchestration for optimization campaigns.

Authority (DLIB-FA-001): FO owns Pareto / MultiFidelity / Plateau / Mutation /
Search / Sealed Test. FA is a consumer of FO evaluation-tier evidence and a
producer of Factor-Library-Assembly artifacts only. See
``FA_OPTIMIZER_MIGRATION_MATRIX.md`` in this directory.

- ``pareto.py``: assembly-only dominance comparator (used by
  ``assembly/engine.py`` to rank already-admitted candidates). VERIFIED
  assembly-only, not search/treatment.
- ``typed_mutation.py``: reference-only adapter for FO MutationSpec/Trial.
  VERIFIED reference-only, no FA mutation execution.
- ``multifidelity.py`` / ``plateau.py`` / ``frozen_candidate.py``: retained for
  compatibility (§108), marked RESEARCH_ONLY, emit DeprecationWarning on import.
  No FA production path consumes them.
"""

from factor_assets.optimizer.typed_mutation import (
    TypedMutation,
    MutationContext,
    MutationResult,
    MutationStatus,
)
from factor_assets.optimizer.diagnosis_mapper import (
    DiagnosisMapper,
    DiagnosisEvidence,
    RepairCandidate,
)
from factor_assets.optimizer.frozen_candidate import (
    FrozenCandidate,
    FrozenState,
    FrozenCandidateStateMachine,
    ContaminationKind,
)
from factor_assets.optimizer.pareto import (
    ParetoPoint,
    ParetoFrontier,
    ParetoOptimizer,
)
from factor_assets.optimizer.multifidelity import (
    FidelityTier,
    FidelitySpec,
    PromotionCriteria,
    MultiFidelityPolicy,
)
from factor_assets.optimizer.plateau import (
    ParameterNeighbor,
    PlateauAnalysis,
    ParameterPlateauDetector,
    NeighborSurvivalAnalyzer,
)

#: The optimizer subpackage is not a production search/treatment authority.
#: Only the assembly-only Pareto dominance comparator and the reference-only
#: typed-mutation adapter are production-relevant; the rest is retained for
#: compatibility and marked research-only.
RESEARCH_ONLY = True

__all__ = [
    # Typed mutation
    "TypedMutation",
    "MutationContext",
    "MutationResult",
    "MutationStatus",
    # Diagnosis mapping
    "DiagnosisMapper",
    "DiagnosisEvidence",
    "RepairCandidate",
    # Frozen candidate
    "FrozenCandidate",
    "FrozenState",
    "FrozenCandidateStateMachine",
    "ContaminationKind",
    # Pareto optimization
    "ParetoPoint",
    "ParetoFrontier",
    "ParetoOptimizer",
    # Multi-fidelity
    "FidelityTier",
    "FidelitySpec",
    "PromotionCriteria",
    "MultiFidelityPolicy",
    # Plateau detection
    "ParameterNeighbor",
    "PlateauAnalysis",
    "ParameterPlateauDetector",
    "NeighborSurvivalAnalyzer",
    "RESEARCH_ONLY",
]
