"""
Factor Assets Optimizer: Typed mutation and diagnosis-driven repair coordination.

This module coordinates with factor_optimizer (FO) through public contracts.
It does NOT implement mutations, search, or evaluation - those belong to FO.
Instead, it provides FA-specific orchestration for optimization campaigns.
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
]
