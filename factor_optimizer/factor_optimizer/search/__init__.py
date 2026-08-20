"""Search orchestration for factor mutation optimization."""

from factor_optimizer.search.runner import SearchRunner, SearchConfig, SearchSession
from factor_optimizer.search.multifidelity import (
    FidelityTier,
    FidelitySpec,
    MultiFidelityScheduler,
    PromotionCriteria,
)
from factor_optimizer.search.pareto import ParetoPoint, ParetoFrontier, ParetoArchive
from factor_optimizer.search.plateau import (
    PlateauDetector,
    PlateauConfig,
    AdaptivePlateauDetector,
    MultiObjectivePlateauDetector,
)
from factor_optimizer.search.lineage import (
    LineageNode,
    LineageTree,
    LineageAnalyzer,
)

__all__ = [
    "SearchRunner",
    "SearchConfig",
    "SearchSession",
    "FidelityTier",
    "FidelitySpec",
    "MultiFidelityScheduler",
    "PromotionCriteria",
    "ParetoPoint",
    "ParetoFrontier",
    "ParetoArchive",
    "PlateauDetector",
    "PlateauConfig",
    "AdaptivePlateauDetector",
    "MultiObjectivePlateauDetector",
    "LineageNode",
    "LineageTree",
    "LineageAnalyzer",
]
