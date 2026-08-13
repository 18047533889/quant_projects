"""
Factor interaction analysis module.

Analyzes relationships between factors: pairwise correlations, conditional IC,
substitution effects, and complementarity detection.
"""

from quant_evaluator.metrics.interactions.pairwise import (
    compute_pairwise_correlation,
    compute_rolling_pairwise_correlation,
    compute_correlation_matrix,
)
from quant_evaluator.metrics.interactions.conditional import (
    compute_conditional_ic,
    compute_incremental_ic,
    compute_partial_ic,
)
from quant_evaluator.metrics.interactions.substitution import (
    compute_substitution_effect,
    detect_substitutable_factors,
    compute_marginal_contribution,
)
from quant_evaluator.metrics.interactions.complementarity import (
    compute_complementarity_score,
    detect_complementary_pairs,
    compute_interaction_strength,
)

__all__ = [
    "compute_pairwise_correlation",
    "compute_rolling_pairwise_correlation",
    "compute_correlation_matrix",
    "compute_conditional_ic",
    "compute_incremental_ic",
    "compute_partial_ic",
    "compute_substitution_effect",
    "detect_substitutable_factors",
    "compute_marginal_contribution",
    "compute_complementarity_score",
    "detect_complementary_pairs",
    "compute_interaction_strength",
]
