"""
Similarity computation and detection.

Correlation-based similarity for factor comparison and ANN-based fast search.
"""

from factor_assets.similarity.exact import (
    SimilarityMethod,
    SimilarityMeasure,
    SimilarityResult,
    CorrelationSimilarity,
    QEPairwiseSimilarity,
)

try:
    from factor_assets.similarity.ann import (
        ANNBackend,
        ANNSearchResult,
        ANNIndex,
        FaissANNIndex,
        AnnoyANNIndex,
        create_ann_index,
        IndexCapability,
    )
    ANN_AVAILABLE = True
except ImportError:
    ANN_AVAILABLE = False
    ANNBackend = None
    ANNSearchResult = None
    ANNIndex = None
    FaissANNIndex = None
    AnnoyANNIndex = None
    create_ann_index = None
    IndexCapability = None

__all__ = [
    "SimilarityMethod",
    "SimilarityMeasure",
    "SimilarityResult",
    "CorrelationSimilarity",
    "QEPairwiseSimilarity",
    "ANNBackend",
    "ANNSearchResult",
    "ANNIndex",
    "FaissANNIndex",
    "AnnoyANNIndex",
    "create_ann_index",
    "IndexCapability",
    "ANN_AVAILABLE",
]
