"""
Novelty detection and QE evidence integration.

Provides protocol boundaries for QE evidence without duplicating evaluation logic.
"""

from factor_assets.novelty.provider import (
    EvidenceProvider,
    EvidenceQuery,
    EvidenceResult,
    MockEvidenceProvider,
)
from factor_assets.novelty.conditional import (
    NoveltyResult,
    ResultIdentity,
    ResultIdentityCache,
    ConditionalNoveltyProvider,
    SimpleConditionalNoveltyAssessor,
)

__all__ = [
    "EvidenceProvider",
    "EvidenceQuery",
    "EvidenceResult",
    "MockEvidenceProvider",
    "NoveltyResult",
    "ResultIdentity",
    "ResultIdentityCache",
    "ConditionalNoveltyProvider",
    "SimpleConditionalNoveltyAssessor",
]
