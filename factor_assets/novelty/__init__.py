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

__all__ = [
    "EvidenceProvider",
    "EvidenceQuery",
    "EvidenceResult",
    "MockEvidenceProvider",
]
