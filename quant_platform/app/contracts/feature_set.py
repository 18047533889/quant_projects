"""FeatureSetArtifact + FeatureSetDiff + ModelRetrainRequired event.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.8 (spec §17,
§18). PURE stdlib frozen dataclasses + enum. ``[RECONCILE]`` against
``factor_assets/contracts/factor_set.py`` before freeze.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

from ._contenthash import content_hash

__all__ = [
    "FeatureSetDiffCategory",
    "FeatureSetArtifact",
    "ModelRetrainRequiredEvent",
    "retrain_required_for_diff",
]


class FeatureSetDiffCategory(enum.Enum):
    """FeatureSetDiff categories (spec §18)."""

    METADATA_ONLY = "METADATA_ONLY"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"
    FEATURE_MEMBERSHIP_CHANGE = "FEATURE_MEMBERSHIP_CHANGE"
    FEATURE_TRANSFORM_CHANGE = "FEATURE_TRANSFORM_CHANGE"
    FEATURE_ORIENTATION_CHANGE = "FEATURE_ORIENTATION_CHANGE"
    FEATURE_SCHEMA_CHANGE = "FEATURE_SCHEMA_CHANGE"
    LABEL_CHANGE = "LABEL_CHANGE"
    DATA_REVISION = "DATA_REVISION"

    @property
    def retrain_required(self) -> bool:
        """Default retrain decision (spec §18)."""
        return self in {
            FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE,
            FeatureSetDiffCategory.FEATURE_TRANSFORM_CHANGE,
            FeatureSetDiffCategory.FEATURE_ORIENTATION_CHANGE,
            FeatureSetDiffCategory.FEATURE_SCHEMA_CHANGE,
            FeatureSetDiffCategory.LABEL_CHANGE,
            FeatureSetDiffCategory.DATA_REVISION,
        }


def retrain_required_for_diff(category: FeatureSetDiffCategory) -> bool:
    """Convenience wrapper for ``FeatureSetDiffCategory.retrain_required``."""
    return category.retrain_required


@dataclass(frozen=True)
class FeatureSetArtifact:
    """Ordered feature-set artifact (spec §17)."""

    feature_set_id: str
    feature_set_version: str
    source_library_versions: tuple[str, ...] = ()
    ordered_feature_manifest: tuple[str, ...] = ()
    content_hash: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.feature_set_id:
            raise ValueError("feature_set_id is required")
        if not self.feature_set_version:
            raise ValueError("feature_set_version is required")
        if not self.content_hash:
            object.__setattr__(
                self,
                "content_hash",
                content_hash(
                    self.feature_set_id,
                    self.feature_set_version,
                    self.source_library_versions,
                    self.ordered_feature_manifest,
                ),
            )


@dataclass(frozen=True)
class ModelRetrainRequiredEvent:
    """Event emitted when a FeatureSetDiff requires model retrain (spec §18)."""

    feature_set_id: str
    diff_category: FeatureSetDiffCategory
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.feature_set_id:
            raise ValueError("feature_set_id is required")
