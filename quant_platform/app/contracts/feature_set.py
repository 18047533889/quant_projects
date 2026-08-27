"""FeatureSet — typed members + versioned artifact + retrain decision.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.8 (spec §17,
§18). PURE stdlib frozen dataclasses + enum. ``[RECONCILE]`` against
``factor_assets/contracts/factor_set.py`` before freeze.

Ordered feature identity MUST participate in the hash.

IMPORTANT content-hash rule (§5.5): ``FeatureSetArtifact.content_hash`` IS
verifiable here because it is a **semantic hash over local fields** (unlike
``ArtifactRef``'s object-bytes hash, which this DTO cannot recompute). When the
caller supplies a ``content_hash``, we VERIFY it against the recomputed value and
fail closed on mismatch.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ._contenthash import content_hash
from .rbac import SecurityClassification
from .timing import TimingContract

__all__ = [
    "FeatureSetDiffCategory",
    "FeatureSetArtifact",
    "FeatureSetVersion",
    "FeatureMemberRef",
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
class FeatureMemberRef:
    """One ordered feature within a FeatureSetVersion (§5.5)."""

    position: int
    feature_name: str
    factor_definition_ref: str
    raw_value_ref: str | None = None
    treatment_selection_ref: str | None = None
    treated_feature_ref: str | None = None
    orientation: str | None = None
    dtype: str | None = None
    channel: str | None = None
    timing_ref: str | None = None
    security_classification: SecurityClassification | None = None

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("position must be >= 0")
        if not self.feature_name:
            raise ValueError("feature_name is required")
        if not self.factor_definition_ref:
            raise ValueError("factor_definition_ref is required")


@dataclass(frozen=True)
class FeatureSetVersion:
    """Versioned feature set with typed, ordered members (§5.5)."""

    feature_set_id: str
    version: str
    ordered_members: tuple[FeatureMemberRef, ...] = ()
    consumer_profile: str | None = None
    source_library_versions: tuple[str, ...] = ()
    schema_hash: str = ""
    semantic_hash: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.feature_set_id:
            raise ValueError("feature_set_id is required")
        if not self.version:
            raise ValueError("version is required")
        if not self.schema_hash:
            object.__setattr__(
                self,
                "schema_hash",
                content_hash(
                    self.feature_set_id,
                    self.version,
                    self.ordered_members,
                ),
            )
        if not self.semantic_hash:
            object.__setattr__(
                self,
                "semantic_hash",
                content_hash(
                    self.feature_set_id,
                    self.version,
                    self.ordered_members,
                    self.source_library_versions,
                    self.consumer_profile,
                ),
            )


@dataclass(frozen=True)
class FeatureSetArtifact:
    """Ordered feature-set artifact (spec §17).

    ``content_hash`` is a **semantic hash over local fields** — it IS recomputable
    by this DTO. When the caller supplies ``content_hash``, we VERIFY it against
    the recomputed value and fail closed on mismatch. (Contrast with
    ``ArtifactRef.content_hash``, an object-bytes hash this DTO cannot recompute.)
    """

    feature_set_id: str
    feature_set_version: str
    source_library_versions: tuple[str, ...] = ()
    ordered_feature_manifest: tuple[str, ...] = ()
    created_at: datetime | None = None
    content_hash: str = ""

    def recomputed_hash(self) -> str:
        """The authoritative hash over all semantic fields incl. ordering."""
        return content_hash(
            self.feature_set_id,
            self.feature_set_version,
            self.source_library_versions,
            self.ordered_feature_manifest,
        )

    def __post_init__(self) -> None:
        if not self.feature_set_id:
            raise ValueError("feature_set_id is required")
        if not self.feature_set_version:
            raise ValueError("feature_set_version is required")
        computed = self.recomputed_hash()
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match recomputed value — fail closed "
                f"(got {self.content_hash!r}, recomputed {computed!r})"
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
