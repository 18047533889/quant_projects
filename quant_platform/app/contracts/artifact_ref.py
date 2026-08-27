"""ArtifactRef — platform-facing artifact reference DTO.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.1 (spec §7.2).
PURE stdlib frozen dataclass. This is the *platform-facing ref / adapter
boundary*; it does not replace domain-native artifacts (e.g.
``factor_assets/contracts/treatment_selection.py``). ``[RECONCILE]`` against
domain-native ref types before freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

__all__ = [
    "ArtifactRef",
    "ARTIFACT_TYPE_FACTOR_CANDIDATE",
    "ARTIFACT_TYPE_FACTOR_DEFINITION",
    "ARTIFACT_TYPE_FACTOR_VALUE",
    "ARTIFACT_TYPE_EVALUATION_BUNDLE",
    "ARTIFACT_TYPE_TREATMENT_SELECTION",
    "ARTIFACT_TYPE_SIMILARITY_FINGERPRINT",
    "ARTIFACT_TYPE_SIMILARITY_GRAPH",
    "ARTIFACT_TYPE_CLUSTER_VERSION",
    "ARTIFACT_TYPE_CLUSTER_ASSIGNMENT",
    "ARTIFACT_TYPE_FACTOR_LIBRARY_VERSION",
    "ARTIFACT_TYPE_FEATURE_SET",
    "ARTIFACT_TYPE_MODEL_DATASET",
    "ARTIFACT_TYPE_MODEL",
    "ARTIFACT_TYPE_BACKTEST",
    "ARTIFACT_TYPE_REPORT_EXPORT",
    "ARTIFACT_TYPES",
]

# spec §7.2 artifact type constants
ARTIFACT_TYPE_FACTOR_CANDIDATE = "FACTOR_CANDIDATE"
ARTIFACT_TYPE_FACTOR_DEFINITION = "FACTOR_DEFINITION"
ARTIFACT_TYPE_FACTOR_VALUE = "FACTOR_VALUE"
ARTIFACT_TYPE_EVALUATION_BUNDLE = "EVALUATION_BUNDLE"
ARTIFACT_TYPE_TREATMENT_SELECTION = "TREATMENT_SELECTION"
ARTIFACT_TYPE_SIMILARITY_FINGERPRINT = "SIMILARITY_FINGERPRINT"
ARTIFACT_TYPE_SIMILARITY_GRAPH = "SIMILARITY_GRAPH"
ARTIFACT_TYPE_CLUSTER_VERSION = "CLUSTER_VERSION"
ARTIFACT_TYPE_CLUSTER_ASSIGNMENT = "CLUSTER_ASSIGNMENT"
ARTIFACT_TYPE_FACTOR_LIBRARY_VERSION = "FACTOR_LIBRARY_VERSION"
ARTIFACT_TYPE_FEATURE_SET = "FEATURE_SET"
ARTIFACT_TYPE_MODEL_DATASET = "MODEL_DATASET"
ARTIFACT_TYPE_MODEL = "MODEL"
ARTIFACT_TYPE_BACKTEST = "BACKTEST"
ARTIFACT_TYPE_REPORT_EXPORT = "REPORT_EXPORT"

ARTIFACT_TYPES: frozenset[str] = frozenset(
    {
        ARTIFACT_TYPE_FACTOR_CANDIDATE,
        ARTIFACT_TYPE_FACTOR_DEFINITION,
        ARTIFACT_TYPE_FACTOR_VALUE,
        ARTIFACT_TYPE_EVALUATION_BUNDLE,
        ARTIFACT_TYPE_TREATMENT_SELECTION,
        ARTIFACT_TYPE_SIMILARITY_FINGERPRINT,
        ARTIFACT_TYPE_SIMILARITY_GRAPH,
        ARTIFACT_TYPE_CLUSTER_VERSION,
        ARTIFACT_TYPE_CLUSTER_ASSIGNMENT,
        ARTIFACT_TYPE_FACTOR_LIBRARY_VERSION,
        ARTIFACT_TYPE_FEATURE_SET,
        ARTIFACT_TYPE_MODEL_DATASET,
        ARTIFACT_TYPE_MODEL,
        ARTIFACT_TYPE_BACKTEST,
        ARTIFACT_TYPE_REPORT_EXPORT,
    }
)


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable reference to a published artifact (spec §7.2).

    Fields match the spec exactly. ``content_hash`` is derived per the
    content-hash rule (DRAFT doc §2); a caller-supplied hash that does not match
    the recomputed value fails closed.
    """

    artifact_id: str
    artifact_type: str
    schema_version: str
    content_hash: str
    storage_uri: str
    size_bytes: int
    created_at: datetime
    producer_type: str
    producer_version: str
    snapshot_id: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id is required")
        if self.artifact_type not in ARTIFACT_TYPES:
            raise ValueError(f"unknown artifact_type: {self.artifact_type!r}")
        if not self.schema_version:
            raise ValueError("schema_version is required")
        if not self.content_hash:
            raise ValueError("content_hash is required (derived, not self-reported)")
        if not self.storage_uri:
            raise ValueError("storage_uri is required")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")
