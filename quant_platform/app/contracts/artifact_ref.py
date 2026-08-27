"""ArtifactRef — platform-facing artifact reference DTO.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.1 (spec §7.2).
PURE stdlib frozen dataclass. This is the *platform-facing ref / adapter
boundary*; it does not replace domain-native artifacts (e.g.
``factor_assets/contracts/treatment_selection.py``). ``[RECONCILE]`` against
domain-native ref types before freeze.

An ``ArtifactRef`` is a **REFERENCE**, not a validator. It does NOT claim to
recompute or verify the real object byte hash — that is the job of the artifact
publisher / resolver (see ``storage.py``), which hashes the actual COS bytes.
This DTO only validates the *shape* of its fields: hash FORMAT, required fields,
URI scheme, and ``size_bytes >= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re

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

# Common sha256 hex form: 64 lowercase hex chars.
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable reference to a *published* artifact (spec §7.2).

    ``ArtifactRef`` is a REFERENCE, not a validator. It does **not** recompute or
    verify the real object byte hash — that is the responsibility of the
    ``ArtifactPublisher``/``ArtifactResolver`` (see ``storage.py``) which hash the
    actual COS bytes. This DTO only validates the *shape* of its fields.

    Two hash slots are distinguished:
      - ``semantic_hash``: a domain-semantics hash (over semantic fields), if
        applicable to this artifact type; may be empty for opaque byte blobs.
      - ``content_hash``: the COS/object-bytes SHA-256 computed by the publisher
        over the stored object. It is *reported* by the publisher/resolver, NOT
        recomputed by this DTO. Here we only validate its hex format.

    A caller-supplied hash is never "verified against a recomputed value" by this
    class — it cannot, because it has no access to the underlying object bytes.
    Verification happens at the storage boundary.
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
    # NEW fields (§5.1)
    semantic_hash: str = ""
    media_type: str = "application/octet-stream"
    producer_source_ref: str | None = None
    snapshot_ref: str | None = None
    universe_ref: str | None = None
    security_classification: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id is required")
        if self.artifact_type not in ARTIFACT_TYPES:
            raise ValueError(f"unknown artifact_type: {self.artifact_type!r}")
        if not self.schema_version:
            raise ValueError("schema_version is required")
        if not self.content_hash:
            raise ValueError(
                "content_hash is required (reported by the publisher/resolver, "
                "not recomputed by ArtifactRef)"
            )
        if not _SHA256_HEX_RE.match(self.content_hash):
            raise ValueError(
                f"content_hash must be a 64-char lowercase hex sha256, got {self.content_hash!r}"
            )
        if not self.storage_uri:
            raise ValueError("storage_uri is required")
        if not _SCHEME_RE.match(self.storage_uri):
            raise ValueError(f"storage_uri must carry a URI scheme, got {self.storage_uri!r}")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")
