"""Storage ports — ArtifactPublisher / ArtifactResolver / ArtifactStoragePort.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.11 (spec §22).
PURE stdlib ``typing.Protocol``.

The platform does NOT own a second ObjectStore authority. ``data_access`` owns the
mature ObjectStore. This module declares *platform-side ports* that will be
implemented later via an adapter over DataAccess's public ObjectStore — that
adapter is a separate task (QRP-P0R-C5). Only the contract types live here.

- ``ArtifactStoragePort`` — minimal durable-byte operations (put/get/head/delete).
- ``ArtifactPublisher`` — publishes bytes, computes ``content_hash``, uploads,
  HEAD-verifies, returns an ``ArtifactRef``.
- ``ArtifactResolver`` — download/open + ``content_hash`` verify.
- ``ArtifactMaterializer`` — download + verify + write to a local path.
- ``ArtifactAccessGate`` / ``FormulaAccessGuard`` / ``AuditEvent`` — artifact
  security classification + formula-endpoint authorization + audit (spec §23).
- ``LocalArtifactCache`` — local disk cache keyed by content_hash.

``ObjectStore``/``ObjectMetadata`` (the old redefinition) is kept ONLY as a
deprecated alias for backward-compat, because nothing outside this package imports
it — we check imports first (grep confirms only ``data_access`` owns an ObjectStore).
If nothing imports it, we keep the alias marked deprecated; it will be removed in
the C5 adapter task.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol, runtime_checkable

from .artifact_ref import ArtifactRef
from .rbac import Permission, SecurityClassification

__all__ = [
    "ArtifactStoragePort",
    "ArtifactPublisher",
    "ArtifactResolver",
    "ArtifactMaterializer",
    "LocalArtifactCache",
    "CacheEvictionPolicy",
    "ArtifactAccessGate",
    "FormulaAccessGuard",
    "AuditEvent",
    "ObjectStore",
    "ObjectMetadata",
]


@runtime_checkable
class ArtifactStoragePort(Protocol):
    """Minimal durable-byte operations (adapter over DataAccess ObjectStore)."""

    def put(self, uri: str, data: bytes) -> "ObjectMetadata":
        """Upload bytes; two-phase publish semantics (spec §7.4)."""
        ...

    def get(self, uri: str) -> bytes:
        """Download the full object."""
        ...

    def head(self, uri: str) -> "ObjectMetadata":
        """Return object metadata (content_hash / size / etag)."""
        ...

    def delete(self, uri: str) -> None:
        """Explicit deletion; never for published artifacts."""
        ...


@runtime_checkable
class ArtifactPublisher(Protocol):
    """Publishes bytes → computes content_hash → uploads → HEAD-verifies →
    returns ArtifactRef (spec §7.4)."""

    def publish(self, artifact: ArtifactRef, data: bytes) -> ArtifactRef:
        """Publish ``data`` for ``artifact``; computes/verifies content_hash.

        Returns the resolved ``ArtifactRef`` (content_hash set from actual bytes).
        """
        ...


@runtime_checkable
class ArtifactResolver(Protocol):
    """Download/open + content_hash verify (spec §22)."""

    def open(self, artifact: ArtifactRef) -> bytes:
        """Download bytes for ``artifact`` and verify content_hash; fail closed."""
        ...

    def verify(self, artifact: ArtifactRef, data: bytes) -> bool:
        """True iff ``data``'s sha256 equals ``artifact.content_hash``."""
        ...


class CacheEvictionPolicy(enum.Enum):
    """LocalArtifactCache eviction policy (spec §22)."""

    LRU = "LRU"


@runtime_checkable
class LocalArtifactCache(Protocol):
    """Local disk cache keyed by content_hash (spec §22)."""

    max_bytes: int
    eviction_policy: CacheEvictionPolicy

    def lookup(self, content_hash: str) -> bytes | None:
        """Return cached bytes for ``content_hash`` or None on miss."""
        ...

    def store(self, content_hash: str, data: bytes) -> None:
        """Store bytes, validating checksum; enforce max_bytes + LRU eviction."""
        ...

    def hit_rate(self) -> float:
        """Cache hit-rate metric (spec §22)."""
        ...


@runtime_checkable
class ObjectMetadata(Protocol):
    """Metadata returned by storage ``head`` (spec §22)."""

    content_hash: str
    size_bytes: int
    etag: str | None


# ---------------------------------------------------------------------------
# Artifact security classification + access gate (spec §5.8, §23)
# ---------------------------------------------------------------------------


@runtime_checkable
class ArtifactAccessGate(Protocol):
    """Authorization gate for artifact access.

    ``authorize`` returns True iff the principal may exercise ``permission`` on a
    resource whose ``security_classification`` is ``classification``. The RBAC
    agent wires this to the platform's permission/principal model (see
    ``quant_platform.app.security.rbac.RBAC``); a minimal pure implementation is
    provided in :mod:`quant_platform.app.storage.access_guard`.
    """

    def authorize(
        self,
        principal: Any,
        permission: Permission,
        classification: SecurityClassification,
        resource: Any = None,
    ) -> bool:
        """True iff ``principal`` may exercise ``permission`` at ``classification``."""
        ...


@dataclass(frozen=True)
class AuditEvent:
    """Audit record for a guarded access decision (spec §23)."""

    principal_id: str
    team: str | None
    role: str | None
    permission: str
    resource: str
    request_id: str | None
    security_digest: str | None
    credential_scope_id: str | None
    action: str
    result: str  # "ALLOW" | "DENY"
    timestamp: datetime


@runtime_checkable
class FormulaAccessGuard(Protocol):
    """Guard for ``GET /factors/{id}/formula`` (spec §23).

    ``guard`` returns an ``AuditEvent`` describing the decision. A denied decision
    must map to HTTP 403 (never 200 + null). The guard MUST record an audit event
    on every invocation (allow or deny).
    """

    def guard(
        self,
        *,
        principal: Any,
        factor_id: str,
        request_id: str | None = None,
        security_digest: str | None = None,
        credential_scope_id: str | None = None,
    ) -> AuditEvent:
        """Authorize formula read; returns the audit event for the decision."""
        ...


# ---------------------------------------------------------------------------
# materializer (spec §22)
# ---------------------------------------------------------------------------


@runtime_checkable
class ArtifactMaterializer(Protocol):
    """Materializes a published artifact to a local path (spec §22).

    ``materialize`` resolves the artifact, verifies its content_hash, and writes
    the bytes to ``dest``. Returns the destination path. Fails closed on any
    content_hash mismatch.
    """

    def materialize(self, artifact: ArtifactRef, dest: str) -> str:
        """Download + verify + write ``artifact`` to ``dest``; return ``dest``."""
        ...


# DEPRECATED alias: the platform must NOT become a second ObjectStore authority.
# DataAccess owns ObjectStore; this alias is retained only for any legacy import
# and will be removed in the C5 adapter task (QRP-P0R-C5).
ObjectStore = ArtifactStoragePort
