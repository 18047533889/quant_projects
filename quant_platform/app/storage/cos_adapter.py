"""Minimal COS ArtifactPublisher — bytes → COS → HEAD verify → ArtifactRef.

Implements the ``ArtifactPublisher`` port declared in
:mod:`quant_platform.app.contracts.storage` for a real COS/S3 bucket, per
QRP-P2 (referencing — not redefining — the ``data_access`` COS multipart
publisher, QRP-P0R-C6).

Publisher semantics (this layer only; **no second ObjectStore authority**):
- ``content_hash`` is computed *here* over the actual object bytes (sha256).
- The bytes are uploaded to the COS object at ``storage_uri`` (single PUT; the
  ``data_access`` ``COSObjectStore`` handles real multipart above its own
  threshold internally, exactly as the C6 publisher does — we defer to it, we do
  not reimplement multipart).
- A HEAD call verifies the object exists and its size matches the bytes. If the
  caller supplied a ``content_hash`` that disagrees with the actual bytes, we
  fail closed before touching COS (never publish bytes under a hash that does
  not match them).
- Returns the resolved ``ArtifactRef`` (``content_hash`` = actual, ``size_bytes``
  = actual).

Injection / testability
-----------------------
``store`` is an optional duck-typed injectable exposing at least
``put_object(key, data)`` and ``head_object(key) -> dict | None`` (with a
``size`` key). When ``None``, the adapter lazily builds a ``data_access``
``COSObjectStore`` so that COS object semantics (credential-aware client,
multipart routing) stay in ``data_access`` — the single ObjectStore authority.

``import data_access`` is guarded: if it is not importable, the module still
imports (documentation form) and marks ``COS_AVAILABLE=False`` with a
``COS_BLOCKED_REASON``; ``COSArtifactPublisher`` then raises
``COSUnavailableError`` when constructed.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import replace
from typing import Any

from quant_platform.app.contracts import ArtifactRef

__all__ = [
    "COSArtifactPublisher",
    "COSUnavailableError",
    "COS_AVAILABLE",
    "COS_BLOCKED_REASON",
    "sha256_bytes",
]

logger = logging.getLogger("quant_platform.storage.cos_adapter")


class COSUnavailableError(RuntimeError):
    """Raised when the COS adapter is constructed but ``data_access`` is not
    importable (the COS object store authority is unavailable)."""


def sha256_bytes(data: bytes) -> str:
    """sha256 hex of ``data`` — the object-bytes content hash (computed here)."""
    return hashlib.sha256(data).hexdigest()


def _uri_to_key(uri: str) -> str:
    """Extract a POSIX-relative object key from a ``cos://.../key`` uri.

    Matches the platform convention in the C5 adapter
    (:func:`quant_platform.app.adapters.data_access_storage._uri_to_key`): the
    path after the scheme *is* the key — the bucket is configured on the store,
    not carried in the uri.
    """
    if "://" in uri:
        uri = uri.split("://", 1)[1]
    return uri.lstrip("/")


def _check_data_access_importable() -> tuple[bool, str]:
    """Return (available, blocked_reason). If unavailable, reason is the exact
    import traceback message; nothing is faked."""
    try:
        import data_access  # noqa: F401
        import data_access.read.object_store as _cos_store  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"COS adapter BLOCKED: data_access import failed: "
            f"{type(exc).__name__}: {exc}"
        )
    return True, ""


COS_AVAILABLE, COS_BLOCKED_REASON = _check_data_access_importable()


def _build_cos_store(bucket: str, **cos_kwargs: Any):
    """Lazily build the ``data_access`` COSObjectStore (single ObjectStore
    authority). Import is deferred so the module loads even when ``data_access``
    is unavailable."""
    from data_access.read.object_store import COSObjectStore  # noqa: PLC0415

    return COSObjectStore(str(bucket), **cos_kwargs)


class COSArtifactPublisher:
    """Minimal publisher: bytes → COS → HEAD verify → ArtifactRef (spec §7.4).

    ``content_hash`` is computed at this layer over the actual bytes. A
    caller-supplied ``content_hash`` that disagrees with the bytes is a hard
    error (fail closed). The publish succeeds only after HEAD verification.
    """

    def __init__(
        self,
        bucket: str | None = None,
        *,
        store: Any = None,
        **cos_kwargs: Any,
    ) -> None:
        if store is not None:
            self.store = store
        else:
            if not COS_AVAILABLE:
                raise COSUnavailableError(COS_BLOCKED_REASON)
            if not bucket:
                raise ValueError("COSArtifactPublisher requires a bucket or an injected store")
            self.store = _build_cos_store(bucket, **cos_kwargs)
        self.bucket = bucket

    def publish(self, artifact: ArtifactRef, data: bytes) -> ArtifactRef:
        """Publish ``data`` to ``artifact.storage_uri``; return resolved ref.

        Steps: compute actual sha256 → fail closed on caller-supplied hash
        mismatch → PUT bytes → HEAD verify (object present, size matches) →
        return resolved ``ArtifactRef``.
        """
        actual = sha256_bytes(data)
        if artifact.content_hash and artifact.content_hash != actual:
            raise ValueError(
                "COS publish fail-closed: ArtifactRef content_hash "
                f"{artifact.content_hash!r} != actual bytes {actual!r}"
            )
        key = _uri_to_key(artifact.storage_uri)
        if not key:
            raise ValueError(f"COS publish requires a non-empty storage_uri key, got {artifact.storage_uri!r}")

        self.store.put_object(key, data)

        head = self.store.head_object(key) if hasattr(self.store, "head_object") else None
        stored_size = int(head.get("size", -1)) if head else -1
        if stored_size != len(data):
            raise RuntimeError(
                "COS publish HEAD-verify failed: stored size "
                f"{stored_size} != {len(data)} for key {key!r}"
            )

        return replace(
            artifact,
            content_hash=actual,
            size_bytes=len(data),
        )