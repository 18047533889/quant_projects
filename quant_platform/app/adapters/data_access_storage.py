"""QRP-P0R-C5 — platform storage adapter runtime over ``data_access`` ObjectStore.

The platform contract layer (:mod:`quant_platform.app.contracts.storage`) declares
PURE-stdlib ``typing.Protocol`` ports — ``ArtifactStoragePort`` /
``ArtifactPublisher`` / ``ArtifactResolver`` / ``LocalArtifactCache``. This module
*implements* those ports over ``data_access``'s mature public ObjectStore (and its
two-phase immutable-generation publisher), NOT by redefining a second storage
authority.

Design
------
- :class:`DataAccessStorageAdapter` implements ``ArtifactStoragePort``. It wraps a
  ``data_access`` ObjectStore (injected, or built as ``LocalObjectStore`` /
  ``COSObjectStore`` from a bucket). Raw byte ops (put/get/head/delete) delegate
  straight to the store, with single-PUT-vs-real-multipart routing for large blobs.
- For *published* artifacts we route through the two-phase
  ``ObjectStoreGenerationPublisher`` so an artifact becomes atomically visible:
  payload uploaded under ``<prefix>/<generation_id>/payload``, immutable manifest
  written, then the ``CURRENT.json`` pointer flipped last. Readers only ever see the
  old complete generation or the new complete generation — never a partial publish.
- :class:`ArtifactPublisherImpl` computes the sha256 ``content_hash`` over the actual
  bytes, validates the ``ArtifactRef`` semantic fields, publishes via the two-phase
  path, HEAD-verifies, and returns a resolved ``ArtifactRef``.
- :class:`ArtifactResolverImpl`` downloads + verifies ``sha256(content) ==
  content_hash`` and fails closed on mismatch.
- :class:`LocalArtifactCache`` is a real on-disk LRU cache keyed by ``content_hash``
  with a ``hit_rate`` metric.
- Security: when a ``DataAccessExecutionContext`` is active, all operations run
  under that context (a ContextVar), so ``data_access.cos.remote`` / the COS store's
  credential-aware client builder resolve per-principal scoped credentials. If the
  store is COS-backed and the context carries a ``CredentialProvider``, we force the
  store client to be (re)built under that context before each op — never reuse a
  credential built for another principal.

Only ``data_access`` public API and ``quant_platform.app.contracts`` are imported.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_access.core.exceptions import DataError
from data_access.read.object_store import (
    COSObjectStore,
    LocalObjectStore,
    ObjectStore,
)
from data_access.security.execution_context import current_execution_context
from data_access.write.object_store_generation_publisher import (
    ObjectStoreGenerationPublisher,
)

from quant_platform.app.contracts import (
    ArtifactRef,
    CacheEvictionPolicy,
    ObjectMetadata,
)

__all__ = [
    "DataAccessStorageAdapter",
    "ArtifactPublisherImpl",
    "ArtifactResolverImpl",
    "LocalArtifactCacheImpl",
    "ObjectMetadataImpl",
    "sha256_bytes",
]

# The fixed object key (relative to the generation dir) holding a published payload.
_PAYLOAD_KEY = "payload"

_DEFAULT_MULTIPART_THRESHOLD = 8 * 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """sha256 hex of the given bytes (the object-bytes content hash)."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ObjectMetadataImpl:
    """Concrete ``ObjectMetadata`` returned by storage head/put."""

    content_hash: str
    size_bytes: int
    etag: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# storage adapter over data_access ObjectStore
# ---------------------------------------------------------------------------


class DataAccessStorageAdapter:
    """Implements ``ArtifactStoragePort`` over a ``data_access`` ObjectStore.

    ``put/get/head/delete`` delegate to the wrapped ObjectStore. Published artifacts
    use :meth:`publish_generation` which routes through the two-phase
    ``ObjectStoreGenerationPublisher`` (manifest + CURRENT-pointer flip) so a
    publish is atomically visible.
    """

    def __init__(
        self,
        store: ObjectStore | None = None,
        *,
        bucket: str | None = None,
        root: str | Path | None = None,
        multipart_threshold: int = _DEFAULT_MULTIPART_THRESHOLD,
        **cos_kwargs: Any,
    ) -> None:
        """Build the adapter.

        ``store`` may be injected directly (any ``data_access`` ObjectStore). If
        ``store`` is None and ``root`` is given we build a :class:`LocalObjectStore`;
        if ``bucket`` is given we build a :class:`COSObjectStore`. At least one of
        ``store``/``root``/``bucket`` must be provided.
        """
        if store is not None:
            self.store = store
        elif root is not None:
            self.store = LocalObjectStore(Path(root))
        elif bucket is not None:
            self.store = COSObjectStore(str(bucket), **cos_kwargs)
        else:
            raise ValueError(
                "DataAccessStorageAdapter requires an injected store, a root "
                "(LocalObjectStore), or a bucket (COSObjectStore)"
            )
        self.bucket = bucket
        self.multipart_threshold = int(multipart_threshold)
        self._publisher = ObjectStoreGenerationPublisher(
            self.store,
            bucket=bucket,
            multipart_threshold=int(multipart_threshold),
        )

    # ---- security routing -------------------------------------------------

    def _route_security(self) -> None:
        """Ensure COS-backed ops run under the active execution context.

        ``data_access.cos.remote.resolve_s3_credentials`` already prefers the
        request-scoped CredentialProvider from the current execution context (a
        ContextVar). The COS store's credential-aware client builder compares its
        client identity against the freshly resolved credentials and rebuilds on any
        identity change — so calling it under an active context never reuses a
        client built for another principal.
        """
        ctx = current_execution_context()
        if ctx is not None and ctx.credential_provider is not None:
            build = getattr(self.store, "_s3", None)
            if callable(build):
                build()  # build/refresh COS client under the current context

    # ---- ArtifactStoragePort ----------------------------------------------

    def put(self, uri: str, data: bytes) -> ObjectMetadata:
        """Upload ``data`` to ``uri``; small blobs via single PUT, large via multipart."""
        self._route_security()
        key = _uri_to_key(uri)
        self._upload(key, data)
        ch = sha256_bytes(data)
        return ObjectMetadataImpl(
            content_hash=ch, size_bytes=len(data), etag=self._head_etag(key)
        )

    def get(self, uri: str) -> bytes:
        """Download the full object at ``uri`` (resolving a generation if needed)."""
        self._route_security()
        key = self._resolve_object_key(uri)
        if key is None:
            raise DataError(f"object not found: {uri!r}")
        return _read_bytes(self.store, key)

    def head(self, uri: str) -> ObjectMetadata:
        """Return object metadata (content_hash / size / etag)."""
        self._route_security()
        key = self._resolve_object_key(uri)
        if key is None:
            raise DataError(f"object not found: {uri!r}")
        meta = self.store.head_object(key)
        if meta is None:
            raise DataError(f"object not found: {uri!r}")
        data = _read_bytes(self.store, key)
        return ObjectMetadataImpl(
            content_hash=sha256_bytes(data),
            size_bytes=int(meta.get("size", len(data))),
            etag=meta.get("etag"),
        )

    def delete(self, uri: str) -> None:
        """Explicit deletion; never called on published artifacts by contract."""
        self._route_security()
        key = self._resolve_object_key(uri)
        if key is None:
            return
        self.store.delete_object(key)

    # ---- two-phase publish (durability for published artifacts) -----------

    def publish_generation(
        self,
        prefix: str,
        data: bytes,
        *,
        key: str = _PAYLOAD_KEY,
        metadata: dict[str, Any] | None = None,
    ) -> ObjectMetadata:
        """Atomically publish ``data`` under ``prefix`` (two-phase).

        Uploads ``<prefix>/<generation_id>/<key>``, writes an immutable manifest,
        verifies generation COMPLETE, then flips ``<prefix>/CURRENT.json`` last.
        Returns metadata for the *resolved* (current) payload.
        """
        self._route_security()
        gid = self._publisher.begin_generation(
            prefix, metadata={"published_at": _now_iso(), **(metadata or {})}
        )
        try:
            self._publisher.add_object(gid, key, data, metadata={"content_hash": sha256_bytes(data)})
            self._publisher.finish_generation(gid)
        except Exception:
            try:
                self._publisher.abort_generation(gid)
            except Exception:
                pass
            raise
        # HEAD-verify the resolved payload content_hash.
        resolved_key = self._resolved_payload_key(prefix, key)
        blob = _read_bytes(self.store, resolved_key)
        ch = sha256_bytes(blob)
        if ch != sha256_bytes(data):
            raise DataError(
                f"publish verification failed: prefix={prefix!r} content_hash "
                f"{ch} != expected {sha256_bytes(data)}"
            )
        head = self.store.head_object(resolved_key) or {}
        return ObjectMetadataImpl(content_hash=ch, size_bytes=len(blob), etag=head.get("etag"))

    # ---- helpers -----------------------------------------------------------

    def _upload(self, key: str, data: bytes) -> None:
        if len(data) >= self.multipart_threshold:
            upload_id = self.store.begin_multipart(key)
            try:
                chunk = self.multipart_threshold
                # S3/COS PartNumber 域 = 1..10000（P0-06）。store 的
                # ``part_index`` 是 0-based 内部索引，UploadPart 前由 store 统一
                # ``+1`` 落 1-based 并 fail-closed 越界校验——此处必须从 0 起传
                # 0-based 索引，切勿自行 +1（会 double-shift 成 PartNumber 2..N+1）。
                for idx, offset in enumerate(range(0, len(data), chunk)):
                    self.store.upload_part(upload_id, key, idx, data[offset : offset + chunk])
                self.store.complete_multipart(upload_id, key)
            except Exception:
                try:
                    self.store.abort_multipart(upload_id, key)
                except Exception:
                    pass
                raise
        else:
            self.store.put_object(key, data)

    def _head_etag(self, key: str) -> str | None:
        try:
            meta = self.store.head_object(key)
            return meta.get("etag") if meta else None
        except Exception:
            return None

    def _resolve_object_key(self, uri: str) -> str | None:
        """Map a uri to a concrete object key.

        If ``<key>/CURRENT.json`` exists the uri is a two-phase publish prefix: we
        resolve the current generation and return the payload object key. Otherwise
        treat the uri path as a direct object key.
        """
        key = _uri_to_key(uri)
        current_key = f"{key}/{_CURRENT_JSON}"
        try:
            head = self.store.head_object(current_key)
        except Exception:
            head = None
        if head is not None:
            return self._resolved_payload_key(key, _PAYLOAD_KEY)
        try:
            head_direct = self.store.head_object(key)
        except Exception:
            head_direct = None
        if head_direct is not None:
            return key
        return None

    def _resolved_payload_key(self, prefix: str, key: str) -> str:
        gid = self._publisher.resolve_current(prefix)
        if not gid:
            raise DataError(f"no CURRENT generation for prefix {prefix!r}")
        return f"{_uri_to_key(prefix)}/{gid}/{key}"


_CURRENT_JSON = "CURRENT.json"


def _uri_to_key(uri: str) -> str:
    """Extract a POSIX-relative object key from a uri (scheme stripped)."""
    if "://" in uri:
        uri = uri.split("://", 1)[1]
    # drop any authority/bucket portion up to the first path segment
    key = uri.lstrip("/")
    return key


def _read_bytes(store: ObjectStore, key: str) -> bytes:
    """Read the full object bytes via open_reader or range_read."""
    reader = store.open_reader(key)
    if reader is not None:
        try:
            return reader.read()
        finally:
            try:
                reader.close()
            except Exception:
                pass
    head = store.head_object(key)
    size = head.get("size") if head else None
    if size is None:
        raise DataError(f"object {key!r} has no size and no reader; cannot download")
    return store.range_read(key, offset=0, length=int(size))


# ---------------------------------------------------------------------------
# publisher
# ---------------------------------------------------------------------------


class ArtifactPublisherImpl:
    """Implements ``ArtifactPublisher`` over :class:`DataAccessStorageAdapter`.

    ``publish`` computes the sha256 ``content_hash`` from the actual bytes, validates
    the artifact's semantic fields, publishes via the two-phase path, HEAD-verifies,
    and returns the resolved ``ArtifactRef``. Fails closed on any hash mismatch.
    """

    def __init__(self, storage: DataAccessStorageAdapter) -> None:
        self.storage = storage

    def publish(self, artifact: ArtifactRef, data: bytes) -> ArtifactRef:
        actual = sha256_bytes(data)

        # Validate the artifact's semantic fields (shape + scheme + type + hash fmt).
        # A caller-supplied content_hash, if present, must match the actual bytes —
        # otherwise fail closed.
        if artifact.content_hash and artifact.content_hash != actual:
            raise ValueError(
                "publish content_hash mismatch: ArtifactRef reports "
                f"{artifact.content_hash!r} but bytes hash to {actual!r}"
            )

        # Re-validate field shape by constructing a resolved ref (raises on bad fields).
        resolved = replace(
            artifact, content_hash=actual, size_bytes=len(data)
        )

        prefix = _uri_to_key(artifact.storage_uri)
        meta = self.storage.publish_generation(
            prefix,
            data,
            metadata={
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "schema_version": artifact.schema_version,
            },
        )

        # HEAD-verify the published object's content_hash matches the bytes.
        head = self.storage.head(artifact.storage_uri)
        if head.content_hash != actual:
            raise DataError(
                "publish HEAD-verify failed: stored content_hash "
                f"{head.content_hash!r} != actual {actual!r}"
            )

        return replace(
            resolved,
            content_hash=actual,
            size_bytes=len(data),
        )


# ---------------------------------------------------------------------------
# resolver
# ---------------------------------------------------------------------------


class ArtifactResolverImpl:
    """Implements ``ArtifactResolver``. Downloads + verifies, fails closed."""

    def __init__(self, storage: DataAccessStorageAdapter) -> None:
        self.storage = storage

    def open(self, artifact: ArtifactRef) -> bytes:
        data = self.storage.get(artifact.storage_uri)
        if sha256_bytes(data) != artifact.content_hash:
            raise DataError(
                "resolver open failed: content_hash mismatch for "
                f"{artifact.storage_uri!r} (expected {artifact.content_hash!r})"
            )
        return data

    def verify(self, artifact: ArtifactRef, data: bytes) -> bool:
        return sha256_bytes(data) == artifact.content_hash


# ---------------------------------------------------------------------------
# local on-disk LRU cache
# ---------------------------------------------------------------------------


class LocalArtifactCacheImpl:
    """Real on-disk LRU cache keyed by content_hash (implements LocalArtifactCache).

    Files live under ``root`` (one file per content_hash). ``max_bytes`` triggers
    LRU eviction. ``store`` validates the checksum before writing (fail closed).
    """

    eviction_policy = CacheEvictionPolicy.LRU

    def __init__(self, root: str | Path, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        # content_hash -> (size, last_access_ns); insertion/access order == LRU order
        self._index: "OrderedDict[str, tuple[int, int]]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._reindex()

    def _reindex(self) -> None:
        """Scan existing files into the LRU index (in file-mtime order)."""
        items: list[tuple[int, Path, int]] = []
        for f in self.root.iterdir():
            if f.is_file():
                try:
                    st = f.stat()
                    items.append((st.st_mtime_ns, f, st.st_size))
                except OSError:
                    continue
        items.sort(key=lambda t: t[0])
        for _, path, size in items:
            if len(path.name) == 64 and all(ch in "0123456789abcdef" for ch in path.name):
                self._index[path.name] = (int(size), int(path.stat().st_mtime_ns))

    def lookup(self, content_hash: str) -> bytes | None:
        with self._lock:
            if content_hash not in self._index:
                self._misses += 1
                return None
            path = self.root / content_hash
            if not path.exists():
                self._index.pop(content_hash, None)
                self._misses += 1
                return None
            self._hits += 1
            data = path.read_bytes()
            size = len(data)
            self._index.move_to_end(content_hash)
            self._index[content_hash] = (size, time.time_ns())
            if sha256_bytes(data) != content_hash:
                # corrupt cache entry → treat as miss
                self._hits -= 1
                self._misses += 1
                self._index.pop(content_hash, None)
                try:
                    path.unlink()
                except OSError:
                    pass
                return None
            return data

    def store(self, content_hash: str, data: bytes) -> None:
        if sha256_bytes(data) != content_hash:
            raise ValueError(
                "cache store failed: data content_hash "
                f"{sha256_bytes(data)!r} != key {content_hash!r}"
            )
        with self._lock:
            path = self.root / content_hash
            path.write_bytes(data)
            size = len(data)
            if content_hash in self._index:
                self._index.pop(content_hash)
            self._index[content_hash] = (size, time.time_ns())
            self._evict_locked()

    def hit_rate(self) -> float:
        with self._lock:
            total = self._hits + self._misses
            return float(self._hits / total) if total else 0.0

    def _evict_locked(self) -> None:
        while self._total_bytes_locked() > self.max_bytes and self._index:
            oldest_hash, _ = next(iter(self._index.items()))
            self._index.pop(oldest_hash, None)
            try:
                (self.root / oldest_hash).unlink()
            except OSError:
                pass

    def _total_bytes_locked(self) -> int:
        return sum(sz for sz, _ in self._index.values())
