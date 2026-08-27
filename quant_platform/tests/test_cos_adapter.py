"""QRP-P2 — COS ArtifactPublisher tests (monkeypatched fake COS client).

The publisher uses a duck-typed injectable store (``put_object`` /
``head_object``), so the tests drive it with an in-memory FakeCosStore — no real
credentials or network required. ``sha256_bytes`` matches the platform behavior.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from quant_platform.app.contracts import ArtifactRef
from quant_platform.app.storage.cos_adapter import (
    COSArtifactPublisher,
    COS_AVAILABLE,
    sha256_bytes,
)


def _ref(data: bytes, uri: str = "cos://artifacts/fc/1", **kw) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=kw.pop("artifact_id", "a1"),
        artifact_type=kw.pop("artifact_type", "FACTOR_CANDIDATE"),
        schema_version=kw.pop("schema_version", "1.0"),
        content_hash=kw.pop("content_hash", sha256_bytes(data)),
        storage_uri=uri,
        size_bytes=kw.pop("size_bytes", len(data)),
        created_at=kw.pop("created_at", datetime.now(timezone.utc)),
        producer_type=kw.pop("producer_type", "test"),
        producer_version=kw.pop("producer_version", "0.1"),
        **kw,
    )


class FakeCosStore:
    """Duck-typed fake COS client: put_object/head_object with a size key."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    def head_object(self, key: str) -> dict | None:
        if key not in self.objects:
            return None
        return {"size": len(self.objects[key]), "etag": None}


# ---- (a) publish → resolved ref round-trip ---------------------------------
def test_publish_roundtrip_resolves_ref():
    store = FakeCosStore()
    publisher = COSArtifactPublisher(store=store)

    data = b"cos payload"
    ref = _ref(data, uri="cos://artifacts/fc/roundtrip")
    published = publisher.publish(ref, data)

    assert published.content_hash == sha256_bytes(data)
    assert published.size_bytes == len(data)
    assert published.storage_uri == ref.storage_uri
    # bytes landed in the fake COS under the uri-derived key
    assert store.objects["artifacts/fc/roundtrip"] == data


# ---- (b) content_hash computed at this layer -------------------------------
def test_content_hash_computed_at_this_layer():
    store = FakeCosStore()
    publisher = COSArtifactPublisher(store=store)

    data = b"hash-from-bytes"
    # Caller supplies the "true" hash for the bytes (the DTO requires one);
    # the publisher still recomputes it at this layer and must agree.
    ref = _ref(data)
    published = publisher.publish(ref, data)
    assert published.content_hash == sha256_bytes(data)


# ---- (c) fail closed on caller-supplied hash mismatch ----------------------
def test_publish_hash_mismatch_fails_closed():
    store = FakeCosStore()
    publisher = COSArtifactPublisher(store=store)

    data = b"real bytes"
    wrong = _ref(data, content_hash=sha256_bytes(b"other"))
    with pytest.raises(ValueError):
        publisher.publish(wrong, data)
    # nothing was uploaded
    assert store.objects == {}


# ---- (d) HEAD-verify failure raises ----------------------------------------
def test_publish_head_verify_failure_raises():
    class HeadlessStore:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def put_object(self, key: str, data: bytes) -> None:
            self.objects[key] = data

        def head_object(self, key: str) -> dict | None:
            # HEAD reports a size that disagrees with the stored object.
            return {"size": 999, "etag": None}

    publisher = COSArtifactPublisher(store=HeadlessStore())
    data = b"payload"
    with pytest.raises(RuntimeError):
        publisher.publish(_ref(data), data)


# ---- (e) absent HEAD (object not visible) is a verification failure --------
def test_publish_head_missing_raises():
    class NoHeadStore:
        objects: dict[str, bytes] = {}

        def put_object(self, key: str, data: bytes) -> None:
            self.objects[key] = data

        def head_object(self, key: str) -> dict | None:
            return None  # object not visible post-PUT

    publisher = COSArtifactPublisher(store=NoHeadStore())
    with pytest.raises(RuntimeError):
        publisher.publish(_ref(b"data"), b"data")


# ---- (f) module loads cleanly with data_access present ----------------------
def test_cos_available_since_data_access_importable():
    # In this workspace data_access is importable and COS object-store semantics
    # are delegated to it (single ObjectStore authority, QRP-P0R-C6 reference).
    assert COS_AVAILABLE is True