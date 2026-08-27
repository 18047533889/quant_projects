"""QRP-P0R-C5 — platform storage adapter runtime tests.

Cover:
(a) publish → resolve round-trip with LocalObjectStore/tmpdir (content_hash verified);
(b) resolver.open fails closed on tampered bytes;
(c) cache hit/miss + LRU eviction + hit_rate;
(d) publish fails closed on content_hash mismatch;
(e) small vs large (multipart) routing on both the raw put and the two-phase publish paths.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data_access.read.object_store import LocalObjectStore

from quant_platform.app.adapters.data_access_storage import (
    ArtifactPublisherImpl,
    ArtifactResolverImpl,
    DataAccessStorageAdapter,
    LocalArtifactCacheImpl,
    sha256_bytes,
)
from quant_platform.app.contracts import ArtifactRef


def _ref(uri: str, data: bytes, **kw) -> ArtifactRef:
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


# ---- (a) publish → resolve round-trip --------------------------------------
def test_publish_resolve_roundtrip(tmp_path):
    root = tmp_path / "store"
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(root))
    publisher = ArtifactPublisherImpl(adapter)
    resolver = ArtifactResolverImpl(adapter)

    data = b"hello platform artifact"
    ref = _ref("cos://artifacts/fc/roundtrip", data)

    published = publisher.publish(ref, data)
    assert published.content_hash == sha256_bytes(data)
    assert published.size_bytes == len(data)

    # resolver.open downloads and verifies
    got = resolver.open(published)
    assert got == data
    assert resolver.verify(published, data) is True
    assert resolver.verify(published, b"tampered") is False


# ---- (b) resolver.open fails closed on tampered bytes ----------------------
def test_resolver_open_fails_closed_on_tampered(tmp_path):
    root = tmp_path / "store"
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(root))
    publisher = ArtifactPublisherImpl(adapter)
    resolver = ArtifactResolverImpl(adapter)

    data = b"payload to be tampered"
    ref = _ref("cos://artifacts/fc/tamper", data)
    published = publisher.publish(ref, data)

    # Tamper the stored payload object directly at the storage layer.
    from quant_platform.app.adapters.data_access_storage import _uri_to_key
    from data_access.security.execution_context import execution_scope

    key = _uri_to_key(published.storage_uri)
    resolved = adapter._resolve_object_key(published.storage_uri)
    assert resolved is not None
    adapter.store.put_object(resolved, b"EVIL TAMPERED BYTES!!")

    with pytest.raises(Exception):
        resolver.open(published)


# ---- (c) cache hit/miss + LRU eviction + hit_rate --------------------------
def test_cache_lru_and_hit_rate(tmp_path):
    cache = LocalArtifactCacheImpl(tmp_path / "cache", max_bytes=300)

    # miss then store then hit
    c1 = sha256_bytes(b"first")
    assert cache.lookup(c1) is None
    assert cache.hit_rate() == 0.0

    cache.store(c1, b"first")  # 5 bytes
    assert cache.lookup(c1) == b"first"
    assert cache.hit_rate() == 0.5  # 1 hit / 2 total

    # fill until eviction kicks in (300 max bytes): use 150-byte payloads so the
    # total of three entries (5+150+150=305) exceeds max_bytes and evicts oldest.
    c2 = sha256_bytes(b"B" * 150)
    c3 = sha256_bytes(b"C" * 150)
    cache.store(c2, b"B" * 150)
    cache.store(c3, b"C" * 150)
    # oldest ("first", 5 bytes) evicted once total exceeds max_bytes
    assert cache.lookup(c1) is None  # evicted
    assert cache.lookup(c2) == b"B" * 150


def test_cache_store_checksum_fail_closed(tmp_path):
    cache = LocalArtifactCacheImpl(tmp_path / "cache")
    with pytest.raises(ValueError):
        cache.store(sha256_bytes(b"x"), b"does-not-match")


# ---- (d) publish fails closed on content_hash mismatch ---------------------
def test_publish_hash_mismatch_raises(tmp_path):
    root = tmp_path / "store"
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(root))
    publisher = ArtifactPublisherImpl(adapter)

    data = b"real bytes"
    wrong_ref = _ref("cos://artifacts/fc/mismatch", data, content_hash=sha256_bytes(b"other"))
    with pytest.raises(ValueError):
        publisher.publish(wrong_ref, data)


# ---- (e) small vs large (multipart) routing ---------------------------------
def test_raw_put_small_vs_large_multipart(tmp_path):
    root = tmp_path / "store"
    adapter = DataAccessStorageAdapter(
        store=LocalObjectStore(root), multipart_threshold=16
    )
    small = b"tiny"
    large = b"x" * 100  # > 16 bytes → multipart

    small_meta = adapter.put("cos://raw/small", small)
    assert small_meta.size_bytes == len(small)
    assert adapter.get("cos://raw/small") == small

    large_meta = adapter.put("cos://raw/large", large)
    assert large_meta.size_bytes == len(large)
    assert adapter.get("cos://raw/large") == large
    assert adapter.head("cos://raw/large").content_hash == sha256_bytes(large)


def test_publish_small_vs_large_multipart(tmp_path):
    root = tmp_path / "store"
    adapter = DataAccessStorageAdapter(
        store=LocalObjectStore(root), multipart_threshold=16
    )
    publisher = ArtifactPublisherImpl(adapter)
    resolver = ArtifactResolverImpl(adapter)

    large = b"y" * 200
    ref = _ref("cos://artifacts/fc/large", large)
    published = publisher.publish(ref, large)
    assert resolver.open(published) == large
    assert published.content_hash == sha256_bytes(large)
