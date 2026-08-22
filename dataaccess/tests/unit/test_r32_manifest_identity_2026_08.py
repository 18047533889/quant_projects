from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from data_access.core.exceptions import SourceSnapshotUnavailable, ValidationError
from data_access.read.read_session import PhysicalResolutionContext, _ResolutionCache
from data_access.security.credentials import EnvCredentialProvider
from data_access.snapshot.resolver import parse_source_manifest
from data_access.snapshot.source_snapshot import (
    ResolvedObject,
    ResolvedSourceSnapshot,
    content_digest_of_objects,
)


def _manifest(objects=None, **overrides):
    objects = [] if objects is None else objects
    values = {
        "manifest_version": "1",
        "dataset": "d",
        "source_generation": "g1",
        "complete": True,
        "objects": objects,
        "object_count": len(objects),
        "prefix": "s3://bucket/d/",
        "published_at": "2026-08-11T00:00:00Z",
    }
    values["content_digest"] = content_digest_of_objects(
        tuple(
            ResolvedObject(
                uri=e["key"],
                content_length=(e["size"] if "size" in e else e.get("content_length")),
                etag=e.get("etag"),
            )
            for e in objects
        )
    )
    values.update(overrides)
    return values


def test_manifest_requires_complete_and_objects_explicitly():
    with pytest.raises(SourceSnapshotUnavailable):
        parse_source_manifest(_manifest(complete=None), strict=True)
    missing = _manifest()
    missing.pop("objects")
    with pytest.raises(SourceSnapshotUnavailable):
        parse_source_manifest(missing, strict=True)


def test_manifest_rejects_future_schema_and_bad_timestamp():
    with pytest.raises(SourceSnapshotUnavailable, match="不兼容"):
        parse_source_manifest(_manifest(manifest_version="99"), strict=True)
    with pytest.raises(SourceSnapshotUnavailable, match="timezone-aware"):
        parse_source_manifest(_manifest(published_at="2026-08-11T00:00:00"), strict=True)
    with pytest.raises(SourceSnapshotUnavailable, match="clock skew"):
        parse_source_manifest(
            _manifest(
                published_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            ),
            strict=True,
        )


def test_empty_and_zero_byte_identity_are_distinct_from_unknown():
    empty_digest = content_digest_of_objects(())
    assert empty_digest
    zero = ResolvedObject(uri="s3://bucket/d/zero", content_length=0, etag="e")
    unknown = ResolvedObject(uri="s3://bucket/d/unknown", content_length=None, etag="u")
    assert ResolvedSourceSnapshot(dataset="d", objects=(zero,)).total_bytes == 0
    assert ResolvedSourceSnapshot(dataset="d", objects=(unknown,)).total_bytes is None
    assert content_digest_of_objects((zero,)) != content_digest_of_objects((unknown,))


def test_local_mtime_and_checksum_change_snapshot_identity():
    base = ResolvedObject(uri="/tmp/x", content_length=4, mtime_ns=1)
    changed_mtime = ResolvedObject(uri="/tmp/x", content_length=4, mtime_ns=2)
    hashed = ResolvedObject(uri="/tmp/x", content_length=4, checksum="a", checksum_algorithm="sha256")
    assert content_digest_of_objects((base,)) != content_digest_of_objects((changed_mtime,))
    assert hashed.identity.algorithm == "sha256"


def test_env_credentials_are_family_atomic(monkeypatch):
    monkeypatch.setenv("COS_SECRET_ID", "cos-id")
    monkeypatch.delenv("COS_SECRET_KEY", raising=False)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    with pytest.raises(ValidationError, match="family"):
        EnvCredentialProvider().resolve()


def test_resolution_cache_clear_drops_context_memo():
    contexts = [PhysicalResolutionContext(mutation_generation="g1"), PhysicalResolutionContext(mutation_generation="g2")]
    cache = _ResolutionCache(context_provider=lambda _: contexts.pop(0))
    key = ("d", "p")
    cache[key] = "old"
    cache.clear()
    cache[key] = "new"
    assert cache.get(key) == "new"
    assert len(cache._memo) == 1
