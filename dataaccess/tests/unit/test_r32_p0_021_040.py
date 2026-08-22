"""R32 P0-021 through P0-040 tests."""
import json
import pytest
from datetime import datetime, timezone, timedelta


def test_p0_022_production_forbid_skip_startup_gate():
    """R32-P0-022: Production mode cannot skip startup gate."""
    from data_access.runtime.startup_gate import run_startup_gate

    # Mock store
    class MockStore:
        pass

    store = MockStore()

    # Production mode with skip=True should raise
    with pytest.raises(RuntimeError, match="R32-P0-022.*cannot skip startup gate"):
        run_startup_gate(store, production=True, skip=True)

    # Research mode with skip=True should succeed
    result = run_startup_gate(store, production=False, skip=True)
    assert result.passed
    assert "startup gate skipped" in result.problems


def test_p0_029_strict_manifest_must_have_explicit_complete():
    """R32-P0-029: Strict publisher manifest must explicitly complete=true."""
    from dataaccess.snapshot.resolver import parse_source_manifest
    from data_access.core.exceptions import SourceSnapshotUnavailable

    # Missing complete field in strict mode
    manifest_no_complete = {
        "source_generation": "g1",
        "manifest_version": "1.0",
        "dataset": "test",
        "objects": [],
        "object_count": 0,
        "content_digest": "abc",
        "prefix": "s3://bucket/path",
        "published_at": "2024-01-01T00:00:00+00:00",
    }

    with pytest.raises(SourceSnapshotUnavailable, match="缺少 complete 字段"):
        parse_source_manifest(manifest_no_complete, strict=True)

    # complete not a bool
    manifest_bad_complete = {**manifest_no_complete, "complete": "yes"}
    with pytest.raises(SourceSnapshotUnavailable, match="complete 必须是 bool"):
        parse_source_manifest(manifest_bad_complete, strict=True)

    # Explicit complete=true should work
    manifest_ok = {**manifest_no_complete, "complete": True}
    result = parse_source_manifest(manifest_ok, strict=True, expected_dataset="test")
    assert result is not None
    assert result.complete is True


def test_p0_030_strict_manifest_must_have_objects_field():
    """R32-P0-030: Strict publisher manifest must explicitly have objects field."""
    from dataaccess.snapshot.resolver import parse_source_manifest
    from data_access.core.exceptions import SourceSnapshotUnavailable

    # Missing objects field in strict mode
    manifest_no_objects = {
        "source_generation": "g1",
        "manifest_version": "1.0",
        "dataset": "test",
        "complete": True,
        "object_count": 0,
        "content_digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "prefix": "s3://bucket/path",
        "published_at": "2024-01-01T00:00:00+00:00",
    }

    with pytest.raises(SourceSnapshotUnavailable, match="缺少 objects 字段"):
        parse_source_manifest(manifest_no_objects, strict=True, expected_dataset="test")

    # Empty objects=[] should work
    manifest_ok = {**manifest_no_objects, "objects": []}
    result = parse_source_manifest(manifest_ok, strict=True, expected_dataset="test")
    assert result is not None
    assert len(result.objects) == 0


def test_p0_032_manifest_zero_byte_object_not_eaten():
    """R32-P0-032: Manifest 0-byte object not eaten by falsy operation."""
    from dataaccess.snapshot.resolver import parse_source_manifest

    # 0-byte object with explicit size=0
    manifest = {
        "source_generation": "g1",
        "manifest_version": "1.0",
        "dataset": "test",
        "complete": True,
        "objects": [
            {"key": "s3://bucket/file.txt", "size": 0, "etag": "d41d8cd98f00b204e9800998ecf8427e"}
        ],
        "object_count": 1,
        "content_digest": "abc",
        "prefix": "s3://bucket/",
        "published_at": "2024-01-01T00:00:00+00:00",
    }

    result = parse_source_manifest(manifest, strict=False, expected_dataset="test")
    assert result is not None
    assert len(result.objects) == 1
    # R32-P0-032: size=0 must be preserved, not treated as None
    assert result.objects[0].content_length == 0


def test_p0_033_manifest_version_gate_compatibility():
    """R32-P0-033: Manifest schema version must truly gate compatibility."""
    from dataaccess.snapshot.resolver import parse_source_manifest
    from data_access.core.exceptions import SourceSnapshotUnavailable

    base_manifest = {
        "source_generation": "g1",
        "dataset": "test",
        "complete": True,
        "objects": [],
        "object_count": 0,
        "content_digest": "abc",
        "prefix": "s3://bucket/path",
        "published_at": "2024-01-01T00:00:00+00:00",
    }

    # Unknown future version should fail
    manifest_future = {**base_manifest, "manifest_version": "99.0"}
    with pytest.raises(SourceSnapshotUnavailable, match="manifest_version.*不支持"):
        parse_source_manifest(manifest_future, strict=True, expected_dataset="test")

    # Supported versions should work
    for version in ["1.0", "1.1", "1"]:
        manifest_ok = {**base_manifest, "manifest_version": version}
        result = parse_source_manifest(manifest_ok, strict=True, expected_dataset="test")
        assert result is not None


def test_p0_034_published_at_timezone_aware():
    """R32-P0-034: published_at must be strict timezone-aware timestamp."""
    from dataaccess.snapshot.resolver import parse_source_manifest
    from data_access.core.exceptions import SourceSnapshotUnavailable

    base_manifest = {
        "source_generation": "g1",
        "manifest_version": "1.0",
        "dataset": "test",
        "complete": True,
        "objects": [],
        "object_count": 0,
        "content_digest": "abc",
        "prefix": "s3://bucket/path",
    }

    # Missing timezone should fail
    manifest_no_tz = {**base_manifest, "published_at": "2024-01-01T00:00:00"}
    with pytest.raises(SourceSnapshotUnavailable, match="缺少时区信息"):
        parse_source_manifest(manifest_no_tz, strict=True, expected_dataset="test")

    # Invalid ISO format
    manifest_bad_format = {**base_manifest, "published_at": "2024/01/01 00:00:00"}
    with pytest.raises(SourceSnapshotUnavailable, match="不是合法 ISO 8601"):
        parse_source_manifest(manifest_bad_format, strict=True, expected_dataset="test")

    # Excessive future skew (> 1h) - using 2h to be safe
    future_time = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    manifest_future = {**base_manifest, "published_at": future_time}
    with pytest.raises(SourceSnapshotUnavailable, match="clock skew"):
        parse_source_manifest(manifest_future, strict=True, expected_dataset="test")

    # Valid timezone-aware timestamp
    manifest_ok = {**base_manifest, "published_at": "2024-01-01T00:00:00+00:00"}
    result = parse_source_manifest(manifest_ok, strict=True, expected_dataset="test")
    assert result is not None
    assert result.published_at == "2024-01-01T00:00:00+00:00"


def test_p0_039_object_identity():
    """R32-P0-039: ObjectIdentity formally model checksum/content hash."""
    from dataaccess.snapshot.source_snapshot import ObjectIdentity, ResolvedObject

    # Create ObjectIdentity with etag
    identity = ObjectIdentity(
        kind="etag",
        value="abc123",
        size=1024,
    )
    assert identity.kind == "etag"
    assert identity.value == "abc123"
    assert identity.size == 1024

    # ResolvedObject with etag should extract identity
    obj = ResolvedObject(
        uri="s3://bucket/file.txt",
        etag="def456",
        content_length=2048,
    )
    extracted = obj.identity
    assert extracted is not None
    assert extracted.kind == "etag"
    assert extracted.value == "def456"
    assert extracted.size == 2048

    # ResolvedObject with version_id
    obj_version = ResolvedObject(
        uri="s3://bucket/file.txt",
        version_id="v123",
        content_length=512,
    )
    extracted_v = obj_version.identity
    assert extracted_v is not None
    assert extracted_v.kind == "version_id"
    assert extracted_v.value == "v123"

    # ResolvedObject with mtime_ns (local file)
    obj_local = ResolvedObject(
        uri="/path/to/file.txt",
        mtime_ns=1234567890000000000,
        content_length=100,
    )
    extracted_local = obj_local.identity
    assert extracted_local is not None
    assert extracted_local.kind == "mtime_ns"
    assert extracted_local.value == "1234567890000000000"

    # ResolvedObject with checksum (highest priority)
    obj_checksum = ResolvedObject(
        uri="/path/to/file.txt",
        checksum="abc123",
        checksum_algorithm="sha256",
        content_length=200,
    )
    extracted_checksum = obj_checksum.identity
    assert extracted_checksum is not None
    assert extracted_checksum.kind == "checksum"
    assert extracted_checksum.value == "abc123"
    assert extracted_checksum.algorithm == "sha256"


def test_p0_040_total_bytes_distinguish_unknown_vs_zero():
    """R32-P0-040: ResolvedSourceSnapshot.total_bytes distinguish unknown vs 0."""
    from dataaccess.snapshot.source_snapshot import ResolvedObject, ResolvedSourceSnapshot

    # Empty object set: total_bytes should be 0 (not None)
    snapshot_empty = ResolvedSourceSnapshot(
        dataset="test",
        objects=(),
    )
    assert snapshot_empty.total_bytes == 0

    # Known sizes: sum correctly
    snapshot_known = ResolvedSourceSnapshot(
        dataset="test",
        objects=(
            ResolvedObject(uri="s3://bucket/a", content_length=100),
            ResolvedObject(uri="s3://bucket/b", content_length=200),
        ),
    )
    assert snapshot_known.total_bytes == 300

    # Unknown size (None): total_bytes should be None
    snapshot_unknown = ResolvedSourceSnapshot(
        dataset="test",
        objects=(
            ResolvedObject(uri="s3://bucket/a", content_length=100),
            ResolvedObject(uri="s3://bucket/b", content_length=None),
        ),
    )
    assert snapshot_unknown.total_bytes is None

    # 0-byte object: should count as 0, not unknown
    snapshot_zero = ResolvedSourceSnapshot(
        dataset="test",
        objects=(
            ResolvedObject(uri="s3://bucket/a", content_length=0),
            ResolvedObject(uri="s3://bucket/b", content_length=100),
        ),
    )
    assert snapshot_zero.total_bytes == 100
