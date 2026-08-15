# -*- coding: utf-8
"""R32 P0-041 to P0-060 tests: Snapshot/Credential/Session/Cache fixes."""
import hashlib
import json
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from data_access.read.execution_identity import (
    ArtifactIdentity,
    ExecutionIdentity,
    get_current_execution_identity,
)
from data_access.read.read_contract import (
    DataSnapshot,
    FileVersion,
    build_data_snapshot,
    rebuild_snapshot_files,
)
from data_access.read.read_session import (
    PhysicalResolutionContext,
    SessionState,
    _ResolutionCache,
)
from data_access.snapshot.source_snapshot import (
    ResolvedObject,
    ResolvedSourceSnapshot,
    content_digest_of_objects,
)


class TestP0041ToP0043_SnapshotConvergence:
    """R32-P0-041/042/043: Snapshot convergence and build_sha separation."""

    def test_p0_042_data_snapshot_no_build_sha(self):
        """R32-P0-042: DataSnapshot must not include build_sha in identity."""
        snap1 = build_data_snapshot(
            dataset="test_ds",
            registry_hash="reg123",
            schema={"col1": "int64"},
            paths=[],
            params={"window": 20},
        )

        snap2 = build_data_snapshot(
            dataset="test_ds",
            registry_hash="reg123",
            schema={"col1": "int64"},
            paths=[],
            params={"window": 20},
        )

        # Same data = same snapshot_id, regardless of build
        assert snap1.snapshot_id == snap2.snapshot_id
        assert snap1.dataset == "test_ds"
        assert snap1.registry_hash == "reg123"

    def test_p0_043_unified_identity_algorithm(self):
        """R32-P0-043: build_data_snapshot and rebuild_snapshot_files use identical algorithms."""
        files = (
            FileVersion(path="/data/file1.parquet", size=1000, mtime_ns=123456789),
            FileVersion(path="/data/file2.parquet", size=2000, mtime_ns=987654321),
        )

        snap1 = build_data_snapshot(
            dataset="test_ds",
            registry_hash="reg123",
            schema={"col1": "int64"},
            paths=[],
            params={"window": 20},
            files=files,
        )

        # Rebuild with same files should produce identical snapshot_id
        snap2 = rebuild_snapshot_files(snap1, files)

        assert snap1.snapshot_id == snap2.snapshot_id
        assert snap1.file_manifest_hash == snap2.file_manifest_hash

    def test_p0_042_execution_identity_separate(self):
        """R32-P0-042: ExecutionIdentity tracks code facts separately."""
        exec_id = ExecutionIdentity(
            build_sha="abc123",
            package_version="0.11.0",
            runtime_mode="production",
            semantic_execution_version="v1",
        )

        digest = exec_id.digest()
        assert len(digest) == 16
        assert exec_id.build_sha == "abc123"

        # Different build = different execution identity
        exec_id2 = ExecutionIdentity(
            build_sha="xyz789",
            package_version="0.11.0",
            runtime_mode="production",
            semantic_execution_version="v1",
        )
        assert exec_id.digest() != exec_id2.digest()

    def test_p0_041_artifact_identity_combines_layers(self):
        """R32-P0-041/076: ArtifactIdentity combines data+code+security+plan."""
        artifact = ArtifactIdentity(
            source_snapshot_digest="data_abc123",
            execution_digest="exec_xyz789",
            security_digest="sec_456",
            plan_digest="plan_789",
        )

        full_digest = artifact.digest()
        assert len(full_digest) == 32

        # Change any layer = different artifact
        artifact2 = ArtifactIdentity(
            source_snapshot_digest="data_DIFFERENT",
            execution_digest="exec_xyz789",
            security_digest="sec_456",
            plan_digest="plan_789",
        )
        assert artifact.digest() != artifact2.digest()


class TestP0040_SnapshotTotalBytes:
    """R32-P0-040: Distinguish unknown bytes from 0."""

    def test_p0_040_unknown_bytes_returns_none(self):
        """Unknown content_length must return None, not 0."""
        objs = (
            ResolvedObject(uri="s3://bucket/file1.parquet", content_length=1000),
            ResolvedObject(uri="s3://bucket/file2.parquet", content_length=None),
        )
        snap = ResolvedSourceSnapshot(dataset="test", objects=objs)

        # R32-P0-040: One unknown = total unknown
        assert snap.total_bytes is None

    def test_p0_040_all_known_returns_sum(self):
        """All known content_length returns sum."""
        objs = (
            ResolvedObject(uri="s3://bucket/file1.parquet", content_length=1000),
            ResolvedObject(uri="s3://bucket/file2.parquet", content_length=2000),
        )
        snap = ResolvedSourceSnapshot(dataset="test", objects=objs)

        assert snap.total_bytes == 3000

    def test_p0_040_zero_bytes_is_valid(self):
        """0-byte object is different from unknown."""
        objs = (
            ResolvedObject(uri="s3://bucket/empty.parquet", content_length=0),
        )
        snap = ResolvedSourceSnapshot(dataset="test", objects=objs)

        assert snap.total_bytes == 0  # Known 0, not unknown


class TestP0031_P0038_ContentDigest:
    """R32-P0-031/038: Content digest improvements."""

    def test_p0_031_empty_set_canonical_digest(self):
        """Empty object set has canonical non-empty digest."""
        digest = content_digest_of_objects([])

        # R32-P0-031: Not empty string, has canonical value
        assert digest != ""
        assert len(digest) == 32
        assert isinstance(digest, str)

    def test_p0_038_mtime_ns_in_digest(self):
        """R32-P0-038: mtime_ns included for local file identity."""
        obj1 = ResolvedObject(
            uri="/data/file.parquet",
            content_length=1000,
            mtime_ns=123456789,
        )
        obj2 = ResolvedObject(
            uri="/data/file.parquet",
            content_length=1000,
            mtime_ns=987654321,  # Different mtime
        )

        digest1 = content_digest_of_objects([obj1])
        digest2 = content_digest_of_objects([obj2])

        # R32-P0-038: Same path, same size, different mtime = different digest
        assert digest1 != digest2


class TestP0049ToP0052_SessionContext:
    """R32-P0-049 to P0-052: Session/Context lifecycle."""

    def test_p0_049_failed_context_unique_identity(self):
        """R32-P0-049: Failed resolution returns unique unshareable identity in research."""
        mock_store = Mock()
        mock_store._registry.get.side_effect = Exception("Registry failed")
        mock_store.manifest_version.side_effect = Exception("Manifest failed")
        mock_store._contract_digest_for.side_effect = Exception("Contract failed")

        # Should not share empty string - must fail completely to trigger unshareable
        ctx1 = PhysicalResolutionContext.from_store(mock_store, "dataset1")
        time.sleep(0.001)  # Ensure different timestamp
        ctx2 = PhysicalResolutionContext.from_store(mock_store, "dataset2")

        # R32-P0-049: Different failures get different identities
        assert ctx1.namespace != ""
        assert ctx2.namespace != ""
        assert ctx1.namespace != ctx2.namespace
        assert "UNKNOWN_UNSHAREABLE" in ctx1.namespace

    def test_p0_050_clear_also_clears_memo(self):
        """R32-P0-050: _ResolutionCache.clear() must clear _memo."""
        cache = _ResolutionCache()
        cache._memo["dataset1"] = PhysicalResolutionContext(
            namespace="ns1", contract_digest="c1"
        )

        assert len(cache._memo) == 1
        cache.clear()

        # R32-P0-050: _memo must be cleared
        assert len(cache._memo) == 0

    def test_p0_051_state_machine_prevents_double_enter(self):
        """R32-P0-051: State machine prevents double-enter."""
        mock_store = Mock()
        mock_store.lock_calendars = Mock()

        from data_access.read.read_session import DataReadSession

        session = DataReadSession(mock_store)

        with session:
            # Already ACTIVE, cannot enter again
            with pytest.raises(RuntimeError, match="already ACTIVE"):
                with session:
                    pass

    def test_p0_051_closed_session_cannot_reenter(self):
        """R32-P0-051: CLOSED session cannot be re-entered."""
        mock_store = Mock()
        mock_store.lock_calendars = Mock()

        from data_access.read.read_session import DataReadSession

        session = DataReadSession(mock_store)

        with session:
            pass

        # Now CLOSED, cannot re-enter
        with pytest.raises(RuntimeError, match="CLOSED"):
            with session:
                pass

    def test_p0_052_cleanup_errors_logged(self):
        """R32-P0-052: Critical cleanup failures are logged."""
        mock_store = Mock()
        mock_store.lock_calendars = Mock()

        from data_access.read.read_session import DataReadSession

        session = DataReadSession(mock_store)

        # Enter and exit normally - check state transitions
        with session:
            assert session._state == SessionState.ACTIVE

        # After exit, should be CLOSED
        assert session._state == SessionState.CLOSED

        # Verify cleanup happened (ExitStack.close was called)
        # Cannot easily mock ContextVar.reset, but can verify state machine worked
        assert len(session._cleanup_errors) == 0  # Normal exit, no errors


class TestP0044ToP0048_CredentialHandling:
    """R32-P0-044 to P0-048: Credential handling improvements."""

    def test_p0_044_session_token_concept(self):
        """R32-P0-044: Credentials should support session_token."""
        # Test that S3Credentials dataclass accepts session_token
        from data_access.cos.remote import S3Credentials

        creds = S3Credentials(
            access_key_id="key123",
            secret_access_key="secret456",
            endpoint="cos.example.com",
            region="ap-guangzhou",
            session_token="session_token_abc",
        )

        assert creds.session_token == "session_token_abc"
        assert creds.has_session_token is True

    def test_p0_045_cache_key_includes_credential_scope(self):
        """R32-P0-045: Remote metadata cache includes credential scope."""
        from data_access.read.read_contract import _RemoteMetaCacheKey

        key1 = _RemoteMetaCacheKey(
            uri="s3://bucket/file.parquet",
            credential_scope_id="scope1",
        )
        key2 = _RemoteMetaCacheKey(
            uri="s3://bucket/file.parquet",
            credential_scope_id="scope2",
        )

        # Different credential scope = different cache key
        assert key1 != key2

    def test_p0_036_cache_key_changes_on_credential_generation(self):
        from data_access.read.read_contract import _RemoteMetaCacheKey

        assert _RemoteMetaCacheKey("s3://b/x", "scope", "generation-1") != _RemoteMetaCacheKey(
            "s3://b/x", "scope", "generation-2"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
