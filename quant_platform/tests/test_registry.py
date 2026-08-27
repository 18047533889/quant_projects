"""QRP-P2 — ArtifactRegistry tests (memory + sqlite unified interface).

Cover:
(a) register → resolve by content_hash round-trip (memory mode);
(b) idempotency: same (content_hash, artifact_type) re-register returns the
    existing entry (no new row, no version bump);
(c) unknown artifact type is rejected;
(d) version listing: list_versions returns all versions in registration order;
(e) persistent sqlite mode round-trip (same interface, survives reopen);
(f) same bytes under a different artifact_type are distinct entries;
(g) resolve/contains on a miss return None/False.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.app.contracts import (
    ARTIFACT_TYPE_BACKTEST,
    ARTIFACT_TYPE_FACTOR_CANDIDATE,
    ArtifactRef,
)
from quant_platform.app.storage.registry import (
    ArtifactRegistry,
    UnknownArtifactTypeError,
)

import hashlib


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ref(
    data: bytes,
    *,
    artifact_id: str = "art-1",
    artifact_type: str = ARTIFACT_TYPE_FACTOR_CANDIDATE,
    uri: str = "cos://artifacts/fc/1",
    version_created: str | None = None,
    **kw,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        schema_version=kw.pop("schema_version", "1.0"),
        content_hash=sha256_bytes(data),
        storage_uri=kw.pop("storage_uri", uri),
        size_bytes=len(data),
        created_at=kw.pop(
            "created_at",
            datetime.fromisoformat(version_created) if version_created
            else datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        producer_type=kw.pop("producer_type", "test"),
        producer_version=kw.pop("producer_version", "0.1"),
        **kw,
    )


# ---- (a) register → resolve round-trip ------------------------------------
def test_register_resolve_roundtrip_memory():
    reg = ArtifactRegistry()
    try:
        data = b"payload-a"
        ref = _ref(data)
        stored = reg.register(ref)
        assert stored.content_hash == sha256_bytes(data)
        resolved = reg.resolve(sha256_bytes(data))
        assert resolved is not None
        assert resolved.artifact_id == ref.artifact_id
        assert resolved.artifact_type == ARTIFACT_TYPE_FACTOR_CANDIDATE
        assert resolved.size_bytes == len(data)
        assert reg.contains(sha256_bytes(data)) is True
    finally:
        reg.close()


def test_resolve_and_contains_on_miss():
    reg = ArtifactRegistry()
    try:
        assert reg.resolve(sha256_bytes(b"never-registered")) is None
        assert reg.contains(sha256_bytes(b"never-registered")) is False
    finally:
        reg.close()


# ---- (b) idempotency ------------------------------------------------------
def test_register_idempotent_same_content_hash_same_type():
    reg = ArtifactRegistry()
    try:
        data = b"idempotent-payload"
        ref = _ref(data, artifact_id="art-idem")
        first = reg.register(ref)
        second = reg.register(ref)  # same content_hash + same type
        assert second == first
        assert second.content_hash == first.content_hash
        # only one entry exists
        assert len(reg.list_versions("art-idem")) == 1
        # no version bump on duplicate
        assert reg.resolve(sha256_bytes(data)) == first
    finally:
        reg.close()


# ---- (c) unknown artifact type rejected -----------------------------------
def test_register_unknown_artifact_type_rejected():
    # ArtifactRef.__post_init__ rejects unknown types at construction, so build
    # the object past the DTO guard to exercise the *registry*'s own defensive
    # unknown-type check (the behavior QRP-P2 requires at the registry boundary).
    bad = object.__new__(ArtifactRef)
    for _field_name, _field in ArtifactRef.__dataclass_fields__.items():
        object.__setattr__(bad, _field_name, _field.default)
    object.__setattr__(bad, "artifact_id", "bad")
    object.__setattr__(bad, "artifact_type", "NOT_A_REAL_TYPE")
    object.__setattr__(bad, "schema_version", "1.0")
    object.__setattr__(bad, "content_hash", sha256_bytes(b"x"))
    object.__setattr__(bad, "storage_uri", "cos://x/y")
    object.__setattr__(bad, "size_bytes", 1)
    object.__setattr__(bad, "created_at", datetime(2026, 1, 1, tzinfo=timezone.utc))
    object.__setattr__(bad, "producer_type", "test")
    object.__setattr__(bad, "producer_version", "0.1")
    reg = ArtifactRegistry()
    try:
        with pytest.raises(UnknownArtifactTypeError):
            reg.register(bad)
        # nothing got registered by the rejected call
        assert reg.contains(sha256_bytes(b"x")) is False
    finally:
        reg.close()


# ---- (d) version listing --------------------------------------------------
def test_list_versions_registration_order():
    reg = ArtifactRegistry()
    try:
        v1 = _ref(b"v1-bytes", artifact_id="art-vers", uri="cos://artifacts/fc/v1")
        v2 = _ref(b"v2-bytes", artifact_id="art-vers", uri="cos://artifacts/fc/v2")
        reg.register(v1)
        reg.register(v2)
        versions = reg.list_versions("art-vers")
        assert [v.content_hash for v in versions] == [
            sha256_bytes(b"v1-bytes"),
            sha256_bytes(b"v2-bytes"),
        ]
        # second register of v2's bytes must be idempotent (no third version)
        reg.register(v2)
        assert len(reg.list_versions("art-vers")) == 2
        # unknown artifact_id → empty list
        assert reg.list_versions("no-such-id") == []
    finally:
        reg.close()


# ---- (e) persistent sqlite mode -------------------------------------------
def test_persistent_sqlite_mode_survives_reopen(tmp_path):
    db = tmp_path / "registry.sqlite"
    reg1 = ArtifactRegistry(db)
    data = b"persisted-payload"
    ref = _ref(data, artifact_id="art-persist")
    reg1.register(ref)
    reg1.close()

    reg2 = ArtifactRegistry(db)  # reopen same file
    try:
        resolved = reg2.resolve(sha256_bytes(data))
        assert resolved is not None
        assert resolved.artifact_id == "art-persist"
        # duplicate register after reopen is still idempotent
        again = reg2.register(ref)
        assert again == resolved
        assert len(reg2.list_versions("art-persist")) == 1
    finally:
        reg2.close()


# ---- (f) same bytes, different type = distinct entries --------------------
def test_same_bytes_different_type_are_distinct():
    reg = ArtifactRegistry()
    try:
        data = b"shared-bytes"
        ref_fc = _ref(data, artifact_type=ARTIFACT_TYPE_FACTOR_CANDIDATE)
        ref_bt = _ref(
            data,
            artifact_type=ARTIFACT_TYPE_BACKTEST,
            artifact_id="art-bt",
            uri="cos://artifacts/bt/1",
        )
        fc = reg.register(ref_fc)
        bt = reg.register(ref_bt)
        assert fc.artifact_type == ARTIFACT_TYPE_FACTOR_CANDIDATE
        assert bt.artifact_type == ARTIFACT_TYPE_BACKTEST
        # same hash resolves deterministically to the first registered entry
        assert reg.resolve(sha256_bytes(data)).artifact_type == ARTIFACT_TYPE_FACTOR_CANDIDATE
    finally:
        reg.close()