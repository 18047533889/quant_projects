"""QRP-P0R-C5 — artifact security classification + formula authorization + audit.

Cover:
(a) classification gate denies access below the permission's minimum classification;
(b) formula guard returns DENY (→ 403) without factor:read_formula and records an
    audit event;
(c) formula guard ALLOWs for a principal holding factor:read_formula at the
    required classification;
(d) audit event includes security_digest + credential_scope_id;
(e) materializer downloads + verifies + writes to a local path (fail closed on
    tampered bytes).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data_access.read.object_store import LocalObjectStore

from quant_platform.app.adapters.data_access_storage import (
    ArtifactPublisherImpl,
    ArtifactResolverImpl,
    DataAccessStorageAdapter,
    sha256_bytes,
)
from quant_platform.app.contracts import ArtifactRef
from quant_platform.app.contracts.rbac import (
    Permission,
    Role,
    SecurityClassification,
    Team,
)
from quant_platform.app.storage import (
    ClassificationAccessGate,
    FormulaAccessGuardImpl,
)
from quant_platform.app.storage.materializer import ArtifactMaterializerImpl


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


# ---- (a) classification gate denies below minimum --------------------------
def test_classification_gate_denies_below_minimum():
    gate = ClassificationAccessGate()

    # A CORE principal holds factor:read_formula (min CONFIDENTIAL_ALPHA).
    class P:
        principal_id = "p1"
        role = Role.CORE
        team = Team.FACTOR_TEAM

    # At/above the floor → allowed.
    assert gate.authorize(
        P(), Permission.FACTOR_READ_FORMULA, SecurityClassification.CONFIDENTIAL_ALPHA
    )
    assert gate.authorize(
        P(), Permission.FACTOR_READ_FORMULA, SecurityClassification.RESTRICTED_RAW_VALUES
    )
    # Below the floor (PUBLIC_METADATA) → denied.
    assert not gate.authorize(
        P(), Permission.FACTOR_READ_FORMULA, SecurityClassification.PUBLIC_METADATA
    )


def test_classification_gate_denies_missing_permission():
    gate = ClassificationAccessGate()

    class Member:
        principal_id = "p2"
        role = Role.MEMBER  # MEMBER lacks factor:read_formula
        team = Team.FACTOR_TEAM

    assert not gate.authorize(
        Member(), Permission.FACTOR_READ_FORMULA, SecurityClassification.CONFIDENTIAL_ALPHA
    )


# ---- (b) formula guard DENY without factor:read_formula + audit ------------
def test_formula_guard_denies_without_permission_and_audits():
    guard = FormulaAccessGuardImpl()

    class Member:
        principal_id = "p2"
        role = Role.MEMBER
        team = Team.FACTOR_TEAM

    event = guard.guard(
        principal=Member(),
        factor_id="fc-123",
        request_id="req-1",
        security_digest="digest-abc",
        credential_scope_id="scope-42",
    )
    # Denied → maps to 403, never 200 + null.
    assert event.result == "DENY"
    assert guard.is_allowed(event) is False
    # Audit event carries the required fields.
    assert event.principal_id == "p2"
    assert event.permission == "factor:read_formula"
    assert event.resource == "factor:fc-123"
    assert event.request_id == "req-1"
    assert event.security_digest == "digest-abc"
    assert event.credential_scope_id == "scope-42"
    assert event.action == "factor:read_formula"


# ---- (c) formula guard ALLOW for authorized principal ----------------------
def test_formula_guard_allows_authorized_principal():
    guard = FormulaAccessGuardImpl()

    class Core:
        principal_id = "p1"
        role = Role.CORE
        team = Team.FACTOR_TEAM

    event = guard.guard(
        principal=Core(),
        factor_id="fc-456",
        request_id="req-2",
        security_digest="digest-xyz",
        credential_scope_id="scope-7",
    )
    assert event.result == "ALLOW"
    assert guard.is_allowed(event) is True


# ---- (d) audit event includes security_digest + credential_scope_id --------
def test_audit_event_fields_present_on_deny():
    guard = FormulaAccessGuardImpl()

    class Member:
        principal_id = "p2"
        role = Role.MEMBER
        team = Team.FACTOR_TEAM

    event = guard.guard(
        principal=Member(),
        factor_id="fc-789",
        security_digest="digest-1",
        credential_scope_id="scope-9",
    )
    assert event.security_digest == "digest-1"
    assert event.credential_scope_id == "scope-9"
    assert event.timestamp is not None


# ---- (e) materializer download + verify + write -----------------------------
def test_materializer_writes_verified_bytes(tmp_path):
    root = tmp_path / "store"
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(root))
    publisher = ArtifactPublisherImpl(adapter)
    resolver = ArtifactResolverImpl(adapter)
    materializer = ArtifactMaterializerImpl(resolver)

    data = b"materialize me"
    ref = _ref("cos://artifacts/fc/mat", data)
    published = publisher.publish(ref, data)

    dest = tmp_path / "out" / "artifact.bin"
    out = materializer.materialize(published, str(dest))
    assert out == str(dest)
    assert dest.read_bytes() == data


def test_materializer_fails_closed_on_tampered(tmp_path):
    root = tmp_path / "store"
    adapter = DataAccessStorageAdapter(store=LocalObjectStore(root))
    publisher = ArtifactPublisherImpl(adapter)
    resolver = ArtifactResolverImpl(adapter)
    materializer = ArtifactMaterializerImpl(resolver)

    data = b"tamper target"
    ref = _ref("cos://artifacts/fc/mat2", data)
    published = publisher.publish(ref, data)

    # Tamper the stored payload object directly.
    from quant_platform.app.adapters.data_access_storage import _uri_to_key

    resolved = adapter._resolve_object_key(published.storage_uri)
    assert resolved is not None
    adapter.store.put_object(resolved, b"EVIL TAMPERED!!")

    dest = tmp_path / "out" / "artifact.bin"
    with pytest.raises(Exception):
        materializer.materialize(published, str(dest))
