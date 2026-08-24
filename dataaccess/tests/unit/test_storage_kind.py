"""R46: StorageKind / StorageRequirement typed storage-requirement closure."""
from __future__ import annotations

import pytest

from data_access.core.engine import DuckDBEngine, StorageKind, StorageRequirement
from data_access.core.exceptions import ValidationError


def test_storage_kind_has_expected_members():
    assert {k.value for k in StorageKind} == {"local", "s3", "gcs", "azure"}
    assert StorageKind.REMOTE_S3.value == "s3"


def test_storage_requirement_typed_fields():
    req = StorageRequirement(kind=StorageKind.REMOTE_S3, credential_scope="scope")
    assert req.kind is StorageKind.REMOTE_S3
    assert req.credential_scope == "scope"
    assert req.is_remote is True


def test_storage_requirement_normalizes_string_kind():
    req = StorageRequirement(kind="s3")  # type: ignore[arg-type]
    assert req.kind is StorageKind.REMOTE_S3


def test_storage_requirement_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        StorageRequirement(kind="cosss")  # type: ignore[arg-type]


def test_storage_requirement_is_frozen():
    req = StorageRequirement(kind=StorageKind.LOCAL)
    with pytest.raises(Exception):
        req.kind = StorageKind.REMOTE_S3  # type: ignore[misc]


def test_unimplemented_remote_kind_fails_closed(monkeypatch):
    engine = DuckDBEngine(threads=1, max_concurrency=1, enable_object_cache=False)
    req = StorageRequirement(kind=StorageKind.REMOTE_GCS)
    with pytest.raises(ValidationError, match="尚未实现"):
        engine.execute_arrow("SELECT 1 AS x", storage_requirement=req)
    engine.close()


def test_local_requirement_is_noop():
    engine = DuckDBEngine(threads=1, max_concurrency=1, enable_object_cache=False)
    table = engine.execute_arrow(
        "SELECT 1 AS x", storage_requirement=StorageRequirement(kind=StorageKind.LOCAL)
    )
    assert table.column(0)[0].as_py() == 1
    engine.close()
