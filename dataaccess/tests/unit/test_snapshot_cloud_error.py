"""Focused fake-cloud tests for typed snapshot failure propagation."""
from __future__ import annotations

import socket

import pytest

from data_access.core.exceptions import SourceSnapshotUnavailable
from data_access.read.read_contract import RemoteMetadataError
from data_access.snapshot.cloud_error import CloudErrorClassifier, CloudErrorKind
from data_access.snapshot.resolver import SourceSnapshotResolver
from data_access.snapshot.source_snapshot import ResolvedObject, ResolvedSourceSnapshot
from data_access.snapshot.verifier import SnapshotVerifier
from data_access.store import DataAccessStore


class ClientError(Exception):
    """Minimal botocore-compatible fake; no botocore or network required."""

    def __init__(self, response):
        self.response = response
        super().__init__(str(response))


def _client_error(code: str, status: int | None = None) -> ClientError:
    response = {"Error": {"Code": code}}
    if status is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": status}
    return ClientError(response)


@pytest.mark.parametrize(
    ("error", "kind", "retryable"),
    [
        (_client_error("NoSuchKey", 404), CloudErrorKind.NOT_FOUND, False),
        (_client_error("AccessDenied", 403), CloudErrorKind.AUTH_FAILED, False),
        (_client_error("InternalError", 500), CloudErrorKind.SERVER_ERROR, True),
        (TimeoutError("timed out"), CloudErrorKind.DEADLINE_EXCEEDED, True),
        (ConnectionRefusedError("refused"), CloudErrorKind.NETWORK_ERROR, True),
        (socket.gaierror(-2, "Name or service not known"), CloudErrorKind.NETWORK_ERROR, True),
    ],
)
def test_cloud_classifier_preserves_common_typed_outcomes(error, kind, retryable):
    classified = CloudErrorClassifier.classify(error)
    assert classified.kind is kind
    assert classified.retryable is retryable


def test_client_error_expired_token_precedes_generic_403():
    classified = CloudErrorClassifier.classify(_client_error("ExpiredToken", 403))
    assert classified.kind is CloudErrorKind.CREDENTIALS_INVALID
    assert classified.security_sensitive is True
    assert classified.original_code == "ExpiredToken"


def test_client_error_without_response_metadata_is_safe():
    classified = CloudErrorClassifier.classify(_client_error("NoSuchKey"))
    assert classified.kind is CloudErrorKind.NOT_FOUND
    assert classified.original_code == "NoSuchKey"


@pytest.mark.parametrize(
    ("error", "expected_kind"),
    [
        (_client_error("AccessDenied", 403), CloudErrorKind.AUTH_FAILED),
        (_client_error("NoSuchKey", 404), CloudErrorKind.NOT_FOUND),
        (TimeoutError("head timeout"), CloudErrorKind.DEADLINE_EXCEEDED),
        (ConnectionRefusedError("refused"), CloudErrorKind.NETWORK_ERROR),
        (_client_error("InternalError", 500), CloudErrorKind.SERVER_ERROR),
    ],
)
def test_snapshot_verifier_attaches_typed_failure_and_preserves_cause(error, expected_kind):
    snapshot = ResolvedSourceSnapshot(
        dataset="table",
        objects=(ResolvedObject("s3://bucket/table/file.parquet", etag="e1", content_length=1),),
    )

    def fail(_uri):
        raise error

    with pytest.raises(SourceSnapshotUnavailable) as raised:
        SnapshotVerifier(remote_meta_fn=fail, strict=True).verify_before_execute(snapshot)

    failure = raised.value
    assert failure.cloud_error.kind is expected_kind
    assert failure.cloud_error.retryable is (expected_kind in {
        CloudErrorKind.DEADLINE_EXCEEDED,
        CloudErrorKind.NETWORK_ERROR,
        CloudErrorKind.SERVER_ERROR,
    })
    assert failure.__cause__ is error
    assert f"kind={expected_kind.value}" in str(failure)


def test_store_remote_meta_head_surfaces_recorded_typed_failure(monkeypatch):
    from data_access.read import read_contract

    error = RemoteMetadataError("auth_failed", "remote metadata HEAD failed")
    monkeypatch.setattr(read_contract, "_remote_snapshot_meta_enabled", lambda: True)

    def fail(*_args, **kwargs):
        assert kwargs["raise_on_error"] is True
        raise error

    monkeypatch.setattr(read_contract, "_remote_object_meta", fail)

    store = object.__new__(DataAccessStore)
    with pytest.raises(RemoteMetadataError) as raised:
        store._remote_meta_head("s3://bucket/table/file.parquet")
    assert raised.value is error


def test_store_to_verifier_preserves_typed_failure(monkeypatch):
    from data_access.read import read_contract

    original = _client_error("AccessDenied", 403)
    error = RemoteMetadataError("auth_failed", "remote metadata HEAD failed", original=original)
    monkeypatch.setattr(read_contract, "_remote_snapshot_meta_enabled", lambda: True)

    def fail(*_args, **kwargs):
        assert kwargs["raise_on_error"] is True
        raise error

    monkeypatch.setattr(read_contract, "_remote_object_meta", fail)

    store = object.__new__(DataAccessStore)
    snapshot = ResolvedSourceSnapshot(
        dataset="table",
        objects=(ResolvedObject("s3://bucket/table/file.parquet", etag="e1", content_length=1),),
    )
    with pytest.raises(SourceSnapshotUnavailable) as raised:
        SnapshotVerifier(remote_meta_fn=store._remote_meta_head, strict=True).verify_before_execute(snapshot)

    failure = raised.value
    assert failure.cloud_error.kind is CloudErrorKind.AUTH_FAILED
    assert failure.__cause__ is error


def test_store_to_verifier_preserves_throttle_retry_after(monkeypatch):
    from data_access.read import read_contract

    original = _client_error("Throttling", 429)
    original.response["ResponseMetadata"]["HTTPHeaders"] = {"retry-after": "2.5"}
    error = RemoteMetadataError("throttled", "remote metadata HEAD failed", original=original)
    monkeypatch.setattr(read_contract, "_remote_snapshot_meta_enabled", lambda: True)

    def fail(*_args, **kwargs):
        assert kwargs["raise_on_error"] is True
        raise error

    monkeypatch.setattr(read_contract, "_remote_object_meta", fail)
    store = object.__new__(DataAccessStore)
    snapshot = ResolvedSourceSnapshot(
        dataset="table",
        objects=(ResolvedObject("s3://bucket/table/file.parquet", etag="e1", content_length=1),),
    )

    with pytest.raises(SourceSnapshotUnavailable) as raised:
        SnapshotVerifier(remote_meta_fn=store._remote_meta_head, strict=True).verify_before_execute(snapshot)

    assert raised.value.cloud_error.kind is CloudErrorKind.THROTTLED
    assert raised.value.cloud_error.retryable is True
    assert raised.value.cloud_error.retry_after == 2.5
    assert raised.value.__cause__ is error


def test_non_strict_store_failure_is_suppressed(monkeypatch):
    from data_access.read import read_contract

    error = RemoteMetadataError("network_error", "remote metadata HEAD failed")
    monkeypatch.setattr(read_contract, "_remote_snapshot_meta_enabled", lambda: True)
    monkeypatch.setattr(
        read_contract,
        "_remote_object_meta",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    store = object.__new__(DataAccessStore)
    snapshot = ResolvedSourceSnapshot(
        dataset="table",
        objects=(ResolvedObject("s3://bucket/table/file.parquet", etag="e1", content_length=1),),
    )

    SnapshotVerifier(remote_meta_fn=store._remote_meta_head, strict=False).verify_before_execute(snapshot)


def test_strict_absent_metadata_remains_fail_closed():
    snapshot = ResolvedSourceSnapshot(
        dataset="table",
        objects=(ResolvedObject("s3://bucket/table/file.parquet", etag="e1", content_length=1),),
    )
    with pytest.raises(SourceSnapshotUnavailable, match="HEAD 失败"):
        SnapshotVerifier(remote_meta_fn=lambda _uri: None, strict=True).verify_before_execute(snapshot)


@pytest.mark.parametrize(
    ("entrypoint", "error", "expected_kind"),
    [
        ("head", _client_error("ExpiredToken", 403), CloudErrorKind.CREDENTIALS_INVALID),
        ("list", socket.gaierror(-2, "Name or service not known"), CloudErrorKind.NETWORK_ERROR),
        ("manifest", TimeoutError("manifest timeout"), CloudErrorKind.DEADLINE_EXCEEDED),
    ],
)
def test_resolver_keeps_typed_failure_in_diagnostic_and_cause(
    entrypoint, error, expected_kind
):
    def fail(*_args):
        raise error

    kwargs = {"strict": True}
    paths = ["s3://bucket/table/file.parquet"]
    if entrypoint == "head":
        kwargs["head_object_fn"] = fail
    elif entrypoint == "list":
        kwargs["list_objects_fn"] = fail
        paths = ["s3://bucket/table/*.parquet"]
    else:
        kwargs["source_manifest_fn"] = fail

    with pytest.raises(SourceSnapshotUnavailable) as raised:
        SourceSnapshotResolver(**kwargs).resolve("table", paths=paths)

    failure = raised.value
    assert failure.cloud_error.kind is expected_kind
    assert failure.__cause__ is error
    assert f"kind={expected_kind.value}" in str(failure)
    assert "retryable=" in str(failure)
