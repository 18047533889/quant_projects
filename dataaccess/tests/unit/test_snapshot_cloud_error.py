"""Focused fake-cloud tests for typed snapshot failure propagation."""
from __future__ import annotations

import socket

import pytest

from data_access.core.exceptions import SourceSnapshotUnavailable
from data_access.snapshot.cloud_error import CloudErrorClassifier, CloudErrorKind
from data_access.snapshot.resolver import SourceSnapshotResolver


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
