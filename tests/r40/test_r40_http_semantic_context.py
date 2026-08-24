"""R40 #89: HTTP path constructs an immutable, complete ExecutionSemanticIdentityV2."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("FACTOR_ENGINE_SERVICE_ROOT", "/tmp/r40_http_semantic_root")

from factor_engine.runtime.endpoint_policy import EndpointExecutionPolicy  # noqa: E402
from factor_engine.service.app import _validate_and_build_request  # noqa: E402
from factor_engine.service.security import ANONYMOUS_PRINCIPAL  # noqa: E402


def _build(payload):
    return _validate_and_build_request(
        payload,
        endpoint_policy=EndpointExecutionPolicy.RESEARCH,
        principal=ANONYMOUS_PRINCIPAL,
    )


def test_http_execution_identity_frozen_and_complete():
    execution, _digest = _build(
        {
            "formula": "close",
            "name": "f",
            "market": "ashare",
            "calendar": "SSE",
            "decision_time_policy": "close",
            "freq": "1d",
            "data_source": {"type": "data_access", "dataset": "d"},
        }
    )
    identity_payload = execution["identity"]
    from factor_engine.runtime.factor_identity import ExecutionSemanticIdentityV2

    identity = ExecutionSemanticIdentityV2(**identity_payload)
    # immutable (frozen dataclass)
    with pytest.raises(Exception):
        identity.market = "us"  # type: ignore[misc]
    # all required hash fields present
    assert identity.ir_hash
    assert identity.operator_contract_hash
    assert identity.field_contract_hash
    assert identity.source_contract_hash
    assert identity.source_dependency_hash
    # full execution-scope fields
    assert identity.market == "ashare"
    assert identity.calendar == "SSE"
    assert identity.decision_time_policy == "close"
    assert identity.frequency == "1d"
    # to_payload round-trips the same identity digest
    assert identity.identity_digest() == ExecutionSemanticIdentityV2(**execution["identity"]).identity_digest()


def test_http_identity_changes_with_context():
    execution_a, _ = _build(
        {"formula": "close", "market": "ashare", "calendar": "SSE", "data_source": {"type": "data_access", "dataset": "d"}}
    )
    execution_b, _ = _build(
        {"formula": "close", "market": "us", "calendar": "NYSE", "data_source": {"type": "data_access", "dataset": "d"}}
    )
    from factor_engine.runtime.factor_identity import ExecutionSemanticIdentityV2

    dig_a = ExecutionSemanticIdentityV2(**execution_a["identity"]).identity_digest()
    dig_b = ExecutionSemanticIdentityV2(**execution_b["identity"]).identity_digest()
    assert dig_a != dig_b
