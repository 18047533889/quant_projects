"""R2 DataAccess query-cache identity regression oracles."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.query_cache import _security_scope_digest, query_cache_key
from data_access.security.principal import DataPrincipal, DEFAULT_ACCESS_POLICY


def _query_key(**overrides):
    values = {
        "dataset": "prices",
        "params": {},
        "time_range": None,
        "instruments": None,
        "columns": None,
        "manifest_token": None,
    }
    values.update(overrides)
    return query_cache_key(**values)


def test_query_cache_correctness_identity_is_full_width_and_typed():
    int_key = _query_key(params={"window": 1})
    float_key = _query_key(params={"window": 1.0})

    assert len(int_key) == 64
    assert int_key != float_key
    assert int_key == _query_key(params={"window": 1})


def test_query_cache_identity_is_order_independent_for_mappings():
    assert _query_key(params={"a": 1, "b": 2}) == _query_key(
        params={"b": 2, "a": 1}
    )


def test_query_cache_identity_rejects_repr_fallback():
    with pytest.raises(ValidationError, match="repr fallback"):
        _query_key(params={"unsupported": object()})


def test_query_cache_identity_enforces_string_parameter_keys():
    with pytest.raises(ValidationError, match="mapping keys must be strings"):
        _query_key(params={1: "integer-key"})


def test_security_scope_identity_is_full_width_and_principal_scoped():
    alice = DataPrincipal(principal_id="alice", clearance="basic")
    bob = DataPrincipal(principal_id="bob", clearance="basic")

    alice_digest = _security_scope_digest(
        principal=alice, access_policy=DEFAULT_ACCESS_POLICY
    )
    bob_digest = _security_scope_digest(
        principal=bob, access_policy=DEFAULT_ACCESS_POLICY
    )

    assert len(alice_digest) == 64
    assert alice_digest != bob_digest


def test_query_cache_identity_rejects_read_contract_unsupported_temporal_values():
    with pytest.raises(ValidationError, match="unsupported parameter value type date"):
        _query_key(params={"as_of": date(2026, 8, 16)})
    with pytest.raises(
        ValidationError, match="unsupported parameter value type datetime"
    ):
        _query_key(
            params={
                "as_of": datetime(2026, 8, 16, 12, 30, tzinfo=timezone.utc)
            }
        )


def test_query_cache_identity_rejects_naive_datetime():
    with pytest.raises(ValidationError, match="datetime must be timezone-aware"):
        _query_key(params={"as_of": datetime(2026, 8, 16, 12, 30)})


def test_query_cache_identity_strictly_encodes_extra_payload():
    base = _query_key()
    typed = _query_key(extra={"window": 1})
    reordered = _query_key(extra={"window": 1})

    assert typed != base
    assert typed == reordered
    with pytest.raises(ValidationError, match="repr fallback"):
        _query_key(extra={"unsupported": object()})
