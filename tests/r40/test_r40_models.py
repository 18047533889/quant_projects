"""R40 #90/#91/#93/#148: service.models remediation tests."""

from __future__ import annotations

import pytest

from service.models import (
    ComputeRequest,
    MaterializeRequest,
    RequestBudgetSnapshot,
    ValidatedFactorRequest,
    size_budget,
)


def _vr(**overrides) -> ValidatedFactorRequest:
    base = dict(
        canonical_formula="close",
        surface="daily",
        dialect="native",
        dialect_version=None,
        frequency="1d",
        market="ashare",
        universe=(),
        calendar="SSE",
        decision_time_policy="close",
        production_policy="research",
    )
    base.update(overrides)
    return ValidatedFactorRequest(**base)


# ---------------------------------------------------------------------------
# #90: backend fields are part of the digest
# ---------------------------------------------------------------------------


def test_different_backend_different_digest():
    a = _vr(backend="pandas")
    b = _vr(backend="polars")
    assert a.digest() != b.digest()


def test_backend_fields_in_payload():
    v = _vr(backend="duckdb_sql", resolved_backend_policy="duckdb_sql")
    payload = v.to_payload()
    assert payload["backend"] == "duckdb_sql"
    assert payload["resolved_backend_policy"] == "duckdb_sql"


# ---------------------------------------------------------------------------
# #91: source contract / profile version are part of the digest
# ---------------------------------------------------------------------------


def test_source_contract_change_changes_digest():
    a = _vr(source_profile="prof1", source_contract_hash="abc")
    b = _vr(source_profile="prof1", source_contract_hash="def")
    assert a.digest() != b.digest()


def test_source_profile_version_change_changes_digest():
    a = _vr(source_profile="prof1", source_profile_version="v1")
    b = _vr(source_profile="prof1", source_profile_version="v2")
    assert a.digest() != b.digest()


def test_source_binding_fields_in_payload():
    v = _vr(
        source_profile="prof1",
        source_profile_version="v1",
        source_contract_hash="abc",
        dataset_contract="ashare_stock_daily",
        snapshot_policy="snapshot_now_only",
        provider_identity="data_access",
    )
    p = v.to_payload()
    assert p["source_profile_version"] == "v1"
    assert p["source_contract_hash"] == "abc"
    assert p["dataset_contract"] == "ashare_stock_daily"
    assert p["snapshot_policy"] == "snapshot_now_only"
    assert p["provider_identity"] == "data_access"


# ---------------------------------------------------------------------------
# #93: timeout_seconds is Pydantic-validated
# ---------------------------------------------------------------------------


def test_compute_timeout_valid():
    m = ComputeRequest(formula="close", timeout_seconds=30.0)
    assert m.timeout_seconds == 30.0


def test_compute_timeout_too_small_rejected():
    with pytest.raises(Exception):
        ComputeRequest(formula="close", timeout_seconds=0.5)


def test_compute_timeout_too_large_rejected():
    with pytest.raises(Exception):
        ComputeRequest(formula="close", timeout_seconds=5000.0)


def test_materialize_timeout_invalid_rejected():
    with pytest.raises(Exception):
        MaterializeRequest(config_path="/tmp/x.yaml", timeout_seconds=0.5)


# ---------------------------------------------------------------------------
# #148: RequestBudgetSnapshot immutable + per-request env snapshot
# ---------------------------------------------------------------------------


def test_request_budget_snapshot_immutable_and_per_request(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_MAX_FORMULA_BYTES", "12345")
    budget = size_budget()
    assert isinstance(budget, RequestBudgetSnapshot)
    assert budget["max_formula_bytes"] == 12345
    assert budget.max_formula_bytes == 12345
    # immutable
    with pytest.raises(Exception):
        budget.max_formula_bytes = 999  # frozen dataclass -> FrozenInstanceError
    # per-request: changing env then re-snapshot gives a new value
    monkeypatch.setenv("FACTOR_ENGINE_MAX_FORMULA_BYTES", "54321")
    budget2 = size_budget()
    assert budget2["max_formula_bytes"] == 54321
    assert budget["max_formula_bytes"] == 12345  # first snapshot unchanged


def test_budget_mapping_compat():
    budget = size_budget()
    assert "max_formula_bytes" in budget
    assert "max_ast_nodes" not in budget
