"""Tests for FE_OPERATOR identity binding and live backend replacement."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

import factor_preprocess.adapters.fe_operator as fe_adapter
from factor_preprocess.errors import GovernanceError


class _FakeOperator:
    def __init__(self, marker: float, implementation_hash: str, contract_hash: str):
        self.marker = marker
        self.implementation_hash = implementation_hash
        self.contract_hash = contract_hash

    def calculate(self, panel):
        return panel + self.marker


class _FakeRegistry:
    def __init__(self, operator, source="test.operator.v1"):
        self.operator = operator
        self.source = source

    def resolve_canonical(self, name):
        return name

    def get(self, name, backend="pandas_numpy"):
        assert backend == "pandas_numpy"
        return self.operator

    def _read_state(self):
        return {}, {}, {
            "rank": {
                "backend_meta": {"pandas_numpy": {"source": self.source}},
                "semantic_version": "test-v1",
            }
        }


@pytest.fixture
def fake_fe(monkeypatch):
    registry = _FakeRegistry(_FakeOperator(10.0, "impl-1", "contract-1"))
    monkeypatch.setattr(fe_adapter, "_fe_operator_registry", lambda: registry)

    import factor_engine.cleaned_operators.registry as fe_registry

    monkeypatch.setattr(
        fe_registry, "_impl_source_hash", lambda operator: operator.implementation_hash
    )
    monkeypatch.setattr(
        fe_registry, "_contract_hash", lambda operator: operator.contract_hash
    )
    return registry


def _input():
    return pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")],
        "asset_id": ["A"],
        "value": [1.0],
    })


def test_identity_binds_selected_backend_adapter_and_runtime(fake_fe):
    executor = fe_adapter.FeOperatorExecutor("rank")

    identity = executor.execution_identity

    assert identity["status"] == "bound"
    assert identity["execution_origin"] == "FE_OPERATOR"
    assert identity["fe_canonical_id"] == "rank"
    assert identity["backend"] == "pandas_numpy"
    assert identity["backend_source"] == "test.operator.v1"
    assert identity["fe_implementation_hash"] == "impl-1"
    assert identity["fe_contract_hash"] == "contract-1"
    assert identity["adapter_implementation_hash"]
    assert identity["runtime_versions"] == {
        "numpy": fe_adapter.np.__version__,
        "pandas": pd.__version__,
    }
    assert "full transitive runtime" in identity["coverage_marker"]


def test_replacement_refreshes_identity_and_actual_executor_binding(fake_fe):
    executor = fe_adapter.FeOperatorExecutor("rank")
    first_identity = executor.execution_identity

    replacement = _FakeOperator(20.0, "impl-2", "contract-2")
    fake_fe.operator = replacement
    fake_fe.source = "test.operator.v2"

    second_identity = executor.execution_identity
    output = executor(values=_input())

    assert second_identity["digest"] != first_identity["digest"]
    assert second_identity["backend_source"] == "test.operator.v2"
    assert second_identity["fe_implementation_hash"] == "impl-2"
    assert second_identity["fe_contract_hash"] == "contract-2"
    assert executor._op is replacement
    assert output.iloc[0] == 21.0


def test_identity_fails_closed_without_backend_provenance(fake_fe):
    executor = fe_adapter.FeOperatorExecutor("rank")
    fake_fe.source = ""

    with pytest.raises(GovernanceError, match="identity metadata is incomplete"):
        _ = executor.execution_identity


def test_registered_fe_operator_routes_have_live_selected_backend_identities():
    from factor_preprocess.registry.transforms import get_default_registry

    registry = get_default_registry()
    expected = {
        "cs_rank": "rank",
        "cs_demean": "cs_demean",
        "cs_winsor": "winsorize",
        "forward_fill": "ffill_limit",
        "missing_indicator": "is_null",
    }

    for transform_name, canonical in expected.items():
        identity = registry.execution_identity(transform_name)
        assert identity["status"] == "bound"
        assert identity["execution_origin"] == "FE_OPERATOR"
        assert identity["identity_kind"] == "FE_OPERATOR_SELECTED_BACKEND"
        assert identity["transform_name"] == transform_name
        assert identity["fe_operator_id"] == canonical
        assert identity["fe_canonical_id"] == canonical
        expected_backend = "polars" if transform_name == "missing_indicator" else "pandas_numpy"
        if transform_name == "missing_indicator":
            assert identity["binding_state"] == "planned_native_candidate"
        assert identity["backend"] == expected_backend
        if expected_backend == "polars":
            assert identity["runtime_versions"]["polars"]
            assert identity["adapter_call_contract"]["identity_scope"] == "default_numeric_route_candidate"
            assert "unsupported dtypes" in identity["adapter_call_contract"]["input_dtype_contract"]
        assert identity["backend_source"]
        assert identity["semantic_version"]
        assert identity["fe_implementation_hash"]
        assert identity["fe_contract_hash"]
        assert identity["adapter_implementation_hash"]
        assert identity["digest"]


def test_missing_indicator_identity_tracks_each_dtype_on_same_executor():
    from factor_preprocess.registry.transforms import get_default_registry

    executor = get_default_registry().get_execution("missing_indicator")
    assert executor.execution_identity["binding_state"] == "planned_native_candidate"
    index = [8, 3, 5]
    common = {
        "date": pd.date_range("2024-01-01", periods=3),
        "asset_id": ["A", "B", "C"],
    }
    numeric = pd.DataFrame({**common, "value": [np.nan, np.inf, 2.0]}, index=index)
    actual = executor(numeric)
    assert actual.tolist() == [1.0, 0.0, 0.0]
    assert executor.execution_identity["backend"] == "polars"
    assert executor.execution_identity["binding_state"] == "last_execution"

    strings = pd.DataFrame({**common, "value": pd.Series(["x", None, "y"], dtype=object, index=index)}, index=index)
    actual = executor(strings)
    assert actual.tolist() == [0.0, 1.0, 0.0]
    assert executor.execution_identity["backend"] == "pandas_numpy"
    assert executor.execution_identity["binding_state"] == "current_selected_binding"

    dates = pd.DataFrame({
        **common,
        "value": pd.Series(pd.to_datetime(["2024-01-01", None, "2024-01-03"]), index=index),
    }, index=index)
    actual = executor(dates)
    assert actual.tolist() == [0.0, 1.0, 0.0]
    assert executor.execution_identity["backend"] == "pandas_numpy"
    assert executor.execution_identity["binding_state"] == "current_selected_binding"

    actual = executor(numeric)
    assert actual.tolist() == [1.0, 0.0, 0.0]
    assert executor.execution_identity["backend"] == "polars"
    assert executor.execution_identity["binding_state"] == "last_execution"


def test_missing_indicator_cold_string_numeric_binding_transitions():
    from factor_preprocess.registry.transforms import get_default_registry

    executor = get_default_registry().get_execution("missing_indicator")
    assert executor.execution_identity["binding_state"] == "planned_native_candidate"
    index = [3, 8]
    common = {
        "date": pd.date_range("2024-01-01", periods=2),
        "asset_id": ["A", "B"],
    }
    strings = pd.DataFrame({**common, "value": ["x", None]}, index=index)
    numeric = pd.DataFrame({**common, "value": [np.nan, np.inf]}, index=index)

    assert executor(strings).tolist() == [0.0, 1.0]
